//! Indexes files in SQLite and maintains the corresponding USearch snapshot.
//!
//! Each pass stages document changes and removes records for unavailable sources.
//! When vectors change or automatic compaction is due, it saves one final snapshot
//! before committing the document changes, counters, and snapshot reference together.

use std::fs;
use std::io;
use std::num::NonZeroUsize;
use std::ops::{ControlFlow, Deref};
use std::path::Path;

use rusqlite::{Connection, Transaction};
use usearch::Index;

use crate::adapter::Document;
use crate::compact;
use crate::config::DatabaseConfig;
use crate::database::{self, Result, VectorIndex};
use crate::embeddings::{Embedder, EmbeddingSettings, format_chunk};
use crate::index_report::{IndexProgress, IndexReport, IndexStatus, PublicationStatus};
use crate::processor;
use crate::progress::{IndexEvent, ProgressEvent};
use crate::snapshot::{self, utf8_path};
use crate::source_path;
use crate::walker::{InputKind, Walker};

/// Counters published with the vector snapshot, including removals from rollbacks.
struct VectorState {
    removed_since_compact: i64,
}

/// File-level outcome after its savepoint and vector changes have been resolved.
#[derive(Debug)]
enum StageOutcome {
    /// The file is prepared in the outer transaction and awaits pass publication.
    Staged(IndexStatus),
    /// File rows and new vectors were restored, so continuation is safe.
    RolledBack(Box<dyn std::error::Error + Send + Sync>),
}

impl VectorState {
    /// Remove `key` from the working `index` and count the graph change in `self`.
    ///
    /// Returns success after removing exactly one vector. The counter is persisted
    /// only if this working index is published, and is discarded on pass rollback.
    ///
    /// # Errors
    ///
    /// Returns an error for counter overflow, a missing key, or a USearch failure.
    fn remove_vector(&mut self, index: &Index, key: u64) -> Result<()> {
        let removed = self
            .removed_since_compact
            .checked_add(1)
            .ok_or_else(|| io::Error::other("Vector removal counter exhausted"))?;
        if index.remove(key)? != 1 {
            return Err(io::Error::other(format!("Could not remove vector {key}")).into());
        }
        self.removed_since_compact = removed;
        Ok(())
    }
}

/// Policy for one indexing pass, independent of storage and embedding services.
struct IndexOptions {
    chunk_size: NonZeroUsize,
    glass: bool,
    no_compact: bool,
}

/// Index the files under `root` and remove unavailable sources in that scope.
///
/// # Parameters
///
/// - `root` is the existing file or directory to index; symlinks resolve to its
///   canonical absolute identity before opening storage.
/// - `chunk_size` is the maximum chunk length in Unicode characters.
/// - `glass` stops the pass on the first traversal, file, or deletion-check error
///   when enabled.
/// - `no_compact` defers automatic rebuilding while preserving removal history.
/// - `database` provides the SQLite and snapshot location.
/// - `initialize_embedder` supplies an owned or borrowed embedder once the
///   writer transaction is held.
/// - `emit` receives synchronous progress and diagnostic events.
///
/// Returns an IndexReport for every attempt, including setup and publication
/// failures. Recoverable errors can coexist with a published pass. Aborted passes
/// keep their staged counts, report zero committed counts, and retain their
/// diagnostics without requiring callers to interpret nested results.
pub(crate) fn run<E: Deref>(
    root: &Path,
    chunk_size: NonZeroUsize,
    glass: bool,
    no_compact: bool,
    database: &DatabaseConfig,
    initialize_embedder: impl FnOnce() -> Result<E>,
    emit: &mut dyn FnMut(ProgressEvent<'_>),
) -> IndexReport
where
    E::Target: Embedder + Sized,
{
    let progress = IndexProgress::new(emit);
    let root = match fs::canonicalize(root) {
        Ok(root) => root,
        Err(error) => {
            progress.issue(Some(root), &error);
            return progress.finish(PublicationStatus::NotPublished);
        }
    };
    progress.notify(IndexEvent::Opening(&root));
    let publication = match execute(
        &root,
        IndexOptions {
            chunk_size,
            glass,
            no_compact,
        },
        database,
        initialize_embedder,
        &progress,
    ) {
        Ok(publication) => publication,
        Err(error) => {
            progress.issue(None, error.as_ref());
            PublicationStatus::NotPublished
        }
    };
    progress.finish(publication)
}

/// Writable vector state shared by preparation, file processing, and publication.
struct WorkingIndex {
    index: Index,
    vectors: VectorState,
    previous: Option<std::path::PathBuf>,
    changed: bool,
}

/// Coordinate preparation, traversal, deletion reconciliation, and publication.
///
/// `root` and `options` define the input scope and error policy. `database`
/// locates storage, `initialize_embedder` supplies the embedding provider after
/// the writer lock is acquired, and `progress` records counters and emits events.
/// Returns the publication outcome; already-reported aborts remain NotPublished.
/// Validates the pass's chunk size before loading vectors or visiting sources.
///
/// # Errors
///
/// Returns unreported setup, database, vector cleanup, rebuild, and publication
/// errors for the outer entry point to add to the report.
fn execute<E: Deref>(
    root: &Path,
    options: IndexOptions,
    database: &DatabaseConfig,
    initialize_embedder: impl FnOnce() -> Result<E>,
    progress: &IndexProgress<'_>,
) -> Result<PublicationStatus>
where
    E::Target: Embedder + Sized,
{
    let mut connection = database::open(&database.sqlite_path())?;
    // Acquire the writer lock before choosing a snapshot or allocating vector keys.
    // Keep it through traversal, embedding calls, and publication.
    let mut transaction = database::begin_write(&mut connection)?;
    database::ensure_chunk_size(&transaction, options.chunk_size)?;
    let embedder = initialize_embedder()?;
    let settings = embedder.settings();
    let Some(mut working) = prepare_index(&transaction, database, settings, progress)? else {
        return Ok(PublicationStatus::NotPublished);
    };
    normalize_document_paths(&transaction, root, progress)?;
    let ControlFlow::Continue(input_kind) = walk_files(
        root,
        &options,
        &mut transaction,
        &mut working,
        &*embedder,
        progress,
    ) else {
        return Ok(PublicationStatus::NotPublished);
    };
    if remove_missing_documents(&transaction, root, input_kind, options.glass, progress)?.is_break()
    {
        return Ok(PublicationStatus::NotPublished);
    }
    publish_index(
        transaction,
        working,
        database,
        settings,
        options.no_compact,
        progress,
    )
}

/// Normalize legacy source identities within canonical `root` in `connection`.
///
/// `progress` records and reports aliases removed when a canonical document
/// already exists. Missing paths resolve through their existing ancestors.
/// An in-scope path must still lead to a regular file without following
/// descendant symlinks before it can be relocated to a different identity.
/// Inaccessible or unresolvable paths stay unchanged for deletion checks.
/// Returns success after staging all renames and duplicate removals in scope.
///
/// # Errors
///
/// Returns an error for source-query, UTF-8 conversion, or SQLite write failures.
fn normalize_document_paths(
    connection: &Connection,
    root: &Path,
    progress: &IndexProgress<'_>,
) -> Result<()> {
    for source in database::document_sources(connection)? {
        let path = Path::new(&source);
        let Ok(canonical) = source_path::normalize_stored(path) else {
            continue;
        };
        if !canonical.starts_with(root) {
            continue;
        }
        if path != canonical
            && let Ok(relative) = path.strip_prefix(root)
            && !matches!(source_path::is_regular_source(root, relative), Ok(true))
        {
            continue;
        }
        if database::relocate_document(connection, &source, &canonical)? {
            progress.removal(path);
        }
    }
    Ok(())
}

/// Load a writable snapshot and initialize this pass's vector bookkeeping.
///
/// `connection` holds the writer transaction, `database` locates snapshots,
/// `settings` defines compatible embeddings, and `progress` records and reports
/// snapshot-load failures.
/// Returns working state, or None after an already-reported snapshot-load failure.
///
/// # Errors
///
/// Returns unreported metadata, compatibility, index creation, and tracking errors.
fn prepare_index(
    connection: &Connection,
    database: &DatabaseConfig,
    settings: &EmbeddingSettings,
    progress: &IndexProgress<'_>,
) -> Result<Option<WorkingIndex>> {
    let active = database::vector_index(connection)?;
    let previous = active
        .as_ref()
        .map(|active| snapshot::active_path(&database.path, active, settings))
        .transpose()?;
    let index = snapshot::new_index(settings.dimensions)?;
    if let Some(path) = &previous {
        // The active file remains unchanged until publication commits a replacement.
        if let Err(error) = index.load(utf8_path(path)?) {
            progress.issue(Some(path), &format!("Cannot load active index: {error}"));
            return Ok(None);
        }
        snapshot::validate(&index, settings.dimensions)?;
    }
    database::track_removed_vectors(connection)?;
    Ok(Some(WorkingIndex {
        index,
        vectors: VectorState {
            removed_since_compact: active
                .as_ref()
                .map_or(0, |active| active.removed_since_compact),
        },
        previous,
        changed: false,
    }))
}

/// Walk `root` and stage supported files according to `options`.
///
/// `transaction` holds file savepoints, `working` receives new vectors,
/// `embedder` generates embeddings, and `progress` records and reports file and traversal outcomes.
/// Returns the input kind on Continue, or Break after a reported fatal error or
/// a glass-mode failure. Every failure is recorded exactly once.
fn walk_files(
    root: &Path,
    options: &IndexOptions,
    transaction: &mut Transaction<'_>,
    working: &mut WorkingIndex,
    embedder: &dyn Embedder,
    progress: &IndexProgress<'_>,
) -> ControlFlow<(), InputKind> {
    let mut aborted = false;
    progress.notify(IndexEvent::Scanning);
    let walked = Walker {
        continue_on_error: !options.glass,
        report_error: &|path, error| {
            progress.issue(Some(path), error);
        },
    }
    .walk(root, |path| {
        let flow = process_file(
            path,
            transaction,
            working,
            options.chunk_size,
            embedder,
            progress,
        )?;
        aborted = flow.is_break();
        Ok(flow)
    });
    match walked {
        Ok(kind) if !aborted => ControlFlow::Continue(kind),
        _ => ControlFlow::Break(()),
    }
}

/// Read and stage one source `path` using `chunk_size` and `embedder`.
///
/// `transaction` owns the file savepoint, `working` holds the mutable vectors,
/// and `progress` records and reports the file's outcome.
/// Returns Continue for StageOutcome::Staged, or Break after reporting an
/// unrecoverable savepoint or vector rollback failure.
///
/// # Errors
///
/// Returns file read errors and StageOutcome::RolledBack failures for Walker to
/// record and apply its continuation policy.
fn process_file(
    path: &Path,
    transaction: &mut Transaction<'_>,
    working: &mut WorkingIndex,
    chunk_size: NonZeroUsize,
    embedder: &dyn Embedder,
    progress: &IndexProgress<'_>,
) -> io::Result<ControlFlow<()>> {
    progress.file(path);
    // Hash and split the same bytes even if the source changes during embedding.
    let source = fs::read_to_string(path)?;
    let previous_size = working.index.size();
    let outcome = stage_file(
        transaction,
        working,
        path,
        &source,
        chunk_size,
        embedder.settings().chunk_format.as_deref(),
        |input, ordinal| {
            progress.notify(IndexEvent::Embedding {
                path,
                chunk: ordinal,
            });
            embedder.embed_chunk(input)
        },
    );
    match outcome {
        Ok(StageOutcome::Staged(status)) => {
            working.changed |= working.index.size() != previous_size;
            progress.staged(path, status);
            Ok(ControlFlow::Continue(()))
        }
        Ok(StageOutcome::RolledBack(error)) => Err(io::Error::other(error)),
        Err(error) => {
            progress.issue(Some(path), error.as_ref());
            Ok(ControlFlow::Break(()))
        }
    }
}

/// Remove unreferenced vectors from `working` and their full hashes from `connection`.
///
/// Returns success after marking the working index changed when vectors were removed.
/// This runs after file savepoints and deletion reconciliation, so rolled-back
/// deletions and chunks still shared by other documents retain their vectors.
///
/// # Errors
///
/// Returns unreported SQLite or vector-removal errors.
fn remove_obsolete_vectors(connection: &Connection, working: &mut WorkingIndex) -> Result<()> {
    for key in database::removed_vector_keys(connection)? {
        database::delete_embedding(connection, key)?;
        working.vectors.remove_vector(&working.index, key)?;
        working.changed = true;
    }
    Ok(())
}

/// Reconcile vectors in `working` and publish the completed `transaction`.
///
/// `database` supplies snapshot storage, `settings` identifies compatible vectors,
/// `no_compact` defers automatic rebuilding, and `progress` emits phase events.
/// Returns Published with the actual snapshot-update and compaction flags.
///
/// # Errors
///
/// Returns unreported cleanup, rebuild, snapshot-write, or commit errors. On failure,
/// the caller records NotPublished while dropping the transaction rolls back staging.
fn publish_index(
    transaction: Transaction<'_>,
    mut working: WorkingIndex,
    database: &DatabaseConfig,
    settings: &EmbeddingSettings,
    no_compact: bool,
    progress: &IndexProgress<'_>,
) -> Result<PublicationStatus> {
    remove_obsolete_vectors(&transaction, &mut working)?;
    let compacted = compact_if_needed(
        &transaction,
        &mut working.index,
        &mut working.vectors,
        settings.dimensions,
        no_compact,
        progress,
    )?;
    let snapshot_updated = working.changed || compacted;
    // Replacements can preserve the size, and deletion-only passes can shrink it.
    let updated_index = snapshot_updated.then_some(&working.index);
    commit_indexing(
        transaction,
        updated_index,
        database,
        settings,
        working.vectors,
        working.previous.as_deref(),
        progress,
    )?;
    Ok(PublicationStatus::Published {
        snapshot_updated,
        compacted,
    })
}

/// Stage one file in a savepoint and add its new vectors to the working index.
///
/// # Parameters
///
/// - `transaction` is the indexing pass's SQLite transaction.
/// - `working` holds the vector graph and counts removals during rollback.
/// - `path` identifies the source document.
/// - `source` contains the text used for both hashing and chunking.
/// - `chunk_size` is the maximum chunk length in Unicode characters.
/// - `chunk_format` configures the exact input to hash and embed.
/// - `embed` produces a vector for a new model input and receives its one-based
///   generation number for this file.
///
/// Returns Staged after releasing the file savepoint, or RolledBack with the
/// original file error after restoring its rows and removing its new vectors.
///
/// # Errors
///
/// Returns an error if creating, releasing, or rolling back the savepoint fails,
/// or if undoing or counting the vector inserts fails. These failures require
/// aborting the entire indexing pass.
fn stage_file(
    transaction: &mut Transaction<'_>,
    working: &mut WorkingIndex,
    path: &Path,
    source: &str,
    chunk_size: NonZeroUsize,
    chunk_format: Option<&str>,
    mut embed: impl FnMut(String, usize) -> Result<Vec<f32>>,
) -> Result<StageOutcome> {
    let index = &working.index;
    let hash = blake3::hash(source.as_bytes()).to_hex();
    let mut added_keys = Vec::new();
    let savepoint = transaction.savepoint()?;
    let result = stage_document(
        &savepoint,
        path,
        hash.as_str(),
        || processor::open_document(path, source, chunk_size),
        chunk_format,
        |key, input| {
            let vector = embed(input, added_keys.len() + 1)?;
            if index.size() == index.capacity() {
                index.reserve(index.capacity().saturating_mul(2).max(1))?;
            }
            index.add(key, &vector)?;
            added_keys.push(key);
            Ok(())
        },
    );
    match result {
        Ok(status) => {
            // Releasing the savepoint keeps this file's changes in the outer transaction.
            savepoint.commit()?;
            Ok(StageOutcome::Staged(status))
        }
        Err(error) => {
            // Report RolledBack only after both rows and vector inserts have been restored.
            savepoint.finish()?;
            for key in added_keys {
                working.vectors.remove_vector(index, key)?;
            }
            Ok(StageOutcome::RolledBack(error))
        }
    }
}

/// Prepare one file inside the caller's transaction or savepoint.
///
/// `connection` holds a pass whose chunk size was already validated. `path` and
/// `content_hash` identify the source; `load_document` lazily supplies its chunks.
/// `chunk_format` prepares the exact model input and `add_vector` generates and
/// stores a vector only for a previously unseen input.
/// Returns whether the source was unchanged, indexed, or reused shared chunks.
///
/// # Errors
///
/// Returns path, adapter, serialization, SQLite, key-allocation, or vector errors.
/// The caller must roll back the file's savepoint on failure.
fn stage_document<'a>(
    connection: &Connection,
    path: &Path,
    content_hash: &str,
    load_document: impl FnOnce() -> io::Result<Document<'a>>,
    chunk_format: Option<&str>,
    mut add_vector: impl FnMut(u64, String) -> Result<()>,
) -> Result<IndexStatus> {
    let source = path
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Source path must be UTF-8"))?;
    // Skip the adapter entirely when this path still has the same content.
    if database::document_hash(connection, source)?.as_deref() == Some(content_hash) {
        return Ok(IndexStatus::Unchanged);
    }

    // The callback is FnOnce: extraction is deferred until needed and runs once.
    let document = load_document()?;
    if document.metadata.source != path {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Adapter returned a different source path",
        )
        .into());
    }
    // Decide reuse using other documents, before replacing this document.
    let reuse = database::has_other_document(connection, source, content_hash)?;
    database::replace_document(connection, source, content_hash, &document.metadata)?;
    if !reuse {
        for (ordinal, chunk) in document.chunks.enumerate() {
            let input = format_chunk(chunk_format, chunk.section_path.as_deref(), &chunk.text)?;
            let hash = blake3::hash(input.as_bytes());
            let key = match database::embedding_key(connection, &hash)? {
                Some(key) => key,
                None => {
                    let key = allocate_vector_key(connection, &hash)?;
                    add_vector(key, input)?;
                    key
                }
            };
            database::insert_chunk(connection, content_hash, ordinal, key, &chunk)?;
        }
    }
    Ok(if reuse {
        IndexStatus::Reused
    } else {
        IndexStatus::Indexed
    })
}

/// Register a new full `hash` in `connection` and return its USearch key.
///
/// Starts with the first eight hash bytes in little-endian order and probes
/// consecutive keys when occupied. Skips USearch's reserved deleted-vector key.
/// Call only after a full-hash miss. Returns an error if SQLite fails or every
/// usable key is occupied. The caller owns the savepoint and vector insertion.
fn allocate_vector_key(connection: &Connection, hash: &blake3::Hash) -> Result<u64> {
    let mut prefix = [0; 8];
    prefix.copy_from_slice(&hash.as_bytes()[..8]);
    let start = u64::from_le_bytes(prefix);
    let mut key = start;
    loop {
        if key != u64::MAX && database::try_insert_embedding(connection, key, hash)? {
            return Ok(key);
        }
        key = key.wrapping_add(1);
        if key == start {
            return Err(io::Error::other("USearch key space exhausted").into());
        }
    }
}

/// Rebuild the working index before snapshot publication when policy requires it.
///
/// - `connection` sees the pass's final staged chunk rows.
/// - `index` is the working vector graph, replaced only after rebuilding succeeds.
/// - `vectors` carries accumulated removals and resets them after rebuilding.
/// - `dimensions` is the validated vector length.
/// - `no_compact` skips rebuilding while preserving the counters.
/// - `progress` emits the compaction phase before rebuilding starts.
///
/// Returns whether the index was rebuilt, including when the removal threshold
/// was reached by a previous pass that deferred compaction.
///
/// # Errors
///
/// Returns an error if reading live chunks or copying their embeddings fails.
fn compact_if_needed(
    connection: &Connection,
    index: &mut Index,
    vectors: &mut VectorState,
    dimensions: NonZeroUsize,
    no_compact: bool,
    progress: &IndexProgress<'_>,
) -> Result<bool> {
    if no_compact || vectors.removed_since_compact == 0 {
        return Ok(false);
    }
    let live_vectors = database::live_vector_count(connection)?;
    if !compact::should_compact(live_vectors, vectors.removed_since_compact) {
        return Ok(false);
    }
    progress.notify(IndexEvent::Compacting);
    let keys = database::live_vector_keys(connection)?;
    *index = compact::rebuild(index, &keys, dimensions)?;
    vectors.removed_since_compact = 0;
    Ok(true)
}

/// Delete stored documents whose in-scope paths no longer lead to regular sources.
///
/// # Parameters
///
/// - `connection` is the SQLite connection used to query and delete records.
/// - `root` is the absolute path of the indexed file or directory.
/// - `input_kind` determines whether the scope covers one file or a directory tree.
/// - `glass` causes unexpected filesystem errors to abort the check when enabled.
/// - `progress` tracks removals and failures and emits events for the phase and outcomes.
///
/// Returns Break when glass mode aborts after a reported filesystem failure,
/// otherwise Continue; nonfatal failures remain in the progress diagnostics.
///
/// # Errors
///
/// Returns an error for unreported database failures.
fn remove_missing_documents(
    connection: &Connection,
    root: &Path,
    input_kind: InputKind,
    glass: bool,
    progress: &IndexProgress<'_>,
) -> Result<ControlFlow<()>> {
    // Check stored paths so skipped traversal entries and inaccessible paths
    // are not mistaken for deleted sources.
    progress.notify(IndexEvent::CheckingDeletions);
    for source in database::document_sources(connection)? {
        let path = Path::new(&source);
        let Ok(relative) = path.strip_prefix(root) else {
            continue;
        };
        if matches!(input_kind, InputKind::File) && relative != Path::new("") {
            continue;
        }
        match source_path::is_regular_source(root, relative) {
            Ok(true) => {}
            Ok(false) => {
                database::delete_document(connection, &source)?;
                progress.removal(path);
            }
            Err(error) => {
                progress.issue(
                    Some(path),
                    &format!("Could not check for deletion: {error}"),
                );
                if glass {
                    return Ok(ControlFlow::Break(()));
                }
            }
        }
    }

    Ok(ControlFlow::Continue(()))
}

/// Commit the document changes and publish an updated vector snapshot when provided.
///
/// # Parameters
///
/// - `transaction` contains the staged SQLite document changes.
/// - `updated_index` is the working vector index to save when vectors were added,
///   removed, or rebuilt by automatic compaction.
/// - `database` supplies the SQLite and snapshot locations.
/// - `settings` identifies the model and digest, dimensions, and chunk format used by the embedder.
/// - `vectors` supplies the accumulated removal count to persist with the snapshot.
/// - `previous` is the old snapshot path to remove after a successful commit, if present.
/// - `progress` forwards snapshot events to the pass's listener.
///
/// Reports old snapshot cleanup failures after the commit and returns success.
/// On failure, removes the new file only if SQLite confirms it is unreferenced.
///
/// # Errors
///
/// Returns an error if the snapshot cannot be written or synced, its path or
/// dimensions cannot be represented, or SQLite cannot stage or commit its metadata.
fn commit_indexing(
    transaction: Transaction<'_>,
    updated_index: Option<&Index>,
    database: &DatabaseConfig,
    settings: &EmbeddingSettings,
    vectors: VectorState,
    previous: Option<&Path>,
    progress: &IndexProgress<'_>,
) -> Result<()> {
    let dimensions = i64::try_from(settings.dimensions.get())?;
    let update = updated_index.map(|index| {
        (
            index,
            VectorIndex {
                format_version: database::VECTOR_INDEX_VERSION,
                file_name: String::new(),
                model: settings.model.clone(),
                model_digest: settings.model_digest.clone(),
                chunk_format: settings.chunk_format.clone(),
                dimensions,
                removed_since_compact: vectors.removed_since_compact,
            },
        )
    });
    snapshot::commit(transaction, update, database, previous, &mut |event| {
        progress.emit(event)
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::path::PathBuf;

    use super::*;
    use crate::index_report::IndexCounts;

    /// Deterministic embeddings with optional failure and source disappearance injection.
    struct TestEmbedder {
        settings: EmbeddingSettings,
        calls: Cell<usize>,
        fail_on: Option<usize>,
        delete_path: Option<PathBuf>,
    }

    impl Embedder for TestEmbedder {
        /// Return this test provider's stable embedding identity.
        fn settings(&self) -> &EmbeddingSettings {
            &self.settings
        }

        /// Ignore model `_input`, injecting the configured test event.
        ///
        /// Returns a fixed vector, or an injected generation or file-removal error.
        fn embed_chunk(&self, _input: String) -> Result<Vec<f32>> {
            let call = self.calls.get() + 1;
            self.calls.set(call);
            if self.fail_on == Some(call) {
                return Err(io::Error::other("injected embedding failure").into());
            }
            if call == 1
                && let Some(path) = &self.delete_path
            {
                fs::remove_file(path)?;
            }
            Ok(vec![1.0, 0.0, 0.0])
        }

        /// Reject `_query` because report tests exercise indexing only.
        ///
        /// Always returns an unsupported-operation error.
        fn embed_query(&self, _query: &str) -> Result<Vec<f32>> {
            Ok(vec![1.0, 0.0, 0.0])
        }
    }

    /// Create a provider that fails on `fail_on` or removes `delete_path` on its first call.
    ///
    /// Returns a deterministic provider with a stable digest and no service dependency.
    fn test_embedder(fail_on: Option<usize>, delete_path: Option<PathBuf>) -> TestEmbedder {
        TestEmbedder {
            settings: EmbeddingSettings {
                model: "test-model".into(),
                model_digest: "a".repeat(64),
                dimensions: NonZeroUsize::new(3).unwrap(),
                chunk_format: None,
            },
            calls: Cell::new(0),
            fail_on,
            delete_path,
        }
    }

    /// Create temporary corpus storage under the canonical system temporary directory.
    ///
    /// Returns the directory guard and configuration, or a path-resolution or
    /// temporary-directory creation error.
    fn corpus() -> Result<(tempfile::TempDir, DatabaseConfig)> {
        let directory = tempfile::tempdir_in(fs::canonicalize(std::env::temp_dir())?)?;
        let database = DatabaseConfig {
            path: directory.path().to_path_buf(),
        };
        Ok((directory, database))
    }

    /// Index `root` using `database` and `embedder`, with the requested `glass` policy.
    ///
    /// Returns the complete report with default test chunk size and automatic compaction.
    fn index(
        root: &Path,
        database: &DatabaseConfig,
        embedder: &TestEmbedder,
        glass: bool,
    ) -> IndexReport {
        run(
            root,
            NonZeroUsize::new(1_000).unwrap(),
            glass,
            false,
            database,
            || Ok(embedder),
            &mut |_| {},
        )
    }

    /// Verify a second writer fails before embedding setup while the published index is searchable.
    ///
    /// Returns success after checking contention, independent databases, and
    /// lock release, or propagates fixture, indexing, or search errors.
    #[test]
    fn one_writer_per_database_allows_parallel_search() -> Result<()> {
        let (directory, database) = corpus()?;
        let (_other_directory, other_database) = corpus()?;
        let note = directory.path().join("note.md");
        let text = "Published content.";
        fs::write(&note, text)?;
        let embedder = test_embedder(None, None);
        assert!(index(&note, &database, &embedder, false).is_success());

        // The outer index run holds the writer lock when its embedder is initialized.
        let report = run(
            &note,
            NonZeroUsize::new(1_000).unwrap(),
            false,
            false,
            &database,
            || {
                let competing = run::<&TestEmbedder>(
                    &note,
                    NonZeroUsize::new(1_000).unwrap(),
                    false,
                    false,
                    &database,
                    || panic!("a busy index invocation must not initialize its embedder"),
                    &mut |_| {},
                );
                assert_eq!(competing.publication, PublicationStatus::NotPublished);
                assert_eq!(competing.files_seen, 0);
                assert_eq!(competing.errors.len(), 1);
                assert!(
                    competing.errors[0]
                        .message
                        .contains("tb index or tb compact")
                );

                let error = compact::run(&database, &mut |_| {})
                    .err()
                    .expect("same database is busy");
                assert_eq!(
                    error.downcast_ref::<io::Error>().map(io::Error::kind),
                    Some(io::ErrorKind::WouldBlock),
                );
                assert!(error.to_string().contains("tb index or tb compact"));
                assert!(compact::run(&other_database, &mut |_| {})?.is_none());

                let hits = crate::search::run("query".into(), 1, &database, &embedder)?;
                assert_eq!(hits.len(), 1);
                assert_eq!(hits[0].text, text);
                Ok(&embedder)
            },
            &mut |_| {},
        );
        assert!(report.is_success(), "{:?}", report.errors);
        assert!(compact::run(&database, &mut |_| {})?.is_some());
        Ok(())
    }

    /// Verify colliding prefixes retain distinct full hashes and stable keys in SQLite.
    /// Returns success for both lookups, USearch key round trips, and sentinel
    /// handling, or a database or vector index error.
    #[test]
    fn conflicting_vector_keys_probe_without_aliasing_full_hashes() -> Result<()> {
        let connection = database::open(Path::new(":memory:"))?;
        let mut bytes = [0; 32];
        bytes[..8].copy_from_slice(&(u64::MAX - 1).to_le_bytes());
        let first = blake3::Hash::from_bytes(bytes);
        bytes[8] = 1;
        let second = blake3::Hash::from_bytes(bytes);
        let first_key = allocate_vector_key(&connection, &first)?;
        let second_key = allocate_vector_key(&connection, &second)?;
        assert_eq!(first_key, u64::MAX - 1);
        assert_eq!(second_key, 0);
        let index = snapshot::new_index(NonZeroUsize::new(3).unwrap())?;
        index.reserve(2)?;
        index.add(first_key, &[1.0_f32, 0.0, 0.0])?;
        index.add(second_key, &[0.0_f32, 1.0, 0.0])?;
        let mut vector = [0.0_f32; 3];
        assert_eq!(index.get(first_key, &mut vector)?, 1);
        assert_eq!(vector, [1.0, 0.0, 0.0]);
        assert_eq!(index.get(second_key, &mut vector)?, 1);
        assert_eq!(vector, [0.0, 1.0, 0.0]);
        assert_eq!(
            database::embedding_key(&connection, &first)?,
            Some(first_key)
        );
        assert_eq!(
            database::embedding_key(&connection, &second)?,
            Some(second_key)
        );
        database::delete_embedding(&connection, first_key)?;
        assert_eq!(
            database::embedding_key(&connection, &second)?,
            Some(second_key)
        );
        Ok(())
    }

    /// Verify edited and repeated chunks retain shared keys until their last reference disappears.
    /// Returns success for reuse, search, and cleanup, or a fixture or storage error.
    #[test]
    fn prepared_input_hash_reuses_vectors_across_edits_and_files() -> Result<()> {
        let (directory, database) = corpus()?;
        let a = directory.path().join("a.md");
        let b = directory.path().join("b.md");
        let shared_hash = blake3::hash(b"Common\n\nShared paragraph.\n");
        let embedder = test_embedder(None, None);
        fs::write(&a, "# Common\n\nShared paragraph.\n")?;
        assert!(index(&a, &database, &embedder, false).is_success());
        let reader = database::open_read_only(&database.sqlite_path())?;
        let key = database::embedding_key(&reader, &shared_hash)?.unwrap();
        let original = database::vector_index(&reader)?.unwrap().file_name;
        assert_eq!(embedder.calls.get(), 1);

        // Moving the same excerpt down a line changes coordinates, but not model input.
        fs::write(&a, "\n# Common\n\nShared paragraph.\n")?;
        let report = index(&a, &database, &embedder, false);
        assert!(report.is_success());
        assert_eq!(embedder.calls.get(), 1);
        assert_eq!(
            database::vector_index(&reader)?.unwrap().file_name,
            original
        );
        let sources = database::chunk_sources(&reader, key)?;
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].location.line_start, 4);

        // The copy has extra content, so reuse must come from the chunk, not the file hash.
        fs::write(
            &b,
            "# Common\n\nShared paragraph.\n\n# Other\n\nA second paragraph.\n",
        )?;
        assert!(index(&b, &database, &embedder, false).is_success());
        assert_eq!(embedder.calls.get(), 2);
        assert_eq!(database::chunk_sources(&reader, key)?.len(), 2);
        assert_eq!(database::live_vector_count(&reader)?, 2);

        let report = compact::run(&database, &mut |_| {})?.unwrap();
        assert_eq!(report.retained_vectors, 2);
        assert_eq!(database::embedding_key(&reader, &shared_hash)?, Some(key));
        fs::remove_file(&a)?;
        assert!(index(directory.path(), &database, &embedder, false).is_success());
        assert_eq!(database::embedding_key(&reader, &shared_hash)?, Some(key));
        let hits = crate::search::run("query".into(), 5, &database, &embedder)?;
        assert_eq!(
            hits.iter()
                .filter(|hit| hit.text == "Shared paragraph.\n")
                .count(),
            1
        );

        // Changing the heading changes the actual model input even with identical body text.
        fs::write(&b, "# Renamed\n\nShared paragraph.\n")?;
        assert!(index(&b, &database, &embedder, false).is_success());
        assert_eq!(embedder.calls.get(), 3);
        assert!(database::embedding_key(&reader, &shared_hash)?.is_none());
        let active = database::vector_index(&reader)?.unwrap();
        let published = snapshot::new_index(embedder.settings.dimensions)?;
        let path = snapshot::stored_path(&database.path, &active)?;
        published.view(utf8_path(&path)?)?;
        assert_eq!(published.size(), 1);
        assert!(!published.contains(key));
        Ok(())
    }

    /// Verify incompatible chunk settings abort before traversal, even for an empty scope.
    ///
    /// Returns success for one diagnostic and unchanged storage/embedding calls;
    /// propagates fixture and SQLite errors.
    #[test]
    fn chunk_size_mismatch_aborts_once_before_visiting_files() -> Result<()> {
        let (directory, database) = corpus()?;
        let note = directory.path().join("note.md");
        let empty = directory.path().join("empty");
        fs::write(&note, "A note.")?;
        fs::create_dir(&empty)?;
        let embedder = test_embedder(None, None);
        assert!(index(&note, &database, &embedder, false).is_success());
        let calls = embedder.calls.get();
        for root in [empty.as_path(), directory.path()] {
            let report = run(
                root,
                NonZeroUsize::new(500).unwrap(),
                false,
                false,
                &database,
                || Ok(&embedder),
                &mut |_| {},
            );
            assert_eq!(report.publication, PublicationStatus::NotPublished);
            assert_eq!(report.files_seen, 0);
            assert_eq!(report.errors.len(), 1);
            assert!(report.errors[0].message.contains("Chunk size differs"));
        }
        assert_eq!(embedder.calls.get(), calls);
        let connection = database::open_read_only(&database.sqlite_path())?;
        assert_eq!(
            database::document_sources(&connection)?,
            [note.to_str().unwrap()],
        );
        Ok(())
    }

    /// Verify legacy aliases merge into one canonical source without re-embedding.
    ///
    /// Returns success for canonical storage and alias traversal; propagates
    /// fixture, SQLite, or path-resolution errors.
    #[cfg(unix)]
    #[test]
    fn canonical_paths_merge_legacy_aliases_without_new_embeddings() -> Result<()> {
        use std::os::unix::fs::symlink;

        let (directory, database) = corpus()?;
        let notes = directory.path().join("notes");
        let alias = directory.path().join("alias");
        fs::create_dir(&notes)?;
        fs::create_dir(notes.join("nested"))?;
        let note = notes.join("note.md");
        fs::write(&note, "Shared note.")?;
        symlink(&notes, &alias)?;
        let embedder = test_embedder(None, None);
        assert!(index(&note, &database, &embedder, false).is_success());
        let calls = embedder.calls.get();
        let connection = database::open(&database.sqlite_path())?;
        connection.execute(
            "INSERT INTO documents (source, content_hash, attributes)
             SELECT ?1, content_hash, attributes FROM documents WHERE source = ?2",
            rusqlite::params![
                alias.join("note.md").to_str().unwrap(),
                note.to_str().unwrap(),
            ],
        )?;
        drop(connection);
        let report = index(&alias.join("nested/.."), &database, &embedder, false);
        assert!(report.is_success());
        assert_eq!(report.committed.unchanged, 1);
        assert_eq!(report.committed.removed, 1);
        assert_eq!(embedder.calls.get(), calls);
        let connection = database::open_read_only(&database.sqlite_path())?;
        assert_eq!(
            database::document_sources(&connection)?,
            [note.to_str().unwrap()],
        );
        Ok(())
    }

    /// Verify that replacing a Markdown source with a directory removes its chunks
    /// and orphaned vector while keeping the vector shared with a surviving note.
    /// Returns success after checking storage, or propagates fixture and storage errors.
    #[test]
    fn removes_source_replaced_by_directory() -> Result<()> {
        let (directory, database) = corpus()?;
        let note = directory.path().join("note.md");
        let survivor = directory.path().join("survivor.md");
        fs::write(
            &note,
            "# Shared\n\nShared paragraph.\n\n# Unique\n\nUnique paragraph.\n",
        )?;
        fs::write(&survivor, "# Shared\n\nShared paragraph.\n")?;
        let embedder = test_embedder(None, None);
        assert!(index(directory.path(), &database, &embedder, false).is_success());
        let reader = database::open_read_only(&database.sqlite_path())?;
        let shared = blake3::hash(b"Shared\n\nShared paragraph.\n");
        let unique = blake3::hash(b"Unique\n\nUnique paragraph.\n");
        let shared_key = database::embedding_key(&reader, &shared)?.unwrap();
        let unique_key = database::embedding_key(&reader, &unique)?.unwrap();
        assert_eq!(database::chunk_sources(&reader, shared_key)?.len(), 2);

        fs::remove_file(&note)?;
        fs::create_dir(&note)?;
        let report = index(directory.path(), &database, &embedder, false);
        assert!(report.is_success(), "{:?}", report.errors);
        assert_eq!(report.committed.removed, 1);
        assert_eq!(
            database::document_sources(&reader)?,
            [survivor.to_str().unwrap()]
        );
        assert_eq!(database::chunk_sources(&reader, shared_key)?.len(), 1);
        assert!(database::chunk_sources(&reader, unique_key)?.is_empty());
        assert!(database::embedding_key(&reader, &unique)?.is_none());
        let active = database::vector_index(&reader)?.unwrap();
        let published = snapshot::new_index(embedder.settings.dimensions)?;
        published.view(utf8_path(&snapshot::stored_path(&database.path, &active)?)?)?;
        assert!(published.contains(shared_key));
        assert!(!published.contains(unique_key));
        Ok(())
    }

    /// Verify that replacing a parent directory with a file removes its stored
    /// descendants in glass mode and retains documents outside the selected scope.
    /// Returns success after checking storage, or propagates fixture and storage errors.
    #[test]
    fn removes_source_beneath_parent_replaced_by_file() -> Result<()> {
        let (directory, database) = corpus()?;
        let scope = directory.path().join("scope");
        let parent = scope.join("parent");
        let note = parent.join("note.md");
        let outside = directory.path().join("outside.md");
        fs::create_dir_all(&parent)?;
        fs::write(&note, "Descendant note.")?;
        fs::write(&outside, "Note outside the selected scope.")?;
        let embedder = test_embedder(None, None);
        assert!(index(directory.path(), &database, &embedder, true).is_success());

        fs::remove_dir_all(&parent)?;
        fs::write(&parent, "Replacement file.")?;
        let report = index(&scope, &database, &embedder, true);
        assert!(report.is_success(), "{:?}", report.errors);
        assert_eq!(report.committed.removed, 1);
        let reader = database::open_read_only(&database.sqlite_path())?;
        assert_eq!(
            database::document_sources(&reader)?,
            [outside.to_str().unwrap()]
        );
        assert_eq!(database::live_vector_count(&reader)?, 1);
        Ok(())
    }

    /// Verify that replacing a source or its parent with a descendant symlink
    /// removes the old identity while indexing the target at its real path.
    /// Returns success after checking storage, or propagates fixture and storage errors.
    #[cfg(unix)]
    #[test]
    fn removes_sources_replaced_by_descendant_symlinks() -> Result<()> {
        use std::os::unix::fs::symlink;

        let (directory, database) = corpus()?;
        let parent = directory.path().join("parent");
        let note = directory.path().join("note.md");
        let target_dir = directory.path().join("target");
        let target = target_dir.join("note.md");
        fs::create_dir(&parent)?;
        fs::write(parent.join("note.md"), "Former descendant.")?;
        fs::write(&note, "Former file.")?;
        let embedder = test_embedder(None, None);
        assert!(index(directory.path(), &database, &embedder, true).is_success());

        fs::remove_dir_all(&parent)?;
        fs::remove_file(&note)?;
        fs::create_dir(&target_dir)?;
        fs::write(&target, "New target.")?;
        symlink(&target_dir, &parent)?;
        symlink(&target, &note)?;
        let report = index(directory.path(), &database, &embedder, true);
        assert!(report.is_success(), "{:?}", report.errors);
        assert_eq!(report.files_seen, 1);
        assert_eq!(report.committed.removed, 2);
        assert_eq!(report.committed.indexed, 1);
        let reader = database::open_read_only(&database.sqlite_path())?;
        assert_eq!(
            database::document_sources(&reader)?,
            [target.to_str().unwrap()]
        );
        assert_eq!(database::live_vector_count(&reader)?, 1);
        Ok(())
    }

    /// Verify committed counts for deduplication, unchanged files, and shared-content deletion.
    ///
    /// Returns success for the expected reports; propagates fixture filesystem errors.
    #[test]
    fn report_distinguishes_new_reused_unchanged_and_removed_documents() -> Result<()> {
        let (directory, database) = corpus()?;
        let a = directory.path().join("a.md");
        let b = directory.path().join("b.md");
        fs::write(&a, "Shared body.")?;
        fs::write(&b, "Shared body.")?;
        let embedder = test_embedder(None, None);
        let first = index(directory.path(), &database, &embedder, false);
        assert!(first.is_success());
        assert_eq!(first.files_seen, 2);
        assert_eq!(
            first.committed,
            IndexCounts {
                indexed: 1,
                reused: 1,
                ..IndexCounts::default()
            }
        );
        assert_eq!(first.staged, first.committed);
        assert_eq!(
            first.publication,
            PublicationStatus::Published {
                snapshot_updated: true,
                compacted: false
            }
        );

        let unchanged = index(directory.path(), &database, &embedder, false);
        assert!(unchanged.is_success());
        assert_eq!(unchanged.committed.unchanged, 2);
        assert_eq!(
            unchanged.publication,
            PublicationStatus::Published {
                snapshot_updated: false,
                compacted: false
            }
        );
        fs::remove_file(a)?;
        let shared = index(directory.path(), &database, &embedder, false);
        assert!(shared.is_success());
        assert_eq!(
            shared.committed,
            IndexCounts {
                unchanged: 1,
                removed: 1,
                ..IndexCounts::default()
            }
        );
        assert_eq!(
            shared.publication,
            PublicationStatus::Published {
                snapshot_updated: false,
                compacted: false
            }
        );
        fs::remove_file(b)?;
        let empty = index(directory.path(), &database, &embedder, false);
        assert!(empty.is_success());
        assert_eq!(empty.files_seen, 0);
        assert_eq!(empty.committed.removed, 1);
        assert_eq!(
            empty.publication,
            PublicationStatus::Published {
                snapshot_updated: true,
                compacted: true
            }
        );
        Ok(())
    }

    /// Verify recoverable file failures publish good documents and count each issue once.
    ///
    /// Returns success on the partial report and stored documents; propagates fixture errors.
    #[test]
    fn report_retains_errors_after_partial_publication() -> Result<()> {
        let (directory, database) = corpus()?;
        let good = directory.path().join("good.md");
        let bad = directory.path().join("bad.md");
        fs::write(&good, "Valid body.")?;
        fs::write(&bad, [0xff])?;
        let report = index(
            directory.path(),
            &database,
            &test_embedder(None, None),
            false,
        );
        assert!(!report.is_success());
        assert!(matches!(
            report.publication,
            PublicationStatus::Published { .. }
        ));
        assert_eq!(report.files_seen, 2);
        assert_eq!(report.committed.indexed, 1);
        assert_eq!(report.staged, report.committed);
        assert_eq!(report.errors.len(), 1);
        assert_eq!(report.errors[0].path.as_deref(), Some(bad.as_path()));
        let connection = database::open_read_only(&database.sqlite_path())?;
        assert_eq!(
            database::document_sources(&connection)?,
            [good.to_str().unwrap()]
        );
        Ok(())
    }

    /// Verify glass mode preserves prepared counts but reports zero committed changes.
    ///
    /// Returns success on the aborted report and rolled-back storage; propagates fixture errors.
    #[test]
    fn glass_report_separates_prepared_work_from_rolled_back_changes() -> Result<()> {
        let (directory, database) = corpus()?;
        fs::write(directory.path().join("a.md"), "First body.")?;
        fs::write(directory.path().join("b.md"), "Second body.")?;
        let report = index(
            directory.path(),
            &database,
            &test_embedder(Some(2), None),
            true,
        );
        assert!(!report.is_success());
        assert_eq!(report.publication, PublicationStatus::NotPublished);
        assert_eq!(report.files_seen, 2);
        assert_eq!(report.staged.indexed, 1);
        assert_eq!(report.committed, IndexCounts::default());
        assert_eq!(report.errors.len(), 1);
        assert!(
            report.errors[0]
                .message
                .contains("injected embedding failure")
        );
        let connection = database::open_read_only(&database.sqlite_path())?;
        assert!(database::document_sources(&connection)?.is_empty());
        assert!(database::vector_index(&connection)?.is_none());
        Ok(())
    }

    /// Verify failed publication emits an issue and never claims staged work committed.
    ///
    /// Returns success on rollback and snapshot cleanup; propagates fixture errors.
    #[test]
    fn publication_failure_returns_an_unpublished_report() -> Result<()> {
        let (directory, database) = corpus()?;
        fs::write(directory.path().join("note.md"), "A body.")?;
        let connection = database::open(&database.sqlite_path())?;
        connection.execute_batch(
            "CREATE TRIGGER fail_publication BEFORE INSERT ON vector_index
             BEGIN SELECT RAISE(ABORT, 'injected publication failure'); END;",
        )?;
        let embedder = test_embedder(None, None);
        let mut events = Vec::new();
        let report = run(
            directory.path(),
            NonZeroUsize::new(1_000).unwrap(),
            false,
            false,
            &database,
            || Ok(&embedder),
            &mut |event| match event {
                ProgressEvent::Index {
                    event: IndexEvent::Staged { .. },
                    staged,
                    seen,
                    ..
                } => {
                    assert_eq!(seen, 1);
                    assert_eq!(staged.indexed, 1);
                    events.push("staged");
                }
                ProgressEvent::SavingSnapshot => events.push("saving"),
                ProgressEvent::Committing => events.push("committing"),
                ProgressEvent::Index {
                    event: IndexEvent::Issue(issue),
                    errors,
                    ..
                } => {
                    assert_eq!(errors, 1);
                    assert!(issue.message.contains("injected publication failure"));
                    events.push("issue");
                }
                _ => {}
            },
        );
        assert_eq!(events, ["staged", "saving", "issue"]);
        assert_eq!(report.publication, PublicationStatus::NotPublished);
        assert!(!report.is_success());
        assert_eq!(report.staged.indexed, 1);
        assert_eq!(report.committed, IndexCounts::default());
        assert_eq!(report.errors.len(), 1);
        assert!(
            report.errors[0]
                .message
                .contains("injected publication failure")
        );
        assert!(database::document_sources(&connection)?.is_empty());
        assert!(database::vector_index(&connection)?.is_none());
        assert!(!fs::read_dir(directory.path())?.any(|entry| {
            entry
                .unwrap()
                .path()
                .extension()
                .is_some_and(|extension| extension == "usearch")
        }));
        Ok(())
    }

    /// Verify vanished prepared files contribute only to removal counts after publication.
    ///
    /// Returns success for reconciled counts; propagates fixture errors.
    #[test]
    fn vanished_prepared_file_is_not_counted_as_indexed() -> Result<()> {
        let (directory, database) = corpus()?;
        let note = directory.path().join("vanishing.md");
        fs::write(&note, "A vanishing body.")?;
        let report = index(
            directory.path(),
            &database,
            &test_embedder(None, Some(note)),
            false,
        );
        assert!(report.is_success());
        assert_eq!(report.files_seen, 1);
        assert_eq!(
            report.committed,
            IndexCounts {
                removed: 1,
                ..IndexCounts::default()
            }
        );
        assert_eq!(report.staged, report.committed);
        Ok(())
    }

    /// Verify traversal and snapshot-load failures each return one diagnostic without publication.
    ///
    /// Returns success on the aborted reports; propagates fixture and metadata errors.
    #[test]
    fn traversal_and_snapshot_failures_return_reports_without_duplicate_errors() -> Result<()> {
        let (directory, database) = corpus()?;
        let embedder = test_embedder(None, None);
        let missing = directory.path().join("missing.md");
        for glass in [false, true] {
            let report = index(&missing, &database, &embedder, glass);
            assert_eq!(report.publication, PublicationStatus::NotPublished);
            assert_eq!(report.errors.len(), 1);
            assert_eq!(report.errors[0].path.as_deref(), Some(missing.as_path()));
            assert_eq!(report.files_seen, 0);
        }
        let note = directory.path().join("note.md");
        fs::write(&note, "A body.")?;
        assert!(index(&note, &database, &embedder, false).is_success());
        let connection = database::open_read_only(&database.sqlite_path())?;
        let active = database::vector_index(&connection)?.unwrap();
        let snapshot = database.path.join(&active.file_name);
        fs::remove_file(&snapshot)?;
        let report = index(&note, &database, &embedder, false);
        assert_eq!(report.publication, PublicationStatus::NotPublished);
        assert_eq!(report.errors.len(), 1);
        assert_eq!(report.errors[0].path.as_deref(), Some(snapshot.as_path()));
        assert_eq!(report.staged, IndexCounts::default());
        assert_eq!(report.committed, IndexCounts::default());
        Ok(())
    }

    /// Verify that deferral preserves history and a later eligible pass rebuilds once.
    ///
    /// Returns success for the expected working index and counters; propagates
    /// SQLite and USearch setup or rebuild errors. Fixture chunks have explicit
    /// source locations, independent of their compaction bookkeeping.
    #[test]
    fn automatic_compaction_respects_deferral_and_resets_history() -> Result<()> {
        let connection = database::open(Path::new(":memory:"))?;
        connection.execute(
            "INSERT INTO documents (source, content_hash) VALUES ('live.md', 'live')",
            [],
        )?;
        let dimensions = NonZeroUsize::new(3).unwrap();
        let mut index = snapshot::new_index(dimensions)?;
        index.reserve(301)?;
        for key in 1_u64..=300 {
            database::try_insert_embedding(&connection, key, &blake3::hash(&key.to_le_bytes()))?;
            connection.execute(
                "INSERT INTO chunks
                     (vector_key, content_hash, ordinal, byte_start, byte_end, line_start, line_end)
                 VALUES (?1, 'live', ?1, 0, 1, 1, 1)",
                [key.cast_signed()],
            )?;
            index.add(key, &[1.0_f32, 0.0, 0.0])?;
        }
        index.add(999, &[0.0_f32, 1.0, 0.0])?;
        let mut emit = |_: ProgressEvent<'_>| {};
        let progress = IndexProgress::new(&mut emit);
        let mut vectors = VectorState {
            removed_since_compact: 99,
        };
        assert!(!compact_if_needed(
            &connection,
            &mut index,
            &mut vectors,
            dimensions,
            false,
            &progress,
        )?);
        vectors.removed_since_compact = 100;
        assert!(!compact_if_needed(
            &connection,
            &mut index,
            &mut vectors,
            dimensions,
            true,
            &progress,
        )?);
        assert_eq!(vectors.removed_since_compact, 100);
        assert_eq!(index.size(), 301);

        // No new removals are needed once a previous pass has reached the threshold.
        assert!(compact_if_needed(
            &connection,
            &mut index,
            &mut vectors,
            dimensions,
            false,
            &progress,
        )?);
        assert_eq!(vectors.removed_since_compact, 0);
        assert_eq!(index.size(), 300);
        let mut vector = vec![0.0_f32; 3];
        assert_eq!(index.get(999, &mut vector)?, 0);
        assert!(!compact_if_needed(
            &connection,
            &mut index,
            &mut vectors,
            dimensions,
            false,
            &progress,
        )?);
        Ok(())
    }

    /// Verify that failed file inserts count graph removals but restore old chunks.
    ///
    /// Returns success when Staged and RolledBack retain their distinct semantics,
    /// preserve the old vector, and track removal of the new vector; propagates
    /// SQLite and USearch setup errors.
    #[test]
    fn failed_file_rollback_counts_only_removed_new_vectors() -> Result<()> {
        let mut connection = database::open(Path::new(":memory:"))?;
        let mut transaction = connection.transaction()?;
        database::track_removed_vectors(&transaction)?;
        let index = snapshot::new_index(NonZeroUsize::new(3).unwrap())?;
        let chunk_size = NonZeroUsize::new(5).unwrap();
        let mut working = WorkingIndex {
            index,
            vectors: VectorState {
                removed_since_compact: 0,
            },
            previous: None,
            changed: false,
        };
        assert!(matches!(
            stage_file(
                &mut transaction,
                &mut working,
                Path::new("note.md"),
                "old",
                chunk_size,
                None,
                |_, _| Ok(vec![1.0_f32, 0.0, 0.0]),
            )?,
            StageOutcome::Staged(IndexStatus::Indexed),
        ));
        let result = stage_file(
            &mut transaction,
            &mut working,
            Path::new("note.md"),
            "first\n\nsecond",
            chunk_size,
            None,
            |_, ordinal| {
                if ordinal == 2 {
                    Err(io::Error::other("embedding failed").into())
                } else {
                    Ok(vec![0.0_f32, 1.0, 0.0])
                }
            },
        )?;
        assert!(matches!(
            result,
            StageOutcome::RolledBack(error) if error.to_string().contains("embedding failed")
        ));
        assert_eq!(working.vectors.removed_since_compact, 1);
        assert_eq!(working.index.size(), 1);
        let old_key = database::embedding_key(&transaction, &blake3::hash(b"old"))?.unwrap();
        assert_eq!(database::chunk_sources(&transaction, old_key)?.len(), 1);
        assert!(database::embedding_key(&transaction, &blake3::hash(b"first"))?.is_none());
        assert!(database::removed_vector_keys(&transaction)?.is_empty());
        Ok(())
    }
}

#[cfg(test)]
#[path = "indexer/storage_tests.rs"]
mod storage_tests;
