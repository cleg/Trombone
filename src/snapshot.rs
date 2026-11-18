//! Defines the USearch index settings and resolves snapshot files recorded in SQLite.
//!
//! The active snapshot's metadata identifies the embedding model and digest, dimensions,
//! and chunk format used to build it. Indexing and search check that metadata
//! before opening the snapshot and validate the index settings afterward.

use std::cell::RefCell;
use std::fs;
use std::io;
use std::mem::DropGuard;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use rusqlite::Transaction;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::config::DatabaseConfig;
use crate::database::{self, Result, VectorIndex};
use crate::embeddings::EmbeddingSettings;
use crate::progress::ProgressEvent;

/// Prefix shared by newly created USearch snapshot filenames.
pub(crate) const INDEX_PREFIX: &str = "trombone-index-";
/// Extension shared by USearch snapshot filenames.
pub(crate) const INDEX_SUFFIX: &str = ".usearch";

/// Create an empty cosine-similarity index with `F32` vectors and one vector per key.
///
/// `dimensions` is the number of components in each vector.
///
/// # Errors
///
/// Returns an error if USearch cannot initialize the index.
pub(crate) fn new_index(dimensions: NonZeroUsize) -> Result<Index> {
    Ok(Index::new(&IndexOptions {
        dimensions: dimensions.get(),
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        multi: false,
        ..IndexOptions::default()
    })?)
}

/// Check that an opened USearch index uses the expected settings.
///
/// - `index` is the loaded or memory-mapped snapshot to check.
/// - `dimensions` is the expected number of components in each vector.
///
/// # Errors
///
/// Returns an error if the dimensions differ or the index settings do not
/// specify cosine similarity, `F32` vectors, and one vector per key.
pub(crate) fn validate(index: &Index, dimensions: NonZeroUsize) -> io::Result<()> {
    if index.dimensions() != dimensions.get()
        || index.metric_kind() != MetricKind::Cos
        || index.scalar_kind() != ScalarKind::F32
        || index.multi()
    {
        return Err(io::Error::other(
            "Active USearch index has incompatible settings",
        ));
    }
    Ok(())
}

/// Return the path of the active snapshot after checking its metadata.
///
/// - `directory` is the directory containing SQLite and the snapshot files.
/// - `active` is the snapshot metadata read from SQLite.
/// - `settings` supplies the embedder's model, digest, dimensions, and chunk format.
///
/// Accepts older snapshot filenames with a leading dot. The returned path is
/// constructed without accessing the filesystem.
///
/// # Errors
///
/// Returns an error if the model, digest, dimensions, or chunk format differ from the
/// stored values, the dimensions exceed SQLite's integer range, or the stored
/// name fails the snapshot filename checks.
pub(crate) fn active_path(
    directory: &Path,
    active: &VectorIndex,
    settings: &EmbeddingSettings,
) -> Result<PathBuf> {
    if active.model != settings.model
        || active.dimensions != i64::try_from(settings.dimensions.get())?
    {
        return Err(io::Error::other(
            "Embedding model or dimensions differ from the active index; use its settings or a new database path",
        )
        .into());
    }
    if active.model_digest != settings.model_digest {
        return Err(io::Error::other(format!(
            "Embedding model digest differs from the active index (indexed {}, current {}); restore the indexed model or rebuild using a fresh database path",
            active.model_digest, settings.model_digest,
        )).into());
    }
    if active.chunk_format != settings.chunk_format {
        return Err(io::Error::other(
            "ollama.chunk_format differs from the active index; use its setting or a new database path",
        )
        .into());
    }
    stored_path(directory, active)
}

/// Resolve a snapshot filename without comparing it to embedding configuration.
///
/// `directory` contains the snapshots and `active` supplies the stored filename.
/// Returns a path without accessing the filesystem. This allows maintenance to
/// use the index's own settings without an embedding service.
///
/// # Errors
///
/// Returns an error if the stored filename fails the snapshot filename checks.
pub(crate) fn stored_path(directory: &Path, active: &VectorIndex) -> Result<PathBuf> {
    let name = &active.file_name;
    // Older snapshots have a leading dot in their filename.
    if !name
        .strip_prefix('.')
        .unwrap_or(name)
        .starts_with(INDEX_PREFIX)
        || !name.ends_with(INDEX_SUFFIX)
        || Path::new(name).file_name().and_then(|name| name.to_str()) != Some(name)
    {
        return Err(io::Error::other("Invalid active USearch index file name").into());
    }
    Ok(directory.join(name))
}

/// Borrow the UTF-8 representation of `path` for the USearch API.
///
/// # Errors
///
/// Returns an error of kind [`io::ErrorKind::InvalidInput`] if `path` cannot
/// be represented as UTF-8.
pub(crate) fn utf8_path(path: &Path) -> io::Result<&str> {
    path.to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Index path must be UTF-8"))
}

/// Publish a new vector snapshot and commit its metadata atomically with SQLite.
///
/// - `transaction` contains the staged document changes.
/// - `update` supplies the index and metadata when vectors changed. The generated
///   snapshot filename replaces the metadata's `file_name` before publication.
/// - `database` supplies the database and snapshot locations.
/// - `previous` is the old snapshot to remove after a successful commit.
/// - `emit` receives save and commit phases, and any cleanup warnings.
///
/// Returns success after committing. Old snapshot cleanup errors are reported
/// without failing the committed operation. On failure, a new snapshot is removed
/// only if committed SQLite metadata confirms it is unreferenced.
///
/// # Errors
///
/// Returns an error if saving, syncing, staging metadata, or committing fails.
pub(crate) fn commit(
    transaction: Transaction<'_>,
    update: Option<(&Index, VectorIndex)>,
    database: &DatabaseConfig,
    previous: Option<&Path>,
    emit: &mut dyn FnMut(ProgressEvent<'_>),
) -> Result<()> {
    // The pending snapshot's guard and the normal path both emit synchronously.
    let emit = RefCell::new(emit);
    let database_path = database.sqlite_path();
    let pending_snapshot = if let Some((index, mut metadata)) = update {
        (emit.borrow_mut())(ProgressEvent::SavingSnapshot);
        let file = tempfile::Builder::new()
            .prefix(INDEX_PREFIX)
            .suffix(INDEX_SUFFIX)
            .tempfile_in(&database.path)?;
        index.save(utf8_path(file.path())?)?;
        file.as_file().sync_all()?;
        metadata.file_name = utf8_path(Path::new(file.path().file_name().unwrap()))?.to_owned();
        let (_, path) = file.keep()?;
        let guard = DropGuard::new(path, |path| {
            if let Err(error) = remove_unreferenced_snapshot(&database_path, &path) {
                (emit.borrow_mut())(ProgressEvent::SnapshotCleanupFailed {
                    path: &path,
                    error: error.as_ref(),
                });
            }
        });
        // Make the snapshot and its directory entry durable before publishing its name.
        fs::File::open(&database.path)?.sync_all()?;
        database::publish_vector_index(&transaction, &metadata)?;
        Some(guard)
    } else {
        None
    };

    (emit.borrow_mut())(ProgressEvent::Committing);
    transaction.commit()?;
    let saved = pending_snapshot.map(DropGuard::dismiss).is_some();
    if saved
        && let Some(path) = previous
        && let Err(error) = fs::remove_file(path)
    {
        (emit.borrow_mut())(ProgressEvent::SnapshotCleanupFailed {
            path,
            error: &error,
        });
    }
    Ok(())
}

/// Remove `path` only if committed metadata in `database_path` leaves it unreferenced.
///
/// Returns success if the snapshot is active or was removed.
///
/// # Errors
///
/// Returns an error if reading metadata or removing the unreferenced file fails.
fn remove_unreferenced_snapshot(database_path: &Path, path: &Path) -> Result<()> {
    let connection = database::open_read_only(database_path)?;
    if let Some(active) = database::vector_index(&connection)?
        && path.file_name() == Some(std::ffi::OsStr::new(&active.file_name))
    {
        return Ok(());
    }
    fs::remove_file(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify cleanup warnings arrive through events for both publication outcomes.
    ///
    /// Returns success if old-file cleanup preserves a successful commit and
    /// failed-commit cleanup preserves the original SQLite error. Propagates
    /// unexpected fixture, filesystem, or database failures.
    #[test]
    fn cleanup_warnings_preserve_publication_outcome() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let database = DatabaseConfig {
            path: directory.path().to_path_buf(),
        };
        let dimensions = NonZeroUsize::new(3).unwrap();
        let index = new_index(dimensions)?;
        let mut connection = database::open(&database.sqlite_path())?;

        // Removal through remove_file must fail for a directory, regardless of permissions.
        let previous = database.path.join("trombone-index-old.usearch");
        fs::create_dir(&previous)?;
        let metadata = VectorIndex {
            format_version: database::VECTOR_INDEX_VERSION,
            file_name: previous.file_name().unwrap().to_str().unwrap().into(),
            model: "test-model".into(),
            model_digest: "a".repeat(64),
            chunk_format: None,
            dimensions: 3,
            removed_since_compact: 0,
        };
        database::publish_vector_index(&connection, &metadata)?;
        let mut phases = Vec::new();
        let mut warnings = Vec::new();
        commit(
            connection.transaction()?,
            Some((&index, metadata)),
            &database,
            Some(&previous),
            &mut |event| match event {
                ProgressEvent::SavingSnapshot => phases.push("saving"),
                ProgressEvent::Committing => phases.push("committing"),
                ProgressEvent::SnapshotCleanupFailed { path, .. } => {
                    phases.push("cleanup warning");
                    warnings.push(path.to_path_buf());
                }
                ProgressEvent::Index { .. } => unreachable!(),
            },
        )?;
        assert_eq!(phases, ["saving", "committing", "cleanup warning"]);
        assert_eq!(warnings.as_slice(), std::slice::from_ref(&previous));
        let active = database::vector_index(&connection)?.unwrap();
        let active_file = stored_path(&database.path, &active)?;
        assert_ne!(active_file, previous);
        assert!(active_file.is_file());

        connection.execute_batch(
            "CREATE TABLE parent (id INTEGER PRIMARY KEY);
             CREATE TABLE child (
                 parent_id INTEGER REFERENCES parent(id) DEFERRABLE INITIALLY DEFERRED
             );",
        )?;
        let transaction = connection.transaction()?;
        transaction.execute("INSERT INTO child (parent_id) VALUES (1)", [])?;
        let mut pending = None;
        let mut cleanup_path = None;
        let failure = commit(
            transaction,
            Some((&index, active)),
            &database,
            Some(&active_file),
            &mut |event| match event {
                ProgressEvent::Committing => {
                    // Simulate an external filesystem change before the deferred commit failure.
                    let path = fs::read_dir(&database.path)
                        .unwrap()
                        .map(|entry| entry.unwrap().path())
                        .find(|path| {
                            path != &active_file
                                && path.is_file()
                                && path.extension().is_some_and(|ext| ext == "usearch")
                        })
                        .unwrap();
                    fs::remove_file(&path).unwrap();
                    pending = Some(path);
                }
                ProgressEvent::SnapshotCleanupFailed { path, .. } => {
                    cleanup_path = Some(path.to_path_buf());
                }
                _ => {}
            },
        )
        .unwrap_err();
        assert!(
            failure
                .to_string()
                .contains("FOREIGN KEY constraint failed")
        );
        assert!(cleanup_path.is_some());
        assert_eq!(cleanup_path, pending);
        let active = database::vector_index(&connection)?.unwrap();
        assert_eq!(stored_path(&database.path, &active)?, active_file);
        assert!(active_file.is_file());
        Ok(())
    }
}
