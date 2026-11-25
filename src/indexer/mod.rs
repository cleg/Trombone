//! Indexes files in SQLite and maintains the corresponding USearch snapshot.
//!
//! Each pass stages document changes and new vectors in SQLite, then removes
//! records absent from a complete walk of the configured roots. Vector changes
//! publish with the documents and their snapshot reference. Automatic
//! compaction may run afterward in its own transaction.

mod document;
mod publication;

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::PathBuf;

use crate::compact;
use crate::config::DatabaseConfig;
use crate::database;
use crate::embeddings::Embedder;
use crate::error::Result;
use crate::index_report::{IndexProgress, IndexReport, PublicationStatus};
use crate::progress::{IndexEvent, ProgressEvent};
use crate::snapshot;
use crate::walker::{self, WalkEvent, WalkOutcome};

use self::document::process_file;
use self::publication::publish_index;

/// Index the files reachable from `roots` and reconcile the entire database.
///
/// # Parameters
///
/// - `roots` is the validated list of configured source directories.
/// - `chunk_size` is the maximum chunk length in Unicode characters.
/// - `max_file_bytes` is the maximum source file size in bytes read into memory.
/// - `no_compact` skips automatic rebuilding for this pass.
/// - `database` provides the SQLite and snapshot location.
/// - `embedder` supplies model settings and generates chunk embeddings.
/// - `emit` receives synchronous progress and diagnostic events.
///
/// Returns an IndexReport for every attempt, including setup and publication
/// failures. Recoverable errors can coexist with a published pass. Aborted passes
/// keep their staged counts, report zero committed counts, and retain their
/// diagnostics without requiring callers to interpret nested results.
/// Automatic compaction failures emit a warning after successful publication.
pub(crate) fn run(
    roots: &[PathBuf],
    chunk_size: NonZeroUsize,
    max_file_bytes: NonZeroU64,
    no_compact: bool,
    database: &DatabaseConfig,
    embedder: &dyn Embedder,
    emit: &mut dyn FnMut(ProgressEvent<'_>),
) -> IndexReport {
    let mut progress = IndexProgress::new(emit);
    if roots.is_empty() {
        progress.issue(None, &"At least one source directory is required");
        return progress.finish(PublicationStatus::NotPublished);
    }
    progress.notify(IndexEvent::Opening(&database.path));
    let mut publication = match execute(
        roots,
        chunk_size,
        max_file_bytes,
        database,
        embedder,
        &mut progress,
    ) {
        Ok(publication) => publication,
        Err(error) => {
            progress.issue(None, error.as_ref());
            PublicationStatus::NotPublished
        }
    };
    if !no_compact
        && let PublicationStatus::Published {
            snapshot_updated,
            compacted,
        } = &mut publication
    {
        match compact::run_if_due(database, &mut |event| progress.emit(event)) {
            Ok(Some(_)) => {
                *compacted = true;
                *snapshot_updated = true;
            }
            Ok(None) => {}
            Err(error) => progress.emit(ProgressEvent::AutomaticCompactionFailed(error.as_ref())),
        }
    }
    progress.finish(publication)
}

/// Coordinate preparation, traversal, deletion reconciliation, and publication.
///
/// `roots` defines the desired documents for the database. `chunk_size` limits
/// chunk length in Unicode characters, and `max_file_bytes` caps each source
/// file read into memory.
/// `database` locates storage, `embedder` supplies model settings and chunk
/// embeddings, and `progress` records counters, deletion deferral, and events.
/// Returns the publication outcome. Recoverable errors allow traversal to
/// continue. An unopened set of roots or a reported snapshot failure returns
/// NotPublished.
/// Validates the pass's chunk size before loading vectors or visiting sources.
/// Checks the active snapshot header and removes orphaned snapshots during setup.
/// Loads the active graph only when publication needs a replacement.
///
/// # Errors
///
/// Returns unreported setup, database, vector cleanup, and publication
/// errors for the outer entry point to add to the report.
fn execute(
    roots: &[PathBuf],
    chunk_size: NonZeroUsize,
    max_file_bytes: NonZeroU64,
    database: &DatabaseConfig,
    embedder: &dyn Embedder,
    progress: &mut IndexProgress<'_>,
) -> Result<PublicationStatus> {
    let mut connection = database::open(&database.sqlite_path())?;
    // Acquire the writer lock before choosing a snapshot or allocating vector keys.
    // Keep it through traversal, embedding calls, and publication.
    let mut transaction = database::begin_write(&mut connection)?;
    database::ensure_chunk_size(&transaction, chunk_size)?;
    let settings = embedder.settings();
    let active = database::vector_index(&transaction)?;
    let previous = active
        .as_ref()
        .map(|active| snapshot::active_path(&database.path, active, settings))
        .transpose()?;
    if let Some(path) = &previous
        && let Err(error) = snapshot::read_header(path, settings.dimensions)
    {
        progress.issue(
            Some(path),
            &format!("Cannot validate active index: {}", error.reason()),
        );
        return Ok(PublicationStatus::NotPublished);
    }
    snapshot::cleanup_orphans(&database.path, previous.as_deref(), &mut |event| {
        progress.emit(event);
    });
    database::track_removed_vectors(&transaction)?;
    database::create_vector_staging(&transaction)?;
    database::create_source_tracking(&transaction)?;
    let mut any_opened = false;
    let mut all_complete = true;
    for root in roots {
        progress.notify(IndexEvent::Scanning(root));
        let outcome = walker::walk(root, |event| {
            match event {
                WalkEvent::File(path) => {
                    if database::mark_source_seen(&transaction, path)? {
                        process_file(
                            path,
                            &mut transaction,
                            chunk_size,
                            max_file_bytes,
                            embedder,
                            progress,
                        )?;
                    }
                }
                WalkEvent::Issue { path, error } => progress.issue(Some(path), error),
            }
            Ok(())
        })?;
        any_opened |= !matches!(outcome, WalkOutcome::Unopened);
        if !matches!(outcome, WalkOutcome::Complete) {
            all_complete = false;
            progress.defer_deletions();
        }
    }
    if !any_opened {
        return Ok(PublicationStatus::NotPublished);
    }
    if all_complete {
        progress.notify(IndexEvent::CheckingDeletions);
        database::delete_unseen_documents(&transaction, |path| progress.removal(path))?;
    }
    publish_index(
        transaction,
        database,
        settings,
        previous.as_deref(),
        progress,
    )
}

#[cfg(test)]
mod tests;
