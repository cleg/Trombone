//! Indexes files in SQLite and maintains the corresponding USearch snapshot.
//!
//! Each pass stages document changes and new vectors in SQLite, then removes
//! records absent from a complete walk of the configured roots. When vectors
//! change or automatic compaction is due, it saves one snapshot before committing
//! the documents, counters, and snapshot reference together.

mod document;
mod publication;

use std::fs;
use std::io;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use crate::config::DatabaseConfig;
use crate::database::{self, Result};
use crate::embeddings::Embedder;
use crate::index_report::{IndexProgress, IndexReport, PublicationStatus};
use crate::progress::{IndexEvent, ProgressEvent};
use crate::snapshot;
use crate::walker::{self, WalkOutcome};

use self::document::process_file;
use self::publication::publish_index;

/// Index the files reachable from `roots` and reconcile the entire database.
///
/// # Parameters
///
/// - `roots` is the validated list of configured source directories.
/// - `chunk_size` is the maximum chunk length in Unicode characters.
/// - `no_compact` defers automatic rebuilding while preserving removal history.
/// - `database` provides the SQLite and snapshot location.
/// - `embedder` supplies model settings and generates chunk embeddings.
/// - `emit` receives synchronous progress and diagnostic events.
///
/// Returns an IndexReport for every attempt, including setup and publication
/// failures. Recoverable errors can coexist with a published pass. Aborted passes
/// keep their staged counts, report zero committed counts, and retain their
/// diagnostics without requiring callers to interpret nested results.
pub(crate) fn run(
    roots: &[PathBuf],
    chunk_size: NonZeroUsize,
    no_compact: bool,
    database: &DatabaseConfig,
    embedder: &dyn Embedder,
    emit: &mut dyn FnMut(ProgressEvent<'_>),
) -> IndexReport {
    let progress = IndexProgress::new(emit);
    if roots.is_empty() {
        progress.issue(None, &"At least one source directory is required");
        return progress.finish(PublicationStatus::NotPublished);
    }
    progress.notify(IndexEvent::Opening(&database.path));
    let publication = match execute(roots, chunk_size, no_compact, database, embedder, &progress) {
        Ok(publication) => publication,
        Err(error) => {
            progress.issue(None, error.as_ref());
            PublicationStatus::NotPublished
        }
    };
    progress.finish(publication)
}

/// Coordinate preparation, traversal, deletion reconciliation, and publication.
///
/// `roots` defines the desired documents for the database. `chunk_size` limits
/// chunk length in Unicode characters and `no_compact` defers automatic compaction.
/// `database` locates storage, `embedder` supplies model settings and chunk
/// embeddings, and `progress` records counters and emits events.
/// Returns the publication outcome. Recoverable errors allow traversal to
/// continue. An unopened set of roots or a reported snapshot failure returns
/// NotPublished.
/// Validates the pass's chunk size before loading vectors or visiting sources.
/// Checks the active snapshot path during setup and loads its graph only when
/// publication needs a replacement.
///
/// # Errors
///
/// Returns unreported setup, database, vector cleanup, rebuild, and publication
/// errors for the outer entry point to add to the report.
fn execute(
    roots: &[PathBuf],
    chunk_size: NonZeroUsize,
    no_compact: bool,
    database: &DatabaseConfig,
    embedder: &dyn Embedder,
    progress: &IndexProgress<'_>,
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
        && let Err(error) = fs::metadata(path).and_then(|metadata| {
            if metadata.is_file() {
                Ok(())
            } else {
                Err(io::Error::other("Expected a regular snapshot file"))
            }
        })
    {
        progress.issue(Some(path), &format!("Cannot access active index: {error}"));
        return Ok(PublicationStatus::NotPublished);
    }
    database::track_removed_vectors(&transaction)?;
    database::create_vector_staging(&transaction)?;
    database::create_source_tracking(&transaction)?;
    let mut any_opened = false;
    let mut complete = true;
    for root in roots {
        progress.notify(IndexEvent::Scanning(root));
        let walked = walker::walk(
            root,
            |path| {
                if database::mark_source_seen(&transaction, path)? {
                    process_file(path, &mut transaction, chunk_size, embedder, progress)?;
                }
                Ok(())
            },
            |path, error| {
                progress.defer_deletions();
                progress.issue(Some(path), error);
            },
        )?;
        any_opened |= !matches!(walked, WalkOutcome::Unopened);
        complete &= matches!(walked, WalkOutcome::Complete);
    }
    if !any_opened {
        return Ok(PublicationStatus::NotPublished);
    }
    if complete {
        progress.notify(IndexEvent::CheckingDeletions);
        database::delete_unseen_documents(&transaction, |path| progress.removal(path))?;
    }
    publish_index(
        transaction,
        database,
        settings,
        previous.as_deref(),
        active
            .as_ref()
            .map_or(0, |active| active.removed_since_compact),
        no_compact,
        progress,
    )
}

#[cfg(test)]
mod tests;
