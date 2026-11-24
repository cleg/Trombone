//! Indexes files in SQLite and maintains the corresponding USearch snapshot.
//!
//! Each pass stages document changes and new vectors in SQLite, then removes
//! records for unavailable sources. When vectors change or automatic compaction
//! is due, it saves one snapshot before committing the documents, counters, and
//! snapshot reference together.

mod document;
mod paths;
mod publication;

use std::fs;
use std::io;
use std::num::NonZeroUsize;
use std::path::Path;

use crate::config::DatabaseConfig;
use crate::database::{self, Result};
use crate::embeddings::Embedder;
use crate::index_report::{IndexProgress, IndexReport, PublicationStatus};
use crate::progress::{IndexEvent, ProgressEvent};
use crate::snapshot;
use crate::walker::{WalkOutcome, Walker};

use self::document::process_file;
use self::paths::remove_missing_documents;
use self::publication::publish_index;

/// Index the files under `root` and remove unavailable sources in that scope.
///
/// # Parameters
///
/// - `root` is the existing file or directory to index; symlinks resolve to its
///   canonical absolute identity before opening storage.
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
    root: &Path,
    chunk_size: NonZeroUsize,
    no_compact: bool,
    database: &DatabaseConfig,
    embedder: &dyn Embedder,
    emit: &mut dyn FnMut(ProgressEvent<'_>),
) -> IndexReport {
    let progress = IndexProgress::new(emit);
    let root = match fs::canonicalize(root) {
        Ok(root) => root,
        Err(error) => {
            progress.issue(Some(root), &error);
            return progress.finish(PublicationStatus::NotPublished);
        }
    };
    progress.notify(IndexEvent::Opening(&root));
    let publication = match execute(&root, chunk_size, no_compact, database, embedder, &progress) {
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
/// `root` defines the input scope, `chunk_size` limits chunk length in Unicode
/// characters, and `no_compact` defers automatic compaction. `database` locates
/// storage, `embedder` supplies model settings and chunk embeddings, and
/// `progress` records counters and emits events.
/// Returns the publication outcome. Recoverable errors allow traversal to
/// continue; already-reported fatal errors return NotPublished.
/// Validates the pass's chunk size before loading vectors or visiting sources.
/// Checks the active snapshot path during setup and loads its graph only when
/// publication needs a replacement.
///
/// # Errors
///
/// Returns unreported setup, database, vector cleanup, rebuild, and publication
/// errors for the outer entry point to add to the report.
fn execute(
    root: &Path,
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
    progress.notify(IndexEvent::Scanning);
    let walked = Walker {
        report_error: &|path, error| progress.issue(Some(path), error),
    }
    .walk(root, |path| {
        process_file(path, &mut transaction, chunk_size, embedder, progress)
    });
    match walked {
        Ok(WalkOutcome::Completed(input_kind)) => {
            remove_missing_documents(&transaction, root, input_kind, progress)?;
        }
        Ok(WalkOutcome::StoppedByProcessor) | Err(_) => return Ok(PublicationStatus::NotPublished),
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
