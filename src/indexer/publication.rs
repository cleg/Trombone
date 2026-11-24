//! Applies staged vector changes and publishes the vector snapshot and SQLite transaction.

use std::io;
use std::path::Path;

use rusqlite::{Connection, Transaction};
use usearch::Index;

use crate::compact;
use crate::config::DatabaseConfig;
use crate::database::{self, Result, VectorIndex};
use crate::embeddings::EmbeddingSettings;
use crate::index_report::{IndexProgress, PublicationStatus};
use crate::progress::IndexEvent;
use crate::snapshot::{self, utf8_path};

/// Remove unreferenced embedding rows and staging data from `connection`.
///
/// Returns the keys that must also be removed from the published graph.
/// Runs after the last file savepoint and path reconciliation so shared
/// vectors can be reused during the pass.
///
/// # Errors
///
/// Returns an error if SQLite cannot read dead keys or delete their rows.
fn prune_obsolete_embeddings(connection: &Connection) -> Result<Vec<u64>> {
    let mut removed = Vec::new();
    for key in database::removed_vector_keys(connection)? {
        if !database::discard_staged_vector(connection, key)? {
            removed.push(key);
        }
        database::delete_embedding(connection, key)?;
    }
    Ok(removed)
}

/// Apply the final vector changes and publish the completed `transaction`.
///
/// `database` locates snapshots, `settings` identifies compatible vectors,
/// and `previous` is the currently active snapshot path, if any.
/// `removed_since_compact` is its committed removal count. `no_compact` defers
/// rebuilding and `progress` emits diagnostic and publication events.
/// Returns the publication outcome and, on success, whether it updated and
/// compacted the snapshot. A pass without vector changes or compaction commits
/// its rows without loading the graph.
///
/// # Errors
///
/// Returns unreported SQL, vector, rebuild, snapshot-write, or commit failures.
/// A reported snapshot-load failure returns NotPublished. Both cases preserve
/// the previously committed index.
pub(super) fn publish_index(
    transaction: Transaction<'_>,
    database: &DatabaseConfig,
    settings: &EmbeddingSettings,
    previous: Option<&Path>,
    mut removed_since_compact: i64,
    no_compact: bool,
    progress: &IndexProgress<'_>,
) -> Result<PublicationStatus> {
    let removed = prune_obsolete_embeddings(&transaction)?;
    removed_since_compact = removed_since_compact
        .checked_add(i64::try_from(removed.len())?)
        .ok_or_else(|| io::Error::other("Vector removal counter exhausted"))?;
    let staged = database::staged_vector_count(&transaction)?;
    let compacted = !no_compact
        && removed_since_compact > 0
        && compact::should_compact(
            database::live_vector_count(&transaction)?,
            removed_since_compact,
        );
    let snapshot_updated = staged > 0 || !removed.is_empty() || compacted;
    let updated_index = if snapshot_updated {
        let dimensions = settings.dimensions;
        let mut index = snapshot::new_index(dimensions)?;
        if let Some(path) = previous {
            if let Err(error) = index.load(utf8_path(path)?) {
                progress.issue(Some(path), &format!("Cannot load active index: {error}"));
                return Ok(PublicationStatus::NotPublished);
            }
            snapshot::validate(&index, dimensions)?;
        }

        let capacity = index
            .size()
            .checked_add(staged)
            .ok_or_else(|| io::Error::other("Vector capacity exhausted"))?;
        if capacity > index.capacity() {
            index.reserve(capacity)?;
        }
        database::visit_staged_vectors(&transaction, dimensions, |key, vector| {
            index.add(key, vector)?;
            Ok(())
        })?;
        for key in removed {
            if index.remove(key)? != 1 {
                return Err(io::Error::other(format!("Could not remove vector {key}")).into());
            }
        }
        if compacted {
            progress.notify(IndexEvent::Compacting);
            let keys = database::live_vector_keys(&transaction)?;
            index = compact::rebuild(&index, &keys, dimensions)?;
            removed_since_compact = 0;
        }
        Some(index)
    } else {
        None
    };

    commit_indexing(
        transaction,
        updated_index.as_ref(),
        database,
        settings,
        removed_since_compact,
        previous,
        progress,
    )?;
    Ok(PublicationStatus::Published {
        snapshot_updated,
        compacted,
    })
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
/// - `removed_since_compact` is the accumulated removal count to persist.
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
    removed_since_compact: i64,
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
                removed_since_compact,
            },
        )
    });
    snapshot::commit(transaction, update, database, previous, &mut |event| {
        progress.emit(event)
    })?;
    Ok(())
}

#[cfg(test)]
mod tests;
