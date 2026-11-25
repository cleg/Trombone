//! Applies staged vector changes and publishes the vector snapshot and SQLite transaction.

use std::io;
use std::path::Path;

use rusqlite::{Connection, Transaction};

use crate::config::DatabaseConfig;
use crate::database;
use crate::embeddings::EmbeddingSettings;
use crate::error::Result;
use crate::index_report::{IndexProgress, PublicationStatus};
use crate::snapshot::{self, OpenMode};

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
/// `progress` emits diagnostic and publication events. Returns whether indexing published and
/// updated the snapshot. A pass without vector changes commits its rows
/// without loading the graph.
///
/// # Errors
///
/// Returns unreported SQL, vector, snapshot-write, or commit failures.
/// A reported snapshot-load failure returns NotPublished. Both cases preserve
/// the previously committed index.
pub(super) fn publish_index(
    transaction: Transaction<'_>,
    database: &DatabaseConfig,
    settings: &EmbeddingSettings,
    previous: Option<&Path>,
    progress: &mut IndexProgress<'_>,
) -> Result<PublicationStatus> {
    let removed = prune_obsolete_embeddings(&transaction)?;
    let staged = database::staged_vector_count(&transaction)?;
    let snapshot_updated = staged > 0 || !removed.is_empty();
    let updated_index = if snapshot_updated {
        let dimensions = settings.dimensions;
        let index = match previous {
            Some(path) => match snapshot::open(path, dimensions, OpenMode::Load) {
                Ok(index) => index,
                Err(error) => {
                    progress.issue(
                        Some(path),
                        &format!("Cannot load active index: {}", error.reason()),
                    );
                    return Ok(PublicationStatus::NotPublished);
                }
            },
            None => snapshot::new_index(dimensions)?,
        };

        for key in removed {
            if index.remove(key)? != 1 {
                return Err(io::Error::other(format!("Could not remove vector {key}")).into());
            }
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
        Some(index)
    } else {
        None
    };

    snapshot::commit(
        transaction,
        updated_index.as_ref().map(|index| (index, settings)),
        database,
        previous,
        &mut |event| progress.emit(event),
    )?;
    Ok(PublicationStatus::Published {
        snapshot_updated,
        compacted: false,
    })
}

#[cfg(test)]
mod tests;
