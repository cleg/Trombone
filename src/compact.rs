//! Rebuilds USearch to reclaim deleted slots and drop vectors SQLite no longer references.

use std::io;
use std::num::NonZeroUsize;

use usearch::Index;

use crate::config::DatabaseConfig;
use crate::database;
use crate::error::Result;
use crate::progress::ProgressEvent;
use crate::snapshot::{self, OpenMode};

/// Counts of searchable vectors before and after a committed compaction.
pub(crate) struct CompactReport {
    pub(crate) previous_vectors: usize,
    pub(crate) retained_vectors: usize,
}

/// Rebuild and publish the active vector snapshot using existing embeddings.
///
/// `database` locates SQLite and its vector snapshots. `emit` receives snapshot
/// publication events. Uses stored model settings
/// and preserves the model digest and vector keys. Holds the writer lock
/// through publication so indexing cannot change the starting snapshot.
/// Returns committed vector counts, or `None` when there is no active index.
/// Removes orphaned snapshot files while holding the writer lock.
/// Source files and the embedding service are not accessed.
///
/// # Errors
///
/// Returns an error for an occupied writer lock, invalid metadata, missing live vectors, SQLite failures,
/// or snapshot read, rebuild, and publication failures. Until publication commits,
/// the previous snapshot stays active.
pub(crate) fn run(
    database: &DatabaseConfig,
    emit: &mut dyn FnMut(ProgressEvent<'_>),
) -> Result<Option<CompactReport>> {
    run_with_policy(database, false, emit)
}

/// Compact the published index in `database` if it has enough holes or surplus vectors.
///
/// `emit` receives progress events. Returns the report after publication or
/// `None` when there is no active snapshot or compaction is not due.
///
/// # Errors
///
/// Returns writer lock, metadata, SQLite, snapshot, and publication errors.
pub(crate) fn run_if_due(
    database: &DatabaseConfig,
    emit: &mut dyn FnMut(ProgressEvent<'_>),
) -> Result<Option<CompactReport>> {
    run_with_policy(database, true, emit)
}

/// Rebuild the committed graph in `database` under an independent writer lock.
///
/// `only_if_due` enables the automatic threshold. `emit` receives progress.
/// Returns committed counts, or `None` if no rebuild was needed.
///
/// # Errors
///
/// Returns an error if locking, reading, rebuilding, or publishing fails.
fn run_with_policy(
    database: &DatabaseConfig,
    only_if_due: bool,
    emit: &mut dyn FnMut(ProgressEvent<'_>),
) -> Result<Option<CompactReport>> {
    let mut connection = database::open(&database.sqlite_path())?;
    let transaction = database::begin_write(&mut connection)?;
    let Some(active) = database::vector_index(&transaction)? else {
        if !only_if_due {
            snapshot::cleanup_orphans(&database.path, None, emit);
        }
        return Ok(None);
    };
    let dimensions = active.settings.dimensions;
    let previous = snapshot::stored_path(&database.path, &active)?;
    if only_if_due {
        let header = snapshot::read_header(&previous, dimensions)?;
        let referenced = u64::try_from(database::live_vector_count(&transaction)?)?;
        if !should_compact(header.count_present, header.count_deleted, referenced) {
            return Ok(None);
        }
    }
    emit(ProgressEvent::Compacting);
    let source = snapshot::open(&previous, dimensions, OpenMode::View)?;
    snapshot::cleanup_orphans(&database.path, Some(&previous), emit);

    let keys = database::live_vector_keys(&transaction)?;
    let rebuilt = rebuild(&source, &keys, dimensions)?;
    let report = CompactReport {
        previous_vectors: source.size(),
        retained_vectors: rebuilt.size(),
    };
    // Release the memory map before publication can remove the old snapshot.
    drop(source);
    snapshot::commit(
        transaction,
        Some((&rebuilt, &active.settings)),
        database,
        Some(&previous),
        emit,
    )?;
    Ok(Some(report))
}

/// Copy live embeddings into a new graph without retaining obsolete vectors.
///
/// - `source` is the validated, single-vector-per-key source index.
/// - `keys` lists current SQLite chunk keys, each exactly once.
/// - `dimensions` is the validated source vector length.
///
/// Returns a fresh index with the original keys and vector values. An empty key
/// list returns an empty index, clearing all obsolete vectors.
///
/// # Errors
///
/// Returns an error for missing embeddings or USearch failures.
fn rebuild(source: &Index, keys: &[u64], dimensions: NonZeroUsize) -> Result<Index> {
    let rebuilt = snapshot::new_index(dimensions)?;
    if !keys.is_empty() {
        rebuilt.reserve(keys.len())?;
    }
    let mut vector = vec![0.0_f32; dimensions.get()];
    for &key in keys {
        if source.get(key, &mut vector)? != 1 {
            return Err(io::Error::other(format!("Missing live vector {key}")).into());
        }
        rebuilt.add(key, &vector)?;
    }
    Ok(rebuilt)
}

/// Decide whether reclaiming holes or dropping surplus vectors justifies rebuilding.
///
/// `present` and `deleted` are the live vectors and still-empty slots in the
/// USearch header. `referenced` counts distinct keys used by SQLite chunks.
/// Returns true for an empty graph with holes or an index with at least 100
/// holes that occupy at least 25% of its slots. Also returns true for any live
/// vectors when SQLite references none, or for a surplus of at least 10 vectors
/// and at least 10% of the SQLite count. A count comparison detects net surplus.
fn should_compact(present: u64, deleted: u64, referenced: u64) -> bool {
    // USearch reuses deleted slots. Its current header reflects the remaining holes.
    let many_holes = (present == 0 && deleted > 0)
        || (deleted >= 100 && u128::from(deleted) * 3 >= u128::from(present));
    // SQLite is the source of truth when the graph still considers extra vectors live.
    let surplus = present.saturating_sub(referenced);
    let many_surplus = (referenced == 0 && surplus > 0)
        || (surplus >= 10 && u128::from(surplus) * 10 >= u128::from(referenced));
    many_holes || many_surplus
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the hole and surplus thresholds, empty-index cases, and large counts.
    #[test]
    fn automatic_compaction_checks_holes_and_live_vector_surplus() {
        assert!(!should_compact(100, 99, 100));
        assert!(!should_compact(301, 100, 301));
        assert!(should_compact(300, 100, 300));
        assert!(!should_compact(0, 0, 0));
        assert!(should_compact(0, 1, 0));
        assert!(should_compact(300, u64::MAX, 300));

        assert!(!should_compact(99, 0, 90));
        assert!(!should_compact(111, 0, 101));
        assert!(should_compact(110, 0, 100));
        assert!(should_compact(1, 0, 0));
        assert!(!should_compact(1, 0, 2));
        assert!(should_compact(u64::MAX, 0, 1));
    }

    /// Verify that rebuilding preserves live embeddings and excludes obsolete keys.
    ///
    /// Returns success on matching vectors and keys; propagates USearch errors.
    #[test]
    fn rebuild_keeps_live_vectors_and_drops_obsolete_keys() -> Result<()> {
        let dimensions = NonZeroUsize::new(3).unwrap();
        let source = snapshot::new_index(dimensions)?;
        source.reserve(3)?;
        source.add(1, &[1.0_f32, 0.0, 0.0])?;
        source.add(2, &[0.0_f32, 1.0, 0.0])?;
        source.add(3, &[0.0_f32, 0.0, 1.0])?;

        let rebuilt = rebuild(&source, &[1, 3], dimensions)?;
        assert_eq!(rebuilt.size(), 2);
        let mut vector = vec![0.0_f32; 3];
        assert_eq!(rebuilt.get(1, &mut vector)?, 1);
        assert_eq!(vector, [1.0, 0.0, 0.0]);
        assert_eq!(rebuilt.get(3, &mut vector)?, 1);
        assert_eq!(vector, [0.0, 0.0, 1.0]);
        assert_eq!(rebuilt.get(2, &mut vector)?, 0);
        assert_eq!(source.size(), 3);
        assert_eq!(rebuild(&source, &[], dimensions)?.size(), 0);
        Ok(())
    }

    /// Verify that a missing live embedding aborts rebuilding instead of being skipped.
    ///
    /// Returns success on the expected error; propagates setup errors.
    #[test]
    fn rebuild_rejects_a_missing_live_vector() -> Result<()> {
        let dimensions = NonZeroUsize::new(3).unwrap();
        let source = snapshot::new_index(dimensions)?;
        source.reserve(1)?;
        source.add(1, &[1.0_f32, 0.0, 0.0])?;
        assert!(rebuild(&source, &[1, 2], dimensions).is_err());
        Ok(())
    }

    /// Verify publication, shared chunks, stored settings, and an empty live index.
    ///
    /// Returns success when compaction preserves metadata and replaces snapshots;
    /// verifies that the model digest and locations survive, and
    /// propagates filesystem, SQLite, and USearch errors.
    #[test]
    fn compact_publishes_live_vectors_and_preserves_metadata() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let database = DatabaseConfig {
            path: directory.path().to_path_buf(),
        };
        let connection = database::open(&database.sqlite_path())?;
        let dimensions = NonZeroUsize::new(3).unwrap();
        let source = snapshot::new_index(dimensions)?;
        source.reserve(2)?;
        source.add(1, &[1.0_f32, 0.0, 0.0])?;
        source.add(2, &[0.0_f32, 1.0, 0.0])?;
        let old_path = directory.path().join("trombone-index-before.usearch");
        source.save(snapshot::utf8_path(&old_path)?)?;
        database::publish_vector_index(
            &connection,
            "trombone-index-before.usearch",
            &crate::embeddings::EmbeddingSettings {
                model: "stored-model".into(),
                model_digest: "a".repeat(64),
                chunk_format: Some("{text}".into()),
                dimensions,
            },
        )?;
        connection.execute_batch(
            "INSERT INTO documents (source, content_hash) VALUES ('a.md', 'shared'), ('b.md', 'shared');
             INSERT INTO embeddings (vector_key, input_hash) VALUES (2, zeroblob(32));
             INSERT INTO chunks
                 (vector_key, content_hash, ordinal, byte_start, byte_end, line_start, line_end)
             VALUES (2, 'shared', 0, 0, 1, 1, 1);",
        )?;

        let report = run(&database, &mut |_| {})?.unwrap();
        assert_eq!(report.previous_vectors, 2);
        assert_eq!(report.retained_vectors, 1);
        let active = database::vector_index(&connection)?.unwrap();
        assert_eq!(active.settings.model, "stored-model");
        assert_eq!(active.settings.model_digest, "a".repeat(64));
        assert_eq!(active.settings.chunk_format.as_deref(), Some("{text}"));
        assert_eq!(active.settings.dimensions, dimensions);
        let sources = database::chunk_sources(&connection, 2, usize::MAX)?;
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].paths.len(), 2);
        assert!(sources.iter().all(|source| source.location
            == crate::adapter::SourceLocation {
                byte_start: 0,
                byte_end: 1,
                line_start: 1,
                line_end: 1,
            }));
        let path = snapshot::stored_path(&database.path, &active)?;
        assert_ne!(path, old_path);
        assert!(!old_path.exists());
        let published = snapshot::new_index(dimensions)?;
        published.view(snapshot::utf8_path(&path)?)?;
        let mut vector = vec![0.0_f32; 3];
        assert_eq!(published.get(2, &mut vector)?, 1);
        assert_eq!(vector, [0.0, 1.0, 0.0]);
        assert_eq!(published.size(), 1);
        drop(published);

        database::delete_document(&connection, "a.md")?;
        database::delete_document(&connection, "b.md")?;
        assert_eq!(run(&database, &mut |_| {})?.unwrap().retained_vectors, 0);
        Ok(())
    }
}
