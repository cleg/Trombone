//! Rebuilds the vector snapshot from keys still referenced by SQLite chunks.

use std::io;
use std::num::NonZeroUsize;

use usearch::Index;

use crate::config::DatabaseConfig;
use crate::database::{self, Result};
use crate::progress::ProgressEvent;
use crate::snapshot;

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
/// Resets accumulated removal history on successful publication. Source files
/// and the embedding service are not accessed.
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
    let mut connection = database::open(&database.sqlite_path())?;
    let transaction = database::begin_write(&mut connection)?;
    let Some(mut active) = database::vector_index(&transaction)? else {
        return Ok(None);
    };
    let dimensions = NonZeroUsize::new(usize::try_from(active.dimensions)?)
        .ok_or_else(|| io::Error::other("Active index has zero dimensions"))?;
    let previous = snapshot::stored_path(&database.path, &active)?;
    let source = snapshot::new_index(dimensions)?;
    source.view(snapshot::utf8_path(&previous)?)?;
    snapshot::validate(&source, dimensions)?;

    let keys = database::live_vector_keys(&transaction)?;
    let rebuilt = rebuild(&source, &keys, dimensions)?;
    let report = CompactReport {
        previous_vectors: source.size(),
        retained_vectors: rebuilt.size(),
    };
    // Release the memory map before publication can remove the old snapshot.
    drop(source);
    active.removed_since_compact = 0;
    snapshot::commit(
        transaction,
        Some((&rebuilt, active)),
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
pub(crate) fn rebuild(source: &Index, keys: &[u64], dimensions: NonZeroUsize) -> Result<Index> {
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

/// Decide whether accumulated graph deletions justify an automatic rebuild.
///
/// `live_vectors` counts current SQLite chunks; `removed_since_compact` counts
/// vector removals since the last published rebuild. Returns true at 100 removals
/// and a removal fraction of at least 25% of live-plus-removed vectors. An empty
/// live index with any removals is rebuilt immediately. Integer arithmetic avoids
/// rounding and overflow even for large counts.
pub(crate) fn should_compact(live_vectors: usize, removed_since_compact: i64) -> bool {
    removed_since_compact > 0
        && (live_vectors == 0
            || (removed_since_compact >= 100
                && (removed_since_compact as u128) * 3 >= live_vectors as u128))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify both threshold boundaries, immediate empty-index cleanup, and large counts.
    #[test]
    fn automatic_compaction_requires_the_minimum_and_removal_fraction() {
        assert!(!should_compact(100, 99));
        assert!(!should_compact(301, 100));
        assert!(should_compact(300, 100));
        assert!(!should_compact(1_000, 333));
        assert!(should_compact(1_000, 334));
        assert!(!should_compact(0, 0));
        assert!(should_compact(0, 1));
        assert!(should_compact(300, i64::MAX));
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
    /// verifies that removal history resets while model digest and locations survive, and
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
            &database::VectorIndex {
                format_version: database::VECTOR_INDEX_VERSION,
                file_name: "trombone-index-before.usearch".into(),
                model: "stored-model".into(),
                model_digest: "a".repeat(64),
                chunk_format: Some("{text}".into()),
                dimensions: 3,
                removed_since_compact: 321,
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
        assert_eq!(active.model, "stored-model");
        assert_eq!(active.model_digest, "a".repeat(64));
        assert_eq!(active.chunk_format.as_deref(), Some("{text}"));
        assert_eq!(active.dimensions, 3);
        assert_eq!(active.removed_since_compact, 0);
        let sources = database::chunk_sources(&connection, 2)?;
        assert_eq!(sources.len(), 2);
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
