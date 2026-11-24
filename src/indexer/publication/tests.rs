use std::num::NonZeroUsize;

use super::{commit_indexing, publish_index};
use crate::database::{self, Result};
use crate::embeddings::Embedder;
use crate::index_report::{IndexProgress, PublicationStatus};
use crate::indexer::tests::{corpus, test_embedder};
use crate::progress::ProgressEvent;
use crate::snapshot::{self, utf8_path};

/// Verify that deferral preserves history and a later eligible pass rebuilds once.
///
/// Returns success if a pass with no vector changes preserves the snapshot
/// while compaction is deferred and publishes a rebuilt graph when eligible.
/// Propagates fixture, storage, or rebuild errors.
#[test]
fn automatic_compaction_respects_deferral_and_resets_history() -> Result<()> {
    let (_directory, database) = corpus()?;
    let mut connection = database::open(&database.sqlite_path())?;
    let embedder = test_embedder(None, None);
    let settings = embedder.settings();
    connection.execute(
        "INSERT INTO documents (source, content_hash) VALUES ('live.md', 'live')",
        [],
    )?;
    let dimensions = NonZeroUsize::new(3).unwrap();
    let index = snapshot::new_index(dimensions)?;
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
    commit_indexing(
        database::begin_write(&mut connection)?,
        Some(&index),
        &database,
        settings,
        100,
        None,
        &progress,
    )?;
    let active = database::vector_index(&connection)?.unwrap();
    let previous = snapshot::stored_path(&database.path, &active)?;

    for (removed, no_compact, expected) in
        [(99, false, false), (100, true, false), (100, false, true)]
    {
        let mut connection = database::open(&database.sqlite_path())?;
        connection.execute(
            "UPDATE vector_index SET removed_since_compact = ?1 WHERE id = 1",
            [removed],
        )?;
        let transaction = database::begin_write(&mut connection)?;
        database::track_removed_vectors(&transaction)?;
        database::create_vector_staging(&transaction)?;
        assert_eq!(
            publish_index(
                transaction,
                &database,
                settings,
                Some(&previous),
                removed,
                no_compact,
                &progress,
            )?,
            PublicationStatus::Published {
                snapshot_updated: expected,
                compacted: expected,
            }
        );
        let active = database::vector_index(&connection)?.unwrap();
        if expected {
            assert_eq!(active.removed_since_compact, 0);
            let rebuilt = snapshot::new_index(dimensions)?;
            let path = snapshot::stored_path(&database.path, &active)?;
            rebuilt.view(utf8_path(&path)?)?;
            assert_eq!(rebuilt.size(), 300);
            let mut vector = vec![0.0_f32; 3];
            assert_eq!(rebuilt.get(999, &mut vector)?, 0);
        } else {
            assert_eq!(active.removed_since_compact, removed);
            assert_eq!(snapshot::stored_path(&database.path, &active)?, previous);
        }
    }
    Ok(())
}
