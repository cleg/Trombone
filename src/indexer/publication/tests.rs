use std::fs;
use std::num::NonZeroUsize;

use crate::database;
use crate::embeddings::Embedder;
use crate::error::Result;
use crate::index_report::PublicationStatus;
use crate::indexer::run;
use crate::indexer::tests::{corpus, test_embedder};
use crate::progress::ProgressEvent;
use crate::snapshot;

/// Verify that either trigger can compact an unchanged pass after an earlier deferral.
///
/// Returns success if a pass with no vector changes preserves the snapshot
/// while compaction is deferred and publishes a rebuilt graph when eligible. A
/// failed rebuild publication leaves already committed document changes visible.
/// Propagates fixture, storage, or rebuild errors.
#[test]
fn automatic_compaction_rebuilds_for_holes_or_surplus() -> Result<()> {
    check_automatic_compaction(100, 0)?;
    check_automatic_compaction(0, 30)
}

/// Build a graph with `holes` deleted slots and `surplus` unreferenced live vectors.
///
/// Returns success when a pass defers compaction, an injected failure leaves
/// indexing committed, and a subsequent pass removes the excess graph slots.
/// Propagates fixture, storage, or rebuild errors.
fn check_automatic_compaction(holes: u64, surplus: u64) -> Result<()> {
    let (directory, database) = corpus()?;
    let roots = [directory.path().to_path_buf()];
    let mut connection = database::open(&database.sqlite_path())?;
    let embedder = test_embedder(None, None);
    let settings = embedder.settings();
    let note = directory.path().join("live.md");
    fs::write(&note, "x")?;
    let content_hash = blake3::hash(b"x").to_hex();
    let chunk_size = NonZeroUsize::new(1000).unwrap();
    database::ensure_chunk_size(&connection, chunk_size)?;
    connection.execute(
        "INSERT INTO documents (source, content_hash) VALUES (?1, ?2)",
        [note.to_str().unwrap(), content_hash.as_str()],
    )?;
    let dimensions = NonZeroUsize::new(3).unwrap();
    let index = snapshot::new_index(dimensions)?;
    index.reserve(usize::try_from(300 + holes + surplus)?)?;
    for key in 1_u64..=300 {
        database::try_insert_embedding(&connection, key, &blake3::hash(&key.to_le_bytes()))?;
        connection.execute(
            "INSERT INTO chunks
                 (vector_key, content_hash, ordinal, byte_start, byte_end, line_start, line_end)
             VALUES (?1, ?2, ?1, 0, 1, 1, 1)",
            rusqlite::params![key.cast_signed(), content_hash.as_str()],
        )?;
        index.add(key, &[1.0_f32, 0.0, 0.0])?;
    }
    for key in 301..=300 + holes + surplus {
        index.add(key, &[0.0_f32, 1.0, 0.0])?;
    }
    for key in 301..=300 + holes {
        assert_eq!(index.remove(key)?, 1);
    }
    snapshot::commit(
        database::begin_write(&mut connection)?,
        Some((&index, settings)),
        &database,
        None,
        &mut |_| {},
    )?;
    let active = database::vector_index(&connection)?.unwrap();
    let previous = snapshot::stored_path(&database.path, &active)?;

    let deferred = run(
        &roots,
        chunk_size,
        crate::config::DEFAULT_MAX_FILE_BYTES,
        true,
        &database,
        &embedder,
        &mut |_| {},
    );
    assert!(deferred.is_success(), "{:?}", deferred.errors);
    assert_eq!(
        deferred.publication,
        PublicationStatus::Published {
            snapshot_updated: false,
            compacted: false,
        }
    );
    let header = snapshot::read_header(&previous, dimensions)?;
    assert_eq!(header.count_deleted, holes);
    assert_eq!(header.count_present, 300 + surplus);

    connection.execute_batch(
        "CREATE TRIGGER fail_compaction BEFORE UPDATE OF file_name ON vector_index
         BEGIN SELECT RAISE(ABORT, 'injected compaction failure'); END;",
    )?;
    let copy = directory.path().join("copy.md");
    fs::write(&copy, "x")?;
    let mut warning = None;
    let report = run(
        &roots,
        chunk_size,
        crate::config::DEFAULT_MAX_FILE_BYTES,
        false,
        &database,
        &embedder,
        &mut |event| {
            if let ProgressEvent::AutomaticCompactionFailed(error) = event {
                warning = Some(error.to_string());
            }
        },
    );
    assert!(report.is_success(), "{:?}", report.errors);
    assert_eq!(report.committed.reused, 1);
    assert_eq!(
        report.publication,
        PublicationStatus::Published {
            snapshot_updated: false,
            compacted: false
        }
    );
    assert!(warning.is_some_and(|warning| warning.contains("injected compaction failure")));
    assert_eq!(
        database::document_hash(&connection, copy.to_str().unwrap())?.as_deref(),
        Some(content_hash.as_str())
    );
    let after = database::vector_index(&connection)?.unwrap();
    assert_eq!(after.file_name, active.file_name);
    connection.execute_batch("DROP TRIGGER fail_compaction")?;

    let rebuilt = run(
        &roots,
        chunk_size,
        crate::config::DEFAULT_MAX_FILE_BYTES,
        false,
        &database,
        &embedder,
        &mut |_| {},
    );
    assert!(rebuilt.is_success(), "{:?}", rebuilt.errors);
    assert_eq!(
        rebuilt.publication,
        PublicationStatus::Published {
            snapshot_updated: true,
            compacted: true,
        }
    );
    let active = database::vector_index(&connection)?.unwrap();
    let path = snapshot::stored_path(&database.path, &active)?;
    assert_ne!(path, previous);
    let header = snapshot::read_header(&path, dimensions)?;
    assert_eq!(header.count_present, 300);
    assert_eq!(header.count_deleted, 0);
    Ok(())
}
