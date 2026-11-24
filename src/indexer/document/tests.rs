use std::io;
use std::num::NonZeroUsize;
use std::path::Path;

use rusqlite::Connection;

use super::{StageOutcome, allocate_vector_key, stage_document, stage_file};
use crate::adapter::{Attributes, Chunk, Document, DocumentMetadata, SourceLocation};
use crate::database::{self, *};
use crate::embeddings::Embedder;
use crate::index_report::IndexStatus;
use crate::indexer::tests::test_embedder;
use crate::snapshot;

/// Validate `chunk_size` and save `path` with `hash` and lazy `load_document`.
///
/// `connection` supplies storage. Returns the committed file status, or a
/// configuration/storage/adapter error after rolling back the transaction.
fn save<'a>(
    connection: &mut Connection,
    path: &Path,
    hash: &str,
    chunk_size: NonZeroUsize,
    load_document: impl FnOnce() -> io::Result<Document<'a>>,
) -> Result<IndexStatus> {
    let transaction = connection.transaction()?;
    ensure_chunk_size(&transaction, chunk_size)?;
    let status = stage_document(&transaction, path, hash, load_document, None, |_, _| Ok(()))?;
    transaction.commit()?;
    Ok(status)
}

/// Build an owned test document for `path` with one chunk spanning all of `text`.
///
/// Returns typed byte and line coordinates plus otherwise empty chunk attributes.
fn document(path: &str, text: &str) -> Document<'static> {
    Document {
        metadata: DocumentMetadata {
            source: path.into(),
            attributes: Attributes::new(),
        },
        chunks: Box::new(std::iter::once(Chunk {
            text: text.into(),
            section_path: Some(vec!["Title".into()]),
            location: SourceLocation {
                byte_start: 0,
                byte_end: text.len(),
                line_start: 1,
                line_end: text.lines().count().max(1),
            },
            attributes: Attributes::new(),
        })),
    }
}

/// Verify source locations survive storage alongside independent custom attributes.
///
/// Returns success for exact Unicode byte spans and inclusive line numbers;
/// propagates database and metadata decoding errors.
#[test]
fn stores_typed_locations_separately_from_extra_attributes() -> Result<()> {
    let mut connection = open(Path::new(":memory:"))?;
    let prefix = "# Назва 🙂\n\n";
    let text = "Перший рядок.\nДругий 🙂.\n";
    let location = SourceLocation {
        byte_start: prefix.len(),
        byte_end: prefix.len() + text.len(),
        line_start: 3,
        line_end: 4,
    };
    save(
        &mut connection,
        Path::new("note.md"),
        "hash",
        NonZeroUsize::new(1_000).unwrap(),
        || {
            let mut document = document("note.md", text);
            document.chunks = Box::new(document.chunks.map(move |mut chunk| {
                chunk.location = location;
                chunk.attributes.insert("custom".into(), "preserved".into());
                chunk
            }));
            Ok(document)
        },
    )?;
    let key = live_vector_keys(&connection)?[0];
    let sources = chunk_sources(&connection, key)?;
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].path, Path::new("note.md"));
    assert_eq!(sources[0].location, location);
    let stored_attributes: String = connection.query_row(
        "SELECT attributes FROM chunks WHERE vector_key = ?1",
        [key.cast_signed()],
        |row| row.get(0),
    )?;
    let attributes: Attributes = serde_json::from_str(&stored_attributes)?;
    assert_eq!(
        attributes,
        Attributes::from_iter([("custom".into(), "preserved".into())])
    );
    Ok(())
}

/// Index fixture `path` and `hash` in `connection`; returns its status or an error.
fn index(connection: &mut Connection, path: &str, hash: &str) -> Result<IndexStatus> {
    save(
        connection,
        Path::new(path),
        hash,
        NonZeroUsize::new(2000).unwrap(),
        || Ok(document(path, hash)),
    )
}

/// Return chunk hashes from `connection` in sorted order, or a SQLite error.
fn hashes(connection: &Connection) -> Result<Vec<String>> {
    Ok(connection
        .prepare("SELECT content_hash FROM chunks ORDER BY content_hash")?
        .query_map([], |row| row.get(0))?
        .collect::<rusqlite::Result<_>>()?)
}

/// Verify duplicate sources retain shared chunks until their last reference is deleted.
///
/// Returns success for shared-content lifetimes, or a fixture/storage error.
#[test]
fn duplicate_files_share_chunks_until_the_last_reference_is_deleted() -> Result<()> {
    let mut connection = open(Path::new(":memory:"))?;
    assert_eq!(
        index(&mut connection, "a.md", "same")?,
        IndexStatus::Indexed
    );
    assert_eq!(index(&mut connection, "b.md", "same")?, IndexStatus::Reused);
    assert_eq!(hashes(&connection)?, ["same"]);
    connection.execute("DELETE FROM documents WHERE source = 'a.md'", [])?;
    assert_eq!(hashes(&connection)?, ["same"]);
    connection.execute("DELETE FROM documents WHERE source = 'b.md'", [])?;
    assert!(hashes(&connection)?.is_empty());
    Ok(())
}

/// Verify reindexing reuses other sources' content and removes unreferenced old chunks.
///
/// Returns success for the expected statuses and stored content, or a storage error.
#[test]
fn changed_file_reuses_existing_content_and_cleans_up_old_chunks() -> Result<()> {
    let mut connection = open(Path::new(":memory:"))?;
    index(&mut connection, "a.md", "old")?;
    index(&mut connection, "b.md", "shared")?;
    assert_eq!(
        index(&mut connection, "a.md", "shared")?,
        IndexStatus::Reused
    );
    assert_eq!(hashes(&connection)?, ["shared"]);
    assert_eq!(index(&mut connection, "a.md", "new")?, IndexStatus::Indexed);
    assert_eq!(hashes(&connection)?, ["new", "shared"]);
    let hash: String = connection.query_row(
        "SELECT content_hash FROM documents WHERE source = 'a.md'",
        [],
        |row| row.get(0),
    )?;
    assert_eq!(hash, "new");
    Ok(())
}

/// Verify unchanged sources skip adapters and incompatible sizes fail before opening them.
///
/// Returns success for both decisions and preserved content, or a storage error.
#[test]
fn unchanged_files_skip_the_adapter_and_reject_a_different_chunk_size() -> Result<()> {
    let mut connection = open(Path::new(":memory:"))?;
    index(&mut connection, "a.md", "same")?;
    let status = save(
        &mut connection,
        Path::new("a.md"),
        "same",
        NonZeroUsize::new(2000).unwrap(),
        || panic!("unchanged files must not be parsed"),
    )?;
    assert_eq!(status, IndexStatus::Unchanged);
    let result = save(
        &mut connection,
        Path::new("a.md"),
        "same",
        NonZeroUsize::new(1000).unwrap(),
        || panic!("incompatible settings must be rejected first"),
    );
    assert!(result.is_err());
    assert_eq!(hashes(&connection)?, ["same"]);
    Ok(())
}

/// Verify a failed chunk insertion restores the previous document and all its chunks.
///
/// Returns success for the injected error and rollback, or a fixture/storage error.
#[test]
fn failed_reindex_restores_the_old_document_and_its_chunks() -> Result<()> {
    let mut connection = open(Path::new(":memory:"))?;
    index(&mut connection, "a.md", "old")?;
    // Fail on the second insert, after replacement and the first insert succeeded.
    connection.execute_batch(
        "CREATE TEMP TRIGGER fail_second_chunk
             BEFORE INSERT ON chunks WHEN NEW.ordinal = 1
             BEGIN SELECT RAISE(ABORT, 'test chunk write failure'); END;",
    )?;
    let result = save(
        &mut connection,
        Path::new("a.md"),
        "new",
        NonZeroUsize::new(2000).unwrap(),
        || {
            let mut document = document("a.md", "new");
            let second = self::document("a.md", "second");
            document.chunks = Box::new(document.chunks.chain(second.chunks));
            Ok(document)
        },
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("test chunk write failure")
    );
    assert_eq!(hashes(&connection)?, ["old"]);
    let hash: String = connection.query_row(
        "SELECT content_hash FROM documents WHERE source = 'a.md'",
        [],
        |row| row.get(0),
    )?;
    assert_eq!(hash, "old");
    Ok(())
}

/// Verify zero-chunk documents can be reused without consuming a new chunk iterator.
///
/// Returns success for shared empty content and absent vectors, or a storage error.
#[test]
fn empty_content_reuses_document_identity_without_reading_chunks() -> Result<()> {
    let mut connection = open(Path::new(":memory:"))?;
    let size = NonZeroUsize::new(2000).unwrap();
    assert_eq!(
        save(
            &mut connection,
            Path::new("empty.md"),
            "empty",
            size,
            || {
                let mut document = document("empty.md", "");
                document.chunks = Box::new(std::iter::empty());
                Ok(document)
            }
        )?,
        IndexStatus::Indexed,
    );
    assert_eq!(
        save(&mut connection, Path::new("copy.md"), "empty", size, || {
            let mut document = document("copy.md", "");
            document.chunks = Box::new(std::iter::once_with(|| {
                panic!("reused content must not consume chunks")
            }));
            Ok(document)
        })?,
        IndexStatus::Reused,
    );
    assert_eq!(document_sources(&connection)?.len(), 2);
    assert_eq!(live_vector_count(&connection)?, 0);
    Ok(())
}

/// Verify an adapter returning the wrong source cannot replace indexed content.
///
/// Returns success for rejection and preserved rows, or a fixture/storage error.
#[test]
fn mismatched_adapter_source_preserves_the_existing_document() -> Result<()> {
    let mut connection = open(Path::new(":memory:"))?;
    index(&mut connection, "note.md", "old")?;
    let result = save(
        &mut connection,
        Path::new("note.md"),
        "new",
        NonZeroUsize::new(2000).unwrap(),
        || Ok(document("different.md", "new")),
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("different source path")
    );
    assert_eq!(
        document_hash(&connection, "note.md")?.as_deref(),
        Some("old")
    );
    assert_eq!(hashes(&connection)?, ["old"]);
    Ok(())
}

/// Verify that vector cleanup follows shared chunks and savepoint rollbacks.
///
/// Returns success when only committed chunk deletions are tracked, including
/// chunks created and deleted after tracking began. Propagates setup and
/// database errors.
#[test]
fn removed_vectors_follow_chunk_lifetimes_and_rollbacks() -> Result<()> {
    let mut connection = open(Path::new(":memory:"))?;
    track_removed_vectors(&connection)?;
    index(&mut connection, "a.md", "shared")?;
    index(&mut connection, "b.md", "shared")?;
    let shared_key = live_vector_keys(&connection)?[0];

    // Removing one duplicate must preserve its shared vector.
    delete_document(&connection, "a.md")?;
    assert!(removed_vector_keys(&connection)?.is_empty());

    // A rolled-back deletion must not leave a vector scheduled for cleanup.
    let savepoint = connection.savepoint()?;
    delete_document(&savepoint, "b.md")?;
    assert_eq!(removed_vector_keys(&savepoint)?, [shared_key]);
    savepoint.finish()?;
    assert!(removed_vector_keys(&connection)?.is_empty());
    assert_eq!(hashes(&connection)?, ["shared"]);

    // Replacing the final reference schedules the old vector for removal.
    index(&mut connection, "b.md", "new")?;
    assert_eq!(removed_vector_keys(&connection)?, [shared_key]);
    assert_eq!(hashes(&connection)?, ["new"]);

    let new_key = live_vector_keys(&connection)?[0];
    // Tracking also covers a vector created and deleted during this pass.
    delete_document(&connection, "b.md")?;
    let mut removed = removed_vector_keys(&connection)?;
    removed.sort_unstable();
    let mut expected = [shared_key, new_key];
    expected.sort_unstable();
    assert_eq!(removed, expected);
    assert!(hashes(&connection)?.is_empty());
    Ok(())
}

/// Verify colliding prefixes retain distinct full hashes and stable keys in SQLite.
/// Returns success for both lookups, USearch key round trips, and sentinel
/// handling, or a database or vector index error.
#[test]
fn conflicting_vector_keys_probe_without_aliasing_full_hashes() -> Result<()> {
    let connection = database::open(Path::new(":memory:"))?;
    let mut bytes = [0; 32];
    bytes[..8].copy_from_slice(&(u64::MAX - 1).to_le_bytes());
    let first = blake3::Hash::from_bytes(bytes);
    bytes[8] = 1;
    let second = blake3::Hash::from_bytes(bytes);
    let first_key = allocate_vector_key(&connection, &first)?;
    let second_key = allocate_vector_key(&connection, &second)?;
    assert_eq!(first_key, u64::MAX - 1);
    assert_eq!(second_key, 0);
    let index = snapshot::new_index(NonZeroUsize::new(3).unwrap())?;
    index.reserve(2)?;
    index.add(first_key, &[1.0_f32, 0.0, 0.0])?;
    index.add(second_key, &[0.0_f32, 1.0, 0.0])?;
    let mut vector = [0.0_f32; 3];
    assert_eq!(index.get(first_key, &mut vector)?, 1);
    assert_eq!(vector, [1.0, 0.0, 0.0]);
    assert_eq!(index.get(second_key, &mut vector)?, 1);
    assert_eq!(vector, [0.0, 1.0, 0.0]);
    assert_eq!(
        database::embedding_key(&connection, &first)?,
        Some(first_key)
    );
    assert_eq!(
        database::embedding_key(&connection, &second)?,
        Some(second_key)
    );
    database::delete_embedding(&connection, first_key)?;
    assert_eq!(
        database::embedding_key(&connection, &second)?,
        Some(second_key)
    );
    Ok(())
}

/// Verify a file savepoint restores chunks and pending vectors together.
///
/// Returns success if the old vector remains staged and the failed file's
/// vector and embedding metadata disappear. Propagates fixture or SQL errors.
#[test]
fn failed_file_rollback_restores_staged_vectors_and_chunks() -> Result<()> {
    let mut connection = database::open(Path::new(":memory:"))?;
    let mut transaction = connection.transaction()?;
    database::track_removed_vectors(&transaction)?;
    database::create_vector_staging(&transaction)?;
    let embedder = test_embedder(None, None);
    let settings = embedder.settings();
    let chunk_size = NonZeroUsize::new(5).unwrap();
    assert!(matches!(
        stage_file(
            &mut transaction,
            Path::new("note.md"),
            "old",
            chunk_size,
            settings,
            |_, _| Ok(vec![1.0_f32, 0.0, 0.0]),
        )?,
        StageOutcome::Staged(IndexStatus::Indexed),
    ));
    let result = stage_file(
        &mut transaction,
        Path::new("note.md"),
        "first\n\nsecond",
        chunk_size,
        settings,
        |_, ordinal| {
            if ordinal == 2 {
                Err(io::Error::other("embedding failed").into())
            } else {
                Ok(vec![0.0_f32, 1.0, 0.0])
            }
        },
    )?;
    assert!(matches!(
        result,
        StageOutcome::RolledBack(error) if error.to_string().contains("embedding failed")
    ));
    assert_eq!(database::staged_vector_count(&transaction)?, 1);
    let old_key = database::embedding_key(&transaction, &blake3::hash(b"old"))?.unwrap();
    database::visit_staged_vectors(&transaction, settings.dimensions, |key, vector| {
        assert_eq!(key, old_key);
        assert_eq!(vector, &[1.0_f32, 0.0, 0.0]);
        Ok(())
    })?;
    assert_eq!(database::chunk_sources(&transaction, old_key)?.len(), 1);
    assert!(database::embedding_key(&transaction, &blake3::hash(b"first"))?.is_none());
    assert!(database::removed_vector_keys(&transaction)?.is_empty());
    Ok(())
}
