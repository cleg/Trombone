use std::error::Error;
use std::io;
use std::num::NonZeroUsize;
use std::path::Path;

use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};

use crate::adapter::{Chunk, Document, DocumentMetadata};

// One result type for SQLite, JSON serialization and document opening errors.
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum IndexStatus {
    Unchanged,
    Indexed,
    Reused,
}

/// Open or create the database file and initialize its schema.
pub(crate) fn open(path: &Path) -> Result<Connection> {
    let mut connection = Connection::open(path)?;
    initialize(&mut connection)?;
    Ok(connection)
}

fn initialize(connection: &mut Connection) -> Result<()> {
    // Commit all schema changes together, or roll them all back on failure.
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    // The previous development schema has no hashes. Only upgrade it when empty.
    let legacy: bool = transaction.query_row(
        "SELECT EXISTS(SELECT 1 FROM pragma_table_info('chunks') WHERE name = 'document_id')",
        [],
        |row| row.get(0),
    )?;
    if legacy {
        let count: i64 = transaction.query_row(
            "SELECT (SELECT count(*) FROM documents) + (SELECT count(*) FROM chunks)",
            [],
            |row| row.get(0),
        )?;
        if count != 0 {
            return Err(io::Error::other(
                "Legacy database contains data; select a new database path to rebuild the index",
            )
            .into());
        }
        transaction.execute_batch("DROP TABLE chunks; DROP TABLE documents;")?;
    }
    // Embed the SQL at compile time; no separate SQL file is needed at runtime.
    transaction.execute_batch(include_str!("schema.sql"))?;
    transaction.commit()?;
    Ok(())
}

/// Save a new or changed file, sharing chunks with identical content.
pub(crate) fn save<'a>(
    connection: &mut Connection,
    path: &Path,
    content_hash: &str,
    chunk_size: NonZeroUsize,
    load_document: impl FnOnce() -> io::Result<Document<'a>>,
) -> Result<IndexStatus> {
    // SQLite INTEGER is signed 64-bit; reject sizes that cannot be represented.
    let chunk_size = i64::try_from(chunk_size.get())?;
    let source = path
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Source path must be UTF-8"))?;
    // Acquire the writer lock before checking existing data, so another writer
    // cannot change our deduplication decisions before we save the document.
    // Any error returned through `?` drops this transaction and rolls it back.
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    // A content hash identifies shared chunks only under the same chunking settings.
    ensure_chunk_size(&transaction, chunk_size)?;
    // Skip the adapter entirely when this path still has the same content.
    if !needs_indexing(&transaction, source, content_hash)? {
        transaction.commit()?;
        return Ok(IndexStatus::Unchanged);
    }

    // The callback is FnOnce: extraction is deferred until needed and runs once.
    let document = load_document()?;
    if document.metadata.source != path {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Adapter returned a different source path",
        )
        .into());
    }
    // Decide reuse using other documents, before replacing this document.
    let reuse = can_reuse_chunks(&transaction, source, content_hash)?;
    replace_document(&transaction, source, content_hash, &document.metadata)?;
    if !reuse {
        save_chunks(&transaction, content_hash, document.chunks)?;
    }
    // Document replacement, trigger cleanup and new chunks become visible together.
    transaction.commit()?;
    Ok(if reuse {
        IndexStatus::Reused
    } else {
        IndexStatus::Indexed
    })
}

/// Replace the source's document row without updating its content hash in place.
fn replace_document(
    transaction: &Transaction<'_>,
    source: &str,
    content_hash: &str,
    metadata: &DocumentMetadata,
) -> Result<()> {
    // The DELETE trigger removes old chunks only if their last reference is gone.
    transaction.execute("DELETE FROM documents WHERE source = ?1", [source])?;
    // Insert a fresh row; the enclosing transaction restores the old row on failure.
    transaction.execute(
        "INSERT INTO documents (source, content_hash, attributes) VALUES (?1, ?2, ?3)",
        params![
            source,
            content_hash,
            serde_json::to_string(&metadata.attributes)?
        ],
    )?;
    Ok(())
}

/// Check whether another document already owns a complete chunk set for this hash.
fn can_reuse_chunks(
    transaction: &Transaction<'_>,
    source: &str,
    content_hash: &str,
) -> Result<bool> {
    // Check documents rather than chunks: an empty file legitimately has no chunks.
    // Exclude this source so it can never count as its own reusable reference.
    Ok(transaction.query_row(
        "SELECT EXISTS(SELECT 1 FROM documents WHERE content_hash = ?1 AND source != ?2)",
        params![content_hash, source],
        |row| row.get(0),
    )?)
}

/// Consume the chunk iterator and insert its chunks within the caller's transaction.
fn save_chunks(
    transaction: &Transaction<'_>,
    content_hash: &str,
    chunks: impl Iterator<Item = Chunk>,
) -> Result<()> {
    // Prepare once and reuse the statement for every chunk.
    let mut insert = transaction.prepare(
        "INSERT INTO chunks (content_hash, ordinal, text, section_path, attributes)
         VALUES (?1, ?2, ?3, ?4, ?5)",
    )?;
    for (ordinal, chunk) in chunks.enumerate() {
        // enumerate() gives the zero-based position in the document.
        let ordinal = i64::try_from(ordinal)?;
        // Some(path) becomes JSON text; None becomes SQL NULL.
        // transpose() changes Option<Result<...>> into Result<Option<...>>.
        let section_path = chunk
            .section_path
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        insert.execute(params![
            content_hash,
            ordinal,
            chunk.text,
            section_path,
            serde_json::to_string(&chunk.attributes)?,
        ])?;
    }
    Ok(())
}

/// Keep one chunk size for all shared content in a nonempty index.
fn ensure_chunk_size(transaction: &Transaction<'_>, chunk_size: i64) -> Result<()> {
    // optional() treats a missing settings row as None instead of an error.
    let stored_size: Option<i64> = transaction
        .query_row(
            "SELECT chunk_size FROM index_config WHERE id = 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    let has_documents: bool =
        transaction.query_row("SELECT EXISTS(SELECT 1 FROM documents)", [], |row| {
            row.get(0)
        })?;
    if has_documents && stored_size != Some(chunk_size) {
        return Err(io::Error::other(
            "Chunk size differs from the existing index; use its chunk size or a new database path",
        )
        .into());
    }
    if stored_size != Some(chunk_size) {
        // A new or empty index can adopt the requested chunk size.
        transaction.execute(
            "INSERT INTO index_config (id, chunk_size) VALUES (1, ?1)
         ON CONFLICT(id) DO UPDATE SET chunk_size = excluded.chunk_size",
            [chunk_size],
        )?;
    }
    Ok(())
}

/// Index an unseen path or a path whose stored content hash has changed.
fn needs_indexing(transaction: &Transaction<'_>, source: &str, content_hash: &str) -> Result<bool> {
    let previous_hash: Option<String> = transaction
        .query_row(
            "SELECT content_hash FROM documents WHERE source = ?1",
            [source],
            |row| row.get(0),
        )
        .optional()?;
    Ok(previous_hash.as_deref() != Some(content_hash))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{Attributes, Chunk, DocumentMetadata};

    fn document(path: &str, text: &str) -> Document<'static> {
        Document {
            metadata: DocumentMetadata {
                source: path.into(),
                attributes: Attributes::new(),
            },
            chunks: Box::new(std::iter::once(Chunk {
                text: text.into(),
                section_path: Some(vec!["Title".into()]),
                attributes: Attributes::from_iter([("line_start".into(), 1.into())]),
            })),
        }
    }

    fn index(connection: &mut Connection, path: &str, hash: &str) -> Result<IndexStatus> {
        save(
            connection,
            Path::new(path),
            hash,
            NonZeroUsize::new(2000).unwrap(),
            || Ok(document(path, hash)),
        )
    }

    fn hashes(connection: &Connection) -> Result<Vec<String>> {
        Ok(connection
            .prepare("SELECT content_hash FROM chunks ORDER BY content_hash")?
            .query_map([], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?)
    }

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
}
