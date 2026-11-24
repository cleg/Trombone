use std::io;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::{
    Connection, ErrorCode, OpenFlags, OptionalExtension, Transaction, TransactionBehavior, params,
};

use crate::adapter::{Chunk, DocumentMetadata, SourceLocation};

pub(crate) use crate::error::Result;

/// Exact SQLite schema version supported by this build, stored in user_version.
const DATABASE_VERSION: i64 = 6;
/// Exact vector representation and text preparation version supported by this build.
pub(crate) const VECTOR_INDEX_VERSION: i64 = 2;

/// SQLite's pointer to the active USearch file and the settings used to build it.
pub(crate) struct VectorIndex {
    pub(crate) format_version: i64,
    pub(crate) file_name: String,
    pub(crate) model: String,
    pub(crate) model_digest: String,
    pub(crate) chunk_format: Option<String>,
    pub(crate) dimensions: i64,
    pub(crate) removed_since_compact: i64,
}

pub(crate) struct ChunkSource {
    pub(crate) path: PathBuf,
    pub(crate) content_hash: String,
    pub(crate) section_path: Option<String>,
    pub(crate) location: SourceLocation,
}

/// Resolve vector `key` to its source paths and typed locations using `connection`.
///
/// Returns paths in lexical order, including all documents and occurrences sharing the vector.
///
/// # Errors
///
/// Returns an error if querying or decoding source metadata fails.
pub(crate) fn chunk_sources(connection: &Connection, key: u64) -> Result<Vec<ChunkSource>> {
    let key = key.cast_signed();
    let mut statement = connection.prepare_cached(
        "SELECT d.source, c.content_hash, c.section_path,
                c.byte_start, c.byte_end, c.line_start, c.line_end
         FROM chunks AS c JOIN documents AS d ON d.content_hash = c.content_hash
         WHERE c.vector_key = ?1
         ORDER BY d.source, c.ordinal",
    )?;
    Ok(statement
        .query_map([key], |row| {
            Ok(ChunkSource {
                path: row.get::<_, String>(0)?.into(),
                content_hash: row.get(1)?,
                section_path: row.get(2)?,
                location: SourceLocation {
                    byte_start: location_coordinate(row, 3)?,
                    byte_end: location_coordinate(row, 4)?,
                    line_start: location_coordinate(row, 5)?,
                    line_end: location_coordinate(row, 6)?,
                },
            })
        })?
        .collect::<rusqlite::Result<_>>()?)
}

/// Decode a source coordinate from `column` in `row` into an address-sized integer.
///
/// Returns the coordinate, or a SQLite decoding error for negative values, values
/// outside `usize`, or an incompatible column type.
fn location_coordinate(row: &rusqlite::Row<'_>, column: usize) -> rusqlite::Result<usize> {
    let value: i64 = row.get(column)?;
    usize::try_from(value).map_err(|_| rusqlite::Error::IntegralValueOutOfRange(column, value))
}

/// Read active snapshot metadata from `connection`.
///
/// Returns `None` when no snapshot has been published. A published snapshot must
/// have exactly the vector format version supported by this build. Its embedding
/// identity includes the digest captured when the snapshot was built.
///
/// # Errors
///
/// Returns an error for SQLite failures, an incompatible vector format version,
/// or chunks without an active snapshot.
pub(crate) fn vector_index(connection: &Connection) -> Result<Option<VectorIndex>> {
    let index = connection
        .query_row(
            "SELECT file_name, model, dimensions, chunk_format,
                    removed_since_compact, format_version, model_digest
             FROM vector_index WHERE id = 1",
            [],
            |row| {
                Ok(VectorIndex {
                    file_name: row.get(0)?,
                    model: row.get(1)?,
                    model_digest: row.get(6)?,
                    dimensions: row.get(2)?,
                    chunk_format: row.get(3)?,
                    removed_since_compact: row.get(4)?,
                    format_version: row.get(5)?,
                })
            },
        )
        .optional()?;
    if let Some(active) = &index
        && active.format_version != VECTOR_INDEX_VERSION
    {
        return Err(io::Error::other(format!(
            "Incompatible vector index version {}; expected {VECTOR_INDEX_VERSION}; recreate the database and run tb index",
            active.format_version
        ))
        .into());
    }
    if index.is_none()
        && connection.query_row("SELECT EXISTS(SELECT 1 FROM chunks)", [], |row| {
            row.get::<_, bool>(0)
        })?
    {
        return Err(io::Error::other("SQLite has chunks but no active USearch index").into());
    }
    Ok(index)
}

/// Stage snapshot publication and removal history in `connection`.
///
/// `index` supplies the filename, format version, embedding identity and digest,
/// and accumulated removal count. On replacement, format and embedding settings
/// stay fixed while the filename and counters change. Returns success after
/// staging the metadata.
///
/// # Errors
///
/// Returns an error if SQLite cannot insert or update the metadata.
pub(crate) fn publish_vector_index(connection: &Connection, index: &VectorIndex) -> Result<()> {
    connection.execute(
        "INSERT INTO vector_index
             (id, file_name, model, dimensions, chunk_format,
              removed_since_compact, format_version, model_digest)
         VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6, ?7)
         ON CONFLICT(id) DO UPDATE SET file_name = excluded.file_name,
             removed_since_compact = excluded.removed_since_compact",
        params![
            index.file_name,
            index.model,
            index.dimensions,
            index.chunk_format,
            index.removed_since_compact,
            index.format_version,
            index.model_digest,
        ],
    )?;
    Ok(())
}

pub(crate) fn document_sources(connection: &Connection) -> Result<Vec<String>> {
    Ok(connection
        .prepare("SELECT source FROM documents")?
        .query_map([], |row| row.get(0))?
        .collect::<rusqlite::Result<_>>()?)
}

pub(crate) fn delete_document(connection: &Connection, source: &str) -> Result<()> {
    // The delete trigger drops chunk rows once the last document sharing them is gone.
    connection.execute("DELETE FROM documents WHERE source = ?1", [source])?;
    Ok(())
}

/// Track deleted chunk keys for the current indexing connection.
///
/// `connection` owns the indexing transaction. The temporary trigger records
/// deletions, including those made by document cleanup, in a temporary table.
/// Savepoint and transaction rollbacks also undo the recorded keys.
/// Returns success once tracking is installed; call once per connection.
///
/// # Errors
///
/// Returns an error if SQLite cannot create the temporary table or trigger.
pub(crate) fn track_removed_vectors(connection: &Connection) -> Result<()> {
    connection.execute_batch(
        "CREATE TEMP TABLE removed_vector_keys (vector_key INTEGER PRIMARY KEY);
         CREATE TEMP TRIGGER track_removed_vectors
         AFTER DELETE ON main.chunks
         BEGIN
             INSERT OR IGNORE INTO removed_vector_keys (vector_key) VALUES (OLD.vector_key);
         END;",
    )?;
    Ok(())
}

/// Create the staging table on the indexing `connection`.
///
/// Returns success once new vectors can be staged under the file savepoints,
/// or an error if SQLite cannot create the temporary table.
pub(crate) fn create_vector_staging(connection: &Connection) -> Result<()> {
    connection.execute_batch(
        "CREATE TEMP TABLE staged_vectors (
             id INTEGER PRIMARY KEY,
             vector_key INTEGER NOT NULL UNIQUE,
             data BLOB NOT NULL
         );",
    )?;
    Ok(())
}

/// Stage the `vector` for `key` inside the file savepoint on `connection`.
///
/// Returns success after writing the vector as little-endian floats, or an error
/// if SQLite cannot insert it. The caller validates the vector dimensions.
pub(crate) fn stage_vector(connection: &Connection, key: u64, vector: &[f32]) -> Result<()> {
    let data: Vec<u8> = vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    connection
        .prepare_cached("INSERT INTO temp.staged_vectors (vector_key, data) VALUES (?1, ?2)")?
        .execute(params![key.cast_signed(), data])?;
    Ok(())
}

/// Remove `key` from staging on `connection` before deleting its embedding metadata.
///
/// Returns true if the vector existed only in staging, false if it must be
/// removed from the published graph, or an error if the SQLite delete fails.
pub(crate) fn discard_staged_vector(connection: &Connection, key: u64) -> Result<bool> {
    Ok(connection
        .prepare_cached("DELETE FROM temp.staged_vectors WHERE vector_key = ?1")?
        .execute([key.cast_signed()])?
        == 1)
}

/// Count the vectors staged on `connection` for the current indexing pass.
///
/// Returns the count or an error if SQLite cannot read it or it exceeds `usize`.
pub(crate) fn staged_vector_count(connection: &Connection) -> Result<usize> {
    let count: i64 =
        connection.query_row("SELECT COUNT(*) FROM temp.staged_vectors", [], |row| {
            row.get(0)
        })?;
    Ok(usize::try_from(count)?)
}

/// Decode staged vectors on `connection` and pass each key and vector to `visit`.
///
/// `dimensions` is the expected number of floats per vector. Visits vectors in
/// insertion order, borrowing one reusable buffer. Returns success after all
/// visits, or an error for invalid data, failed SQLite reads, or a failed visit.
pub(crate) fn visit_staged_vectors(
    connection: &Connection,
    dimensions: NonZeroUsize,
    mut visit: impl FnMut(u64, &[f32]) -> Result<()>,
) -> Result<()> {
    let mut statement =
        connection.prepare("SELECT vector_key, data FROM temp.staged_vectors ORDER BY id")?;
    let mut rows = statement.query([])?;
    let mut vector = Vec::new();
    while let Some(row) = rows.next()? {
        let key = row.get::<_, i64>(0)?.cast_unsigned();
        let data = row.get_ref(1)?.as_blob()?;
        let (values, remainder) = data.as_chunks::<4>();
        if !remainder.is_empty() || values.len() != dimensions.get() {
            return Err(
                io::Error::other(format!("Invalid staged vector data for key {key}")).into(),
            );
        }
        vector.clear();
        vector.extend(values.iter().copied().map(f32::from_le_bytes));
        visit(key, &vector)?;
    }
    Ok(())
}

/// Return tracked vector keys that no longer have a chunk in `connection`.
///
/// `connection` must have tracking installed by `track_removed_vectors` and see
/// the indexing pass's staged changes. Returns keys in SQLite's signed order,
/// excluding keys still referenced by any chunk.
///
/// # Errors
///
/// Returns an error if SQLite cannot query or decode the recorded keys.
pub(crate) fn removed_vector_keys(connection: &Connection) -> Result<Vec<u64>> {
    Ok(connection
        .prepare(
            "SELECT r.vector_key FROM temp.removed_vector_keys AS r
             WHERE NOT EXISTS (SELECT 1 FROM chunks AS c WHERE c.vector_key = r.vector_key)
             ORDER BY r.vector_key",
        )?
        .query_map([], |row| Ok(row.get::<_, i64>(0)?.cast_unsigned()))?
        .collect::<rusqlite::Result<_>>()?)
}

/// Return the vector keys referenced by current chunks in `connection`.
///
/// `connection` must see the transaction whose snapshot is being rebuilt.
/// Returns each key once in SQLite's signed order, regardless of how many chunks reference it.
///
/// # Errors
///
/// Returns an error if SQLite cannot query or decode the keys.
pub(crate) fn live_vector_keys(connection: &Connection) -> Result<Vec<u64>> {
    Ok(connection
        .prepare("SELECT DISTINCT vector_key FROM chunks ORDER BY vector_key")?
        .query_map([], |row| Ok(row.get::<_, i64>(0)?.cast_unsigned()))?
        .collect::<rusqlite::Result<_>>()?)
}

/// Count distinct live chunk vectors in `connection`'s current transaction.
///
/// Returns the number of distinct vector keys referenced by current chunks.
///
/// # Errors
///
/// Returns an error if SQLite cannot count chunks or the count exceeds `usize`.
pub(crate) fn live_vector_count(connection: &Connection) -> Result<usize> {
    let count: i64 =
        connection.query_row("SELECT COUNT(DISTINCT vector_key) FROM chunks", [], |row| {
            row.get(0)
        })?;
    Ok(usize::try_from(count)?)
}

/// Open or create the database file and initialize its schema.
///
/// `path` identifies the database. Returns a writable connection or an error
/// if opening, acquiring the writer lock, or initializing the schema fails.
pub(crate) fn open(path: &Path) -> Result<Connection> {
    let mut connection = Connection::open(path)?;
    connection.pragma_update(None, "foreign_keys", true)?;
    initialize(&mut connection)?;
    Ok(connection)
}

/// Acquire the SQLite writer lock on `connection` without waiting for another writer.
///
/// Returns a transaction with the usual five-second busy timeout restored so
/// publication can wait for brief readers. On acquisition failure the caller
/// should abandon the connection. Returns an actionable error for lock contention,
/// or propagates other SQLite failures.
pub(crate) fn begin_write(connection: &mut Connection) -> Result<Transaction<'_>> {
    connection.busy_timeout(Duration::ZERO)?;
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|error| match error.sqlite_error_code() {
            Some(ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked) => io::Error::new(
                io::ErrorKind::WouldBlock,
                format!(
                    "Database is busy. If another tb index or tb compact is running on this database, let it finish and retry. SQLite: {error}"
                ),
            )
            .into(),
            _ => Box::new(error) as Box<dyn std::error::Error + Send + Sync>,
        })?;
    transaction.busy_timeout(Duration::from_secs(5))?;
    Ok(transaction)
}

/// Open an existing database at `path` without permitting writes.
///
/// Returns a connection only when its schema version exactly matches this build.
///
/// # Errors
///
/// Returns an error if opening the file or checking its database version fails.
pub(crate) fn open_read_only(path: &Path) -> Result<Connection> {
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    check_database_version(&connection)?;
    Ok(connection)
}

/// Initialize a new database or accept the exact supported schema version.
///
/// `connection` is the writable database connection. Returns success after
/// creating a fresh schema atomically or accepting an existing current schema.
/// Existing databases are never migrated, including empty older schemas.
///
/// # Errors
///
/// Returns an error for an occupied writer lock, incompatible versions, or
/// SQLite initialization failures.
fn initialize(connection: &mut Connection) -> Result<()> {
    let transaction = begin_write(connection)?;
    let version = database_version(&transaction)?;
    let has_tables: bool = transaction.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE type = 'table')",
        [],
        |row| row.get(0),
    )?;
    if version == 0 && !has_tables {
        transaction.execute_batch(include_str!("schema.sql"))?;
        transaction.pragma_update(None, "user_version", DATABASE_VERSION)?;
    } else {
        check_database_version(&transaction)?;
    }
    transaction.commit()?;
    Ok(())
}

/// Require `connection` to have exactly this build's database schema version.
///
/// Returns success for the expected version without changing the database.
///
/// # Errors
///
/// Returns an error if reading the version fails or the version is incompatible.
fn check_database_version(connection: &Connection) -> Result<()> {
    let version = database_version(connection)?;
    if version != DATABASE_VERSION {
        return Err(io::Error::other(format!(
            "Incompatible database version {version}; expected {DATABASE_VERSION}; recreate the database and run tb index"
        ))
        .into());
    }
    Ok(())
}

/// Read the application's schema version from `connection`'s user_version field.
///
/// Returns the stored version, or a SQLite error if the pragma cannot be read.
fn database_version(connection: &Connection) -> rusqlite::Result<i64> {
    connection.pragma_query_value(None, "user_version", |row| row.get(0))
}

/// Replace `source` in `transaction` with `content_hash` and document `metadata`.
///
/// Returns success after delete-and-insert, letting triggers remove unreferenced
/// chunks. Returns serialization or SQLite errors; the caller owns rollback.
pub(crate) fn replace_document(
    transaction: &Connection,
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

/// Check `transaction` for another source with `content_hash`, excluding `source`.
///
/// Returns whether such a document exists, including zero-chunk documents, or a
/// SQLite error. This query must precede replacement of the current source.
pub(crate) fn has_other_document(
    transaction: &Connection,
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

/// Look up a vector by its full `hash` in `connection`.
///
/// Returns the stored USearch key if the input is known, or a SQLite error.
pub(crate) fn embedding_key(connection: &Connection, hash: &blake3::Hash) -> Result<Option<u64>> {
    let mut statement =
        connection.prepare_cached("SELECT vector_key FROM embeddings WHERE input_hash = ?1")?;
    Ok(statement
        .query_row([hash.as_bytes().as_slice()], |row| {
            Ok(row.get::<_, i64>(0)?.cast_unsigned())
        })
        .optional()?)
}

/// Try to associate `key` with the full input `hash` in `connection`.
///
/// Returns false if the numeric key is already taken, true on insertion,
/// or an error for other SQLite failures. The caller must use a savepoint
/// and ensure the key's vector is added before publishing.
pub(crate) fn try_insert_embedding(
    connection: &Connection,
    key: u64,
    hash: &blake3::Hash,
) -> Result<bool> {
    let mut statement = connection.prepare_cached(
        "INSERT INTO embeddings (vector_key, input_hash) VALUES (?1, ?2)
         ON CONFLICT(vector_key) DO NOTHING",
    )?;
    Ok(statement.execute(params![key.cast_signed(), hash.as_bytes().as_slice()])? == 1)
}

/// Delete the unreferenced embedding for `key` within `connection`.
///
/// Returns success after deleting exactly one row, or an error if the row
/// is missing, still referenced, or SQLite cannot delete it.
pub(crate) fn delete_embedding(connection: &Connection, key: u64) -> Result<()> {
    let mut statement =
        connection.prepare_cached("DELETE FROM embeddings WHERE vector_key = ?1")?;
    if statement.execute([key.cast_signed()])? != 1 {
        return Err(
            io::Error::other(format!("Missing embedding metadata for vector {key}")).into(),
        );
    }
    Ok(())
}

/// Insert metadata for one `chunk` into the caller's `connection` transaction.
///
/// `content_hash` identifies the source bytes, `ordinal` is its zero-based chunk
/// position, and `key` references an existing embedding. Returns success after storing
/// metadata using a cached statement; does not generate or store vectors.
///
/// # Errors
///
/// Returns integer-conversion, serialization, or SQLite errors. The caller owns rollback.
pub(crate) fn insert_chunk(
    connection: &Connection,
    content_hash: &str,
    ordinal: usize,
    key: u64,
    chunk: &Chunk,
) -> Result<()> {
    let mut insert = connection.prepare_cached(
        "INSERT INTO chunks (vector_key, content_hash, ordinal, section_path, attributes,
                             byte_start, byte_end, line_start, line_end)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
    )?;
    let section_path = chunk
        .section_path
        .as_ref()
        .map(serde_json::to_string)
        .transpose()?;
    insert.execute(params![
        key.cast_signed(),
        content_hash,
        i64::try_from(ordinal)?,
        section_path,
        serde_json::to_string(&chunk.attributes)?,
        i64::try_from(chunk.location.byte_start)?,
        i64::try_from(chunk.location.byte_end)?,
        i64::try_from(chunk.location.line_start)?,
        i64::try_from(chunk.location.line_end)?,
    ])?;
    Ok(())
}

/// Validate and stage `chunk_size` once in the pass's `transaction` before traversal.
///
/// Returns success when the size matches a nonempty index or is adopted by an
/// empty index. Staged settings are rolled back if the pass is not published.
///
/// # Errors
///
/// Returns an error for sizes outside SQLite's integer range, incompatible
/// settings, or SQLite read/write failures.
pub(crate) fn ensure_chunk_size(transaction: &Connection, chunk_size: NonZeroUsize) -> Result<()> {
    let chunk_size = i64::try_from(chunk_size.get())?;
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

/// Read the indexed hash for `source` using `connection`.
///
/// Returns None for an unknown source, its hash otherwise, or a SQLite error.
pub(crate) fn document_hash(connection: &Connection, source: &str) -> Result<Option<String>> {
    Ok(connection
        .query_row(
            "SELECT content_hash FROM documents WHERE source = ?1",
            [source],
            |row| row.get(0),
        )
        .optional()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that incompatible index and database versions are rejected unchanged.
    ///
    /// Returns success when readers and writers reject unsupported versions
    /// without migrating digest-bearing metadata; propagates setup and SQLite errors.
    #[test]
    fn incompatible_versions_are_rejected_without_migration() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("trombone.sqlite");
        let connection = open(&path)?;
        publish_vector_index(
            &connection,
            &VectorIndex {
                format_version: VECTOR_INDEX_VERSION,
                file_name: "trombone-index-version.usearch".into(),
                model: "model".into(),
                model_digest: "a".repeat(64),
                chunk_format: None,
                dimensions: 3,
                removed_since_compact: 100,
            },
        )?;
        connection.execute(
            "UPDATE vector_index SET format_version = ?1 WHERE id = 1",
            [VECTOR_INDEX_VERSION + 1],
        )?;
        assert!(vector_index(&connection).is_err());

        connection.pragma_update(None, "user_version", DATABASE_VERSION - 1)?;
        drop(connection);
        assert!(open_read_only(&path).is_err());
        assert!(open(&path).is_err());

        let unchanged = Connection::open(&path)?;
        assert_eq!(database_version(&unchanged)?, DATABASE_VERSION - 1);
        let metadata: (i64, i64) = unchanged.query_row(
            "SELECT format_version, removed_since_compact FROM vector_index WHERE id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        assert_eq!(metadata, (VECTOR_INDEX_VERSION + 1, 100));
        Ok(())
    }
}
