//! Stages a Markdown document and its new embeddings in a SQLite savepoint.

use std::io;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;

use rusqlite::{Connection, Transaction};

use crate::adapter::Document;
use crate::database;
use crate::embeddings::{Embedder, EmbeddingSettings, format_chunk};
use crate::error::{Error, Result};
use crate::index_report::{IndexProgress, IndexStatus};
use crate::processor;
use crate::progress::{IndexEvent, ProgressEvent};

#[derive(Debug)]
enum StageError {
    Processing(Error),
    Storage(Error),
}

/// File-level outcome after its savepoint and vector changes have been resolved.
#[derive(Debug)]
enum StageOutcome {
    /// The file is prepared in the outer transaction and awaits pass publication.
    Staged(IndexStatus),
    /// File rows and new vectors were restored, so continuation is safe.
    RolledBack(Error),
}

/// Read and stage the discovered source `path` using `chunk_size`, `max_file_bytes`, and `embedder`.
///
/// `transaction` owns the file savepoint. `progress` records success, emits a
/// nonfatal warning when the file exceeds `max_file_bytes` and skips it, or
/// records a recoverable read or processing error. Returns success after any of
/// these outcomes.
///
/// # Errors
///
/// Returns fatal storage or savepoint failures to abort the pass.
pub(super) fn process_file(
    path: &Path,
    transaction: &mut Transaction<'_>,
    chunk_size: NonZeroUsize,
    max_file_bytes: NonZeroU64,
    embedder: &dyn Embedder,
    progress: &mut IndexProgress<'_>,
) -> Result<()> {
    progress.file(path);
    // Hash and split the same bytes even if the source changes during embedding.
    let source = match processor::read_source_file(path, max_file_bytes) {
        Ok(source) => source,
        Err(error) if error.kind() == io::ErrorKind::FileTooLarge => {
            progress.emit(ProgressEvent::FileTooLarge {
                path,
                max_bytes: max_file_bytes,
            });
            return Ok(());
        }
        Err(error) => {
            progress.issue(Some(path), &error);
            return Ok(());
        }
    };
    match stage_file(
        transaction,
        path,
        &source,
        chunk_size,
        embedder.settings(),
        |input, ordinal| {
            progress.notify(IndexEvent::Embedding {
                path,
                chunk: ordinal,
            });
            embedder.embed_chunk(input)
        },
    )? {
        StageOutcome::Staged(status) => progress.staged(path, status),
        StageOutcome::RolledBack(error) => progress.issue(Some(path), error.as_ref()),
    }
    Ok(())
}

/// Stage one file and its new vectors in the same SQLite savepoint.
///
/// # Parameters
///
/// - `transaction` is the indexing pass's SQLite transaction.
/// - `path` identifies the source document.
/// - `source` contains the text used for both hashing and chunking.
/// - `chunk_size` is the maximum chunk length in Unicode characters.
/// - `settings` defines the input format and the expected vector dimensions.
/// - `embed` produces a vector for a new model input and receives its one-based
///   generation number for this file.
///
/// Returns Staged after releasing the file savepoint, or RolledBack with the
/// original file error after restoring its rows, including staged vectors.
/// Rejects vectors whose length differs from `settings.dimensions` or that
/// contain non-finite floats as recoverable processing errors.
///
/// # Errors
///
/// Returns an error if writing to storage or manipulating the savepoint fails.
/// These failures require aborting the entire indexing pass.
fn stage_file(
    transaction: &mut Transaction<'_>,
    path: &Path,
    source: &str,
    chunk_size: NonZeroUsize,
    settings: &EmbeddingSettings,
    mut embed: impl FnMut(String, usize) -> Result<Vec<f32>>,
) -> Result<StageOutcome> {
    let hash = blake3::hash(source.as_bytes()).to_hex();
    let mut ordinal = 0;
    let savepoint = transaction.savepoint()?;
    let result = stage_document(
        &savepoint,
        path,
        hash.as_str(),
        || processor::open_document(path, source, chunk_size),
        settings.chunk_format.as_deref(),
        |key, input| {
            ordinal += 1;
            let vector = embed(input, ordinal).map_err(StageError::Processing)?;
            if vector.len() != settings.dimensions.get() {
                return Err(StageError::Processing(
                    io::Error::other(format!(
                        "Expected {} vector dimensions, got {}",
                        settings.dimensions,
                        vector.len()
                    ))
                    .into(),
                ));
            }
            if vector.iter().any(|value| !value.is_finite()) {
                return Err(StageError::Processing(
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Embedding vector contains a non-finite value",
                    )
                    .into(),
                ));
            }
            database::stage_vector(&savepoint, key, &vector).map_err(StageError::Storage)
        },
    );
    match result {
        Ok(status) => {
            // Releasing the savepoint keeps this file's changes in the outer transaction.
            savepoint.commit()?;
            Ok(StageOutcome::Staged(status))
        }
        Err(StageError::Processing(error)) => {
            savepoint.finish()?;
            Ok(StageOutcome::RolledBack(error))
        }
        Err(StageError::Storage(error)) => Err(error),
    }
}

/// Prepare one file inside the caller's transaction or savepoint.
///
/// `connection` holds a pass whose chunk size was already validated. `path` and
/// `content_hash` identify the source; `load_document` lazily supplies its chunks.
/// `chunk_format` prepares the exact model input and `add_vector` generates and
/// stores a vector only for a previously unseen input.
/// Returns whether the source was unchanged, indexed, or reused shared chunks.
///
/// # Errors
///
/// Returns recoverable source, adapter, formatting, and embedding errors, or
/// fatal storage errors. The caller owns rollback.
fn stage_document<'a>(
    connection: &Connection,
    path: &Path,
    content_hash: &str,
    load_document: impl FnOnce() -> io::Result<Document<'a>>,
    chunk_format: Option<&str>,
    mut add_vector: impl FnMut(u64, String) -> std::result::Result<(), StageError>,
) -> std::result::Result<IndexStatus, StageError> {
    let source = path.to_str().ok_or_else(|| {
        StageError::Processing(io::Error::other("Source path must be UTF-8").into())
    })?;
    // Skip the adapter entirely when this path still has the same content.
    if database::document_hash(connection, source)
        .map_err(StageError::Storage)?
        .as_deref()
        == Some(content_hash)
    {
        return Ok(IndexStatus::Unchanged);
    }

    // The callback is FnOnce: extraction is deferred until needed and runs once.
    let document = load_document().map_err(|error| StageError::Processing(error.into()))?;
    if document.metadata.source != path {
        return Err(StageError::Processing(
            io::Error::other("Adapter returned a different source path").into(),
        ));
    }
    // Decide reuse using other documents, before replacing this document.
    let reuse = database::has_other_document(connection, source, content_hash)
        .map_err(StageError::Storage)?;
    database::replace_document(connection, source, content_hash, &document.metadata)
        .map_err(StageError::Storage)?;
    if !reuse {
        for (ordinal, chunk) in document.chunks.enumerate() {
            let input = format_chunk(chunk_format, chunk.section_path.as_deref(), &chunk.text)
                .map_err(|error| StageError::Processing(error.into()))?;
            let hash = blake3::hash(input.as_bytes());
            let key =
                match database::embedding_key(connection, &hash).map_err(StageError::Storage)? {
                    Some(key) => key,
                    None => {
                        let key =
                            allocate_vector_key(connection, &hash).map_err(StageError::Storage)?;
                        add_vector(key, input)?;
                        key
                    }
                };
            database::insert_chunk(connection, content_hash, ordinal, key, &chunk)
                .map_err(StageError::Storage)?;
        }
    }
    Ok(if reuse {
        IndexStatus::Reused
    } else {
        IndexStatus::Indexed
    })
}

/// Register a new full `hash` in `connection` and return its USearch key.
///
/// Starts with the first eight hash bytes in little-endian order and probes
/// consecutive keys when occupied. Skips USearch's reserved deleted-vector key.
/// Call only after a full-hash miss. Returns an error if SQLite fails or every
/// usable key is occupied. The caller owns the savepoint and vector insertion.
fn allocate_vector_key(connection: &Connection, hash: &blake3::Hash) -> Result<u64> {
    let mut prefix = [0; 8];
    prefix.copy_from_slice(&hash.as_bytes()[..8]);
    let start = u64::from_le_bytes(prefix);
    let mut key = start;
    loop {
        if key != u64::MAX && database::try_insert_embedding(connection, key, hash)? {
            return Ok(key);
        }
        key = key.wrapping_add(1);
        if key == start {
            return Err(io::Error::other("USearch key space exhausted").into());
        }
    }
}

#[cfg(test)]
mod tests;
