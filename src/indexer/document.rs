//! Stages a Markdown document and its new embeddings in a SQLite savepoint.

use std::fs;
use std::io;
use std::num::NonZeroUsize;
use std::ops::ControlFlow;
use std::path::Path;

use rusqlite::{Connection, Transaction};

use crate::adapter::Document;
use crate::database::{self, Result};
use crate::embeddings::{Embedder, EmbeddingSettings, format_chunk};
use crate::index_report::{IndexProgress, IndexStatus};
use crate::processor;
use crate::progress::IndexEvent;

/// File-level outcome after its savepoint and vector changes have been resolved.
#[derive(Debug)]
enum StageOutcome {
    /// The file is prepared in the outer transaction and awaits pass publication.
    Staged(IndexStatus),
    /// File rows and new vectors were restored, so continuation is safe.
    RolledBack(Box<dyn std::error::Error + Send + Sync>),
}

/// Read and stage one source `path` using `chunk_size` and `embedder`.
///
/// `transaction` owns the file savepoint and `progress` records the outcome.
/// Returns Continue for a staged file, or Break after reporting an
/// unrecoverable savepoint failure.
///
/// # Errors
///
/// Returns file read errors and StageOutcome::RolledBack failures for Walker to
/// report before continuing with the remaining files.
pub(super) fn process_file(
    path: &Path,
    transaction: &mut Transaction<'_>,
    chunk_size: NonZeroUsize,
    embedder: &dyn Embedder,
    progress: &IndexProgress<'_>,
) -> io::Result<ControlFlow<()>> {
    progress.file(path);
    // Hash and split the same bytes even if the source changes during embedding.
    let source = fs::read_to_string(path)?;
    let outcome = stage_file(
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
    );
    match outcome {
        Ok(StageOutcome::Staged(status)) => {
            progress.staged(path, status);
            Ok(ControlFlow::Continue(()))
        }
        Ok(StageOutcome::RolledBack(error)) => Err(io::Error::other(error)),
        Err(error) => {
            progress.issue(Some(path), error.as_ref());
            Ok(ControlFlow::Break(()))
        }
    }
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
///
/// # Errors
///
/// Returns an error if creating, releasing, or rolling back the savepoint fails.
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
            let vector = embed(input, ordinal)?;
            if vector.len() != settings.dimensions.get() {
                return Err(io::Error::other(format!(
                    "Expected {} vector dimensions, got {}",
                    settings.dimensions,
                    vector.len()
                ))
                .into());
            }
            database::stage_vector(&savepoint, key, &vector)
        },
    );
    match result {
        Ok(status) => {
            // Releasing the savepoint keeps this file's changes in the outer transaction.
            savepoint.commit()?;
            Ok(StageOutcome::Staged(status))
        }
        Err(error) => {
            savepoint.finish()?;
            Ok(StageOutcome::RolledBack(error))
        }
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
/// Returns path, adapter, serialization, SQLite, key-allocation, or vector errors.
/// The caller must roll back the file's savepoint on failure.
fn stage_document<'a>(
    connection: &Connection,
    path: &Path,
    content_hash: &str,
    load_document: impl FnOnce() -> io::Result<Document<'a>>,
    chunk_format: Option<&str>,
    mut add_vector: impl FnMut(u64, String) -> Result<()>,
) -> Result<IndexStatus> {
    let source = path
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Source path must be UTF-8"))?;
    // Skip the adapter entirely when this path still has the same content.
    if database::document_hash(connection, source)?.as_deref() == Some(content_hash) {
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
    let reuse = database::has_other_document(connection, source, content_hash)?;
    database::replace_document(connection, source, content_hash, &document.metadata)?;
    if !reuse {
        for (ordinal, chunk) in document.chunks.enumerate() {
            let input = format_chunk(chunk_format, chunk.section_path.as_deref(), &chunk.text)?;
            let hash = blake3::hash(input.as_bytes());
            let key = match database::embedding_key(connection, &hash)? {
                Some(key) => key,
                None => {
                    let key = allocate_vector_key(connection, &hash)?;
                    add_vector(key, input)?;
                    key
                }
            };
            database::insert_chunk(connection, content_hash, ordinal, key, &chunk)?;
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
