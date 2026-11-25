//! Defines the USearch index settings and resolves snapshot files recorded in SQLite.
//!
//! The active snapshot's metadata identifies the embedding model and digest, dimensions,
//! and chunk format used to build it. Indexing and search check that metadata
//! before opening the snapshot and validate the index settings afterward.

use std::cell::RefCell;
use std::fmt;
use std::fs;
use std::io;
use std::mem::DropGuard;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use rusqlite::Transaction;
use usearch::{Index, IndexMetadata, IndexOptions, MetricKind, ScalarKind};

use crate::config::DatabaseConfig;
use crate::database::{self, VectorIndex};
use crate::embeddings::EmbeddingSettings;
use crate::error::{Error, Result};
use crate::progress::ProgressEvent;

/// Prefix shared by newly created USearch snapshot filenames.
pub(crate) const INDEX_PREFIX: &str = "trombone-index-";
/// Extension shared by USearch snapshot filenames.
pub(crate) const INDEX_SUFFIX: &str = ".usearch";
/// Number of random alphanumeric characters in a generated snapshot name.
const SNAPSHOT_ID_LEN: usize = 6;

/// Create an empty cosine-similarity index with `F32` vectors and one vector per key.
///
/// `dimensions` is the number of components in each vector.
///
/// # Errors
///
/// Returns an error if USearch cannot initialize the index.
pub(crate) fn new_index(dimensions: NonZeroUsize) -> Result<Index> {
    Ok(Index::new(&IndexOptions {
        dimensions: dimensions.get(),
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        multi: false,
        ..IndexOptions::default()
    })?)
}

/// How to open an existing snapshot's graph.
pub(crate) enum OpenMode {
    Load,
    View,
}

/// Failure to read or use a particular snapshot.
#[derive(Debug)]
pub(crate) struct OpenError {
    path: PathBuf,
    failure: OpenFailure,
}

#[derive(Debug)]
enum OpenFailure {
    Access(io::Error),
    Engine(Error),
    Incompatible,
}

impl OpenError {
    /// Return whether the snapshot file was missing when the open failed.
    pub(crate) fn is_missing(&self) -> bool {
        matches!(&self.failure, OpenFailure::Access(error) if error.kind() == io::ErrorKind::NotFound)
    }

    /// Borrow the failure details for callers that already display the snapshot path.
    pub(crate) fn reason(&self) -> &dyn fmt::Display {
        match &self.failure {
            OpenFailure::Access(error) => error,
            OpenFailure::Engine(error) => error,
            OpenFailure::Incompatible => &"Active USearch index has incompatible settings",
        }
    }
}

impl fmt::Display for OpenError {
    /// Write the failed snapshot path and reason to `formatter`, propagating formatting errors.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Cannot open active index {}: {}",
            self.path.display(),
            self.reason(),
        )
    }
}

impl std::error::Error for OpenError {
    /// Return the underlying filesystem or USearch error, if any.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.failure {
            OpenFailure::Access(error) => Some(error),
            OpenFailure::Engine(error) => Some(error.as_ref()),
            OpenFailure::Incompatible => None,
        }
    }
}

/// Open the graph at `path` with the expected vector `dimensions`.
///
/// `mode` chooses whether to load a writable copy or memory-map the graph.
/// Returns the opened index after checking its settings.
///
/// # Errors
///
/// Returns a path-specific error if USearch cannot initialize the index, the
/// graph cannot be read, or its settings differ.
pub(crate) fn open(
    path: &Path,
    dimensions: NonZeroUsize,
    mode: OpenMode,
) -> std::result::Result<Index, OpenError> {
    read_snapshot(path, |filename| {
        let index = new_index(dimensions).map_err(OpenFailure::Engine)?;
        match mode {
            OpenMode::Load => index.load(filename),
            OpenMode::View => index.view(filename),
        }
        .map_err(|error| native_failure(path, error.into()))?;
        if index.dimensions() != dimensions.get()
            || !compatible_settings(index.metric_kind(), index.scalar_kind(), index.multi())
        {
            return Err(OpenFailure::Incompatible);
        }
        Ok(index)
    })
}

/// Read the serialized header at `path` and check the expected `dimensions`.
///
/// Returns metadata for a regular snapshot file with compatible settings. This
/// reads only metadata and leaves graph integrity checks to the full open.
///
/// # Errors
///
/// Returns an error for inaccessible files, invalid headers, or incompatible settings.
pub(crate) fn read_header(
    path: &Path,
    dimensions: NonZeroUsize,
) -> std::result::Result<IndexMetadata, OpenError> {
    read_snapshot(path, |filename| {
        check_regular_file(path).map_err(OpenFailure::Access)?;
        let header =
            Index::metadata(filename).map_err(|error| native_failure(path, error.into()))?;
        if usize::try_from(header.dimensions).ok() != Some(dimensions.get())
            || !compatible_settings(header.metric, header.quantization, header.multi)
        {
            return Err(OpenFailure::Incompatible);
        }
        Ok(header)
    })
}

/// Return whether `metric`, `scalar` and `multi` match Trombone's graph settings.
fn compatible_settings(metric: MetricKind, scalar: ScalarKind, multi: bool) -> bool {
    metric == MetricKind::Cos && scalar == ScalarKind::F32 && !multi
}

/// Run `read` with the UTF-8 filename for `path`, attaching the path to any failure.
///
/// Returns the reader's value or an error for an unsupported path or failed read.
fn read_snapshot<T>(
    path: &Path,
    read: impl FnOnce(&str) -> std::result::Result<T, OpenFailure>,
) -> std::result::Result<T, OpenError> {
    utf8_path(path)
        .map_err(OpenFailure::Access)
        .and_then(read)
        .map_err(|failure| OpenError {
            path: path.to_path_buf(),
            failure,
        })
}

/// Classify a USearch `error` for `path` using a filesystem probe after the failure.
///
/// Returns the filesystem error when the file cannot be opened, or the original
/// engine error when the filesystem probe succeeds.
fn native_failure(path: &Path, error: Error) -> OpenFailure {
    match check_regular_file(path).and_then(|()| fs::File::open(path)) {
        Ok(_) => OpenFailure::Engine(error),
        Err(error) => OpenFailure::Access(error),
    }
}

/// Check that `path` resolves to a regular file, returning an I/O error otherwise.
fn check_regular_file(path: &Path) -> io::Result<()> {
    if fs::metadata(path)?.is_file() {
        Ok(())
    } else {
        Err(io::Error::other("Expected a regular snapshot file"))
    }
}

/// Return the path of the active snapshot after checking its metadata.
///
/// - `directory` is the directory containing SQLite and the snapshot files.
/// - `active` is the snapshot metadata read from SQLite.
/// - `settings` supplies the embedder's model, digest, dimensions, and chunk format.
///
/// Accepts older snapshot filenames with a leading dot. The returned path is
/// constructed without accessing the filesystem.
///
/// # Errors
///
/// Returns an error if the model, digest, dimensions, or chunk format differ from the
/// stored values or the stored name fails the snapshot filename checks.
pub(crate) fn active_path(
    directory: &Path,
    active: &VectorIndex,
    settings: &EmbeddingSettings,
) -> Result<PathBuf> {
    let stored = &active.settings;
    if stored.model != settings.model || stored.dimensions != settings.dimensions {
        return Err(io::Error::other(
            "Embedding model or dimensions differ from the active index; use its settings or a new database path",
        )
        .into());
    }
    if stored.model_digest != settings.model_digest {
        return Err(io::Error::other(format!(
            "Embedding model digest differs from the active index (indexed {}, current {}); restore the indexed model or rebuild using a fresh database path",
            stored.model_digest, settings.model_digest,
        )).into());
    }
    if stored.chunk_format != settings.chunk_format {
        return Err(io::Error::other(
            "ollama.chunk_format differs from the active index; use its setting or a new database path",
        )
        .into());
    }
    stored_path(directory, active)
}

/// Resolve a snapshot filename without comparing it to embedding configuration.
///
/// `directory` contains the snapshots and `active` supplies the stored filename.
/// Returns a path without accessing the filesystem. This allows maintenance to
/// use the index's own settings without an embedding service.
///
/// # Errors
///
/// Returns an error if the stored filename fails the snapshot filename checks.
pub(crate) fn stored_path(directory: &Path, active: &VectorIndex) -> Result<PathBuf> {
    let name = &active.file_name;
    // Older snapshots have a leading dot in their filename.
    if !name
        .strip_prefix('.')
        .unwrap_or(name)
        .starts_with(INDEX_PREFIX)
        || !name.ends_with(INDEX_SUFFIX)
        || Path::new(name).file_name().and_then(|name| name.to_str()) != Some(name)
    {
        return Err(io::Error::other("Invalid active USearch index file name").into());
    }
    Ok(directory.join(name))
}

/// Borrow the UTF-8 representation of `path` for the USearch API.
///
/// # Errors
///
/// Returns an error of kind [`io::ErrorKind::InvalidInput`] if `path` cannot
/// be represented as UTF-8.
pub(crate) fn utf8_path(path: &Path) -> io::Result<&str> {
    path.to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Index path must be UTF-8"))
}

/// Remove orphaned Trombone snapshot files from `directory` under the caller's writer lock.
///
/// `active` is the snapshot referenced by committed SQLite metadata, if present.
/// If its directory entry cannot be identified as a regular file, the sweep is
/// skipped so it cannot remove a symlink's target. `emit` receives nonfatal
/// directory and removal errors. The sweep ignores unrelated names and nonfiles.
pub(crate) fn cleanup_orphans(
    directory: &Path,
    active: Option<&Path>,
    emit: &mut dyn FnMut(ProgressEvent<'_>),
) {
    if active
        .is_some_and(|path| !fs::symlink_metadata(path).is_ok_and(|metadata| metadata.is_file()))
    {
        return;
    }
    let active_name = active
        .and_then(Path::file_name)
        .and_then(|name| name.to_str());
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) => {
            emit(ProgressEvent::SnapshotCleanupFailed {
                path: directory,
                error: &error,
            });
            return;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                emit(ProgressEvent::SnapshotCleanupFailed {
                    path: directory,
                    error: &error,
                });
                continue;
            }
        };
        let filename = entry.file_name();
        let Some(name) = filename.to_str() else {
            continue;
        };
        if !is_generated_snapshot_name(name)
            || active_name.is_some_and(|active| active.eq_ignore_ascii_case(name))
        {
            continue;
        }
        let path = entry.path();
        let result = entry.file_type().and_then(|kind| {
            if kind.is_file() {
                fs::remove_file(&path)?;
            }
            Ok(())
        });
        if let Err(error) = result
            && error.kind() != io::ErrorKind::NotFound
        {
            emit(ProgressEvent::SnapshotCleanupFailed {
                path: &path,
                error: &error,
            });
        }
    }
}

/// Return whether `name` matches Trombone's current or legacy generated filenames.
fn is_generated_snapshot_name(name: &str) -> bool {
    name.strip_prefix('.')
        .unwrap_or(name)
        .strip_prefix(INDEX_PREFIX)
        .and_then(|name| name.strip_suffix(INDEX_SUFFIX))
        .is_some_and(|id| {
            id.len() == SNAPSHOT_ID_LEN && id.bytes().all(|byte| byte.is_ascii_alphanumeric())
        })
}

/// Publish a new vector snapshot and commit its metadata atomically with SQLite.
///
/// - `transaction` contains the staged document changes.
/// - `update` supplies the index and embedding settings when vectors changed.
///   The snapshot filename is generated when saving the file.
/// - `database` supplies the database and snapshot locations.
/// - `previous` is the old snapshot to remove after a successful commit.
/// - `emit` receives save and commit phases, and any cleanup warnings.
///
/// Returns success after committing. Old snapshot cleanup errors are reported
/// without failing the committed operation. An already removed old file is fine.
/// On failure, a new snapshot is removed
/// only if committed SQLite metadata confirms it is unreferenced.
///
/// # Errors
///
/// Returns an error if saving, syncing, staging metadata, or committing fails.
pub(crate) fn commit(
    transaction: Transaction<'_>,
    update: Option<(&Index, &EmbeddingSettings)>,
    database: &DatabaseConfig,
    previous: Option<&Path>,
    emit: &mut dyn FnMut(ProgressEvent<'_>),
) -> Result<()> {
    // The pending snapshot's guard and the normal path both emit synchronously.
    let emit = RefCell::new(emit);
    let database_path = database.sqlite_path();
    let pending_snapshot = if let Some((index, settings)) = update {
        (emit.borrow_mut())(ProgressEvent::SavingSnapshot);
        let file = tempfile::Builder::new()
            .prefix(INDEX_PREFIX)
            .suffix(INDEX_SUFFIX)
            .rand_bytes(SNAPSHOT_ID_LEN)
            .tempfile_in(&database.path)?;
        index.save(utf8_path(file.path())?)?;
        file.as_file().sync_all()?;
        let file_name = file
            .path()
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| io::Error::other("Invalid generated snapshot filename"))?
            .to_owned();
        let (_, path) = file.keep()?;
        let guard = DropGuard::new(path, |path| {
            if let Err(error) = remove_unreferenced_snapshot(&database_path, &path) {
                (emit.borrow_mut())(ProgressEvent::SnapshotCleanupFailed {
                    path: &path,
                    error: error.as_ref(),
                });
            }
        });
        // Make the snapshot and its directory entry durable before publishing its name.
        fs::File::open(&database.path)?.sync_all()?;
        database::publish_vector_index(&transaction, &file_name, settings)?;
        Some(guard)
    } else {
        None
    };

    (emit.borrow_mut())(ProgressEvent::Committing);
    transaction.commit()?;
    let saved = pending_snapshot.map(DropGuard::dismiss).is_some();
    if saved
        && let Some(path) = previous
        && let Err(error) = fs::remove_file(path)
        && error.kind() != io::ErrorKind::NotFound
    {
        (emit.borrow_mut())(ProgressEvent::SnapshotCleanupFailed {
            path,
            error: &error,
        });
    }
    Ok(())
}

/// Remove `path` only if committed metadata in `database_path` leaves it unreferenced.
///
/// Returns success if the snapshot is active or was removed.
///
/// # Errors
///
/// Returns an error if reading metadata or removing the unreferenced file fails.
fn remove_unreferenced_snapshot(database_path: &Path, path: &Path) -> Result<()> {
    let connection = database::open_read_only(database_path)?;
    if let Some(active) = database::vector_index(&connection)?
        && path.file_name() == Some(std::ffi::OsStr::new(&active.file_name))
    {
        return Ok(());
    }
    fs::remove_file(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify cleanup warnings arrive through events for both publication outcomes.
    ///
    /// Returns success if old-file cleanup preserves a successful commit and
    /// failed-commit cleanup preserves the original SQLite error. Propagates
    /// unexpected fixture, filesystem, or database failures.
    #[test]
    fn cleanup_warnings_preserve_publication_outcome() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let database = DatabaseConfig {
            path: directory.path().to_path_buf(),
        };
        let dimensions = NonZeroUsize::new(3).unwrap();
        let index = new_index(dimensions)?;
        let mut connection = database::open(&database.sqlite_path())?;

        // Removal through remove_file must fail for a directory, regardless of permissions.
        let previous = database.path.join("trombone-index-old.usearch");
        fs::create_dir(&previous)?;
        let settings = EmbeddingSettings {
            model: "test-model".into(),
            model_digest: "a".repeat(64),
            chunk_format: None,
            dimensions,
        };
        database::publish_vector_index(&connection, "trombone-index-old.usearch", &settings)?;
        let mut phases = Vec::new();
        let mut warnings = Vec::new();
        commit(
            connection.transaction()?,
            Some((&index, &settings)),
            &database,
            Some(&previous),
            &mut |event| match event {
                ProgressEvent::SavingSnapshot => phases.push("saving"),
                ProgressEvent::Committing => phases.push("committing"),
                ProgressEvent::SnapshotCleanupFailed { path, .. } => {
                    phases.push("cleanup warning");
                    warnings.push(path.to_path_buf());
                }
                ProgressEvent::Index { .. }
                | ProgressEvent::FileTooLarge { .. }
                | ProgressEvent::Compacting
                | ProgressEvent::AutomaticCompactionFailed(_) => unreachable!(),
            },
        )?;
        assert_eq!(phases, ["saving", "committing", "cleanup warning"]);
        assert_eq!(warnings.as_slice(), std::slice::from_ref(&previous));
        let active = database::vector_index(&connection)?.unwrap();
        let active_file = stored_path(&database.path, &active)?;
        assert_ne!(active_file, previous);
        assert!(active_file.is_file());

        connection.execute_batch(
            "CREATE TABLE parent (id INTEGER PRIMARY KEY);
             CREATE TABLE child (
                 parent_id INTEGER REFERENCES parent(id) DEFERRABLE INITIALLY DEFERRED
             );",
        )?;
        let transaction = connection.transaction()?;
        transaction.execute("INSERT INTO child (parent_id) VALUES (1)", [])?;
        let mut pending = None;
        let mut cleanup_path = None;
        let failure = commit(
            transaction,
            Some((&index, &active.settings)),
            &database,
            Some(&active_file),
            &mut |event| match event {
                ProgressEvent::Committing => {
                    // Simulate an external filesystem change before the deferred commit failure.
                    let path = fs::read_dir(&database.path)
                        .unwrap()
                        .map(|entry| entry.unwrap().path())
                        .find(|path| {
                            path != &active_file
                                && path.is_file()
                                && path.extension().is_some_and(|ext| ext == "usearch")
                        })
                        .unwrap();
                    fs::remove_file(&path).unwrap();
                    pending = Some(path);
                }
                ProgressEvent::SnapshotCleanupFailed { path, .. } => {
                    cleanup_path = Some(path.to_path_buf());
                }
                _ => {}
            },
        )
        .unwrap_err();
        assert!(
            failure
                .to_string()
                .contains("FOREIGN KEY constraint failed")
        );
        assert!(cleanup_path.is_some());
        assert_eq!(cleanup_path, pending);
        let active = database::vector_index(&connection)?.unwrap();
        assert_eq!(stored_path(&database.path, &active)?, active_file);
        assert!(active_file.is_file());
        Ok(())
    }
}
