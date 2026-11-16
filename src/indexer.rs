use std::fs;
use std::io;
use std::num::NonZeroUsize;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use ollama_rs::Ollama;
use rusqlite::TransactionBehavior;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::config::Config;
use crate::database::{self, IndexStatus, Result, VectorIndex};
use crate::processor;
use crate::reporter::{Phase, Reporter};
use crate::walker::{InputKind, Walker};

const INDEX_PREFIX: &str = "trombone-index-";
const INDEX_SUFFIX: &str = ".usearch";

pub(crate) fn run(
    root: &Path,
    chunk_size: NonZeroUsize,
    glass: bool,
    config: &Config,
    ollama: &Ollama,
) -> Result<bool> {
    let root = std::path::absolute(root)?;
    let reporter = Reporter::new(&root);
    let dimensions = processor::expected_dimensions(&config.ollama)?;
    let model = &config.ollama.model;
    // The walker is synchronous. One Tokio runtime runs the Ollama requests for this pass.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let mut connection = database::open(&config.database.sqlite_path())?;
    // Lock out other writers before choosing the starting snapshot or allocating keys.
    // The transaction spans the walk and the final publication, including Ollama calls.
    let mut transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let active = database::vector_index(&transaction)?;
    let directory = &config.database.path;
    let previous = active
        .as_ref()
        .map(|active| active_path(directory, active, model, dimensions))
        .transpose()?;
    let index = Index::new(&IndexOptions {
        dimensions: dimensions.get(),
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        multi: false,
        ..IndexOptions::default()
    })?;
    if let Some(path) = &previous {
        // Load a writable copy in memory. The file stays available as the active snapshot
        // until SQLite commits the name of its replacement.
        if let Err(error) = index.load(utf8_path(path)?) {
            reporter.snapshot_load_failed(path, &error);
            return Ok(false);
        }
        if index.dimensions() != dimensions.get()
            || index.metric_kind() != MetricKind::Cos
            || index.scalar_kind() != ScalarKind::F32
            || index.multi()
        {
            return Err(io::Error::other("Active USearch index has incompatible settings").into());
        }
    }
    // Persist the counter with the snapshot so superseded vectors keep distinct keys.
    let mut next_key = active.map_or(1, |active| active.next_key);
    let original_size = index.size();
    let mut failed = false;
    let mut fatal = false;

    reporter.phase(Phase::Scanning);
    let input_kind = Walker {
        continue_on_error: !glass,
        report_error: &|path, error| reporter.file_error(path, error),
    }
    .walk(&root, |path| {
        reporter.file(path);
        // Hash and split the same bytes, even if the file is edited while we index it.
        let source = fs::read_to_string(path).inspect_err(|_| failed = true)?;
        let hash = blake3::hash(source.as_bytes()).to_hex();
        let mut added_keys = Vec::new();
        // The inner result is the file's outcome. An outer error means opening, releasing
        // or rolling back its savepoint failed, or USearch cleanup failed.
        let outcome = (|| -> Result<Result<IndexStatus>> {
            let savepoint = transaction.savepoint()?;
            let result = database::save_file(
                &savepoint,
                path,
                hash.as_str(),
                chunk_size,
                || processor::open_document(path, &source, chunk_size),
                &mut next_key,
                |key, text| {
                    reporter.embedding(path, added_keys.len() + 1);
                    let vector =
                        runtime.block_on(processor::embed(ollama, model, dimensions, text))?;
                    if index.size() == index.capacity() {
                        index.reserve(index.capacity().saturating_mul(2).max(1))?;
                    }
                    index.add(key, &vector)?;
                    added_keys.push(key);
                    Ok(())
                },
            );
            if result.is_ok() {
                // Releasing the savepoint keeps this file's changes in the outer transaction.
                savepoint.commit()?;
            } else {
                // The savepoint rolls back the rows. Undo the matching in-memory inserts
                // so a later successful file can still publish the working index.
                savepoint.finish()?;
                for key in added_keys {
                    if index.remove(key)? != 1 {
                        return Err(
                            io::Error::other(format!("Could not roll back vector {key}")).into(),
                        );
                    }
                }
            }
            Ok(result)
        })();

        match outcome {
            Ok(Ok(status)) => reporter.staged(path, status),
            Ok(Err(error)) => {
                failed = true;
                return Err(io::Error::other(error));
            }
            Err(error) => {
                reporter.indexing_aborted_at(path, error.as_ref());
                fatal = true;
                return Ok(ControlFlow::Break(()));
            }
        }
        Ok(ControlFlow::Continue(()))
    })?;
    if fatal {
        return Ok(false);
    }

    // Start from stored paths to catch entire directories that have disappeared.
    // A file missed by the walker is removed only if a direct check returns NotFound.
    reporter.phase(Phase::CheckingDeletions);
    for source in database::document_sources(&transaction)? {
        let path = Path::new(&source);
        let in_scope = match input_kind {
            InputKind::File => path == root,
            InputKind::Directory => path.starts_with(&root),
        };
        if !in_scope {
            continue;
        }
        match fs::metadata(path) {
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                database::delete_document(&transaction, &source)?;
                reporter.removal(path);
            }
            Err(error) => {
                reporter.deletion_check_error(path, &error);
                if glass {
                    return Err(error.into());
                }
                failed = true;
            }
        }
    }

    // Superseded vectors remain until vacuum. An increase in size is the signal
    // to serialize the working index once for this pass.
    let saved = index.size() > original_size;
    if saved {
        reporter.phase(Phase::SavingSnapshot);
        let file = tempfile::Builder::new()
            .prefix(INDEX_PREFIX)
            .suffix(INDEX_SUFFIX)
            .tempfile_in(directory)?;
        index.save(utf8_path(file.path())?)?;
        // Sync both the file and its directory entry before SQLite can publish its name.
        // A crash before commit can leave an extra file, while the old snapshot stays active.
        file.as_file().sync_all()?;
        let file_name = utf8_path(Path::new(file.path().file_name().unwrap()))?.to_owned();
        file.keep()?;
        fs::File::open(directory)?.sync_all()?;
        database::publish_vector_index(
            &transaction,
            &VectorIndex {
                file_name,
                model: model.clone(),
                dimensions: i64::try_from(dimensions.get())?,
                next_key,
            },
        )?;
    }
    // The new filename and the rows that reference its vector keys become visible together.
    reporter.phase(Phase::Committing);
    transaction.commit()?;
    reporter.committed();
    if saved
        && let Some(path) = previous
        && let Err(error) = fs::remove_file(&path)
    {
        Reporter::snapshot_cleanup_failed(&path, &error);
    }
    Ok(!failed)
}

fn active_path(
    directory: &Path,
    active: &VectorIndex,
    model: &str,
    dimensions: NonZeroUsize,
) -> Result<PathBuf> {
    if active.model != model || active.dimensions != i64::try_from(dimensions.get())? {
        return Err(io::Error::other(
            "Embedding model or dimensions differ from the active index; use its settings or a new database path",
        )
        .into());
    }
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

fn utf8_path(path: &Path) -> io::Result<&str> {
    path.to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Index path must be UTF-8"))
}
