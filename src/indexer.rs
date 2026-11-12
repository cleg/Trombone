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
use crate::walker::Walker;

const INDEX_PREFIX: &str = ".trombone-index-";
const INDEX_SUFFIX: &str = ".usearch";

pub(crate) fn run(
    root: &Path,
    chunk_size: NonZeroUsize,
    glass: bool,
    config: &Config,
    ollama: &Ollama,
) -> Result<bool> {
    let dimensions = processor::expected_dimensions(&config.ollama)?;
    let model = &config.ollama.model;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let mut connection = database::open(&config.database.path)?;
    let mut transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let active = database::vector_index(&transaction)?;
    let directory = config
        .database
        .path
        .parent()
        .unwrap_or_else(|| Path::new("."));
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
        index.load(utf8_path(path)?).map_err(|error| {
            io::Error::other(format!(
                "Cannot load active index {}: {error}",
                path.display()
            ))
        })?;
        if index.dimensions() != dimensions.get()
            || index.metric_kind() != MetricKind::Cos
            || index.scalar_kind() != ScalarKind::F32
            || index.multi()
        {
            return Err(io::Error::other("Active USearch index has incompatible settings").into());
        }
    }
    let mut next_key = active.map_or(1, |active| active.next_key);
    let original_size = index.size();
    let mut completed = Vec::new();
    let mut failed = false;
    let mut fatal = None;

    Walker {
        continue_on_error: !glass,
    }
    .walk(root, |path| {
        let source = fs::read_to_string(path).inspect_err(|_| failed = true)?;
        let hash = blake3::hash(source.as_bytes()).to_hex();
        let mut added_keys = Vec::new();
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
                savepoint.commit()?;
            } else {
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
            Ok(Ok(status)) => completed.push((path.to_path_buf(), status)),
            Ok(Err(error)) => {
                failed = true;
                return Err(io::Error::other(error));
            }
            Err(error) => {
                fatal = Some(io::Error::other(format!("{}: {error}", path.display())));
                return Ok(ControlFlow::Break(()));
            }
        }
        Ok(ControlFlow::Continue(()))
    })?;
    if let Some(error) = fatal {
        return Err(error.into());
    }

    // Old vectors remain until vacuum, so a larger index means there are new vectors.
    let saved = index.size() > original_size;
    if saved {
        let file = tempfile::Builder::new()
            .prefix(INDEX_PREFIX)
            .suffix(INDEX_SUFFIX)
            .tempfile_in(directory)?;
        index.save(utf8_path(file.path())?)?;
        file.as_file().sync_all()?;
        let file_name = utf8_path(Path::new(file.path().file_name().unwrap()))?.to_owned();
        // Keep the complete snapshot before SQLite can publish its name.
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
    transaction.commit()?;
    if saved
        && let Some(path) = previous
        && let Err(error) = fs::remove_file(&path)
    {
        eprintln!(
            "Could not remove previous index {}: {error}",
            path.display()
        );
    }
    for (path, status) in completed {
        let action = match status {
            IndexStatus::Unchanged => "Unchanged",
            IndexStatus::Indexed => "Indexed",
            IndexStatus::Reused => "Reused chunks",
        };
        println!("{action}: {}", path.display());
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
    if !name.starts_with(INDEX_PREFIX)
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
