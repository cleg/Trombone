mod adapter;
mod adapters;
mod cli;
mod compact;
mod config;
mod database;
mod embeddings;
mod error;
mod index_report;
mod indexer;
mod processor;
mod progress;
mod reporter;
mod search;
mod search_output;
mod snapshot;
mod template;
mod walker;

use clap::Parser;
use std::process::ExitCode;

use cli::{Cli, Commands, OutputFormat};
use config::Config;
use embeddings::{Embedder, OllamaEmbedder};
use index_report::PublicationStatus;
use reporter::Reporter;
use search_output::{ConsoleOutput, JsonOutput, SearchOutput};

/// Dispatch the CLI command using the selected configuration.
///
/// Returns success after a complete operation, or failure after reporting a
/// configuration, indexing, compaction, search, or output error. Indexing forwards
/// the rebuild and compaction flags and derives the exit code from IndexReport.
/// The report can include published work alongside recoverable errors. Indexing
/// validates its root list before initializing embeddings or opening storage.
/// Indexing and search initialize the embedding client before entering their core operations.
/// A published indexing pass also checks for model changes and warns on a mismatch.
/// Progress events are rendered by the console reporter here.
fn main() -> ExitCode {
    let cli = Cli::parse();
    let config_path = match cli.config.map_or_else(config::default_path, Ok) {
        Ok(path) => path,
        Err(error) => {
            Reporter::config_location_failed(&error);
            return ExitCode::FAILURE;
        }
    };
    let config = match Config::load(&config_path) {
        Ok(config) => config,
        Err(error) => {
            Reporter::config_load_failed(&config_path, error.as_ref());
            return ExitCode::FAILURE;
        }
    };
    let mut reporter = Reporter::default();
    match cli.command {
        Commands::Compact => {
            let result = compact::run(&config.database, &mut |event| reporter.handle(event))
                .and_then(|report| {
                    Reporter::compacted(report.as_ref())?;
                    Ok(())
                });
            if let Err(error) = result {
                Reporter::compaction_failed(error.as_ref());
                return ExitCode::FAILURE;
            }
        }
        Commands::Index {
            chunk_size,
            full,
            no_compact,
        } => {
            let roots = match config.index_roots() {
                Ok(roots) => roots,
                Err(error) => {
                    Reporter::config_load_failed(&config_path, &error);
                    return ExitCode::FAILURE;
                }
            };
            let embedder = match initialize_embedder(&config) {
                Ok(embedder) => embedder,
                Err(status) => return status,
            };
            let report = indexer::run(
                roots,
                chunk_size.unwrap_or(config.chunking.size),
                config.max_file_bytes,
                indexer::IndexOptions { full, no_compact },
                &config.database,
                &embedder,
                &mut |event| reporter.handle(event),
            );
            reporter.finish();
            if matches!(report.publication, PublicationStatus::Published { .. })
                && let Ok(Some(current)) = embedder.changed_model_digest()
            {
                let settings = embedder.settings();
                Reporter::model_digest_changed(&settings.model, &settings.model_digest, &current);
            }
            if let Err(error) = Reporter::indexed(&report) {
                Reporter::indexing_failed(&error);
                return ExitCode::FAILURE;
            }
            if !report.is_success() {
                return ExitCode::FAILURE;
            }
        }
        Commands::Search {
            query,
            limit,
            output,
        } => {
            let embedder = match initialize_embedder(&config) {
                Ok(embedder) => embedder,
                Err(status) => return status,
            };
            let result = search::run(
                &query,
                limit,
                config.max_file_bytes,
                &config.database,
                &embedder,
            )
            .and_then(|hits| {
                let stdout = std::io::stdout().lock();
                match output {
                    OutputFormat::Console => ConsoleOutput::new(stdout).write_results(&hits),
                    OutputFormat::Json => JsonOutput::new(stdout).write_results(&hits),
                }?;
                Ok(())
            });
            if let Err(error) = result {
                Reporter::searching_failed(error.as_ref());
                return ExitCode::FAILURE;
            }
        }
    }

    ExitCode::SUCCESS
}

/// Initialize the embedding component using `config` for indexing or search.
///
/// Returns the embedder, or a failure exit status after reporting invalid settings
/// or a runtime initialization error.
fn initialize_embedder(config: &Config) -> Result<OllamaEmbedder, ExitCode> {
    OllamaEmbedder::new(&config.ollama).map_err(|error| {
        Reporter::embedding_init_failed(error.as_ref());
        ExitCode::FAILURE
    })
}
