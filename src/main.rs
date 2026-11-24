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
mod source_path;
mod template;
mod walker;

use clap::Parser;
use std::process::ExitCode;

use cli::{Cli, Commands, OutputFormat};
use config::Config;
use embeddings::OllamaEmbedder;
use reporter::Reporter;
use search_output::{ConsoleOutput, JsonOutput, SearchOutput};

/// Dispatch the CLI command using the selected configuration.
///
/// Returns success after a complete operation, or failure after reporting a
/// configuration, indexing, compaction, search, or output error. Indexing forwards
/// its automatic-compaction preference and derives the exit code from IndexReport.
/// The report can include published work alongside recoverable errors.
/// Indexing and search initialize the embedding client before entering their core operations.
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
            path,
            chunk_size,
            no_compact,
        } => {
            let embedder = match initialize_embedder(&config) {
                Ok(embedder) => embedder,
                Err(status) => return status,
            };
            let report = indexer::run(
                &path,
                chunk_size.unwrap_or(config.chunking.size),
                no_compact,
                &config.database,
                &embedder,
                &mut |event| reporter.handle(event),
            );
            reporter.finish();
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
            let result = search::run(query, limit, &config.database, &embedder).and_then(|hits| {
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
