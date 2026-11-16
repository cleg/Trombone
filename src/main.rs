mod adapter;
mod adapters;
mod cli;
mod config;
mod database;
mod indexer;
mod processor;
mod reporter;
mod walker;

use clap::Parser;
use ollama_rs::Ollama;
use std::process::ExitCode;

use cli::{Cli, Commands};
use config::Config;
use reporter::Reporter;

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
    let ollama = match Ollama::try_new(&config.ollama.url) {
        Ok(ollama) => ollama,
        Err(error) => {
            Reporter::ollama_init_failed(&error);
            return ExitCode::FAILURE;
        }
    };
    match cli.command {
        Commands::Index {
            path,
            chunk_size,
            glass,
        } => {
            match indexer::run(
                &path,
                chunk_size.unwrap_or(config.chunking.size),
                glass,
                &config,
                &ollama,
            ) {
                Ok(true) => {}
                Ok(false) => return ExitCode::FAILURE,
                Err(error) => {
                    Reporter::indexing_failed(error.as_ref());
                    return ExitCode::FAILURE;
                }
            }
        }
        Commands::Search { query, limit } => {
            Reporter::search_preview(&query, limit);
        }
    }

    ExitCode::SUCCESS
}
