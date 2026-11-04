mod adapter;
mod adapters;
mod cli;
mod config;
mod database;
mod processor;
mod walker;

use clap::Parser;
use ollama_rs::Ollama;
use std::ops::ControlFlow;
use std::process::ExitCode;

use cli::{Cli, Commands};
use config::Config;
use walker::Walker;

fn main() -> ExitCode {
    let cli = Cli::parse();
    let config_path = match cli.config.map_or_else(config::default_path, Ok) {
        Ok(path) => path,
        Err(error) => {
            eprintln!("Failed to locate config: {error}");
            return ExitCode::FAILURE;
        }
    };
    let config = match Config::load(&config_path) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("Failed to load config {}: {error}", config_path.display());
            return ExitCode::FAILURE;
        }
    };
    let _ollama = match Ollama::try_new(&config.ollama.url) {
        Ok(ollama) => ollama,
        Err(error) => {
            eprintln!("Failed to initialize Ollama client: {error}");
            return ExitCode::FAILURE;
        }
    };
    // Ollama selects the model per request, not when constructing the client.
    let _embedding_model = config.ollama.model;
    let mut database = match database::open(&config.database.path) {
        Ok(connection) => connection,
        Err(error) => {
            eprintln!(
                "Failed to initialize database {}: {error}",
                config.database.path.display()
            );
            return ExitCode::FAILURE;
        }
    };

    match cli.command {
        Commands::Index {
            path,
            chunk_size,
            glass,
        } => {
            let chunk_size = chunk_size.unwrap_or(config.chunking.size);
            let walker = Walker {
                continue_on_error: !glass,
            };
            if walker
                .walk(&path, |file| {
                    processor::process(file, chunk_size, &mut database)?;
                    Ok(ControlFlow::Continue(()))
                })
                .is_err()
            {
                return ExitCode::FAILURE;
            }
        }
        Commands::Search { query, limit } => {
            println!("Searching: {query}, limit: {limit}");
        }
    }

    ExitCode::SUCCESS
}
