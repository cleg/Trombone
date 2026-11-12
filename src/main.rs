mod adapter;
mod adapters;
mod cli;
mod config;
mod database;
mod indexer;
mod processor;
mod vector_probe;
mod walker;

use clap::Parser;
use ollama_rs::Ollama;
use std::process::ExitCode;

use cli::{Cli, Commands};
use config::Config;

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
    let ollama = match Ollama::try_new(&config.ollama.url) {
        Ok(ollama) => ollama,
        Err(error) => {
            eprintln!("Failed to initialize Ollama client: {error}");
            return ExitCode::FAILURE;
        }
    };
    match cli.command {
        Commands::ProbeEmbedding { path, usearch } => {
            let result = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(Box::<dyn std::error::Error + Send + Sync>::from)
                .and_then(|runtime| {
                    runtime.block_on(processor::probe_embedding(
                        &path,
                        config.chunking.size,
                        &ollama,
                        &config.ollama,
                    ))
                });
            match result {
                Ok(embedding) => {
                    println!("Embedding dimensions: {}", embedding.len());
                    if usearch {
                        match vector_probe::round_trip(&embedding) {
                            Ok((key, distance)) => {
                                println!("USearch matched key {key}, cosine distance: {distance}");
                            }
                            Err(error) => {
                                eprintln!("USearch probe failed: {error}");
                                return ExitCode::FAILURE;
                            }
                        }
                    }
                }
                Err(error) => {
                    eprintln!("Failed to embed first chunk of {}: {error}", path.display());
                    return ExitCode::FAILURE;
                }
            }
        }
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
                    eprintln!("Indexing failed: {error}");
                    return ExitCode::FAILURE;
                }
            }
        }
        Commands::Search { query, limit } => {
            println!("Searching: {query}, limit: {limit}");
        }
    }

    ExitCode::SUCCESS
}
