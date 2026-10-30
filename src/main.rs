mod adapter;
mod adapters;
mod cli;
mod processor;
mod walker;

use clap::Parser;
use std::ops::ControlFlow;
use std::process::ExitCode;

use cli::{Cli, Commands};
use walker::Walker;

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Commands::Index {
            path,
            chunk_size,
            glass,
        } => {
            let walker = Walker {
                continue_on_error: !glass,
            };
            if walker
                .walk(&path, |file| {
                    processor::process(file, chunk_size)?;
                    // Preview one document during this iteration.
                    Ok(ControlFlow::Break(()))
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
