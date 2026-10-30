use std::num::NonZeroUsize;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(version, about = "Local content indexing and search")]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Commands,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Commands {
    /// Index Markdown files in a directory
    Index {
        path: PathBuf,

        /// Maximum chunk size in Unicode characters
        #[arg(long, default_value = "2000")]
        chunk_size: NonZeroUsize,

        /// Glass mode: stop on the first traversal or processing error
        #[arg(long)]
        glass: bool,
    },

    /// Search indexed content
    Search {
        query: String,

        #[arg(short, long, default_value_t = 5)]
        limit: usize,
    },
}
