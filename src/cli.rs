use std::num::NonZeroUsize;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(version, about = "Local content indexing and search")]
pub(crate) struct Cli {
    /// Path to the TOML configuration file (default: ~/.config/trombone/trombone.toml)
    #[arg(long, global = true)]
    pub(crate) config: Option<PathBuf>,

    #[command(subcommand)]
    pub(crate) command: Commands,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Commands {
    /// Generate and check an embedding for the first chunk of a Markdown file
    ProbeEmbedding {
        path: PathBuf,

        /// Save, reopen, and search a temporary USearch index
        #[arg(long)]
        usearch: bool,
    },

    /// Index Markdown files in a directory
    Index {
        path: PathBuf,

        /// Maximum chunk size in Unicode characters
        #[arg(long)]
        chunk_size: Option<NonZeroUsize>,

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
