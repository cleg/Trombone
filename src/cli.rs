use std::num::NonZeroUsize;
use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

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
    /// Rebuild the vector index from current chunks without regenerating embeddings
    Compact,

    /// Index Markdown from all configured source directories
    Index {
        /// Maximum chunk size in Unicode characters
        #[arg(long)]
        chunk_size: Option<NonZeroUsize>,

        /// Defer automatic compaction while retaining accumulated removal counts
        #[arg(long)]
        no_compact: bool,
    },

    /// Search indexed content
    Search {
        query: String,

        #[arg(short, long, default_value_t = 5)]
        limit: usize,

        /// Format for search results
        #[arg(long, value_enum, default_value_t = OutputFormat::Console)]
        output: OutputFormat,
    },
}

#[derive(Clone, Debug, ValueEnum)]
pub(crate) enum OutputFormat {
    Console,
    Json,
}
