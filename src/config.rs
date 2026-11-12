use std::error::Error;
use std::fs;
use std::io;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use serde::Deserialize;

pub(crate) fn default_path() -> io::Result<PathBuf> {
    let home = std::env::home_dir().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "Cannot determine home directory; specify --config",
        )
    })?;
    Ok(home.join(".config/trombone/trombone.toml"))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Config {
    pub(crate) ollama: OllamaConfig,
    pub(crate) database: DatabaseConfig,
    #[serde(default)]
    pub(crate) chunking: ChunkingConfig,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DatabaseConfig {
    pub(crate) path: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OllamaConfig {
    #[serde(default = "default_ollama_url")]
    pub(crate) url: String,
    pub(crate) model: String,
    pub(crate) expected_dimensions: Option<NonZeroUsize>,
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct ChunkingConfig {
    pub(crate) size: NonZeroUsize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            size: NonZeroUsize::new(2000).unwrap(),
        }
    }
}

fn default_ollama_url() -> String {
    "http://localhost:11434".to_owned()
}

impl Config {
    pub(crate) fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        let text = fs::read_to_string(path)?;
        let mut config: Self = toml::from_str(&text)?;
        if config.database.path.is_relative() {
            let config_path = fs::canonicalize(path)?;
            let directory = config_path.parent().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "Config path has no parent")
            })?;
            config.database.path = directory.join(&config.database.path);
        }
        Ok(config)
    }
}
