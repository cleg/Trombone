use std::error::Error;
use std::fs;
use std::io;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::embeddings;

pub(crate) fn default_path() -> io::Result<PathBuf> {
    let home = std::env::home_dir().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "Cannot determine home directory; specify --config",
        )
    })?;
    Ok(home.join(".config/trombone/trombone.toml"))
}

/// Default maximum source file size read into memory (50 MiB).
pub(crate) const DEFAULT_MAX_FILE_BYTES: NonZeroU64 = NonZeroU64::new(50 * 1024 * 1024).unwrap();

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Config {
    #[serde(default)]
    roots: Vec<PathBuf>,
    /// Maximum source file size in bytes that indexing and search will read into memory.
    #[serde(default = "default_max_file_bytes")]
    pub(crate) max_file_bytes: NonZeroU64,
    pub(crate) ollama: OllamaConfig,
    pub(crate) database: DatabaseConfig,
    #[serde(default)]
    pub(crate) chunking: ChunkingConfig,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DatabaseConfig {
    /// Directory that holds the SQLite database and USearch snapshots.
    pub(crate) path: PathBuf,
}

impl DatabaseConfig {
    pub(crate) fn sqlite_path(&self) -> PathBuf {
        self.path.join("trombone.sqlite")
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OllamaConfig {
    #[serde(default = "default_ollama_url")]
    pub(crate) url: String,
    pub(crate) model: String,
    pub(crate) expected_dimensions: Option<NonZeroUsize>,
    pub(crate) chunk_format: Option<String>,
    pub(crate) query_format: Option<String>,
    /// Maximum time allowed to establish an Ollama connection, in seconds.
    #[serde(default = "default_connect_timeout_secs")]
    pub(crate) connect_timeout_secs: NonZeroU64,
    /// Maximum total duration of one Ollama request, in seconds.
    #[serde(default = "default_request_timeout_secs")]
    pub(crate) request_timeout_secs: NonZeroU64,
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

/// Return the nonzero default maximum source file size (50 MiB).
fn default_max_file_bytes() -> NonZeroU64 {
    DEFAULT_MAX_FILE_BYTES
}

/// Return the nonzero default Ollama connection timeout of ten seconds.
fn default_connect_timeout_secs() -> NonZeroU64 {
    NonZeroU64::new(10).unwrap()
}

/// Return the nonzero default total Ollama request timeout of two minutes.
fn default_request_timeout_secs() -> NonZeroU64 {
    NonZeroU64::new(120).unwrap()
}

impl Config {
    /// Load the TOML configuration at `path` and validate its templates and storage directory.
    ///
    /// Returns the configuration with relative database and source paths resolved against
    /// the real config file location after following symlinks.
    ///
    /// # Errors
    ///
    /// Returns an error for unreadable or invalid TOML, invalid embedding input
    /// templates, a config location that cannot be resolved, or an inaccessible
    /// or non-directory database location.
    pub(crate) fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        let text = fs::read_to_string(path)?;
        let mut config: Self = toml::from_str(&text)?;
        embeddings::validate_formats(&config.ollama)?;
        if config.database.path.is_relative() || config.roots.iter().any(|root| root.is_relative())
        {
            let config_path = fs::canonicalize(path)?;
            let directory = config_path.parent().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "Config path has no parent")
            })?;
            if config.database.path.is_relative() {
                config.database.path = directory.join(&config.database.path);
            }
            for root in &mut config.roots {
                if !root.as_os_str().is_empty() && root.is_relative() {
                    *root = directory.join(&*root);
                }
            }
        }
        let directory = &config.database.path;
        let metadata = fs::metadata(directory).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "Cannot access database directory {}: {error}",
                    directory.display()
                ),
            )
        })?;
        if !metadata.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                format!(
                    "database.path must point to a directory: {}",
                    directory.display()
                ),
            )
            .into());
        }
        Ok(config)
    }

    /// Return the configured roots for indexing without checking their availability.
    ///
    /// Returns a configuration error if the list or any of its paths is empty.
    pub(crate) fn index_roots(&self) -> io::Result<&[PathBuf]> {
        if self.roots.is_empty() || self.roots.iter().any(|root| root.as_os_str().is_empty()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "roots must contain at least one directory path and no empty paths",
            ));
        }
        Ok(&self.roots)
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// Verify relative roots and database paths are anchored to the real TOML.
    ///
    /// Returns success for the resolved paths, including an unavailable root,
    /// or propagates fixture and configuration errors.
    #[test]
    fn relative_paths_follow_the_real_config_location() -> Result<(), Box<dyn Error>> {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir()?;
        let real = fs::canonicalize(directory.path())?.join("config");
        fs::create_dir(&real)?;
        let config_path = real.join("trombone.toml");
        let alias = directory.path().join("linked.toml");
        fs::write(
            &config_path,
            "roots = ['unavailable', '/absolute']\n[database]\npath = '.'\n[ollama]\nmodel = 'test'",
        )?;
        symlink(&config_path, &alias)?;

        let config = Config::load(&alias)?;
        assert_eq!(config.database.path, real);
        assert_eq!(
            config.index_roots()?,
            [real.join("unavailable"), PathBuf::from("/absolute")]
        );
        Ok(())
    }
}
