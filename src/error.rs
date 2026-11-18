//! Shared error type for fallible application operations.

/// Application result that can carry filesystem, storage, or embedding errors.
pub(crate) type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
