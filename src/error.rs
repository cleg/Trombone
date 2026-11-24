//! Shared error type for fallible application operations.

/// A shared error at application boundaries.
pub(crate) type Error = Box<dyn std::error::Error + Send + Sync>;

/// Application result that can carry filesystem, storage, or embedding errors.
pub(crate) type Result<T> = std::result::Result<T, Error>;
