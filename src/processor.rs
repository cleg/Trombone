use std::fs::File;
use std::io::{self, Read};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;

use crate::adapter::{Adapter, Document};
use crate::adapters::markdown::MarkdownAdapter;

/// Read a UTF-8 source file at `path`, rejecting files larger than `max_bytes`.
///
/// Checks the opened file's metadata length first and caps reading with
/// [`Read::take`] so concurrent growth cannot exceed `max_bytes` in memory.
///
/// # Errors
///
/// Returns an I/O error if opening or reading fails, [`io::ErrorKind::FileTooLarge`]
/// if the file exceeds `max_bytes`, or [`io::ErrorKind::InvalidData`] if it contains
/// invalid UTF-8.
pub(crate) fn read_source_file(path: &Path, max_bytes: NonZeroU64) -> io::Result<String> {
    let limit = max_bytes.get();
    let file = File::open(path)?;
    let len = file.metadata()?.len();
    if len > limit {
        return Err(io::Error::new(
            io::ErrorKind::FileTooLarge,
            format!("File exceeds maximum size of {limit} bytes"),
        ));
    }
    let mut bytes = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
    file.take(limit.saturating_add(1)).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(io::Error::new(
            io::ErrorKind::FileTooLarge,
            format!("File exceeds maximum size of {limit} bytes"),
        ));
    }
    String::from_utf8(bytes).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

/// Open the document at `path` from its already-read UTF-8 `source`.
///
/// `chunk_size` is the maximum number of Unicode characters per chunk. Returns
/// document metadata and a lazy chunk iterator borrowing the supplied source.
///
/// # Errors
///
/// Returns an error when the extension is unsupported or the adapter cannot open it.
pub(crate) fn open_document<'a>(
    path: &Path,
    source: &'a str,
    chunk_size: NonZeroUsize,
) -> io::Result<Document<'a>> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("md") => MarkdownAdapter.open(path, source, chunk_size),
        _ => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "no adapter for this file extension",
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    /// Verify `read_source_file` accepts files up to the limit and rejects oversized files.
    ///
    /// Returns success when boundary sizes behave as expected, or propagates
    /// temporary-directory and write errors.
    #[test]
    fn read_source_file_enforces_max_bytes() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("note.md");
        fs::write(&path, "12345")?;
        assert_eq!(
            read_source_file(&path, NonZeroU64::new(5).unwrap())?,
            "12345"
        );
        let error = read_source_file(&path, NonZeroU64::new(4).unwrap()).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::FileTooLarge);
        assert!(error.to_string().contains("maximum size"));
        Ok(())
    }
}
