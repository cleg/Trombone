use std::io;
use std::num::NonZeroUsize;
use std::path::Path;

use crate::adapter::{Adapter, Document};
use crate::adapters::markdown::MarkdownAdapter;

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
