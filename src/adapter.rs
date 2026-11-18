use std::io;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

pub(crate) type Attributes = Map<String, Value>;

pub(crate) struct DocumentMetadata {
    pub(crate) source: PathBuf,
    pub(crate) attributes: Attributes,
}

/// Coordinates of a nonempty excerpt in the original UTF-8 source text.
///
/// Byte offsets form a zero-based, half-open range. Line numbers are one-based
/// and inclusive; a trailing newline belongs to the line it terminates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SourceLocation {
    /// Byte offset of the first source byte in the excerpt.
    pub(crate) byte_start: usize,
    /// Byte offset immediately after the excerpt's final source byte.
    pub(crate) byte_end: usize,
    /// Number of the first source line touched by the excerpt.
    pub(crate) line_start: usize,
    /// Number of the last source line touched by the excerpt.
    pub(crate) line_end: usize,
}

pub(crate) struct Chunk {
    pub(crate) text: String,
    pub(crate) section_path: Option<Vec<String>>,
    pub(crate) location: SourceLocation,
    pub(crate) attributes: Attributes,
}

// Chunks may borrow the source text, which must outlive their iterator.
pub(crate) struct Document<'a> {
    pub(crate) metadata: DocumentMetadata,
    pub(crate) chunks: Box<dyn Iterator<Item = Chunk> + 'a>,
}

pub(crate) trait Adapter {
    fn open<'a>(
        &self,
        path: &Path,
        source: &'a str,
        chunk_size: NonZeroUsize,
    ) -> io::Result<Document<'a>>;
}
