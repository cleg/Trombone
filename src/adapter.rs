use std::io;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

pub(crate) type Attributes = Map<String, Value>;

pub(crate) struct DocumentMetadata {
    pub(crate) source: PathBuf,
    pub(crate) attributes: Attributes,
}

pub(crate) struct Chunk {
    pub(crate) text: String,
    pub(crate) section_path: Option<Vec<String>>,
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
