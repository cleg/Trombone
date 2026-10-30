use std::io;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

pub(crate) type Attributes = Map<String, Value>;

pub(crate) struct DocumentMetadata {
    pub(crate) source: PathBuf,
    #[expect(dead_code, reason = "Metadata persistence is not implemented yet")]
    pub(crate) attributes: Attributes,
}

pub(crate) struct Chunk {
    pub(crate) text: String,
    pub(crate) section_path: Option<Vec<String>>,
    pub(crate) attributes: Attributes,
}

pub(crate) struct Document {
    pub(crate) metadata: DocumentMetadata,
    pub(crate) chunks: Box<dyn Iterator<Item = io::Result<Chunk>>>,
}

pub(crate) trait Adapter {
    fn open(&self, path: &Path, chunk_size: NonZeroUsize) -> io::Result<Document>;
}
