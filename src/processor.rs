use std::io::{self, Write};
use std::num::NonZeroUsize;
use std::path::Path;

use crate::adapter::{Adapter, Document};
use crate::adapters::markdown::MarkdownAdapter;

pub(crate) fn process(path: &Path, chunk_size: NonZeroUsize) -> io::Result<()> {
    let document = open_document(path, chunk_size)?;
    let mut output = io::stdout().lock();
    writeln!(output, "File: {}", document.metadata.source.display())?;
    for (index, chunk) in document.chunks.enumerate() {
        let chunk = chunk?;
        writeln!(output, "\n--- Chunk {} ---", index + 1)?;
        if let Some(path) = chunk.section_path {
            writeln!(output, "Section: {}", path.join(" → "))?;
        }
        writeln!(
            output,
            "Location: {}",
            serde_json::to_string(&chunk.attributes)?
        )?;
        writeln!(output, "{}", chunk.text)?;
    }
    Ok(())
}

fn open_document(path: &Path, chunk_size: NonZeroUsize) -> io::Result<Document> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("md") => MarkdownAdapter.open(path, chunk_size),
        _ => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "no adapter for this file extension",
        )),
    }
}
