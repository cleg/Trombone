use std::io::{self, Write};
use std::num::NonZeroUsize;
use std::path::Path;

use rusqlite::Connection;

use crate::adapter::{Adapter, Document};
use crate::adapters::markdown::MarkdownAdapter;
use crate::database::{self, IndexStatus};

pub(crate) fn process(
    path: &Path,
    chunk_size: NonZeroUsize,
    connection: &mut Connection,
) -> io::Result<()> {
    let source = std::fs::read_to_string(path)?;
    let content_hash = blake3::hash(source.as_bytes()).to_hex().to_string();
    let status = database::save(connection, path, &content_hash, chunk_size, || {
        open_document(path, &source, chunk_size)
    })
    .map_err(io::Error::other)?;
    let action = match status {
        IndexStatus::Unchanged => "Unchanged",
        IndexStatus::Indexed => "Indexed",
        IndexStatus::Reused => "Reused chunks",
    };
    writeln!(io::stdout().lock(), "{action}: {}", path.display())
}

fn open_document<'a>(
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
