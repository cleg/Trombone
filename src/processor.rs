use std::io;
use std::num::NonZeroUsize;
use std::path::Path;

use ollama_rs::{Ollama, generation::embeddings::request::GenerateEmbeddingsRequest};

use crate::adapter::{Adapter, Document};
use crate::adapters::markdown::MarkdownAdapter;
use crate::config::OllamaConfig;
use crate::database::Result;

pub(crate) async fn probe_embedding(
    path: &Path,
    chunk_size: NonZeroUsize,
    ollama: &Ollama,
    config: &OllamaConfig,
) -> Result<Vec<f32>> {
    let expected = expected_dimensions(config)?;
    let source = std::fs::read_to_string(path)?;
    let chunk = open_document(path, &source, chunk_size)?
        .chunks
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "No chunks in the file"))?;
    embed(ollama, &config.model, expected, chunk.text).await
}

pub(crate) fn expected_dimensions(config: &OllamaConfig) -> io::Result<NonZeroUsize> {
    config.expected_dimensions.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Set ollama.expected_dimensions in the config",
        )
    })
}

pub(crate) async fn embed(
    ollama: &Ollama,
    model: &str,
    expected: NonZeroUsize,
    text: String,
) -> Result<Vec<f32>> {
    let request = GenerateEmbeddingsRequest::new(model.to_owned(), text.into());
    let response = ollama.generate_embeddings(request).await?;
    let [embedding] = <[_; 1]>::try_from(response.embeddings).map_err(|embeddings| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Expected one embedding, got {}", embeddings.len()),
        )
    })?;
    let actual = embedding.len();
    if actual != expected.get() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Model {model} returned {actual} dimensions, expected {expected}"),
        )
        .into());
    }
    Ok(embedding)
}

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
