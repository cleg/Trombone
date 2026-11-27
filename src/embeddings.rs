//! Prepares model input and obtains validated embeddings behind a synchronous API.

use std::future::Future;
use std::io;
use std::num::NonZeroUsize;
use std::time::Duration;

use ollama_rs::{Ollama, generation::embeddings::request::GenerateEmbeddingsRequest};
use serde::Deserialize;
use tokio::runtime::Runtime;

use crate::config::OllamaConfig;
use crate::error::Result;
use crate::template;

/// Settings that identify the embedding space used by a persisted index.
pub(crate) struct EmbeddingSettings {
    pub(crate) model: String,
    pub(crate) model_digest: String,
    pub(crate) dimensions: NonZeroUsize,
    pub(crate) chunk_format: Option<String>,
}

/// Synchronous embedding boundary used by indexing and search.
pub(crate) trait Embedder {
    /// Return this embedder's model, resolved digest, dimensions, and document input format.
    fn settings(&self) -> &EmbeddingSettings;

    /// Embed the exact prepared model `input` for a chunk.
    ///
    /// Returns one vector matching `settings().dimensions`, or an error if
    /// generation or response validation fails.
    fn embed_chunk(&self, input: String) -> Result<Vec<f32>>;

    /// Embed `query` using the model's configured search-query format.
    ///
    /// Returns one vector matching `settings().dimensions`, or an error if input
    /// formatting, generation, or response validation fails.
    fn embed_query(&self, query: &str) -> Result<Vec<f32>>;
}

/// Ollama implementation owning the client, input formats, and one Tokio runtime.
pub(crate) struct OllamaEmbedder {
    client: Ollama,
    http_client: reqwest::Client,
    runtime: Runtime,
    settings: EmbeddingSettings,
    query_format: Option<String>,
    request_timeout: Duration,
}

impl OllamaEmbedder {
    /// Construct an embedder from `config` and resolve its model digest through Ollama.
    ///
    /// Returns an embedder with one runtime and HTTP client reused by all requests.
    /// Model metadata is fetched without generating embeddings. The configured
    /// connection timeout and total request deadline apply to all service calls.
    ///
    /// # Errors
    ///
    /// Returns an error for missing dimensions, invalid templates or URL, or
    /// failure to initialize the runtime, unavailable model metadata, or an
    /// absent or invalid model digest, HTTP client construction, or a request timeout.
    pub(crate) fn new(config: &OllamaConfig) -> Result<Self> {
        validate_formats(config)?;
        let dimensions = config.expected_dimensions.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Set ollama.expected_dimensions in the config",
            )
        })?;
        let http_client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(config.connect_timeout_secs.get()))
            .build()?;
        let url = normalize_base_url(&config.url)?;
        let client = Ollama::builder()
            .url(url)
            .reqwest_client(http_client.clone())
            .build();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let request_timeout = Duration::from_secs(config.request_timeout_secs.get());
        let model_digest = resolve_model_digest(
            &runtime,
            request_timeout,
            &http_client,
            &client,
            &config.model,
        )?;
        Ok(Self {
            client,
            http_client,
            runtime,
            settings: EmbeddingSettings {
                model: config.model.clone(),
                model_digest,
                dimensions,
                chunk_format: config.chunk_format.clone(),
            },
            query_format: config.query_format.clone(),
            request_timeout,
        })
    }

    /// Return the current digest if the model tag now resolves to a different version.
    ///
    /// Returns `None` when the tag still resolves to the digest captured at startup.
    /// The original embedding settings remain available for snapshot compatibility.
    ///
    /// # Errors
    ///
    /// Returns an error if Ollama is unavailable or no longer reports a valid digest.
    pub(crate) fn changed_model_digest(&self) -> Result<Option<String>> {
        let current = resolve_model_digest(
            &self.runtime,
            self.request_timeout,
            &self.http_client,
            &self.client,
            &self.settings.model,
        )?;
        Ok((current != self.settings.model_digest).then_some(current))
    }

    /// Request one embedding for already formatted `text` using this client's runtime.
    ///
    /// Returns the validated vector, or an error for service, timeout, or response failures.
    fn embed(&self, text: String) -> Result<Vec<f32>> {
        let request = GenerateEmbeddingsRequest::new(self.settings.model.clone(), text.into());
        let response = request_with_timeout(
            &self.runtime,
            self.request_timeout,
            "embedding generation",
            async { Ok(self.client.generate_embeddings(request).await?) },
        )?;
        validate_response(
            response.embeddings,
            &self.settings.model,
            self.settings.dimensions,
        )
    }
}

/// Run `request` on `runtime` with the total duration bounded by `timeout`.
///
/// `operation` names the service call in diagnostics. Returns the request's value,
/// propagates its error, or returns a timed-out I/O error after the deadline.
/// The timer is created inside the runtime so initialization needs no async context.
fn request_with_timeout<T>(
    runtime: &Runtime,
    timeout: Duration,
    operation: &str,
    request: impl Future<Output = Result<T>>,
) -> Result<T> {
    runtime.block_on(async {
        tokio::time::timeout(timeout, request).await.map_err(|_| {
            io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "Ollama {operation} timed out after {} seconds; adjust ollama.request_timeout_secs if needed",
                    timeout.as_secs(),
                ),
            )
        })?
    })
}

/// Model metadata retained from Ollama's tags endpoint, including fields omitted by ollama-rs.
#[derive(Deserialize)]
struct AvailableModel {
    name: String,
    digest: Option<String>,
}

/// Response envelope for the model-list metadata request.
#[derive(Deserialize)]
struct AvailableModels {
    models: Vec<AvailableModel>,
}

/// Parse `raw_url` as an Ollama base URL with a trailing path slash.
///
/// Ensuring the path ends with `/` lets relative endpoints resolve beneath
/// reverse-proxy path prefixes as well as root URLs.
///
/// # Errors
///
/// Returns an error if `raw_url` cannot be parsed as a URL.
fn normalize_base_url(raw_url: &str) -> Result<reqwest::Url> {
    let mut url = reqwest::Url::parse(raw_url)?;
    if !url.path().ends_with('/') {
        url.set_path(&format!("{}/", url.path()));
    }
    Ok(url)
}

/// Build the Ollama `/api/tags` endpoint URL from `base_url`.
///
/// Preserves any path prefix configured on `base_url`.
///
/// # Errors
///
/// Returns an error if `base_url` is invalid or cannot be joined with `api/tags`.
fn tags_url(base_url: &str) -> Result<reqwest::Url> {
    Ok(normalize_base_url(base_url)?.join("api/tags")?)
}

/// Resolve `model` through `client`'s server using the shared `http_client`.
///
/// Runs on `runtime` with the specified `timeout`. Returns the normalized
/// SHA-256 digest without generating an embedding.
///
/// # Errors
///
/// Returns an error for timeouts, HTTP or JSON failures, a missing model, or an invalid digest.
fn resolve_model_digest(
    runtime: &Runtime,
    timeout: Duration,
    http_client: &reqwest::Client,
    client: &Ollama,
    model: &str,
) -> Result<String> {
    let url = tags_url(client.url_str())?;
    request_with_timeout(runtime, timeout, "model metadata lookup", async {
        let available: AvailableModels = http_client
            .get(url)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        select_model_digest(&available.models, model)
    })
}

/// Select `model` from `available`, treating an omitted tag as `latest`.
///
/// Returns the normalized 64-digit SHA-256 digest. Missing metadata or an invalid
/// digest returns an error rather than accepting the model name as its identity.
fn select_model_digest(available: &[AvailableModel], model: &str) -> Result<String> {
    let name = model_with_tag(model);
    let selected = available
        .iter()
        .find(|candidate| model_with_tag(&candidate.name) == name)
        .ok_or_else(|| {
            io::Error::other(format!(
                "Embedding model {model} is not available in Ollama"
            ))
        })?;
    let digest = selected.digest.as_deref().unwrap_or("");
    let digest = digest.strip_prefix("sha256:").unwrap_or(digest);
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(io::Error::other(format!(
            "Ollama returned a missing or invalid digest for embedding model {model}"
        ))
        .into());
    }
    Ok(digest.to_ascii_lowercase())
}

/// Return `model` with an explicit tag, preserving namespace and registry port separators.
fn model_with_tag(model: &str) -> String {
    if model.rsplit('/').next().unwrap_or(model).contains(':') {
        model.to_owned()
    } else {
        format!("{model}:latest")
    }
}

impl Embedder for OllamaEmbedder {
    /// Return the resolved identity and formatting settings shared by this client's embeddings.
    fn settings(&self) -> &EmbeddingSettings {
        &self.settings
    }

    /// Send prepared `input` to Ollama, returning a validated vector or a service/response error.
    fn embed_chunk(&self, input: String) -> Result<Vec<f32>> {
        self.embed(input)
    }

    /// Format `query` for search and return its embedding.
    ///
    /// Returns an error for invalid formatting, service failures, or an invalid vector.
    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let text = format_query(self.query_format.as_deref(), query)?;
        self.embed(text)
    }
}

/// Validate optional input templates in `config` without requiring a client or model.
///
/// Returns success for valid document and query templates. This lets configuration
/// loading reject invalid formats even for commands that do not generate vectors.
///
/// # Errors
///
/// Returns an invalid-input error for unsupported placeholders or malformed braces.
pub(crate) fn validate_formats(config: &OllamaConfig) -> io::Result<()> {
    format_chunk(config.chunk_format.as_deref(), None, "")?;
    format_query(config.query_format.as_deref(), "")?;
    Ok(())
}

/// Format chunk `text` with optional ancestor `headings` using `format` when set.
///
/// Returns the model input, or an invalid-input error for a malformed template.
pub(crate) fn format_chunk(
    format: Option<&str>,
    headings: Option<&[String]>,
    text: &str,
) -> io::Result<String> {
    let title = headings.map(|headings| headings.join(" > "));
    match format {
        Some(format) => render_format(
            format,
            "chunk_format",
            &[
                ("text", text),
                ("title", title.as_deref().unwrap_or("none")),
            ],
            &["text"],
        ),
        None => Ok(match title {
            Some(title) => format!("{title}\n\n{text}"),
            None => text.to_owned(),
        }),
    }
}

/// Format a search `query` using `format`, or return it unchanged when unset.
///
/// Returns the model input, or an invalid-input error for a malformed template.
fn format_query(format: Option<&str>, query: &str) -> io::Result<String> {
    match format {
        Some(format) => render_format(format, "query_format", &[("query", query)], &["query"]),
        None => Ok(query.to_owned()),
    }
}

/// Substitute `values` in `format`, requiring all placeholders in `required`.
///
/// `setting` identifies the configuration field in error messages. Returns the
/// rendered string, or an invalid-input error describing the template failure.
fn render_format(
    format: &str,
    setting: &str,
    values: &[(&str, &str)],
    required: &[&str],
) -> io::Result<String> {
    template::render(format, values, required).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("ollama.{setting}: {error}"),
        )
    })
}

/// Validate a single-input response from `model` against `expected` dimensions.
///
/// `embeddings` contains all returned vectors. Returns exactly one vector with
/// all finite components, or an invalid-data error if the response cardinality,
/// vector length, or float values are invalid.
fn validate_response(
    embeddings: Vec<Vec<f32>>,
    model: &str,
    expected: NonZeroUsize,
) -> Result<Vec<f32>> {
    let [embedding] = <[_; 1]>::try_from(embeddings).map_err(|embeddings| {
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
    if embedding.iter().any(|value| !value.is_finite()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Model {model} returned a non-finite embedding value"),
        )
        .into());
    }
    Ok(embedding)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::fs;

    use super::*;
    use crate::config::DatabaseConfig;
    use crate::{compact, indexer, search};

    /// Verify request deadlines return a recognizable timeout and successful values pass through.
    ///
    /// Returns success on both outcomes, or a runtime/request error.
    #[test]
    fn request_deadline_bounds_a_stalled_service_call() -> Result<()> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        assert_eq!(
            request_with_timeout(&runtime, Duration::from_secs(1), "test request", async {
                Ok(42)
            })?,
            42,
        );
        let error = request_with_timeout(
            &runtime,
            Duration::from_millis(1),
            "test request",
            std::future::pending::<Result<()>>(),
        )
        .unwrap_err();
        assert_eq!(
            error.downcast_ref::<io::Error>().unwrap().kind(),
            io::ErrorKind::TimedOut,
        );
        assert!(error.to_string().contains("test request"));
        Ok(())
    }

    /// Verify default tags, registry ports, and SHA-256 spelling resolve one model identity.
    ///
    /// Returns success on the expected digest; propagates model-resolution errors.
    #[test]
    fn resolves_model_digest_with_explicit_and_default_tags() -> Result<()> {
        let available = vec![AvailableModel {
            name: "registry:11434/team/model:latest".into(),
            digest: Some(format!("sha256:{}", "A".repeat(64))),
        }];
        assert_eq!(
            select_model_digest(&available, "registry:11434/team/model")?,
            "a".repeat(64)
        );
        assert_eq!(
            select_model_digest(&available, "registry:11434/team/model:latest")?,
            "a".repeat(64)
        );
        assert!(select_model_digest(&available, "registry:11434/team/model:other").is_err());
        let bare = vec![AvailableModel {
            name: "model".into(),
            digest: Some("b".repeat(64)),
        }];
        assert_eq!(select_model_digest(&bare, "model:latest")?, "b".repeat(64));
        Ok(())
    }

    /// Verify missing models and unusable digests cannot fall back to name-only identity.
    #[test]
    fn rejects_missing_models_and_invalid_digests() {
        assert!(select_model_digest(&[], "model").is_err());
        for digest in [
            None,
            Some(String::new()),
            Some("unknown".into()),
            Some("z".repeat(64)),
        ] {
            let available = vec![AvailableModel {
                name: "model:latest".into(),
                digest,
            }];
            assert!(select_model_digest(&available, "model").is_err());
        }
    }

    /// Verify default and custom inputs preserve heading paths and raw user text.
    ///
    /// Returns success for expected model inputs; propagates template errors.
    #[test]
    fn formats_documents_and_queries_without_changing_input_semantics() -> Result<()> {
        let headings = vec!["Root".into(), "Child".into()];
        assert_eq!(
            format_chunk(None, Some(&headings), "Body {text}.\n")?,
            "Root > Child\n\nBody {text}.\n",
        );
        assert_eq!(format_chunk(None, None, "Body.\n")?, "Body.\n");
        let format = Some("title: {title} | text: {text}");
        assert_eq!(
            format_chunk(format, Some(&headings), "Body.")?,
            "title: Root > Child | text: Body.",
        );
        assert_eq!(
            format_chunk(format, None, "Body.")?,
            "title: none | text: Body.",
        );
        assert_eq!(format_query(None, "raw {query}")?, "raw {query}");
        assert_eq!(
            format_query(Some("task: search result | query: {query}"), "raw {query}")?,
            "task: search result | query: raw {query}",
        );
        Ok(())
    }

    /// Verify template errors identify the setting and reject missing input placeholders.
    #[test]
    fn rejects_invalid_document_and_query_formats() {
        let error = format_chunk(Some("title: {title}"), None, "body").unwrap_err();
        assert!(error.to_string().contains("ollama.chunk_format"));
        assert!(format_query(Some("{text}"), "query").is_err());
    }

    /// Verify the embedding boundary rejects wrong response counts, dimensions, and non-finite values.
    ///
    /// Returns success for expected validation outcomes; propagates valid-response errors.
    #[test]
    fn validates_response_cardinality_and_vector_dimensions() -> Result<()> {
        let dimensions = NonZeroUsize::new(3).unwrap();
        assert!(validate_response(vec![], "model", dimensions).is_err());
        assert!(validate_response(vec![vec![1.0; 3], vec![1.0; 3]], "model", dimensions).is_err());
        assert!(validate_response(vec![vec![1.0; 2]], "model", dimensions).is_err());
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error =
                validate_response(vec![vec![1.0, bad, 0.0]], "model", dimensions).unwrap_err();
            assert_eq!(
                error.downcast_ref::<io::Error>().unwrap().kind(),
                io::ErrorKind::InvalidData,
            );
        }
        assert_eq!(
            validate_response(vec![vec![1.0, 0.0, 0.0]], "model", dimensions)?,
            [1.0, 0.0, 0.0],
        );
        Ok(())
    }

    /// Verify Ollama endpoint URLs preserve reverse-proxy path prefixes with or without a trailing slash.
    ///
    /// Returns success when every base URL resolves to the expected `/api/tags` endpoint,
    /// or propagates URL parsing errors.
    #[test]
    fn tags_url_preserves_base_path_prefixes() -> Result<()> {
        assert_eq!(
            tags_url("http://localhost:11434")?.as_str(),
            "http://localhost:11434/api/tags",
        );
        assert_eq!(
            tags_url("http://localhost:11434/")?.as_str(),
            "http://localhost:11434/api/tags",
        );
        assert_eq!(
            tags_url("http://localhost:11434/ollama")?.as_str(),
            "http://localhost:11434/ollama/api/tags",
        );
        assert_eq!(
            tags_url("http://localhost:11434/ollama/")?.as_str(),
            "http://localhost:11434/ollama/api/tags",
        );
        Ok(())
    }

    /// Embedding test implementation recording inputs received from the pipeline.
    struct RecordingEmbedder {
        settings: EmbeddingSettings,
        chunks: RefCell<Vec<String>>,
        queries: RefCell<Vec<String>>,
    }

    impl Embedder for RecordingEmbedder {
        /// Return the test embedding identity without constructing a service client.
        fn settings(&self) -> &EmbeddingSettings {
            &self.settings
        }

        /// Record the exact prepared `input`, returning a fixed three-dimensional vector.
        fn embed_chunk(&self, input: String) -> Result<Vec<f32>> {
            self.chunks.borrow_mut().push(input);
            Ok(vec![1.0, 0.0, 0.0])
        }

        /// Record raw `query` input and return a fixed three-dimensional vector.
        fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
            self.queries.borrow_mut().push(query.to_owned());
            Ok(vec![1.0, 0.0, 0.0])
        }
    }

    /// Verify typed excerpts, report success, and name/digest compatibility through the API.
    ///
    /// Returns success when both operations can borrow the same service-free
    /// embedder in the real storage pipeline. Propagates filesystem, SQLite,
    /// index, and search errors.
    #[test]
    fn indexing_and_search_depend_only_on_the_embedding_boundary() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let database = DatabaseConfig {
            path: directory.path().to_path_buf(),
        };
        let note = directory.path().join("note.md");
        let body = "Перший рядок.\nДругий 🙂.\n";
        fs::write(&note, format!("# Корінь 🙂\n\n{body}"))?;
        let mut embedder = RecordingEmbedder {
            settings: EmbeddingSettings {
                model: "test-model".into(),
                model_digest: "a".repeat(64),
                dimensions: NonZeroUsize::new(3).unwrap(),
                chunk_format: Some("title: {title} | text: {text}".into()),
            },
            chunks: RefCell::new(Vec::new()),
            queries: RefCell::new(Vec::new()),
        };
        let roots = [directory.path().to_path_buf()];
        let chunk_size = NonZeroUsize::new(1_000).unwrap();
        let max_file_bytes = crate::config::DEFAULT_MAX_FILE_BYTES;
        assert!(
            indexer::run(
                &roots,
                chunk_size,
                max_file_bytes,
                indexer::IndexOptions::default(),
                &database,
                &embedder,
                &mut |_| {}
            )
            .is_success()
        );
        assert_eq!(
            *embedder.chunks.borrow(),
            [format!("title: Корінь 🙂 | text: {body}")]
        );
        assert!(
            indexer::run(
                &roots,
                chunk_size,
                max_file_bytes,
                indexer::IndexOptions::default(),
                &database,
                &embedder,
                &mut |_| {}
            )
            .is_success()
        );
        assert_eq!(embedder.chunks.borrow().len(), 1);

        let hits = search::run("raw query", 5, max_file_bytes, &database, &embedder)?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text.as_deref(), Some(body));
        assert_eq!(hits[0].line_start, 3);
        assert_eq!(hits[0].line_end, 4);
        assert_eq!(hits[0].section_path, Some(vec!["Корінь 🙂".into()]));
        assert_eq!(*embedder.queries.borrow(), ["raw query"]);
        compact::run(&database, &mut |_| {})?;
        assert_eq!(embedder.chunks.borrow().len(), 1);
        assert_eq!(embedder.queries.borrow().len(), 1);

        let digest = embedder.settings.model_digest.clone();
        embedder.settings.model_digest = "b".repeat(64);
        let error = search::run("changed digest", 5, max_file_bytes, &database, &embedder)
            .err()
            .unwrap();
        assert!(error.to_string().contains("digest differs"));
        fs::write(
            &note,
            "New content that must not be embedded with the replacement model.",
        )?;
        let report = indexer::run(
            &roots,
            chunk_size,
            max_file_bytes,
            indexer::IndexOptions::default(),
            &database,
            &embedder,
            &mut |_| {},
        );
        assert!(!report.is_success());
        assert_eq!(
            report.publication,
            crate::index_report::PublicationStatus::NotPublished
        );
        assert!(report.errors[0].message.contains("digest differs"));
        assert_eq!(embedder.chunks.borrow().len(), 1);
        assert_eq!(embedder.queries.borrow().len(), 1);
        let connection = crate::database::open_read_only(&database.sqlite_path())?;
        assert_eq!(
            crate::database::vector_index(&connection)?
                .unwrap()
                .settings
                .model_digest,
            digest
        );

        embedder.settings.model_digest = digest;
        embedder.settings.model = "another-model".into();
        assert!(search::run("incompatible", 5, max_file_bytes, &database, &embedder).is_err());
        assert!(
            !indexer::run(
                &roots,
                chunk_size,
                max_file_bytes,
                indexer::IndexOptions::default(),
                &database,
                &embedder,
                &mut |_| {}
            )
            .is_success()
        );
        assert_eq!(embedder.chunks.borrow().len(), 1);
        assert_eq!(embedder.queries.borrow().len(), 1);
        Ok(())
    }
}
