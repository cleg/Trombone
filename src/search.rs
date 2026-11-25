use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::config::DatabaseConfig;
use crate::database::{self, ChunkSource};
use crate::embeddings::{Embedder, EmbeddingSettings};
use crate::error::Result;
use crate::processor;
use crate::snapshot::{self, OpenMode};

#[derive(Serialize)]
pub(crate) struct SearchHit {
    pub(crate) sources: Vec<SearchSource>,
    pub(crate) line_start: usize,
    pub(crate) line_end: usize,
    pub(crate) section_path: Option<Vec<String>>,
    #[serde(rename = "cosine_distance")]
    pub(crate) distance: f32,
    pub(crate) text: Option<String>,
}

/// An indexed source path and its availability at query time.
#[derive(Serialize)]
pub(crate) struct SearchSource {
    pub(crate) path: PathBuf,
    pub(crate) line_start: usize,
    pub(crate) line_end: usize,
    pub(crate) status: SourceStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SourceStatus {
    Available,
    Missing,
    TooLarge,
    ExcerptUnavailable,
}

/// Ranked chunk and its source metadata, owned independently of SQLite.
struct SearchCandidate {
    distance: f32,
    chunk: ChunkSource,
}

/// Search the active vector snapshot and resolve current source excerpts.
///
/// - `query` is the search text passed to the embedder's query formatter.
/// - `limit` is the maximum chunk count; zero returns immediately.
/// - `max_file_bytes` is the maximum source file size in bytes read for excerpts.
/// - `database` locates SQLite and its vector snapshots.
/// - `embedder` supplies compatible settings and the query embedding.
///
/// Returns ranked chunks with their indexed source paths, even when an excerpt
/// cannot be read. Duplicate files share a hit, and distinct chunks can share
/// a vector. After releasing the SQLite snapshot, reads each path at most once
/// and retains at most one excerpt per hit.
///
/// # Errors
///
/// Returns an error for incompatible settings, embedding failures, invalid stored
/// metadata, or SQLite and snapshot access failures.
pub(crate) fn run(
    query: &str,
    limit: usize,
    max_file_bytes: NonZeroU64,
    database: &DatabaseConfig,
    embedder: &dyn Embedder,
) -> Result<Vec<SearchHit>> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    let mut connection = database::open_read_only(&database.sqlite_path())?;
    let directory = &database.path;
    let settings = embedder.settings();
    {
        // Check compatibility before embedding the query, then release the read lock.
        let transaction = connection.transaction()?;
        let Some(active) = database::vector_index(&transaction)? else {
            return Ok(Vec::new());
        };
        snapshot::active_path(directory, &active, settings)?;
        transaction.commit()?;
    }

    let embedding = embedder.embed_query(query)?;
    let candidates = collect_candidates(&mut connection, database, settings, &embedding, limit)?;
    resolve_hits(&candidates, |path, needs_text| {
        read_source(path, needs_text, max_file_bytes)
    })
}

/// Maximum retries when a concurrent publication replaces the active snapshot before `view`.
///
/// Two retries (three attempts total) cover back-to-back snapshot swaps such as
/// `tb index` followed immediately by automatic compaction.
const MISSING_SNAPSHOT_RETRIES: usize = 2;

/// Collect ranked candidate metadata from one consistent SQLite/vector snapshot.
///
/// `connection` supplies read transactions, `database` locates snapshot files,
/// `settings` defines the expected embedding space, `embedding` is the query
/// vector, and `limit` bounds the number of chunks collected across vector matches.
/// Returns owned candidates after releasing the mapping and read transaction,
/// or an empty list when no snapshot is active.
/// Retries up to [`MISSING_SNAPSHOT_RETRIES`] times if concurrent publications
/// remove the selected snapshot before opening it.
///
/// # Errors
///
/// Returns metadata, compatibility, SQLite, or vector snapshot/search errors.
fn collect_candidates(
    connection: &mut rusqlite::Connection,
    database: &DatabaseConfig,
    settings: &EmbeddingSettings,
    embedding: &[f32],
    limit: usize,
) -> Result<Vec<SearchCandidate>> {
    let directory = &database.path;
    let dimensions = settings.dimensions;
    let mut remaining_retries = MISSING_SNAPSHOT_RETRIES;
    loop {
        // The file name and chunk metadata must come from the same SQLite snapshot.
        let transaction = connection.transaction()?;
        let Some(active) = database::vector_index(&transaction)? else {
            return Ok(Vec::new());
        };
        let path = snapshot::active_path(directory, &active, settings)?;
        let index = match snapshot::open(&path, dimensions, OpenMode::View) {
            Ok(index) => index,
            Err(error) if remaining_retries > 0 && error.is_missing() => {
                // Indexing or automatic compaction may have published a new file and removed this one before view.
                remaining_retries -= 1;
                continue;
            }
            Err(error) => return Err(error.into()),
        };

        let mut candidates = Vec::new();
        let vector_count = limit.min(index.size());
        if vector_count > 0 {
            let matches = index.search(embedding, vector_count)?;
            for (key, distance) in matches.keys.into_iter().zip(matches.distances) {
                let remaining = limit - candidates.len();
                let chunks = database::chunk_sources(&transaction, key, remaining)?;
                candidates.extend(
                    chunks
                        .into_iter()
                        .map(|chunk| SearchCandidate { distance, chunk }),
                );
                if candidates.len() == limit {
                    break;
                }
            }
        }
        drop(index);
        transaction.commit()?;
        return Ok(candidates);
    }
}

/// Resolve ranked `candidates` by grouping every source occurrence by path.
///
/// `read` hashes each path once and includes its text if any affected hit still
/// needs an excerpt. Returns hits in candidate order, each with its indexed
/// paths and at most one verified excerpt. Releases each file's text before
/// moving on to the next path.
///
/// # Errors
///
/// Returns an error for an invalid stored byte range. Source
/// read errors are recorded as availability statuses.
fn resolve_hits(
    candidates: &[SearchCandidate],
    mut read: impl FnMut(&Path, bool) -> io::Result<(blake3::Hash, Option<String>)>,
) -> Result<Vec<SearchHit>> {
    let mut by_path = BTreeMap::<&Path, Vec<(usize, usize)>>::new();
    let mut hits = Vec::with_capacity(candidates.len());
    for (hit_index, candidate) in candidates.iter().enumerate() {
        let chunk = &candidate.chunk;
        let location = chunk.location;
        let mut sources = Vec::with_capacity(chunk.paths.len());
        for (source_index, path) in chunk.paths.iter().enumerate() {
            by_path
                .entry(path)
                .or_default()
                .push((hit_index, source_index));
            sources.push(SearchSource {
                path: path.clone(),
                line_start: location.line_start,
                line_end: location.line_end,
                status: SourceStatus::ExcerptUnavailable,
            });
        }
        hits.push(SearchHit {
            sources,
            line_start: location.line_start,
            line_end: location.line_end,
            section_path: chunk.section_path.clone(),
            distance: candidate.distance,
            text: None,
        });
    }

    for (path, occurrences) in by_path {
        let needs_text = occurrences.iter().any(|&(hit, _)| hits[hit].text.is_none());
        let document = read(path, needs_text).map(|(hash, text)| (hash.to_hex(), text));
        for (hit_index, source_index) in occurrences {
            let hit = &mut hits[hit_index];
            let chunk = &candidates[hit_index].chunk;
            hit.sources[source_index].status = match &document {
                Ok((hash, text)) if hash.as_str() == chunk.content_hash => {
                    if hit.text.is_none()
                        && let Some(text) = text
                    {
                        let location = chunk.location;
                        let excerpt = text
                            .get(location.byte_start..location.byte_end)
                            .ok_or_else(|| {
                                io::Error::other(format!(
                                    "Invalid chunk byte range for {}",
                                    path.display()
                                ))
                            })?;
                        hit.text = Some(excerpt.to_owned());
                    }
                    SourceStatus::Available
                }
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
                    ) =>
                {
                    SourceStatus::Missing
                }
                Err(error) if error.kind() == io::ErrorKind::FileTooLarge => SourceStatus::TooLarge,
                _ => SourceStatus::ExcerptUnavailable,
            };
        }
    }
    Ok(hits)
}

/// Hash the file at `path` and return its UTF-8 text when `needs_text` is true.
///
/// Enforces `max_file_bytes` before loading text or hashing so oversized files
/// consistently report [`io::ErrorKind::FileTooLarge`]. Streams the bytes into
/// the hash when only verification is needed. Returns the hash and optional
/// text, or an error on open, read, size limit, or UTF-8 decoding.
fn read_source(
    path: &Path,
    needs_text: bool,
    max_file_bytes: NonZeroU64,
) -> io::Result<(blake3::Hash, Option<String>)> {
    if needs_text {
        let text = processor::read_source_file(path, max_file_bytes)?;
        return Ok((blake3::hash(text.as_bytes()), Some(text)));
    }
    let file = fs::File::open(path)?;
    if file.metadata()?.len() > max_file_bytes.get() {
        return Err(io::Error::new(
            io::ErrorKind::FileTooLarge,
            format!("File exceeds maximum size of {max_file_bytes} bytes"),
        ));
    }
    let mut hash = blake3::Hasher::new();
    hash.update_reader(file)?;
    Ok((hash.finalize(), None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{Attributes, Chunk, DocumentMetadata, SourceLocation};

    /// Test provider whose stored vectors rank notes by the integer in their text.
    struct RankedEmbedder(EmbeddingSettings);

    impl Embedder for RankedEmbedder {
        /// Return the settings shared by fixture indexing and search.
        fn settings(&self) -> &EmbeddingSettings {
            &self.0
        }

        /// Embed numeric prepared `text` as a vector.
        /// Returns a vector or an error if the fixture text is not numeric.
        fn embed_chunk(&self, text: String) -> Result<Vec<f32>> {
            Ok(vec![1.0, text.trim().parse()?, 0.0])
        }

        /// Ignore `_query` and return a vector that ranks smaller fixture numbers first.
        fn embed_query(&self, _query: &str) -> Result<Vec<f32>> {
            Ok(vec![1.0, 0.0, 0.0])
        }
    }

    /// Verify the limit counts indexed chunks, including hits with no readable excerpt.
    ///
    /// Returns success when changed and missing sources retain their ranking and
    /// duplicate files share a hit. Propagates fixture, indexing, or search errors.
    #[test]
    fn search_counts_chunks_with_unavailable_sources() -> Result<()> {
        use std::num::NonZeroUsize;

        let directory = tempfile::tempdir_in(fs::canonicalize(std::env::temp_dir())?)?;
        let database = DatabaseConfig {
            path: directory.path().to_path_buf(),
        };
        let changed = directory.path().join("0.md");
        let missing_copy = directory.path().join("0-copy.md");
        let missing = directory.path().join("1.md");
        let available = directory.path().join("2.md");
        for (path, text) in [
            (&changed, "0"),
            (&missing_copy, "0"),
            (&missing, "1"),
            (&available, "2"),
        ] {
            fs::write(path, text)?;
        }
        let embedder = RankedEmbedder(EmbeddingSettings {
            model: "test-model".into(),
            model_digest: "a".repeat(64),
            dimensions: NonZeroUsize::new(3).unwrap(),
            chunk_format: None,
        });
        let report = crate::indexer::run(
            &[directory.path().to_path_buf()],
            NonZeroUsize::new(1_000).unwrap(),
            crate::config::DEFAULT_MAX_FILE_BYTES,
            false,
            &database,
            &embedder,
            &mut |_| {},
        );
        assert!(report.is_success(), "{:?}", report.errors);
        fs::write(&changed, "Changed.")?;
        fs::remove_file(&missing_copy)?;
        fs::remove_file(&missing)?;

        let hits = run(
            "query",
            2,
            crate::config::DEFAULT_MAX_FILE_BYTES,
            &database,
            &embedder,
        )?;
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].text, None);
        assert_eq!(hits[0].sources.len(), 2);
        assert_eq!(hits[0].sources[0].path, missing_copy);
        assert_eq!(hits[0].sources[0].status, SourceStatus::Missing);
        assert_eq!(hits[0].sources[1].path, changed);
        assert_eq!(hits[0].sources[1].status, SourceStatus::ExcerptUnavailable);
        assert_eq!(hits[1].sources[0].path, missing);
        assert_eq!(hits[1].text, None);
        let hits = run(
            "query",
            3,
            crate::config::DEFAULT_MAX_FILE_BYTES,
            &database,
            &embedder,
        )?;
        assert_eq!(hits[2].text.as_deref(), Some("2"));
        assert_eq!(hits[2].sources[0].status, SourceStatus::Available);
        Ok(())
    }

    /// Verify candidate collection releases its lock and owns all duplicate-source metadata.
    ///
    /// Publishes a replacement before reading any excerpts. Returns success when
    /// the old candidates still resolve through a valid alternate file, or a
    /// fixture, storage, snapshot, or search error.
    #[test]
    fn candidates_survive_publication_without_holding_a_read_transaction() -> Result<()> {
        use rusqlite::TransactionBehavior;
        use std::num::NonZeroUsize;

        let directory = tempfile::tempdir()?;
        let database = DatabaseConfig {
            path: directory.path().to_path_buf(),
        };
        let first = directory.path().join("a.md");
        let alternate = directory.path().join("b.md");
        let text = "Shared indexed text.\n";
        fs::write(&first, text)?;
        fs::write(&alternate, text)?;
        let hash = blake3::hash(text.as_bytes()).to_hex().to_string();
        let settings = EmbeddingSettings {
            model: "test-model".into(),
            model_digest: "a".repeat(64),
            dimensions: NonZeroUsize::new(3).unwrap(),
            chunk_format: None,
        };
        let mut writer = database::open(&database.sqlite_path())?;
        writer.busy_timeout(std::time::Duration::ZERO)?;
        let transaction = writer.transaction()?;
        for path in [&first, &alternate] {
            database::replace_document(
                &transaction,
                path.to_str().unwrap(),
                &hash,
                &DocumentMetadata {
                    source: path.clone(),
                    attributes: Attributes::new(),
                },
            )?;
        }
        let source = chunk_source(
            &first,
            text,
            SourceLocation {
                byte_start: 0,
                byte_end: text.len(),
                line_start: 1,
                line_end: 1,
            },
        );
        database::try_insert_embedding(&transaction, 1, &blake3::hash(text.as_bytes()))?;
        database::insert_chunk(
            &transaction,
            &hash,
            0,
            1,
            &Chunk {
                text: text.into(),
                section_path: Some(vec!["Original heading".into()]),
                location: source.location,
                attributes: Attributes::new(),
            },
        )?;
        let index = snapshot::new_index(settings.dimensions)?;
        index.reserve(1)?;
        index.add(1, &[1.0_f32, 0.0, 0.0])?;
        snapshot::commit(
            transaction,
            Some((&index, &settings)),
            &database,
            None,
            &mut |_| {},
        )?;
        drop(index);

        let mut reader = database::open_read_only(&database.sqlite_path())?;
        let candidates =
            collect_candidates(&mut reader, &database, &settings, &[1.0, 0.0, 0.0], 5)?;
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].chunk.paths.len(), 2);
        assert!(reader.is_autocommit());

        // Publish a replacement while the owned candidates still exist.
        let transaction = writer.transaction_with_behavior(TransactionBehavior::Exclusive)?;
        let active = database::vector_index(&transaction)?.unwrap();
        let previous = snapshot::stored_path(&database.path, &active)?;
        database::delete_document(&transaction, first.to_str().unwrap())?;
        database::delete_document(&transaction, alternate.to_str().unwrap())?;
        let empty = snapshot::new_index(settings.dimensions)?;
        snapshot::commit(
            transaction,
            Some((&empty, &active.settings)),
            &database,
            Some(&previous),
            &mut |_| {},
        )?;
        assert!(!previous.exists());
        assert!(database::document_sources(&reader)?.is_empty());

        fs::write(&first, "Edited after candidate collection.\n")?;
        let hits = resolve_hits(&candidates, |path, needs_text| {
            read_source(path, needs_text, crate::config::DEFAULT_MAX_FILE_BYTES)
        })?;
        let hit = &hits[0];
        assert_eq!(hit.sources[0].status, SourceStatus::ExcerptUnavailable);
        assert_eq!(hit.sources[1].path, alternate);
        assert_eq!(hit.sources[1].status, SourceStatus::Available);
        assert_eq!(hit.text.as_deref(), Some(text));
        assert_eq!(hit.section_path, Some(vec!["Original heading".into()]));
        assert!(hit.distance.abs() < 0.0001);
        Ok(())
    }

    /// Build a candidate for `path`, its full `indexed_text`, and the excerpt `location`.
    ///
    /// Returns source metadata with the indexed file hash and no heading path.
    fn chunk_source(path: &Path, indexed_text: &str, location: SourceLocation) -> ChunkSource {
        ChunkSource {
            paths: vec![path.to_path_buf()],
            content_hash: blake3::hash(indexed_text.as_bytes()).to_hex().to_string(),
            section_path: None,
            location,
        }
    }

    /// Verify that one read resolves multiple chunks and duplicates are only hashed.
    ///
    /// Returns success when every hit preserves its original ranking and excerpt,
    /// and missing or invalid copies keep their indexed ranges. Propagates fixture
    /// and candidate-resolution errors.
    #[test]
    fn reads_each_path_once_and_only_streams_hash_for_extra_copies() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("a.md");
        let copy = directory.path().join("b.md");
        let missing = directory.path().join("c.md");
        let invalid = directory.path().join("d.md");
        let first = "Перший 🙂.\n";
        let second = "Другий рядок.\n";
        let markdown = format!("{first}{second}");
        fs::write(&path, &markdown)?;
        fs::write(&copy, &markdown)?;
        fs::write(&invalid, [0xff])?;
        let candidates: Vec<_> = [(0, first.len(), 1), (first.len(), markdown.len(), 2)]
            .into_iter()
            .map(|(byte_start, byte_end, line)| {
                let mut chunk = chunk_source(
                    &path,
                    &markdown,
                    SourceLocation {
                        byte_start,
                        byte_end,
                        line_start: line,
                        line_end: line,
                    },
                );
                chunk
                    .paths
                    .extend([copy.clone(), missing.clone(), invalid.clone()]);
                SearchCandidate {
                    chunk,
                    distance: line as f32 * 0.25,
                }
            })
            .collect();
        let mut reads = Vec::new();
        let hits = resolve_hits(&candidates, |path, needs_text| {
            reads.push((path.to_path_buf(), needs_text));
            read_source(path, needs_text, crate::config::DEFAULT_MAX_FILE_BYTES)
        })?;
        assert_eq!(
            reads,
            [
                (path, true),
                (copy, false),
                (missing, false),
                (invalid, false)
            ]
        );
        for (index, (hit, excerpt)) in hits.iter().zip([first, second]).enumerate() {
            assert_eq!(hit.text.as_deref(), Some(excerpt));
            assert_eq!(hit.distance, candidates[index].distance);
            assert_eq!(
                hit.sources
                    .iter()
                    .map(|source| source.status)
                    .collect::<Vec<_>>(),
                [
                    SourceStatus::Available,
                    SourceStatus::Available,
                    SourceStatus::Missing,
                    SourceStatus::ExcerptUnavailable,
                ]
            );
            let line = index + 1;
            assert!(
                hit.sources
                    .iter()
                    .all(|source| (source.line_start, source.line_end) == (line, line))
            );
        }
        Ok(())
    }

    /// Verify search preserves exact Unicode byte spans and indexed line coordinates.
    ///
    /// Returns success on the expected hit; propagates filesystem and search errors.
    #[test]
    fn typed_location_extracts_the_original_unicode_excerpt() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("note.md");
        let prefix = "# Назва 🙂\n\n";
        let excerpt = "Перший рядок.\nДругий 🙂.\n";
        let markdown = format!("{prefix}{excerpt}\n# Інший розділ\n\nІнший текст.\n");
        fs::write(&path, &markdown)?;
        let hits = resolve_hits(
            &[SearchCandidate {
                chunk: ChunkSource {
                    paths: vec![path.clone()],
                    content_hash: blake3::hash(markdown.as_bytes()).to_hex().to_string(),
                    section_path: Some(vec!["Назва 🙂".into()]),
                    location: SourceLocation {
                        byte_start: prefix.len(),
                        byte_end: prefix.len() + excerpt.len(),
                        line_start: 3,
                        line_end: 4,
                    },
                },
                distance: 0.25,
            }],
            |path, needs_text| read_source(path, needs_text, crate::config::DEFAULT_MAX_FILE_BYTES),
        )?;
        let hit = &hits[0];
        assert_eq!(hit.sources[0].path, path);
        assert_eq!(hit.text.as_deref(), Some(excerpt));
        assert_eq!((hit.line_start, hit.line_end), (3, 4));
        assert_eq!(hit.section_path, Some(vec!["Назва 🙂".into()]));
        assert_eq!(hit.distance, 0.25);
        Ok(())
    }

    /// Verify search rejects invalid UTF-8 boundaries in stored coordinates.
    ///
    /// Returns success on the expected range error; propagates fixture write errors.
    #[test]
    fn typed_location_rejects_a_byte_offset_inside_a_unicode_character() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("note.md");
        let markdown = "🙂text";
        fs::write(&path, markdown)?;
        let result = resolve_hits(
            &[SearchCandidate {
                chunk: ChunkSource {
                    paths: vec![path],
                    content_hash: blake3::hash(markdown.as_bytes()).to_hex().to_string(),
                    section_path: None,
                    location: SourceLocation {
                        byte_start: 1,
                        byte_end: 4,
                        line_start: 1,
                        line_end: 1,
                    },
                },
                distance: 0.25,
            }],
            |path, needs_text| read_source(path, needs_text, crate::config::DEFAULT_MAX_FILE_BYTES),
        );
        assert!(result.is_err());
        Ok(())
    }

    /// Verify oversized files are skipped with a warning during indexing and marked TooLarge in search.
    ///
    /// Returns success when indexing publishes without errors after skipping an
    /// oversized file and search reports `SourceStatus::TooLarge` for an indexed
    /// file that later exceeds `max_file_bytes`. Propagates fixture or search errors.
    #[test]
    fn oversized_files_warn_on_index_and_report_too_large_on_search() -> Result<()> {
        use std::num::NonZeroUsize;

        let directory = tempfile::tempdir_in(fs::canonicalize(std::env::temp_dir())?)?;
        let database = DatabaseConfig {
            path: directory.path().to_path_buf(),
        };
        let small = directory.path().join("0.md");
        let oversized = directory.path().join("1.md");
        fs::write(&small, "0")?;
        fs::write(&oversized, "0123456789")?;
        let embedder = RankedEmbedder(EmbeddingSettings {
            model: "test-model".into(),
            model_digest: "a".repeat(64),
            dimensions: NonZeroUsize::new(3).unwrap(),
            chunk_format: None,
        });
        let limit = NonZeroU64::new(5).unwrap();
        let mut warned = Vec::new();
        let report = crate::indexer::run(
            &[directory.path().to_path_buf()],
            NonZeroUsize::new(1_000).unwrap(),
            limit,
            false,
            &database,
            &embedder,
            &mut |event| {
                if let crate::progress::ProgressEvent::FileTooLarge { path, .. } = event {
                    warned.push(path.to_path_buf());
                }
            },
        );
        assert!(report.is_success(), "{:?}", report.errors);
        assert_eq!(report.committed.indexed, 1);
        assert_eq!(warned, [oversized]);

        // Grow the indexed file beyond the limit and verify search reports TooLarge.
        fs::write(&small, "0123456789")?;
        let hits = run("query", 5, limit, &database, &embedder)?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, None);
        assert_eq!(hits[0].sources[0].path, small);
        assert_eq!(hits[0].sources[0].status, SourceStatus::TooLarge);
        Ok(())
    }
}
