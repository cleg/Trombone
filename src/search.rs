use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::config::DatabaseConfig;
use crate::database::{self, ChunkSource, Result};
use crate::embeddings::{Embedder, EmbeddingSettings};
use crate::snapshot;

#[derive(Serialize)]
pub(crate) struct SearchHit {
    pub(crate) path: PathBuf,
    pub(crate) line_start: usize,
    pub(crate) line_end: usize,
    pub(crate) section_path: Option<Vec<String>>,
    #[serde(rename = "cosine_distance")]
    pub(crate) distance: f32,
    pub(crate) text: String,
}

/// Ranked vector candidate with all source metadata owned independently of SQLite.
struct SearchCandidate {
    distance: f32,
    sources: Vec<ChunkSource>,
}

/// Owned candidates from one round and the size of the vector snapshot they came from.
struct CandidateBatch {
    candidates: Vec<SearchCandidate>,
    index_size: usize,
}

/// One source version observed by this query, with its hash computed once.
struct CachedSource {
    text: String,
    content_hash: String,
}

/// Query-local source snapshots; None entries also cache unreadable or missing files.
#[derive(Default)]
struct SourceCache {
    files: HashMap<PathBuf, Option<CachedSource>>,
}

impl SourceCache {
    /// Read and hash `path` on its first access, then reuse that observation.
    ///
    /// Returns the cached source text and hash, or None after any read or UTF-8
    /// decoding failure. Later file changes are observed only by a fresh cache.
    fn read(&mut self, path: &Path) -> Option<&CachedSource> {
        self.files
            .entry(path.to_path_buf())
            .or_insert_with(|| {
                fs::read_to_string(path).ok().map(|text| CachedSource {
                    content_hash: blake3::hash(text.as_bytes()).to_hex().to_string(),
                    text,
                })
            })
            .as_ref()
    }
}

/// Search the active vector snapshot and resolve current source excerpts.
///
/// - `query` is the search text passed to the embedder's query formatter.
/// - `limit` is the maximum result count; zero returns immediately.
/// - `database` locates SQLite and its vector snapshots.
/// - `embedder` supplies compatible settings and the query embedding.
///
/// Returns ranked hits whose source contents match their indexed hashes when
/// first read during this query. All excerpts from one path reuse the same text
/// and hash, including across retries. Search starts with `limit` candidates and
/// doubles the request if needed, up to four times `limit` or the index size.
/// Each round reads vectors and metadata from one snapshot and releases its
/// transaction before reading Markdown. Returned hits come from the final round.
///
/// # Errors
///
/// Returns an error for incompatible settings, embedding failures, invalid stored
/// metadata, or SQLite and snapshot access failures.
pub(crate) fn run(
    query: String,
    limit: usize,
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

    let embedding = embedder.embed_query(&query)?;
    let max_candidates = limit.saturating_mul(4);
    let mut candidate_count = limit;
    let mut cache = SourceCache::default();
    loop {
        let batch = collect_candidates(
            &mut connection,
            database,
            settings,
            &embedding,
            candidate_count,
        )?;
        // Approximate search can reorder results at a higher limit, and publication
        // can replace the snapshot between rounds. Resolve in the new ranking.
        let hits = resolve_candidates(batch.candidates, limit, &mut cache)?;
        let next_count = candidate_count
            .saturating_mul(2)
            .min(max_candidates)
            .min(batch.index_size);
        if hits.len() == limit || next_count <= candidate_count {
            return Ok(hits);
        }
        candidate_count = next_count;
    }
}

/// Collect ranked candidate metadata from one consistent SQLite/vector snapshot.
///
/// `connection` supplies read transactions, `database` locates snapshot files,
/// `settings` defines the expected embedding space, `embedding` is the query
/// vector, and `candidate_count` is the requested number of vector matches.
/// Returns owned candidates and snapshot size after releasing the vector mapping
/// and read transaction, or an empty batch when no snapshot is active.
/// Retries once if a concurrent publication removes the selected snapshot before opening it.
///
/// # Errors
///
/// Returns metadata, compatibility, SQLite, or vector snapshot/search errors.
fn collect_candidates(
    connection: &mut rusqlite::Connection,
    database: &DatabaseConfig,
    settings: &EmbeddingSettings,
    embedding: &[f32],
    candidate_count: usize,
) -> Result<CandidateBatch> {
    let directory = &database.path;
    let dimensions = settings.dimensions;
    let mut retry_missing = true;
    loop {
        // The file name and chunk metadata must come from the same SQLite snapshot.
        let transaction = connection.transaction()?;
        let Some(active) = database::vector_index(&transaction)? else {
            return Ok(CandidateBatch {
                candidates: Vec::new(),
                index_size: 0,
            });
        };
        let path = snapshot::active_path(directory, &active, settings)?;
        let index = snapshot::new_index(dimensions)?;
        if let Err(error) = index.view(snapshot::utf8_path(&path)?) {
            if retry_missing
                && path
                    .metadata()
                    .is_err_and(|error| error.kind() == io::ErrorKind::NotFound)
            {
                // Indexing may have published a new file and removed this one before view.
                retry_missing = false;
                continue;
            }
            return Err(io::Error::other(format!(
                "Cannot open active index {}: {error}",
                path.display()
            ))
            .into());
        }
        snapshot::validate(&index, dimensions)?;

        let index_size = index.size();
        let candidates = if index_size == 0 {
            Vec::new()
        } else {
            let matches = index.search(embedding, candidate_count.min(index_size))?;
            matches
                .keys
                .into_iter()
                .zip(matches.distances)
                .map(|(key, distance)| {
                    Ok(SearchCandidate {
                        distance,
                        sources: database::chunk_sources(&transaction, key)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?
        };
        drop(index);
        transaction.commit()?;
        return Ok(CandidateBatch {
            candidates,
            index_size,
        });
    }
}

/// Resolve owned `candidates` into at most `limit` ranked, current source excerpts.
///
/// Returns one hit per candidate, trying alternate sources in order. This phase
/// reads and hashes files through the shared query `cache` and holds no SQLite transaction.
///
/// # Errors
///
/// Returns errors for invalid stored ranges or malformed heading metadata.
fn resolve_candidates(
    candidates: Vec<SearchCandidate>,
    limit: usize,
    cache: &mut SourceCache,
) -> Result<Vec<SearchHit>> {
    let mut hits = Vec::new();
    for candidate in candidates {
        if hits.len() == limit {
            break;
        }
        for source in candidate.sources {
            if let Some(hit) = read_hit(source, candidate.distance, cache)? {
                hits.push(hit);
                break;
            }
        }
    }
    Ok(hits)
}

/// Resolve `source` at its typed location using `cache` and attach vector `distance`.
///
/// Returns a hit only if the source version first observed by this query has its
/// indexed content hash. Missing, unreadable, and already-edited files return None;
/// subsequent changes do not affect other excerpts from that path in this query.
///
/// # Errors
///
/// Returns an error if the byte range is invalid for the UTF-8 source or the
/// stored heading path cannot be decoded.
fn read_hit(
    source: ChunkSource,
    distance: f32,
    cache: &mut SourceCache,
) -> Result<Option<SearchHit>> {
    let Some(document) = cache.read(&source.path) else {
        return Ok(None);
    };
    if document.content_hash != source.content_hash {
        return Ok(None);
    }

    let location = source.location;
    let text = document
        .text
        .get(location.byte_start..location.byte_end)
        .ok_or_else(|| {
            io::Error::other(format!(
                "Invalid chunk byte range for {}",
                source.path.display()
            ))
        })?;
    let section_path = source
        .section_path
        .as_deref()
        .map(serde_json::from_str)
        .transpose()?;
    Ok(Some(SearchHit {
        path: source.path,
        line_start: location.line_start,
        line_end: location.line_end,
        section_path,
        distance,
        text: text.to_owned(),
    }))
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

    /// Verify search starts at the requested count, expands past stale notes,
    /// and stops at the candidate budget even if a valid note lies beyond it.
    /// Indexing and search borrow the same ranking fixture.
    /// Returns success for all three outcomes or propagates fixture and search errors.
    #[test]
    fn search_expands_only_as_needed_within_candidate_budget() -> Result<()> {
        use std::num::NonZeroUsize;

        let directory = tempfile::tempdir()?;
        let database = DatabaseConfig {
            path: directory.path().to_path_buf(),
        };
        let notes: Vec<_> = (0..5)
            .map(|rank| directory.path().join(format!("{rank}.md")))
            .collect();
        for (rank, note) in notes.iter().enumerate() {
            fs::write(note, rank.to_string())?;
        }
        let embedder = RankedEmbedder(EmbeddingSettings {
            model: "test-model".into(),
            model_digest: "a".repeat(64),
            dimensions: NonZeroUsize::new(3).unwrap(),
            chunk_format: None,
        });
        let report = crate::indexer::run(
            directory.path(),
            NonZeroUsize::new(1_000).unwrap(),
            false,
            &database,
            &embedder,
            &mut |_| {},
        );
        assert!(report.is_success(), "{:?}", report.errors);
        let mut reader = database::open_read_only(&database.sqlite_path())?;
        let batch = collect_candidates(
            &mut reader,
            &database,
            embedder.settings(),
            &[1.0, 0.0, 0.0],
            1,
        )?;
        assert_eq!(batch.index_size, 5);
        assert_eq!(batch.candidates.len(), 1);
        let hits = run("query".into(), 1, &database, &embedder)?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "0");

        fs::write(&notes[0], "Changed.")?;
        fs::remove_file(&notes[1])?;
        let hits = run("query".into(), 1, &database, &embedder)?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "2");

        fs::remove_file(&notes[2])?;
        fs::remove_file(&notes[3])?;
        assert!(run("query".into(), 1, &database, &embedder)?.is_empty());
        let hits = run("query".into(), 2, &database, &embedder)?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "4");
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
            Some((
                &index,
                database::VectorIndex {
                    format_version: database::VECTOR_INDEX_VERSION,
                    file_name: String::new(),
                    model: settings.model.clone(),
                    model_digest: settings.model_digest.clone(),
                    dimensions: 3,
                    chunk_format: None,
                    removed_since_compact: 0,
                },
            )),
            &database,
            None,
            &mut |_| {},
        )?;
        drop(index);

        let mut reader = database::open_read_only(&database.sqlite_path())?;
        let candidates =
            collect_candidates(&mut reader, &database, &settings, &[1.0, 0.0, 0.0], 5)?.candidates;
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].sources.len(), 2);
        assert!(reader.is_autocommit());

        // An exclusive writer would fail immediately if candidate collection retained its lock.
        let transaction = writer.transaction_with_behavior(TransactionBehavior::Exclusive)?;
        let mut active = database::vector_index(&transaction)?.unwrap();
        let previous = snapshot::stored_path(&database.path, &active)?;
        database::delete_document(&transaction, first.to_str().unwrap())?;
        database::delete_document(&transaction, alternate.to_str().unwrap())?;
        let empty = snapshot::new_index(settings.dimensions)?;
        active.removed_since_compact = 0;
        snapshot::commit(
            transaction,
            Some((&empty, active)),
            &database,
            Some(&previous),
            &mut |_| {},
        )?;
        assert!(!previous.exists());
        assert!(database::document_sources(&reader)?.is_empty());

        fs::write(&first, "Edited after candidate collection.\n")?;
        let hits = resolve_candidates(candidates, 5, &mut SourceCache::default())?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, alternate);
        assert_eq!(hits[0].text, text);
        assert_eq!(hits[0].section_path, Some(vec!["Original heading".into()]));
        assert!(hits[0].distance.abs() < 0.0001);
        Ok(())
    }

    /// Build a candidate for `path`, its full `indexed_text`, and the excerpt `location`.
    ///
    /// Returns source metadata with the indexed file hash and no heading path.
    fn chunk_source(path: &Path, indexed_text: &str, location: SourceLocation) -> ChunkSource {
        ChunkSource {
            path: path.to_path_buf(),
            content_hash: blake3::hash(indexed_text.as_bytes()).to_hex().to_string(),
            section_path: None,
            location,
        }
    }

    /// Verify every excerpt in a query uses the first file version and a new query rereads it.
    ///
    /// Returns success for cached Unicode excerpts and stale-file rejection;
    /// propagates filesystem and candidate-resolution errors.
    #[test]
    fn excerpts_share_one_source_version_until_the_next_query() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("note.md");
        let first = "Перший 🙂.\n";
        let second = "Другий рядок.\n";
        let markdown = format!("{first}{second}");
        fs::write(&path, &markdown)?;
        let mut cache = SourceCache::default();
        let first_hit = read_hit(
            chunk_source(
                &path,
                &markdown,
                SourceLocation {
                    byte_start: 0,
                    byte_end: first.len(),
                    line_start: 1,
                    line_end: 1,
                },
            ),
            0.25,
            &mut cache,
        )?
        .unwrap();
        assert_eq!(first_hit.text, first);

        fs::write(&path, "Edited after the first excerpt.\n")?;
        let location = SourceLocation {
            byte_start: first.len(),
            byte_end: markdown.len(),
            line_start: 2,
            line_end: 2,
        };
        let second_hit =
            read_hit(chunk_source(&path, &markdown, location), 0.5, &mut cache)?.unwrap();
        assert_eq!(second_hit.text, second);
        assert_eq!((second_hit.line_start, second_hit.line_end), (2, 2));
        assert!(
            read_hit(
                chunk_source(&path, &markdown, location),
                0.5,
                &mut SourceCache::default(),
            )?
            .is_none()
        );
        Ok(())
    }

    /// Verify failed reads remain cached until a new query observes newly available text.
    ///
    /// Returns success for missing and invalid-UTF-8 sources; propagates fixture and
    /// candidate-resolution errors.
    #[test]
    fn missing_and_unreadable_sources_are_not_retried_in_the_same_query() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let missing = directory.path().join("missing.md");
        let invalid = directory.path().join("invalid.md");
        fs::write(&invalid, [0xff])?;
        let markdown = "Available 🙂.\n";
        let location = SourceLocation {
            byte_start: 0,
            byte_end: markdown.len(),
            line_start: 1,
            line_end: 1,
        };
        let mut cache = SourceCache::default();
        for path in [&missing, &invalid] {
            assert!(read_hit(chunk_source(path, markdown, location), 0.25, &mut cache)?.is_none());
            fs::write(path, markdown)?;
            assert!(read_hit(chunk_source(path, markdown, location), 0.25, &mut cache)?.is_none());
            let fresh_hit = read_hit(
                chunk_source(path, markdown, location),
                0.25,
                &mut SourceCache::default(),
            )?
            .unwrap();
            assert_eq!(fresh_hit.text, markdown);
        }
        Ok(())
    }

    /// Verify a cached stale path does not hide another path with matching indexed content.
    ///
    /// Returns success for stale-source caching and valid duplicate fallback;
    /// propagates fixture and candidate-resolution errors.
    #[test]
    fn stale_source_does_not_poison_other_paths_with_the_same_indexed_hash() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let stale = directory.path().join("stale.md");
        let valid = directory.path().join("valid.md");
        let markdown = "Shared indexed text.\n";
        fs::write(&stale, "Changed text.\n")?;
        fs::write(&valid, markdown)?;
        let location = SourceLocation {
            byte_start: 0,
            byte_end: markdown.len(),
            line_start: 1,
            line_end: 1,
        };
        let mut cache = SourceCache::default();
        assert!(read_hit(chunk_source(&stale, markdown, location), 0.25, &mut cache)?.is_none());
        fs::write(&stale, markdown)?;
        assert!(read_hit(chunk_source(&stale, markdown, location), 0.25, &mut cache)?.is_none());
        let hit = read_hit(chunk_source(&valid, markdown, location), 0.25, &mut cache)?.unwrap();
        assert_eq!(hit.path, valid);
        assert_eq!(hit.text, markdown);
        assert!(
            read_hit(
                chunk_source(&stale, markdown, location),
                0.25,
                &mut SourceCache::default(),
            )?
            .is_some()
        );
        Ok(())
    }

    /// Verify cached text preserves exact Unicode byte spans and their line coordinates.
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
        let hit = read_hit(
            ChunkSource {
                path: path.clone(),
                content_hash: blake3::hash(markdown.as_bytes()).to_hex().to_string(),
                section_path: Some(serde_json::to_string(&["Назва 🙂"])?),
                location: SourceLocation {
                    byte_start: prefix.len(),
                    byte_end: prefix.len() + excerpt.len(),
                    line_start: 3,
                    line_end: 4,
                },
            },
            0.25,
            &mut SourceCache::default(),
        )?
        .unwrap();
        assert_eq!(hit.path, path);
        assert_eq!(hit.text, excerpt);
        assert_eq!((hit.line_start, hit.line_end), (3, 4));
        assert_eq!(hit.section_path, Some(vec!["Назва 🙂".into()]));
        assert_eq!(hit.distance, 0.25);
        Ok(())
    }

    /// Verify cached text still rejects invalid UTF-8 boundaries in typed coordinates.
    ///
    /// Returns success on the expected range error; propagates fixture write errors.
    #[test]
    fn typed_location_rejects_a_byte_offset_inside_a_unicode_character() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("note.md");
        let markdown = "🙂text";
        fs::write(&path, markdown)?;
        let result = read_hit(
            ChunkSource {
                path,
                content_hash: blake3::hash(markdown.as_bytes()).to_hex().to_string(),
                section_path: None,
                location: SourceLocation {
                    byte_start: 1,
                    byte_end: 4,
                    line_start: 1,
                    line_end: 1,
                },
            },
            0.25,
            &mut SourceCache::default(),
        );
        assert!(result.is_err());
        Ok(())
    }
}
