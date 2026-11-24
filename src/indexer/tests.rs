mod paths;

use std::cell::Cell;
use std::fs;
use std::io;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use super::run;
use crate::compact;
use crate::config::DatabaseConfig;
use crate::database::{self, Result};
use crate::embeddings::{Embedder, EmbeddingSettings};
use crate::index_report::{IndexCounts, IndexReport, PublicationStatus};
use crate::progress::{IndexEvent, ProgressEvent};
use crate::snapshot::{self, utf8_path};

/// Deterministic embeddings with optional failure and source disappearance injection.
pub(super) struct TestEmbedder {
    settings: EmbeddingSettings,
    calls: Cell<usize>,
    fail_on: Option<usize>,
    delete_path: Option<PathBuf>,
}

impl Embedder for TestEmbedder {
    /// Return this test provider's stable embedding identity.
    fn settings(&self) -> &EmbeddingSettings {
        &self.settings
    }

    /// Ignore model `_input`, injecting the configured test event.
    ///
    /// Returns a fixed vector, or an injected generation or file-removal error.
    fn embed_chunk(&self, _input: String) -> Result<Vec<f32>> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if self.fail_on == Some(call) {
            return Err(io::Error::other("injected embedding failure").into());
        }
        if call == 1
            && let Some(path) = &self.delete_path
        {
            fs::remove_file(path)?;
        }
        Ok(vec![1.0, 0.0, 0.0])
    }

    /// Reject `_query` because report tests exercise indexing only.
    ///
    /// Always returns an unsupported-operation error.
    fn embed_query(&self, _query: &str) -> Result<Vec<f32>> {
        Ok(vec![1.0, 0.0, 0.0])
    }
}

/// Create a provider that fails on `fail_on` or removes `delete_path` on its first call.
///
/// Returns a deterministic provider with a stable digest and no service dependency.
pub(super) fn test_embedder(fail_on: Option<usize>, delete_path: Option<PathBuf>) -> TestEmbedder {
    TestEmbedder {
        settings: EmbeddingSettings {
            model: "test-model".into(),
            model_digest: "a".repeat(64),
            dimensions: NonZeroUsize::new(3).unwrap(),
            chunk_format: None,
        },
        calls: Cell::new(0),
        fail_on,
        delete_path,
    }
}

/// Create temporary corpus storage under the canonical system temporary directory.
///
/// Returns the directory guard and configuration, or a path-resolution or
/// temporary-directory creation error.
pub(super) fn corpus() -> Result<(tempfile::TempDir, DatabaseConfig)> {
    let directory = tempfile::tempdir_in(fs::canonicalize(std::env::temp_dir())?)?;
    let database = DatabaseConfig {
        path: directory.path().to_path_buf(),
    };
    Ok((directory, database))
}

/// Index `root` using `database` and the borrowed `embedder`.
///
/// Returns the complete report with default test chunk size and automatic compaction.
fn index(root: &Path, database: &DatabaseConfig, embedder: &TestEmbedder) -> IndexReport {
    run(
        root,
        NonZeroUsize::new(1_000).unwrap(),
        false,
        database,
        embedder,
        &mut |_| {},
    )
}

/// Verify a second writer fails before scanning while the published index is searchable.
///
/// Returns success after checking contention, independent databases, and
/// lock release, or propagates fixture, indexing, or search errors.
#[test]
fn one_writer_per_database_allows_parallel_search() -> Result<()> {
    let (directory, database) = corpus()?;
    let (_other_directory, other_database) = corpus()?;
    let note = directory.path().join("note.md");
    let text = "Published content.";
    fs::write(&note, text)?;
    let embedder = test_embedder(None, None);
    assert!(index(&note, &database, &embedder).is_success());

    let mut checked_contention = false;
    let report = run(
        &note,
        NonZeroUsize::new(1_000).unwrap(),
        false,
        &database,
        &embedder,
        &mut |event| {
            if !matches!(
                event,
                ProgressEvent::Index {
                    event: IndexEvent::Scanning,
                    ..
                }
            ) {
                return;
            }
            checked_contention = true;
            let calls = embedder.calls.get();
            let competing = run(
                &note,
                NonZeroUsize::new(1_000).unwrap(),
                false,
                &database,
                &embedder,
                &mut |_| {},
            );
            assert_eq!(competing.publication, PublicationStatus::NotPublished);
            assert_eq!(competing.files_seen, 0);
            assert_eq!(embedder.calls.get(), calls);
            assert_eq!(competing.errors.len(), 1);
            assert!(
                competing.errors[0]
                    .message
                    .contains("tb index or tb compact")
            );

            let error = compact::run(&database, &mut |_| {})
                .err()
                .expect("same database is busy");
            assert_eq!(
                error.downcast_ref::<io::Error>().map(io::Error::kind),
                Some(io::ErrorKind::WouldBlock),
            );
            assert!(error.to_string().contains("tb index or tb compact"));
            assert!(
                compact::run(&other_database, &mut |_| {})
                    .expect("a separate database is available for writing")
                    .is_none()
            );

            let hits = crate::search::run("query".into(), 1, &database, &embedder)
                .expect("the published index remains searchable during indexing");
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].text, text);
        },
    );
    assert!(report.is_success(), "{:?}", report.errors);
    assert!(checked_contention);
    assert!(compact::run(&database, &mut |_| {})?.is_some());
    Ok(())
}

/// Verify edited and repeated chunks retain shared keys until their last reference disappears.
/// Returns success for reuse, search, and cleanup, or a fixture or storage error.
#[test]
fn prepared_input_hash_reuses_vectors_across_edits_and_files() -> Result<()> {
    let (directory, database) = corpus()?;
    let a = directory.path().join("a.md");
    let b = directory.path().join("b.md");
    let shared_hash = blake3::hash(b"Common\n\nShared paragraph.\n");
    let embedder = test_embedder(None, None);
    fs::write(&a, "# Common\n\nShared paragraph.\n")?;
    assert!(index(&a, &database, &embedder).is_success());
    let reader = database::open_read_only(&database.sqlite_path())?;
    let key = database::embedding_key(&reader, &shared_hash)?.unwrap();
    let original = database::vector_index(&reader)?.unwrap().file_name;
    assert_eq!(embedder.calls.get(), 1);

    // Moving the same excerpt down a line changes coordinates, but not model input.
    fs::write(&a, "\n# Common\n\nShared paragraph.\n")?;
    let report = index(&a, &database, &embedder);
    assert!(report.is_success());
    assert_eq!(embedder.calls.get(), 1);
    assert_eq!(
        database::vector_index(&reader)?.unwrap().file_name,
        original
    );
    let sources = database::chunk_sources(&reader, key)?;
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].location.line_start, 4);

    // The copy has extra content, so reuse must come from the chunk, not the file hash.
    fs::write(
        &b,
        "# Common\n\nShared paragraph.\n\n# Other\n\nA second paragraph.\n",
    )?;
    assert!(index(&b, &database, &embedder).is_success());
    assert_eq!(embedder.calls.get(), 2);
    assert_eq!(database::chunk_sources(&reader, key)?.len(), 2);
    assert_eq!(database::live_vector_count(&reader)?, 2);

    let report = compact::run(&database, &mut |_| {})?.unwrap();
    assert_eq!(report.retained_vectors, 2);
    assert_eq!(database::embedding_key(&reader, &shared_hash)?, Some(key));
    fs::remove_file(&a)?;
    assert!(index(directory.path(), &database, &embedder).is_success());
    assert_eq!(database::embedding_key(&reader, &shared_hash)?, Some(key));
    let hits = crate::search::run("query".into(), 5, &database, &embedder)?;
    assert_eq!(
        hits.iter()
            .filter(|hit| hit.text == "Shared paragraph.\n")
            .count(),
        1
    );

    // Changing the heading changes the actual model input even with identical body text.
    fs::write(&b, "# Renamed\n\nShared paragraph.\n")?;
    assert!(index(&b, &database, &embedder).is_success());
    assert_eq!(embedder.calls.get(), 3);
    assert!(database::embedding_key(&reader, &shared_hash)?.is_none());
    let active = database::vector_index(&reader)?.unwrap();
    let published = snapshot::new_index(embedder.settings.dimensions)?;
    let path = snapshot::stored_path(&database.path, &active)?;
    published.view(utf8_path(&path)?)?;
    assert_eq!(published.size(), 1);
    assert!(!published.contains(key));
    Ok(())
}

/// Verify incompatible chunk settings abort before traversal or embedding, even for an empty scope.
///
/// Returns success for one diagnostic and unchanged storage/embedding calls;
/// propagates fixture and SQLite errors.
#[test]
fn chunk_size_mismatch_aborts_once_before_visiting_files() -> Result<()> {
    let (directory, database) = corpus()?;
    let note = directory.path().join("note.md");
    let empty = directory.path().join("empty");
    fs::write(&note, "A note.")?;
    fs::create_dir(&empty)?;
    let embedder = test_embedder(None, None);
    assert!(index(&note, &database, &embedder).is_success());
    let calls = embedder.calls.get();
    for root in [empty.as_path(), directory.path()] {
        let report = run(
            root,
            NonZeroUsize::new(500).unwrap(),
            false,
            &database,
            &embedder,
            &mut |_| {},
        );
        assert_eq!(report.publication, PublicationStatus::NotPublished);
        assert_eq!(report.files_seen, 0);
        assert_eq!(report.errors.len(), 1);
        assert!(report.errors[0].message.contains("Chunk size differs"));
    }
    assert_eq!(embedder.calls.get(), calls);
    let connection = database::open_read_only(&database.sqlite_path())?;
    assert_eq!(
        database::document_sources(&connection)?,
        [note.to_str().unwrap()],
    );
    Ok(())
}

/// Verify committed counts for deduplication, unchanged files, and shared-content deletion.
///
/// Returns success for the expected reports; propagates fixture filesystem errors.
#[test]
fn report_distinguishes_new_reused_unchanged_and_removed_documents() -> Result<()> {
    let (directory, database) = corpus()?;
    let a = directory.path().join("a.md");
    let b = directory.path().join("b.md");
    fs::write(&a, "Shared body.")?;
    fs::write(&b, "Shared body.")?;
    let embedder = test_embedder(None, None);
    let first = index(directory.path(), &database, &embedder);
    assert!(first.is_success());
    assert_eq!(first.files_seen, 2);
    assert_eq!(
        first.committed,
        IndexCounts {
            indexed: 1,
            reused: 1,
            ..IndexCounts::default()
        }
    );
    assert_eq!(first.staged, first.committed);
    assert_eq!(
        first.publication,
        PublicationStatus::Published {
            snapshot_updated: true,
            compacted: false
        }
    );

    let unchanged = index(directory.path(), &database, &embedder);
    assert!(unchanged.is_success());
    assert_eq!(unchanged.committed.unchanged, 2);
    assert_eq!(
        unchanged.publication,
        PublicationStatus::Published {
            snapshot_updated: false,
            compacted: false
        }
    );
    fs::remove_file(a)?;
    let shared = index(directory.path(), &database, &embedder);
    assert!(shared.is_success());
    assert_eq!(
        shared.committed,
        IndexCounts {
            unchanged: 1,
            removed: 1,
            ..IndexCounts::default()
        }
    );
    assert_eq!(
        shared.publication,
        PublicationStatus::Published {
            snapshot_updated: false,
            compacted: false
        }
    );
    fs::remove_file(b)?;
    let empty = index(directory.path(), &database, &embedder);
    assert!(empty.is_success());
    assert_eq!(empty.files_seen, 0);
    assert_eq!(empty.committed.removed, 1);
    assert_eq!(
        empty.publication,
        PublicationStatus::Published {
            snapshot_updated: true,
            compacted: true
        }
    );
    Ok(())
}

/// Verify recoverable file failures publish good documents and count each issue once.
///
/// Returns success on the partial report and stored documents; propagates fixture errors.
#[test]
fn report_retains_errors_after_partial_publication() -> Result<()> {
    let (directory, database) = corpus()?;
    let good = directory.path().join("good.md");
    let bad = directory.path().join("bad.md");
    fs::write(&good, "Valid body.")?;
    fs::write(&bad, [0xff])?;
    let report = index(directory.path(), &database, &test_embedder(None, None));
    assert!(!report.is_success());
    assert!(matches!(
        report.publication,
        PublicationStatus::Published { .. }
    ));
    assert_eq!(report.files_seen, 2);
    assert_eq!(report.committed.indexed, 1);
    assert_eq!(report.staged, report.committed);
    assert_eq!(report.errors.len(), 1);
    assert_eq!(report.errors[0].path.as_deref(), Some(bad.as_path()));
    let connection = database::open_read_only(&database.sqlite_path())?;
    assert_eq!(
        database::document_sources(&connection)?,
        [good.to_str().unwrap()]
    );
    Ok(())
}

/// Verify an embedding failure preserves the affected file and the pass continues.
/// Returns success if both other updates and the confirmed deletion are committed.
/// Propagates fixture and SQLite errors.
#[test]
fn embedding_failure_preserves_the_old_document_and_continues() -> Result<()> {
    let (directory, database) = corpus()?;
    let notes = ["a.md", "b.md", "c.md"];
    for name in notes {
        fs::write(directory.path().join(name), name)?;
    }
    let missing = directory.path().join("missing.md");
    fs::write(&missing, "A source that will disappear.")?;
    assert!(index(directory.path(), &database, &test_embedder(None, None)).is_success());

    fs::remove_file(&missing)?;
    for name in notes {
        fs::write(directory.path().join(name), format!("Updated {name}"))?;
    }
    let report = index(directory.path(), &database, &test_embedder(Some(2), None));

    assert!(!report.is_success());
    assert_eq!(
        report.publication,
        PublicationStatus::Published {
            snapshot_updated: true,
            compacted: false,
        },
    );
    assert_eq!(report.files_seen, 3);
    assert_eq!(report.committed.indexed, 2);
    assert_eq!(report.committed.removed, 1);
    assert_eq!(report.staged, report.committed);
    assert_eq!(report.errors.len(), 1);
    let issue = &report.errors[0];
    assert!(issue.message.contains("injected embedding failure"));

    let connection = database::open_read_only(&database.sqlite_path())?;
    let mut updated = 0;
    for name in notes {
        let path = directory.path().join(name);
        let hash = database::document_hash(&connection, path.to_str().unwrap())?.unwrap();
        let old_hash = blake3::hash(name.as_bytes()).to_hex();
        if issue.path.as_deref() == Some(path.as_path()) {
            assert_eq!(hash, old_hash.as_str());
            continue;
        }
        let new_hash = blake3::hash(format!("Updated {name}").as_bytes()).to_hex();
        assert_eq!(hash, new_hash.as_str());
        updated += 1;
    }
    assert_eq!(updated, 2);
    assert!(database::document_hash(&connection, missing.to_str().unwrap())?.is_none());
    Ok(())
}

/// Verify a publication failure after successful embedding reports an issue and zero committed work.
///
/// Returns success on rollback and snapshot cleanup; propagates fixture errors.
#[test]
fn publication_failure_returns_an_unpublished_report() -> Result<()> {
    let (directory, database) = corpus()?;
    fs::write(directory.path().join("note.md"), "A body.")?;
    let connection = database::open(&database.sqlite_path())?;
    connection.execute_batch(
        "CREATE TRIGGER fail_publication BEFORE INSERT ON vector_index
         BEGIN SELECT RAISE(ABORT, 'injected publication failure'); END;",
    )?;
    let embedder = test_embedder(None, None);
    let mut events = Vec::new();
    let report = run(
        directory.path(),
        NonZeroUsize::new(1_000).unwrap(),
        false,
        &database,
        &embedder,
        &mut |event| match event {
            ProgressEvent::Index {
                event: IndexEvent::Staged { .. },
                staged,
                seen,
                ..
            } => {
                assert_eq!(seen, 1);
                assert_eq!(staged.indexed, 1);
                events.push("staged");
            }
            ProgressEvent::SavingSnapshot => events.push("saving"),
            ProgressEvent::Committing => events.push("committing"),
            ProgressEvent::Index {
                event: IndexEvent::Issue(issue),
                errors,
                ..
            } => {
                assert_eq!(errors, 1);
                assert!(issue.message.contains("injected publication failure"));
                events.push("issue");
            }
            _ => {}
        },
    );
    assert_eq!(events, ["staged", "saving", "issue"]);
    assert_eq!(report.publication, PublicationStatus::NotPublished);
    assert!(!report.is_success());
    assert_eq!(report.staged.indexed, 1);
    assert_eq!(report.committed, IndexCounts::default());
    assert_eq!(report.errors.len(), 1);
    assert!(
        report.errors[0]
            .message
            .contains("injected publication failure")
    );
    assert!(database::document_sources(&connection)?.is_empty());
    assert!(database::vector_index(&connection)?.is_none());
    assert!(!fs::read_dir(directory.path())?.any(|entry| {
        entry
            .unwrap()
            .path()
            .extension()
            .is_some_and(|extension| extension == "usearch")
    }));
    Ok(())
}

/// Verify vanished prepared files contribute only to removal counts after publication.
///
/// Returns success for reconciled counts; propagates fixture errors.
#[test]
fn vanished_prepared_file_is_not_counted_as_indexed() -> Result<()> {
    let (directory, database) = corpus()?;
    let note = directory.path().join("vanishing.md");
    fs::write(&note, "A vanishing body.")?;
    let report = index(
        directory.path(),
        &database,
        &test_embedder(None, Some(note)),
    );
    assert!(report.is_success());
    assert_eq!(report.files_seen, 1);
    assert_eq!(
        report.committed,
        IndexCounts {
            removed: 1,
            ..IndexCounts::default()
        }
    );
    assert_eq!(report.staged, report.committed);
    assert_eq!(
        report.publication,
        PublicationStatus::Published {
            snapshot_updated: false,
            compacted: false,
        }
    );
    let connection = database::open_read_only(&database.sqlite_path())?;
    assert!(database::vector_index(&connection)?.is_none());
    assert!(database::embedding_key(&connection, &blake3::hash(b"A vanishing body."))?.is_none());
    Ok(())
}

/// Verify traversal and snapshot-load failures each return one diagnostic without publication.
///
/// Returns success on the aborted reports; propagates fixture and metadata errors.
#[test]
fn traversal_and_snapshot_failures_return_reports_without_duplicate_errors() -> Result<()> {
    let (directory, database) = corpus()?;
    let embedder = test_embedder(None, None);
    let missing = directory.path().join("missing.md");
    let report = index(&missing, &database, &embedder);
    assert_eq!(report.publication, PublicationStatus::NotPublished);
    assert_eq!(report.errors.len(), 1);
    assert_eq!(report.errors[0].path.as_deref(), Some(missing.as_path()));
    assert_eq!(report.files_seen, 0);
    let note = directory.path().join("note.md");
    fs::write(&note, "A body.")?;
    assert!(index(&note, &database, &embedder).is_success());
    let connection = database::open_read_only(&database.sqlite_path())?;
    let active = database::vector_index(&connection)?.unwrap();
    let snapshot = database.path.join(&active.file_name);
    fs::remove_file(&snapshot)?;
    let report = index(&note, &database, &embedder);
    assert_eq!(report.publication, PublicationStatus::NotPublished);
    assert_eq!(report.errors.len(), 1);
    assert_eq!(report.errors[0].path.as_deref(), Some(snapshot.as_path()));
    assert_eq!(report.staged, IndexCounts::default());
    assert_eq!(report.committed, IndexCounts::default());
    Ok(())
}
