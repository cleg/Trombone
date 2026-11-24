use std::fs;

use super::{corpus, index, test_embedder};
use crate::database::{self, Result};
use crate::snapshot::{self, utf8_path};

/// Verify indexing through a root symlink stores the real path for later passes.
///
/// Returns success if a pass through the real path reuses that record; propagates
/// fixture, SQLite, or path-resolution errors.
#[cfg(unix)]
#[test]
fn root_symlink_and_real_path_share_one_document() -> Result<()> {
    use std::os::unix::fs::symlink;

    let (directory, database) = corpus()?;
    let notes = directory.path().join("notes");
    let alias = directory.path().join("alias");
    fs::create_dir(&notes)?;
    fs::create_dir(notes.join("nested"))?;
    let note = notes.join("note.md");
    fs::write(&note, "Shared note.")?;
    symlink(&notes, &alias)?;
    let embedder = test_embedder(None, None);
    let first = index(&alias.join("nested/.."), &database, &embedder);
    assert!(first.is_success());
    assert_eq!(first.committed.indexed, 1);
    let calls = embedder.calls.get();
    let connection = database::open_read_only(&database.sqlite_path())?;
    assert_eq!(
        database::document_sources(&connection)?,
        [note.to_str().unwrap()],
    );
    let second = index(&note, &database, &embedder);
    assert!(second.is_success());
    assert_eq!(second.committed.unchanged, 1);
    assert_eq!(second.committed.removed, 0);
    assert_eq!(embedder.calls.get(), calls);
    Ok(())
}

/// Verify that replacing a Markdown source with a directory removes its chunks
/// and orphaned vector while keeping the vector shared with a surviving note.
/// Returns success after checking storage, or propagates fixture and storage errors.
#[test]
fn removes_source_replaced_by_directory() -> Result<()> {
    let (directory, database) = corpus()?;
    let note = directory.path().join("note.md");
    let survivor = directory.path().join("survivor.md");
    fs::write(
        &note,
        "# Shared\n\nShared paragraph.\n\n# Unique\n\nUnique paragraph.\n",
    )?;
    fs::write(&survivor, "# Shared\n\nShared paragraph.\n")?;
    let embedder = test_embedder(None, None);
    assert!(index(directory.path(), &database, &embedder).is_success());
    let reader = database::open_read_only(&database.sqlite_path())?;
    let shared = blake3::hash(b"Shared\n\nShared paragraph.\n");
    let unique = blake3::hash(b"Unique\n\nUnique paragraph.\n");
    let shared_key = database::embedding_key(&reader, &shared)?.unwrap();
    let unique_key = database::embedding_key(&reader, &unique)?.unwrap();
    assert_eq!(database::chunk_sources(&reader, shared_key)?.len(), 2);

    fs::remove_file(&note)?;
    fs::create_dir(&note)?;
    let report = index(directory.path(), &database, &embedder);
    assert!(report.is_success(), "{:?}", report.errors);
    assert_eq!(report.committed.removed, 1);
    assert_eq!(
        database::document_sources(&reader)?,
        [survivor.to_str().unwrap()]
    );
    assert_eq!(database::chunk_sources(&reader, shared_key)?.len(), 1);
    assert!(database::chunk_sources(&reader, unique_key)?.is_empty());
    assert!(database::embedding_key(&reader, &unique)?.is_none());
    let active = database::vector_index(&reader)?.unwrap();
    let published = snapshot::new_index(embedder.settings.dimensions)?;
    published.view(utf8_path(&snapshot::stored_path(&database.path, &active)?)?)?;
    assert!(published.contains(shared_key));
    assert!(!published.contains(unique_key));
    Ok(())
}

/// Verify that replacing a parent directory with a file removes its stored
/// descendants and retains documents outside the selected scope.
/// Returns success after checking storage, or propagates fixture and storage errors.
#[test]
fn removes_source_beneath_parent_replaced_by_file() -> Result<()> {
    let (directory, database) = corpus()?;
    let scope = directory.path().join("scope");
    let parent = scope.join("parent");
    let note = parent.join("note.md");
    let outside = directory.path().join("outside.md");
    fs::create_dir_all(&parent)?;
    fs::write(&note, "Descendant note.")?;
    fs::write(&outside, "Note outside the selected scope.")?;
    let embedder = test_embedder(None, None);
    assert!(index(directory.path(), &database, &embedder).is_success());

    fs::remove_dir_all(&parent)?;
    fs::write(&parent, "Replacement file.")?;
    let report = index(&scope, &database, &embedder);
    assert!(report.is_success(), "{:?}", report.errors);
    assert_eq!(report.committed.removed, 1);
    let reader = database::open_read_only(&database.sqlite_path())?;
    assert_eq!(
        database::document_sources(&reader)?,
        [outside.to_str().unwrap()]
    );
    assert_eq!(database::live_vector_count(&reader)?, 1);
    Ok(())
}

/// Verify that replacing a source or its parent with a descendant symlink
/// removes the old identity while indexing the target at its real path.
/// Returns success after checking storage, or propagates fixture and storage errors.
#[cfg(unix)]
#[test]
fn removes_sources_replaced_by_descendant_symlinks() -> Result<()> {
    use std::os::unix::fs::symlink;

    let (directory, database) = corpus()?;
    let parent = directory.path().join("parent");
    let note = directory.path().join("note.md");
    let target_dir = directory.path().join("target");
    let target = target_dir.join("note.md");
    fs::create_dir(&parent)?;
    fs::write(parent.join("note.md"), "Former descendant.")?;
    fs::write(&note, "Former file.")?;
    let embedder = test_embedder(None, None);
    assert!(index(directory.path(), &database, &embedder).is_success());

    fs::remove_dir_all(&parent)?;
    fs::remove_file(&note)?;
    fs::create_dir(&target_dir)?;
    fs::write(&target, "New target.")?;
    symlink(&target_dir, &parent)?;
    symlink(&target, &note)?;
    let report = index(directory.path(), &database, &embedder);
    assert!(report.is_success(), "{:?}", report.errors);
    assert_eq!(report.files_seen, 1);
    assert_eq!(report.committed.removed, 2);
    assert_eq!(report.committed.indexed, 1);
    let reader = database::open_read_only(&database.sqlite_path())?;
    assert_eq!(
        database::document_sources(&reader)?,
        [target.to_str().unwrap()]
    );
    assert_eq!(database::live_vector_count(&reader)?, 1);
    Ok(())
}
