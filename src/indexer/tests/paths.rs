use std::fs;

use super::{corpus, index, index_roots, test_embedder};
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
    let second = index(&notes, &database, &embedder);
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

/// Verify replacing a descendant directory with a file drops its stored notes,
/// and removing another root drops its notes from the same index.
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
    let survivor = scope.join("survivor.md");
    fs::write(&survivor, "A surviving note.")?;
    let embedder = test_embedder(None, None);
    assert!(index(directory.path(), &database, &embedder).is_success());

    fs::remove_dir_all(&parent)?;
    fs::write(&parent, "Replacement file.")?;
    let report = index(&scope, &database, &embedder);
    assert!(report.is_success(), "{:?}", report.errors);
    assert_eq!(report.committed.removed, 2);
    let reader = database::open_read_only(&database.sqlite_path())?;
    assert_eq!(
        database::document_sources(&reader)?,
        [survivor.to_str().unwrap()]
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

/// Verify overlapping roots share one processing attempt and retain a note
/// reachable from a remaining root after removing the broader root.
///
/// Returns success for the final membership and counts, or a fixture/storage error.
#[test]
fn removing_an_overlapping_root_only_drops_unreachable_notes() -> Result<()> {
    let (directory, database) = corpus()?;
    let outer = directory.path().join("outer");
    let inner = outer.join("inner");
    fs::create_dir_all(&inner)?;
    let shared = inner.join("shared.md");
    fs::write(&shared, "Reachable through both roots.")?;
    fs::write(outer.join("exclusive.md"), "Reachable through one root.")?;
    let embedder = test_embedder(None, None);

    let first = index_roots(&[inner.clone(), outer], &database, &embedder);
    assert!(first.is_success(), "{:?}", first.errors);
    assert_eq!(first.files_seen, 2);
    assert_eq!(first.committed.indexed, 2);
    let second = index(&inner, &database, &embedder);
    assert!(second.is_success(), "{:?}", second.errors);
    assert_eq!(second.committed.unchanged, 1);
    assert_eq!(second.committed.removed, 1);
    let connection = database::open_read_only(&database.sqlite_path())?;
    assert_eq!(
        database::document_sources(&connection)?,
        [shared.to_str().unwrap()]
    );
    Ok(())
}

/// Verify an unavailable root prevents all global deletions while a reachable
/// root can publish changes. Recovery to an empty directory enables deletion.
///
/// Returns success for both passes and database states, or a fixture/storage error.
#[test]
fn unavailable_root_defers_global_deletion_until_it_can_be_opened() -> Result<()> {
    let (directory, database) = corpus()?;
    let retained_root = directory.path().join("retained");
    let removed_root = directory.path().join("removed");
    let missing_root = directory.path().join("missing");
    for root in [&retained_root, &removed_root, &missing_root] {
        fs::create_dir(root)?;
        fs::write(root.join("note.md"), root.to_str().unwrap())?;
    }
    let embedder = test_embedder(None, None);
    let first = index_roots(
        &[retained_root.clone(), removed_root, missing_root.clone()],
        &database,
        &embedder,
    );
    assert!(first.is_success(), "{:?}", first.errors);

    fs::remove_dir_all(&missing_root)?;
    let edited = retained_root.join("note.md");
    fs::write(&edited, "Updated content.")?;
    let roots = [missing_root.clone(), retained_root];
    let partial = index_roots(&roots, &database, &embedder);
    assert!(!partial.is_success());
    assert!(partial.deletions_deferred);
    assert_eq!(partial.errors.len(), 1);
    assert_eq!(
        partial.errors[0].path.as_deref(),
        Some(missing_root.as_path())
    );
    assert_eq!(partial.committed.indexed, 1);
    assert_eq!(partial.committed.removed, 0);
    let connection = database::open_read_only(&database.sqlite_path())?;
    assert_eq!(database::document_sources(&connection)?.len(), 3);

    fs::create_dir(&missing_root)?;
    let complete = index_roots(&roots, &database, &embedder);
    assert!(complete.is_success(), "{:?}", complete.errors);
    assert!(!complete.deletions_deferred);
    assert_eq!(complete.committed.removed, 2);
    assert_eq!(
        database::document_sources(&connection)?,
        [edited.to_str().unwrap()]
    );
    Ok(())
}

/// Verify a root replaced by a regular file cannot publish a pass on its own.
///
/// Returns success when the previous documents and snapshot survive, or a
/// fixture/storage error.
#[test]
fn no_open_root_preserves_the_previous_publication() -> Result<()> {
    let (directory, database) = corpus()?;
    let root = directory.path().join("root");
    fs::create_dir(&root)?;
    let note = root.join("note.md");
    fs::write(&note, "Indexed note.")?;
    let embedder = test_embedder(None, None);
    assert!(index(&root, &database, &embedder).is_success());
    let reader = database::open_read_only(&database.sqlite_path())?;
    let previous = database::vector_index(&reader)?.unwrap().file_name;

    fs::remove_dir_all(&root)?;
    fs::write(&root, "The root is now a regular file.")?;
    let report = index(&root, &database, &embedder);
    assert_eq!(
        report.publication,
        crate::index_report::PublicationStatus::NotPublished
    );
    assert!(report.deletions_deferred);
    assert_eq!(report.errors.len(), 1);
    assert_eq!(
        database::document_sources(&reader)?,
        [note.to_str().unwrap()]
    );
    assert_eq!(
        database::vector_index(&reader)?.unwrap().file_name,
        previous
    );
    Ok(())
}

/// Verify removal and retargeting of links reconcile canonical external files.
///
/// Returns success when a shared path survives through its last remaining route
/// and the retargeted link replaces its old identity, or a fixture/storage error.
#[cfg(unix)]
#[test]
fn removing_and_retargeting_symlinks_reconciles_their_canonical_targets() -> Result<()> {
    use std::os::unix::fs::symlink;

    let (directory, database) = corpus()?;
    let root = directory.path().join("sources");
    let external = directory.path().join("external");
    fs::create_dir(&root)?;
    fs::create_dir(&external)?;
    let original = external.join("original.md");
    let shared = directory.path().join("shared.md");
    let replacement = directory.path().join("replacement.md");
    fs::write(&original, "Original target.")?;
    fs::write(&shared, "Shared target.")?;
    fs::write(&replacement, "Replacement target.")?;
    let link = root.join("directory-link");
    let redundant = root.join("first-shared-link");
    let survivor = root.join("second-shared-link");
    symlink(&external, &link)?;
    symlink(&shared, &redundant)?;
    symlink(&shared, &survivor)?;
    let embedder = test_embedder(None, None);
    let first = index(&root, &database, &embedder);
    assert!(first.is_success(), "{:?}", first.errors);
    assert_eq!(first.files_seen, 2);

    fs::remove_file(&redundant)?;
    fs::remove_file(&link)?;
    symlink(&replacement, &link)?;
    let second = index(&root, &database, &embedder);
    assert!(second.is_success(), "{:?}", second.errors);
    assert_eq!(second.committed.removed, 1);
    assert_eq!(second.committed.unchanged, 1);
    assert_eq!(second.committed.indexed, 1);
    let reader = database::open_read_only(&database.sqlite_path())?;
    assert_eq!(
        database::document_sources(&reader)?,
        [replacement.to_str().unwrap(), shared.to_str().unwrap()]
    );

    fs::remove_file(&survivor)?;
    assert_eq!(index(&root, &database, &embedder).committed.removed, 1);
    assert_eq!(
        database::document_sources(&reader)?,
        [replacement.to_str().unwrap()]
    );
    Ok(())
}

/// Verify hidden-directory and format filtering use the traversal name and
/// canonical file extension respectively, while recognized link cycles are skipped.
///
/// Returns success if only eligible documents are visited and stale notes are
/// removed, or a fixture/storage error.
#[cfg(unix)]
#[test]
fn skips_cycles_and_hidden_routes_while_using_the_target_format() -> Result<()> {
    use std::os::unix::fs::symlink;

    let (directory, database) = corpus()?;
    let root = directory.path().join("sources");
    let hidden_root = directory.path().join(".explicit");
    let external_hidden = directory.path().join(".external");
    let hidden_route_target = directory.path().join("excluded");
    for dir in [
        &root,
        &hidden_root,
        &external_hidden,
        &hidden_route_target,
        &root.join(".hidden"),
    ] {
        fs::create_dir(dir)?;
        fs::write(dir.join("note.md"), dir.to_str().unwrap())?;
    }
    let gone = root.join("gone.md");
    fs::write(&gone, "Will be deleted.")?;
    let plain = directory.path().join("plain.txt");
    fs::write(&plain, "Unsupported format.")?;
    let embedder = test_embedder(None, None);
    let roots = [root.clone(), hidden_root.clone()];
    assert!(index_roots(&roots, &database, &embedder).is_success());

    fs::remove_file(&gone)?;
    symlink("self", root.join("self"))?;
    symlink("other", root.join("mutual"))?;
    symlink("mutual", root.join("other"))?;
    symlink(&root, root.join("ancestor"))?;
    symlink(&hidden_route_target, root.join(".hidden-alias"))?;
    symlink(
        external_hidden.join("note.md"),
        root.join(".visible-file.txt"),
    )?;
    symlink(&plain, root.join("unsupported.md"))?;
    let report = index_roots(&roots, &database, &embedder);
    assert!(report.is_success(), "{:?}", report.errors);
    assert_eq!(report.files_seen, 3);
    assert_eq!(report.committed.removed, 1);
    let reader = database::open_read_only(&database.sqlite_path())?;
    let sources = database::document_sources(&reader)?;
    assert!(sources.contains(&external_hidden.join("note.md").to_str().unwrap().to_owned()));
    assert!(sources.contains(&hidden_root.join("note.md").to_str().unwrap().to_owned()));
    Ok(())
}

/// Verify a dangling link leaves global membership unchanged until it is removed.
///
/// Returns success if only the complete pass removes the stale record, or a
/// fixture/storage error.
#[cfg(unix)]
#[test]
fn dangling_symlink_defers_deletion_even_with_an_open_root() -> Result<()> {
    use std::os::unix::fs::symlink;

    let (directory, database) = corpus()?;
    let root = directory.path().join("sources");
    fs::create_dir(&root)?;
    let external = directory.path().join("external.md");
    fs::write(&external, "Reachable at first.")?;
    let link = root.join("link");
    symlink(&external, &link)?;
    let embedder = test_embedder(None, None);
    assert!(index(&root, &database, &embedder).is_success());

    fs::remove_file(&external)?;
    let partial = index(&root, &database, &embedder);
    assert!(!partial.is_success());
    assert_eq!(partial.errors.len(), 1);
    assert!(partial.deletions_deferred);
    assert_eq!(partial.committed.removed, 0);
    let reader = database::open_read_only(&database.sqlite_path())?;
    assert_eq!(
        database::document_sources(&reader)?,
        [external.to_str().unwrap()]
    );

    fs::remove_file(&link)?;
    let complete = index(&root, &database, &embedder);
    assert!(complete.is_success(), "{:?}", complete.errors);
    assert_eq!(complete.committed.removed, 1);
    assert!(database::document_sources(&reader)?.is_empty());
    Ok(())
}
