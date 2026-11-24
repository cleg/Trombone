//! Discovers canonical Markdown paths through directory trees and symlinks.

use std::collections::HashSet;
use std::fs::{self, DirEntry, FileType};
use std::io;
use std::path::{Path, PathBuf};

use crate::error::Result;

/// A discovered file or a filesystem issue that made discovery incomplete.
pub(crate) enum WalkEvent<'a> {
    File(&'a Path),
    Issue {
        path: &'a Path,
        error: &'a io::Error,
    },
}

/// Completeness of a traversal of one configured root.
pub(crate) enum WalkOutcome {
    Unopened,
    Partial,
    Complete,
}

/// Visit the canonical Markdown files reachable from directory `root`.
///
/// `emit` receives each discovered route's canonical file and any filesystem
/// issues. Returns `Unopened` if the root cannot be opened, `Partial` if a
/// filesystem issue interrupts part of the walk, or `Complete` otherwise.
/// Hidden descendant directories and recognized loops are skipped.
///
/// # Errors
///
/// Propagates fatal errors from `emit`, stopping traversal immediately.
pub(crate) fn walk(
    root: &Path,
    mut emit: impl FnMut(WalkEvent<'_>) -> Result<()>,
) -> Result<WalkOutcome> {
    let opened = fs::canonicalize(root).and_then(|path| {
        let entries = fs::read_dir(&path)?;
        Ok((path, entries))
    });
    let mut stack = match opened {
        Ok(opened) => vec![opened],
        Err(error) => {
            emit(WalkEvent::Issue {
                path: root,
                error: &error,
            })?;
            return Ok(WalkOutcome::Unopened);
        }
    };
    let mut outcome = WalkOutcome::Complete;
    while let Some((directory, entries)) = stack.last_mut() {
        let Some(entry) = entries.next() else {
            stack.pop();
            continue;
        };
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                outcome = WalkOutcome::Partial;
                emit(WalkEvent::Issue {
                    path: directory,
                    error: &error,
                })?;
                continue;
            }
        };
        let (path, kind) = match source_entry(&entry) {
            Ok(Some(source)) => source,
            Ok(None) => continue,
            Err(error) => {
                outcome = WalkOutcome::Partial;
                emit(WalkEvent::Issue {
                    path: &entry.path(),
                    error: &error,
                })?;
                continue;
            }
        };

        if kind.is_file() {
            emit(WalkEvent::File(&path))?;
        } else if !stack.iter().any(|(ancestor, _)| ancestor == &path) {
            match fs::read_dir(&path) {
                Ok(entries) => stack.push((path, entries)),
                Err(error) => {
                    outcome = WalkOutcome::Partial;
                    emit(WalkEvent::Issue {
                        path: &path,
                        error: &error,
                    })?;
                }
            }
        }
    }
    Ok(outcome)
}

/// Resolve `entry` to its canonical path and target type when it can be visited.
///
/// Returns None for hidden directories (using the entry name), recognized link
/// cycles, non-regular entries, and files with an unsupported canonical extension.
/// Returns an error if the type or canonical path cannot be determined.
fn source_entry(entry: &DirEntry) -> io::Result<Option<(PathBuf, FileType)>> {
    let path = entry.path();
    let mut kind = entry.file_type()?;
    if kind.is_symlink() {
        kind = match fs::metadata(&path) {
            Ok(metadata) => metadata.file_type(),
            Err(_) if is_symlink_loop(&path) => return Ok(None),
            Err(error) => return Err(error),
        };
    }
    if kind.is_dir() {
        if entry.file_name().as_encoded_bytes().starts_with(b".") {
            return Ok(None);
        }
        return Ok(Some((fs::canonicalize(path)?, kind)));
    }
    if kind.is_file() {
        let canonical = fs::canonicalize(path)?;
        if canonical
            .extension()
            .is_some_and(|extension| extension == "md")
        {
            return Ok(Some((canonical, kind)));
        }
    }
    Ok(None)
}

/// Confirm that following the named symlink at `path` repeats a link.
///
/// Called after target lookup fails. Returns false when a link or its parent
/// cannot be resolved, including chains that merely exceed the OS hop limit.
fn is_symlink_loop(path: &Path) -> bool {
    let mut path = path.to_path_buf();
    let mut seen = HashSet::new();
    while seen.insert(path.clone()) {
        let Ok(target) = fs::read_link(&path) else {
            return false;
        };
        let target = path.with_file_name(target);
        let (Some(parent), Some(name)) = (target.parent(), target.file_name()) else {
            return false;
        };
        let Ok(parent) = fs::canonicalize(parent) else {
            return false;
        };
        path = parent.join(name);
    }
    true
}
