//! Discovers canonical Markdown paths through directory trees and symlinks.

use std::collections::HashSet;
use std::fs::{self, DirEntry, FileType};
use std::io;
use std::path::{Path, PathBuf};

use crate::error::Result;

/// Whether a configured root could be opened and fully enumerated.
pub(crate) enum WalkOutcome {
    Unopened,
    Partial,
    Complete,
}

/// Visit the canonical Markdown files reachable from directory `root`.
///
/// `visit` receives each discovered route's canonical file. `report_error`
/// receives traversal failures with the path where available. Returns whether
/// the root opened and whether discovery was complete. Hidden descendant
/// directories and recognized symlink loops are skipped.
///
/// # Errors
///
/// Propagates fatal errors from `visit`, stopping traversal immediately.
pub(crate) fn walk(
    root: &Path,
    mut visit: impl FnMut(&Path) -> Result<()>,
    mut report_error: impl FnMut(&Path, &io::Error),
) -> Result<WalkOutcome> {
    let opened = fs::canonicalize(root).and_then(|path| {
        let entries = fs::read_dir(&path)?;
        Ok((path, entries))
    });
    let mut stack = match opened {
        Ok(opened) => vec![opened],
        Err(error) => {
            report_error(root, &error);
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
                report_error(directory, &error);
                outcome = WalkOutcome::Partial;
                continue;
            }
        };
        let (path, kind) = match source_entry(&entry) {
            Ok(Some(source)) => source,
            Ok(None) => continue,
            Err(error) => {
                report_error(&entry.path(), &error);
                outcome = WalkOutcome::Partial;
                continue;
            }
        };

        if kind.is_file() {
            visit(&path)?;
        } else if !stack.iter().any(|(ancestor, _)| ancestor == &path) {
            match fs::read_dir(&path) {
                Ok(entries) => stack.push((path, entries)),
                Err(error) => {
                    report_error(&path, &error);
                    outcome = WalkOutcome::Partial;
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
