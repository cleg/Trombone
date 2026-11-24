//! Filesystem checks for indexed source paths.

use std::fs;
use std::io;
use std::path::Path;

/// Check whether the stored path `relative` to canonical `root` is a regular file.
///
/// An empty `relative` checks `root` itself. Returns false for missing paths,
/// non-directory ancestors, symlinks, and non-regular leaves. Checks ancestors
/// first so symlink targets are never used to determine source eligibility.
///
/// # Errors
///
/// Returns filesystem errors that leave the source type unknown.
pub(crate) fn is_regular_source(root: &Path, relative: &Path) -> io::Result<bool> {
    let mut path = root.to_path_buf();
    let mut components = relative.components();
    loop {
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
                ) =>
            {
                return Ok(false);
            }
            Err(error) => return Err(error),
        };
        let Some(component) = components.next() else {
            return Ok(metadata.is_file());
        };
        if !metadata.is_dir() {
            return Ok(false);
        }
        path.push(component);
    }
}
