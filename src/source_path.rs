//! Canonical source identities, including stored paths whose files have disappeared.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Normalize stored `path` through its deepest existing ancestor.
///
/// Returns the canonical path when the source exists. For a missing source,
/// resolves symlinks in the existing ancestor and appends the missing suffix so
/// deletion reconciliation uses the same identity as indexing. Does not create files.
///
/// # Errors
///
/// Returns filesystem errors other than not-found and not-a-directory, or an
/// invalid-input error if no existing ancestor or missing filename can be found.
pub(crate) fn normalize_stored(path: &Path) -> io::Result<PathBuf> {
    let absolute = std::path::absolute(path)?;
    let mut ancestor = absolute.as_path();
    let mut missing = Vec::new();
    loop {
        match fs::canonicalize(ancestor) {
            Ok(mut canonical) => {
                for component in missing.into_iter().rev() {
                    canonical.push(component);
                }
                return Ok(canonical);
            }
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
                ) =>
            {
                let name = ancestor.file_name().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "Cannot normalize missing source path",
                    )
                })?;
                missing.push(name.to_os_string());
                ancestor = ancestor.parent().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "Source path has no existing ancestor",
                    )
                })?;
            }
            Err(error) => return Err(error),
        }
    }
}

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

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::symlink;

    use super::*;

    /// Verify aliases and missing descendants share the real directory's identity.
    ///
    /// Returns success for existing and deleted sources, or a fixture/path error.
    #[test]
    fn resolves_symlink_ancestors_for_existing_and_missing_sources() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let real = directory.path().join("real");
        let alias = directory.path().join("alias");
        fs::create_dir(&real)?;
        fs::create_dir(real.join("nested"))?;
        fs::write(real.join("note.md"), "A note.")?;
        symlink(&real, &alias)?;
        let canonical = fs::canonicalize(&real)?;
        assert_eq!(
            normalize_stored(&alias.join("nested/../note.md"))?,
            canonical.join("note.md"),
        );
        assert_eq!(
            normalize_stored(&alias.join("missing/deleted.md"))?,
            canonical.join("missing/deleted.md"),
        );
        Ok(())
    }
}
