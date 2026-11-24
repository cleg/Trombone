use std::io;
use std::ops::ControlFlow;
use std::path::Path;

use walkdir::{DirEntry, WalkDir};

pub(crate) struct Walker<'a> {
    pub(crate) report_error: &'a dyn Fn(&Path, &io::Error),
}

pub(crate) enum InputKind {
    File,
    Directory,
}

pub(crate) enum WalkOutcome {
    Completed(InputKind),
    StoppedByProcessor,
}

impl Walker<'_> {
    /// Visit Markdown sources under existing `root` using canonical absolute paths.
    ///
    /// `processor` receives each source and may explicitly stop with `Break`.
    /// Reports recoverable traversal and processor errors through this walker's
    /// callback and continues. Returns the input kind on completion or
    /// `StoppedByProcessor` after a break.
    /// Root symlinks are resolved; descendant symlinks are not followed.
    ///
    /// # Errors
    /// Reports and returns an error if the root is inaccessible or unsupported.
    pub(crate) fn walk<F>(&self, root: &Path, mut processor: F) -> io::Result<WalkOutcome>
    where
        F: FnMut(&Path) -> io::Result<ControlFlow<()>>,
    {
        let root =
            std::fs::canonicalize(root).inspect_err(|error| (self.report_error)(root, error))?;

        let metadata =
            std::fs::metadata(&root).inspect_err(|error| (self.report_error)(&root, error))?;
        if metadata.is_file() && is_markdown(&root) {
            match processor(&root) {
                Ok(ControlFlow::Break(())) => return Ok(WalkOutcome::StoppedByProcessor),
                Ok(ControlFlow::Continue(())) => {}
                Err(error) => (self.report_error)(&root, &error),
            }
            return Ok(WalkOutcome::Completed(InputKind::File));
        }
        if !metadata.is_dir() {
            let error = io::Error::new(
                io::ErrorKind::InvalidInput,
                "expected a Markdown (.md) file or a directory",
            );
            (self.report_error)(&root, &error);
            return Err(error);
        }

        let entries = WalkDir::new(&root).into_iter().filter_entry(should_visit);

        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    let path = error.path().unwrap_or(&root).to_path_buf();
                    (self.report_error)(&path, &io::Error::other(error));
                    continue;
                }
            };

            if !is_supported_file(&entry) {
                continue;
            }

            match processor(entry.path()) {
                Ok(ControlFlow::Break(())) => return Ok(WalkOutcome::StoppedByProcessor),
                Ok(ControlFlow::Continue(())) => {}
                Err(error) => (self.report_error)(entry.path(), &error),
            }
        }

        Ok(WalkOutcome::Completed(InputKind::Directory))
    }
}

fn should_visit(entry: &DirEntry) -> bool {
    entry.depth() == 0
        || !entry.file_type().is_dir()
        || !entry.file_name().as_encoded_bytes().starts_with(b".")
}

fn is_supported_file(entry: &DirEntry) -> bool {
    entry.file_type().is_file() && is_markdown(entry.path())
}

fn is_markdown(path: &Path) -> bool {
    path.extension().is_some_and(|ext| ext == "md")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        /// Create and return a temporary directory with a canonical absolute path.
        ///
        /// Panics if temporary storage cannot be created or canonicalized.
        fn new() -> Self {
            static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
            loop {
                let path = std::env::temp_dir().join(format!(
                    "trombone-walker-{}-{}",
                    std::process::id(),
                    NEXT_ID.fetch_add(1, Ordering::Relaxed),
                ));
                match std::fs::create_dir(&path) {
                    Ok(()) => return Self(std::fs::canonicalize(path).unwrap()),
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(error) => panic!("cannot create test directory: {error}"),
                }
            }
        }

        fn write(&self, relative: &str) {
            let path = self.0.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "test").unwrap();
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Verify unsupported or missing roots fail before any file is processed.
    #[test]
    fn invalid_roots_are_fatal() {
        let directory = TestDirectory::new();
        directory.write("file.txt");
        let walker = Walker {
            report_error: &|_, _| {},
        };
        for name in ["file.txt", "missing"] {
            let result = walker.walk(&directory.0.join(name), |_| {
                panic!("processor must not run")
            });
            assert!(result.is_err());
        }
    }

    /// Verify a single-file walk completes after visiting only the selected file.
    #[test]
    fn a_file_input_processes_only_that_file() {
        let directory = TestDirectory::new();
        directory.write("selected.md");
        directory.write("neighbor.md");
        let selected = directory.0.join("selected.md");
        let mut found = Vec::new();
        let kind = Walker {
            report_error: &|_, _| {},
        }
        .walk(&selected, |path| {
            found.push(path.to_path_buf());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
        assert!(matches!(kind, WalkOutcome::Completed(InputKind::File)));
        assert_eq!(found, [selected]);
    }

    /// Verify a processor error is reported and the remaining files are visited.
    #[test]
    fn processor_errors_are_reported_without_stopping_the_walk() {
        let directory = TestDirectory::new();
        for name in ["a.md", "nested/b.md", "c.md"] {
            directory.write(name);
        }
        let reported = std::cell::Cell::new(0);
        let walker = Walker {
            report_error: &|_, error| {
                assert_eq!(error.to_string(), "processing failed");
                reported.set(reported.get() + 1);
            },
        };
        let mut calls = 0;
        let result = walker
            .walk(&directory.0, |_| {
                calls += 1;
                if calls == 1 {
                    Err(io::Error::other("processing failed"))
                } else {
                    Ok(ControlFlow::Continue(()))
                }
            })
            .unwrap();
        assert!(matches!(
            result,
            WalkOutcome::Completed(InputKind::Directory)
        ));
        assert_eq!(calls, 3);
        assert_eq!(reported.get(), 1);
    }

    /// Verify an explicit processor stop ends traversal immediately.
    #[test]
    fn explicit_processor_break_stops_the_walk() {
        let directory = TestDirectory::new();
        directory.write("a.md");
        directory.write("b.md");
        let mut calls = 0;
        let outcome = Walker {
            report_error: &|_, _| {},
        }
        .walk(&directory.0, |_| {
            calls += 1;
            Ok(ControlFlow::Break(()))
        })
        .unwrap();
        assert!(matches!(outcome, WalkOutcome::StoppedByProcessor));
        assert_eq!(calls, 1);
    }

    /// Verify traversal filters unsupported files and hidden descendant directories.
    #[test]
    fn filters_files_and_hidden_directories_but_allows_a_hidden_root() {
        let directory = TestDirectory::new();
        for name in [
            ".notes/a.md",
            ".notes/nested/b.md",
            ".notes/.hidden/c.md",
            ".notes/nested/.hidden/deep/d.md",
            ".notes/skip.txt",
            ".notes/UPPER.MD",
            ".notes/.visible-file.md",
            ".notes/directory.md/skip.txt",
        ] {
            directory.write(name);
        }

        let root = directory.0.join(".notes");
        let mut found = BTreeSet::new();
        Walker {
            report_error: &|_, _| {},
        }
        .walk(&root, |path| {
            assert!(path.is_absolute());
            found.insert(path.strip_prefix(&root).unwrap().to_path_buf());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();

        let expected = ["a.md", "nested/b.md", ".visible-file.md"]
            .into_iter()
            .map(PathBuf::from)
            .collect();
        assert_eq!(found, expected);
    }
}
