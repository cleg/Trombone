use std::io;
use std::ops::ControlFlow;
use std::path::Path;

use walkdir::{DirEntry, WalkDir};

pub(crate) struct Walker<'a> {
    pub(crate) continue_on_error: bool,
    pub(crate) report_error: &'a dyn Fn(&Path, &io::Error),
}

pub(crate) enum InputKind {
    File,
    Directory,
}

impl Walker<'_> {
    /// Visit Markdown sources under existing `root` using canonical absolute paths.
    ///
    /// `processor` receives each source and may stop traversal with Break. Returns
    /// the input kind, or a root/traversal/processing error under the chosen policy.
    /// Root symlinks are resolved; descendant symlinks are not followed.
    pub(crate) fn walk<F>(&self, root: &Path, mut processor: F) -> io::Result<InputKind>
    where
        F: FnMut(&Path) -> io::Result<ControlFlow<()>>,
    {
        let root =
            std::fs::canonicalize(root).inspect_err(|error| (self.report_error)(root, error))?;

        let metadata =
            std::fs::metadata(&root).inspect_err(|error| (self.report_error)(&root, error))?;
        if metadata.is_file() && is_markdown(&root) {
            if let Err(error) = processor(&root) {
                self.handle_conditional_error(&root, error)?;
            }
            return Ok(InputKind::File);
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
                    self.handle_conditional_error(&path, io::Error::other(error))?;
                    continue;
                }
            };

            if !is_supported_file(&entry) {
                continue;
            }

            match processor(entry.path()) {
                Ok(ControlFlow::Break(())) => break,
                Ok(ControlFlow::Continue(())) => {}
                Err(error) => self.handle_conditional_error(entry.path(), error)?,
            }
        }

        Ok(InputKind::Directory)
    }

    fn handle_conditional_error(&self, path: &Path, error: io::Error) -> io::Result<()> {
        (self.report_error)(path, &error);
        if self.continue_on_error {
            Ok(())
        } else {
            Err(error)
        }
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

    #[test]
    fn invalid_roots_are_fatal_regardless_of_continuation() {
        let directory = TestDirectory::new();
        directory.write("file.txt");
        for continue_on_error in [true, false] {
            let walker = Walker {
                continue_on_error,
                report_error: &|_, _| {},
            };
            for name in ["file.txt", "missing"] {
                let result = walker.walk(&directory.0.join(name), |_| {
                    panic!("processor must not run")
                });
                assert!(result.is_err());
            }
        }
    }

    #[test]
    fn a_file_input_processes_only_that_file() {
        let directory = TestDirectory::new();
        directory.write("selected.md");
        directory.write("neighbor.md");
        let selected = directory.0.join("selected.md");
        let mut found = Vec::new();
        let kind = Walker {
            continue_on_error: true,
            report_error: &|_, _| {},
        }
        .walk(&selected, |path| {
            found.push(path.to_path_buf());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
        assert!(matches!(kind, InputKind::File));
        assert_eq!(found, [selected]);
    }

    #[test]
    fn processor_errors_continue_or_stop_the_actual_walk() {
        let directory = TestDirectory::new();
        for name in ["a.md", "nested/b.md", "c.md"] {
            directory.write(name);
        }
        for continue_on_error in [true, false] {
            let walker = Walker {
                continue_on_error,
                report_error: &|_, _| {},
            };
            let mut calls = 0;
            let result = walker.walk(&directory.0, |_| {
                calls += 1;
                if calls == 1 {
                    Err(io::Error::other("processing failed"))
                } else {
                    Ok(ControlFlow::Continue(()))
                }
            });
            assert_eq!(result.is_ok(), continue_on_error);
            assert_eq!(calls, if continue_on_error { 3 } else { 1 });
        }
    }

    #[test]
    fn processor_can_stop_successfully_regardless_of_error_policy() {
        let directory = TestDirectory::new();
        directory.write("a.md");
        directory.write("b.md");
        for continue_on_error in [true, false] {
            let mut calls = 0;
            Walker {
                continue_on_error,
                report_error: &|_, _| {},
            }
            .walk(&directory.0, |_| {
                calls += 1;
                Ok(ControlFlow::Break(()))
            })
            .unwrap();
            assert_eq!(calls, 1);
        }
    }

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
            continue_on_error: true,
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
