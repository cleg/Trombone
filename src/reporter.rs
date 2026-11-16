use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt::Display;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};

use crate::database::IndexStatus;

#[derive(Clone, Copy, Default)]
struct Counts {
    indexed: usize,
    reused: usize,
    unchanged: usize,
}

impl Counts {
    fn record(&mut self, status: &IndexStatus) {
        match status {
            IndexStatus::Indexed => self.indexed += 1,
            IndexStatus::Reused => self.reused += 1,
            IndexStatus::Unchanged => self.unchanged += 1,
        }
    }
}

pub(crate) enum Phase {
    Scanning,
    CheckingDeletions,
    SavingSnapshot,
    Committing,
}

pub(crate) struct Reporter<'a> {
    root: &'a Path,
    spinner: Option<ProgressBar>,
    files_seen: Cell<usize>,
    staged: Cell<Counts>,
    completed: RefCell<HashMap<PathBuf, IndexStatus>>,
    removed: Cell<usize>,
    errors: Cell<usize>,
}

impl<'a> Reporter<'a> {
    pub(crate) fn new(root: &'a Path) -> Self {
        let spinner = if io::stdout().is_terminal() && io::stderr().is_terminal() {
            let bar = ProgressBar::new_spinner().with_finish(ProgressFinish::AndClear);
            if bar.is_hidden() {
                None
            } else {
                bar.set_style(
                    ProgressStyle::with_template(
                        "{spinner:.cyan} {elapsed_precise}  {wide_msg}\n  {prefix:.dim}",
                    )
                    .expect("valid indexing progress template"),
                );
                bar.enable_steady_tick(Duration::from_millis(100));
                Some(bar)
            }
        } else {
            None
        };
        let reporter = Self {
            root,
            spinner,
            files_seen: Cell::new(0),
            staged: Cell::new(Counts::default()),
            completed: RefCell::new(HashMap::new()),
            removed: Cell::new(0),
            errors: Cell::new(0),
        };
        reporter.stage(format_args!("Opening index for {}", path_label(root)));
        reporter
    }

    pub(crate) fn phase(&self, phase: Phase) {
        self.stage(match phase {
            Phase::Scanning => "Scanning Markdown",
            Phase::CheckingDeletions => "Checking for deleted files",
            Phase::SavingSnapshot => "Saving USearch snapshot",
            Phase::Committing => "Committing SQLite",
        });
    }

    fn stage(&self, message: impl Display) {
        if let Some(spinner) = &self.spinner {
            spinner.set_message(message.to_string());
            self.refresh_counts();
            spinner.force_draw();
        } else {
            log_line(message);
        }
    }

    pub(crate) fn file(&self, path: &Path) {
        self.files_seen.set(self.files_seen.get() + 1);
        if let Some(spinner) = &self.spinner {
            spinner.set_message(format!("Reading {}", self.short_path_label(path)));
            self.refresh_counts();
        }
    }

    pub(crate) fn embedding(&self, path: &Path, chunk: usize) {
        if let Some(spinner) = &self.spinner {
            spinner.set_message(format!(
                "Embedding chunk {chunk} · {}",
                self.short_path_label(path)
            ));
        } else if chunk == 1 {
            log_line(format_args!("Embedding: {}", self.short_path_label(path)));
        }
    }

    pub(crate) fn removal(&self, path: &Path) {
        // The file may have disappeared after it was staged in this run.
        self.completed.borrow_mut().remove(path);
        self.removed.set(self.removed.get() + 1);
        if self.spinner.is_none() {
            log_line(format_args!("Removal ready: {}", path_label(path)));
        }
    }

    pub(crate) fn staged(&self, path: &Path, status: IndexStatus) {
        let mut counts = self.staged.get();
        counts.record(&status);
        self.staged.set(counts);
        if self.spinner.is_some() {
            self.refresh_counts();
        } else {
            let label = match status {
                IndexStatus::Indexed => "Ready to save",
                IndexStatus::Reused => "Reuse ready",
                IndexStatus::Unchanged => "Unchanged",
            };
            log_line(format_args!("{label}: {}", path_label(path)));
        }
        self.completed
            .borrow_mut()
            .insert(path.to_path_buf(), status);
    }

    pub(crate) fn file_error(&self, path: &Path, error: &dyn Display) {
        self.error(format_args!("{}: {error}", path_label(path)));
    }

    pub(crate) fn indexing_aborted_at(&self, path: &Path, error: &dyn Display) {
        self.error(format_args!(
            "Indexing failed: {}: {error}",
            path_label(path)
        ));
    }

    pub(crate) fn deletion_check_error(&self, path: &Path, error: &io::Error) {
        self.error(format_args!(
            "Could not check {}: {error}",
            path_label(path)
        ));
    }

    pub(crate) fn snapshot_load_failed(&self, path: &Path, error: &dyn Display) {
        self.error(format_args!(
            "Indexing failed: Cannot load active index {}: {error}",
            path_label(path)
        ));
    }

    pub(crate) fn snapshot_cleanup_failed(path: &Path, error: &io::Error) {
        eprintln!(
            "Could not remove previous index {}: {error}",
            path_label(path)
        );
    }

    pub(crate) fn indexing_failed(error: &dyn Display) {
        eprintln!("Indexing failed: {error}");
    }

    pub(crate) fn config_location_failed(error: &dyn Display) {
        eprintln!("Failed to locate config: {error}");
    }

    pub(crate) fn config_load_failed(path: &Path, error: &dyn Display) {
        eprintln!("Failed to load config {}: {error}", path_label(path));
    }

    pub(crate) fn ollama_init_failed(error: &dyn Display) {
        eprintln!("Failed to initialize Ollama client: {error}");
    }

    pub(crate) fn search_preview(query: &str, limit: usize) {
        log_line(format_args!("Searching: {query}, limit: {limit}"));
    }

    fn error(&self, message: impl Display) {
        self.errors.set(self.errors.get() + 1);
        if let Some(spinner) = &self.spinner {
            self.refresh_counts();
            spinner.suspend(|| eprintln!("{message}"));
        } else {
            eprintln!("{message}");
        }
    }

    pub(crate) fn committed(self) {
        if let Some(spinner) = &self.spinner {
            spinner.finish_and_clear();
        }
        let mut counts = Counts::default();
        for status in self.completed.into_inner().into_values() {
            counts.record(&status);
        }
        log_line(format_args!(
            "Indexed: {} · Reused: {} · Unchanged: {} · Removed: {} · Errors: {}",
            counts.indexed,
            counts.reused,
            counts.unchanged,
            self.removed.get(),
            self.errors.get()
        ));
    }

    fn short_path_label(&self, path: &Path) -> String {
        let relative = path.strip_prefix(self.root).unwrap_or(path);
        let label = if relative.as_os_str().is_empty() {
            path.file_name().map(Path::new).unwrap_or(path)
        } else {
            relative
        };
        path_label(label)
    }

    fn refresh_counts(&self) {
        if let Some(spinner) = &self.spinner {
            let counts = self.staged.get();
            spinner.set_prefix(format!(
                "Seen {}  Ready {}  Reuse ready {}  Unchanged {}  Errors {}",
                self.files_seen.get(),
                counts.indexed,
                counts.reused,
                counts.unchanged,
                self.errors.get()
            ));
        }
    }
}

fn log_line(message: impl Display) {
    let mut stdout = io::stdout().lock();
    let _ = writeln!(stdout, "{message}").and_then(|()| stdout.flush());
}

fn path_label(path: &Path) -> String {
    path.to_string_lossy()
        .chars()
        .map(|character| {
            if character.is_control() {
                '�'
            } else {
                character
            }
        })
        .collect()
}
