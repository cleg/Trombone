use std::fmt::Display;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};

use crate::compact::CompactReport;
use crate::index_report::{IndexCounts, IndexIssue, IndexReport, IndexStatus, PublicationStatus};
use crate::progress::{IndexEvent, ProgressEvent};

/// Console presentation for synchronous indexing and snapshot events.
#[derive(Default)]
pub(crate) struct Reporter {
    indexing: bool,
    root: Option<PathBuf>,
    spinner: Option<ProgressBar>,
}

impl Reporter {
    /// Display `event` with console formatting and a live spinner where available.
    pub(crate) fn handle(&mut self, event: ProgressEvent<'_>) {
        match event {
            ProgressEvent::Index {
                event,
                seen,
                staged,
                errors,
            } => {
                if matches!(event, IndexEvent::Opening(_)) {
                    self.open_index();
                }
                self.refresh_counts(seen, staged, errors);
                match event {
                    IndexEvent::Opening(database) => {
                        self.stage(format_args!("Opening index in {}", path_label(database)))
                    }
                    IndexEvent::Scanning(root) => {
                        self.root = Some(root.to_path_buf());
                        self.stage(format_args!("Scanning Markdown in {}", path_label(root)));
                    }
                    IndexEvent::File(path) => self.file(path),
                    IndexEvent::Embedding { path, chunk } => self.embedding(path, chunk),
                    IndexEvent::Staged { path, status } => self.staged(path, status),
                    IndexEvent::CheckingDeletions => self.stage("Checking for deleted files"),
                    IndexEvent::Removal(path) => self.removal(path),
                    IndexEvent::Issue(issue) => self.issue(issue),
                }
            }
            // Standalone compaction prints its summary and warnings only.
            ProgressEvent::SavingSnapshot if self.indexing => self.stage("Saving USearch snapshot"),
            ProgressEvent::Committing if self.indexing => self.stage("Committing SQLite"),
            ProgressEvent::Compacting if self.indexing => self.stage("Compacting vector index"),
            ProgressEvent::SavingSnapshot
            | ProgressEvent::Committing
            | ProgressEvent::Compacting => {}
            ProgressEvent::AutomaticCompactionFailed(error) => {
                self.error(format_args!(
                    "Automatic compaction failed after indexing: {error}"
                ));
            }
            ProgressEvent::SnapshotCleanupFailed { path, error } => self.error(format_args!(
                "Could not clean up index snapshot {}: {error}",
                path_label(path),
            )),
        }
    }

    /// Mark the reporter as indexing and start its terminal display.
    fn open_index(&mut self) {
        self.indexing = true;
        self.spinner = if io::stdout().is_terminal() && io::stderr().is_terminal() {
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
    }

    /// Display a progress `message` in the spinner or as a plain-text line.
    fn stage(&self, message: impl Display) {
        if let Some(spinner) = &self.spinner {
            spinner.set_message(message.to_string());
            spinner.force_draw();
        } else {
            log_line(message);
        }
    }

    /// Display the source `path` currently being processed without changing pass statistics.
    fn file(&self, path: &Path) {
        if let Some(spinner) = &self.spinner {
            spinner.set_message(format!("Reading {}", self.short_path_label(path)));
        }
    }

    /// Display embedding generation for `chunk` of `path`.
    fn embedding(&self, path: &Path, chunk: usize) {
        if let Some(spinner) = &self.spinner {
            spinner.set_message(format!(
                "Embedding chunk {chunk} · {}",
                self.short_path_label(path)
            ));
        } else if chunk == 1 {
            log_line(format_args!("Embedding: {}", self.short_path_label(path)));
        }
    }

    /// Display the staged removal of `path`; indexing owns the removal count.
    fn removal(&self, path: &Path) {
        if self.spinner.is_none() {
            log_line(format_args!("Removal ready: {}", path_label(path)));
        }
    }

    /// Display the prepared `status` of source `path` using indexing-owned counters.
    fn staged(&self, path: &Path, status: IndexStatus) {
        if self.spinner.is_none() {
            let label = match status {
                IndexStatus::Indexed => "Ready to save",
                IndexStatus::Reused => "Reuse ready",
                IndexStatus::Unchanged => "Unchanged",
            };
            log_line(format_args!("{label}: {}", path_label(path)));
        }
    }

    /// Display an indexing-owned `issue` without recording or counting it again.
    fn issue(&self, issue: &IndexIssue) {
        match &issue.path {
            Some(path) if self.indexing => {
                self.error(format_args!("{}: {}", path_label(path), issue.message))
            }
            _ => self.error(format_args!("Indexing failed: {}", issue.message)),
        }
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

    /// Report the embedding component initialization failure described by `error`.
    pub(crate) fn embedding_init_failed(error: &dyn Display) {
        eprintln!("Failed to initialize embeddings: {error}");
    }

    /// Warn on stderr that `model` resolved to `current` after indexing started with `initial`.
    /// Ignores write errors so the warning cannot affect the command's exit status.
    pub(crate) fn model_digest_changed(model: &str, initial: &str, current: &str) {
        let _ = writeln!(
            io::stderr().lock(),
            "Warning: Ollama model {model} changed during indexing ({initial} → {current}). The published index may contain embeddings from different model versions."
        );
    }

    pub(crate) fn searching_failed(error: &dyn Display) {
        eprintln!("Search failed: {error}");
    }

    /// Print committed compaction counts, or an empty-index message for `None`.
    ///
    /// `report` contains vector counts after successful publication.
    /// Returns success after flushing stdout, or an error if writing fails.
    pub(crate) fn compacted(report: Option<&CompactReport>) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        match report {
            Some(report) => writeln!(
                stdout,
                "Compacted: {} → {} vectors · Removed: {}",
                report.previous_vectors,
                report.retained_vectors,
                report.previous_vectors - report.retained_vectors,
            )?,
            None => writeln!(stdout, "No active vector index to compact.")?,
        }
        stdout.flush()
    }

    /// Report the compaction or output failure described by `error` to stderr.
    pub(crate) fn compaction_failed(error: &dyn Display) {
        eprintln!("Compaction failed: {error}");
    }

    /// Display diagnostic `message`, temporarily suspending the terminal spinner.
    fn error(&self, message: impl Display) {
        if let Some(spinner) = &self.spinner {
            spinner.suspend(|| eprintln!("{message}"));
        } else {
            eprintln!("{message}");
        }
    }

    /// Clear the live display after the indexing pass finishes, without printing a summary.
    pub(crate) fn finish(self) {
        if let Some(spinner) = &self.spinner {
            spinner.finish_and_clear();
        }
    }

    /// Print publication status and counters from the completed indexing `report`.
    ///
    /// Reports deferred deletion and includes prepared counts if publication failed.
    /// Returns success after flushing stdout, or an error if output fails.
    pub(crate) fn indexed(report: &IndexReport) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        match report.publication {
            PublicationStatus::Published {
                snapshot_updated,
                compacted,
            } => writeln!(
                stdout,
                "Publication: published · Snapshot: {} · Compacted: {}",
                if snapshot_updated {
                    "updated"
                } else {
                    "unchanged"
                },
                if compacted { "yes" } else { "no" },
            )?,
            PublicationStatus::NotPublished => writeln!(stdout, "Publication: not published")?,
        }
        if report.deletions_deferred {
            writeln!(
                stdout,
                "Global deletion deferred because source traversal was incomplete."
            )?;
        }
        let counts = report.committed;
        writeln!(
            stdout,
            "Indexed: {} · Reused: {} · Unchanged: {} · Removed: {} · Errors: {} · Seen: {}",
            counts.indexed,
            counts.reused,
            counts.unchanged,
            counts.removed,
            report.errors.len(),
            report.files_seen,
        )?;
        if report.staged != report.committed {
            let staged = report.staged;
            writeln!(
                stdout,
                "Prepared but not published: Indexed: {} · Reused: {} · Unchanged: {} · Removed: {}",
                staged.indexed, staged.reused, staged.unchanged, staged.removed,
            )?;
        }
        stdout.flush()
    }

    /// Return a display label for `path` relative to the current indexing root where possible.
    fn short_path_label(&self, path: &Path) -> String {
        let relative = self
            .root
            .as_ref()
            .and_then(|root| path.strip_prefix(root).ok())
            .unwrap_or(path);
        let label = if relative.as_os_str().is_empty() {
            path.file_name().map(Path::new).unwrap_or(path)
        } else {
            relative
        };
        path_label(label)
    }

    /// Display the `seen` file count, `staged` counts, and `errors` in the spinner.
    fn refresh_counts(&self, seen: usize, staged: IndexCounts, errors: usize) {
        if let Some(spinner) = &self.spinner {
            spinner.set_prefix(format!(
                "Seen {}  Ready {}  Reuse ready {}  Unchanged {}  Errors {}",
                seen, staged.indexed, staged.reused, staged.unchanged, errors,
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
