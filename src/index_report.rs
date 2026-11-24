//! Indexing outcomes and counters independent of console presentation.

use std::fmt::Display;
use std::path::{Path, PathBuf};

use crate::progress::{IndexEvent, ProgressEvent};

/// Outcome of preparing one source document for publication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IndexStatus {
    Unchanged,
    Indexed,
    Reused,
}

/// Document counts for staged work or a successfully committed indexing pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct IndexCounts {
    pub(crate) indexed: usize,
    pub(crate) reused: usize,
    pub(crate) unchanged: usize,
    pub(crate) removed: usize,
}

impl IndexCounts {
    /// Return the counter associated with the prepared file's `status` in `self`.
    fn counter(&mut self, status: IndexStatus) -> &mut usize {
        match status {
            IndexStatus::Indexed => &mut self.indexed,
            IndexStatus::Reused => &mut self.reused,
            IndexStatus::Unchanged => &mut self.unchanged,
        }
    }
}

/// Whether this pass committed, independently of recoverable indexing errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PublicationStatus {
    /// This pass published no document or vector changes.
    NotPublished,
    /// SQLite committed; unchanged passes can reuse the previous snapshot.
    Published {
        snapshot_updated: bool,
        compacted: bool,
    },
}

/// One indexing failure, with its source path when the failure is path-specific.
#[derive(Clone, Debug)]
pub(crate) struct IndexIssue {
    pub(crate) path: Option<PathBuf>,
    pub(crate) message: String,
}

/// Complete outcome of an indexing attempt, including attempts that were rolled back.
#[derive(Debug)]
pub(crate) struct IndexReport {
    pub(crate) publication: PublicationStatus,
    /// Number of Markdown files whose processing was attempted.
    pub(crate) files_seen: usize,
    /// Final staged counts before publication.
    pub(crate) staged: IndexCounts,
    /// Counts made visible by this pass; all zero when nothing was published.
    pub(crate) committed: IndexCounts,
    /// Failures in observation order, including the fatal cause of a rollback.
    pub(crate) errors: Vec<IndexIssue>,
    /// Whether incomplete discovery prevented a global deletion check.
    pub(crate) deletions_deferred: bool,
}

impl IndexReport {
    /// Return whether this report records a committed pass without indexing errors.
    pub(crate) fn is_success(&self) -> bool {
        matches!(self.publication, PublicationStatus::Published { .. }) && self.errors.is_empty()
    }
}

/// Mutable pass statistics shared by synchronous traversal and processing callbacks.
pub(crate) struct IndexProgress<'a> {
    emit: &'a mut dyn FnMut(ProgressEvent<'_>),
    files_seen: usize,
    counts: IndexCounts,
    deletions_deferred: bool,
    errors: Vec<IndexIssue>,
}

impl<'a> IndexProgress<'a> {
    /// Create empty pass statistics that send events to `emit`.
    pub(crate) fn new(emit: &'a mut dyn FnMut(ProgressEvent<'_>)) -> Self {
        Self {
            emit,
            files_seen: 0,
            counts: IndexCounts::default(),
            deletions_deferred: false,
            errors: Vec::new(),
        }
    }

    /// Forward `event` to the pass's event sink.
    pub(crate) fn emit(&mut self, event: ProgressEvent<'_>) {
        (self.emit)(event);
    }

    /// Send `event` with a copy of the current pass counters.
    pub(crate) fn notify(&mut self, event: IndexEvent<'_>) {
        self.emit(ProgressEvent::Index {
            event,
            seen: self.files_seen,
            staged: self.counts,
            errors: self.errors.len(),
        });
    }

    /// Count and report another Markdown file attempted at `path`.
    pub(crate) fn file(&mut self, path: &Path) {
        self.files_seen += 1;
        self.notify(IndexEvent::File(path));
    }

    /// Record and report the prepared `status` for the unique source `path`.
    pub(crate) fn staged(&mut self, path: &Path, status: IndexStatus) {
        *self.counts.counter(status) += 1;
        self.notify(IndexEvent::Staged { path, status });
    }

    /// Record and report removal of the undiscovered source `path`.
    pub(crate) fn removal(&mut self, path: &Path) {
        self.counts.removed += 1;
        self.notify(IndexEvent::Removal(path));
    }

    /// Record the indexer's decision to defer deletion after incomplete discovery.
    pub(crate) fn defer_deletions(&mut self) {
        self.deletions_deferred = true;
    }

    /// Record and report `error` with an optional source `path`.
    pub(crate) fn issue(&mut self, path: Option<&Path>, error: &dyn Display) {
        let issue = IndexIssue {
            path: path.map(Path::to_path_buf),
            message: error.to_string(),
        };
        self.errors.push(issue.clone());
        self.notify(IndexEvent::Issue(&issue));
    }

    /// Consume the statistics and build a report for the final `publication` status.
    ///
    /// Returns staged counts and the deletion status for both outcomes, with
    /// committed counts populated only after successful publication.
    pub(crate) fn finish(self, publication: PublicationStatus) -> IndexReport {
        let staged = self.counts;
        let committed = match publication {
            PublicationStatus::Published { .. } => staged,
            PublicationStatus::NotPublished => IndexCounts::default(),
        };
        IndexReport {
            publication,
            files_seen: self.files_seen,
            staged,
            committed,
            errors: self.errors,
            deletions_deferred: self.deletions_deferred,
        }
    }
}
