//! Indexing outcomes and counters independent of console presentation.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
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
    fn counter(&mut self, status: &IndexStatus) -> &mut usize {
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
    /// Final staged counts, excluding prepared files subsequently deleted in this pass.
    pub(crate) staged: IndexCounts,
    /// Counts made visible by this pass; all zero when nothing was published.
    pub(crate) committed: IndexCounts,
    /// Failures in observation order, including the fatal cause of a rollback.
    pub(crate) errors: Vec<IndexIssue>,
}

impl IndexReport {
    /// Return whether this report records a committed pass without indexing errors.
    pub(crate) fn is_success(&self) -> bool {
        matches!(self.publication, PublicationStatus::Published { .. }) && self.errors.is_empty()
    }
}

/// Mutable pass statistics shared by synchronous traversal and processing callbacks.
pub(crate) struct IndexProgress<'a> {
    emit: RefCell<&'a mut dyn FnMut(ProgressEvent<'_>)>,
    files_seen: Cell<usize>,
    counts: Cell<IndexCounts>,
    prepared: RefCell<HashMap<PathBuf, IndexStatus>>,
    errors: RefCell<Vec<IndexIssue>>,
}

impl<'a> IndexProgress<'a> {
    /// Create empty pass statistics that send events to `emit`.
    pub(crate) fn new(emit: &'a mut dyn FnMut(ProgressEvent<'_>)) -> Self {
        Self {
            emit: RefCell::new(emit),
            files_seen: Cell::new(0),
            counts: Cell::new(IndexCounts::default()),
            prepared: RefCell::new(HashMap::new()),
            errors: RefCell::new(Vec::new()),
        }
    }

    /// Forward `event` to the pass's event sink.
    pub(crate) fn emit(&self, event: ProgressEvent<'_>) {
        (self.emit.borrow_mut())(event);
    }

    /// Send `event` with a copy of the current pass counters.
    pub(crate) fn notify(&self, event: IndexEvent<'_>) {
        self.emit(ProgressEvent::Index {
            event,
            seen: self.files_seen.get(),
            staged: self.counts.get(),
            errors: self.errors.borrow().len(),
        });
    }

    /// Count and report another Markdown file attempted at `path`.
    pub(crate) fn file(&self, path: &Path) {
        self.files_seen.set(self.files_seen.get() + 1);
        self.notify(IndexEvent::File(path));
    }

    /// Record and report prepared `status` for `path`, replacing any previous status.
    pub(crate) fn staged(&self, path: &Path, status: IndexStatus) {
        let mut counts = self.counts.get();
        *counts.counter(&status) += 1;
        if let Some(previous) = self
            .prepared
            .borrow_mut()
            .insert(path.to_path_buf(), status)
        {
            *counts.counter(&previous) -= 1;
        }
        self.counts.set(counts);
        self.notify(IndexEvent::Staged { path, status });
    }

    /// Record and report removal of `path`, discarding any prepared status.
    pub(crate) fn removal(&self, path: &Path) {
        let mut counts = self.counts.get();
        if let Some(previous) = self.prepared.borrow_mut().remove(path) {
            *counts.counter(&previous) -= 1;
        }
        counts.removed += 1;
        self.counts.set(counts);
        self.notify(IndexEvent::Removal(path));
    }

    /// Record and report `error` with an optional source `path`.
    pub(crate) fn issue(&self, path: Option<&Path>, error: &dyn Display) {
        let issue = IndexIssue {
            path: path.map(Path::to_path_buf),
            message: error.to_string(),
        };
        self.errors.borrow_mut().push(issue.clone());
        self.notify(IndexEvent::Issue(&issue));
    }

    /// Consume the statistics and build a report for the final `publication` status.
    ///
    /// Returns staged counts for both outcomes, but committed counts only after
    /// publication has succeeded.
    pub(crate) fn finish(self, publication: PublicationStatus) -> IndexReport {
        let staged = self.counts.get();
        let committed = match publication {
            PublicationStatus::Published { .. } => staged,
            PublicationStatus::NotPublished => IndexCounts::default(),
        };
        IndexReport {
            publication,
            files_seen: self.files_seen.get(),
            staged,
            committed,
            errors: self.errors.into_inner(),
        }
    }
}
