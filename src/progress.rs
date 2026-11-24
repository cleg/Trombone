//! Operation events shared by the indexing core and its callers.

use std::fmt::Display;
use std::path::Path;

use crate::index_report::{IndexCounts, IndexIssue, IndexStatus};

/// A synchronous progress or diagnostic notification.
pub(crate) enum ProgressEvent<'a> {
    Index {
        event: IndexEvent<'a>,
        seen: usize,
        staged: IndexCounts,
        errors: usize,
    },
    SavingSnapshot,
    Committing,
    Compacting,
    AutomaticCompactionFailed(&'a dyn Display),
    SnapshotCleanupFailed {
        path: &'a Path,
        error: &'a dyn Display,
    },
}

/// Indexing activity and diagnostics with no console formatting.
pub(crate) enum IndexEvent<'a> {
    Opening(&'a Path),
    Scanning(&'a Path),
    File(&'a Path),
    Embedding { path: &'a Path, chunk: usize },
    Staged { path: &'a Path, status: IndexStatus },
    CheckingDeletions,
    Removal(&'a Path),
    Issue(&'a IndexIssue),
}
