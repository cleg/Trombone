//! Reconciles stored source paths with the files in the indexing scope.

use std::path::Path;

use rusqlite::Connection;

use crate::database::{self, Result};
use crate::index_report::IndexProgress;
use crate::progress::IndexEvent;
use crate::source_path;
use crate::walker::InputKind;

/// Delete stored documents whose in-scope paths no longer lead to regular sources.
///
/// # Parameters
///
/// - `connection` is the SQLite connection used to query and delete records.
/// - `root` is the absolute path of the indexed file or directory.
/// - `input_kind` determines whether the scope covers one file or a directory tree.
/// - `progress` tracks removals and failures and emits events for the phase and outcomes.
///
/// Returns success after staging deletions. Reports filesystem errors and
/// leaves the affected records in the index.
///
/// # Errors
///
/// Returns an error for unreported database failures.
pub(super) fn remove_missing_documents(
    connection: &Connection,
    root: &Path,
    input_kind: InputKind,
    progress: &IndexProgress<'_>,
) -> Result<()> {
    // Check stored paths so skipped traversal entries and inaccessible paths
    // are not mistaken for deleted sources.
    progress.notify(IndexEvent::CheckingDeletions);
    for source in database::document_sources(connection)? {
        let path = Path::new(&source);
        let Ok(relative) = path.strip_prefix(root) else {
            continue;
        };
        if matches!(input_kind, InputKind::File) && relative != Path::new("") {
            continue;
        }
        match source_path::is_regular_source(root, relative) {
            Ok(true) => {}
            Ok(false) => {
                database::delete_document(connection, &source)?;
                progress.removal(path);
            }
            Err(error) => {
                progress.issue(
                    Some(path),
                    &format!("Could not check for deletion: {error}"),
                );
            }
        }
    }

    Ok(())
}
