![A fluffy black cat playing a rusty trombone](banner.jpg)

# Trombone

Trombone is a Rust CLI for building a local semantic index of Markdown notes. The `tb` binary splits notes into chunks, generates embeddings through Ollama, and stores chunk metadata in SQLite and vectors in USearch.

## Quick start

You need a current Rust toolchain, a C/C++ compiler for the native dependencies, and a running Ollama server. The [example config](trombone.toml) uses `embeddinggemma-2:740m-bf16` with 768 dimensions.

Set the top-level `roots` array in the TOML before running `tb index`. Put it before the first table header.

```toml
roots = ["./notes", "/path/to/other/notes"]
```

```sh
ollama pull embeddinggemma-2:740m-bf16
cargo build --locked
./target/debug/tb --config trombone.toml index
```

`--chunk-size` overrides `[chunking].size` for the pass. In a terminal, indexing shows live counts, the current file and chunk, and the save and commit stages. When output is redirected, for example through `tee`, it streams plain-text progress lines. The final counts report committed changes.

`tb index --full` clears the existing index before scanning the configured roots. It regenerates embeddings for all indexed content and accepts new model or chunking settings. The reset is committed before scanning, so search has no active index during the rebuild. If the rebuild fails before publication, the index stays empty. Source files and configuration remain in place.

Only one `tb index` or `tb compact` can write to a given database at a time.
A command that encounters an occupied writer lock exits with a busy message.
`tb index` checks the Ollama model before attempting to acquire the lock.
Different databases can be written independently. `tb index` and `tb compact`
enable SQLite write-ahead logging (WAL), including for an existing compatible
database on its next writable open. Search can read the previously published
index while a writer prepares or commits changes. Keep `database.path` on a
local filesystem. SQLite manages the `trombone.sqlite-wal` and
`trombone.sqlite-shm` files alongside the database.
While holding the writer lock, both commands remove unreferenced files with
Trombone-generated snapshot names, including files left by interrupted runs.
Cleanup failures are reported as warnings.

The final summary explicitly states whether the pass was published, whether its vector snapshot changed, and whether automatic compaction ran. It includes indexed, reused, unchanged, and removed document counts, attempted file count, and error count. With a fatal error, committed counts are zero and any prepared work is reported separately. A pass with recoverable errors can publish its successful changes but still exits with a failure status.

Without `--config`, `tb` reads `~/.config/trombone/trombone.toml`. Set `[database].path` to an existing directory. Relative paths in `database.path` and `roots` resolve against the directory containing the real config file after following symlinks. Indexing requires at least one directory path and rejects empty paths before opening SQLite or initializing Ollama. Search and compaction can use an existing database without `roots`. Trombone stores `trombone.sqlite` and `trombone-index-*.usearch` inside that directory. Set `[ollama].model` and `expected_dimensions` together when choosing a model. Run `tb index --full` after changing the model or chunk size for an existing index.

Run `tb index --full` to apply sentence-aware Markdown chunking to previously indexed files. Ordinary indexing skips files whose content is unchanged.

Ollama calls allow 10 seconds to connect and 120 seconds for a complete request,
including response decoding. Override these positive integer limits with
`[ollama].connect_timeout_secs` and `request_timeout_secs`. The total deadline
also covers model metadata lookup. Increase the request timeout for models that
take longer to load or generate embeddings on your machine. An embedding timeout
is reported for the affected file, and the indexing pass continues.

Set the optional `[ollama]` templates to customize the text sent to the embedding model. For EmbeddingGemma:

```toml
chunk_format = "title: {title} | text: {text}"
query_format = "task: search result | query: {query}"
```

The chunk template requires `{text}` and can use `{title}` for the heading path (or `none` if there is no heading). The query template requires `{query}`. Use `{{` and `}}` for literal braces. Omit the templates to use the heading path followed by the chunk body, and the raw search query.

After changing `chunk_format`, rebuild with `tb index --full`. Changing only `query_format` takes effect on the next search.

At the start of indexing or search, Trombone reads the selected model's digest from [Ollama's model list](https://docs.ollama.com/api/tags). Incremental indexing and search require the digest stored with the active snapshot to match, even when the model name and dimensions are unchanged. On a mismatch, restore the indexed model or rebuild with `tb index --full`. Missing models or unavailable digest metadata at startup are errors.

Trombone checks the digest again after publication, including when some files failed to index. If it changed, the command prints a warning on stderr and keeps the published result, the startup digest, and its exit status. The final lookup is silent if it fails or the digest matches. Search only checks at startup, and `compact` preserves the stored digest without contacting Ollama.

## How indexing works

The configured `roots` describe the entire desired document set for `database.path`. Each complete pass reconciles the whole database. Configs that share a database but list different roots can therefore remove one another’s documents.

```mermaid
flowchart TD
    files["Configured roots"] --> walk["Discover and record canonical Markdown paths"]
    walk -->|Complete walk| cleanup["Remove undiscovered documents"]
    cleanup --> rows
    walk -->|New content| chunk["Markdown adapter yields chunks"]
    chunk --> input["Hash the prepared model input"]
    input -->|New hash| ollama["Ollama generates embeddings"]
    input -->|Existing hash| rows
    chunk -->|Chunk metadata| rows["SQLite transaction"]
    ollama -->|Validated vectors| staged["Temporary SQLite table"]
    staged -->|Apply after source reconciliation| working["USearch working index"]
    working -->|Save on vector changes| snapshot["USearch snapshot on disk"]
    snapshot -->|Snapshot filename| rows
    rows -->|Commit| active["Active index"]
    active -->|After commit, if due| compact["Separate compaction transaction"]
```

- A file whose content hash matches its saved version is skipped. Byte-identical files share chunk rows and vectors. SQLite stores source paths, byte and line ranges, heading paths, and the keys that identify vectors in USearch. The Markdown files hold the text.
- The walker follows file and directory symlinks, including targets outside the configured roots. It skips recognized cycles and hidden descendant directories by their name on the traversal route. An explicitly configured hidden root is traversed. The canonical file must have the `.md` extension. The index stores and search reports that canonical absolute path. Overlapping roots and multiple routes to the same file share one processing attempt.
- Chunk size compatibility is checked once before loading the vector snapshot or scanning files. A mismatch aborts the pass with one diagnostic, including for an empty input directory, and leaves existing data unchanged.
- If a snapshot is already active, indexing checks its USearch header and settings before scanning files, including on unchanged passes. It can do this without loading the graph.
- Markdown headings mark section boundaries. Each chunk spans body text. The configured template controls the input sent to Ollama. Search displays the heading path separately from the original excerpt and its line range.
- Line ranges recognize LF, CRLF, and standalone CR. To refresh previously stored line numbers for files containing standalone CR, build a fresh index. Unchanged files reuse their stored chunk metadata.
- Chunks with identical prepared Ollama input share one vector, even across different files. SQLite stores the full 32-byte BLAKE3 hash and the 64-bit USearch key. The key starts with the first eight hash bytes in little-endian order. If already assigned to a different hash, Trombone probes consecutive keys and skips USearch's reserved key. Existing assignments stay stable while referenced.
- The walker records each discovered canonical Markdown path in a temporary SQLite set before reading it, outside the file savepoint. Read, processing, and embedding errors preserve any previous record and allow work on other files to continue. A processing or embedding error rolls back that file’s rows and new vectors. Storage or savepoint failures abort the transaction.
- After a complete walk, Trombone removes any document that was undiscovered across all configured roots. Removing a root or symlink drops a document when its last route disappears. An accessible empty directory counts as complete. Shared chunks survive while any document still references them.
- An unavailable or invalid root, a dangling symlink, or an error enumerating or resolving sources makes the walk incomplete. The summary reports that global deletion was deferred. Successful updates from roots that opened can still publish. If no root opened, the pass ends without publication. Any reported error produces a nonzero exit status.
- Before committing, Trombone checks keys whose chunk references were removed during the pass. It discards their staged vectors or removes them from the graph, and deletes their SQLite hashes, only if no current chunk references the key. This lets an edited note relink an unchanged chunk before cleanup.
- When indexing adds or removes vectors, it saves a USearch snapshot before committing SQLite. The commit publishes the snapshot filename and the updated chunk rows together. The indexing transaction reuses the existing snapshot without loading its graph when vectors stay the same. A failure while applying or publishing vector changes rolls back the pass. The final summary counts indexed and removed files after the commit.

## Search indexed notes

```sh
./target/debug/tb --config trombone.toml search "how are embeddings stored?" --limit 5
```

Search returns up to `--limit` ranked chunks (five by default). A chunk lists all its indexed source paths, including byte-identical copies. A single file can contribute several chunks. Distinct chunks count separately even when they share one embedding and cosine distance.

Console output shows up to five source paths per chunk, followed by the number of hidden paths and the chunk excerpt. Available paths show their line range. In a terminal, paths use status colors and are shortened to fit on one row. JSON output includes every path.

Search collects chunk metadata from one consistent SQLite/vector snapshot, then releases the mapping and transaction before reading Markdown. It reads and hashes each source path at most once per query and releases each file's text before reading the next file. A copy whose hash matches the indexed version supplies one excerpt per chunk. Further copies are checked by streaming their bytes through the hash when no other excerpt is needed from that file. Every indexed path keeps its place in the result with its indexed line range and availability, even when no copy can supply the excerpt.

If a writer retires the selected USearch file before search can open it, search
retries once in a new read transaction. A second publication can race that
retry and surface a snapshot-open error.

Use `--output=json` to get ranked chunks with `sources`, `line_start`, `line_end`, `section_path`, `cosine_distance`, and `text`. Each entry in `sources` has a `path`, its indexed `line_start` and `line_end`, and a `status` of `available`, `missing`, or `excerpt_unavailable`. The last status covers changed files and other read failures. Line numbers are one-based and inclusive. `section_path` is an array of headings or `null`, and `text` is `null` if there is no verified excerpt. An empty result is `[]`.

## Compact the vector index

```sh
./target/debug/tb --config trombone.toml compact
```

Compaction builds a fresh USearch graph containing only vector keys referenced by current SQLite chunks. It rebuilds the graph after deletions and preserves existing embeddings, keys, and model settings. It does not read source files or contact Ollama, and it uses the stored index settings rather than the current embedding model configuration.

The command reports vector counts before and after compaction. A fresh graph can change approximate search ordering even though the embeddings stay the same. Compaction does not reindex edited files or remove missing document paths; run `tb index` to reconcile those changes first.

Compaction holds the writer lock and publishes its snapshot through the same save, sync, and SQLite commit sequence as indexing. If rebuilding or publication fails, the previous snapshot stays active. The old snapshot is removed after a successful commit. With no active index, the command reports that there is nothing to compact.

After indexing commits, automatic compaction takes a separate writer lock. It reads the current USearch header and counts the distinct vector keys referenced by SQLite chunks. It rebuilds the graph when either condition holds:

- **Deleted slots.** At least 100 slots are still marked deleted and they make up at least 25% of the graph's slots. A graph with no live vectors and any deleted slots is also rebuilt. USearch can reuse deleted slots, so the header reflects how many holes remain.
- **Surplus live vectors.** USearch reports at least 10 more live vectors than SQLite references, and the surplus is at least 10% of the SQLite count. If SQLite references no vectors, any live vector in USearch triggers a rebuild. This compares counts across the two stores. Compaction itself uses the exact SQLite keys to decide what to keep.

A pass that changes vectors and triggers compaction writes two snapshots in sequence. If automatic compaction fails, indexing remains committed and reports a warning. The number of scanned files does not affect the threshold.

To defer automatic compaction for one indexing pass:

```sh
./target/debug/tb --config trombone.toml index --no-compact
```

A later pass can compact even if indexing made no vector changes. The decision always uses the snapshot that is active when compaction takes its writer lock.

Database schema and vector format versions must exactly match the versions supported by the binary. The current database schema is version 6, and the logical vector format is version 2 (hash-derived keys). There are no migrations or compatibility fallbacks during development. When either version differs, the command exits with an error; use a fresh database directory and reindex the notes.

## Development

Embedding generation is isolated in `src/embeddings.rs`. Indexing and search use the synchronous `Embedder` interface for chunk embeddings, query embeddings, and embedding settings. `OllamaEmbedder` owns model-specific input formatting, the Ollama client, one reusable Tokio runtime, and response validation. Compaction does not construct an embedder. Document opening and Markdown chunking remain outside this component.

`indexer` owns the file-processing sequence: unchanged-file decisions, content
reuse, lazy adapter opening, chunk iteration, vector-key allocation, and vector
staging. Its file savepoint covers the document and chunk metadata as well as
new vectors. `database` supplies storage queries and writes, including cached
chunk and vector inserts, without adapter or embedding callbacks.
`IndexStatus` belongs to the indexing report types. Persistence orchestration
regressions live in `src/indexer/tests.rs` and its submodules.

Every chunk carries a typed `SourceLocation` with zero-based, half-open byte offsets and one-based, inclusive line numbers. SQLite stores these coordinates in dedicated integer columns and reconstructs the same type when resolving search candidates. JSON attributes hold additional metadata. Search JSON keeps its existing `line_start` and `line_end` fields.

`indexer::run` returns an `IndexReport` for every attempt, including aborted attempts. The report separates staged and committed `IndexCounts`, records files attempted and diagnostics with optional source paths, and exposes a `PublicationStatus`. Published status includes snapshot-update and compaction flags. `is_success()` requires publication without indexing errors. Indexing owns these statistics and emits structured progress events, including current counters. Snapshot publication emits save and commit events and reports cleanup failures as warnings. The CLI passes a callback to `Reporter`, which formats progress, warnings, and the final report. Other callers can capture events or ignore them.

```sh
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

Licensed under the [MIT License](LICENSE).
