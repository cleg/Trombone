CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    attributes TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(attributes) AND json_type(attributes) = 'object')
);

CREATE INDEX IF NOT EXISTS documents_content_hash ON documents(content_hash);

CREATE TABLE IF NOT EXISTS embeddings (
    vector_key INTEGER PRIMARY KEY CHECK (vector_key <> -1),
    input_hash BLOB NOT NULL UNIQUE CHECK (typeof(input_hash) = 'blob' AND length(input_hash) = 32)
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    vector_key INTEGER NOT NULL REFERENCES embeddings(vector_key),
    content_hash TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    byte_start INTEGER NOT NULL CHECK (byte_start >= 0),
    byte_end INTEGER NOT NULL CHECK (byte_end > byte_start),
    line_start INTEGER NOT NULL CHECK (line_start >= 1),
    line_end INTEGER NOT NULL CHECK (line_end >= line_start),
    section_path TEXT
        CHECK (section_path IS NULL OR
            (json_valid(section_path) AND json_type(section_path) = 'array')),
    attributes TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(attributes) AND json_type(attributes) = 'object'),
    UNIQUE (content_hash, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_vector_key ON chunks(vector_key);

CREATE TRIGGER IF NOT EXISTS delete_unreferenced_chunks
AFTER DELETE ON documents
WHEN NOT EXISTS (SELECT 1 FROM documents WHERE content_hash = OLD.content_hash)
BEGIN
    DELETE FROM chunks WHERE content_hash = OLD.content_hash;
END;

CREATE TABLE IF NOT EXISTS index_config (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    chunk_size INTEGER NOT NULL CHECK (chunk_size > 0)
);

-- Singleton row that publishes the active USearch snapshot alongside chunk changes.
CREATE TABLE IF NOT EXISTS vector_index (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    format_version INTEGER NOT NULL CHECK (format_version > 0),
    file_name TEXT NOT NULL,
    model TEXT NOT NULL,
    model_digest TEXT NOT NULL CHECK (length(model_digest) > 0),
    chunk_format TEXT,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    -- Legacy column retained for compatibility with existing version 6 databases.
    removed_since_compact INTEGER NOT NULL DEFAULT 0 CHECK (removed_since_compact >= 0)
);
