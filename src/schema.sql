CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    attributes TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(attributes) AND json_type(attributes) = 'object')
);

CREATE INDEX IF NOT EXISTS documents_content_hash ON documents(content_hash);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    vector_key INTEGER NOT NULL UNIQUE CHECK (vector_key > 0),
    content_hash TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    section_path TEXT
        CHECK (section_path IS NULL OR
            (json_valid(section_path) AND json_type(section_path) = 'array')),
    attributes TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(attributes) AND json_type(attributes) = 'object'),
    UNIQUE (content_hash, ordinal)
);

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
    file_name TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    next_key INTEGER NOT NULL CHECK (next_key > 0)
);
