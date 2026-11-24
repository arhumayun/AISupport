CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_releases (
    release_id text PRIMARY KEY,
    audience text NOT NULL CHECK (audience IN ('student', 'staff')),
    manifest jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_documents (
    release_id text NOT NULL REFERENCES rag_releases(release_id) ON DELETE CASCADE,
    document_id text NOT NULL,
    audience text NOT NULL CHECK (audience IN ('student', 'staff')),
    department text NOT NULL,
    source text NOT NULL,
    source_version text NOT NULL,
    content_hash text NOT NULL,
    approval_status text NOT NULL,
    review_date date NOT NULL,
    effective_date date NOT NULL,
    access_level text NOT NULL CHECK (access_level IN ('student', 'staff')),
    PRIMARY KEY (release_id, document_id),
    CHECK (audience = access_level)
);

CREATE TABLE IF NOT EXISTS rag_chunks (
    release_id text NOT NULL,
    chunk_id text NOT NULL,
    document_id text NOT NULL,
    audience text NOT NULL CHECK (audience IN ('student', 'staff')),
    department text NOT NULL,
    source text NOT NULL,
    source_version text NOT NULL,
    access_level text NOT NULL CHECK (access_level IN ('student', 'staff')),
    ordinal integer NOT NULL,
    content text NOT NULL,
    content_hash text NOT NULL,
    embedding vector(384) NOT NULL,
    textsearch tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    PRIMARY KEY (release_id, chunk_id),
    FOREIGN KEY (release_id, document_id)
        REFERENCES rag_documents(release_id, document_id) ON DELETE CASCADE,
    CHECK (audience = access_level)
);

CREATE INDEX IF NOT EXISTS rag_chunks_policy_idx
    ON rag_chunks (release_id, audience, department, access_level);
CREATE INDEX IF NOT EXISTS rag_chunks_textsearch_idx
    ON rag_chunks USING gin (textsearch);
CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);
