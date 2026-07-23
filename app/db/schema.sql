-- Schema for the hierarchical-RAG `nodes` table.
--
-- Matches db/db2.py (PostgresStore.insert_nodes_bulk + the LLM-selection
-- SELECTs). The LLM-guided retriever needs no embeddings; the `embedding`
-- column is kept optional/nullable for the (currently unused) vector path.
--
-- Apply with, e.g.:
--   psql "postgresql://rag:rag@localhost:5432/<db>" -f db/schema.sql

CREATE EXTENSION IF NOT EXISTS vector;   -- only needed if you use the embedding column

CREATE TABLE IF NOT EXISTS nodes (
    id                TEXT PRIMARY KEY,
    level             INTEGER,
    title             TEXT,
    title_path        TEXT,
    url               TEXT,
    canonical_url     TEXT,
    content           TEXT,

    -- Evidence side (this node's own content): drives the "is this evidence?" decision.
    evidence_summary  TEXT,
    evidence_keywords TEXT[],
    evidence_text     TEXT,

    -- Routing side (the child subtree): drives the "should I explore?" decision.
    routing_summary   TEXT,
    routing_keywords  TEXT[],
    routing_text      TEXT,

    parent_id         TEXT,
    children          TEXT[],
    anchor            TEXT,
    outgoing_links    TEXT[],

    embedding         vector(1024)      -- optional; unused by the LLM retriever
);

CREATE INDEX IF NOT EXISTS nodes_parent_id_idx ON nodes (parent_id);

-- ---------------------------------------------------------------------------
-- In-place migration from the OLD schema (summary / keywords / embedding_text)
-- instead of dropping the table. Run these once, then re-summarize + re-load:
--
--   ALTER TABLE nodes
--     ADD COLUMN IF NOT EXISTS evidence_summary  TEXT,
--     ADD COLUMN IF NOT EXISTS evidence_keywords TEXT[],
--     ADD COLUMN IF NOT EXISTS evidence_text     TEXT,
--     ADD COLUMN IF NOT EXISTS routing_summary   TEXT,
--     ADD COLUMN IF NOT EXISTS routing_keywords  TEXT[],
--     ADD COLUMN IF NOT EXISTS routing_text      TEXT;
--
--   -- optional cleanup of columns no longer written:
--   ALTER TABLE nodes
--     DROP COLUMN IF EXISTS summary,
--     DROP COLUMN IF EXISTS keywords,
--     DROP COLUMN IF EXISTS embedding_text;
-- ---------------------------------------------------------------------------
