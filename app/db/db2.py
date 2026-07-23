import psycopg2
from psycopg2.extras import execute_values
from typing import List

from core.errors import ConfigError


class PostgresStore:
    def __init__(self, dsn: str | None = None):
        # The DSN (which names the target database) must be supplied
        # explicitly — there is no silent fallback to a default database, so a
        # missing/empty DSN fails loudly instead of writing to the wrong place.
        if not dsn:
            raise ConfigError(
                "No Postgres DSN provided. Pass dsn=... (including the database "
                "name), e.g. postgresql://rag:rag@localhost:5432/ragdb, or set "
                "POSTGRES_DSN in the env / .env."
            )
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True

    def insert_nodes_bulk(self, nodes_by_id: dict):
        node_ids = list(nodes_by_id.keys())
        nodes = [nodes_by_id[nid] for nid in node_ids]

        rows = [
            (
                n["id"],
                n["level"],
                n["title"],
                n["title_path"],
                n["url"],
                n["canonical_url"],
                n["content"],
                n.get("evidence_summary", ""),
                n.get("evidence_keywords", []),
                n.get("evidence_text"),
                n.get("routing_summary", ""),
                n.get("routing_keywords", []),
                n.get("routing_text"),
                n.get("parent_id"),
                n.get("children", []),
                n.get("metadata", {}).get("anchor"),
                n.get("metadata", {}).get("outgoing_links", []),
            )
            for n in nodes
        ]

        with self.conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO nodes (
                    id, level, title, title_path, url, canonical_url,
                    content, evidence_summary, evidence_keywords, evidence_text,
                    routing_summary, routing_keywords, routing_text,
                    parent_id, children, anchor, outgoing_links
                )
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    content          = EXCLUDED.content,
                    evidence_summary = EXCLUDED.evidence_summary,
                    evidence_keywords = EXCLUDED.evidence_keywords,
                    evidence_text    = EXCLUDED.evidence_text,
                    routing_summary  = EXCLUDED.routing_summary,
                    routing_keywords = EXCLUDED.routing_keywords,
                    routing_text     = EXCLUDED.routing_text
                """,
                rows,
                page_size=1000,
            )
    
    def get_roots_for_llm(self) -> list:
        """
        Fetch all root nodes with evidence_text + routing_text for LLM
        selection. No embedding needed — the LLM reads those texts directly.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    level,
                    title,
                    title_path,
                    url,
                    canonical_url,
                    content,
                    evidence_text,
                    routing_text,
                    anchor,
                    parent_id
                FROM nodes
                WHERE parent_id IS NULL
                ORDER BY id
                """
            )
            return cur.fetchall()

    def get_children_for_llm(self, parent_id: str) -> list:
        """
        Fetch children with evidence_text + routing_text for LLM selection.
        No embedding needed — the LLM reads those texts directly.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    level,
                    title,
                    title_path,
                    url,
                    canonical_url,
                    content,
                    evidence_text,
                    routing_text,
                    anchor,
                    parent_id
                FROM nodes
                WHERE parent_id = %s
                ORDER BY id
                """,
                (parent_id,),
            )
            return cur.fetchall()

    def get_node_by_id(self, node_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, level, title, title_path, url, canonical_url,
                       content, evidence_summary, evidence_keywords,
                       routing_summary, routing_keywords, parent_id, children,
                       anchor, outgoing_links
                FROM nodes WHERE id = %s
                """,
                (node_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            keys = [
                "id", "level", "title", "title_path", "url", "canonical_url",
                "content", "evidence_summary", "evidence_keywords",
                "routing_summary", "routing_keywords", "parent_id", "children",
                "anchor", "outgoing_links",
            ]
            return dict(zip(keys, row))

    def get_children(self, parent_id: str) -> list:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, level, url FROM nodes WHERE parent_id = %s",
                (parent_id,),
            )
            return cur.fetchall()

    def close(self):
        self.conn.close()