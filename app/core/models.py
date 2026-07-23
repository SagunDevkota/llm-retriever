"""Data models shared across the pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel


@dataclass
class Node:
    """A single section of a documentation page in the hierarchical tree.

    Level 0 nodes are whole pages (``parent_id is None``); deeper levels mirror
    the heading structure (h1..h6) of the source HTML.
    """

    # -------------------------
    # Identity
    # -------------------------
    id: str
    parent_id: Optional[str]
    # -------------------------
    # Structure
    # -------------------------
    level: int
    title: str
    title_path: str
    children: List[str]
    # -------------------------
    # Raw document data
    # -------------------------
    url: str
    canonical_url: str
    content: str
    # -------------------------
    # Generated hierarchical data
    # -------------------------
    # Two decoupled summaries (see the summarizer):
    #  - evidence_* describes THIS node's own content — "if I retrieve this
    #    node, what do I actually get?" — and drives the evidence decision.
    #  - routing_* describes only the CHILD subtree — "is it worth descending?"
    #    — and drives the explore/routing decision. Empty for leaves.
    evidence_summary: Optional[str] = ""
    evidence_keywords: List[str] = field(default_factory=list)
    routing_summary: Optional[str] = ""
    routing_keywords: List[str] = field(default_factory=list)
    # -------------------------
    # Retrieval data
    # -------------------------
    # Concatenated text the retriever actually reads. The *_summary / *_keywords
    # above are kept only for traceability / rebuilding these without re-calling
    # the LLM.
    evidence_text: Optional[str] = None
    routing_text: Optional[str] = None
    # -------------------------
    # Metadata
    # -------------------------
    metadata: dict = field(default_factory=dict)


class SummaryResponse(BaseModel):
    """Structured output expected from the summarization LLM.

    A single call returns both summaries: ``evidence_*`` derived from the node's
    own content, ``routing_*`` derived only from the child context (empty when
    the node has no children).
    """

    evidence_summary: str
    evidence_keywords: List[str]
    routing_summary: str = ""
    routing_keywords: List[str] = []
