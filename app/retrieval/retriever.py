"""LLM-guided hierarchical retriever (RAG, evidence-set variant).

Walks the node tree stored in Postgres level by level. At each node the LLM
makes TWO independent decisions from two decoupled texts:

  - **evidence?** — judged from the node's ``evidence_text`` (its own content):
    "if I retrieve THIS node, do I get something that helps answer the query?"
  - **explore?** — judged from the node's ``routing_text`` (its subtree):
    "is it worth descending into the children?"

Selected nodes are accumulated into an *evidence set* regardless of whether
their children are explored or selected — so a parent that holds the answer is
kept even when none of its children are relevant, and vice versa. No vector
search / embeddings are involved; every decision is made by the LLM.
"""

import json
import time

from openai import OpenAI

from core.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, SUMMARIZER_MODEL
from core.errors import ConfigError, RetrieverError

# Column indices for rows from get_roots_for_llm / get_children_for_llm:
# id, level, title, title_path, url, canonical_url, content,
# evidence_text, routing_text, anchor, parent_id
ID = 0
TITLE_PATH = 3
CONTENT = 6
EVIDENCE_TEXT = 7
ROUTING_TEXT = 8
ANCHOR = 9
PARENT_ID = 10


def _clamp_score(value) -> float:
    """Coerce an LLM-supplied relevance score into [0.0, 1.0]; 0.0 on garbage."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


class LLMRetriever:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = SUMMARIZER_MODEL,
        *,
        base_url: str = OPENROUTER_BASE_URL,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ):
        api_key = api_key or OPENROUTER_API_KEY
        if not api_key:
            raise ConfigError(
                "No OpenRouter API key. Set OPENROUTER_API_KEY (env/.env) "
                "or pass api_key=... explicitly."
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def _chat(self, prompt: str, *, max_tokens: int = 500) -> str:
        """Single chat call with linear-backoff retries on transient errors."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)

        raise RetrieverError(
            f"LLM selection request failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def _llm_evaluate_nodes(self, query, nodes, max_explore=7):
        """
        Ask the LLM, in a single call for the whole level, two independent
        decisions per node:

          - ``evidence``: does this node's OWN content (evidence_text) help
            answer the query?
          - ``explore``: do its subsections (routing_text) plausibly hold
            relevant info worth descending into?

        Returns ``{node_id: {"evidence": bool, "explore": bool}}`` for the
        nodes the LLM named; nodes it omits are treated as both-false.
        """
        if not nodes:
            return {}

        nodes_formatted = ""
        for row in nodes:
            title_path = (row[TITLE_PATH] or "").strip()
            evidence_text = (row[EVIDENCE_TEXT] or "").strip()
            routing_text = (row[ROUTING_TEXT] or "").strip()

            # evidence_text / routing_text both begin with the title path. When
            # nothing follows it, the node has no own content (evidence) / no
            # subtree info (routing) — show "(none)" so the LLM doesn't mistake
            # a bare heading for retrievable content.
            evidence_display = (
                evidence_text
                if evidence_text and evidence_text != title_path
                else "(none — organizational node, no own content)"
            )
            routing_display = (
                routing_text
                if routing_text and routing_text != title_path
                else "(none — leaf, no children)"
            )

            nodes_formatted += (
                f"ID: {row[ID]}\n"
                f"Title Path: {title_path}\n"
                f"EVIDENCE (this node's OWN retrievable content):\n{evidence_display}\n"
                f"ROUTING (content that lives in this node's CHILDREN):\n{routing_display}\n\n"
            )

        prompt = f"""You are navigating a hierarchical documentation tree to answer a query.

User Query: {query}

Each node has two texts:
  - EVIDENCE — the node's OWN retrievable content. This is exactly what you get
    if you select the node. "(none ...)" means the node has no own content.
  - ROUTING — a summary of content that lives in the node's CHILDREN. This
    content is NOT inside this node; the ONLY way to retrieve it is to explore
    into the children.

For EACH node make TWO independent decisions:

1. evidence — Judge the EVIDENCE text ALONE. Set true ONLY if the EVIDENCE text
   DIRECTLY answers or materially contributes to answering the query — not
   merely because it is topically related, mentions a keyword, or gives
   background. If EVIDENCE is "(none ...)" or just a heading/title path with no
   substantive text, evidence MUST be false — selecting it would retrieve
   nothing useful. Do NOT set evidence true because of something you saw in
   ROUTING. When in doubt about evidence, prefer false.

   evidence_score — a number from 0.0 to 1.0: your confidence that this node's
   EVIDENCE text DIRECTLY helps answer the query (1.0 = clearly and directly
   answers it; 0.5 = partially relevant; 0.0 = not at all). Set 0.0 whenever
   evidence is false. This score is used to rank the final results, so be
   discriminating rather than clustering everything near the top.

2. explore — Judge the ROUTING text. Set true if ROUTING indicates the query's
   topic is covered anywhere in the children. That content lives in a child, not
   in this node, so exploring is the ONLY way to reach it: if ROUTING mentions or
   relates to the query's topic, you MUST set explore=true — do not assume the
   node already contains it. If ROUTING is "(none ...)" (a leaf), explore MUST be
   false.

The two decisions are independent: a node can be evidence AND explore (its intro
answers partly and its subsections add detail), evidence only, explore only, or
neither. When unsure whether to EXPLORE, lean toward true — deeper exploration
can still be pruned, so this protects recall. Hold EVIDENCE to a higher bar:
prefer to omit a node whose own content is only tangentially related.

Select at most {max_explore} nodes to explore (the most promising branches).

Include a node in the output ONLY if evidence or explore (or both) is true.
OMIT any node where both are false — do not emit entries like
{{"id": "...", "evidence": false, "explore": false}}. If no node qualifies,
return an empty list.

Nodes:
{nodes_formatted}
Return ONLY valid JSON in this exact format, with no other text:
{{"decisions": [{{"id": "<id>", "evidence": true, "evidence_score": 0.0, "explore": true}}, ...]}}"""

        raw = self._chat(prompt)

        try:
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())
            decisions = {}
            for item in parsed.get("decisions", []):
                node_id = str(item.get("id"))
                evidence = bool(item.get("evidence", False))
                decisions[node_id] = {
                    "evidence": evidence,
                    # Only trust a score when evidence is asserted; otherwise 0.0.
                    "evidence_score": _clamp_score(item.get("evidence_score", 0.0)) if evidence else 0.0,
                    "explore": bool(item.get("explore", False)),
                }
            return decisions
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            return {}

    @staticmethod
    def _has_own_content(row) -> bool:
        """
        True only if the node carries substantive content of its own — i.e. the
        ``content`` that would actually be handed to the generator is non-empty.
        Organizational/heading-only nodes (empty ``content``) contribute nothing
        retrievable, so they are hard-gated out of the evidence set regardless
        of what the LLM decided.
        """
        return bool((row[CONTENT] or "").strip())

    def _search_level(self, nodes, level, query,
                      store, evidence, visited,
                      max_explore, max_level):
        """
        Evaluate one level in a single LLM call. Nodes marked ``evidence`` (and
        which actually carry own content) are appended to the evidence pool as
        ``(score, row)`` pairs, regardless of whether they are explored or any
        descendant is selected; the children of nodes marked ``explore`` are
        pooled and the search descends into them.

        The pool is intentionally uncapped here — ranking and the final cap are
        applied once in ``retrieve`` so the top-k is chosen globally rather than
        greedily by traversal order.
        """
        if level > max_level or not nodes:
            return

        unvisited = [row for row in nodes if str(row[ID]) not in visited]
        if not unvisited:
            return

        decisions = self._llm_evaluate_nodes(query, unvisited, max_explore=max_explore)
        if not decisions:
            return

        id_to_row = {str(row[ID]): row for row in unvisited}

        next_level = []
        for node_id, decision in decisions.items():
            row = id_to_row.get(node_id)
            if row is None:
                continue
            visited.add(node_id)

            if decision["evidence"] and self._has_own_content(row):
                evidence.append((decision["evidence_score"], row))

            if decision["explore"]:
                next_level.extend(store.get_children_for_llm(node_id))

        self._search_level(
            next_level, level + 1, query,
            store, evidence, visited,
            max_explore, max_level,
        )

    def retrieve(self, query, store,
                 max_explore=7, max_results=7, max_level=6):
        """
        Evidence-set retrieval with LLM-based, decoupled node decisions.
        No embedder needed — all decisions by LLM using evidence_text /
        routing_text.

        The full traversal collects an ``(score, row)`` pool; it is then ranked
        by the LLM's relevance score (descending) and capped to ``max_results``.
        Ties keep traversal order (shallower nodes first) — Python's sort is
        stable and the pool is built shallow-first. No extra LLM call: the score
        rides along on the per-level evaluation that already happened.
        """
        root_nodes = store.get_roots_for_llm()

        evidence = []
        visited = set()

        self._search_level(
            root_nodes, 0, query,
            store, evidence, visited,
            max_explore, max_level,
        )

        ranked = sorted(evidence, key=lambda pair: pair[0], reverse=True)
        return [row for _score, row in ranked[:max_results]]

    def answer(self, query, store,
               max_explore=7, max_results=7, max_level=6,
               max_tokens=1500):
        """
        End-to-end RAG: collect the evidence set, then generate the final
        answer from it. Returns ``(answer, contexts)``.
        """
        rows = self.retrieve(
            query, store,
            max_explore=max_explore, max_results=max_results, max_level=max_level,
        )
        if not rows:
            return "No relevant documentation was found for this question.", []

        prompt = build_llm_context_from_rows(query, rows)
        contexts = [row[CONTENT] for row in rows]
        return self._chat(prompt, max_tokens=max_tokens), contexts


def build_llm_context_from_rows(query, evidence_rows):
    """
    Build the generation prompt from the evidence set. Each evidence node
    contributes its own ``content`` (for an internal node, that is the section's
    intro text), so answers that live at any level of the hierarchy are present
    — no separate parent bubble-up is needed.
    """
    segments = []
    for rank, row in enumerate(evidence_rows, 1):
        title_path = (row[TITLE_PATH] or "").strip()
        content = (row[CONTENT] or "").strip()
        anchor = (row[ANCHOR] or "").strip()
        block = f"[{rank}] {title_path}\nAnchor: {anchor}\n{content}"
        segments.append(block)

    context = "\n\n---\n\n".join(segments)

    return (
        f"Answer the following question using only the provided context.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:"
    )
