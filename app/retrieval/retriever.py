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

from core.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, RETRIEVER_MODEL
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


ROUTING_SYSTEM_PROMPT = """You navigate a hierarchical documentation tree to answer a query. \
You are shown one level of the tree at a time and decide, per node, what to retrieve and \
where to descend. You never answer the query here — you only route.

Each node has two texts:
  - EVIDENCE — the node's OWN retrievable content. This is exactly what you get if you
    select the node. "(none ...)" means the node has no own content.
  - ROUTING — a summary of content that lives in the node's CHILDREN. This content is NOT
    inside this node; the ONLY way to retrieve it is to explore into the children.

For EACH node make TWO independent decisions:

1. evidence — Judge the EVIDENCE text ALONE. Set true ONLY if the EVIDENCE text DIRECTLY
   answers or materially contributes to answering the query — not merely because it is
   topically related, mentions a keyword, or gives background. If EVIDENCE is "(none ...)"
   or just a heading/title path with no substantive text, evidence MUST be false —
   selecting it would retrieve nothing useful. Do NOT set evidence true because of
   something you saw in ROUTING. When in doubt about evidence, prefer false.

   evidence_score — a number from 0.0 to 1.0: your confidence that this node's EVIDENCE
   text DIRECTLY helps answer the query (1.0 = clearly and directly answers it; 0.5 =
   partially relevant; 0.0 = not at all). Set 0.0 whenever evidence is false. This score
   ranks the final results, so be discriminating rather than clustering everything near
   the top. Reserve scores above 0.8 for text that states the answer outright.

2. explore — Judge the ROUTING text. Set true if ROUTING indicates the query's topic is
   covered anywhere in the children. That content lives in a child, not in this node, so
   exploring is the ONLY way to reach it: if ROUTING mentions or relates to the query's
   topic, you MUST set explore=true — do not assume the node already contains it. If
   ROUTING is "(none ...)" (a leaf), explore MUST be false.

The two decisions are independent: a node can be evidence AND explore (its intro answers
partly and its subsections add detail), evidence only, explore only, or neither. When
unsure whether to EXPLORE, lean toward true — deeper exploration can still be pruned, so
this protects recall. Hold EVIDENCE to a higher bar: prefer to omit a node whose own
content is only tangentially related.

Additional rules:

- COVER EVERY PART OF THE QUERY. If the query has several parts, asks "X and Y", or needs
  facts from more than one area of the documentation, make sure your explored set covers
  each part. Do not spend your whole explore budget on the single most obvious topic.
  A branch you fail to explore can never be recovered at a later level.
- PREFER CURRENT REFERENCE MATERIAL. Release notes, version announcements, changelogs and
  migration guides describe what changed in one past version and often name APIs that were
  later renamed or removed. Treat them as weak evidence: prefer the equivalent reference or
  guide page, and only select a release note as evidence when the query is itself about
  that version's history. Still explore them if nothing else covers the topic.
- Judge relevance to what the user is actually trying to do, including any constraint they
  state ("I'd rather not…", "without…", "instead of…"). A node describing an approach the
  user has explicitly ruled out is not good evidence.

Return ONLY valid JSON, with no prose, no explanation and no code fences, in exactly this
format:
{"decisions": [{"id": "<id>", "evidence": true, "evidence_score": 0.0, "explore": true}]}

Include a node ONLY if evidence or explore (or both) is true. OMIT any node where both are
false — never emit entries like {"id": "...", "evidence": false, "explore": false}. If no
node qualifies, return {"decisions": []}."""


ANSWER_SYSTEM_PROMPT = """You answer a question using only the documentation excerpts \
provided in the user message. The excerpts are numbered and each carries the title path of \
the section it came from.

Rules:

- GROUND EVERY CLAIM IN THE EXCERPTS. Do not introduce names, values, options or behaviour
  that do not appear there. If the excerpts genuinely do not answer the question, say so
  plainly and state what they do cover — but only after reading all of them, since the
  answer is often in a later excerpt rather than the first.
- BE SPECIFIC. Name the exact thing the reader has to use or change, rather than describing
  in general terms how the system behaves. Where the excerpts give a concrete form, show it.
- ANSWER EVERY PART OF THE QUESTION. When a question asks for more than one thing, address
  each part explicitly instead of answering only the most obvious one.
- RESPECT STATED CONSTRAINTS. When the question rules something out, or states a
  requirement the answer has to satisfy, do not recommend an approach that violates it even
  if the excerpts describe that approach. Choose the option that meets the constraint, and
  say why.
- FOLLOW THE DOCUMENTATION'S OWN RECOMMENDATION. If the excerpts present several supported
  approaches, lead with whichever the documentation itself calls recommended, preferred or
  built in, and mention the alternatives afterwards. Do not lead with a custom or
  hand-built approach when a supported one appears in the excerpts.
- WHEN SEVERAL APPROACHES GENUINELY APPLY, name them, say when each is the right choice,
  then commit to a recommendation for the case in the question.
- PREFER CURRENT MATERIAL OVER HISTORICAL. If an excerpt is a release note, changelog,
  announcement or migration guide, treat it as a record of one past version: what it names
  may since have changed. Do not present it as the current state unless another excerpt
  agrees. If it is your only source, say which version it describes.
- Be direct and concise. Do not preface the answer with remarks about the excerpts.
- If the answer is not present in the reference document respond with "I don't know" and nothing else."""


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
        model: str = RETRIEVER_MODEL,
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
        self._reset_usage()

    @staticmethod
    def _empty_bucket() -> dict:
        return {
            "cost": 0.0,
            "upstream_inference_cost": 0.0,
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _reset_usage(self) -> None:
        """
        Zero the per-question usage accumulators. Usage is split by call kind:
        ``routing`` (the one-per-level tree-navigation calls, of which there are
        as many as levels descended) and ``generation`` (the single final
        answer call), since the two have very different cost profiles.
        """
        self._usage = {
            "routing": self._empty_bucket(),
            "generation": self._empty_bucket(),
        }

    @property
    def last_usage(self) -> dict:
        """
        Cost/token usage of the most recent ``answer`` call, split into
        ``routing`` and ``generation`` buckets with the combined figures kept at
        the top level.
        """
        routing = dict(self._usage["routing"])
        generation = dict(self._usage["generation"])
        totals = {
            key: routing[key] + generation[key] for key in self._empty_bucket()
        }
        return {"model": self.model, **totals, "routing": routing, "generation": generation}

    def _track_usage(self, response, kind: str) -> None:
        """Accumulate OpenRouter usage accounting into the ``kind`` bucket."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        bucket = self._usage[kind]
        bucket["calls"] += 1
        bucket["prompt_tokens"] += usage.prompt_tokens or 0
        bucket["completion_tokens"] += usage.completion_tokens or 0
        bucket["total_tokens"] += usage.total_tokens or 0
        bucket["cost"] += getattr(usage, "cost", None) or 0.0
        cost_details = getattr(usage, "cost_details", None) or {}
        if not isinstance(cost_details, dict):  # pydantic extra model
            cost_details = dict(cost_details)
        bucket["upstream_inference_cost"] += (
            cost_details.get("upstream_inference_cost") or 0.0
        )

    def _chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 500,
        kind: str = "routing",
    ) -> str:
        """
        Single chat call with linear-backoff retries on transient errors.

        ``system`` carries the standing rules for the task and ``prompt`` carries only
        the volatile payload (the query and the candidate nodes / evidence). Keeping
        them apart raises instruction adherence and lets the provider cache the
        invariant half across the many calls one question makes.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=messages,
                    temperature=0.0,
                    # OpenRouter usage accounting: include cost in the response.
                    extra_body={"usage": {"include": True}},
                )
                self._track_usage(response, kind)
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

        prompt = f"""User Query: {query}

Select at most {max_explore} nodes to explore (the most promising branches), making sure
they cover every distinct part of the query.

Nodes:
{nodes_formatted}"""

        raw = self._chat(prompt, system=ROUTING_SYSTEM_PROMPT)

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
        answer from it. Returns ``(answer, contexts)``; the aggregated
        cost/token usage of all LLM calls made for this question is available
        afterwards via ``last_usage``.
        """
        self._reset_usage()
        rows = self.retrieve(
            query, store,
            max_explore=max_explore, max_results=max_results, max_level=max_level,
        )
        if not rows:
            return "No relevant documentation was found for this question.", []

        prompt = build_llm_context_from_rows(query, rows)
        contexts = [row[CONTENT] for row in rows]
        answer = self._chat(
            prompt,
            system=ANSWER_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            kind="generation",
        )
        return answer, contexts


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

    # Only the volatile half: the standing rules live in ANSWER_SYSTEM_PROMPT.
    return f"Question: {query}\n\nDocumentation excerpts:\n{context}"
