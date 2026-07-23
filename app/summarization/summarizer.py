"""LLM-backed hierarchical summarizer (OpenRouter, OpenAI-compatible)."""

import json
import time

from openai import OpenAI

from core.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, SUMMARIZER_MODEL
from core.errors import ConfigError, SummarizationError
from core.models import SummaryResponse


def build_prompt(node, child_context=""):
    """Builds the dual-summary hierarchical summarization prompt.

    A single call produces two decoupled summaries from two disjoint inputs:
      - ``evidence_*`` from CURRENT SECTION CONTENT only (what this node yields
        if retrieved), and
      - ``routing_*`` from CHILD CONTEXT only (what lives deeper, for deciding
        whether to descend). CHILD CONTEXT is the children's summaries — never
        their raw content — and is empty for leaves, so the routing fields must
        come back empty in that case.
    """

    content = node.get("content", "").strip()

    return f"""
You are generating hierarchical retrieval metadata for a documentation RAG system.

You are given two SEPARATE inputs for one node:
  - CURRENT SECTION CONTENT: the text that belongs directly to this heading.
  - CHILD CONTEXT: summaries of this node's subsections (its descendants).

Produce TWO independent, non-overlapping summaries:

1. evidence_summary / evidence_keywords
   - Derived ONLY from CURRENT SECTION CONTENT.
   - Describes what a reader obtains if THIS node's own content is retrieved.
   - Must NOT include information that appears only in CHILD CONTEXT.

2. routing_summary / routing_keywords
   - Derived ONLY from CHILD CONTEXT.
   - Describes what the subsections collectively cover, to decide whether it is
     worth exploring deeper.
   - Must NOT restate this node's own content.
   - If CHILD CONTEXT is empty, return "" and [] for these two fields.

Guidelines:
- Preserve technical meaning; include important concepts/tools/features.
- Keywords should help semantic retrieval; avoid generic keywords.
- Summaries should be concise but information dense.
- Do NOT hallucinate. Do NOT cross-contaminate the two summaries.
- Respond in the following json structure:
{{
    "evidence_summary": str,
    "evidence_keywords": list[str],
    "routing_summary": str,
    "routing_keywords": list[str]
}}

TITLE PATH:
{node.get("title_path", "")}

CURRENT SECTION TITLE:
{node.get("title", "")}

CURRENT SECTION CONTENT:
{content}

CHILD CONTEXT:
{child_context}
"""


class GroqSummarizer:
    """Calls an OpenRouter chat model and parses structured JSON output.

    Transient API errors and unparseable responses are retried with a simple
    linear backoff before being surfaced as :class:`SummarizationError`.
    """

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

    def summarize(self, node, child_context="") -> SummaryResponse:
        prompt = build_prompt(node, child_context)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You generate structured summaries "
                                "for hierarchical RAG systems."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
                raw = completion.choices[0].message.content
                data = json.loads(raw)
                return SummaryResponse(**data)
            except Exception as e:  # API error, JSON error, or schema mismatch
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
                    continue

        raise SummarizationError(
            f"Summarization failed after {self.max_retries} attempts "
            f"for node {node.get('id')!r}: {last_error}"
        ) from last_error
