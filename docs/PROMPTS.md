# Prompts

The three prompts the system uses: one to summarize each node, one to navigate the tree, and
one to generate the answer. Text is verbatim from the source files linked under each heading.
Placeholders in double angle brackets (`<<query>>`) are filled in at call time.

These are the prompts published as Online Resource 1 of *Structure-Grounded Retrieval-Augmented
Generation for Technical Documentation Using Bottom-Up Summarization and LLM Tree Traversal*
(Devkota and Baral, Journal of Intelligent Information Systems). The PDF typesets em dashes as
hyphens and "…" as "...", but the wording is identical.

| Prompt | Model role | Called | Source |
|---|---|---|---|
| [A. Summarization](#a-summarization-prompt) | `summarizer` | Once per node, children before parents | [app/summarization/summarizer.py](../app/summarization/summarizer.py) |
| [B. Traversal](#b-traversal-prompt) | `routing` | Once per tree level | [app/retrieval/retriever.py](../app/retrieval/retriever.py) |
| [C. Answer generation](#c-answer-generation-prompt) | `answer` | Once per query | [app/retrieval/retriever.py](../app/retrieval/retriever.py) |

## A. Summarization prompt

Sent once per node during the post-order pass (stage 3 of the pipeline), so every child is
summarized before its parent. The request uses a JSON response format so the four fields come
back parsed.

`<<child_evidence_and_routing_texts>>` is the children's already-generated evidence and routing
texts, never their raw content. It is empty for leaf nodes, which is why the routing fields must
come back empty for them.

```text
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
{
    "evidence_summary": str,
    "evidence_keywords": list[str],
    "routing_summary": str,
    "routing_keywords": list[str]
}

TITLE PATH:
<<title_path>>

CURRENT SECTION TITLE:
<<title>>

CURRENT SECTION CONTENT:
<<content>>

CHILD CONTEXT:
<<child_evidence_and_routing_texts>>
```

## B. Traversal prompt

A fixed system prompt, plus a user prompt rebuilt for each tree level from every unvisited node
at that level.

**System prompt**

```text
You navigate a hierarchical documentation tree to answer a query. You are shown one level of the tree at a time and decide, per node, what to retrieve and where to descend. You never answer the query here — you only route.

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
node qualifies, return {"decisions": []}.
```

**User prompt**

The explore limit is 7 by default (`max_explore`).

```text
User Query: <<query>>

Select at most 7 nodes to explore (the most promising branches), making sure
they cover every distinct part of the query.

Nodes:
<<nodes>>
```

Each entry in `<<nodes>>` looks like this:

```text
ID: <<id>>
Title Path: <<title_path>>
EVIDENCE (this node's OWN retrievable content):
<<evidence_text>>
ROUTING (content that lives in this node's CHILDREN):
<<routing_text>>
```

When a node has no content of its own, the evidence line reads
`(none — organizational node, no own content)`. A leaf node's routing line reads
`(none — leaf, no children)`.

## C. Answer generation prompt

A fixed system prompt, plus a user prompt built once traversal has finished and the evidence
pool has been ranked and capped (7 nodes by default, `max_results`). Every baseline uses this
same pair, so differences between answers come from what was retrieved, not from how the
question was asked.

**System prompt**

```text
You answer a question using only the documentation excerpts provided in the user message. The excerpts are numbered and each carries the title path of the section it came from.

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
- If the answer is not present in the reference document respond with "I don't know" and nothing else.
```

**User prompt**

```text
Question: <<query>>

Documentation excerpts:
<<context>>
```

`<<context>>` is the selected nodes' raw content, ranked and separated by `---`. Each excerpt
looks like this:

```text
[<<rank>>] <<title_path>>
Anchor: <<anchor>>
<<content>>
```
