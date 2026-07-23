"""Hierarchical summarization pipeline.

Walks ``ordered_nodes`` (expected in post-order so children precede parents),
summarizes each node with its children's summaries as context, builds the
``evidence_text`` / ``routing_text``, and checkpoints after every success so the
run is resumable.
"""

import time

import tqdm

from core.checkpoint import save_checkpoint


def hierarchical_summarization_pipeline(ordered_nodes, nodes_by_id, summarizer):
    for node in tqdm.tqdm(ordered_nodes):

        # -------------------------------------------------
        # SKIP already processed nodes (resume support)
        # -------------------------------------------------
        if node.get("evidence_text"):
            print(f"Skipping: {node['title']}")
            continue

        # CHILD CONTEXT is built from each child's evidence_text + routing_text
        # (never its raw content). Those two fields already hold
        # title_path + summary + keywords for each side, so this carries the
        # child's own evidence AND its rolled-up subtree. Because a child's
        # routing was itself built from ITS children on the earlier post-order
        # pass, feeding only immediate children transitively rolls the whole
        # subtree's evidence up to this node.
        child_context_parts = []

        for child_id in node["children"]:
            child = nodes_by_id[child_id]

            child_parts = [
                child.get("evidence_text", ""),
                child.get("routing_text", ""),
            ]
            child_block = "\n\n".join(part for part in child_parts if part)
            if child_block:
                child_context_parts.append(child_block)

        child_context = "\n\n---\n\n".join(child_context_parts)

        try:
            result = summarizer.summarize(node=node, child_context=child_context)

            node["evidence_summary"] = result.evidence_summary
            node["evidence_keywords"] = result.evidence_keywords
            node["routing_summary"] = result.routing_summary
            node["routing_keywords"] = result.routing_keywords

            title_path = node.get("title_path", "")

            # Output texts stay strictly separated: evidence_text never carries
            # routing fields and vice versa.
            evidence_parts = [
                title_path,
                result.evidence_summary,
                ", ".join(result.evidence_keywords),
            ]
            node["evidence_text"] = "\n".join(
                part for part in evidence_parts if part
            )

            routing_parts = [
                title_path,
                result.routing_summary,
                ", ".join(result.routing_keywords),
            ]
            node["routing_text"] = "\n".join(
                part for part in routing_parts if part
            )

            # Save checkpoint after every success.
            save_checkpoint(nodes_by_id)

            print(f"Processed: {node['title']}")
            time.sleep(0.5)

        except Exception as e:
            print(f"FAILED: {node['title']}")
            print(e)

            # Save partial progress before propagating.
            save_checkpoint(nodes_by_id)
            raise
