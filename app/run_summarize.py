"""Phase 3 — summarize: hierarchical summaries + keywords -> checkpoint.

Reads the checkpoint from phase 2, summarizes every node (children first), and
checkpoints after each one (resumable). Uses the SUMMARIZER model, which runs on
OpenRouter — needs OPENROUTER_API_KEY in the env.
    python run_summarize.py
"""

from core.checkpoint import load_checkpoint
from summarization.pipeline import hierarchical_summarization_pipeline
from summarization.summarizer import GroqSummarizer
from summarization.tree import post_order_traversal

# Model id from SUMMARIZER_MODEL, key from OPENROUTER_API_KEY (env / .env).
summarizer = GroqSummarizer()

nodes_by_id = load_checkpoint()
if not nodes_by_id:
    raise SystemExit("No checkpoint found. Run run_parse.py first.")

ordered_nodes = post_order_traversal(nodes_by_id)

hierarchical_summarization_pipeline(
    ordered_nodes=ordered_nodes,
    nodes_by_id=nodes_by_id,
    summarizer=summarizer,
)
print("Summarization complete -> checkpoint")
