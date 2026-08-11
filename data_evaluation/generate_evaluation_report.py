"""Compare RAGAS scores across RAG types, averaged per question type.

For each retriever:
  questions_<retriever>.json    -> the `category` (question type) per question,
                                   plus the `llm_cost` breakdown of the run
  ragas_results_<retriever>.xlsx -> per-question metric scores + retrieved_contexts

Output: evaluation_report.xlsx, a single sheet with one row per
(question type, RAG type) so the RAGs sit side by side for comparison, plus the
average character length of the retrieved contexts and the average LLM
cost/token usage.

Cost is `upstream_inference_cost` (the `cost` field is 0 on the free tier). It is
reported three ways: total, routing (the tree-walk calls) and generation (the
final answer call). Only the LLM retriever spends tokens on routing, so
routing_* is blank for the single-shot retrievers; for those, generation_*
equals the total by construction.
"""

import ast
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "django-rest-framework"
OUTPUT_PATH = Path(__file__).resolve().parent / "evaluation_report.xlsx"

# retriever key -> label shown in the report.
RETRIEVERS = {"llm_retriever": "LLM Retriever", "naive_rag": "Naive RAG", "raptor": "RAPTOR"}

METRICS = [
    "context_precision",
    "context_recall",
    "helpfulness",
    "answer_accuracy",
    "factual_correctness",
    "rouge_score(mode=precision)",
    "rouge_score(mode=recall)",
]
COST_COLS = [
    "llm_calls",
    "total_cost",
    "total_tokens",
    "routing_cost",
    "routing_tokens",
    "generation_cost",
    "generation_tokens",
]
CATEGORY_ORDER = ["straightforward", "multi_hop", "intent_driven", "debugging"]


def context_length(raw):
    """retrieved_contexts is a stringified list; total character length of its items."""
    try:
        return sum(len(str(c)) for c in ast.literal_eval(raw))
    except (ValueError, SyntaxError):
        return 0


def cost_breakdown(usage):
    """Flatten one question's `llm_cost` into the routing / generation split.

    Single-shot retrievers have no `routing` sub-dict: their routing columns stay
    NaN (blank in Excel) and the whole spend is attributed to generation.
    """
    routing = usage.get("routing")
    generation = usage.get("generation", usage)
    return {
        "llm_calls": usage.get("calls"),
        "total_cost": usage.get("upstream_inference_cost"),
        "total_tokens": usage.get("total_tokens"),
        "routing_cost": routing.get("upstream_inference_cost") if routing else float("nan"),
        "routing_tokens": routing.get("total_tokens") if routing else float("nan"),
        "generation_cost": generation.get("upstream_inference_cost"),
        "generation_tokens": generation.get("total_tokens"),
    }


def load(retriever):
    questions = json.loads((DATA_DIR / f"questions_{retriever}.json").read_text())
    cats = {q["user_input"]: q["category"] for q in questions}
    costs = {q["user_input"]: cost_breakdown(q["llm_cost"]) for q in questions}

    df = pd.read_excel(DATA_DIR / f"ragas_results_{retriever}.xlsx")
    df["question_type"] = df["user_input"].map(cats)
    df["context_length"] = df["retrieved_contexts"].map(context_length)
    for col in COST_COLS:
        df[col] = df["user_input"].map(lambda q, c=col: costs.get(q, {}).get(c, float("nan")))
    df["rag"] = RETRIEVERS[retriever]
    return df


rows = pd.concat([load(r) for r in RETRIEVERS], ignore_index=True)

report = (
    rows.groupby(["question_type", "rag"])[METRICS + ["context_length"] + COST_COLS]
    .mean()
    .reset_index()
)
# costs are fractions of a cent, so they need more decimals than the 0-1 metrics.
report[METRICS + ["context_length"]] = report[METRICS + ["context_length"]].round(4)
report[COST_COLS] = report[COST_COLS].round(6)
report["question_type"] = pd.Categorical(report["question_type"], CATEGORY_ORDER, ordered=True)
report = report.sort_values(["question_type", "rag"]).reset_index(drop=True)

report.to_excel(OUTPUT_PATH, index=False, sheet_name="RAG Comparison")
print(report.to_string(index=False))
print(f"\nWrote report -> {OUTPUT_PATH}")
