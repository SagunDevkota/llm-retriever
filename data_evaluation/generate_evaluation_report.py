"""Compare RAGAS scores across RAG types, averaged per question type.

For each retriever:
  questions_<retriever>.json    -> the `category` (question type) per question
  ragas_results_<retriever>.xlsx -> per-question metric scores + retrieved_contexts

Output: evaluation_report.xlsx, a single sheet with one row per
(question type, RAG type) so the RAGs sit side by side for comparison, plus the
average character length of the retrieved contexts.
"""

import ast
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "playwright"
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
CATEGORY_ORDER = ["straightforward", "multi_hop", "intent_driven", "debugging"]


def context_length(raw):
    """retrieved_contexts is a stringified list; total character length of its items."""
    try:
        return sum(len(str(c)) for c in ast.literal_eval(raw))
    except (ValueError, SyntaxError):
        return 0


def load(retriever):
    cats = {q["user_input"]: q["category"]
            for q in json.loads((DATA_DIR / f"questions_{retriever}.json").read_text())}
    df = pd.read_excel(DATA_DIR / f"ragas_results_{retriever}.xlsx")
    df["question_type"] = df["user_input"].map(cats)
    df["context_length"] = df["retrieved_contexts"].map(context_length)
    df["rag"] = RETRIEVERS[retriever]
    return df


rows = pd.concat([load(r) for r in RETRIEVERS], ignore_index=True)

report = (
    rows.groupby(["question_type", "rag"])[METRICS + ["context_length"]]
    .mean()
    .round(4)
    .reset_index()
)
report["question_type"] = pd.Categorical(report["question_type"], CATEGORY_ORDER, ordered=True)
report = report.sort_values(["question_type", "rag"]).reset_index(drop=True)

report.to_excel(OUTPUT_PATH, index=False, sheet_name="RAG Comparison")
print(report.to_string(index=False))
print(f"\nWrote report -> {OUTPUT_PATH}")
