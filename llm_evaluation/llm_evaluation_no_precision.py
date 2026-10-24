"""Re-score a long-context RAGAS spreadsheet (e.g. full KETRAG) without context_precision.

Same as llm_evaluation.py (same model, concurrency, checkpointing and in-place
write-back), with two differences for retrievers that return many chunks (~56):

* ``context_precision`` is skipped: it makes one judge call per chunk and is
  rank-weighted, so with 56 chunks it is costly and mostly reflects the tail.
* ``retrieved_contexts`` is read from the source questions JSON, not the xlsx:
  Excel truncates cells at 32,767 characters, which cuts the list mid-chunk and
  would leave ``context_recall`` scoring a single broken string.

    uv run --project .. python llm_evaluation_no_precision.py --xlsx questions_ketrag_full.xlsx
    # JSON defaults to the xlsx name with .json; override with --json path/to/file.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

import llm_evaluation

# rescore() reads this module-level list at call time.
llm_evaluation.SCORE_COLUMNS = [
    column for column in llm_evaluation.SCORE_COLUMNS  if column not in ["context_precision","context_recall"]
]


def _use_full_contexts(json_path: Path) -> None:
    """Swap the truncated xlsx retrieved_contexts for the JSON's llm_references."""
    questions = json.loads(json_path.read_text())
    full_contexts = {
        q["user_input"]: [str(c) for c in q["llm_references"]] for q in questions
    }
    compute = llm_evaluation._compute

    async def _compute(column: str, row: pd.Series) -> float:
        user_input = str(row.get("user_input", "") or "")
        if user_input not in full_contexts:
            raise KeyError(f"question not found in {json_path.name}: {user_input[:60]!r}")
        row = row.copy()
        row["retrieved_contexts"] = full_contexts[user_input]
        return await compute(column, row)

    # rescore() looks _compute up in the module at call time.
    llm_evaluation._compute = _compute
    print(f"Using full retrieved_contexts from {json_path} ({len(full_contexts)} questions)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--xlsx", required=True, help="Path to the .xlsx evaluation file to re-score")
    parser.add_argument("--json", help="Questions JSON with full llm_references (default: xlsx path with .json)")
    args = parser.parse_args()

    path = Path(args.xlsx)
    json_path = Path(args.json) if args.json else path.with_suffix(".json")
    for p in (path, json_path):
        if not p.exists():
            raise SystemExit(f"File not found: {p}")

    _use_full_contexts(json_path)
    asyncio.run(llm_evaluation.rescore(path))


if __name__ == "__main__":
    main()
