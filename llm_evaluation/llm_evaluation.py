"""Re-score a RAGAS evaluation spreadsheet with the ragas 0.4.x API.

Reads an .xlsx (path passed via CLI), and for every row that is missing a score
recomputes it in place. Columns that don't exist yet are created. The workbook is
written back to the same file.

Migration note (ragas v0.3 -> v0.4):
    * metrics now live in ``ragas.metrics.collections``
    * metrics are scored with ``await metric.ascore(**kwargs)`` (not
      ``single_turn_ascore(sample)``) and return a ``MetricResult`` whose numeric
      value is ``result.value``.

Scores computed:
    answer_accuracy, factual_correctness, rouge_score(mode=precision),
    rouge_score(mode=recall), context_precision, context_recall

The LLM (model id + api key) is read from the environment / .env only -- nothing
is hardcoded. No embedding model is used; every metric here is LLM-only or
lexical (rouge), so no embeddings are required.

    uv run --project .. python llm_evaluation.py results.xlsx
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# --- ragas 0.4.x metrics (collections API) ---------------------------------
from ragas.metrics.collections import (
    AnswerAccuracy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    RougeScore,
)
from ragas.metrics import RubricsScore  # legacy metric (scored via single_turn_ascore)
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory


# --- Configuration: model + key come from .env only ------------------------
# The model registry lives in app/core/config.py; app/ is put on sys.path so the
# EVALUATION model is defined in exactly one place alongside the other three.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "app"))

from core.config import init_model  # noqa: E402  (needs the sys.path line above)

# Max simultaneous metric computations (LLM calls). Tune down if you hit rate
# limits, up if the provider allows more throughput.
MAX_CONCURRENCY = int(os.environ.get("EVAL_MAX_CONCURRENCY", "5"))
# Flush partial results to disk after this many computed cells (crash-resume).
CHECKPOINT_EVERY = int(os.environ.get("EVAL_CHECKPOINT_EVERY", "5"))

# Async client so the metrics' ``agenerate`` calls work.
_client, EVALUATION_MODEL = init_model("evaluation", async_client=True)
evaluator_llm = llm_factory(
    EVALUATION_MODEL, client=_client, temperature=0.0, max_tokens=4096
)
# --- Metric instances (all LLM-only; no embedding model needed) ------------
answer_accuracy = AnswerAccuracy(llm=evaluator_llm)
factual_correctness = FactualCorrectness(llm=evaluator_llm)
context_precision = ContextPrecision(llm=evaluator_llm)
context_recall = ContextRecall(llm=evaluator_llm)
rouge_precision = RougeScore(rouge_type="rougeL", mode="precision")
rouge_recall = RougeScore(rouge_type="rougeL", mode="recall")

# Helpfulness: a 1-5 rubric score of the response against the user's question.
# (adapted from Google's helpfulness_prompt_template)
helpfulness_rubrics = {
    "score1_description": "Response is useless/irrelevant, contains inaccurate/deceptive/misleading information, and/or contains harmful/offensive content. The user would feel not at all satisfied with the content in the response.",
    "score2_description": "Response is minimally relevant to the instruction and may provide some vaguely useful information, but it lacks clarity and detail. It might contain minor inaccuracies. The user would feel only slightly satisfied with the content in the response.",
    "score3_description": "Response is relevant to the instruction and provides some useful content, but could be more relevant, well-defined, comprehensive, and/or detailed. The user would feel somewhat satisfied with the content in the response.",
    "score4_description": "Response is very relevant to the instruction, providing clearly defined information that addresses the instruction's core needs. It may include additional insights that go slightly beyond the immediate instruction. The user would feel quite satisfied with the content in the response.",
    "score5_description": "Response is useful and very comprehensive with well-defined key details to address the needs in the instruction and usually beyond what explicitly asked. The user would feel very satisfied with the content in the response.",
}
helpfulness = RubricsScore(
    name="helpfulness", rubrics=helpfulness_rubrics, llm=evaluator_llm
)

# Rouge column names match the existing spreadsheet's headers so the already
# populated values are reused (and skipped) rather than duplicated.
ROUGE_PRECISION_COL = "rouge_score(mode=precision)"
ROUGE_RECALL_COL = "rouge_score(mode=recall)"

# All output columns, in order.
SCORE_COLUMNS = [
    "answer_accuracy",
    "factual_correctness",
    "helpfulness",
    ROUGE_PRECISION_COL,
    ROUGE_RECALL_COL,
    "context_precision",
    "context_recall",
]


# --- Helpers ---------------------------------------------------------------
def _as_list(value) -> list[str]:
    """Coerce a cell holding a (repr of a) list of strings into a real list."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(parsed, (list, tuple)):
        return [str(v) for v in parsed]
    return [str(parsed)]


def _is_missing(value) -> bool:
    """True when a score cell has no usable value yet.

    Handles every "empty" form a cell can take: ``None``, float ``NaN`` (read
    from xlsx), pandas ``NA`` (from a freshly-created column), and blank strings.
    """
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


async def _value(coro) -> float:
    """Await an ``ascore`` coroutine and return its numeric value."""
    result = await coro
    return float(result.value)


async def _compute(column: str, row: pd.Series) -> float:
    """Compute a single metric for a single row via the v0.4 ``ascore`` API."""
    user_input = str(row.get("user_input", "") or "")
    response = str(row.get("response", "") or "")
    reference = str(row.get("reference", "") or "")
    retrieved_contexts = _as_list(row.get("retrieved_contexts"))

    if column == "answer_accuracy":
        return await _value(
            answer_accuracy.ascore(
                user_input=user_input,
                response=response,
                reference=reference,
            )
        )
    if column == "factual_correctness":
        return await _value(
            factual_correctness.ascore(response=response, reference=reference)
        )
    if column == "helpfulness":
        # Legacy metric: scored via single_turn_ascore, returns a raw number.
        sample = SingleTurnSample(user_input=user_input, response=response)
        return float(await helpfulness.single_turn_ascore(sample))
    if column == "context_precision":
        return await _value(
            context_precision.ascore(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
        )
    if column == "context_recall":
        return await _value(
            context_recall.ascore(
                user_input=user_input,
                retrieved_contexts=retrieved_contexts,
                reference=reference,
            )
        )
    if column == ROUGE_PRECISION_COL:
        return await _value(
            rouge_precision.ascore(reference=reference, response=response)
        )
    if column == ROUGE_RECALL_COL:
        return await _value(
            rouge_recall.ascore(reference=reference, response=response)
        )
    raise ValueError(f"Unknown metric column: {column}")


async def rescore(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Ensure every requested score column exists.
    for column in SCORE_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    # Collect every missing (row, metric) cell as an independent unit of work,
    # then run them all concurrently -- bounded by a semaphore so we don't fire
    # hundreds of simultaneous LLM calls and trip provider rate limits.
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _worker(idx, column, row):
        async with sem:
            try:
                value = await _compute(column, row)
                return idx, column, value
            except Exception as exc:  # keep going; leave the cell blank on failure
                print(f"row {idx} / {column}: failed ({exc})")
                return idx, column, None

    tasks = [
        asyncio.create_task(_worker(idx, column, df.loc[idx]))
        for idx in df.index
        for column in SCORE_COLUMNS
        if _is_missing(df.at[idx, column])
    ]

    completed = 0
    for coro in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="scoring"
    ):
        idx, column, value = await coro
        if value is not None:
            df.at[idx, column] = value
        completed += 1
        # Checkpoint to disk periodically so a crash keeps partial progress.
        # Re-running skips already-scored cells, resuming where it left off.
        if completed % CHECKPOINT_EVERY == 0:
            _save(df, path)

    _save(df, path)
    print(f"Wrote {path}")
    return df


def _save(df: pd.DataFrame, path: Path) -> None:
    """Atomically write the sheet (temp file + replace) so a crash mid-write
    can't corrupt the existing results."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_excel(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", help="Path to the .xlsx evaluation file to re-score")
    args = parser.parse_args()

    path = Path(args.xlsx)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    asyncio.run(rescore(path))


if __name__ == "__main__":
    main()
