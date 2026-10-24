"""
Across-questions bootstrap CI: llm_retriever vs every baseline, pooled over both corpora.

Captures how much the average depends on which 200 questions happened to be asked. Resamples the
per-question scores already in ragas_results_*.xlsx (no extra cost). Each resample draws the same
question indices for every pipeline, so paired differences stay paired. Pairs are matched on
(corpus, user_input); a baseline run on only one corpus (e.g. KETRAG, DRF only) is compared on the
questions both pipelines answered -- see the n column.

One row per (metric, baseline): mean of each pipeline, mean paired difference (llm_retriever minus
baseline), percentile CI on that difference, and whether the CI excludes 0.

Usage:  python report_bootstrap_ci.py [results_dir]
        results_dir holds playwright/ and django-rest-framework/ (default: this folder)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
CORPORA = ["playwright", "django-rest-framework"]
BASELINES = [("naive_rag", "Naive RAG"), ("raptor", "RAPTOR"), ("hybrid_rerank", "Hybrid+Rerank"),
             ("ketrag", "KETRAG"), ("rankgpt", "RankGPT")]
METRICS = [("context_precision", "Context precision"), ("context_recall", "Context recall"),
           ("answer_accuracy", "Answer accuracy"), ("helpfulness", "Helpfulness"),
           ("factual_correctness", "Factual correctness")]
LEVEL = 0.95
BOOT = 10_000
SEED = 20240728


def load(retriever):
    """Every corpus the pipeline was run on, tagged with its corpus; None if it has no results."""
    parts = []
    for c in CORPORA:
        path = ROOT / c / f"ragas_results_{retriever}.xlsx"
        if not path.exists():
            print(f"[load] {retriever}: no results for {c}, skipped")
            continue
        parts.append(pd.read_excel(path).sort_values("user_input", kind="stable").assign(corpus=c))
    return pd.concat(parts, ignore_index=True) if parts else None


def resample_idx(n):
    """Bootstrap indices for n paired questions. Same seed for every row, so all baselines
    compared on the same questions get the same resamples (paired across rows too)."""
    if n not in _idx:
        _idx[n] = np.random.default_rng(SEED).integers(0, n, size=(BOOT, n))
    return _idx[n]


_idx = {}
llm = load("llm_retriever")
lo_q, hi_q = (1 - LEVEL) / 2 * 100, (1 + LEVEL) / 2 * 100

rows = []
for base, base_label in BASELINES:
    other = load(base)
    if other is None:
        continue
    # Matched on (corpus, user_input); row order follows llm_retriever.
    pair = llm.merge(other, on=["corpus", "user_input"], suffixes=("_llm", "_base"), validate="one_to_one")
    assert len(pair) == len(other), f"{base}: some questions have no llm_retriever counterpart"
    idx = resample_idx(len(pair))
    for key, label in METRICS:
        x, y = pair[f"{key}_llm"].to_numpy(float), pair[f"{key}_base"].to_numpy(float)
        ok = ~(np.isnan(x) | np.isnan(y))
        d = (x - y)
        d = np.where(ok, d, np.nan)
        means = np.nanmean(d[idx], axis=1)
        lo, hi = np.percentile(means, [lo_q, hi_q])
        rows.append(dict(metric=label, baseline=base_label, n=int(ok.sum()),
                         mean_llm=np.nanmean(x[ok]), mean_baseline=np.nanmean(y[ok]),
                         diff=np.nanmean(d), ci_lo=lo, ci_hi=hi,
                         excludes_zero="Yes" if lo > 0 or hi < 0 else "No"))

res = pd.DataFrame(rows)
res["_o"] = res["metric"].map({l: i for i, (_, l) in enumerate(METRICS)})
res["_b"] = res["baseline"].map({l: i for i, (_, l) in enumerate(BASELINES)})
res = res.sort_values(["_o", "_b"], kind="stable").drop(columns=["_o", "_b"])

print(f"Source: {ROOT}\n{BOOT:,} bootstrap resamples of the paired questions, {LEVEL:.0%} percentile CI "
      "on mean(llm_retriever - baseline)\n")
print(f"{'':<20}{'':<15}{'n':>5}{'LLM':>7}{'base':>7}{'diff':>8}   {'95% CI':<18}0 excluded")
shown = None
for _, r in res.iterrows():
    m = r.metric if r.metric != shown else ""
    shown = r.metric
    print(f"{m:<20}{r.baseline:<15}{r.n:5d}{r.mean_llm:7.3f}{r.mean_baseline:7.3f}{r["diff"]:+8.3f}   "
          f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  {r.excludes_zero}")
out = ROOT / "bootstrap_ci_summary.csv"
res.to_csv(out, index=False)
print(f"\nSaved -> {out}")
