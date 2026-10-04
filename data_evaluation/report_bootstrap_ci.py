"""
Across-questions bootstrap CI: llm_retriever vs naive_rag / raptor, pooled over both corpora.

Captures how much the average depends on which 200 questions happened to be asked. Resamples the
per-question scores already in ragas_results_*.xlsx (no extra cost). Each resample draws the same
question indices for every pipeline, so paired differences stay paired.

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
BASELINES = [("naive_rag", "Naive RAG"), ("raptor", "RAPTOR")]
METRICS = [("context_precision", "Context precision"), ("context_recall", "Context recall"),
           ("answer_accuracy", "Answer accuracy"), ("helpfulness", "Helpfulness"),
           ("factual_correctness", "Factual correctness")]
LEVEL = 0.95
BOOT = 10_000
SEED = 20240728


def load(retriever):
    return pd.concat([pd.read_excel(ROOT / c / f"ragas_results_{retriever}.xlsx")
                      .sort_values("user_input", kind="stable") for c in CORPORA],
                     ignore_index=True)


llm = load("llm_retriever")
n = len(llm)
idx = np.random.default_rng(SEED).integers(0, n, size=(BOOT, n))  # shared across all rows -> paired
lo_q, hi_q = (1 - LEVEL) / 2 * 100, (1 + LEVEL) / 2 * 100

rows = []
for base, base_label in BASELINES:
    other = load(base)
    assert (llm["user_input"].values == other["user_input"].values).all(), "question misalignment"
    for key, label in METRICS:
        x, y = llm[key].to_numpy(float), other[key].to_numpy(float)
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
res = res.sort_values(["_o", "baseline"], kind="stable").drop(columns="_o")

print(f"Source: {ROOT}\n{BOOT:,} bootstrap resamples of {n} questions, {LEVEL:.0%} percentile CI "
      "on mean(llm_retriever - baseline)\n")
print(f"{'':<20}{'':<11}{'LLM':>7}{'base':>7}{'diff':>8}   {'95% CI':<18}0 excluded")
shown = None
for _, r in res.iterrows():
    m = r.metric if r.metric != shown else ""
    shown = r.metric
    print(f"{m:<20}{r.baseline:<11}{r.mean_llm:7.3f}{r.mean_baseline:7.3f}{r["diff"]:+8.3f}   "
          f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  {r.excludes_zero}")
out = ROOT / "bootstrap_ci_summary.csv"
res.to_csv(out, index=False)
print(f"\nSaved -> {out}")
