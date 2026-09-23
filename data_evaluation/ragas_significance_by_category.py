"""
Per-category significance testing: does hierarchical retrieval pay off more on
some question types than others?
=============================================================================

Motivation: the pooled analysis (ragas_significance.py) finds a large, robust
context_precision win for llm_retriever but no measurable answer-quality gain.
One explanation is dilution -- if the advantage is concentrated in question types
where a naive top-k retrieval is likely to miss (multi_hop, debugging), then
averaging over all 200 questions, most of which are simple lookups, will wash it
out. This script tests that directly.

Design: the 200 pooled questions are evenly split across 4 categories
(50 each: 25 from playwright + 25 from DRF), so each category is a balanced
paired subsample. Pairing is preserved within category -- all three pipelines
answer the same 50 questions.

Every RAGAS metric in the result files is tested per category, EXCEPT the two
rouge_score columns. ROUGE is excluded deliberately: it is n-gram overlap against
a reference answer, a weak proxy for answer quality, and it is mechanically
biased here -- llm_retriever returns more context, which depresses rouge
precision regardless of answer quality. State this exclusion in the paper's
methods rather than dropping it silently.

Two derived outcomes are added:
  - answer_quality    -- composite of the 3 answer metrics on a common 0-1 scale,
                         included because the individual answer metrics are too
                         tied to have resolution (88% ties on helpfulness)
  - retrieval_failed  -- binary (context_precision == 0), tested with an exact
                         McNemar; the pooled analysis showed the effect lives here

Holm correction is applied WITHIN each outcome family (4 categories x 2
comparisons = 8 tests). Correcting within outcome is the standard choice when
outcomes answer distinct questions; state it in the paper. Because 7 outcomes x
8 tests = 56 tests is substantial multiplicity, a `p_holm_global` column applies
Holm across ALL tests as a conservative sensitivity check -- report both, since
some findings are sensitive to which one you use.

!! INTERPRETIVE WARNING !!
This is a subgroup analysis with n=50 per cell. Subgroup analyses are the
classic route to false positives: with 24 tests over 4 slices, something will
look significant by chance. Two rules for reporting it honestly:
  - Treat the category split as PRE-SPECIFIED (it is -- the categories were
    assigned at question-generation time, not chosen after seeing results).
  - Report the direction/effect size across ALL categories, not just the
    winners. A real effect shows a consistent gradient across question types;
    a false positive shows one category spiking with the rest flat.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm, rankdata, wilcoxon

BASE_DIR = Path(__file__).resolve().parent
CORPORA = {"playwright": BASE_DIR / "playwright", "drf": BASE_DIR / "django-rest-framework"}
RETRIEVERS = ["naive_rag", "llm_retriever", "raptor"]
COMPARISONS = [("llm_retriever", "naive_rag"), ("llm_retriever", "raptor")]
# Ordered easiest -> hardest, which is the gradient the hypothesis predicts.
CATEGORIES = ["straightforward", "intent_driven", "multi_hop", "debugging"]

# All RAGAS metrics in the files except rouge_score(precision|recall) -- see the
# module docstring for why ROUGE is excluded.
FILE_METRICS = [
    "answer_accuracy",
    "factual_correctness",
    "helpfulness",
    "context_precision",
    "context_recall",
]
DERIVED = ["answer_quality", "retrieval_failed"]
OUTCOMES = FILE_METRICS + DERIVED
ALPHA = 0.05
RESULTS_CSV = BASE_DIR / "ragas_significance_by_category.csv"


def load(retriever):
    """Merge both corpora for one pipeline, attaching each question's category."""
    parts = []
    for corpus, folder in CORPORA.items():
        df = pd.read_excel(folder / f"ragas_results_{retriever}.xlsx")
        cats = {q["user_input"]: q["category"] for q in json.loads((folder / f"questions_{retriever}.json").read_text())}
        df.insert(0, "corpus", corpus)
        df.insert(1, "category", df["user_input"].map(cats))
        parts.append(df)
    merged = pd.concat(parts, ignore_index=True)
    if merged["category"].isna().any():
        raise ValueError(f"{retriever}: some questions have no category (join by user_input failed)")
    # Composite answer quality: 3 metrics mapped onto a common 0-1 scale.
    # Equal weighting is a deliberate pre-specified choice, not tuned.
    merged["answer_quality"] = (
        merged["answer_accuracy"] + merged["factual_correctness"] + merged["helpfulness"] / 5
    ) / 3
    merged["retrieval_failed"] = (merged["context_precision"] == 0).astype(int)
    return merged.sort_values(["corpus", "user_input"], kind="stable").reset_index(drop=True)


dfs = {r: load(r) for r in RETRIEVERS}

reference = dfs["llm_retriever"][["corpus", "user_input", "category"]]
for name, df in dfs.items():
    pd.testing.assert_frame_equal(
        df[["corpus", "user_input", "category"]].reset_index(drop=True),
        reference.reset_index(drop=True),
        obj=f"{name} question alignment",
    )


def rank_biserial(diffs):
    """Signed matched-pairs rank-biserial: (T+ - T-) / (T+ + T-). See
    ragas_significance.py for why this rather than Z/sqrt(N)."""
    nz = diffs[diffs != 0]
    if nz.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nz))
    return (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()


def signed_rank(x, y):
    """Wilcoxon signed-rank, in the terms a stats package reports it.

    Identical to signed_rank() in ragas_significance_pooled.py -- kept in step
    so the pooled and per-category tables are the same test reported the same
    way. Columns: N (pairs left after zero differences are dropped), T_plus /
    T_minus (rank sums for x > y and x < y), T = min(T_plus, T_minus), Z
    (tie- and continuity-corrected), and the three p-values.

    method="approx" is pinned rather than "auto" only to make the choice
    explicit: every test here has tied |differences|, so scipy's "auto" already
    falls back to the normal approximation in all of them -- the exact method is
    never reachable on this data, small n=50 cells included. correction=True is
    the standard reported form, and is conservative.
    """
    diffs = x - y
    nz = diffs[diffs != 0]
    n = nz.size
    if n == 0:  # every pair tied: no signed-rank information at all
        return dict(N=0, T_plus=0.0, T_minus=0.0, T=np.nan, Z=np.nan,
                    p_greater=1.0, p_less=1.0, p_two_tailed=1.0,
                    method="all pairs tied (no test)")
    ranks = rankdata(np.abs(nz))
    t_plus, t_minus = float(ranks[nz > 0].sum()), float(ranks[nz < 0].sum())
    t = min(t_plus, t_minus)
    mu = n * (n + 1) / 4
    _, counts = np.unique(np.abs(nz), return_counts=True)
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24 - np.sum(counts ** 3 - counts) / 48)
    # Clamp at 0: when T lands within half a rank of its null mean the continuity
    # correction overshoots and the raw expression goes negative (p = 1 there).
    z = max((abs(mu - t) - 0.5) / sd, 0.0) if sd > 0 else 0.0
    kw = dict(zero_method="wilcox", correction=True, method="approx")
    return dict(N=n, T_plus=t_plus, T_minus=t_minus, T=t, Z=float(z),
                p_greater=float(wilcoxon(x, y, alternative="greater", **kw)[1]),
                p_less=float(wilcoxon(x, y, alternative="less", **kw)[1]),
                p_two_tailed=float(wilcoxon(x, y, **kw)[1]),
                method="normal with continuity correction")


def hodges_lehmann(diffs, alpha=ALPHA):
    """Hodges-Lehmann point estimate of the median paired difference, with the
    distribution-free CI that inverts the signed-rank test.

    The interval estimate a Wilcoxon result should be reported with: it puts the
    effect on the metric's own scale, which the rank-biserial r cannot. Walsh
    averages (all pairwise means of the differences) are the signed-rank test's
    natural estimator; the CI is the k-th smallest/largest Walsh average, with k
    read off the null distribution of W.
    """
    nz = diffs[diffs != 0]
    n = nz.size
    if n == 0:
        return 0.0, 0.0, 0.0
    walsh = np.add.outer(nz, nz)[np.triu_indices(n)] / 2.0
    walsh.sort()
    estimate = float(np.median(walsh))
    # k = number of Walsh averages to trim from each tail. Normal approximation
    # to the signed-rank null; exact enough at these n and never below 1.
    z = norm.ppf(1 - alpha / 2)
    k = int(round(n * (n + 1) / 4 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    k = min(max(k, 0), walsh.size // 2)
    return estimate, float(walsh[k]), float(walsh[walsh.size - 1 - k])


def wilson(successes, total, alpha=ALPHA):
    """Wilson score interval -- for reporting a raw failure rate, not a contrast."""
    if total == 0:
        return 0.0, 0.0
    z = norm.ppf(1 - alpha / 2)
    p, z2 = successes / total, z * z
    centre = (p + z2 / (2 * total)) / (1 + z2 / total)
    half = z * np.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / (1 + z2 / total)
    return float(centre - half), float(centre + half)


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    n, order, adjusted, run = len(pvalues), np.argsort(pvalues), np.empty(len(pvalues)), 0.0
    for rank, idx in enumerate(order):
        run = max(run, (n - rank) * pvalues[idx])
        adjusted[idx] = min(run, 1.0)
    return adjusted


rows = []
for outcome in OUTCOMES:
    for cat in CATEGORIES:
        for a, b in COMPARISONS:
            mask = dfs[a]["category"] == cat
            x = dfs[a].loc[mask, outcome].to_numpy(float)
            y = dfs[b].loc[mask, outcome].to_numpy(float)
            ok = ~(np.isnan(x) | np.isnan(y))
            x, y = x[ok], y[ok]
            diffs = x - y

            row = dict(outcome=outcome, category=cat, comparison=f"{a} vs {b}",
                       n=len(x), mean_a=x.mean(), mean_b=y.mean(), diff=x.mean() - y.mean(),
                       median_a=float(np.median(x)), median_b=float(np.median(y)),
                       iqr_a=float(np.subtract(*np.percentile(x, [75, 25]))),
                       iqr_b=float(np.subtract(*np.percentile(y, [75, 25]))))

            if outcome == "retrieval_failed":
                # Paired binary -> exact McNemar on the discordant pairs. On a 0/1
                # outcome the signed-rank test degenerates to the sign test, of
                # which this is the exact form; scipy's wilcoxon would fall back to
                # the normal approximation here (every |difference| is tied) and is
                # roughly 2x anti-conservative at these discordant-pair counts.
                a_only, b_only = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
                n_disc = a_only + b_only
                # Contrast is a rate difference, so report it with its own CI
                # rather than a Hodges-Lehmann median (which is 0/±1 on binary data).
                lo_a, hi_a = wilson(int(x.sum()), len(x))
                lo_b, hi_b = wilson(int(y.sum()), len(y))
                # Same column names as the signed-rank rows; T_plus/T_minus/Z do
                # not exist for a sign test, so they stay blank, not faked.
                bt = lambda alt: binomtest(a_only, n_disc, 0.5, alternative=alt).pvalue if n_disc else 1.0
                row.update(N=n_disc, T_plus=np.nan, T_minus=np.nan,
                           T=float(a_only), Z=np.nan, p_greater=bt("greater"),
                           p_less=bt("less"), p_two_tailed=bt("two-sided"),
                           # Signed so + always favours llm_retriever, matching the
                           # other outcomes (fewer failures = better).
                           effect=(b_only - a_only) / n_disc if n_disc else 0.0,
                           hl=np.nan, hl_lo=lo_a - hi_b, hl_hi=hi_a - lo_b,
                           discordant_a=a_only, discordant_b=b_only,
                           method="exact (binomial)")
            else:
                hl, hl_lo, hl_hi = hodges_lehmann(diffs)
                row.update(effect=rank_biserial(diffs),
                           hl=hl, hl_lo=hl_lo, hl_hi=hl_hi,
                           discordant_a=np.nan, discordant_b=np.nan,
                           **signed_rank(x, y))
            rows.append(row)

results = pd.DataFrame(rows)
# Primary: Holm within each outcome family (8 tests each).
results["p_holm"] = results.groupby("outcome")["p_two_tailed"].transform(lambda s: holm(s.values))
results[f"sig_{ALPHA}"] = results["p_holm"] < ALPHA
# Sensitivity: Holm across every test run here, the conservative reading.
results["p_holm_global"] = holm(results["p_two_tailed"].values)
results[f"sig_global_{ALPHA}"] = results["p_holm_global"] < ALPHA

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
LABELS = {"context_precision": "context_precision (primary — the pooled analysis's robust win)",
          "answer_quality": "answer_quality (composite 0-1, derived)",
          "retrieval_failed": "retrieval failure rate (binary, exact McNemar)"}
for outcome in OUTCOMES:
    sub = results[results["outcome"] == outcome].copy()
    sub["category"] = pd.Categorical(sub["category"], CATEGORIES, ordered=True)
    sub = sub.sort_values(["comparison", "category"])
    print(f"\n=== {LABELS.get(outcome, outcome)} — Holm within {len(sub)} tests ===")
    print(sub[["category", "comparison", "n", "N", "T_plus", "T_minus", "T", "Z",
               "p_greater", "p_less", "p_two_tailed", "method"]].round(6).to_string(index=False))
    print(sub[["category", "comparison", "mean_a", "mean_b", "diff", "hl", "hl_lo", "hl_hi",
               "p_two_tailed", "p_holm", f"sig_{ALPHA}", "p_holm_global",
               f"sig_global_{ALPHA}", "effect"]].round(4).to_string(index=False))

hits = results[results[f"sig_{ALPHA}"]].sort_values("p_two_tailed")
print(f"\n=== SUMMARY: {len(hits)} of {len(results)} tests significant within-family; "
      f"{int(results[f'sig_global_{ALPHA}'].sum())} survive global Holm ===")
print(hits[["outcome", "category", "comparison", "diff", "T", "Z", "p_two_tailed", "p_holm",
            "p_holm_global", f"sig_global_{ALPHA}", "effect"]].round(4).to_string(index=False))

results.to_csv(RESULTS_CSV, index=False)
print(f"\nSaved -> {RESULTS_CSV}")
