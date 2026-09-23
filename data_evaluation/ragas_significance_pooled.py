"""
Pooled significance testing: the primary analysis over all 200 questions.
=========================================================================

Companion to ragas_significance_by_category.py, which slices the same data by
question type. This script produces the numbers behind the paper's sections
5.1 (retrieval quality), 5.2 (answer quality + equivalence) and 5.4 (cost).

Design: all three pipelines answer the same 200 questions (100 Playwright +
100 DRF), so every comparison is paired at the question level. The two corpora
are pooled with pairing preserved; per-corpus results are printed alongside as
a robustness check, since effect magnitudes differ between them.

Tests mirror the by-category script exactly, so the two analyses are directly
comparable:
  - Wilcoxon signed-rank, zeros discarded (zero_method="wilcox"), on every
    outcome except retrieval_failed. Reported in the standard normal-approximation
    form -- rank sums W+/W-, T = min(W+, W-), tie-corrected z with continuity
    correction -- so the numbers reproduce in any Wilcoxon calculator or SPSS.
    See signed_rank() for why the exact method is not in play here.
  - retrieval_failed is binary; there the signed-rank test degenerates to the
    sign test, computed in its exact form (exact McNemar on discordant pairs).
    scipy's wilcoxon would fall back to a normal approximation here -- every
    |difference| is tied -- and is materially anti-conservative at these counts.
  - Effect size: matched-pairs rank-biserial r [Kerby, 2014].
  - Interval estimate: Hodges-Lehmann median paired difference with the
    distribution-free CI that inverts the signed-rank test.

Holm correction is applied across the family of pooled tests (7 outcomes x 2
comparisons = 14 tests).

ROUGE is excluded for the reasons given in ragas_significance_by_category.py.

Equivalence: a non-significant Wilcoxon is not evidence of no difference. For
the composite answer-quality outcome we additionally run TOST (two one-sided
tests) against a +/-EQUIV_MARGIN band, which CAN support a claim of practical
equivalence. Read the two together: TOST significant = equivalent within the
margin; both tests non-significant = inconclusive at this sample size.
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
# Practical-equivalence band for answer quality, on the composite 0-1 scale.
# Fixed a priori: 0.05 is half the smallest gap the LLM judge can express on
# answer_accuracy, so differences inside it are below the judge's resolution.
EQUIV_MARGIN = 0.05
BOOT = 10_000
SEED = 20240728
RESULTS_CSV = BASE_DIR / "ragas_significance_pooled.csv"


def load(retriever):
    """Merge both corpora for one pipeline, attaching each question's category."""
    parts = []
    for corpus, folder in CORPORA.items():
        df = pd.read_excel(folder / f"ragas_results_{retriever}.xlsx")
        cats = {q["user_input"]: q["category"]
                for q in json.loads((folder / f"questions_{retriever}.json").read_text())}
        df.insert(0, "corpus", corpus)
        df.insert(1, "category", df["user_input"].map(cats))
        parts.append(df)
    merged = pd.concat(parts, ignore_index=True)
    if merged["category"].isna().any():
        raise ValueError(f"{retriever}: some questions have no category (join by user_input failed)")
    merged["answer_quality"] = (
        merged["answer_accuracy"] + merged["factual_correctness"] + merged["helpfulness"] / 5
    ) / 3
    merged["retrieval_failed"] = (merged["context_precision"] == 0).astype(int)
    return merged.sort_values(["corpus", "user_input"], kind="stable").reset_index(drop=True)


def rank_biserial(diffs):
    nz = diffs[diffs != 0]
    if nz.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nz))
    return (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()


def signed_rank(x, y):
    """Wilcoxon signed-rank, in the terms a stats package reports it.

    Column names follow the standard Wilcoxon vocabulary so a row can be checked
    against any calculator or SPSS without translation:

      N          pairs remaining after zero differences are dropped
      T_plus     sum of ranks where x > y  ("condition 1 > condition 2")
      T_minus    sum of ranks where x < y ; T_plus + T_minus = N(N+1)/2
      T          the test statistic, min(T_plus, T_minus)
      Z          tie-corrected, continuity-corrected normal deviate
      p_greater  one-sided, x > y        p_less  one-sided, x < y
      p_two_tailed   the p everything downstream (Holm, sig flags) uses

    method="approx" is pinned rather than "auto" only to make the choice
    explicit: every test in this file and in the by-category script has tied
    |differences|, so scipy's "auto" falls back to the normal approximation in
    all of them anyway -- the exact method is never actually reachable on this
    data. correction=True is the standard reported form and is what makes these
    p-values reproduce elsewhere; it costs at most ~0.001 of p and is
    conservative, so it never manufactures significance.
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
    # correction overshoots and the raw expression goes negative. p is taken from
    # scipy, which clamps the same way (that case is p = 1).
    z = max((abs(mu - t) - 0.5) / sd, 0.0) if sd > 0 else 0.0
    kw = dict(zero_method="wilcox", correction=True, method="approx")
    return dict(N=n, T_plus=t_plus, T_minus=t_minus, T=t, Z=float(z),
                p_greater=float(wilcoxon(x, y, alternative="greater", **kw)[1]),
                p_less=float(wilcoxon(x, y, alternative="less", **kw)[1]),
                p_two_tailed=float(wilcoxon(x, y, **kw)[1]),
                method="normal with continuity correction")


def hodges_lehmann(diffs, alpha=ALPHA):
    """HL point estimate + distribution-free CI (see by-category script)."""
    nz = diffs[diffs != 0]
    n = nz.size
    if n == 0:
        return 0.0, 0.0, 0.0
    walsh = np.add.outer(nz, nz)[np.triu_indices(n)] / 2.0
    walsh.sort()
    z = norm.ppf(1 - alpha / 2)
    k = int(round(n * (n + 1) / 4 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    k = min(max(k, 0), walsh.size // 2)
    return float(np.median(walsh)), float(walsh[k]), float(walsh[walsh.size - 1 - k])


def wilson(successes, total, alpha=ALPHA):
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


def paired_test(x, y, outcome):
    """One paired comparison -> the full row a paper needs to report it."""
    diffs = x - y
    row = dict(n=len(x), mean_a=x.mean(), mean_b=y.mean(), diff=x.mean() - y.mean(),
               median_a=float(np.median(x)), median_b=float(np.median(y)))

    if outcome == "retrieval_failed":
        a_only = int(((x == 1) & (y == 0)).sum())
        b_only = int(((x == 0) & (y == 1)).sum())
        n_disc = a_only + b_only
        lo_a, hi_a = wilson(int(x.sum()), len(x))
        lo_b, hi_b = wilson(int(y.sum()), len(y))
        # Same column names as the signed-rank rows; T_plus/T_minus/Z do not
        # exist for a sign test, so they stay blank rather than being faked.
        bt = lambda alt: binomtest(a_only, n_disc, 0.5, alternative=alt).pvalue if n_disc else 1.0
        row.update(N=n_disc, T_plus=np.nan, T_minus=np.nan, T=float(a_only), Z=np.nan,
                   p_greater=bt("greater"), p_less=bt("less"), p_two_tailed=bt("two-sided"),
                   effect=(b_only - a_only) / n_disc if n_disc else 0.0,
                   hl=np.nan, hl_lo=lo_a - hi_b, hl_hi=hi_a - lo_b,
                   discordant_a=a_only, discordant_b=b_only,
                   test="exact McNemar", method="exact (binomial)")
        return row

    hl, hl_lo, hl_hi = hodges_lehmann(diffs)
    row.update(effect=rank_biserial(diffs), hl=hl, hl_lo=hl_lo, hl_hi=hl_hi,
               discordant_a=np.nan, discordant_b=np.nan, test="Wilcoxon signed-rank",
               **signed_rank(x, y))
    return row


def tost(diffs, margin=EQUIV_MARGIN):
    """Two one-sided Wilcoxon tests against +/-margin. Equivalence is declared
    only if BOTH one-sided tests reject, i.e. p = max(p_lower, p_upper) < alpha."""
    kw = dict(zero_method="wilcox", correction=True, method="approx")  # see signed_rank()
    lower = wilcoxon(diffs - (-margin), alternative="greater", **kw)[1]
    upper = wilcoxon(diffs - margin, alternative="less", **kw)[1]
    return max(lower, upper), lower, upper


def boot_mean_ci(diffs, level=0.90, n_boot=BOOT, seed=SEED):
    """Percentile bootstrap CI on the MEAN paired difference.

    Secondary, and a DIFFERENT ESTIMAND from everything else here: these
    distributions are skewed enough that the mean and the pseudomedian diverge
    substantially (answer_quality: mean 0.044 vs HL 0.013). The equivalence
    claim is made on the pseudomedian, to stay consistent with the signed-rank
    tests; this is reported only so the mean-scale reader is not misled.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


dfs = {r: load(r) for r in RETRIEVERS}

reference = dfs["llm_retriever"][["corpus", "user_input", "category"]]
for name, df in dfs.items():
    pd.testing.assert_frame_equal(
        df[["corpus", "user_input", "category"]].reset_index(drop=True),
        reference.reset_index(drop=True),
        obj=f"{name} question alignment",
    )

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", None)

# ---------------------------------------------------------------- pooled tests
rows = []
for outcome in OUTCOMES:
    for a, b in COMPARISONS:
        x = dfs[a][outcome].to_numpy(float)
        y = dfs[b][outcome].to_numpy(float)
        ok = ~(np.isnan(x) | np.isnan(y))
        rows.append(dict(outcome=outcome, comparison=f"{a} vs {b}",
                         **paired_test(x[ok], y[ok], outcome)))

results = pd.DataFrame(rows)
results["p_holm"] = holm(results["p_two_tailed"].values)
results[f"sig_{ALPHA}"] = results["p_holm"] < ALPHA

# Two views of the same rows: the test as a calculator reports it, then the
# effect sizes and the multiplicity-corrected decision.
print(f"=== 5.1/5.2 POOLED (n=200) -- Wilcoxon signed-rank statistics ===")
print(results[["outcome", "comparison", "n", "N", "T_plus", "T_minus", "T", "Z",
               "p_greater", "p_less", "p_two_tailed", "method"]].round(6).to_string(index=False))
print("  n = pairs; N = pairs left after dropping zero differences;"
      " p_greater is 'a > b' (a = first pipeline named).")

print(f"\n=== 5.1/5.2 POOLED -- effect sizes and decision (Holm across {len(results)} tests) ===")
print(results[["outcome", "comparison", "N", "mean_a", "mean_b", "diff",
               "hl", "hl_lo", "hl_hi", "effect", "T", "p_two_tailed", "p_holm",
               f"sig_{ALPHA}"]].round(4).to_string(index=False))

# ------------------------------------------------------- per-corpus robustness
print("\n=== 5.1 per-corpus replication (direction check, uncorrected) ===")
per_corpus = []
for corpus in CORPORA:
    for outcome in OUTCOMES:
        for a, b in COMPARISONS:
            ma, mb = dfs[a]["corpus"] == corpus, dfs[b]["corpus"] == corpus
            x, y = dfs[a].loc[ma, outcome].to_numpy(float), dfs[b].loc[mb, outcome].to_numpy(float)
            ok = ~(np.isnan(x) | np.isnan(y))
            per_corpus.append(dict(corpus=corpus, outcome=outcome, comparison=f"{a} vs {b}",
                                   **paired_test(x[ok], y[ok], outcome)))
per_corpus = pd.DataFrame(per_corpus)
pivot = per_corpus.pivot_table(index=["outcome", "comparison"], columns="corpus",
                               values=["diff", "effect", "p_two_tailed"])
print(pivot.round(4).to_string())
agree = (per_corpus.pivot_table(index=["outcome", "comparison"], columns="corpus", values="diff")
         .apply(lambda r: np.sign(r).nunique() == 1, axis=1))
print(f"\ndirection agrees across both corpora in {int(agree.sum())}/{len(agree)} outcome-comparisons")

# ---------------------------------------------------------------- 5.2 equivalence
print(f"\n=== 5.2 equivalence on answer_quality (TOST, margin +/-{EQUIV_MARGIN}) ===")
eq_rows = []
for a, b in COMPARISONS:
    d = (dfs[a]["answer_quality"].to_numpy(float) - dfs[b]["answer_quality"].to_numpy(float))
    d = d[~np.isnan(d)]
    p_tost, p_lo, p_hi = tost(d)
    # 90% (= 1-2*alpha) HL interval: the interval that corresponds to TOST at
    # alpha, on the same estimand the signed-rank test is about. Equivalence
    # holds exactly when this sits inside +/-EQUIV_MARGIN.
    hl, hl_lo, hl_hi = hodges_lehmann(d, alpha=2 * ALPHA)
    mean_lo, mean_hi = boot_mean_ci(d)
    eq_rows.append(dict(comparison=f"{a} vs {b}", hl=hl, hl90_lo=hl_lo, hl90_hi=hl_hi,
                        p_tost=p_tost, p_lower=p_lo, p_upper=p_hi, equivalent=p_tost < ALPHA,
                        mean_diff=d.mean(), mean90_lo=mean_lo, mean90_hi=mean_hi))
equivalence = pd.DataFrame(eq_rows)
print(equivalence.round(4).to_string(index=False))
print(f"equivalent=True: the pseudomedian difference is bounded inside +/-{EQUIV_MARGIN}.\n"
      "equivalent=False with a non-significant Wilcoxon above means INCONCLUSIVE, not equal.\n"
      "mean_diff columns are a different estimand (see boot_mean_ci) -- do not read them\n"
      "as the equivalence interval; these distributions are skewed and the two diverge.")

# --------------------------------------------- 5.2 saturation / conditional means
print("\n=== 5.2 answer_accuracy conditional on retrieval success ===")
cond = pd.DataFrame([
    dict(pipeline=r,
         **{"ctx_precision = 0": dfs[r].loc[dfs[r]["context_precision"] == 0, "answer_accuracy"].mean(),
            "n_failed": int((dfs[r]["context_precision"] == 0).sum()),
            "ctx_precision > 0.8": dfs[r].loc[dfs[r]["context_precision"] > 0.8, "answer_accuracy"].mean(),
            "n_good": int((dfs[r]["context_precision"] > 0.8).sum())})
    for r in RETRIEVERS])
print(cond.round(4).to_string(index=False))

print("\n=== retrieval failures out of 200 (Wilson 95% CI) ===")
for r in RETRIEVERS:
    k = int(dfs[r]["retrieval_failed"].sum())
    lo, hi = wilson(k, len(dfs[r]))
    print(f"  {r:15} {k:3d}/{len(dfs[r])}  = {k / len(dfs[r]):.3f}  [{lo:.3f}, {hi:.3f}]")

# ---------------------------------------------------------------------- 5.4 cost
print("\n=== 5.4 cost per query ===")
cost = pd.DataFrame([
    dict(pipeline=r, cost_per_query=dfs[r]["cost"].mean(),
         tokens_per_query=dfs[r]["total_tokens"].mean(), total_cost=dfs[r]["cost"].sum())
    for r in RETRIEVERS])
cost["relative"] = cost["cost_per_query"] / cost.loc[cost.pipeline == "naive_rag",
                                                     "cost_per_query"].iloc[0]
print(cost.round(6).to_string(index=False))
# NOTE: LLM calls per query are not recorded in the RAGAS result files; the
# draft's 4.35 figure has to come from the retriever's own logs, not from here.

failures_avoided = {b: int((dfs[b]["retrieval_failed"].sum()
                            - dfs["llm_retriever"]["retrieval_failed"].sum()))
                    for _, b in COMPARISONS}
extra_cost = {b: float(dfs["llm_retriever"]["cost"].sum() - dfs[b]["cost"].sum())
              for _, b in COMPARISONS}
for b in failures_avoided:
    n_av, extra = failures_avoided[b], extra_cost[b]
    per = f"${extra / n_av:.4f}" if n_av > 0 else "n/a (no failures avoided)"
    print(f"  vs {b:12} avoids {n_av:3d} failures for +${extra:.4f} -> {per} per avoided failure")

results.to_csv(RESULTS_CSV, index=False)
per_corpus.to_csv(BASE_DIR / "ragas_significance_pooled_by_corpus.csv", index=False)
print(f"\nSaved -> {RESULTS_CSV}")
print(f"Saved -> {BASE_DIR / 'ragas_significance_pooled_by_corpus.csv'}")
