# Variant-Confusion Analysis — Methodology

**Purpose of this document:** a complete, self-contained explanation of every
metric computed by `scripts/variant_confusion.py`, written so it can be
lifted directly into a presentation (one section ≈ one or two slides). It
covers *what* is computed, *how* it's computed, and *why* — plus a worked
numeric example using a real result from this project.

Code lives in [`scripts/variant_confusion.py`](../../scripts/variant_confusion.py).
Per-dataset outputs live in `analysis/variant_confusion/{dataset}/{tables,figures,reports}/`.

---

## 1. The research question (slide: "What are we testing?")

> Does rewriting a question/sentence to be more **idiomatic** hurt a model's
> accuracy more than rewriting it as a plain **paraphrase** — using the
> **original** text as the control?

Every dataset row has 3 text variants: `original`, `paraphrase`, `idiomatic`
(the same underlying question/sentence, rewritten by an LLM augmenter). A
model is evaluated on all 3 variants independently, and this analysis
compares the pattern of *where it goes wrong* across the three.

---

## 2. Data preparation — "aligned" items (slide: "What counts as a valid comparison?")

Not every row survives augmentation for every variant (the LLM augmenter
sometimes fails validation for a given row — see Stage 2). An item is
**aligned** only if the model has a scored result for **all three** variants
of that row. Anything missing a variant is dropped before any statistics are
computed — this keeps every comparison strictly paired (same row, same
model, all three variants present).

```
aligned = { id: (original_correct, paraphrase_correct, idiomatic_correct) }
```

`n` = number of aligned items for a model = the denominator for every
statistic below.

---

## 3. Metric 1 — Joint confusion matrix (slide: "The full 8-way breakdown")

For each aligned item there are 2³ = **8 possible outcomes** — every
combination of right/wrong across `(original, paraphrase, idiomatic)`. We
count how many items fall into each of the 8 buckets and report count + % of
`n` for each. This is the most granular view — useful for appendix slides or
raw-data sanity checks, but too detailed for a headline slide.

**Output:** `{model}_confusion.csv` / `.md` / a colored HTML table.

---

## 4. Metric 2 — Sole-cause mistakes (slide: "The key quantity — did the rewrite alone break it?")

This is the metric everything else is built on. We restrict to items where
the model got **`original` right**, and ask: did switching to *one specific*
rewrite alone break it, while the other rewrite stayed correct?

| Quantity | Definition | Interpretation |
|---|---|---|
| **`idiomatic_only_wrong`** (b) | original ✅, paraphrase ✅, idiomatic ❌ | Idiomatic rewrite alone broke it |
| **`paraphrase_only_wrong`** (c) | original ✅, paraphrase ❌, idiomatic ✅ | Paraphrase rewrite alone broke it |
| `original_only_wrong` | original ❌, paraphrase ✅, idiomatic ✅ | Reported for reference only — does not enter any test |

Why "sole cause"? Items where **both** rewrites broke it (or neither did)
give no information about which rewrite is *worse* — they're excluded from
every significance test below, exactly like any paired test discards
concordant pairs.

**Output:** `{model}_sole_cause.csv` / `.md` / a colored bar chart.

---

## 5. Statistical Test 1 — McNemar's test (slide: "Is the difference between b and c real?")

**Question it answers:** given b = `idiomatic_only_wrong` and c =
`paraphrase_only_wrong`, is the gap between them larger than we'd expect from
random chance alone?

**Null hypothesis (H0):** b and c come from the same underlying 50/50 rate —
i.e. among items where exactly one rewrite broke it, it's a coin flip
whether it was the idiomatic or the paraphrase rewrite.

**Formula** (with Yates' continuity correction):

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad \text{df} = 1$$

**Where the formula comes from:** under H0, with n = b + c held fixed, the
count of items assigned to "idiomatic broke it" follows a Binomial(n, 0.5)
distribution, so Var(B) = n/4 and Var(B − C) = n. The z-statistic
(B − C)/√n is approximately standard normal, and z² ~ χ²(1). The "−1" is
Yates' correction, compensating for approximating a discrete distribution
with a continuous one — standard practice for small paired 2×2 designs.

**p-value:** `scipy.stats.chi2.sf(chi2_stat, df=1)`.

**How to read it:**
- **p < 0.05** → statistically significant: one rewrite reliably breaks
  correct answers more than the other.
- **p ≥ 0.05** → not significant: the observed gap between b and c is
  plausibly just noise given this sample size.

**Known limitation:** the chi-square approximation can be inaccurate when
`b + c` is small (our per-model discordant counts range from ~45 to ~150) —
this motivated adding Metric/Test 2 below as a robustness check.

**Output:** the `## Test statistic` and `## Result` sections of
`{model}_chi2_report.md`, and the McNemar line in the sole-cause bar chart's
subtitle.

---

## 6. Statistical Test 2 — Non-parametric bootstrap (slide: "A second, assumption-free check")

**Why add this:** McNemar's p-value relies on a *theoretical* approximation.
The bootstrap instead builds an *empirical* sampling distribution directly
from the data, making no assumption about normality or chi-square shape —
particularly valuable when discordant counts are small or a result is
borderline (e.g. p≈0.04).

**Procedure** (`bootstrap_diff_ci()`):

1. Take all `n` aligned items for a model (the full paired
   `(original, paraphrase, idiomatic)` correctness triples).
2. **Resample n items with replacement** — some items get picked multiple
   times, others not at all — simulating "what if we'd tested on a
   slightly different but same-size sample."
3. On this resample, recompute the difference:
   `diff = idiomatic_only_wrong − paraphrase_only_wrong`.
4. Repeat steps 2–3 **10,000 times** (configurable via `--n-boot`).
5. From the resulting 10,000 values of `diff`, take:
   - **95% CI** = the [2.5th, 97.5th] percentile range.
   - **Bootstrap p-value** = 2 × (fraction of resamples landing on the
     *opposite* side of zero from the observed diff), capped at 1.0 — a
     two-sided empirical test.

**How to read it:**
- **CI excludes zero** (both bounds same sign) → significant: consistent
  direction holds up across resampled variation.
- **CI includes zero** → not significant: could easily go either way with a
  slightly different sample.
- If McNemar and bootstrap **disagree**, prefer the bootstrap's (typically
  more conservative) verdict — this is called out explicitly in the
  generated report's Conclusion section.

**Output:** `{model}_bootstrap.csv` / `.md`, the second subtitle line in the
sole-cause bar chart, and the `## Non-parametric bootstrap check` section +
Conclusion addendum in `{model}_chi2_report.md`.

---

## 7. Pooled analysis across models (slide: "What if we combine all models?")

When more than one `--model` is passed, all of the above (McNemar +
bootstrap) is repeated once more on the **concatenation** of every model's
aligned items — i.e. summing b and c across models for McNemar, and
resampling from the full pooled item list for the bootstrap.

**Important caveat (always stated in the pooled report):** this is a naive
pooling — it treats every model's items as freely interchangeable, which
isn't strictly true (different models aren't independent identically-
distributed draws). Treat the pooled result as **indicative, not
definitive** — a proper analysis would use a mixed-effects model with model
as a random effect.

**Output:** `pooled_sole_cause.{csv,md}`, `pooled_bootstrap.{csv,md}`,
`pooled_sole_cause.html`, `pooled_chi2_report.md`.

---

## 8. Worked numeric example (slide: "Putting it together — qwen3.5-1.5b-instruct on mmlu")

- n = 1085 aligned items (model answered `original` correctly).
- b (idiomatic_only_wrong) = 59, c (paraphrase_only_wrong) = 24.
- **McNemar:** χ² = 13.928, **p = 0.00019** → significant.
- **Bootstrap** (10,000 resamples): observed diff = 35, 95% CI = **[17, 53]**
  (excludes zero), **p = 0.00020** → significant, agreeing with McNemar.
- **Conclusion:** for this model on mmlu, the idiomatic rewrite reliably
  broke more previously-correct answers than the paraphrase rewrite did —
  both tests agree this is a real effect, not noise.

Contrast with `mistral-7b-instruct-v0.3` on the same dataset: b=41, c=38,
McNemar p=0.822, bootstrap CI=[−15, 21] (includes zero), p=0.786 — both
tests agree there's **no real effect** here, just noise around a near-equal
split.

---

## 9. One-slide cheat sheet

| Metric | Question it answers | Formula/procedure | Output |
|---|---|---|---|
| Joint confusion matrix | Full 8-way right/wrong breakdown | Simple counting | `*_confusion.*` |
| Sole-cause counts (b, c) | Did one rewrite alone break it? | Counting restricted to original-correct items | `*_sole_cause.*` |
| McNemar χ²/p | Is b vs c gap real? (theoretical) | `(\|b-c\|-1)² / (b+c)`, χ²(1) | `*_chi2_report.md` |
| Bootstrap CI/p | Is b vs c gap real? (empirical, assumption-free) | 10,000× resample-with-replacement, recompute diff | `*_bootstrap.*` |
| Pooled (all above) | Same questions, across all models combined | Same math on concatenated data | `pooled_*` |
