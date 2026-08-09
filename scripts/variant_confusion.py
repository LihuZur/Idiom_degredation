"""Cross-variant confusion analysis for Stage 3 results (ARCHITECTURE §2.7 adjacent).

For a list of datasets and a list of models, loads each model's
``results/{dataset}/{model}.json``, restricts to items present in all three
variants (``original``, ``paraphrase``, ``idiomatic``), and for every
dataset writes a report folder containing:

- the full joint (2x2x2) per-variant correctness confusion matrix per model,
  as a CSV/Markdown table and a colored Plotly HTML table
- "sole-cause" mistake counts per model: how often is a single variant wrong
  while the other two are correct (i.e. the rewrite alone broke an
  otherwise-correct answer), as a table + colored bar chart
- a Markdown report per model fully explaining the paired McNemar
  significance test comparing idiomatic-only-wrong vs paraphrase-only-wrong
  counts (contingency table, variance derivation, computed statistic, and a
  plain-language conclusion)
- a paired, non-parametric bootstrap CI/p-value (see
  ``bootstrap_diff_ci``) alongside every McNemar test, as a robustness
  check that doesn't rely on the chi-square approximation
- the same sole-cause table/chart/report pooled across all given models for
  that dataset (only when more than one model is given)

Unlike ``analysis/results.py::load_result``, this does **not** enforce the
complete-triple invariant across an entire result file — it simply skips any
``per_task`` item missing a variant, since real Stage 2 runs drop a
significant fraction of rows (validator rejections) for the paraphrase and
idiomatic variants relative to original.

See ``analysis/variant_confusion/METHODOLOGY.md`` for a full, presentation-
ready explanation of every metric computed here (joint confusion matrix,
sole-cause counts, McNemar test, and the bootstrap check), including a
worked numeric example.

MAINTENANCE: any change to the calculation logic in this module (a new
statistic, a changed formula, different pooling behavior, etc.) must be
mirrored in ``analysis/variant_confusion/METHODOLOGY.md`` in the same
change — keep the two in sync.
"""

import csv
import json
from itertools import product
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from scipy.stats import chi2 as chi2_dist

from analysis.io import write_markdown, write_sidecar, write_table

VARIANTS: tuple[str, str, str] = ("original", "paraphrase", "idiomatic")
ALPHA = 0.05  # significance threshold used in the written conclusions
N_BOOT_DEFAULT = 10_000  # default number of bootstrap resamples


def load_per_task(results_dir: Path, dataset: str, model_id: str) -> list[dict[str, Any]]:
    """Load the `per_task` list from a Stage 3 result file.

    Args:
        results_dir: Root directory containing `{dataset}/{model}.json` files.
        dataset: Dataset name (e.g. "sst2", "mmlu", "mnli").
        model_id: Model id, matching the result file's stem.

    Returns:
        The raw `per_task` list of dicts, exactly as written by Stage 3.

    Raises:
        click.ClickException: If no result file exists at the expected path.
    """
    path = results_dir / dataset / f"{model_id}.json"
    if not path.exists():
        raise click.ClickException(f"No result file found at {path}")
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    per_task: list[dict[str, Any]] = data["per_task"]
    return per_task


def aligned_correctness(per_task: list[dict[str, Any]]) -> dict[str, tuple[bool, bool, bool]]:
    """Restrict to items with all 3 variants present.

    Args:
        per_task: The raw `per_task` list from a result file.

    Returns:
        A mapping from task id to `(original_correct, paraphrase_correct,
        idiomatic_correct)`, including only ids where all three variant
        sub-objects are present in the entry.
    """
    out: dict[str, tuple[bool, bool, bool]] = {}
    for entry in per_task:
        original = entry.get("original")
        paraphrase = entry.get("paraphrase")
        idiomatic = entry.get("idiomatic")
        if original is None or paraphrase is None or idiomatic is None:
            continue
        out[str(entry["id"])] = (
            bool(original["correct"]),
            bool(paraphrase["correct"]),
            bool(idiomatic["correct"]),
        )
    return out


def load_row_text(datasets_dir: Path, dataset: str) -> dict[tuple[str, str], dict[str, str]]:
    """Load `{(id, variant): row}` from `datasets_out/{dataset}/*.csv` for source-text lookups.

    Args:
        datasets_dir: Root directory containing `{dataset}/{variant}.csv` files.
        dataset: Dataset name (e.g. "sst2", "mmlu", "mnli").

    Returns:
        A mapping from `(task id, variant name)` to that CSV row as a dict.
        Datasets whose variant CSV is missing are simply skipped.
    """
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for variant in VARIANTS:
        path = datasets_dir / dataset / f"{variant}.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows[(row["id"], variant)] = row
    return rows


def choice_text(meta_json: str, answer_index: int) -> str:
    """Extract the gold-answer text from a row's `meta` JSON blob, if present.

    Args:
        meta_json: The raw `meta` column value (a JSON-encoded object).
        answer_index: Index into `meta["choices"]`, if that key exists.

    Returns:
        The choice text, or `""` if `meta` has no `choices` (e.g. sst2 rows).
    """
    try:
        meta: dict[str, Any] = json.loads(meta_json)
        choices: list[str] = meta["choices"]
        return str(choices[answer_index])
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return ""


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """Paired McNemar test (with continuity correction) on two discordant-pair counts.

    Args:
        b: Count of pairs discordant in favor of the first condition.
        c: Count of pairs discordant in favor of the second condition.

    Returns:
        `(chi2_statistic, p_value)`. `(0.0, 1.0)` if there are no discordant
        pairs (`b + c == 0`).
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / n
    p_value = float(chi2_dist.sf(stat, 1))
    return stat, p_value


def bootstrap_diff_ci(
    items: list[tuple[bool, bool, bool]],
    *,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = 0,
) -> tuple[int, float, float, float]:
    """Non-parametric bootstrap CI/p-value for the sole-cause count difference.

    Complements `mcnemar()` (which relies on a chi-square approximation) with
    an empirical estimate that makes no distributional assumption: resamples
    `items` (whole aligned rows, i.e. paired `(original, paraphrase,
    idiomatic)` correctness triples) with replacement `n_boot` times, and for
    each resample recomputes `idiomatic_only_wrong - paraphrase_only_wrong`.
    The spread of that statistic across resamples approximates its true
    sampling distribution.

    Args:
        items: The aligned `(original, paraphrase, idiomatic)` correctness
            tuples to resample from (e.g. `list(aligned.values())`, or a
            concatenation across models for a pooled bootstrap).
        n_boot: Number of bootstrap resamples to draw.
        seed: RNG seed, for reproducibility across runs.

    Returns:
        `(observed_diff, ci_low, ci_high, p_value)`:
        - `observed_diff`: the actual `idiom_only_wrong - para_only_wrong` on `items`.
        - `ci_low`/`ci_high`: the 2.5th/97.5th percentile of the bootstrap
          distribution of that difference (a 95% percentile CI).
        - `p_value`: a two-sided empirical p-value — twice the fraction of
          resamples landing on the opposite side of zero from `observed_diff`
          (capped at 1.0). `0.0` diff-and-p-value edge case: if `items` is
          empty, returns `(0, 0.0, 0.0, 1.0)`.
    """
    n = len(items)
    if n == 0:
        return 0, 0.0, 0.0, 1.0

    orig = np.array([o for o, _, _ in items], dtype=bool)
    para = np.array([p for _, p, _ in items], dtype=bool)
    idiom = np.array([i for _, _, i in items], dtype=bool)
    sole_idiom_mask = orig & para & ~idiom
    sole_para_mask = orig & ~para & idiom

    observed = int(sole_idiom_mask.sum()) - int(sole_para_mask.sum())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_diffs = sole_idiom_mask[idx].sum(axis=1).astype(np.int64) - sole_para_mask[idx].sum(
        axis=1
    ).astype(np.int64)

    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))

    opposite = int(np.sum(boot_diffs <= 0)) if observed >= 0 else int(np.sum(boot_diffs >= 0))
    p_value = float(min(1.0, 2 * opposite / n_boot))

    return observed, ci_low, ci_high, p_value


def joint_counts(aligned: dict[str, tuple[bool, bool, bool]]) -> dict[tuple[bool, bool, bool], int]:
    """Count occurrences of each of the 8 (original, paraphrase, idiomatic) correctness combos."""
    combos: list[tuple[bool, bool, bool]] = [
        (o, p, i) for o, p, i in product([True, False], repeat=3)
    ]
    counts: dict[tuple[bool, bool, bool], int] = dict.fromkeys(combos, 0)
    for combo in aligned.values():
        counts[combo] += 1
    return counts


def sole_cause_counts(counts: dict[tuple[bool, bool, bool], int]) -> tuple[int, int, int]:
    """Return (idiomatic_only_wrong, paraphrase_only_wrong, original_only_wrong) counts."""
    return (
        counts[(True, True, False)],
        counts[(True, False, True)],
        counts[(False, True, True)],
    )


def confusion_dataframe(counts: dict[tuple[bool, bool, bool], int], n: int) -> pd.DataFrame:
    """Build the joint 8-row confusion table, sorted by count descending."""
    rows: list[dict[str, bool | int | float]] = [
        {
            "original": combo[0],
            "paraphrase": combo[1],
            "idiomatic": combo[2],
            "count": count,
            "pct": 100 * count / n if n else 0.0,
        }
        for combo, count in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
    return pd.DataFrame(rows, columns=["original", "paraphrase", "idiomatic", "count", "pct"])


def sole_cause_dataframe(sole_idiom: int, sole_para: int, sole_orig: int, n: int) -> pd.DataFrame:
    """Build the 3-row sole-cause-mistake summary table."""
    rows: list[dict[str, str | int | float]] = [
        {
            "variant": "idiomatic_only_wrong",
            "count": sole_idiom,
            "pct": 100 * sole_idiom / n if n else 0.0,
        },
        {
            "variant": "paraphrase_only_wrong",
            "count": sole_para,
            "pct": 100 * sole_para / n if n else 0.0,
        },
        {
            "variant": "original_only_wrong",
            "count": sole_orig,
            "pct": 100 * sole_orig / n if n else 0.0,
        },
    ]
    return pd.DataFrame(rows, columns=["variant", "count", "pct"])


def bootstrap_dataframe(
    boot_diff: int, boot_ci_low: float, boot_ci_high: float, boot_p: float, n_boot: int
) -> pd.DataFrame:
    """Build the 1-row bootstrap CI/p-value summary table (companion to `sole_cause_dataframe`)."""
    rows: list[dict[str, int | float]] = [
        {
            "observed_diff": boot_diff,
            "ci_low": boot_ci_low,
            "ci_high": boot_ci_high,
            "p_value": boot_p,
            "n_boot": n_boot,
        }
    ]
    return pd.DataFrame(rows, columns=["observed_diff", "ci_low", "ci_high", "p_value", "n_boot"])


def detail_dataframe(
    aligned: dict[str, tuple[bool, bool, bool]],
    row_text: dict[tuple[str, str], dict[str, str]],
) -> pd.DataFrame:
    """Build a per-item detail table: id, per-variant correctness, and source text."""
    rows: list[dict[str, str | bool]] = []
    for item_id, combo in aligned.items():
        row = row_text.get((item_id, "original"))
        x = row["x"] if row else ""
        gold_answer = choice_text(row["meta"], int(row["y"])) if row and "meta" in row else ""
        rows.append(
            {
                "id": item_id,
                "orig_correct": combo[0],
                "para_correct": combo[1],
                "idiom_correct": combo[2],
                "x": x,
                "gold_answer": gold_answer,
            }
        )
    return pd.DataFrame(
        rows, columns=["id", "orig_correct", "para_correct", "idiom_correct", "x", "gold_answer"]
    )


def _bool_fill_color(is_correct: bool) -> str:
    return "#c6efce" if is_correct else "#ffc7ce"  # Excel-style green / red


def build_confusion_figure(label: str, df: pd.DataFrame) -> go.Figure:
    """Build a colored Plotly table of the joint (original, paraphrase, idiomatic) confusion matrix.

    The three correctness columns are colored green/red (right/wrong); the
    count and percentage columns are colored on a blue heatmap scale by
    magnitude.
    """
    pct_values = df["pct"].tolist()
    max_pct = max(pct_values) if pct_values else 1.0
    pct_colors = sample_colorscale("Blues", [p / max_pct if max_pct else 0.0 for p in pct_values])

    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": [
                        "Original",
                        "Paraphrase",
                        "Idiomatic",
                        "Count",
                        "% of aligned items",
                    ],
                    "fill_color": "#333333",
                    "font": {"color": "white", "size": 12},
                    "align": "center",
                },
                cells={
                    "values": [
                        ["right" if v else "WRONG" for v in df["original"]],
                        ["right" if v else "WRONG" for v in df["paraphrase"]],
                        ["right" if v else "WRONG" for v in df["idiomatic"]],
                        df["count"].tolist(),
                        [f"{p:.2f}%" for p in pct_values],
                    ],
                    "fill_color": [
                        [_bool_fill_color(v) for v in df["original"]],
                        [_bool_fill_color(v) for v in df["paraphrase"]],
                        [_bool_fill_color(v) for v in df["idiomatic"]],
                        pct_colors,
                        pct_colors,
                    ],
                    "align": "center",
                },
            )
        ]
    )
    fig.update_layout(title=f"Joint variant confusion matrix — {label}")
    return fig


def build_sole_cause_figure(
    label: str,
    sole_idiom: int,
    sole_para: int,
    sole_orig: int,
    stat: float,
    p_value: float,
    *,
    boot_ci_low: float | None = None,
    boot_ci_high: float | None = None,
    boot_p: float | None = None,
) -> go.Figure:
    """Build a colored bar chart of sole-cause mistake counts, annotated with the McNemar result.

    If bootstrap results are supplied (`boot_ci_low`/`boot_ci_high`/`boot_p`), a
    second subtitle line reports the bootstrap 95% CI and empirical p-value
    alongside the McNemar chi2/p-value, without altering the McNemar line.
    """
    categories = ["idiomatic-only-wrong", "paraphrase-only-wrong", "original-only-wrong"]
    values = [sole_idiom, sole_para, sole_orig]
    colors = ["#EF553B", "#636EFA", "#B6E880"]
    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=values,
                marker={"color": colors},
                text=values,
                textposition="outside",
            )
        ]
    )
    significance = "significant" if p_value < ALPHA else "not significant"
    subtitle = (
        f"<sup>McNemar (idiomatic-only vs paraphrase-only): "
        f"chi2={stat:.3f}, p={p_value:.5f} ({significance} at alpha={ALPHA})"
    )
    if boot_ci_low is not None and boot_ci_high is not None and boot_p is not None:
        boot_significance = "significant" if boot_p < ALPHA else "not significant"
        subtitle += (
            f"<br>Bootstrap 95% CI on (idiom_only - para_only): "
            f"[{boot_ci_low:.0f}, {boot_ci_high:.0f}], p={boot_p:.5f} "
            f"({boot_significance} at alpha={ALPHA})"
        )
    subtitle += "</sup>"
    fig.update_layout(
        title=(f"Sole-cause mistakes — {label}<br>{subtitle}"),
        yaxis_title="count",
    )
    return fig


def render_chi2_report(
    label: str,
    n: int,
    sole_idiom: int,
    sole_para: int,
    sole_orig: int,
    stat: float,
    p_value: float,
    *,
    pooled: bool,
    boot_ci_low: float,
    boot_ci_high: float,
    boot_p: float,
    n_boot: int,
) -> str:
    """Render a full Markdown explanation of the McNemar test for one model (or pooled).

    Also includes a non-parametric bootstrap CI/p-value section (see
    `bootstrap_diff_ci`) as a robustness check alongside the McNemar test, and
    folds its verdict into the final Conclusion section.
    """
    if sole_idiom > sole_para:
        direction = (
            f"the **idiomatic** rewrite broke a previously-correct answer more often than the "
            f"**paraphrase** rewrite did (b={sole_idiom} vs c={sole_para})"
        )
    elif sole_para > sole_idiom:
        direction = (
            f"the **paraphrase** rewrite broke a previously-correct answer more often than the "
            f"**idiomatic** rewrite did (c={sole_para} vs b={sole_idiom})"
        )
    else:
        direction = f"idiomatic and paraphrase broke a previously-correct answer equally often (b=c={sole_idiom})"

    if p_value < ALPHA:
        conclusion = f"**Statistically significant at alpha={ALPHA}** (p={p_value:.5f} < {ALPHA}): {direction}. "
        if sole_idiom > sole_para:
            conclusion += (
                "This supports the research hypothesis that idiomatic phrasing degrades "
                "accuracy more than plain paraphrasing."
            )
        elif sole_para > sole_idiom:
            conclusion += (
                "This is the OPPOSITE of the research hypothesis (paraphrase degraded "
                "accuracy more than idiomatic here)."
            )
    else:
        conclusion = (
            f"**Not statistically significant at alpha={ALPHA}** (p={p_value:.5f} >= {ALPHA}). "
            f"Although {direction}, with only {sole_idiom + sole_para} discordant items this "
            f"difference could plausibly be due to chance alone — no reliable conclusion should "
            f"be drawn from this result in isolation."
        )
    if pooled:
        conclusion += (
            "\n\n**Pooling caveat:** this pooled test naively sums discordant-pair counts across "
            "multiple models. It is *not* a rigorous mixed-effects test (the models are not "
            "strictly exchangeable independent draws) — treat it as indicative, not definitive."
        )

    boot_diff = sole_idiom - sole_para
    boot_ci_includes_zero = boot_ci_low <= 0 <= boot_ci_high
    if boot_p < ALPHA and not boot_ci_includes_zero:
        boot_agreement = (
            f"The bootstrap check **agrees**: its 95% CI on (idiomatic_only minus "
            f"paraphrase_only) is [{boot_ci_low:.0f}, {boot_ci_high:.0f}], which excludes zero, "
            f"and its empirical p-value ({boot_p:.5f}) is also below alpha={ALPHA}."
        )
    else:
        disagree_note = (
            "When the McNemar test and the bootstrap disagree on significance, prefer the "
            "more conservative (non-significant) verdict, especially for small discordant counts."
            if (p_value < ALPHA) != (boot_p < ALPHA)
            else "This is broadly consistent with the McNemar result above."
        )
        boot_agreement = (
            f"The bootstrap check is **more cautious**: its 95% CI on (idiomatic_only minus "
            f"paraphrase_only) is [{boot_ci_low:.0f}, {boot_ci_high:.0f}] "
            f"({'includes' if boot_ci_includes_zero else 'excludes'} zero), and its empirical "
            f"p-value is {boot_p:.5f}. {disagree_note}"
        )
    conclusion += f"\n\n{boot_agreement}"

    return f"""# McNemar significance report — {label}

## What is being compared

Restricting to the {n} items where this {"pooled set of models" if pooled else "model"} answered
the **original** variant correctly, we compare whether the **paraphrase** or the **idiomatic**
rewrite of that same item broke the answer, while the other rewrite did not:

|                        | idiomatic WRONG | idiomatic right |
|------------------------|------------------|------------------|
| **paraphrase WRONG**   | (both broke it — excluded from the test) | c = {sole_para} |
| **paraphrase right**   | b = {sole_idiom} | (both fine — excluded from the test) |

Only the *discordant* cells (b, c) — where exactly one rewrite broke an otherwise-correct
answer — enter McNemar's test. The concordant cells (both right / both wrong) carry no
information about which rewrite is worse, so they are excluded, as in any paired McNemar test.
For reference, `original_only_wrong` = {sole_orig} (cases where the original itself was wrong
but both rewrites were correct) is reported separately and does not enter this test.

## Test statistic

McNemar's test (with continuity correction) is:

$$\\chi^2 = \\frac{{(|b - c| - 1)^2}}{{b + c}}$$

**Null hypothesis H0:** b and c come from the same underlying rate — i.e. a paraphrase rewrite
and an idiomatic rewrite are equally likely to break an originally-correct answer (each
discordant item is effectively a coin flip, p=0.5, for "idiomatic broke it" vs "paraphrase
broke it").

### Where the formula comes from (variance derivation)

Under H0, with n = b + c discordant items held fixed, the count B (of those n items assigned to
"idiomatic broke it") follows a Binomial(n, 0.5) distribution. Then:

$$\\mathrm{{Var}}(B) = n \\cdot 0.5 \\cdot (1 - 0.5) = n / 4$$

Since C = n - B, the difference B - C = 2B - n, so:

$$\\mathrm{{Var}}(B - C) = 4 \\cdot \\mathrm{{Var}}(B) = n$$

The (uncorrected) test statistic $z = (B - C) / \\sqrt{{\\mathrm{{Var}}(B - C)}} = (B - C) /
\\sqrt{{n}}$ is then approximately standard normal under H0, and $z^2 \\sim \\chi^2(1)$. The
"-1" in $|b - c| - 1$ is Yates' continuity correction, which compensates for approximating a
discrete binomial difference with a continuous chi-square distribution — a standard adjustment
for paired 2x2 designs, especially with small counts.

## Result

- b (idiomatic-only-wrong) = {sole_idiom}
- c (paraphrase-only-wrong) = {sole_para}
- n = b + c = {sole_idiom + sole_para}
- chi2 = {stat:.3f}, df = 1
- p-value = {p_value:.5f}

## Non-parametric bootstrap check

McNemar's test above relies on a chi-square approximation, which can be inaccurate when the
discordant count (b + c) is small. As a robustness check, we instead **resample the aligned
items with replacement** {n_boot} times (paired bootstrap — each resample redraws whole
`(original, paraphrase, idiomatic)` rows, keeping the pairing intact), recomputing
`idiomatic_only_wrong - paraphrase_only_wrong` on each resample. This builds an empirical
sampling distribution with no assumption of normality/chi-square, from which we read off a 95%
percentile confidence interval and a two-sided empirical p-value (the fraction of resamples
landing on the opposite side of zero from the observed difference, doubled).

- observed diff (b - c) = {boot_diff}
- 95% bootstrap CI = [{boot_ci_low:.0f}, {boot_ci_high:.0f}]
- bootstrap p-value = {boot_p:.5f} (n_boot={n_boot})

## Conclusion

{conclusion}
"""


def process_model(
    dataset: str,
    model_id: str,
    results_dir: Path,
    row_text: dict[tuple[str, str], dict[str, str]],
    dataset_out_dir: Path,
    *,
    n_boot: int,
    boot_seed: int,
) -> tuple[int, int, int, int, list[tuple[bool, bool, bool]]]:
    """Run the full per-model analysis and write its table/figure/report outputs.

    Returns:
        `(n, sole_idiom_wrong, sole_para_wrong, sole_orig_wrong, aligned_items)` —
        the last element (the raw aligned correctness tuples) is for the
        caller to pool into a combined bootstrap across models.
    """
    per_task = load_per_task(results_dir, dataset, model_id)
    aligned = aligned_correctness(per_task)
    n = len(aligned)
    if n == 0:
        click.echo(f"{model_id}: no items with all 3 variants present - skipping")
        return 0, 0, 0, 0, []

    counts = joint_counts(aligned)
    sole_idiom, sole_para, sole_orig = sole_cause_counts(counts)
    stat, p_value = mcnemar(sole_idiom, sole_para)
    aligned_items = list(aligned.values())
    _, boot_ci_low, boot_ci_high, boot_p = bootstrap_diff_ci(
        aligned_items, n_boot=n_boot, seed=boot_seed
    )
    verdict = (
        "IDIOMATIC causes more sole mistakes"
        if sole_idiom > sole_para
        else "PARAPHRASE causes more sole mistakes"
        if sole_para > sole_idiom
        else "TIE"
    )
    click.echo(
        f"{model_id}: n={n} idiom_only={sole_idiom} para_only={sole_para} "
        f"chi2={stat:.3f} p={p_value:.5f} boot_ci=[{boot_ci_low:.0f}, {boot_ci_high:.0f}] "
        f"boot_p={boot_p:.5f} -> {verdict}"
    )

    tables_dir = dataset_out_dir / "tables"
    figures_dir = dataset_out_dir / "figures"
    reports_dir = dataset_out_dir / "reports"

    confusion_df = confusion_dataframe(counts, n)
    write_table(confusion_df, tables_dir / f"{model_id}_confusion.csv")
    write_markdown(confusion_df, tables_dir / f"{model_id}_confusion.md")

    sole_df = sole_cause_dataframe(sole_idiom, sole_para, sole_orig, n)
    write_table(sole_df, tables_dir / f"{model_id}_sole_cause.csv")
    write_markdown(sole_df, tables_dir / f"{model_id}_sole_cause.md")

    boot_diff = sole_idiom - sole_para
    boot_df = bootstrap_dataframe(boot_diff, boot_ci_low, boot_ci_high, boot_p, n_boot)
    write_table(boot_df, tables_dir / f"{model_id}_bootstrap.csv")
    write_markdown(boot_df, tables_dir / f"{model_id}_bootstrap.md")

    write_table(detail_dataframe(aligned, row_text), tables_dir / f"{model_id}_detail.csv")

    figures_dir.mkdir(parents=True, exist_ok=True)
    build_confusion_figure(model_id, confusion_df).write_html(
        figures_dir / f"{model_id}_confusion.html"
    )
    build_sole_cause_figure(
        model_id,
        sole_idiom,
        sole_para,
        sole_orig,
        stat,
        p_value,
        boot_ci_low=boot_ci_low,
        boot_ci_high=boot_ci_high,
        boot_p=boot_p,
    ).write_html(figures_dir / f"{model_id}_sole_cause.html")

    report = render_chi2_report(
        model_id,
        n,
        sole_idiom,
        sole_para,
        sole_orig,
        stat,
        p_value,
        pooled=False,
        boot_ci_low=boot_ci_low,
        boot_ci_high=boot_ci_high,
        boot_p=boot_p,
        n_boot=n_boot,
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{model_id}_chi2_report.md").write_text(report, encoding="utf-8")

    return n, sole_idiom, sole_para, sole_orig, aligned_items


def process_pooled(
    models: tuple[str, ...],
    pooled_n: int,
    pooled_idiom: int,
    pooled_para: int,
    pooled_orig: int,
    pooled_items: list[tuple[bool, bool, bool]],
    dataset_out_dir: Path,
    *,
    n_boot: int,
    boot_seed: int,
) -> None:
    """Write the pooled-across-models sole-cause table/figure/report."""
    stat, p_value = mcnemar(pooled_idiom, pooled_para)
    boot_diff, boot_ci_low, boot_ci_high, boot_p = bootstrap_diff_ci(
        pooled_items, n_boot=n_boot, seed=boot_seed
    )
    verdict = (
        "IDIOMATIC causes more sole mistakes overall"
        if pooled_idiom > pooled_para
        else "PARAPHRASE causes more sole mistakes overall"
        if pooled_para > pooled_idiom
        else "TIE overall"
    )
    click.echo(
        f"POOLED ({len(models)} models): n={pooled_n} idiom_only={pooled_idiom} "
        f"para_only={pooled_para} chi2={stat:.3f} p={p_value:.5f} "
        f"boot_ci=[{boot_ci_low:.0f}, {boot_ci_high:.0f}] boot_p={boot_p:.5f} -> {verdict}"
    )

    label = f"pooled ({len(models)} models)"
    sole_df = sole_cause_dataframe(pooled_idiom, pooled_para, pooled_orig, pooled_n)
    write_table(sole_df, dataset_out_dir / "tables" / "pooled_sole_cause.csv")
    write_markdown(sole_df, dataset_out_dir / "tables" / "pooled_sole_cause.md")

    boot_df = bootstrap_dataframe(boot_diff, boot_ci_low, boot_ci_high, boot_p, n_boot)
    write_table(boot_df, dataset_out_dir / "tables" / "pooled_bootstrap.csv")
    write_markdown(boot_df, dataset_out_dir / "tables" / "pooled_bootstrap.md")

    figures_dir = dataset_out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    build_sole_cause_figure(
        label,
        pooled_idiom,
        pooled_para,
        pooled_orig,
        stat,
        p_value,
        boot_ci_low=boot_ci_low,
        boot_ci_high=boot_ci_high,
        boot_p=boot_p,
    ).write_html(figures_dir / "pooled_sole_cause.html")

    report = render_chi2_report(
        label,
        pooled_n,
        pooled_idiom,
        pooled_para,
        pooled_orig,
        stat,
        p_value,
        pooled=True,
        boot_ci_low=boot_ci_low,
        boot_ci_high=boot_ci_high,
        boot_p=boot_p,
        n_boot=n_boot,
    )
    reports_dir = dataset_out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "pooled_chi2_report.md").write_text(report, encoding="utf-8")


def process_dataset(
    dataset: str,
    models: tuple[str, ...],
    results_dir: Path,
    datasets_dir: Path,
    out_dir: Path,
    *,
    force: bool,
    n_boot: int,
    boot_seed: int,
) -> None:
    """Run the full per-model (+ pooled) analysis for one dataset and write all outputs."""
    dataset_out_dir = out_dir / dataset
    if dataset_out_dir.exists() and not force:
        raise click.ClickException(
            f"Output dir exists (use --force to overwrite): {dataset_out_dir}"
        )

    click.echo(f"\n{'=' * 70}\nDataset: {dataset}")
    row_text = load_row_text(datasets_dir, dataset)

    pooled_n = pooled_idiom = pooled_para = pooled_orig = 0
    pooled_items: list[tuple[bool, bool, bool]] = []
    for model_id in models:
        n, sole_idiom, sole_para, sole_orig, aligned_items = process_model(
            dataset,
            model_id,
            results_dir,
            row_text,
            dataset_out_dir,
            n_boot=n_boot,
            boot_seed=boot_seed,
        )
        pooled_n += n
        pooled_idiom += sole_idiom
        pooled_para += sole_para
        pooled_orig += sole_orig
        pooled_items.extend(aligned_items)

    if len(models) > 1:
        process_pooled(
            models,
            pooled_n,
            pooled_idiom,
            pooled_para,
            pooled_orig,
            pooled_items,
            dataset_out_dir,
            n_boot=n_boot,
            boot_seed=boot_seed,
        )

    write_sidecar(
        {"dataset": dataset, "models": list(models), "results_dir": str(results_dir)},
        dataset_out_dir / "provenance.json",
    )
    click.echo(f"Wrote outputs to {dataset_out_dir}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--dataset",
    "datasets",
    required=True,
    multiple=True,
    help="Registered dataset name (repeat --dataset for multiple, e.g. sst2, mmlu, mnli).",
)
@click.option(
    "--model",
    "models",
    required=True,
    multiple=True,
    help="Model id to include (repeat --model for multiple). Must match results/{dataset}/{model}.json.",
)
@click.option(
    "--results-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results"),
    show_default=True,
    help="Root directory containing results/{dataset}/{model}.json files.",
)
@click.option(
    "--datasets-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("datasets_out"),
    show_default=True,
    help="Root directory containing datasets_out/{dataset}/{variant}.csv files (used for source-text lookups).",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("analysis/variant_confusion"),
    show_default=True,
    help="Root output directory; one subfolder per dataset is created under it.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite an existing dataset output folder instead of raising an error.",
)
@click.option(
    "--n-boot",
    type=int,
    default=N_BOOT_DEFAULT,
    show_default=True,
    help="Number of bootstrap resamples for the non-parametric CI/p-value check.",
)
@click.option(
    "--boot-seed",
    type=int,
    default=0,
    show_default=True,
    help="RNG seed for the bootstrap resampling, for reproducibility.",
)
def main(
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    results_dir: Path,
    datasets_dir: Path,
    out_dir: Path,
    force: bool,
    n_boot: int,
    boot_seed: int,
) -> None:
    """Write per-variant confusion tables, colored figures, and McNemar reports for each dataset."""
    for dataset in datasets:
        process_dataset(
            dataset,
            models,
            results_dir,
            datasets_dir,
            out_dir,
            force=force,
            n_boot=n_boot,
            boot_seed=boot_seed,
        )


if __name__ == "__main__":
    main()
