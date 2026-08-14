"""Regenerate the two LaTeX result tables embedded in NLP_2024_Reichman/main.tex.

Run from the repo root:  uv run idiom-report-tables
Writes t_summary.tex and t_permodel.tex into analysis/tables/; paste each over the
matching \\begin{table}...\\end{table} block in main.tex (match by \\label). Those
two blocks are machine-generated -- edit this file, never main.tex, or the two
drift apart.

PREREQUISITE: analysis/tables/summary.csv must exist and be current. It is
gitignored, so in a fresh clone regenerate it first:
    uv run idiom-analyze --results results/ --out-dir analysis \\
        --n-resamples 10000 --ci 0.95 --seed 0 --force
(note --out-dir analysis, not analysis/tables: the CLI appends /tables itself).

Replaces the four large tables (three 18-row per-dataset tables + a robustness
table) with two: a 3-row dataset-level summary that folds in the parse-clean
robustness check, and one 18-row per-model table carrying the key quantity
(Delta_idi) for all three datasets at once. Numbers come from the same
artifacts as before, so nothing is hand-transcribed.
"""

import csv
import json
import re
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "analysis" / "tables"
DATASETS = ("sst2", "mmlu", "mnli")
DS_LABEL = {"sst2": "SST-2", "mmlu": "MMLU", "mnli": "MNLI"}

DISPLAY = {
    "qwen3.5-0.5b-instruct": "Qwen2.5-0.5B",
    "olmo-2-1b-instruct": "OLMo-2-1B",
    "qwen3.5-1.5b-instruct": "Qwen2.5-1.5B",
    "stablelm-2-1.6b-chat": "StableLM-2-1.6B",
    "smollm2-1.7b-instruct": "SmolLM2-1.7B",
    "gemma-4-2b-instruct": "Gemma-2-2B",
    "granite-3.1-2b-instruct": "Granite-3.1-2B",
    "qwen3.5-3b-instruct": "Qwen2.5-3B",
    "falcon3-3b-instruct": "Falcon3-3B",
    "phi-4-mini-instruct": "Phi-4-mini-3.8B",
    "h2o-danube3-4b-chat": "H2O-Danube3-4B",
    "yi-1.5-6b-chat": "Yi-1.5-6B",
    "qwen3.5-7b-instruct": "Qwen2.5-7B",
    "falcon3-7b-instruct": "Falcon3-7B",
    "mistral-7b-instruct-v0.3": "Mistral-7B-v0.3",
    "olmo-2-7b-instruct": "OLMo-2-7B",
    "gemma-4-9b-instruct": "Gemma-2-9B",
}
ORDER = list(DISPLAY)  # already ascending in parameter count

# deepseek-r1-distill-qwen-7b is excluded from the report. Its result files and
# per-model variant_confusion artifacts still exist on disk; leaving it out of
# DISPLAY is what drops it from both tables. The pooled sole-cause numbers read
# below must therefore come from a variant_confusion run over these 17 models
# only -- re-run scripts.variant_confusion with this --model list if in doubt.


ALPHA = 0.05
# Below P_FLOOR the table prints "<0.0001"; below P_FOUR_DP it keeps four decimals.
P_FLOOR = 1e-4
P_FOUR_DP = 0.1


def fmt_p(p: float) -> str:
    if p < P_FLOOR:
        return "$<$0.0001"
    return f"{p:.4f}" if p < P_FOUR_DP else f"{p:.2f}"


def pp_body(x: float) -> str:
    """Signed percentage-point value, without the surrounding math delimiters."""
    v = x * 100
    return f"{'+' if v >= 0 else '-'}{abs(v):.2f}"


def pp(x: float) -> str:
    return f"${pp_body(x)}$"


def sole_cause(ds: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    base = REPO / "analysis/variant_confusion" / ds
    for p in sorted((base / "tables").glob("*_sole_cause.csv")):
        model = p.name.replace("_sole_cause.csv", "")
        with p.open(encoding="utf-8") as f:
            counts = {r["variant"]: int(r["count"]) for r in csv.DictReader(f)}
        with (base / "tables" / f"{model}_bootstrap.csv").open(encoding="utf-8") as f:
            boot = next(iter(csv.DictReader(f)))
        rep = (base / "reports" / f"{model}_chi2_report.md").read_text(encoding="utf-8")
        out[model] = {
            "b": counts["idiomatic_only_wrong"],
            "c": counts["paraphrase_only_wrong"],
            "chi2": float(re.search(r"chi2 = ([\d.]+), df", rep).group(1)),
            "mcn_p": float(re.search(r"p-value = ([\d.eE+-]+)", rep).group(1)),
            "boot_p": float(boot["p_value"]),
            "ci_low": float(boot["ci_low"]),
            "ci_high": float(boot["ci_high"]),
        }
    return out


def parse_clean(ds: str) -> pd.DataFrame:
    recs = []
    for path in sorted((REPO / "results" / ds).glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if d["model_id"] not in DISPLAY:
            continue
        pt = d["per_task"]
        keep = [
            it
            for it in pt
            if all(it[v]["parse_status"] == "ok" for v in ("original", "paraphrase", "idiomatic"))
        ]

        def acc(items: list[dict], v: str) -> float:
            return sum(1 for it in items if it[v]["correct"]) / len(items)

        b = sum(1 for it in keep if it["idiomatic"]["correct"] and not it["paraphrase"]["correct"])
        c = sum(1 for it in keep if not it["idiomatic"]["correct"] and it["paraphrase"]["correct"])
        recs.append(
            {
                "model": d["model_id"],
                "dropped": (len(pt) - len(keep)) / len(pt),
                "d_clean": acc(keep, "idiomatic") - acc(keep, "paraphrase"),
                "p_clean": min(1.0, binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0,
            }
        )
    return pd.DataFrame(recs)


def main() -> None:
    summary = pd.read_csv(REPO / "analysis/tables/summary.csv")
    summary = summary[summary.model_id.isin(DISPLAY)]
    OUT.mkdir(parents=True, exist_ok=True)
    digest = {}

    # ---- Table 1: dataset-level summary, including the parse-clean check ----
    rows = []
    for ds in DATASETS:
        sub = summary[summary.dataset == ds]
        sc = sole_cause(ds)
        pooled = sc["pooled"]
        clean = parse_clean(ds)
        sig_neg = int(((sub.delta_idiom_p < ALPHA) & (sub.delta_idiom < 0)).sum())
        sig_pos = int(((sub.delta_idiom_p < ALPHA) & (sub.delta_idiom > 0)).sum())
        cl_neg = int(((clean.p_clean < ALPHA) & (clean.d_clean < 0)).sum())
        cl_pos = int(((clean.p_clean < ALPHA) & (clean.d_clean > 0)).sum())
        rows.append(
            " & ".join(
                [
                    DS_LABEL[ds],
                    f"{len(sub)}",
                    f"{int(sub.n.iloc[0]):,}".replace(",", "{,}"),
                    pp(sub.delta_paraphrase.mean()),
                    pp(sub.delta_idiom.mean()),
                    f"{sig_neg} / {sig_pos}",
                    pp(clean.d_clean.mean()),
                    f"{cl_neg} / {cl_pos}",
                    f"{int(pooled['b'])} / {int(pooled['c'])}",
                    f"{pooled['chi2']:.1f}",
                    fmt_p(pooled["mcn_p"]),
                    fmt_p(pooled["boot_p"]),
                ]
            )
            + r" \\"
        )
        digest[ds] = {
            "sig_neg": sig_neg,
            "sig_pos": sig_pos,
            "cl_neg": cl_neg,
            "cl_pos": cl_pos,
            "mean_clean": round(clean.d_clean.mean() * 100, 2),
            "dropped": round(clean.dropped.mean() * 100, 2),
        }

    t1 = (
        r"""\begin{table}[t]
\centering
\caption{Per-dataset summary over the aligned variant triples. $\Delta$ values are
means across models in percentage points; \emph{sig.} counts models with a
significant negative / positive $\Delta_{\text{idi}}$ ($p < 0.05$, McNemar exact).
\emph{Parse-clean} repeats both after dropping every triple in which any variant
failed to parse. The
pooled sole-cause columns sum $b$ (idiomatic-only-wrong) and $c$
(paraphrase-only-wrong) across models and test the gap with continuity-corrected
McNemar and a 10{,}000-resample paired bootstrap.}
\label{tab:summary}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrcrcrrrr}
\toprule
& & & \multicolumn{3}{c}{All triples} & \multicolumn{2}{c}{Parse-clean}
  & \multicolumn{4}{c}{Pooled sole-cause} \\
\cmidrule(lr){4-6} \cmidrule(lr){7-8} \cmidrule(lr){9-12}
Dataset & Models & $n$ & $\Delta_{\text{par}}$ & $\Delta_{\text{idi}}$ & sig.
  & $\Delta_{\text{idi}}$ & sig. & $b$ / $c$ & $\chi^2$ & $p_{\chi^2}$ & $p_{\text{boot}}$ \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )
    (OUT / "t_summary.tex").write_text(t1, encoding="utf-8")

    # ---- Table 2: per-model Delta_idi, two side-by-side panels ------------
    def row(mid: str) -> str:
        cells = [DISPLAY[mid]]
        for ds in DATASETS:
            sub = summary[(summary.dataset == ds) & (summary.model_id == mid)]
            if sub.empty:
                cells.append("---")
                continue
            r = sub.iloc[0]
            sc = sole_cause(ds)[mid]
            # One math group per cell: \bm reaches inside math mode, \textbf does
            # not, and the dagger has to ride the same group or it detaches.
            body = pp_body(r.delta_idiom)
            if r.delta_idiom_p < ALPHA:
                body = rf"\bm{{{body}}}"
            if sc["mcn_p"] < ALPHA and sc["boot_p"] < ALPHA:
                body += r"^{\dagger}"
            cells.append(f"${body}$")
        return " & ".join(cells)

    # With an odd model count the right panel is one short; pad it with empty
    # cells rather than letting zip() silently drop the last left-panel model.
    half = (len(ORDER) + 1) // 2
    left, right = ORDER[:half], ORDER[half:]
    blank = " & ".join([""] * 4)
    lines = [
        row(a) + " & " + (row(right[i]) if i < len(right) else blank) + r" \\"
        for i, a in enumerate(left)
    ]

    t2 = (
        r"""\begin{table}[t]
\centering
\caption{Per-model $\Delta_{\text{idi}}$ (pp) on each dataset, ordered by
parameter count and split into two panels. \textbf{Bold} marks
$\Delta_{\text{idi}}$ significant at $p < 0.05$ (McNemar exact); $\dagger$ marks
models whose sole-cause gap between $b$ and $c$ is significant on \emph{both}
the McNemar and bootstrap tests, in whichever direction.
Negative values support the hypothesis that idiomatic phrasing costs
accuracy beyond paraphrasing; note that the SST-2 columns are almost entirely
positive.}
\label{tab:permodel}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrr@{\hskip 1.4em}lrrr}
\toprule
Model & SST-2 & MMLU & MNLI & Model & SST-2 & MMLU & MNLI \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )
    (OUT / "t_permodel.tex").write_text(t2, encoding="utf-8")
    print(json.dumps(digest, indent=1))


if __name__ == "__main__":
    main()
