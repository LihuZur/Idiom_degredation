"""Naive original->idiomatic contrast: the figure a study run without a paraphrase
control arm would have reported (report §4, "The control arm changes what the data
say"). Cross-checked against the Stage 4 summary table so it fails loudly rather
than printing stale numbers if the underlying results change.
"""

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

from analysis.stats import mcnemar_exact_p

# The 14 models the report displays, which is now the whole registry: every model
# dropped from the report has also been removed from the codebase.
DISPLAY = [
    "qwen3.5-0.5b-instruct",
    "olmo-2-1b-instruct",
    "qwen3.5-1.5b-instruct",
    "smollm2-1.7b-instruct",
    "gemma-4-2b-instruct",
    "granite-3.1-2b-instruct",
    "falcon3-3b-instruct",
    "phi-4-mini-instruct",
    "h2o-danube3-4b-chat",
    "yi-1.5-6b-chat",
    "qwen3.5-7b-instruct",
    "falcon3-7b-instruct",
    "mistral-7b-instruct-v0.3",
    "olmo-2-7b-instruct",
]
VARIANTS = ("original", "paraphrase", "idiomatic")
ALPHA = 0.05


def _per_model_naive(results_dir: Path, dataset: str) -> pd.DataFrame:
    """One row per displayed model: accuracies and the naive idiomatic-vs-original gap."""
    rows = []
    for path in sorted((results_dir / dataset).glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["model_id"] not in DISPLAY:
            continue
        per_task = run["per_task"]
        correct = {
            variant: np.array([float(item[variant]["correct"]) for item in per_task])
            for variant in VARIANTS
        }
        rows.append(
            {
                "model": run["model_id"],
                "n_aligned": len(per_task),
                "acc_o": correct["original"].mean(),
                "acc_i": correct["idiomatic"].mean(),
                "naive": correct["idiomatic"].mean() - correct["original"].mean(),
                "naive_p": mcnemar_exact_p(correct["original"], correct["idiomatic"]),
            }
        )
    return pd.DataFrame(rows)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--results",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("results"),
    show_default=True,
    help="Root directory containing results/{dataset}/{model}.json files.",
)
@click.option(
    "--summary",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("analysis/tables/summary.csv"),
    show_default=True,
    help="Stage 4 summary table to cross-check against (regenerate with idiom-analyze).",
)
@click.option(
    "--per-model/--no-per-model",
    default=True,
    show_default=True,
    help="Also print the per-model SST-2 breakdown behind the reported 1-of-14 count.",
)
def main(results: Path, summary: Path, per_model: bool) -> None:
    """Print the control-free contrast per dataset, with its decomposition."""
    shipped = pd.read_csv(summary)
    shipped = shipped[shipped.model_id.isin(DISPLAY)]

    for dataset in ("sst2", "mmlu", "mnli"):
        df = _per_model_naive(results, dataset).merge(
            shipped[shipped.dataset == dataset], left_on="model", right_on="model_id"
        )
        # Recomputed accuracies must match the shipped summary, and the naive gap must
        # equal the two reported deltas chained together (analysis/stats.py).
        assert np.allclose(df.acc_o, df.acc_original), dataset
        assert np.allclose(df.acc_i, df.acc_idiomatic), dataset
        assert np.allclose(df.naive, df.delta_paraphrase + df.delta_idiom), dataset

        sig_pos = int(((df.naive_p < ALPHA) & (df.naive > 0)).sum())
        sig_neg = int(((df.naive_p < ALPHA) & (df.naive < 0)).sum())
        click.echo(
            f"{dataset.upper():5s} n={df.n_aligned.iloc[0]:5d} models={len(df)}"
            f"  d_para={df.delta_paraphrase.mean() * 100:+.2f}"
            f"  d_idi={df.delta_idiom.mean() * 100:+.2f}"
            f"  NAIVE(idi-orig)={df.naive.mean() * 100:+.2f}"
            f"  sig +/-: {sig_pos}/{sig_neg}"
            f"  positive-signed: {int((df.naive > 0).sum())}/{len(df)}"
        )
        if per_model and dataset == "sst2":
            table = df[["model", "naive", "naive_p"]].assign(
                naive=lambda x: (x.naive * 100).round(2)
            )
            click.echo(table.to_string(index=False))


if __name__ == "__main__":
    main()
