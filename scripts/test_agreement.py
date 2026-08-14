"""Do the two sole-cause significance tests ever disagree?

Report §Statistical framework says a result counts as robustly significant "only
when both checks agree". This walks every per-model run and recomputes the
continuity-corrected McNemar p from the stored b/c counts, comparing it against
the stored bootstrap p, so the report can state how often the two verdicts
actually differ rather than only what the rule would do if they did.
"""

from pathlib import Path

import click
import pandas as pd
from scipy.stats import chi2

from scripts.gen_report_tables import ALPHA, DATASETS, DISPLAY, DS_LABEL

B_KEY = "idiomatic_only_wrong"
C_KEY = "paraphrase_only_wrong"


def _mcnemar_cc_p(b: int, c: int) -> float:
    """Continuity-corrected McNemar p-value, matching report eq. (2)."""
    if b + c == 0:
        return 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return float(chi2.sf(stat, 1))


def _runs(root: Path, dataset: str):
    """Yield (model_id, b, c, mcnemar_p, bootstrap_p) for each model in the panel."""
    tables = root / dataset / "tables"
    for model_id in DISPLAY:
        sole = tables / f"{model_id}_sole_cause.csv"
        boot = tables / f"{model_id}_bootstrap.csv"
        if not (sole.exists() and boot.exists()):
            continue
        counts: dict[str, int] = (
            pd.read_csv(sole).set_index("variant")["count"].astype(int).to_dict()
        )
        b, c = counts[B_KEY], counts[C_KEY]
        p_boot = float(pd.read_csv(boot)["p_value"].astype(float).to_list()[0])
        yield model_id, b, c, _mcnemar_cc_p(b, c), p_boot


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("analysis/variant_confusion"),
    show_default=True,
    help="Directory holding {dataset}/tables/{model}_{sole_cause,bootstrap}.csv.",
)
def main(root: Path) -> None:
    """Print how many per-model runs the two sole-cause tests disagree on."""
    total = 0
    disagreements = []
    for dataset in DATASETS:
        runs = list(_runs(root, dataset))
        total += len(runs)
        for model_id, b, c, p_mc, p_boot in runs:
            if (p_mc < ALPHA) != (p_boot < ALPHA):
                disagreements.append((dataset, model_id, b, c, p_mc, p_boot))
        click.echo(f"{DS_LABEL[dataset]:6s} {len(runs):2d} runs checked")

    click.echo(f"\n{total} per-model runs; {len(disagreements)} disagreement(s) at alpha={ALPHA}.")
    for dataset, model_id, b, c, p_mc, p_boot in disagreements:
        click.echo(
            f"  {DS_LABEL[dataset]:6s} {DISPLAY[model_id]:16s} b={b} c={c}"
            f"  mcnemar={p_mc:.4f}  bootstrap={p_boot:.4f}"
        )


if __name__ == "__main__":
    main()
