"""Scale- and capability-trend statistics: the Spearman correlations behind report
§5, "No capability or scale trend", plus the binomial figures behind the
multiple-comparisons paragraph of §Threats to validity.

Parameter counts are the one quantity the report reasons about that no artifact
carries -- results JSONs record hf_repo and revision, not model size -- so they are
declared here and checked against the display order rather than left implicit.
Cross-checked against the Stage 4 per-dataset tables so it fails loudly rather
than printing stale numbers if the underlying results change.
"""

from pathlib import Path
from typing import TypedDict

import click
import numpy as np
import pandas as pd
from scipy.stats import binom, spearmanr, t

from scripts.gen_report_tables import ALPHA, DATASETS, DS_LABEL, ORDER


class Trend(TypedDict):
    """Spearman results and significant-model counts for one dataset."""

    n: int
    rho_acc: float
    p_acc: float
    rho_par: float
    p_par: float
    sig_neg: int
    sig_pos: int


# Billions of parameters, keyed by registry id. Sources are each model's own card;
# where a name states the size (Qwen2.5-0.5B) that is the figure used, and for
# Phi-4-mini the card's 3.8B rather than the "mini" label.
PARAMS_B = {
    "qwen3.5-0.5b-instruct": 0.5,
    "olmo-2-1b-instruct": 1.0,
    "qwen3.5-1.5b-instruct": 1.5,
    "smollm2-1.7b-instruct": 1.7,
    "gemma-4-2b-instruct": 2.0,
    "granite-3.1-2b-instruct": 2.0,
    "falcon3-3b-instruct": 3.0,
    "phi-4-mini-instruct": 3.8,
    "h2o-danube3-4b-chat": 4.0,
    "yi-1.5-6b-chat": 6.0,
    "qwen3.5-7b-instruct": 7.0,
    "falcon3-7b-instruct": 7.0,
    "mistral-7b-instruct-v0.3": 7.0,
    "olmo-2-7b-instruct": 7.0,
}

# The parameter range the report calls "14-fold" (7B / 0.5B).
REPORTED_FOLD = 14.0


def _min_detectable_rho(n: int, alpha: float) -> float:
    """Smallest |rho| a two-sided Spearman test at this n would call significant.

    Uses the t approximation rho*sqrt(n-2)/sqrt(1-rho^2) ~ t(n-2), inverted.
    """
    crit = t.ppf(1 - alpha / 2, n - 2)
    return float(crit / np.sqrt(n - 2 + crit**2))


def _trends(tables: Path, dataset: str) -> Trend:
    """Spearman rho/p for delta_idiom against original accuracy and parameter count."""
    df = pd.read_csv(tables / f"{dataset}_cross_model.csv")
    assert set(df.model_id) == set(PARAMS_B), f"{dataset}: panel != PARAMS_B keys"
    params = df.model_id.map(PARAMS_B)

    rho_acc, p_acc = spearmanr(df.delta_idiom, df.acc_original)
    rho_par, p_par = spearmanr(df.delta_idiom, params)
    sig = df.delta_idiom_p < ALPHA
    return Trend(
        n=len(df),
        rho_acc=float(rho_acc),
        p_acc=float(p_acc),
        rho_par=float(rho_par),
        p_par=float(p_par),
        sig_neg=int((sig & (df.delta_idiom < 0)).sum()),
        sig_pos=int((sig & (df.delta_idiom > 0)).sum()),
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--tables",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("analysis/tables"),
    show_default=True,
    help="Stage 4 table directory holding {dataset}_cross_model.csv.",
)
def main(tables: Path) -> None:
    """Print the scale/capability correlations and their power and multiplicity context."""
    # The report orders Table 2 by parameter count and calls the panel a 14-fold
    # range; both are properties of PARAMS_B, so assert them rather than trust them.
    sizes = [PARAMS_B[mid] for mid in ORDER]
    assert sizes == sorted(sizes), "ORDER is not ascending in parameter count"
    fold = max(sizes) / min(sizes)
    assert fold == REPORTED_FOLD, f"parameter range is {fold:g}-fold, report says 14-fold"

    rows = {ds: _trends(tables, ds) for ds in DATASETS}
    n = rows[DATASETS[0]]["n"]
    floor = _min_detectable_rho(n, ALPHA)

    click.echo(
        f"Panel: {n} models, {min(sizes)}B-{max(sizes)}B ({fold:g}-fold range).\n"
        f"Smallest |rho| detectable at n={n}, alpha={ALPHA}: {floor:.2f}\n"
    )
    for ds, r in rows.items():
        click.echo(
            f"{DS_LABEL[ds]:6s} d_idi vs orig-acc: rho={r['rho_acc']:+.2f} p={r['p_acc']:.2f}"
            f"   vs params: rho={r['rho_par']:+.2f} p={r['p_par']:.2f}"
            f"   {'(both under the detection floor)' if max(abs(r['rho_acc']), abs(r['rho_par'])) < floor else ''}"
        )

    strongest = max(
        (abs(r[key]), ds, label)
        for ds, r in rows.items()
        for key, label in (("rho_par", "params"), ("rho_acc", "orig-acc"))
    )
    click.echo(
        f"\nLargest |rho| of the six: {strongest[0]:.2f} ({DS_LABEL[strongest[1]]} vs {strongest[2]})"
    )

    # Multiple comparisons: how surprising is each dataset's count of individually
    # significant models under a true null of 14 independent tests at alpha?
    click.echo(f"\nBinomial null, {n} tests at alpha={ALPHA} (expected {n * ALPHA:.1f} hits):")
    for ds, r in rows.items():
        for label, count in (("neg", r["sig_neg"]), ("pos", r["sig_pos"])):
            if count:
                click.echo(
                    f"  {DS_LABEL[ds]:6s} {count:2d} significant {label}"
                    f"  P(X>={count}) = {binom.sf(count - 1, n, ALPHA):.3g}"
                )


if __name__ == "__main__":
    main()
