"""RQ1 mixed-effects CLI — fits the idiomatic-vs-paraphrase arm effect for one dataset.

Fits (RQ1 plan §5 step 8):

    is_correct ~ arm + (1 | row_id) + (1 | model_id)

restricted to `paraphrase` vs `idiomatic` — the course paper's established
headline metric `Δ_idi = acc_idi - acc_par` ("extra cost of idiomaticity",
measured over and above a non-idiomatic rewrite, NOT vs the raw original).

Usage:
    uv run idiom-arm-effect sst2
    uv run idiom-arm-effect sst2 --accepted-only
"""

from pathlib import Path

import click

from analysis.mixed_effects import build_arm_dataframe, fit_arm_effect
from analysis.results import load_accepted_ids


@click.command()
@click.argument("dataset")
@click.option(
    "--results-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results"),
    show_default=True,
    help="Root directory containing `{dataset}/{model_id}.json` Stage 3 result files.",
)
@click.option(
    "--datasets-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("datasets_out"),
    show_default=True,
    help="Root directory containing `{dataset}/meaning_accept.csv` (for --accepted-only).",
)
@click.option(
    "--accepted-only",
    is_flag=True,
    default=False,
    help="Restrict to rows with `accepted_pair=True` in `{dataset}/meaning_accept.csv` "
    "(RQ1 plan step 5's meaning-validated subset — the chapter's primary analysis).",
)
def main(dataset: str, results_dir: Path, datasets_dir: Path, *, accepted_only: bool) -> None:
    """Fit the RQ1 idiomatic-vs-original arm effect for DATASET (e.g. `sst2`)."""
    accepted_ids = None
    if accepted_only:
        accepted_ids = load_accepted_ids(datasets_dir / dataset / "meaning_accept.csv")

    df = build_arm_dataframe(dataset, results_dir=results_dir, accepted_ids=accepted_ids)
    if df.empty:
        raise click.ClickException(
            f"no rows to fit for {dataset!r} — check that results/{dataset}/*.json exist"
        )
    click.echo(
        f"[{dataset}] fitting on {len(df)} (row_id, model_id, arm) observations "
        f"({df['row_id'].nunique()} rows x {df['model_id'].nunique()} models x 2 arms)"
    )

    result = fit_arm_effect(df)
    click.echo(f"converged: {result.converged}")
    click.echo(
        f"arm (paraphrase vs idiomatic) coef={result.coef:+.4f}  "
        f"CI=[{result.ci_low:+.4f}, {result.ci_high:+.4f}]  p={result.p_value:.4g}"
    )


if __name__ == "__main__":
    main()
