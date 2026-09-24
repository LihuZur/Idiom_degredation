"""RQ2 mixed-effects CLI — fits the opacity/frequency/scale regression for one dataset.

Fits H2a/H2b/H2c (RQ2 plan §5 step 5, corrected 2026-09-14 — see
`analysis/mixed_effects.py`'s `fit_opacity_frequency_effect` docstring for
why `log_params` needed a standalone main effect and centering):

    correct_idi ~ opacity_c + log_freq + log_params_c + opacity_c:log_params_c
                  + correct_par + (1 | row_id) + (1 | model_id)

Usage:
    uv run idiom-mixed-effects sst2
    uv run idiom-mixed-effects sst2 --accepted-only
"""

from pathlib import Path

import click

from analysis.mixed_effects import build_rq2_dataframe, fit_opacity_frequency_effect
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
    help="Root directory containing `{dataset}/idiom_norms.csv`.",
)
@click.option(
    "--accepted-only",
    is_flag=True,
    default=False,
    help="Restrict to rows with `accepted_pair=True` in `{dataset}/meaning_accept.csv` "
    "(RQ2 plan §8 meaning-drift confound sensitivity check).",
)
def main(dataset: str, results_dir: Path, datasets_dir: Path, *, accepted_only: bool) -> None:
    """Fit the RQ2 opacity/frequency/scale regression for DATASET (e.g. `sst2`)."""
    accepted_ids = None
    if accepted_only:
        accepted_ids = load_accepted_ids(datasets_dir / dataset / "meaning_accept.csv")

    df = build_rq2_dataframe(
        dataset, results_dir=results_dir, datasets_dir=datasets_dir, accepted_ids=accepted_ids
    )
    if df.empty:
        raise click.ClickException(
            f"no rows to fit for {dataset!r} — check that idiom_norms.csv and "
            f"results/{dataset}/*.json both exist and share ids"
        )
    click.echo(
        f"[{dataset}] fitting on {len(df)} (row_id, model_id) observations "
        f"({df['row_id'].nunique()} rows x {df['model_id'].nunique()} models)"
    )

    result = fit_opacity_frequency_effect(df)
    click.echo(f"converged: {result.converged}")
    click.echo(
        f"opacity      coef={result.opacity_coef:+.4f}  "
        f"CI=[{result.opacity_ci[0]:+.4f}, {result.opacity_ci[1]:+.4f}]  p={result.opacity_p:.4g}"
    )
    click.echo(
        f"log_freq     coef={result.log_freq_coef:+.4f}  "
        f"CI=[{result.log_freq_ci[0]:+.4f}, {result.log_freq_ci[1]:+.4f}]  p={result.log_freq_p:.4g}"
    )
    click.echo(
        f"log_params   coef={result.log_params_coef:+.4f}  "
        f"CI=[{result.log_params_ci[0]:+.4f}, {result.log_params_ci[1]:+.4f}]  p={result.log_params_p:.4g}"
    )
    click.echo(
        f"opacity:log_params coef={result.interaction_coef:+.4f}  "
        f"CI=[{result.interaction_ci[0]:+.4f}, {result.interaction_ci[1]:+.4f}]  "
        f"p={result.interaction_p:.4g}"
    )
    click.echo(f"correct_par  coef={result.correct_par_coef:+.4f}  (control)")


if __name__ == "__main__":
    main()
