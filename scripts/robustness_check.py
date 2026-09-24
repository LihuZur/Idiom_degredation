"""RQ1 meaning-preservation robustness check — fits the arm effect across 4
judge-filtering subsets: all rows, both-judges-accept, OpenAI-only-accept,
Claude-only-accept.

Generalizes the ad-hoc analyses run for sst2/mmlu's `Delta_idi` (paraphrase vs
idiomatic) and `Delta_par` (original vs paraphrase) tables in
`local/RQ_STATUS.md`. A judge's "accept" for a given arm pair requires that
judge's verdict to be `True` on every arm in the pair that has a
meaning-preservation column (`original` has none -- it is unmodified, so
`--arms original,paraphrase` only checks the `paraphrase` arm's verdicts).

Usage:
    uv run idiom-robustness sst2
    uv run idiom-robustness mmlu --arms paraphrase,idiomatic
    uv run idiom-robustness sst2 --arms original,paraphrase
"""

from pathlib import Path
from typing import cast

import click
import pandas as pd

from analysis.mixed_effects import build_arm_dataframe, fit_arm_effect

# Maps a `build_arm_dataframe` arm name to the `meaning_accept.csv` column suffix.
# `original` is intentionally absent -- it is unmodified, so has no judge verdict.
ARM_SUFFIX = {"paraphrase": "par", "idiomatic": "idi"}

_N_ARMS = 2


def _judge_accept_mask(
    accept_df: pd.DataFrame, judge_substr: str, arms: tuple[str, str]
) -> pd.Series:
    cols = [
        col
        for arm in arms
        if (suffix := ARM_SUFFIX.get(arm)) is not None
        for col in accept_df.columns
        if col.endswith(f"__{suffix}") and judge_substr in col
    ]
    if not cols:
        raise click.ClickException(f"no meaning_accept.csv columns matched judge {judge_substr!r}")
    return cast(pd.Series, accept_df[cols].eq(True).all(axis=1))


@click.command()
@click.argument("dataset")
@click.option(
    "--arms",
    default="paraphrase,idiomatic",
    show_default=True,
    help="Comma-separated pair of arms to compare (e.g. 'original,paraphrase' for Delta_par).",
)
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
    help="Root directory containing `{dataset}/meaning_accept.csv`.",
)
def main(dataset: str, arms: str, results_dir: Path, datasets_dir: Path) -> None:
    """Fit the DATASET arm effect across 4 meaning-preservation filtering subsets."""
    arm_pair = tuple(a.strip() for a in arms.split(","))
    if len(arm_pair) != _N_ARMS:
        raise click.ClickException(
            f"--arms must be exactly two comma-separated names, got {arms!r}"
        )

    accept_csv = datasets_dir / dataset / "meaning_accept.csv"
    if not accept_csv.exists():
        raise click.ClickException(f"no {accept_csv} found -- run idiom-accept-rule first")
    acc = pd.read_csv(accept_csv, dtype={"id": str})

    subsets: dict[str, frozenset[str] | None] = {
        "all": None,
        "both_judges_accept": frozenset(acc.loc[acc["accepted_pair"], "id"]),
        "openai_only_accept": frozenset(acc.loc[_judge_accept_mask(acc, "openai", arm_pair), "id"]),
        "claude_only_accept": frozenset(
            acc.loc[_judge_accept_mask(acc, "anthropic", arm_pair), "id"]
        ),
    }

    click.echo(f"=== {dataset}  (arms: {arm_pair[0]} vs {arm_pair[1]}) ===")
    for name, ids in subsets.items():
        df = build_arm_dataframe(dataset, results_dir=results_dir, accepted_ids=ids, arms=arm_pair)
        if df.empty:
            click.echo(f"{name:24s} -- no rows, skipped")
            continue
        n_rows = df["row_id"].nunique()
        result = fit_arm_effect(df)
        click.echo(
            f"{name:24s} n_rows={n_rows:5d} obs={len(df):6d} "
            f"coef={result.coef:+.4f} CI=[{result.ci_low:+.4f},{result.ci_high:+.4f}] "
            f"p={result.p_value:.4g} converged={result.converged}"
        )


if __name__ == "__main__":
    main()
