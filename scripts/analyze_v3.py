"""v3 analysis CLI (THESIS_PLAN_v3 §5): `idiom-analyze-v3 --suffix _v3poc`.

Main test per task (mixed-effects logistic, Holm over tasks), per-model check (McNemar + idiom-cluster bootstrap),
accuracy and parse rates. Writes CSV tables to --out-dir.
"""

from pathlib import Path

import click
import pandas as pd

from analysis.v3_analysis import (
    SECONDARY_TERMS,
    accuracy_table,
    fit_main,
    fit_secondary,
    holm,
    load_long,
    per_model,
)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--results-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("results"),
)
@click.option(
    "--suffix", default="_v3poc", show_default=True, help="Dataset name suffix, e.g. _v3poc."
)
@click.option("--tasks", default="sst2,mnli,mmlu,agnews", show_default=True)
@click.option(
    "--idiom-effect", is_flag=True, help="Add an idiom random effect to the main test (plan O27)."
)
@click.option(
    "--valid-from",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Review CSV with `key` (task:id) and `action`: analyse only the rows marked valid (e.g. the pairs that "
    "pass both the filter and a reader; POC-2 clean subset).",
)
@click.option("--n-boot", type=int, default=5000, show_default=True)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("analysis/tables/v3poc"),
    show_default=True,
)
def main(
    results_dir: Path,
    suffix: str,
    tasks: str,
    idiom_effect: bool,
    valid_from: Path | None,
    n_boot: int,
    out_dir: Path,
) -> None:
    """Analyse v3 results: main test, per-model check, accuracy / parse rates."""
    frames = []
    for t in tasks.split(","):
        ds = f"{t}{suffix}"
        if not (results_dir / ds).exists():
            click.echo(f"skip {ds}: no results")
            continue
        frames.append(load_long(results_dir, ds))
    if not frames:
        raise click.ClickException("no results found")
    df = pd.concat(frames, ignore_index=True)
    if valid_from is not None:
        review = pd.read_csv(valid_from, dtype=str)
        valid = set(review.loc[review["action"] == "valid", "key"].tolist())
        keys = df["dataset"].str.replace(suffix, "", regex=False) + ":" + df["item"]
        before = df["item"].nunique()
        df = pd.DataFrame(df[keys.isin(valid)]).reset_index(drop=True)
        click.echo(f"--valid-from: kept {df['item'].nunique()} of {before} items")
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = accuracy_table(df)
    acc.to_csv(out_dir / "accuracy_parse.csv", index=False)
    main_rows = [fit_main(g, idiom_effect=idiom_effect) for _, g in df.groupby("dataset")]
    for r, p in zip(main_rows, holm([r["p_approx"] for r in main_rows]), strict=True):
        r["p_holm"] = p
    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(out_dir / "main_test.csv", index=False)
    sens = [fit_main(g, idiom_effect=True) for _, g in df.groupby("dataset")]
    pd.DataFrame(sens).to_csv(out_dir / "main_test_with_idiom_effect.csv", index=False)
    sec = pd.DataFrame([r for t in SECONDARY_TERMS for r in fit_secondary(df, t)])
    sec.to_csv(out_dir / "secondary.csv", index=False)
    pm = per_model(df, n_boot=n_boot)
    pm.to_csv(out_dir / "per_model.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        click.echo(
            "== main test (per task)\n" + main_df.drop(columns=["estimator"]).to_string(index=False)
        )
        click.echo(
            "\n== main test with an idiom random effect (sensitivity, O27)\n"
            + pd.DataFrame(sens)[
                ["dataset", "log_odds_idiomatic", "ci_low", "ci_high", "p_approx"]
            ].to_string(index=False)
        )
        click.echo("\n== secondary questions (exploratory)\n" + sec.to_string(index=False))
        click.echo("\n== per-model check\n" + pm.to_string(index=False))
    click.echo(f"\ntables -> {out_dir}")


if __name__ == "__main__":
    main()
