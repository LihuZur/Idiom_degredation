"""Write raw per-variant accuracy tables (CSV + Markdown) for the report/thesis."""

from pathlib import Path
from typing import Literal, cast

import click

from analysis.aggregate import bootstrap_by_run
from analysis.io import write_markdown, write_table
from analysis.raw_accuracy import raw_accuracy_table, sole_cause_table, to_wide


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--results",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("results"),
    show_default=True,
)
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("analysis/tables"))
@click.option(
    "--family",
    type=click.Choice(["legacy", "v2"]),
    default="legacy",
    show_default=True,
    help="legacy = pilot datasets (original/paraphrase/idiomatic); "
    "v2 = original/span_literal/idiomatic.",
)
@click.option(
    "--datasets",
    default=None,
    help="Comma-separated datasets (default: sst2,mmlu,mnli for legacy; "
    "sst2_v2,mnli_v2,mmlu_v2,agnews_v2 for v2).",
)
@click.option("--n-resamples", type=int, default=10_000, show_default=True)
@click.option("--ci", type=float, default=0.95, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def main(
    results: Path,
    out_dir: Path,
    family: str,
    datasets: str | None,
    n_resamples: int,
    ci: float,
    seed: int,
) -> None:
    """Raw accuracy (%) per model x variant with CIs, plus sole-cause b/c counts and McNemar p."""
    if datasets is None:
        datasets = "sst2,mmlu,mnli" if family == "legacy" else "sst2_v2,mnli_v2,mmlu_v2,agnews_v2"
    suffix = "" if family == "legacy" else "_v2"
    wanted = tuple(d.strip() for d in datasets.split(",") if d.strip())
    runs = bootstrap_by_run(
        results,
        n_resamples=n_resamples,
        ci=ci,
        seed=seed,
        family=cast(Literal["legacy", "v2"], family),
    )
    long = raw_accuracy_table(runs, wanted)
    sole = sole_cause_table(runs, wanted)
    write_table(long, out_dir / f"raw_accuracy{suffix}.csv")
    write_table(sole, out_dir / f"sole_cause_counts{suffix}.csv")
    write_markdown(to_wide(long, sole), out_dir / f"raw_accuracy{suffix}.md")
    click.echo(
        f"wrote {out_dir / f'raw_accuracy{suffix}.csv'} and {out_dir / f'raw_accuracy{suffix}.md'}"
    )
