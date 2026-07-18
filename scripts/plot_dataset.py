"""Per-dataset cross-model Plotly figure over all three variants (ARCHITECTURE §2.7)."""

from pathlib import Path

import click

from analysis.plots import plot_dataset_cross_model
from analysis.results import discover_results, load_result


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--dataset", type=str, default=None, help="Registered dataset name.")
@click.option(
    "--all", "all_datasets", is_flag=True, default=False, help="Plot every dataset found."
)
@click.option(
    "--results",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results"),
    show_default=True,
    help="Root directory containing results/{dataset}/{model}.json files.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("analysis"),
    show_default=True,
    help="Root output directory; figures go to out-dir/figures, tables to out-dir/tables.",
)
@click.option(
    "--n-resamples",
    type=int,
    default=10000,
    show_default=True,
    help="Number of bootstrap resamples to draw per run.",
)
@click.option(
    "--ci",
    type=float,
    default=0.95,
    show_default=True,
    help="Confidence level in (0, 1) for the bootstrap intervals.",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="RNG seed for bootstrap resampling.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing output files.",
)
def main(
    dataset: str | None,
    all_datasets: bool,
    results: Path,
    out_dir: Path,
    n_resamples: int,
    ci: float,
    seed: int,
    force: bool,
) -> None:
    """Build the per-dataset cross-model Plotly figure(s)."""
    if bool(dataset) == all_datasets:
        raise click.ClickException("Pass exactly one of --dataset or --all.")

    if all_datasets:
        datasets = sorted({load_result(path).dataset for path in discover_results(results)})
        if not datasets:
            raise click.ClickException(f"No result files found under {results}")
    else:
        assert dataset is not None
        datasets = [dataset]

    for ds in datasets:
        html_path = out_dir / "figures" / f"{ds}_cross_model.html"
        if html_path.exists() and not force:
            raise click.ClickException(
                f"Output file exists (use --force to overwrite): {html_path}"
            )

    for ds in datasets:
        written_html = plot_dataset_cross_model(
            ds, results, out_dir=out_dir, n_resamples=n_resamples, ci=ci, seed=seed
        )
        click.echo(f"Wrote {written_html}")
        click.echo(f"Wrote {out_dir / 'tables' / f'{ds}_cross_model.csv'}")
        click.echo(f"Wrote {out_dir / 'tables' / f'{ds}_cross_model.md'}")
        click.echo(f"Wrote {out_dir / 'tables' / f'{ds}_cross_model.meta.json'}")


if __name__ == "__main__":
    main()
