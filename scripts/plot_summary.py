"""Cross-dataset paraphrase/idiom delta summary Plotly figure (ARCHITECTURE §2.7)."""

from pathlib import Path

import click

from analysis.plots import plot_cross_dataset_summary


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
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
    results: Path,
    out_dir: Path,
    n_resamples: int,
    ci: float,
    seed: int,
    force: bool,
) -> None:
    """Build the cross-dataset paraphrase/idiom delta summary Plotly figure."""
    html_path = out_dir / "figures" / "cross_dataset_summary.html"
    if html_path.exists() and not force:
        raise click.ClickException(f"Output file exists (use --force to overwrite): {html_path}")

    written_html = plot_cross_dataset_summary(
        results, out_dir=out_dir, n_resamples=n_resamples, ci=ci, seed=seed
    )
    click.echo(f"Wrote {written_html}")
    click.echo(f"Wrote {out_dir / 'tables' / 'cross_dataset_summary.csv'}")
    click.echo(f"Wrote {out_dir / 'tables' / 'cross_dataset_summary.md'}")
    click.echo(f"Wrote {out_dir / 'tables' / 'cross_dataset_summary.meta.json'}")

    # The `_significant` cut is only written when at least one run has a delta
    # CI clear of zero, so report what actually landed rather than assuming.
    sig_html = out_dir / "figures" / "cross_dataset_summary_significant.html"
    if sig_html.exists():
        click.echo(f"Wrote {sig_html}")
        for suffix in ("csv", "md", "meta.json"):
            click.echo(
                f"Wrote {out_dir / 'tables' / f'cross_dataset_summary_significant.{suffix}'}"
            )
    else:
        click.echo("No run has a delta CI clear of zero — no _significant cut.")


if __name__ == "__main__":
    main()
