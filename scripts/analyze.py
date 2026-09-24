"""Post-Stage-3 CLI — aggregate results/ into summary tables (ARCHITECTURE §2.7)."""

from pathlib import Path

import click

from analysis.aggregate import aggregate
from analysis.io import build_provenance, write_markdown, write_sidecar, write_table
from analysis.results import discover_results, load_result


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--results",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results"),
    show_default=True,
    help="Root directory containing results/{dataset}/{model}.json files.",
)
@click.option(
    "--datasets-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("datasets_out"),
    show_default=True,
    help="Root directory containing datasets_out/{dataset}/meaning_accept.csv "
    "files (only consulted with --accepted-only).",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("analysis"),
    show_default=True,
    help="Root output directory; tables are written to out-dir/tables.",
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
@click.option(
    "--accepted-only",
    is_flag=True,
    default=False,
    help=(
        "Restrict each run to ids with accepted_pair=True in "
        "datasets_out/{dataset}/meaning_accept.csv (RQ1 §5 step 7). Writes to "
        "summary_accepted.{csv,md,meta.json} so the original, unfiltered "
        "summary is never overwritten."
    ),
)
def main(
    results: Path,
    datasets_dir: Path,
    out_dir: Path,
    n_resamples: int,
    ci: float,
    seed: int,
    force: bool,
    accepted_only: bool,
) -> None:
    """Aggregate results across runs into summary.csv / summary.md / summary.meta.json."""
    tables_dir = out_dir / "tables"
    stem = "summary_accepted" if accepted_only else "summary"
    csv_path = tables_dir / f"{stem}.csv"
    md_path = tables_dir / f"{stem}.md"
    meta_path = tables_dir / f"{stem}.meta.json"

    if csv_path.exists() and not force:
        raise click.ClickException(f"Output file exists (use --force to overwrite): {csv_path}")

    df = aggregate(
        results,
        n_resamples=n_resamples,
        ci=ci,
        seed=seed,
        accepted_only=accepted_only,
        datasets_dir=datasets_dir,
    )
    result_files = [load_result(path) for path in discover_results(results)]
    provenance = build_provenance(result_files, n_resamples=n_resamples, ci=ci, seed=seed)
    provenance["accepted_only"] = accepted_only

    write_table(df, csv_path)
    write_markdown(df, md_path)
    write_sidecar(provenance, meta_path)

    click.echo(f"Wrote {csv_path}")
    click.echo(f"Wrote {md_path}")
    click.echo(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
