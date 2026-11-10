"""Write the extra metric tables: unrobustness U, parse-failure rates and rank stability."""

from pathlib import Path

import click

from analysis.extra_metrics import parse_failure_table, rank_stability_table, unrobustness_table
from analysis.io import write_table
from analysis.results import VARIANTS, VARIANTS_V2, discover_results, load_result, variants_for


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--results",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("results"),
    show_default=True,
)
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("analysis/tables"))
@click.option("--family", type=click.Choice(["legacy", "v2"]), default="legacy", show_default=True)
def main(results: Path, out_dir: Path, family: str) -> None:
    """FLUKE-style U, parse-failure rates and model-rank stability across variants."""
    wanted = VARIANTS if family == "legacy" else VARIANTS_V2
    loaded = [
        r
        for r in (load_result(p) for p in discover_results(results))
        if variants_for(r.dataset) == wanted
    ]
    suffix = "" if family == "legacy" else "_v2"
    tables = {
        f"extra_unrobustness{suffix}.csv": unrobustness_table(loaded),
        f"extra_parse_failures{suffix}.csv": parse_failure_table(loaded),
        f"extra_rank_stability{suffix}.csv": rank_stability_table(loaded),
    }
    for name, table in tables.items():
        write_table(table, out_dir / name)
        click.echo(f"wrote {out_dir / name} ({len(table)} rows)")


if __name__ == "__main__":
    main()
