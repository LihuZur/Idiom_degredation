"""Post-Stage-3 CLI — aggregate results/ into tables + Plotly figures.

Scaffold stub. See README §10 and ARCHITECTURE §2.7.
"""

from pathlib import Path

import click


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--results",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results"),
    show_default=True,
    help="Root directory containing results/{dataset}/{model}.json files.",
)
def main(results: Path) -> None:
    """Aggregate results across runs."""
    raise NotImplementedError(f"scripts/analyze.py is a scaffold stub (results: {results})")


if __name__ == "__main__":
    main()
