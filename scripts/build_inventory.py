"""Build the closed idiom inventory (MIDAS-EN x human norms) for the v2 generator."""

from collections import Counter
from pathlib import Path

import click

from data.norms.inventory import DEFAULT_OUT_PATH, write_inventory


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--out", type=click.Path(path_type=Path), default=DEFAULT_OUT_PATH, show_default=True)
def main(out: Path) -> None:
    """Match norm idioms to MIDAS entries and write idiom_inventory.csv for review (A1)."""
    rows, unmatched = write_inventory(out)
    by_status = Counter(r["match_status"] for r in rows)
    click.echo(f"matched {len(rows)} norm idioms: {dict(by_status)}")
    click.echo(f"unmatched norm idioms: {len(unmatched)}")
    click.echo(f"wrote {out} (fill the match_checked column: accept / reject / fix)")


if __name__ == "__main__":
    main()
