"""Apply the author's A1 review to the inventory and print before/after counts."""

from pathlib import Path

import click

from data.norms.inventory import (
    DEFAULT_FINAL_PATH,
    DEFAULT_OUT_PATH,
    write_final_inventory,
)

_STATUSES = ("exact", "key", "near", "ambiguous")
_DECISIONS = ("accept", "fix", "reject", "unreviewed")
_REUSE_CAP = 15


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--review",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_OUT_PATH,
    show_default=True,
)
@click.option(
    "--out", type=click.Path(path_type=Path), default=DEFAULT_FINAL_PATH, show_default=True
)
def main(review: Path, out: Path) -> None:
    """Write idiom_inventory_final.csv from the reviewed CSV; print a markdown count table."""
    rows, counts, unreviewed, merged = write_final_inventory(review, out)
    click.echo("| match_status | " + " | ".join(_DECISIONS) + " | total |")
    click.echo("|---|" + "---|" * (len(_DECISIONS) + 1))
    for st in _STATUSES:
        cells = [counts[(st, d)] for d in _DECISIONS]
        click.echo(f"| {st} | " + " | ".join(map(str, cells)) + f" | {sum(cells)} |")
    totals = [sum(counts[(st, d)] for st in _STATUSES) for d in _DECISIONS]
    click.echo("| **all** | " + " | ".join(map(str, totals)) + f" | {sum(totals)} |")
    two_plus = sum(1 for r in rows if int(r["n_example_sentences"]) >= 2)  # noqa: PLR2004
    kept_total = len(rows) + merged
    click.echo(f"\nkept after review: {kept_total} norm idioms")
    click.echo(f"merged duplicates (same MIDAS idiom in both norm studies): {merged}")
    click.echo(f"final inventory: {len(rows)} idioms ({two_plus} with >= 2 example sentences)")
    click.echo(f"capacity at {_REUSE_CAP} uses per idiom: {len(rows) * _REUSE_CAP} items")
    if unreviewed:
        click.echo(
            f"\nWARNING: {len(unreviewed)} rows left blank and EXCLUDED, e.g. {unreviewed[:5]}"
        )
    click.echo(f"wrote {out}")


if __name__ == "__main__":
    main()
