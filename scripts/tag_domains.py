"""Tag inventory idioms with a source domain (needed for AG News cross-domain stratification).

Without `--run` this only reports the plan (number of calls); with `--run` it calls the LLM
(about 14 calls, a few cents) and writes idiom_domains.csv plus a 50-row spot-check file.
"""

import csv
import json
import random
from pathlib import Path

import click

from augmentation.providers.base import build_client
from data.norms.domains import IdiomLine, batches, tag_all, write_tags
from data.norms.inventory import DEFAULT_FINAL_PATH

_INV = DEFAULT_FINAL_PATH.parent


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--final",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_FINAL_PATH,
)
@click.option("--out", type=click.Path(path_type=Path), default=_INV / "idiom_domains.csv")
@click.option("--provider", default="gemini", show_default=True)
@click.option("--model", default="gemini-3.6-flash", show_default=True)
@click.option("--run", is_flag=True, help="Actually call the LLM (costs a few cents).")
def main(final: Path, out: Path, provider: str, model: str, run: bool) -> None:
    """Assign each inventory idiom to sports / finance-business / war-politics / science-tech / everyday."""
    with final.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    lines = [
        IdiomLine(
            r["midas_ids"].split("|")[0],
            json.loads(r["midas_forms"])[0],
            json.loads(r["definitions"])[0].rstrip("."),
        )
        for r in rows
    ]
    click.echo(f"{len(lines)} idioms in {len(batches(lines))} calls")
    if not run:
        click.echo("dry run: pass --run to call the LLM")
        return
    client = build_client(provider, model)
    tags = tag_all(
        lines,
        lambda prompt: client.complete(
            system="", user=prompt, temperature=0.0, max_output_tokens=4096
        ),
    )
    write_tags(out, lines, tags)
    missing = [x.idiom_id for x in lines if x.idiom_id not in tags]
    click.echo(f"tagged {len(tags)} of {len(lines)}; wrote {out}; untagged: {len(missing)}")
    check = random.Random(0).sample(lines, min(50, len(lines)))
    with out.with_name("idiom_domains_check.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idiom_id", "idiom", "tag", "author_ok"])
        for x in check:
            w.writerow([x.idiom_id, x.form, tags.get(x.idiom_id, ""), ""])


if __name__ == "__main__":
    main()
