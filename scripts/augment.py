"""Stage 2 CLI — cleaned CSV + augmenter → paraphrase.csv + idiomatic.csv.

Scaffold stub. See README §10 and ARCHITECTURE §2.3.
"""

from pathlib import Path

import click


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input",
    "input_",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to a Stage 1 CSV (e.g. datasets_out/sst2/original.csv).",
)
@click.option(
    "--augmenter",
    type=str,
    required=True,
    help="Registered augmenter model id (see augmentation/registry.py).",
)
def main(input_: Path, augmenter: str) -> None:
    """Stage 2: augment a cleaned CSV into paraphrase + idiomatic variants."""
    raise NotImplementedError(
        f"scripts/augment.py is a scaffold stub (input: {input_}, augmenter: {augmenter})"
    )


if __name__ == "__main__":
    main()
