"""Stage 1 CLI — clean a raw dataset into `datasets_out/{ds}/original.csv`.

Scaffold stub. See README §10 and ARCHITECTURE §2.2 for the intended
behaviour.
"""

from pathlib import Path

import click


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to a configs/clean/*.yaml file.",
)
def main(config: Path) -> None:
    """Stage 1: clean a raw dataset."""
    raise NotImplementedError(f"scripts/clean.py is a scaffold stub (config: {config})")


if __name__ == "__main__":
    main()
