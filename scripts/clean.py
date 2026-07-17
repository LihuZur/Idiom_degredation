"""Stage 1 CLI — clean a raw dataset into `datasets_out/{ds}/original.csv`.

See README §10, ARCHITECTURE §2.2, and STAGE1_PLAN §3.5 for the intended
behaviour.
"""

from pathlib import Path

import click

import data  # noqa: F401  # pyright: ignore[reportUnusedImport]  — triggers @register_dataset (Q11)
from cleaning.config import load_config
from cleaning.pipeline import Pipeline
from data.registry import list_datasets

_CONFIG_DIR = Path("configs/clean")


def _clean_one(config: Path, out_dir: Path, seed: int | None) -> None:
    cfg = load_config(config)
    if seed is not None:
        cfg = cfg.model_copy(update={"seed": seed})

    click.echo(f"[clean] dataset={cfg.dataset}  hf_revision={cfg.hf_revision}")
    pipeline = Pipeline(cfg, out_dir=out_dir, config_path=config)
    out_path = pipeline.run()
    for stage, count in pipeline.last_counts.items():
        click.echo(f"[clean] {stage}={count}")
    click.echo(f"[clean] wrote {out_path}")
    click.echo(f"[clean] wrote {out_path.parent / 'original.meta.json'}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a configs/clean/*.yaml file. Required unless --all is set.",
)
@click.option(
    "--all",
    "all_",
    is_flag=True,
    help=(
        "Clean every registered dataset (see data/registry.py), using "
        f"{_CONFIG_DIR}/<dataset>.yaml for each. Future datasets are picked "
        "up automatically once registered and configured."
    ),
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("datasets_out"),
    show_default=True,
    help="Root output directory; written to <out-dir>/<dataset>/original.csv.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Override the config's seed (CLI takes precedence over YAML).",
)
def main(config: Path | None, all_: bool, out_dir: Path, seed: int | None) -> None:
    """Stage 1: clean a raw dataset (or every registered dataset with --all)."""
    if all_ and config is not None:
        raise click.UsageError("pass either --config or --all, not both.")
    if not all_ and config is None:
        raise click.UsageError("--config is required unless --all is set.")

    if not all_:
        assert config is not None  # narrowed above
        _clean_one(config, out_dir, seed)
        return

    names = list_datasets()
    if not names:
        raise click.ClickException("no datasets registered in data/registry.py.")

    missing = [name for name in names if not (_CONFIG_DIR / f"{name}.yaml").exists()]
    if missing:
        raise click.ClickException(
            f"missing {_CONFIG_DIR}/<dataset>.yaml for registered dataset(s): {missing}"
        )

    for i, name in enumerate(names):
        if i:
            click.echo("")
        _clean_one(_CONFIG_DIR / f"{name}.yaml", out_dir, seed)


if __name__ == "__main__":
    main()
