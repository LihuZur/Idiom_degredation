"""Stage 2 CLI — augment a cleaned dataset into paraphrase + idiomatic variants.

See README §10, ARCHITECTURE §2.3, and STAGE2_CONTRACT for the intended
behaviour.
"""

from pathlib import Path

import click

import augmentation.anthropic_augmenter  # pyright: ignore[reportUnusedImport]  — triggers @register_augmenter
import augmentation.gemini_augmenter  # pyright: ignore[reportUnusedImport]  — triggers @register_augmenter
import augmentation.llm_validators  # pyright: ignore[reportUnusedImport]  — triggers @register_validator
import augmentation.openai_augmenter  # pyright: ignore[reportUnusedImport]  — triggers @register_augmenter
import augmentation.validators  # noqa: F401  # pyright: ignore[reportUnusedImport]  — triggers @register_validator
import data  # noqa: F401  # pyright: ignore[reportUnusedImport]  — triggers @register_dataset (needed by --all)
from augmentation.config import load_config
from augmentation.pipeline import AugmentPipeline
from data.registry import list_datasets

_CONFIG_DIR = Path("configs/augment")


def _augment_one(
    config: Path,
    out_dir: Path,
    input_: Path | None,
    augmenter: str | None,
    seed: int | None,
) -> None:
    cfg = load_config(config)
    if augmenter is not None:
        cfg = cfg.model_copy(update={"augmenter": augmenter})
    if seed is not None:
        cfg = cfg.model_copy(update={"seed": seed})

    input_csv = input_ if input_ is not None else out_dir / cfg.dataset / "original.csv"

    click.echo(f"[augment] dataset={cfg.dataset}  augmenter={cfg.augmenter}")
    pipeline = AugmentPipeline(cfg, input_csv=input_csv, out_dir=out_dir, config_path=config)
    paraphrase_csv, idiomatic_csv = pipeline.run()

    out_paths = {"paraphrase": paraphrase_csv, "idiomatic": idiomatic_csv}
    # Report only variants that were actually attempted this run (a variant that
    # stops early stops the run before later variants start).
    for variant, out_path in out_paths.items():
        if variant not in pipeline.last_counts:
            continue
        counts = pipeline.last_counts[variant]
        cache_stats = pipeline.last_cache_stats.get(variant, {})
        click.echo(f"[augment] [{variant}] counts={counts}")
        click.echo(f"[augment] [{variant}] cache={cache_stats}")
        if variant in pipeline.incomplete_variants:
            click.echo(
                f"[augment] [{variant}] INCOMPLETE — {counts.get('written', 0)} row(s) "
                f"saved to {out_path}; re-run the same command to resume."
            )
        else:
            click.echo(f"[augment] [{variant}] wrote {out_path}")
            click.echo(f"[augment] [{variant}] wrote {out_path.parent / f'{variant}.meta.json'}")

    if pipeline.incomplete_variants:
        click.echo(
            "[augment] run incomplete — re-run the same command to resume from where it stopped."
        )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a configs/augment/*.yaml file. Required unless --all is set.",
)
@click.option(
    "--all",
    "all_",
    is_flag=True,
    help=(
        "Augment every registered dataset (see data/registry.py), using "
        f"{_CONFIG_DIR}/<dataset>.yaml for each. Future datasets are picked "
        "up automatically once registered and configured."
    ),
)
@click.option(
    "--input",
    "input_",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Path to a Stage 1 CSV, overriding the derived "
        "<out-dir>/<dataset>/original.csv. Only valid with --config."
    ),
)
@click.option(
    "--augmenter",
    type=str,
    default=None,
    help="Override the config's augmenter id. Only valid with --config.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("datasets_out"),
    show_default=True,
    help="Root output directory; written to <out-dir>/<dataset>/{paraphrase,idiomatic}.csv.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Override the config's seed (CLI takes precedence over YAML).",
)
def main(
    config: Path | None,
    all_: bool,
    input_: Path | None,
    augmenter: str | None,
    out_dir: Path,
    seed: int | None,
) -> None:
    """Stage 2: augment a cleaned dataset (or every registered dataset with --all)."""
    if all_ and config is not None:
        raise click.UsageError("pass either --config or --all, not both.")
    if not all_ and config is None:
        raise click.UsageError("--config is required unless --all is set.")
    if all_ and (input_ is not None or augmenter is not None):
        raise click.UsageError("--input / --augmenter are only valid with --config, not --all.")

    if not all_:
        assert config is not None  # narrowed above
        _augment_one(config, out_dir, input_, augmenter, seed)
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
        _augment_one(_CONFIG_DIR / f"{name}.yaml", out_dir, None, None, seed)


if __name__ == "__main__":
    main()
