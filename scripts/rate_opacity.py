"""RQ2 opacity-rating CLI — runs `analysis.opacity_rating.run_rating`.

Fills the human opacity-norms coverage gap by asking an LLM to rate every
extracted idiom that isn't already covered by an exact or fuzzy match against
`data/norms/opacity.py`'s human norm tables. Ratings are cached per idiom type
and shared across all datasets (rating the same idiom twice never costs
twice), and persisted to `data/norms/_files/opacity/llm_ratings.csv` so
`analysis.idiom_annotation.link_to_norms` can use them as a fallback tier.

Usage:
    uv run idiom-rate-opacity configs/annotate/opacity_rating.yaml sst2
    uv run idiom-rate-opacity configs/annotate/opacity_rating.yaml sst2 mmlu mnli
"""

from pathlib import Path

import click

from analysis.opacity_rating import load_config, run_rating
from augmentation.providers.base import LLMError


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("datasets", nargs=-1, required=True)
def main(config_path: Path, datasets: tuple[str, ...]) -> None:
    """Rate opacity for every not-yet-covered idiom extracted across DATASETS."""
    cfg = load_config(config_path)
    try:
        stats = run_rating(cfg, list(datasets))
    except LLMError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"opacity ratings: {stats.cache_hits} cache-hits, {stats.api_calls} api-calls, "
        f"{stats.skipped} skipped, out of {stats.total_idioms} idioms needing a rating."
    )


if __name__ == "__main__":
    main()
