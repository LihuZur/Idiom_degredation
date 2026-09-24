"""RQ2 idiom-extraction CLI — runs `analysis.idiom_annotation.run_extraction`.

Usage:
    uv run idiom-annotate-idioms configs/annotate/sst2.yaml
    uv run idiom-annotate-idioms configs/annotate/sst2.yaml --max-rows 10
    uv run idiom-annotate-idioms configs/annotate/sst2.yaml --no-finalize
"""

from pathlib import Path

import click

from analysis.idiom_annotation import (
    annotate_and_link,
    finalize_partial_csv,
    load_config,
    run_extraction,
)
from augmentation.providers.base import LLMError


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--max-rows",
    type=int,
    default=None,
    help=(
        "Cap on NEW extractor API calls this invocation. Cached completions do not "
        "count. Overrides `limits.max_rows` in the YAML config. Use small values "
        "(e.g. 5) for a dry-run spot check."
    ),
)
@click.option(
    "--finalize/--no-finalize",
    default=True,
    help=(
        "After a run finishes cleanly, rewrite the partial CSV as the canonical "
        "`idiom_extractions.csv` and join it to the opacity/frequency norm tables "
        "into `idiom_norms.csv`. Skipped when the `--max-rows` cap stops the run early."
    ),
)
def main(config_path: Path, max_rows: int | None, finalize: bool) -> None:
    """Run the RQ2 idiom-extraction pass for one dataset."""
    cfg = load_config(config_path)
    if max_rows is not None:
        cfg = cfg.model_copy(
            update={"limits": cfg.limits.model_copy(update={"max_rows": max_rows})}
        )

    try:
        stats = run_extraction(cfg)
    except LLMError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"\n[{cfg.dataset}] extractions: {stats.cache_hits} cache-hits, "
        f"{stats.api_calls} api-calls, {stats.skipped_rows} skipped."
    )
    click.echo(
        f"[{cfg.dataset}] rough token totals: in≈{stats.tokens_in_est:,}, "
        f"out≈{stats.tokens_out_est:,}, USD≈{stats.dollars_est:.2f}."
    )

    hit_cap = (
        cfg.limits.max_rows is not None
        and (stats.api_calls + stats.skipped_rows) >= cfg.limits.max_rows
    )
    dataset_dir = Path(cfg.output_dir) / cfg.dataset
    if finalize and not hit_cap:
        final = finalize_partial_csv(dataset_dir)
        click.echo(f"[{cfg.dataset}] wrote {final}")
        norms_path = annotate_and_link(dataset_dir)
        click.echo(f"[{cfg.dataset}] wrote {norms_path}")
    else:
        click.echo(
            f"[{cfg.dataset}] partial CSV kept at "
            f"{dataset_dir / 'idiom_extractions.partial.csv'}"
            "  (rerun same command to continue)"
        )


if __name__ == "__main__":
    main()
