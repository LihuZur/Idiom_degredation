"""RQ1 meaning-preservation CLI — runs `analysis.meaning_preservation.run_validation`.

Usage:
    uv run idiom-validate configs/validate/mmlu.yaml
    uv run idiom-validate configs/validate/mmlu.yaml --max-pairs 10
    uv run idiom-validate configs/validate/mmlu.yaml --finalize
"""

from pathlib import Path

import click

from analysis.meaning_preservation import (
    finalize_partial_csv,
    load_config,
    run_validation,
)
from augmentation.providers.base import LLMError


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--max-pairs",
    type=int,
    default=None,
    help=(
        "Cap on NEW (id, arm, judge) API calls this invocation. Cached completions "
        "do not count. Overrides `limits.max_pairs` in the YAML config. Use small "
        "values (e.g. 20) for a dry-run spot check."
    ),
)
@click.option(
    "--finalize/--no-finalize",
    default=True,
    help=(
        "After a run finishes cleanly, rewrite the partial CSV as the canonical "
        "`meaning_validated.csv` (sorted by id, arm, judge). Skipped when the "
        "`--max-pairs` cap stops the run early."
    ),
)
def main(config_path: Path, max_pairs: int | None, finalize: bool) -> None:
    """Run the RQ1 meaning-preservation judge pass for one dataset."""
    cfg = load_config(config_path)
    if max_pairs is not None:
        cfg = cfg.model_copy(
            update={"limits": cfg.limits.model_copy(update={"max_pairs": max_pairs})}
        )

    try:
        stats = run_validation(cfg)
    except LLMError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"\n[{cfg.dataset}] verdicts: {stats.cache_hits} cache-hits, "
        f"{stats.api_calls} api-calls, {stats.skipped_rows} skipped."
    )
    click.echo(
        f"[{cfg.dataset}] rough token totals: in≈{stats.tokens_in_est:,}, "
        f"out≈{stats.tokens_out_est:,}, USD≈{stats.dollars_est:.2f}."
    )
    for label, per in stats.per_judge.items():
        click.echo(
            f"[{cfg.dataset}]   {label}: PASS={per['preserved_true']}, "
            f"FAIL={per['preserved_false']}, cache-hits={per['cache_hits']}, "
            f"api-calls={per['api_calls']}, skipped={per['skipped']}."
        )

    hit_cap = (
        cfg.limits.max_pairs is not None
        and (stats.api_calls + stats.skipped_rows) >= cfg.limits.max_pairs
    )
    if finalize and not hit_cap:
        dataset_dir = Path(cfg.output_dir) / cfg.dataset
        final = finalize_partial_csv(dataset_dir)
        click.echo(f"[{cfg.dataset}] wrote {final}")
    else:
        click.echo(
            f"[{cfg.dataset}] partial CSV kept at "
            f"{Path(cfg.output_dir) / cfg.dataset / 'meaning_validated.partial.csv'}"
            "  (rerun same command to continue)"
        )


if __name__ == "__main__":
    main()
