"""MMLU-replacement baseline (plan v3 O47, 2026-11-18): `python -m scripts.mc_baseline prepare` (on Colab: downloads
Social IQa and CosmosQA from Hugging Face) and `python -m scripts.mc_baseline report` (accuracy per model).

`prepare` writes datasets_out/{siqa,cosmosqa}_{base,noctx}/original.csv: the same 200 items per source (balanced over
the gold answer position), with and without the context.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import click

from data.mc_baseline import SOURCES, normalize, sample_balanced, write_dataset


@click.group()
def main() -> None:
    """MMLU-replacement baseline check."""


def _load_hf(hf_id: str, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset  # noqa: PLC0415

    try:
        ds = load_dataset(hf_id, split=split)
    except (
        Exception
    ) as exc:  # script-based datasets: fall back to the auto-converted parquet branch
        click.echo(f"load_dataset({hf_id}) failed ({exc}); trying the parquet branch")
        ds = load_dataset(hf_id, split=split, revision="refs/convert/parquet")
    return [dict(r) for r in ds]


@main.command()
@click.option("--n", type=int, default=200, show_default=True)
@click.option("--seed", type=int, default=7, show_default=True)
@click.option("--out-root", type=click.Path(path_type=Path), default=Path("datasets_out"))
def prepare(n: int, seed: int, out_root: Path) -> None:
    """Download, normalise, sample n balanced items per source, write the with/without-context datasets."""
    for source, spec in SOURCES.items():
        rows = normalize(_load_hf(spec["hf"], spec["split"]), source)
        sample = sample_balanced(rows, n, seed)
        for suffix, ctx in (("base", True), ("noctx", False)):
            write_dataset(sample, out_root / f"{source}_{suffix}", with_context=ctx)
        words = sorted(len(r["context"].split()) for r in sample)
        click.echo(
            f"{source}: {len(rows)} rows in {spec['hf']}:{spec['split']} -> {len(sample)} sampled; "
            f"{len(sample[0]['choices'])} options; labels {dict(sorted(Counter(r['label'] for r in sample).items()))}; "
            f"context words median {words[len(words) // 2]}, max {words[-1]}"
        )
        ex = sample[0]
        click.echo(
            f"  example: {ex['context'][:200]!r}\n  Q: {ex['question']}  options: {ex['choices']}  gold: {ex['label']}"
        )


@main.command()
@click.option("--results-dir", type=click.Path(path_type=Path), default=Path("results"))
def report(results_dir: Path) -> None:
    """Accuracy and readable-answer rate per model, with and without context, against chance."""
    names = [f"{s}_{x}" for s in SOURCES for x in ("base", "noctx")]
    table: dict[str, dict[str, str]] = {}
    for ds in names:
        for f in sorted((results_dir / ds).glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            tasks = r["per_task"]
            acc = sum(bool(t["original"]["correct"]) for t in tasks) / len(tasks) * 100
            ok = sum(t["original"].get("parse_status") == "ok" for t in tasks) / len(tasks) * 100
            table.setdefault(str(r["model_id"]), {})[ds] = f"{acc:5.1f} ({ok:3.0f}%)"
    if not table:
        raise click.ClickException(f"no results under {results_dir} for {names}")
    click.echo("accuracy % (readable answers %); chance: siqa 33.3, cosmosqa 25.0")
    click.echo(f"{'model':28} " + " ".join(f"{n:>16}" for n in names))
    for model in sorted(table):
        click.echo(f"{model:28} " + " ".join(f"{table[model].get(n, '-'):>16}" for n in names))


@main.command()
@click.option("--source", type=click.Choice(sorted(SOURCES)), default="siqa", show_default=True)
@click.option("--split", default="train", show_default=True)
@click.option("--out-root", type=click.Path(path_type=Path), default=Path("datasets_out/wide"))
def pool(source: str, split: str, out_root: Path) -> None:
    """Write a whole split as a generation pool (`<out-root>/<source>/original.csv`, context = the field to edit)."""
    rows = normalize(_load_hf(SOURCES[source]["hf"], split), source)
    path = write_dataset(rows, out_root / source, with_context=True)
    words = sorted(len(r["context"].split()) for r in rows)
    click.echo(
        f"{source}:{split}: {len(rows)} items -> {path}; labels {dict(sorted(Counter(r['label'] for r in rows).items()))}; "
        f"context words median {words[len(words) // 2]}"
    )


if __name__ == "__main__":
    main()
