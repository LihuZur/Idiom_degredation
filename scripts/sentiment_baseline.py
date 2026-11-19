"""SST-2-replacement baseline (plan v3 O40, 2026-11-19), run on Colab:

  python -m scripts.sentiment_baseline inspect   # configs, splits, columns, one example per dataset (no files)
  python -m scripts.sentiment_baseline prepare   # 200 balanced positive/negative items per candidate
  python -m scripts.sentiment_baseline report    # accuracy, p_gold, model disagreement per item

Datasets written: dynasent_r1_base, dynasent_r2_base, besstie_google_base / besstie_reddit_base (or besstie_base when
the source column is missing). Evaluated with the SST-2 prompt and parser.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import click

from data.sentiment_baseline import normalize, sample_balanced, write_dataset

DYNASENT = "dynabench/dynasent"
BESSTIE = "unswnlporg/BESSTIE"
_SPLIT_LO, _SPLIT_HI = 0.3, 0.7
NAMES = (
    "dynasent_r1_base",
    "dynasent_r2_base",
    "besstie_google_base",
    "besstie_reddit_base",
    "besstie_base",
)


def _configs(hf_id: str) -> list[str]:
    from datasets import get_dataset_config_names  # noqa: PLC0415

    for revision in (
        None,
        "refs/convert/parquet",
    ):  # script-based datasets list configs only on the parquet branch
        try:
            names = list(get_dataset_config_names(hf_id, revision=revision))
            if names:
                return names
        except Exception as exc:
            click.echo(f"  config names for {hf_id} (revision {revision}) not available ({exc})")
    return []


def _load(hf_id: str, config: str | None, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset  # noqa: PLC0415

    try:
        ds = load_dataset(hf_id, config, split=split)
    except Exception as exc:  # script-based datasets: the auto-converted parquet branch
        click.echo(
            f"  load_dataset({hf_id}, {config}, {split}) failed ({exc}); trying the parquet branch"
        )
        ds = load_dataset(hf_id, config, split=split, revision="refs/convert/parquet")
    return [dict(r) for r in ds]


def _split_names(hf_id: str, config: str | None) -> list[str]:
    from datasets import get_dataset_split_names  # noqa: PLC0415

    for revision in (None, "refs/convert/parquet"):
        try:
            names = list(get_dataset_split_names(hf_id, config, revision=revision))
            if names:
                return names
        except Exception:
            continue
    return ["validation", "train"]


def _round_config(configs: list[str], tag: str) -> str:
    hits = [c for c in configs if tag in c.lower()]
    best = [c for c in hits if "all" in c.lower()] or hits
    if not best:
        raise click.ClickException(f"no DynaSent config containing {tag!r} among {configs}")
    return best[0]


@click.group()
def main() -> None:
    """SST-2-replacement baseline check."""


@main.command()
def inspect() -> None:
    """Print configs, splits, columns and one example of each candidate (writes nothing)."""
    for hf_id in (DYNASENT, BESSTIE):
        configs = _configs(hf_id)
        click.echo(f"== {hf_id}: configs {configs}")
        for config in configs[:6] or [None]:
            splits = _split_names(hf_id, config)
            rows = _load(hf_id, config, splits[0])
            click.echo(
                f"  [{config}] splits {splits}; {splits[0]}: {len(rows)} rows; columns {list(rows[0])}"
            )
            click.echo(f"    example: {json.dumps(rows[0], default=str)[:300]}")


def _prefer_split(hf_id: str, config: str | None) -> str:
    splits = _split_names(hf_id, config)
    for s in ("validation", "dev", "test", "train"):
        if s in splits:
            return s
    return splits[0]


@main.command()
@click.option("--n", type=int, default=200, show_default=True)
@click.option("--seed", type=int, default=7, show_default=True)
@click.option("--out-root", type=click.Path(path_type=Path), default=Path("datasets_out"))
def prepare(n: int, seed: int, out_root: Path) -> None:
    """Write n balanced positive/negative items per candidate; a failing candidate is reported and skipped."""
    failures: list[str] = []
    try:
        configs = _configs(DYNASENT)
        for tag, name in (("r1", "dynasent_r1_base"), ("r2", "dynasent_r2_base")):
            try:
                config = _round_config(configs, tag)
                split = _prefer_split(DYNASENT, config)
                rows = normalize(_load(DYNASENT, config, split), name)
                _write(rows, n, seed, out_root / name, f"{DYNASENT}:{config}:{split}")
            except Exception as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
    except Exception as exc:
        failures.append(f"dynasent: {type(exc).__name__}: {exc}")
    try:
        bconfigs = _configs(BESSTIE)
        records: list[dict[str, Any]] = []
        for config in bconfigs or [None]:
            records += _load(BESSTIE, config, _prefer_split(BESSTIE, config))
        rows = normalize(records, "besstie")
        sources = Counter(r["source"].lower() for r in rows)
        click.echo(f"besstie source values: {dict(sources)}")
        groups = {
            "besstie_google_base": [r for r in rows if "google" in r["source"].lower()],
            "besstie_reddit_base": [r for r in rows if "reddit" in r["source"].lower()],
        }
        if not any(groups.values()):
            groups = {"besstie_base": rows}
        for name, part in groups.items():
            if part:
                _write(part, n, seed, out_root / name, f"{BESSTIE} ({name})")
    except Exception as exc:
        failures.append(f"besstie: {type(exc).__name__}: {exc}")
    for f in failures:
        click.echo(f"FAILED {f}")
    if len(failures) >= 3:  # noqa: PLR2004 - nothing could be prepared
        raise click.ClickException("no candidate could be prepared; see the FAILED lines above")


def _write(rows: list[dict[str, Any]], n: int, seed: int, out_dir: Path, where: str) -> None:
    sample = sample_balanced(rows, n, seed)
    write_dataset(sample, out_dir)
    labels = Counter(r["label"] for r in rows)
    words = sorted(len(r["text"].split()) for r in sample)
    click.echo(
        f"{out_dir.name}: {where}: {len(rows)} positive/negative rows (labels {dict(sorted(labels.items()))}) -> "
        f"{len(sample)} sampled; words median {words[len(words) // 2]}\n  example: {sample[0]['text'][:160]!r} -> {sample[0]['label']}"
    )


@main.command()
@click.option("--results-dir", type=click.Path(path_type=Path), default=Path("results_prob"))
def report(results_dir: Path) -> None:
    """Accuracy, mean p_gold and model disagreement per candidate (with SST-2 POC-3 originals as reference)."""
    names = [*NAMES, "sst2_v3poc3"]
    click.echo("accuracy % / mean p_gold % per model; chance 50")
    table: dict[str, dict[str, str]] = {}
    split_items: dict[str, list[float]] = {}
    for ds in names:
        files = sorted((results_dir / ds).glob("*.json"))
        if not files:
            continue
        per_item: dict[str, list[int]] = {}
        for f in files:
            r = json.loads(f.read_text(encoding="utf-8"))
            tasks = r["per_task"]
            acc = sum(bool(t["original"]["correct"]) for t in tasks) / len(tasks) * 100
            pg = [
                t["original"].get("p_gold")
                for t in tasks
                if t["original"].get("p_gold") is not None
            ]
            table.setdefault(str(r["model_id"]), {})[ds] = (
                f"{acc:5.1f} / {sum(pg) / len(pg) * 100:5.1f}" if pg else f"{acc:5.1f}"
            )
            for t in tasks:
                per_item.setdefault(str(t["id"]), []).append(int(bool(t["original"]["correct"])))
        split_items[ds] = [sum(v) / len(v) for v in per_item.values()]
    click.echo(f"{'model':26} " + " ".join(f"{n:>20}" for n in split_items))
    for model in sorted(table):
        click.echo(
            f"{model:26} " + " ".join(f"{table[model].get(n, '-'):>20}" for n in split_items)
        )
    click.echo(
        "\nitems where models split (between 30% and 70% of models correct): a rough label-noise / difficulty signal"
    )
    for ds, fr in split_items.items():
        mid = sum(_SPLIT_LO <= x <= _SPLIT_HI for x in fr)
        allwrong = sum(x < _SPLIT_LO for x in fr)
        click.echo(
            f"  {ds:22} items {len(fr)}; split {mid} ({mid / len(fr):.0%}); mostly wrong {allwrong} ({allwrong / len(fr):.0%})"
        )


if __name__ == "__main__":
    main()
