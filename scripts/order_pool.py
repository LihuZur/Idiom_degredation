"""Write a pool in run order: the already-processed prefix unchanged, the rest ranked by keep chance.

Plan: `local/HANDOFF.md` §4b. The ranking comes from `augmentation.keep_predictor` (logistic regression on
frozen BGE sentence embeddings, fitted on earlier generation outcomes); no API calls. Generating over the
ordered pool with the same settings replays the prefix's waves from the response cache, so a run can be
extended (Small → Medium → Max) without paying twice.
"""

import csv
import json
from datetime import date
from pathlib import Path

import click
import numpy as np

from augmentation.idiom_generation import filter_items, load_pool
from augmentation.idiom_retrieval import bge_embedder
from augmentation.keep_predictor import order_pool

# Not processed (skipped before call 1) or no usable reply: no information about the item.
_UNLABELLED = frozenset({"original_has_idiom", "empty_response", "bad_json"})


def read_labels(logs: tuple[Path, ...]) -> dict[str, bool]:
    """kept / not kept per item id from generation_log.csv files (kept in any log wins)."""
    labels: dict[str, bool] = {}
    for path in logs:
        with path.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["reason"] in _UNLABELLED:
                    continue
                labels[r["id"]] = labels.get(r["id"], False) or r["ok"] == "True"
    return labels


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--dataset", type=click.Choice(["sst2", "mnli", "mmlu", "agnews"]), required=True)
@click.option("--pool", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--where", multiple=True, help="Same filter as idiom-generate --where.")
@click.option(
    "--prefix",
    type=int,
    required=True,
    help="Items (after --where, in pool order) already processed; they stay first, unchanged.",
)
@click.option(
    "--log",
    "logs",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    required=True,
    help="generation_log.csv file(s) with outcomes to fit the predictor on.",
)
@click.option("--out", type=click.Path(path_type=Path), default=None)
def main(
    dataset: str,
    pool: Path,
    where: tuple[str, ...],
    prefix: int,
    logs: tuple[Path, ...],
    out: Path | None,
) -> None:
    """Rank the unprocessed part of a pool by predicted keep chance (free, local)."""
    items_all, rows = load_pool(pool, dataset)
    try:
        items = filter_items(items_all, where)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    if prefix > len(items):
        raise click.UsageError(f"--prefix {prefix} exceeds the {len(items)} items of the pool")
    labels = read_labels(logs)
    embed = bge_embedder()
    ids = [i.id for i in items]
    order, quality = order_pool(
        ids,
        [i.x for i in items],
        labels,
        prefix=prefix,
        embed=lambda texts: np.asarray(embed(texts, False), dtype=np.float64),
    )
    out = out or Path("datasets_out/ordered") / dataset / "original.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[ids[0]]))
        w.writeheader()
        w.writerows(rows[i] for i in order)
    known = [i for i in ids if i in labels]
    meta = {
        "dataset": dataset,
        "source_pool": str(pool),
        "where": list(where),
        "prefix_unchanged": prefix,
        "n_items": len(order),
        "n_labelled": len(known),
        "n_kept_labels": sum(labels[i] for i in known),
        "cross_validated_auc": round(quality, 3),
        "logs": [str(p) for p in logs],
        "date": date.today().isoformat(),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    click.echo(
        f"wrote {out}: {len(order)} items, first {prefix} unchanged, rest ranked; predictor "
        f"fitted on {len(known)} labelled items ({meta['n_kept_labels']} kept), "
        f"cross-validated AUC {quality:.2f}"
    )


if __name__ == "__main__":
    main()
