"""Label Studio helpers: build gold / blind-relabel tasks from generated data and import annotations."""

import csv
import json
from pathlib import Path

import click

from augmentation.labelstudio import (
    V3_FIELDS,
    agreement,
    blind_accuracy,
    blind_config_xml,
    blind_tasks,
    flatten_export,
    flatten_v3_export,
    gold_config_xml,
    gold_tasks,
    load_generated,
    load_v3,
    stratified_gold_sample,
    v3_agreement,
    v3_config_xml,
    v3_tasks,
)


def _dir(pools: Path, dataset: str) -> Path:
    return pools / f"{dataset}_v2"


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Human validation tooling: v3 tasks / import (in use); v2 gold and blind-relabel commands (history)."""


@main.command("gold-tasks")
@click.option("--dataset", type=click.Choice(["sst2", "mnli", "mmlu", "agnews"]), required=True)
@click.option("--n", type=int, default=300, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--pools",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("datasets_out"),
)
def gold_cmd(dataset: str, n: int, seed: int, pools: Path) -> None:
    """Stratified gold sample (label x length x idiom opacity) as Label Studio tasks."""
    out = _dir(pools, dataset) / "labelstudio"
    out.mkdir(parents=True, exist_ok=True)
    sample = stratified_gold_sample(load_generated(_dir(pools, dataset)), n, seed)
    (out / "tasks_gold.json").write_text(
        json.dumps(gold_tasks(sample), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out / "config_gold.xml").write_text(gold_config_xml(), encoding="utf-8")
    click.echo(f"{len(sample)} gold items -> {out}")


@main.command("blind-tasks")
@click.option(
    "--dataset",
    type=click.Choice(["sst2", "mnli", "agnews"]),
    required=True,
    help="MMLU is skipped: answering needs knowledge.",
)
@click.option("--n", type=int, default=150, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--pools",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("datasets_out"),
)
def blind_cmd(dataset: str, n: int, seed: int, pools: Path) -> None:
    """Blind relabel tasks: each item twice (idiomatic, span-literal), no original or label shown."""
    out = _dir(pools, dataset) / "labelstudio"
    out.mkdir(parents=True, exist_ok=True)
    sample = stratified_gold_sample(load_generated(_dir(pools, dataset)), n, seed)
    tasks = blind_tasks(dataset, sample, seed)
    (out / "tasks_blind.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out / "config_blind.xml").write_text(blind_config_xml(dataset), encoding="utf-8")
    click.echo(f"{len(tasks)} blind tasks ({len(sample)} items x 2 variants) -> {out}")


@main.command("import")
@click.option(
    "--export",
    "export_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--kind", type=click.Choice(["gold", "blind"]), required=True)
@click.option("--dataset", type=click.Choice(["sst2", "mnli", "mmlu", "agnews"]), required=True)
@click.option(
    "--pools",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("datasets_out"),
)
def import_cmd(export_path: Path, kind: str, dataset: str, pools: Path) -> None:
    """Flatten a Label Studio JSON export; report kappa (gold) or human accuracy per variant (blind)."""
    rows = flatten_export(json.loads(export_path.read_text(encoding="utf-8")))
    out = _dir(pools, dataset)
    path = out / ("human_gold_long.csv" if kind == "gold" else "human_blind_long.csv")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["item_id", "variant", "gold", "annotator", "criterion", "value"]
        )
        w.writeheader()
        w.writerows(rows)
    click.echo(f"{len(rows)} judgments -> {path}")
    for rec in agreement(rows) if kind == "gold" else blind_accuracy(rows, dataset):
        click.echo(
            "  "
            + ", ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()
            )
        )


_V3_DATASETS = ("sst2", "mnli", "mmlu", "agnews")


@main.command("v3-tasks")
@click.option(
    "--run",
    "runs",
    multiple=True,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Generation output folder(s) holding <dataset>/ subfolders, e.g. datasets_out/_exp_v3_e3 (repeatable).",
)
@click.option(
    "--annotator", required=True, help="Short name; also seeds this annotator's task order."
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("datasets_out/_labelstudio_v3"),
    show_default=True,
)
def v3_tasks_cmd(runs: tuple[Path, ...], annotator: str, out_dir: Path) -> None:
    """Build one Label Studio project's files (tasks.json + config.xml) for one annotator. No model calls."""
    items = []
    for run in runs:
        for d in _V3_DATASETS:
            if (run / d / "idiomatic.csv").exists():
                items += load_v3(run / d, d)
    seed = int.from_bytes(annotator.encode("utf-8"), "big") % (2**31)
    target = out_dir / annotator
    target.mkdir(parents=True, exist_ok=True)
    (target / "tasks.json").write_text(
        json.dumps(v3_tasks(items, seed), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (target / "config.xml").write_text(v3_config_xml(), encoding="utf-8")
    with (target / "items.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "dataset", "id", "source_dir"])
        w.writeheader()
        w.writerows({k: i[k] for k in ("key", "dataset", "id", "source_dir")} for i in items)
    click.echo(
        f"{len(items)} items for {annotator} -> {target} (tasks.json, config.xml, items.csv)"
    )


@main.command("v3-import")
@click.option(
    "--export",
    "exports",
    multiple=True,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Label Studio JSON export(s), one per annotator project (repeatable).",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path("datasets_out/_labelstudio_v3/human_labels.csv"),
    show_default=True,
)
def v3_import_cmd(exports: tuple[Path, ...], out: Path) -> None:
    """Flatten export(s) to one row per item x annotator; print agreement if there are two annotators."""
    rows = []
    for path in exports:
        rows += flatten_v3_export(json.loads(path.read_text(encoding="utf-8")))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "dataset", "id", "annotator", *V3_FIELDS])
        w.writeheader()
        w.writerows(rows)
    click.echo(f"{len(rows)} annotations -> {out}")
    for rec in v3_agreement(rows):
        click.echo(
            "  "
            + ", ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()
            )
        )


if __name__ == "__main__":
    main()
