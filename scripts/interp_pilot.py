"""Interpretability pilot, analysis 1 (plan v3 F1, 2026-11-20); run on Colab (GPU), see `analysis/interp.py`.

  python -m scripts.interp_pilot run --config configs/eval/qwen3p5_7b_instruct.yaml --tasks sst2,mnli,agnews,siqa
  python -m scripts.interp_pilot summary

`run` writes one CSV per model and task to --out-dir (one row per item and layer) and skips pairs already written,
so a stopped Colab session resumes. Correctness and p_gold come from the evaluation results (--results-dir), so the
states describe the same prompts the panel answered.
"""

import json
import warnings
from pathlib import Path
from typing import Any, cast

import click
import numpy as np
import pandas as pd
import yaml

import eval as eval_package  # pyright: ignore[reportUnusedImport]  # noqa: F401  (registers evaluators)
import models as models_package  # pyright: ignore[reportUnusedImport]  # noqa: F401
from analysis.interp import (
    depth_bins,
    group_of,
    item_rows,
    layer_means,
    pooled_flip_test,
    summarize,
)
from eval.config import EvalConfig
from eval.registry import get_evaluator
from models.decoder_runner import DecoderRunner
from models.registry import get_model_class, get_model_spec

VARIANTS = ("original", "idiomatic", "span_literal")


def _results(results_dir: Path, dataset: str, model: str) -> dict[str, dict[str, Any]]:
    path = results_dir / dataset / f"{model.replace('/', '_')}.json"
    if not path.exists():
        raise click.ClickException(f"no evaluation result {path}; the pilot joins on it")
    return {str(t["id"]): t for t in json.loads(path.read_text(encoding="utf-8"))["per_task"]}


def _span(meta: dict[str, Any], key: str) -> tuple[int, int] | None:
    s = (meta.get("idiom") or {}).get(key)
    return (int(s[0]), int(s[1])) if s else None


@click.group()
def main() -> None:
    """Interpretability pilot (analysis 1: layer-wise distance between idiomatic and literal states)."""


@main.command()
@click.option(
    "--config",
    "configs",
    multiple=True,
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--tasks", default="sst2,mnli,agnews,siqa", show_default=True)
@click.option("--suffix", default="_v3poc3", show_default=True)
@click.option("--results-dir", type=click.Path(path_type=Path), default=Path("results_prob"))
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("results_interp"))
@click.option("--limit", type=int, default=None, help="First N items per task (smoke test).")
def run(
    configs: tuple[Path, ...],
    tasks: str,
    suffix: str,
    results_dir: Path,
    out_dir: Path,
    limit: int | None,
) -> None:
    """Hidden states for every POC pair; distances per layer; one CSV per model and task."""
    for config in configs:
        cfg = EvalConfig.model_validate(yaml.safe_load(config.read_text()))
        todo = [
            t
            for t in tasks.split(",")
            if not (out_dir / f"{t}{suffix}" / f"{cfg.model.replace('/', '_')}.csv").exists()
        ]
        if not todo:
            click.echo(f"{cfg.model}: all tasks done, skipped")
            continue
        runner = cast(type[DecoderRunner], get_model_class(cfg.model))(
            get_model_spec(cfg.model), precision=cfg.precision
        )
        for task in todo:
            _run_task(runner, cfg, f"{task}{suffix}", results_dir, out_dir, limit)
        del runner
        try:
            import torch  # noqa: PLC0415

            torch.cuda.empty_cache()
        except Exception:
            pass


def _run_task(
    runner: DecoderRunner,
    cfg: EvalConfig,
    dataset: str,
    results_dir: Path,
    out_dir: Path,
    limit: int | None,
) -> None:
    evaluator = get_evaluator(dataset)(cfg)
    rows = {
        v: {r.id: r for r in evaluator.load_rows(Path("datasets_out") / dataset / f"{v}.csv")}
        for v in VARIANTS
    }
    ids = [i for i in rows["idiomatic"] if all(i in rows[v] for v in VARIANTS)][:limit]
    res = _results(results_dir, dataset, cfg.model)
    items: list[
        tuple[dict[str, object], dict[str, np.ndarray], np.ndarray | None, np.ndarray | None]
    ] = []
    no_span = 0
    for n, i in enumerate(ids):
        t = res.get(i)
        if t is None:
            continue
        last: dict[str, np.ndarray] = {}
        spans: dict[str, np.ndarray | None] = {}
        for v in VARIANTS:
            ex = rows[v][i]
            key = {"idiomatic": "idiomatic_span", "span_literal": "literal_span"}.get(v)
            span = _span(ex.meta, key) if key else None
            h_last, h_span = runner.hidden_states(evaluator.format(ex), ex.x, span)
            last[v] = h_last.numpy()
            spans[v] = h_span.numpy() if h_span is not None else None
        no_span += spans["idiomatic"] is None or spans["span_literal"] is None
        lit_c, idi_c = int(bool(t["span_literal"]["correct"])), int(bool(t["idiomatic"]["correct"]))
        meta: dict[str, object] = {
            "model": cfg.model,
            "task": dataset,
            "id": i,
            "idiom": (rows["idiomatic"][i].meta.get("idiom") or {}).get("idiom", ""),
            "lit_correct": lit_c,
            "idi_correct": idi_c,
            "group": group_of(lit_c, idi_c),
            "pg_lit": t["span_literal"].get("p_gold"),
            "pg_idi": t["idiomatic"].get("p_gold"),
        }
        items.append((meta, last, spans["idiomatic"], spans["span_literal"]))
        if (n + 1) % 20 == 0:
            click.echo(f"  {cfg.model} {dataset}: {n + 1}/{len(ids)}")
    if not items:
        raise click.ClickException(f"{dataset}: no item has an evaluation result for {cfg.model}")
    # per-layer means over the whole task (all items, all variants) for the mean-centred distances
    center_last = layer_means([v for _, last, _, _ in items for v in last.values()])
    span_vecs = [s for _, _, si, sl in items for s in (si, sl) if s is not None]
    center_span = layer_means(span_vecs) if span_vecs else None
    out: list[dict[str, object]] = []
    for meta, last, si, sl in items:
        out += item_rows(meta, last, si, sl, center_last, center_span)
    path = out_dir / dataset / f"{cfg.model.replace('/', '_')}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(path, index=False)
    click.echo(f"{path}: {len(ids)} items; span not located in {no_span}")


@main.command()
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("results_interp"))
def summary(out_dir: Path) -> None:
    """Per-layer summary and relative-depth quarters; writes summary.csv and depth.csv to --out-dir."""
    files = sorted(out_dir.glob("*/*.csv"))
    if not files:
        raise click.ClickException(f"no pilot CSVs under {out_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    warnings.filterwarnings(
        "ignore", message="An input array is constant"
    )  # layer 0: identical last token
    s = summarize(df)
    s.to_csv(out_dir / "summary.csv", index=False)
    d = depth_bins(s)
    d.to_csv(out_dir / "depth.csv", index=False)
    with pd.option_context(
        "display.width", 200, "display.max_rows", 500, "display.float_format", "{:.4f}".format
    ):
        click.echo(d.to_string(index=False))
    pooled = pooled_flip_test(df)
    pooled.to_csv(out_dir / "pooled_flip_test.csv", index=False)
    with pd.option_context(
        "display.width", 200, "display.max_rows", 200, "display.float_format", "{:.3f}".format
    ):
        click.echo(
            "\npooled over models: relative d_il (1 = typical item), flipped items vs items right in both"
        )
        click.echo(pooled.to_string(index=False))
    groups = df[df["layer"] == 0].groupby(["model", "task", "group"]).size().unstack(fill_value=0)
    click.echo(f"\nitems per group (idiom_only_wrong = literal right, idiomatic wrong):\n{groups}")


if __name__ == "__main__":
    main()
