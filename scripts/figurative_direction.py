"""Interpretability test A (plan v3 O42, 2026-11-20): figurative direction per model, validity on held-out idioms,
and the figurative reading of our POC idioms vs the model's errors. See `analysis/figurative.py`.

  python -m scripts.figurative_direction prepare      # local: MAGPIE -> datasets_out/figurative_magpie/sentences.jsonl
  python -m scripts.figurative_direction run --config configs/eval/qwen3p5_7b_instruct.yaml   # Colab GPU
  python -m scripts.figurative_direction summary      # local, from the downloaded CSVs

`run` writes `<model>_validity.csv` (held-out AUC per context and layer) and `<model>_scores.csv` (POC idiom scores
per context and layer, joined with correctness and p_gold from --results-dir) to --out-dir; a model with both files
is skipped, so a stopped Colab session resumes.
"""

import csv
import json
from pathlib import Path
from typing import Any, cast

import click
import numpy as np
import pandas as pd
import yaml

import eval as eval_package  # pyright: ignore[reportUnusedImport]  # noqa: F401  (registers evaluators)
import models as models_package  # pyright: ignore[reportUnusedImport]  # noqa: F401
from analysis.figurative import (
    at_best,
    best_layers,
    cv_auc_by_layer,
    depth_table,
    group_tests,
    idiom_folds,
    md_direction,
    project,
    rescale,
    scale_reference,
    score_vs_pgold,
)
from analysis.interp import group_of
from data.figurative import read_jsonl, select_magpie, write_jsonl
from eval.config import EvalConfig
from eval.registry import get_evaluator
from models.base import FormattedInput
from models.decoder_runner import DecoderRunner
from models.registry import get_model_class, get_model_spec

MAGPIE = Path("data/norms/_files/magpie/MAGPIE_unfiltered.jsonl")
SENTENCES = Path("datasets_out/figurative_magpie/sentences.jsonl")
INSTRUCTION = "Read the following text.\n\nText: "  # neutral wrapper for the `template` context
CONTEXTS = ("bare", "template")


@click.group()
def main() -> None:
    """Figurative direction (interpretability test A)."""


@main.command()
@click.option("--magpie", type=click.Path(exists=True, path_type=Path), default=MAGPIE)
@click.option("--out", type=click.Path(path_type=Path), default=SENTENCES)
@click.option("--suffix", default="_v3poc3", show_default=True)
def prepare(magpie: Path, out: Path, suffix: str) -> None:
    """Select balanced figurative / literal MAGPIE sentences, excluding the idioms of our POC pairs."""
    poc: set[str] = set()
    for f in Path("datasets_out").glob(f"*{suffix}/idiomatic.csv"):
        with f.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                poc.add(str(json.loads(r["meta"])["idiom"]["idiom"]).lower())
    rows = select_magpie(read_jsonl(magpie), poc)
    write_jsonl(rows, out)
    types = {r["idiom"] for r in rows}
    fig = sum(r["label"] for r in rows)
    click.echo(
        f"{out}: {len(rows)} sentences ({fig} figurative, {len(rows) - fig} literal), {len(types)} idioms; "
        f"{len(poc)} POC idioms excluded. Upload the folder to Drive: Idiom_degredation/datasets_out/"
    )


def _template_prompt(runner: DecoderRunner, text: str) -> tuple[str, int]:
    fi = FormattedInput(
        id="x", prompt="", meta={"messages": [{"role": "user", "content": INSTRUCTION + text}]}
    )
    prompt = runner.prompt_text(fi)
    return prompt, prompt.find(text)


def _collect(
    runner: DecoderRunner, sents: list[dict[str, Any]], context: str
) -> tuple[np.ndarray, np.ndarray]:
    acts: list[np.ndarray] = []
    keep: list[int] = []
    for n, s in enumerate(sents):
        lo, hi = s["span"]
        if context == "bare":
            prompt, start = s["text"], 0
        else:
            prompt, start = _template_prompt(runner, s["text"])
        if start < 0:
            continue
        h = runner.residual_at(prompt, start + lo, start + hi)
        if h is not None:
            acts.append(h.numpy().astype(np.float16))
            keep.append(n)
    return np.stack(acts), np.array(keep)


def _poc_rows(
    runner: DecoderRunner,
    cfg: EvalConfig,
    dataset: str,
    results_dir: Path,
    directions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    evaluator = get_evaluator(dataset)(cfg)
    rows = {
        v: {r.id: r for r in evaluator.load_rows(Path("datasets_out") / dataset / f"{v}.csv")}
        for v in ("idiomatic", "span_literal")
    }
    res_path = results_dir / dataset / f"{cfg.model.replace('/', '_')}.json"
    res = {str(t["id"]): t for t in json.loads(res_path.read_text(encoding="utf-8"))["per_task"]}
    out: list[dict[str, Any]] = []
    for i, ex_idi in rows["idiomatic"].items():
        ex_lit, t = rows["span_literal"].get(i), res.get(i)
        if ex_lit is None or t is None:
            continue
        acts: dict[str, np.ndarray] = {}
        for name, ex, key in (("idi", ex_idi, "idiomatic_span"), ("lit", ex_lit, "literal_span")):
            span = (ex.meta.get("idiom") or {}).get(key)
            prompt = runner.prompt_text(evaluator.format(ex))
            start = prompt.find(ex.x)
            h = (
                runner.residual_at(prompt, start + int(span[0]), start + int(span[1]))
                if span and start >= 0
                else None
            )
            if h is not None:
                acts[name] = h.numpy()[None]
        if len(acts) < 2:  # noqa: PLR2004 - both versions needed
            continue
        lit_c, idi_c = int(bool(t["span_literal"]["correct"])), int(bool(t["idiomatic"]["correct"]))
        base = {
            "model": cfg.model,
            "task": dataset,
            "id": i,
            "idiom": (ex_idi.meta.get("idiom") or {}).get("idiom", ""),
            "lit_correct": lit_c,
            "idi_correct": idi_c,
            "group": group_of(lit_c, idi_c),
            "pg_lit": t["span_literal"].get("p_gold"),
            "pg_idi": t["idiomatic"].get("p_gold"),
        }
        for context, (v, mid, half) in directions.items():
            s_idi = rescale(project(acts["idi"], v), mid, half)[0]
            s_lit = rescale(project(acts["lit"], v), mid, half)[0]
            n_layers = len(s_idi)
            out += [
                {
                    **base,
                    "context": context,
                    "layer": k,
                    "depth": k / max(n_layers - 1, 1),
                    "score_idi": float(s_idi[k]),
                    "score_lit": float(s_lit[k]),
                }
                for k in range(n_layers)
            ]
    return out


@main.command()
@click.option(
    "--config",
    "configs",
    multiple=True,
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--sentences", type=click.Path(exists=True, path_type=Path), default=SENTENCES)
@click.option("--tasks", default="sst2,mnli,agnews,siqa", show_default=True)
@click.option("--suffix", default="_v3poc3", show_default=True)
@click.option("--results-dir", type=click.Path(path_type=Path), default=Path("results_prob"))
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("results_figurative"))
def run(
    configs: tuple[Path, ...],
    sentences: Path,
    tasks: str,
    suffix: str,
    results_dir: Path,
    out_dir: Path,
) -> None:
    """Directions, held-out validity and POC scores for each model."""
    sents = read_jsonl(sentences)
    out_dir.mkdir(parents=True, exist_ok=True)
    for config in configs:
        cfg = EvalConfig.model_validate(yaml.safe_load(config.read_text()))
        stem = cfg.model.replace("/", "_")
        vpath, spath = out_dir / f"{stem}_validity.csv", out_dir / f"{stem}_scores.csv"
        if vpath.exists() and spath.exists():
            click.echo(f"{cfg.model}: done, skipped")
            continue
        runner = cast(type[DecoderRunner], get_model_class(cfg.model))(
            get_model_spec(cfg.model), precision=cfg.precision
        )
        validity: list[dict[str, Any]] = []
        directions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for context in CONTEXTS:
            acts, keep = _collect(runner, sents, context)
            labels = np.array([sents[n]["label"] for n in keep])
            folds = idiom_folds([sents[n]["idiom"] for n in keep])
            cv = cv_auc_by_layer(acts, labels, folds)
            n_layers = cv.shape[1]
            validity += [
                {
                    "model": cfg.model,
                    "context": context,
                    "layer": k,
                    "depth": k / max(n_layers - 1, 1),
                    "sentences": len(keep),
                    "auc_mean": float(np.nanmean(cv[:, k])),
                    "auc_min": float(np.nanmin(cv[:, k])),
                }
                for k in range(n_layers)
            ]
            v = md_direction(acts, labels)
            directions[context] = (v, *scale_reference(acts, labels, v))
            click.echo(
                f"  {cfg.model} {context}: {len(keep)} sentences, best held-out AUC {np.nanmax(cv.mean(0)):.3f}"
            )
            del acts
        scores: list[dict[str, Any]] = []
        for task in tasks.split(","):
            scores += _poc_rows(runner, cfg, f"{task}{suffix}", results_dir, directions)
        pd.DataFrame(validity).to_csv(vpath, index=False)
        pd.DataFrame(scores).to_csv(spath, index=False)
        click.echo(f"{cfg.model}: wrote {vpath.name}, {spath.name} ({len(scores)} score rows)")
        del runner
        try:
            import torch  # noqa: PLC0415

            torch.cuda.empty_cache()
        except Exception:
            pass


@main.command()
@click.option(
    "--out-dir", type=click.Path(exists=True, path_type=Path), default=Path("results_figurative")
)
def summary(out_dir: Path) -> None:
    """Validity per model and depth; POC idiom scores; idiom score vs errors and p_gold drop."""
    validity = pd.concat(
        [pd.read_csv(f) for f in sorted(out_dir.glob("*_validity.csv"))], ignore_index=True
    )
    scores = pd.concat(
        [pd.read_csv(f) for f in sorted(out_dir.glob("*_scores.csv"))], ignore_index=True
    )
    best = best_layers(validity)
    chosen = at_best(scores, best)
    fmt = {"display.width": 220, "display.max_rows": 300, "display.float_format": "{:.3f}".format}
    with pd.option_context(*[x for kv in fmt.items() for x in kv]):
        click.echo("1. Held-out AUC (idiom-disjoint folds), best layer per depth quarter:")
        click.echo(depth_table(validity).to_string(index=False))
        click.echo(
            "\n2. Chosen context and layer per model (best mean AUC; chosen on MAGPIE only):"
        )
        click.echo(best.to_string(index=False))
        click.echo(
            "\n3. POC idioms: mean score (+1 = typical figurative use, -1 = typical literal use);"
        )
        click.echo(
            "   score_lit = the literal replacement read along the same direction (reference):"
        )
        means = cast(pd.DataFrame, chosen.groupby("task")[["score_idi", "score_lit"]].mean())
        click.echo(means.to_string())
        click.echo(
            "\n4. Idiom score: items wrong only with the idiom vs right in both (control: wrong only with literal):"
        )
        click.echo(group_tests(chosen).to_string(index=False))
        click.echo(
            "\n5. Idiom score vs p_gold drop (literal - idiomatic), centred within model and task:"
        )
        click.echo(score_vs_pgold(chosen).to_string(index=False))
    best.to_csv(out_dir / "chosen_layers.csv", index=False)
    chosen.to_csv(out_dir / "scores_at_chosen.csv", index=False)


if __name__ == "__main__":
    main()
