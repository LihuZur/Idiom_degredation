"""Stage 3 CLI — run evaluation on a dataset's variant triple (STAGE3_PLAN §3.4)."""

import csv
import hashlib
import json
import platform
import random
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import click
import numpy as np
import pandas as pd
import torch
import transformers
import yaml

import eval as eval_package  # pyright: ignore[reportUnusedImport]  # noqa: F401
import models as models_package  # pyright: ignore[reportUnusedImport]  # noqa: F401
from augmentation.base import AugmentedRow
from eval.base import RunResult
from eval.config import EvalConfig
from eval.io import load_result, write_result
from eval.registry import get_evaluator
from models.base import Prediction
from models.decoder_runner import DecoderRunner
from models.device import select_device
from models.registry import get_model_class, get_model_spec


def resolve_variants(
    dataset: str,
    variants_arg: str | None,
    base_dir: Path = Path("datasets_out"),
) -> list[str]:
    """Resolve which variants to run based on exists on disk and user filter (STAGE3_PLAN §1)."""
    possible = ["original", "paraphrase", "idiomatic"]
    target_dir = base_dir / dataset

    if variants_arg is None:
        existing = []
        for v in possible:
            if (target_dir / f"{v}.csv").exists():
                existing.append(v)
        return existing
    else:
        requested = [v.strip() for v in variants_arg.split(",")]
        for r in requested:
            if r not in possible:
                raise click.BadParameter(f"Invalid variant: {r}. Must be one of {possible}")
            csv_path = target_dir / f"{r}.csv"
            if not csv_path.exists():
                raise click.ClickException(f"Requested variant CSV does not exist: {csv_path}")
        return requested


def _seed_everything(seed: int) -> None:
    """Ensure complete reproducibility where possible (STAGE3_PLAN §1)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _replay_results(
    dataset: str,
    variants_to_run: list[str],
    limit: int | None,
    from_results: Path,
    evaluator: Any,
) -> dict[str, RunResult]:
    """Replay parsing + metrics from a prior result JSON (STAGE3_PLAN §1)."""
    prior = load_result(from_results)

    # Calculate current prompt hash
    current_system = getattr(evaluator, "system_prompt", "")
    current_user_template = getattr(evaluator, "user_template", "")
    prompt_hash = hashlib.sha256(
        (current_system + "\x1e" + current_user_template).encode("utf-8")
    ).hexdigest()[:16]

    if prior.get("prompt_hash") != prompt_hash:
        raise click.ClickException(
            f"Prompt hash mismatch. Prior: {prior.get('prompt_hash')}, Current: {prompt_hash}."
        )

    run_results = {}
    for var in variants_to_run:
        csv_path = Path("datasets_out") / dataset / f"{var}.csv"
        # Read examples
        examples = []
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                meta_str = row.get("meta", "{}")
                meta_val = json.loads(meta_str) if meta_str else {}
                meta = meta_val if isinstance(meta_val, dict) else {}
                examples.append(
                    Prediction(
                        id=row["id"],
                        raw="",
                        parsed=None,
                        meta={"ex_y": row["y"], "ex": row, "meta": meta},
                    )
                )

        if limit is not None:
            examples = examples[:limit]

        id_to_raw = {}
        for task in prior.get("per_task", []):
            if var in task:
                id_to_raw[task["id"]] = task[var]["raw"]

        scored_preds = []
        n_correct = 0
        n_unparseable = 0
        unparseable_ids = []

        for ex_pred in examples:
            if ex_pred.id not in id_to_raw:
                raise click.ClickException(
                    f"Example ID {ex_pred.id} not found in prior results for variant {var}"
                )
            raw = id_to_raw[ex_pred.id]
            ex_meta = ex_pred.meta["meta"]

            ex = AugmentedRow(
                id=ex_pred.id,
                variant=var,  # type: ignore
                x=ex_pred.meta["ex"]["x"],
                y=ex_pred.meta["ex_y"],
                augmenter_model=ex_pred.meta["ex"].get("augmenter_model", ""),
                prompt_hash=ex_pred.meta["ex"].get("prompt_hash", ""),
                meta=ex_meta,
            )

            parsed, parse_status = evaluator.parse(raw, ex)
            correct = evaluator.score(parsed, ex.y)

            if parse_status == "unparseable":
                unparseable_ids.append(ex.id)
                n_unparseable += 1
            if correct:
                n_correct += 1

            scored_preds.append(
                Prediction(
                    id=ex.id,
                    raw=raw,
                    parsed=parsed,
                    meta={"parse_status": parse_status, "correct": correct},
                )
            )

        n = len(examples)
        accuracy = n_correct / n if n > 0 else 0.0
        unparseable_rate = n_unparseable / n if n > 0 else 0.0

        metrics = {
            "accuracy": accuracy,
            "unparseable_rate": unparseable_rate,
            "n": float(n),
            "n_unparseable": float(n_unparseable),
        }

        wall_time = prior.get("wall_time_seconds", {}).get(var, 0.0)

        run_results[var] = RunResult(
            variant=var,  # type: ignore
            metrics=metrics,
            predictions=scored_preds,
            meta={"unparseable_ids": unparseable_ids, "wall_time_seconds": wall_time},
        )
    return run_results


def _prediction_from_checkpoint(d: dict[str, Any]) -> Prediction:
    return Prediction(id=d["id"], raw=d["raw"], parsed=d["parsed"], meta=d["meta"])


def _predictions_to_checkpoint(predictions: list[Prediction]) -> list[dict[str, Any]]:
    return [{"id": p.id, "raw": p.raw, "parsed": p.parsed, "meta": p.meta} for p in predictions]


def _make_checkpoint_callback(
    checkpoint_file: Path,
    checkpoint_state: dict[str, Any],
    var: str,
    merged_so_far: list[Prediction],
    prior_wall_time: float,
) -> Callable[[list[Prediction]], None]:
    """Build a per-variant callback that appends a batch and re-persists the checkpoint."""

    def _on_new_predictions(batch: list[Prediction]) -> None:
        merged_so_far.extend(batch)
        checkpoint_state.setdefault("variants", {})[var] = {
            "predictions": _predictions_to_checkpoint(merged_so_far),
            "wall_time_seconds": prior_wall_time,
        }
        write_result(checkpoint_file, checkpoint_state)

    return _on_new_predictions


def _run_inference(
    cfg: EvalConfig,
    dataset: str,
    variants_to_run: list[str],
    limit: int | None,
    evaluator: Any,
    checkpoint_file: Path | None = None,
    checkpoint_state: dict[str, Any] | None = None,
) -> dict[str, RunResult]:
    """Run model inference normally, checkpointing progress after every batch if requested."""
    model_spec = get_model_spec(cfg.model)
    runner_cls = get_model_class(cfg.model)

    concrete_runner_cls = cast(type[DecoderRunner], runner_cls)
    runner = concrete_runner_cls(model_spec, precision=cfg.precision)

    checkpoint_state = checkpoint_state if checkpoint_state is not None else {}

    run_results = {}
    for var in variants_to_run:
        csv_path = Path("datasets_out") / dataset / f"{var}.csv"

        prior_variant = checkpoint_state.get("variants", {}).get(var, {})
        already_done = {
            d["id"]: _prediction_from_checkpoint(d) for d in prior_variant.get("predictions", [])
        }
        prior_wall_time = prior_variant.get("wall_time_seconds", 0.0)
        if already_done:
            click.echo(f"  {dataset}/{var}: resuming, {len(already_done)} example(s) already done")

        on_new_predictions: Callable[[list[Prediction]], None] | None = None
        if checkpoint_file is not None:
            on_new_predictions = _make_checkpoint_callback(
                checkpoint_file,
                checkpoint_state,
                var,
                list(already_done.values()),
                prior_wall_time,
            )

        res = evaluator.run_variant(
            runner,
            csv_path,
            limit=limit,
            already_done=already_done,
            prior_wall_time=prior_wall_time,
            on_new_predictions=on_new_predictions,
        )
        run_results[var] = res

        if checkpoint_file is not None:
            checkpoint_state.setdefault("variants", {})[var] = {
                "predictions": _predictions_to_checkpoint(res.predictions),
                "wall_time_seconds": res.meta.get("wall_time_seconds", 0.0),
            }
            write_result(checkpoint_file, checkpoint_state)

    return run_results


def _resolve_checkpoint(
    checkpoint_dir: Path | None,
    dataset: str,
    model: str,
    prompt_hash: str,
    config_hash_val: str,
) -> tuple[Path | None, dict[str, Any]]:
    """Load a prior checkpoint for this dataset+model, or seed a fresh one."""
    if checkpoint_dir is None:
        return None, {}

    checkpoint_file = checkpoint_dir / dataset / f"{model.replace('/', '_')}.checkpoint.json"
    if not checkpoint_file.exists():
        checkpoint_state = {
            "dataset": dataset,
            "model_id": model,
            "prompt_hash": prompt_hash,
            "config_hash": config_hash_val,
            "variants": {},
        }
        return checkpoint_file, checkpoint_state

    prior_checkpoint = load_result(checkpoint_file)
    if prior_checkpoint.get("prompt_hash") != prompt_hash:
        raise click.ClickException(
            "Checkpoint prompt hash mismatch (prior run used a different prompt); "
            f"refusing to resume from stale checkpoint: {checkpoint_file}"
        )
    if prior_checkpoint.get("config_hash") != config_hash_val:
        raise click.ClickException(
            "Checkpoint config hash mismatch (prior run used a different config); "
            f"refusing to resume from stale checkpoint: {checkpoint_file}"
        )
    click.echo(f"Resuming from checkpoint: {checkpoint_file}")
    return checkpoint_file, prior_checkpoint


def _build_per_task(
    dataset: str,
    variants_to_run: list[str],
    run_results: dict[str, RunResult],
) -> list[dict[str, Any]]:
    first_var = variants_to_run[0]
    csv_path = Path("datasets_out") / dataset / f"{first_var}.csv"
    id_to_y = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_to_y[row["id"]] = row["y"]

    pred_maps = {var: {p.id: p for p in res.predictions} for var, res in run_results.items()}

    # analysis.results.load_result() requires every per_task row to have all
    # variants_to_run present - build only over ids common to every variant (original.csv
    # can have ids that augmentation's reconciliation dropped from paraphrase/idiomatic).
    common_ids = set(pred_maps[first_var])
    for var in variants_to_run[1:]:
        common_ids &= set(pred_maps[var])

    per_task = []
    for p_first in run_results[first_var].predictions:
        task_id = p_first.id
        if task_id not in common_ids:
            continue
        task_obj: dict[str, Any] = {
            "id": task_id,
            "y": id_to_y.get(task_id, ""),
        }
        for var in variants_to_run:
            p = pred_maps[var][task_id]
            task_obj[var] = {
                "raw": p.raw,
                "parsed": p.parsed,
                "parse_status": p.meta.get("parse_status", "ok"),
                "correct": p.meta.get("correct", False),
            }
        per_task.append(task_obj)
    return per_task


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the per-model YAML config.",
)
@click.option(
    "--dataset", required=True, type=str, help="Registered dataset name (sst2, mmlu, mnli)."
)
@click.option(
    "--variants",
    default=None,
    type=str,
    help="Comma-separated subset of original,paraphrase,idiomatic.",
)
@click.option(
    "--out-dir",
    default=Path("results"),
    type=click.Path(path_type=Path),
    help="Directory to write results JSON to.",
)
@click.option("--seed", default=None, type=int, help="Override seed value in config.")
@click.option("--limit", default=None, type=int, help="Limit number of rows evaluated per variant.")
@click.option("--force", is_flag=True, default=False, help="Overwrite existing result file.")
@click.option(
    "--from-results",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Replay parsing + metrics from a prior result JSON.",
)
@click.option(
    "--checkpoint-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Directory to save/read mid-run checkpoints (e.g. a mounted Drive folder), so a "
    "crashed/disconnected run can resume from the last completed batch instead of "
    "restarting a variant from scratch.",
)
def main(
    config: Path,
    dataset: str,
    variants: str | None,
    out_dir: Path,
    seed: int | None,
    limit: int | None,
    force: bool,
    from_results: Path | None,
    checkpoint_dir: Path | None,
) -> None:
    """Stage 3: evaluate model on a dataset's variant triple."""
    # Load and validate config
    cfg = EvalConfig.model_validate(yaml.safe_load(config.read_text()))
    if seed is not None:
        cfg = cfg.model_copy(update={"seed": seed})

    _seed_everything(cfg.seed)

    evaluator_cls = get_evaluator(dataset)
    evaluator = evaluator_cls(cfg)

    # Calculate current prompt hash
    current_system = getattr(evaluator, "system_prompt", "")
    current_user_template = getattr(evaluator, "user_template", "")
    prompt_hash = hashlib.sha256(
        (current_system + "\x1e" + current_user_template).encode("utf-8")
    ).hexdigest()[:16]

    resolved_config_dict = cfg.model_dump()
    resolved_config_json = json.dumps(resolved_config_dict, sort_keys=True)
    config_hash_val = hashlib.sha256(resolved_config_json.encode("utf-8")).hexdigest()[:16]

    # Resolve variants to run
    variants_to_run = resolve_variants(dataset, variants)
    if not variants_to_run:
        raise click.ClickException(f"No variant CSV files found in datasets_out/{dataset}")

    out_path = out_dir / dataset / f"{cfg.model.replace('/', '_')}.json"
    if out_path.exists() and not force:
        raise click.ClickException(f"Result file exists (use --force to overwrite): {out_path}")

    model_spec = get_model_spec(cfg.model)
    precision_str = cfg.precision or model_spec.default_precision
    device_type = select_device().type

    checkpoint_file, checkpoint_state = (
        _resolve_checkpoint(checkpoint_dir, dataset, cfg.model, prompt_hash, config_hash_val)
        if from_results is None
        else (None, {})
    )

    if from_results is not None:
        run_results = _replay_results(dataset, variants_to_run, limit, from_results, evaluator)
    else:
        run_results = _run_inference(
            cfg, dataset, variants_to_run, limit, evaluator, checkpoint_file, checkpoint_state
        )

    # Get y labels and build per_task
    per_task = _build_per_task(dataset, variants_to_run, run_results)

    metrics_dict = {var: res.metrics for var, res in run_results.items()}
    unparseable_ids_dict = {
        var: res.meta.get("unparseable_ids", []) for var, res in run_results.items()
    }
    wall_time_dict = {
        var: res.meta.get("wall_time_seconds", 0.0) for var, res in run_results.items()
    }

    result_json = {
        "stage": "eval",
        "dataset": dataset,
        "model_id": cfg.model,
        "model_revision": model_spec.revision,
        "hf_repo": model_spec.hf_repo,
        "precision": precision_str,
        "device": device_type,
        "config_path": str(config.resolve()),
        "config_hash": config_hash_val,
        "resolved_config": resolved_config_dict,
        "prompt_hash": prompt_hash,
        "seed": cfg.seed,
        "variants_run": variants_to_run,
        "tool_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "pandas": pd.__version__,
        },
        "metrics": metrics_dict,
        "unparseable_ids": unparseable_ids_dict,
        "per_task": per_task,
        "wall_time_seconds": wall_time_dict,
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }

    write_result(out_path, result_json)
    click.echo(f"Results successfully written to {out_path}")

    if checkpoint_file is not None and checkpoint_file.exists():
        checkpoint_file.unlink()


if __name__ == "__main__":
    main()
