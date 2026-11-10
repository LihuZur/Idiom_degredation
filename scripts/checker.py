"""AI checker and repair CLI for v3 pairs (THESIS_PLAN_v3 §4 steps 3-8). `idiom-check --help`.

Commands
- run      Checker verdicts for every pair in one or more generation folders (dry-run: prompts only, free).
- split    Dev / test split of the human-labelled items (per dataset, bad and good items spread over both).
- compare  Checker vs human labels on dev or test: bad items passed, recall, precision, kappa; disagreements.
- repair   Rejected pairs back to the generator with the checker's reason, re-checked, at most 2 rounds.

Paid modes (`--mode sync`) need the provider and model named explicitly: which checker models to try is open
(plan O18), so there is no default. Replies are cached, so a rerun costs nothing.
"""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import click

from augmentation.cache import ResponseCache
from augmentation.checker import (
    TEMPLATES,
    Verdict,
    checker_hash,
    compare,
    disagreements,
    per_dataset,
    render_checker_prompt,
    run_checker,
    split_dev_test,
)
from augmentation.idiom_generation import sync_completer
from augmentation.labelstudio import load_v3
from augmentation.providers.base import build_client
from augmentation.repair import (
    REPAIR_TEMPLATE,
    ROUNDS,
    RepairTask,
    item_from_v3,
    render_repair_prompt,
    repair_prompt_hash,
    repair_reason,
    repair_shortlist,
    run_repair,
)

_DATASETS = ("sst2", "mnli", "mmlu", "agnews")
_CHARS_PER_TOKEN = 4


def _items(runs: tuple[Path, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run in runs:
        for d in _DATASETS:
            if (run / d / "idiomatic.csv").exists():
                out += load_v3(run / d, d)
    return out


def _read_keys(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8", newline="") as f:
        return {r["key"] for r in csv.DictReader(f)}


def _verdict_row(key: str, raw: str | None, v: Verdict) -> dict[str, Any]:
    return {
        "key": key,
        "ok": v.ok,
        "error": v.error,
        "steps": v.steps,
        "reasons": v.reasons,
        "naturalness": v.naturalness,
        "action": v.action,
        "primary_reason": v.primary_reason,
        "rejects": v.rejects,
        "raw": raw,
    }


def _load_verdicts(path: Path) -> dict[str, Verdict]:
    out: dict[str, Verdict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r["key"]] = Verdict(
            r["ok"],
            r["steps"],
            r["reasons"],
            r["naturalness"],
            r["action"],
            r["primary_reason"],
            r["error"],
        )
    return out


def _load_human(path: Path, annotator: str | None) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    names = sorted({r["annotator"] for r in rows})
    if annotator is None:
        if len(names) != 1:
            raise click.UsageError(
                f"several annotators {names}: pass --annotator (or adjudicated labels)"
            )
        annotator = names[0]
    return {r["key"]: r for r in rows if r["annotator"] == annotator}


def _fmt(d: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """AI checker (tuned on human dev labels, scored on test) and repair loop for v3 pairs."""


_run_opt = click.option(
    "--run",
    "runs",
    multiple=True,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Generation output folder(s) with <dataset>/ subfolders (repeatable).",
)


@main.command("run")
@_run_opt
@click.option(
    "--template", type=click.Choice(TEMPLATES), required=True, help="Checker prompt variant."
)
@click.option(
    "--mode", type=click.Choice(["dry-run", "sync"]), default="dry-run", show_default=True
)
@click.option("--provider", type=click.Choice(["openai", "anthropic", "gemini"]), default=None)
@click.option("--model", default=None, help="Exact model id (required for sync).")
@click.option(
    "--keys-from",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="CSV with a `key` column (e.g. the split file): check only these items.",
)
@click.option("--temperature", type=float, default=0.0, show_default=True)
@click.option("--max-output-tokens", type=int, default=1024, show_default=True)
@click.option("--workers", type=int, default=8, show_default=True)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("datasets_out/_checker"),
    show_default=True,
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=Path(".hf_cache/checker"),
    show_default=True,
)
def run_cmd(
    runs: tuple[Path, ...],
    template: str,
    mode: str,
    provider: str | None,
    model: str | None,
    keys_from: Path | None,
    temperature: float,
    max_output_tokens: int,
    workers: int,
    out_dir: Path,
    cache_dir: Path,
) -> None:
    """Checker verdict for every pair (dry-run renders the prompts and estimates tokens; no call)."""
    keys = _read_keys(keys_from)
    items = [i for i in _items(runs) if keys is None or i["key"] in keys]
    stem = Path(template).stem
    if mode == "dry-run":
        target = out_dir / f"{stem}__dry_run"
        target.mkdir(parents=True, exist_ok=True)
        prompts = [render_checker_prompt(i, template) for i in items]
        with (target / "prompts.jsonl").open("w", encoding="utf-8") as f:
            for i, p in zip(items, prompts, strict=True):
                f.write(json.dumps({"key": i["key"], "prompt": p}, ensure_ascii=False) + "\n")
        mean = sum(map(len, prompts)) / max(1, len(prompts))
        click.echo(
            f"dry run: {len(prompts)} prompts ({template}, hash {checker_hash(template)}), "
            f"mean ≈ {mean / _CHARS_PER_TOKEN:.0f} input tokens (estimate) -> {target}"
        )
        if prompts:
            click.echo("--- first prompt ---\n" + prompts[0])
        return
    if provider is None or model is None:
        raise click.UsageError(
            "sync mode needs --provider and --model (plan O18: no default checker)"
        )
    if provider == "gemini":
        click.echo(
            "warning: the generator is Gemini; the plan requires a checker from another family (V1)"
        )
    client = build_client(provider, model)
    complete = sync_completer(
        client,
        ResponseCache(cache_dir),
        checker_hash(template),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        workers=workers,
    )
    results = run_checker(items, complete, template)
    target = out_dir / f"{stem}__{provider}_{model.replace('/', '_')}"
    target.mkdir(parents=True, exist_ok=True)
    with (target / "verdicts.jsonl").open("w", encoding="utf-8") as f:
        for key, (raw, v) in results.items():
            f.write(json.dumps(_verdict_row(key, raw, v), ensure_ascii=False) + "\n")
    (target / "meta.json").write_text(
        json.dumps(
            {
                "template": template,
                "prompt_hash": checker_hash(template),
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "n_items": len(items),
                "runs": [str(r) for r in runs],
                "keys_from": str(keys_from) if keys_from else None,
                "usage_this_run": dict(getattr(client, "usage", {}) or {}),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    c = Counter(
        "unparsed" if not v.ok else ("reject" if v.rejects else "accept")
        for _, v in results.values()
    )
    click.echo(f"{len(results)} verdicts -> {target / 'verdicts.jsonl'}; {dict(c)}")


@main.command("split")
@click.option(
    "--human",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Output of `idiom-labelstudio v3-import`.",
)
@click.option(
    "--annotator", default=None, help="Whose labels are gold (needed if several annotators)."
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--dev-share", type=float, default=0.5, show_default=True)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path("datasets_out/_labelstudio_v3/split.csv"),
    show_default=True,
)
def split_cmd(human: Path, annotator: str | None, seed: int, dev_share: float, out: Path) -> None:
    """Fix the dev / test split once, before any checker is tuned."""
    gold = _load_human(human, annotator)
    split = split_dev_test(gold, seed, dev_share)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "dataset", "split", "human_action"])
        w.writeheader()
        for k, part in sorted(split.items()):
            w.writerow(
                {
                    "key": k,
                    "dataset": gold[k]["dataset"],
                    "split": part,
                    "human_action": gold[k]["action"],
                }
            )
    c = Counter((gold[k]["dataset"], part) for k, part in split.items())
    click.echo(
        f"{len(split)} items -> {out}; "
        + ", ".join(f"{d}/{p}={n}" for (d, p), n in sorted(c.items()))
    )


@main.command("compare")
@click.option(
    "--verdicts",
    "verdict_files",
    multiple=True,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="verdicts.jsonl of one checker configuration (repeatable, to compare candidates).",
)
@click.option(
    "--human", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True
)
@click.option("--annotator", default=None)
@click.option(
    "--split",
    "split_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--part",
    type=click.Choice(["dev", "test"]),
    required=True,
    help="Tune on dev; run test once, after the configuration is frozen.",
)
def compare_cmd(
    verdict_files: tuple[Path, ...], human: Path, annotator: str | None, split_file: Path, part: str
) -> None:
    """Checker vs human: bad items passed (key number), recall, precision, kappa; disagreements to read."""
    gold = _load_human(human, annotator)
    with split_file.open(encoding="utf-8", newline="") as f:
        keys = [r["key"] for r in csv.DictReader(f) if r["split"] == part]
    for vf in verdict_files:
        verdicts = _load_verdicts(vf)
        click.echo(f"== {vf.parent.name} [{part}]  {_fmt(compare(verdicts, gold, keys))}")
        for d, m in per_dataset(verdicts, gold, keys).items():
            click.echo(f"   {d}: {_fmt(m)}")
        rows = disagreements(verdicts, gold, keys)
        out = vf.parent / f"disagreements_{part}.csv"
        if rows:
            with out.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            click.echo(f"   {len(rows)} disagreements -> {out}")


@main.command("repair")
@click.option(
    "--verdicts",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="verdicts.jsonl of the frozen checker; rejected items are repaired.",
)
@_run_opt
@click.option(
    "--mode", type=click.Choice(["dry-run", "sync"]), default="dry-run", show_default=True
)
@click.option("--rounds", type=int, default=ROUNDS, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--gen-model", default="gemini-3.6-flash", show_default=True, help="Generator (frozen setting)."
)
@click.option("--thinking", default="low", show_default=True)
@click.option("--checker-provider", type=click.Choice(["openai", "anthropic"]), default=None)
@click.option("--checker-model", default=None)
@click.option("--checker-template", type=click.Choice(TEMPLATES), default=None)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("datasets_out/_repair"),
    show_default=True,
)
@click.option(
    "--gen-cache-dir",
    type=click.Path(path_type=Path),
    default=Path(".hf_cache/augment_v3"),
    show_default=True,
)
@click.option(
    "--checker-cache-dir",
    type=click.Path(path_type=Path),
    default=Path(".hf_cache/checker"),
    show_default=True,
)
def repair_cmd(
    verdicts: Path,
    runs: tuple[Path, ...],
    mode: str,
    rounds: int,
    seed: int,
    gen_model: str,
    thinking: str,
    checker_provider: str | None,
    checker_model: str | None,
    checker_template: str | None,
    out_dir: Path,
    gen_cache_dir: Path,
    checker_cache_dir: Path,
) -> None:
    """Repair rejected pairs: regenerate with the checker's reason, code checks, re-check (max rounds)."""
    # Lazy: loads the BGE retriever (run with HF_HUB_OFFLINE=1, as for generation).
    from scripts.generate_v3 import load_bank  # noqa: PLC0415

    vmap = _load_verdicts(verdicts)
    by_key = {i["key"]: i for i in _items(runs)}
    idioms, index, retriever = load_bank(gen_cache_dir)
    tasks = [
        RepairTask(
            k,
            item_from_v3(by_key[k]),
            by_key[k]["idiomatic"],
            by_key[k]["literal"],
            by_key[k]["idiom_id"],
            repair_reason(v),
        )
        for k, v in vmap.items()
        if v.rejects and k in by_key
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "dry-run":
        _repair_dry_run(tasks, idioms, retriever, seed, out_dir)
        return
    if not (checker_provider and checker_model and checker_template):
        raise click.UsageError(
            "sync mode needs --checker-provider, --checker-model and --checker-template "
            "(the frozen checker configuration)"
        )
    gen_client = build_client("gemini", gen_model, thinking_level=thinking)
    gen_cache = ResponseCache(gen_cache_dir)

    def gen_for(dataset: str):  # type: ignore[no-untyped-def]
        ns = repair_prompt_hash(dataset)
        ns = ns if thinking == "low" else f"{ns}:thinking={thinking}"
        return sync_completer(
            gen_client, gen_cache, ns, temperature=0.0, max_output_tokens=2048, workers=8
        )

    chk = sync_completer(
        build_client(checker_provider, checker_model),
        ResponseCache(checker_cache_dir),
        checker_hash(checker_template),
        temperature=0.0,
        max_output_tokens=1024,
        workers=8,
    )
    template = checker_template
    outcome = run_repair(
        tasks,
        by_key,
        idioms,
        index,
        retriever,
        gen_for,
        lambda items: {k: v for k, (_, v) in run_checker(items, chk, template).items()},
        seed=seed,
        rounds=rounds,
    )
    unresolved = [{"key": t.key, "last_reason": t.reason} for t in outcome.unresolved]
    for name, rows in (
        ("repair_log.csv", outcome.log),
        ("repaired.csv", outcome.repaired),
        ("needs_human.csv", unresolved),
    ):
        if rows:
            with (out_dir / name).open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    by_round = Counter(r["round"] for r in outcome.repaired)
    click.echo(
        f"funnel: rejected {len(tasks)} -> repaired "
        + (", ".join(f"round {k}: {v}" for k, v in sorted(by_round.items())) or "0")
        + f" -> needs a person: {len(unresolved)}  (files in {out_dir})"
    )


def _repair_dry_run(
    tasks: list[RepairTask], idioms: list[Any], retriever: Any, seed: int, out_dir: Path
) -> None:
    with (out_dir / "dry_run_prompts.jsonl").open("w", encoding="utf-8") as f:
        for t in tasks:
            short = repair_shortlist(
                t.item,
                idioms,
                retriever,
                seed=seed,
                round_=1,
                previous_idiom_id=t.previous_idiom_id,
            )
            p = render_repair_prompt(
                t.item,
                short,
                previous_idiomatic=t.previous_idiomatic,
                previous_literal=t.previous_literal,
                reason=t.reason,
            )
            f.write(
                json.dumps({"key": t.key, "reason": t.reason, "prompt": p}, ensure_ascii=False)
                + "\n"
            )
    click.echo(
        f"dry run: {len(tasks)} rejected items -> {out_dir / 'dry_run_prompts.jsonl'} "
        f"({REPAIR_TEMPLATE}); no call made"
    )


if __name__ == "__main__":
    main()
