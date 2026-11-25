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
import math
from collections import Counter
from collections.abc import Mapping, Sequence
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
from augmentation.selection import Pair, length_bin, length_bins, select_pairs

_DATASETS = ("sst2", "mnli", "mmlu", "agnews", "siqa")
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


def majority(
    runs: Sequence[Mapping[str, tuple[str | None, Verdict]]],
) -> dict[str, tuple[str | None, Verdict]]:
    """Per key: reject if most runs reject (an unparsable run counts as a reject); the kept verdict is the first run
    that agrees with the majority (IdioLink §3.2.4, plan O20)."""
    if len(runs) == 1:
        return dict(runs[0])
    out: dict[str, tuple[str | None, Verdict]] = {}
    for key in runs[0]:
        rej = [r[key][1].rejects for r in runs]
        decided = sum(rej) * 2 > len(rej)
        out[key] = next(r[key] for r in runs if r[key][1].rejects == decided)
    return out


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
@click.option(
    "--votes",
    type=int,
    default=1,
    show_default=True,
    help="Run the checker N times and reject by majority (IdioLink §3.2.4: 3 runs; plan O20). Run 1 reuses the "
    "single-run cache; runs 2..N get their own cache keys.",
)
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
    votes: int,
    workers: int,
    out_dir: Path,
    cache_dir: Path,
) -> None:
    """Checker verdict for every pair (dry-run renders the prompts and estimates tokens; no call)."""
    keys = _read_keys(keys_from)
    items = [i for i in _items(runs) if keys is None or i["key"] in keys]
    dup = [k for k, n in Counter(i["key"] for i in items).items() if n > 1]
    if (
        dup
    ):  # verdicts are keyed by task:id, so the same item from two runs would overwrite (2026-11-16)
        raise click.UsageError(
            f"{len(dup)} items appear in more than one --run (e.g. {dup[0]}); check each run separately "
            "with its own --out-dir"
        )
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
    runs_out: list[dict[str, tuple[str | None, Verdict]]] = []
    for n in range(1, votes + 1):
        ns = checker_hash(template) if n == 1 else f"{checker_hash(template)}:vote{n}"
        complete = sync_completer(
            client,
            ResponseCache(cache_dir),
            ns,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            workers=workers,
        )
        runs_out.append(run_checker(items, complete, template))
    results = majority(runs_out)
    suffix = "" if votes == 1 else f"__votes{votes}"
    target = out_dir / f"{stem}__{provider}_{model.replace('/', '_')}{suffix}"
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
                "votes": votes,
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


_MIN_KEPT = 0.80  # plan O2, decided 2026-11-25
_POOL_DIRS = {
    "sst2": "sst2",
    "mnli": "mnli_wide",
    "mmlu": "mmlu_wide",
    "agnews": "agnews_wide",
    "siqa": "siqa",
}


def label_balance(
    drawn: dict[str, str],
    kept: set[str],
    accepted: set[str] | None,
    pool: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Per label: count and share in the pool / drawn / kept by code / accepted by the filter; χ² p-value of the
    drawn sample against the pool shares and of each later stage against the drawn shares (plan O39)."""
    from scipy.stats import chisquare  # noqa: PLC0415

    stages: dict[str, list[str]] = {"drawn": list(drawn), "kept": [i for i in drawn if i in kept]}
    if accepted is not None:
        stages["accepted"] = [i for i in stages["kept"] if i in accepted]
    labels = sorted(set(drawn.values()))
    base = Counter(drawn.values())
    rows: list[dict[str, Any]] = []
    if pool is not None:
        pc = Counter(pool)
        n_d = len(drawn)
        p_d = float(
            chisquare(
                [base[lab] for lab in labels], [pc[lab] / len(pool) * n_d for lab in labels]
            ).pvalue
        )
        for lab in labels:
            rows.append(
                {
                    "stage": "pool",
                    "label": lab,
                    "n": pc[lab],
                    "share": round(pc[lab] / len(pool) * 100, 1),
                    "p_vs_drawn": float("nan"),
                }
            )
        rows += [
            {
                "stage": "drawn_vs_pool",
                "label": "",
                "n": n_d,
                "share": float("nan"),
                "p_vs_drawn": p_d,
            }
        ]
    for stage, ids in stages.items():
        c = Counter(drawn[i] for i in ids)
        n = len(ids)
        p = float("nan")
        if stage != "drawn" and n:
            exp = [base[lab] / len(drawn) * n for lab in labels]
            p = float(chisquare([c[lab] for lab in labels], exp).pvalue)
        for lab in labels:
            rows.append(
                {
                    "stage": stage,
                    "label": lab,
                    "n": c[lab],
                    "share": round(c[lab] / n * 100, 1) if n else float("nan"),
                    "p_vs_drawn": p,
                }
            )
    return rows


@main.command("balance")
@_run_opt
@click.option(
    "--verdicts",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="verdicts.jsonl of a filter run; accepted = not rejected.",
)
@click.option(
    "--pools", type=click.Path(path_type=Path), default=Path("datasets_out/wide"), show_default=True
)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), default=None)
def balance_cmd(
    runs: tuple[Path, ...], verdicts: Path | None, pools: Path, out: Path | None
) -> None:
    """Label distribution of drawn vs code-kept vs filter-accepted rows, per dataset (plan O39)."""
    accepted: set[str] | None = None
    if verdicts is not None:
        accepted = {k for k, v in _load_verdicts(verdicts).items() if not v.rejects}
    all_rows: list[dict[str, Any]] = []
    for d in _DATASETS:
        logs = [
            r / d / "generation_log.csv" for r in runs if (r / d / "generation_log.csv").exists()
        ]
        if not logs:
            continue
        with (pools / _POOL_DIRS[d] / "original.csv").open(encoding="utf-8", newline="") as f:
            pool_rows = list(csv.DictReader(f))
        pool = {r["id"]: r["y"] for r in pool_rows}
        words = sorted(len(r["x"].split()) for r in pool_rows)
        cuts = (words[len(words) // 3], words[2 * len(words) // 3])
        length = {
            r["id"]: ("short", "medium", "long")[
                (len(r["x"].split()) >= cuts[0]) + (len(r["x"].split()) >= cuts[1])
            ]
            for r in pool_rows
        }
        drawn: dict[str, str] = {}
        for log in logs:
            with log.open(encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    drawn[f"{d}:{r['id']}"] = pool[r["id"]]
        kept = {
            i["key"] for r in runs if (r / d / "idiomatic.csv").exists() for i in load_v3(r / d, d)
        }
        for row in label_balance(drawn, kept, accepted, list(pool.values())):
            all_rows.append({"dataset": d, **row})
        # same report by text length (pool word-count terciles): do skipped / rejected items lean one way?
        drawn_len = {k: length[k.split(":", 1)[1]] for k in drawn}
        for row in label_balance(drawn_len, kept, accepted, list(length.values())):
            all_rows.append({"dataset": f"{d} (length)", **row})
    for d in dict.fromkeys(r["dataset"] for r in all_rows):
        rows = [r for r in all_rows if r["dataset"] == d]
        if not rows:
            continue
        click.echo(f"== {d}")
        for stage in dict.fromkeys(r["stage"] for r in rows):
            part = [r for r in rows if r["stage"] == stage]
            p = part[0]["p_vs_drawn"]
            if stage == "drawn_vs_pool":
                click.echo(f"  drawn vs pool: χ² p = {p:.2f}")
                continue
            cells = "  ".join(f"{r['label']}: {r['n']} ({r['share']}%)" for r in part)
            click.echo(
                f"  {stage:<9} {cells}" + ("" if math.isnan(p) else f"   χ² p vs drawn = {p:.2f}")
            )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["dataset", "stage", "label", "n", "share", "p_vs_drawn"]
            )
            w.writeheader()
            w.writerows(all_rows)
        click.echo(f"-> {out}")


@main.command("keep")
@_run_opt
@click.option(
    "--verdicts",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="verdicts.jsonl of the filter run; rows it rejects (or could not parse) are dropped.",
)
@click.option("--suffix", default="_v3poc2", show_default=True, help="Output dataset name suffix.")
@click.option(
    "--out-root", type=click.Path(path_type=Path), default=Path("datasets_out"), show_default=True
)
def keep_cmd(runs: tuple[Path, ...], verdicts: Path, suffix: str, out_root: Path) -> None:
    """Write the filter-accepted pairs as evaluation datasets <task><suffix>/ (plan O38, step 8d)."""
    accepted = {k for k, v in _load_verdicts(verdicts).items() if not v.rejects}
    for d in _DATASETS:
        rows: dict[str, list[dict[str, str]]] = {
            "original": [],
            "idiomatic": [],
            "span_literal": [],
        }
        fields: dict[str, list[str]] = {}
        for run in runs:
            if not (run / d / "idiomatic.csv").exists():
                continue
            for version in rows:
                with (run / d / f"{version}.csv").open(encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    fields[version] = list(reader.fieldnames or [])
                    rows[version] += [r for r in reader if f"{d}:{r['id']}" in accepted]
        if not fields:
            continue
        target = out_root / f"{d}{suffix}"
        target.mkdir(parents=True, exist_ok=True)
        ids = {r["id"] for r in rows["idiomatic"]} & {r["id"] for r in rows["span_literal"]}
        for version, rs in rows.items():
            with (target / f"{version}.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields[version])
                w.writeheader()
                w.writerows(r for r in rs if r["id"] in ids)
        click.echo(f"{d}: {len(ids)} accepted pairs -> {target}")


@main.command("positions")
@_run_opt
def positions_cmd(runs: tuple[Path, ...]) -> None:
    """Where the chosen idiom sat in the list shown (quarters, top first) and idiom variety, per run (plan O41)."""
    for run in runs:
        pos: list[float] = []
        names: list[str] = []
        for d in _DATASETS:
            f = run / d / "idiomatic.csv"
            if not f.exists():
                continue
            with f.open(encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh):
                    rec = json.loads(r["meta"])["idiom"]
                    shown = list(rec.get("shown_ids") or [])
                    chosen = str(rec["midas_ids"]).split("|")[0]
                    names.append(str(rec["idiom"]))
                    if chosen in shown:
                        pos.append(shown.index(chosen) / len(shown))
        if not pos:
            click.echo(f"{run}: no recorded lists (shown_ids)")
            continue
        q = [sum(a <= x < a + 0.25 for x in pos) / len(pos) for a in (0, 0.25, 0.5, 0.75)]
        c = Counter(names)
        top10 = sum(v for _, v in c.most_common(10)) / len(names)
        click.echo(
            f"{run}: {len(pos)} picks; by quarter of the list (top first) "
            + " / ".join(f"{x:.0%}" for x in q)
            + f"; {len(c)} distinct idioms in {len(names)} pairs, top-10 share {top10:.0%}"
        )


def _pool_info(pools: Path, d: str) -> tuple[dict[str, dict[str, str]], tuple[int, int]]:
    with (pools / _POOL_DIRS[d] / "original.csv").open(encoding="utf-8", newline="") as f:
        pool = {r["id"]: r for r in csv.DictReader(f)}
    return pool, length_bins([len(r["x"].split()) for r in pool.values()])


def _stratum(row: Mapping[str, str], d: str, cuts: tuple[int, int]) -> str:
    b = length_bin(len(row["x"].split()), cuts)
    if d == "mnli":
        b += "|" + str(json.loads(row.get("meta") or "{}").get("genre", ""))
    return b


def _pool_shares(
    pool: Mapping[str, Mapping[str, str]], d: str, cuts: tuple[int, int]
) -> dict[str, dict[str, float]]:
    counts: dict[str, Counter[str]] = {}
    for r in pool.values():
        counts.setdefault(r["y"], Counter())[_stratum(r, d, cuts)] += 1
    return {lab: {k: v / sum(c.values()) for k, v in c.items()} for lab, c in counts.items()}


def position_bins(
    picks: Sequence[tuple[int, int]], n_bins: int = 5
) -> tuple[list[int], list[float]]:
    """Chosen-idiom positions (index, list length) counted in `n_bins` blocks of the list, with the counts expected
    if the choice ignored position (each block's share = the positions it holds / list length)."""
    obs = [0] * n_bins
    exp = [0.0] * n_bins
    for idx, length in picks:
        obs[min(n_bins - 1, idx * n_bins // length)] += 1
        for k in range(length):
            exp[min(n_bins - 1, k * n_bins // length)] += 1 / length
    return obs, exp


def _sha256(path: Path) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(path.read_bytes()).hexdigest()


@main.command("combine")
@click.option(
    "--verdicts",
    "verdict_files",
    multiple=True,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="verdicts.jsonl of checker runs (repeatable; e.g. the main run and a top-up).",
)
@click.option(
    "--review",
    "review_files",
    multiple=True,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Second reviewer CSV(s) with columns key and action (valid / fix / invalid); repeatable.",
)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def combine_cmd(verdict_files: tuple[Path, ...], review_files: tuple[Path, ...], out: Path) -> None:
    """One verdicts.jsonl where a pair passes only if the checker AND the second review pass it (plan O53 e)."""
    checker: dict[str, Verdict] = {}
    for f in verdict_files:
        checker.update(_load_verdicts(f))
    review: dict[str, str] = {}
    for f in review_files:
        with f.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["key"] in review:
                    raise click.ClickException(f"{r['key']} appears in more than one review file")
                review[r["key"]] = r["action"]
    missing = sorted(set(checker) ^ set(review))
    if missing:
        raise click.ClickException(
            f"{len(missing)} keys in only one of checker / review, e.g. {missing[:3]}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with out.open("w", encoding="utf-8") as f:
        for key in sorted(checker):
            v = checker[key]
            rev_ok = review[key] == "valid"
            counts[
                ("pass" if not v.rejects else "reject") + "/" + ("pass" if rev_ok else "reject")
            ] += 1
            action = v.action if v.rejects else ("valid" if rev_ok else "fix")
            reason = v.primary_reason if v.rejects else ("none" if rev_ok else "second_review")
            row = {
                "key": key,
                "ok": v.ok,
                "steps": v.steps,
                "reasons": v.reasons,
                "naturalness": v.naturalness,
                "action": action,
                "primary_reason": reason,
                "error": v.error,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    click.echo(
        f"checker/review: {dict(sorted(counts.items()))}; both pass: {counts['pass/pass']} -> {out}"
    )


@main.command("select")
@_run_opt
@click.option("--dataset", type=click.Choice(_DATASETS), required=True)
@click.option(
    "--verdicts",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="verdicts.jsonl of the checker run; rejected (or unparsable) pairs are dropped.",
)
@click.option("--n", type=int, required=True, help="Final pairs wanted (split evenly over labels).")
@click.option(
    "--min-kept", type=float, default=0.80, show_default=True, help="Plan O2 (decided 80%)."
)
@click.option(
    "--cap", type=int, default=5, show_default=True, help="Max pairs per idiom (O53, decided 5)."
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--suffix", default="_full", show_default=True)
@click.option(
    "--pools", type=click.Path(path_type=Path), default=Path("datasets_out/wide"), show_default=True
)
@click.option(
    "--out-root", type=click.Path(path_type=Path), default=Path("datasets_out"), show_default=True
)
@click.option(
    "--allow-short",
    is_flag=True,
    help="Write the dataset even if a label is short of its quota (default: report and write nothing).",
)
def select_cmd(
    runs: tuple[Path, ...],
    dataset: str,
    verdicts: Path,
    n: int,
    min_kept: float,
    cap: int,
    seed: int,
    suffix: str,
    pools: Path,
    out_root: Path,
    allow_short: bool,
) -> None:
    """Final full-study dataset: words-kept threshold, idiom cap, equal labels, pool-like strata (plan O53)."""
    accepted = {k for k, v in _load_verdicts(verdicts).items() if not v.rejects}
    pool, cuts = _pool_info(pools, dataset)
    pairs: list[Pair] = []
    rows: dict[str, dict[str, dict[str, str]]] = {
        "original": {},
        "idiomatic": {},
        "span_literal": {},
    }
    fields: dict[str, list[str]] = {}
    for run in runs:
        folder = run / dataset
        if not (folder / "idiomatic.csv").exists():
            continue
        for version, by_id in rows.items():
            with (folder / f"{version}.csv").open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fields[version] = list(reader.fieldnames or [])
                by_id.update({r["id"]: r for r in reader})
    for i, r in rows["idiomatic"].items():
        if f"{dataset}:{i}" not in accepted or i not in rows["span_literal"] or i not in pool:
            continue
        rec = json.loads(r["meta"])["idiom"]
        pairs.append(
            Pair(
                id=i,
                label=pool[i]["y"],
                stratum=_stratum(pool[i], dataset, cuts),
                idiom_id=str(rec["midas_ids"]).split("|")[0],
                words_kept=float(rec.get("original_words_kept", 1.0)),
            )
        )
    sel = select_pairs(
        pairs, n, _pool_shares(pool, dataset, cuts), min_kept=min_kept, cap=cap, seed=seed
    )
    rep = sel.report
    click.echo(
        f"{dataset}: {rep['checked_pairs']} pairs passed the checker; dropped {rep['dropped_words_kept']} "
        f"(words kept < {min_kept:.0%}), {rep['dropped_idiom_cap']} (idiom cap {cap}); selected {rep['selected']} of {n}"
    )
    for lab, v in rep["per_label"].items():
        short = f"  SHORT BY {v['short_by']}" if v["short_by"] else ""
        click.echo(
            f"  label {lab}: quota {v['quota']}, available {v['available']}, selected {v['selected']}{short}"
        )
    if not sel.complete and not allow_short:
        raise click.ClickException(
            "a label is short of its quota: generate more items (a top-up run with --usage-from and "
            "--exclude-ids-from), or pass --allow-short; nothing written"
        )
    target = out_root / f"{dataset}{suffix}"
    target.mkdir(parents=True, exist_ok=True)
    wanted = set(sel.ids)
    manifest: dict[str, Any] = {
        "dataset": dataset,
        "runs": [str(r) for r in runs],
        "verdicts": str(verdicts),
        "verdicts_sha256": _sha256(verdicts),
        "n": n,
        "min_kept": min_kept,
        "cap": cap,
        "seed": seed,
        "report": rep,
        "files": {},
    }
    for version, rs in rows.items():
        path = target / f"{version}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields[version])
            w.writeheader()
            w.writerows(rs[i] for i in sorted(wanted))
        manifest["files"][path.name] = _sha256(path)
    (target / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    click.echo(f"-> {target} (file hashes in selection_manifest.json; freeze before evaluation)")


@main.command("gates")
@click.option(
    "--data",
    "data_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="A final dataset folder written by `select` (e.g. datasets_out/mnli_full).",
)
@click.option("--dataset", type=click.Choice(_DATASETS), required=True)
@click.option(
    "--pools", type=click.Path(path_type=Path), default=Path("datasets_out/wide"), show_default=True
)
def gates_cmd(data_dir: Path, dataset: str, pools: Path) -> None:
    """Data-only reports to read before any model is run (plan O53 f): labels, strata vs pool, words kept, idiom
    reuse, list positions, idiom properties vs the bank, file hashes. Nothing is removed."""
    from scipy.stats import chisquare  # noqa: PLC0415

    from augmentation.idiom_generation import load_inventory  # noqa: PLC0415
    from data.norms.inventory import DEFAULT_FINAL_PATH, DEFAULT_MAGPIE_EXT_PATH  # noqa: PLC0415

    pool, cuts = _pool_info(pools, dataset)
    with (data_dir / "idiomatic.csv").open(encoding="utf-8", newline="") as f:
        idi = list(csv.DictReader(f))
    recs = [json.loads(r["meta"])["idiom"] for r in idi]
    labels = Counter(pool[r["id"]]["y"] for r in idi)
    p_lab = float(chisquare(list(labels.values())).pvalue) if len(labels) > 1 else float("nan")
    click.echo(f"1. labels: {dict(sorted(labels.items()))}  (χ² p vs equal = {p_lab:.2f})")
    shares = _pool_shares(pool, dataset, cuts)
    for lab in sorted(labels):
        got = Counter(
            _stratum(pool[r["id"]], dataset, cuts) for r in idi if pool[r["id"]]["y"] == lab
        )
        keys = sorted(shares[lab])
        exp = [shares[lab][k] * labels[lab] for k in keys]
        p = float(chisquare([got[k] for k in keys], exp).pvalue) if len(keys) > 1 else float("nan")
        worst = max(keys, key=lambda k: abs(got[k] - shares[lab][k] * labels[lab]))
        click.echo(
            f"2. label {lab}: strata vs pool χ² p = {p:.2f}; largest gap {worst}: "
            f"{got[worst]} vs {shares[lab][worst] * labels[lab]:.1f} expected"
        )
    kept = sorted(float(x.get("original_words_kept", 1.0)) for x in recs)
    click.echo(
        f"3. words kept: min {kept[0]:.2f}, median {kept[len(kept) // 2]:.2f}, below {_MIN_KEPT:.2f}: {sum(k < _MIN_KEPT for k in kept)}"
    )
    uses = Counter(str(x["midas_ids"]).split("|")[0] for x in recs)
    top10 = sum(v for _, v in uses.most_common(10)) / len(recs)
    click.echo(
        f"4. idioms: {len(uses)} distinct in {len(recs)} pairs; max uses {max(uses.values())}; top-10 share {top10:.0%}"
    )
    picks = [
        (list(x["shown_ids"]).index(str(x["midas_ids"]).split("|")[0]), len(x["shown_ids"]))
        for x in recs
        if x.get("shown_ids") and str(x["midas_ids"]).split("|")[0] in x["shown_ids"]
    ]
    q, exp = position_bins(picks)
    p_pos = float(chisquare(q, exp).pvalue) if sum(q) else float("nan")
    click.echo(
        f"5. position of the chosen idiom in the list shown (5 equal blocks, top first): {q} "
        f"(expected {[round(e, 1) for e in exp]}; χ² p = {p_pos:.3f})"
    )
    inv = DEFAULT_FINAL_PATH.parent
    bank, _ = load_inventory(
        DEFAULT_FINAL_PATH,
        impli_csv=inv / "impli_gloss_seed.csv",
        domains_csv=[inv / "idiom_domains.csv", inv / "idiom_domains_magpie.csv"],
        extra_csvs=[DEFAULT_MAGPIE_EXT_PATH],
    )
    by_id = {b.idiom_id: b for b in bank}
    picked = [by_id[i] for i in uses.elements() if i in by_id]

    def _mean(xs: Sequence[float | None]) -> str:
        v = [x for x in xs if x is not None]
        return f"{sum(v) / len(v):.2f} (n {len(v)})" if v else "-"

    for prop in ("log_freq", "familiarity", "opacity"):
        click.echo(
            f"6. {prop}: picked {_mean([getattr(b, prop) for b in picked])} vs bank {_mean([getattr(b, prop) for b in bank])}"
        )
    dom_p, dom_b = Counter(b.domain for b in picked), Counter(b.domain for b in bank)
    click.echo(
        "6. domain share picked / bank: "
        + ", ".join(
            f"{k}: {dom_p[k] / len(picked):.0%} / {dom_b[k] / len(bank):.0%}"
            for k in sorted(dom_b, key=str)
        )
    )
    for f in sorted(data_dir.glob("*.csv")):
        click.echo(f"7. sha256 {f.name}: {_sha256(f)}")


if __name__ == "__main__":
    main()
