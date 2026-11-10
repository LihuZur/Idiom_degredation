"""v2 generation CLI (THESIS_PLAN_v2 §3): sample items, build idiomatic + span-literal variants.

Default mode is `dry-run`: it renders the prompts and writes them for inspection without any API call.
`sync` and `batch` call the LLM and cost money; use a small `--n` first (step 5 pilot).
"""

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import click

from augmentation.cache import ResponseCache
from augmentation.idiom_generation import (
    SPANS_TEMPLATE_FILE,
    TEMPLATE_FILE,
    CompleteFn,
    Item,
    Suggester,
    cached_completer,
    filter_items,
    generate,
    load_inventory,
    load_pool,
    prepare,
    prompt_hash,
    render_span_prompt,
    span_prompt_hash,
    stratified_sample,
    sync_completer,
    write_outputs,
)
from augmentation.idiom_retrieval import IdiomRetriever, bge_embedder, parse_proposals
from augmentation.providers.base import build_client
from data.norms.idiom_text import IdiomIndex
from data.norms.inventory import DEFAULT_FINAL_PATH, DEFAULT_MAGPIE_EXT_PATH, load_midas

_INV = DEFAULT_FINAL_PATH.parent
_MIN_TOKENS_FOR_DEEP_THINKING = 4096


def _check_token_budget(max_output_tokens: int, *levels: str) -> None:
    """Thought tokens count against max_output_tokens: deep thinking needs a larger budget."""
    if max_output_tokens < _MIN_TOKENS_FOR_DEEP_THINKING and {"medium", "high"} & set(levels):
        raise click.UsageError(
            f"thinking medium/high needs --max-output-tokens >= {_MIN_TOKENS_FOR_DEEP_THINKING} "
            "(thought tokens count against the limit and cut the answer off)"
        )


def _filter_items(items: list[Item], where: Sequence[str]) -> list[Item]:
    """Pool items whose meta fields match every `field=value[,value]` condition."""
    try:
        return filter_items(items, where)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


def _write_proposals(
    out_dir: Path,
    items: Sequence[Item],
    complete_spans: CompleteFn,
    spans_template: str,
    usage: dict[str, Counter[str]],
) -> None:
    """Call 1 only: one line per item with its usable proposals (for screening tests)."""
    replies = complete_spans([render_span_prompt(i, spans_template) for i in items])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "span_proposals.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for item, raw in zip(items, replies, strict=True):
            props = parse_proposals(raw, item.x)
            f.write(
                json.dumps(
                    {"id": item.id, "n": len(props), "spans": [p.span for p in props]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    click.echo(f"wrote {path} ({len(items)} items); billed: {dict(usage['span'])}")


def _report_usage(out_dir: Path, usage: dict[str, Counter[str]]) -> None:
    """Print the billed tokens per stage and add them to generation.meta.json."""
    billed = {stage: dict(c) for stage, c in usage.items() if c}
    if not billed:
        return
    click.echo(f"billed this run (cached replies excluded): {billed}")
    meta_path = out_dir / "generation.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(
        json.dumps({**meta, "usage_this_run": billed}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--dataset", type=click.Choice(["sst2", "mnli", "mmlu", "agnews"]), required=True)
@click.option(
    "--n",
    type=int,
    default=0,
    show_default=True,
    help="Items to sample from the pool (stratified); 0 = the whole pool in its shuffled order.",
)
@click.option(
    "--where",
    multiple=True,
    help="Keep only pool items whose meta field matches, e.g. --where genre=telephone,travel "
    "(applied before sampling; repeat for several fields).",
)
@click.option(
    "--max-items",
    type=int,
    default=0,
    show_default=True,
    help="With --n 0: process only the first N items of the pool, in the same order and waves as the "
    "full run, so a later full run reuses every cached reply (0 = no limit).",
)
@click.option(
    "--target-kept",
    type=int,
    default=0,
    show_default=True,
    help="Stop after the wave in which this many items passed (0 = process everything).",
)
@click.option(
    "--pool",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Candidate pool CSV; default <pools>/<dataset>/original.csv (use datasets_out/wide/...).",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--mode", type=click.Choice(["dry-run", "sync", "batch"]), default="dry-run", show_default=True
)
@click.option("--provider", default="gemini", show_default=True)
@click.option("--model", default="gemini-3.6-flash", show_default=True)
@click.option("--k", type=int, default=12, show_default=True, help="Idioms offered per item.")
@click.option("--cap", type=int, default=15, show_default=True, help="Max uses per idiom.")
@click.option(
    "--wave",
    type=int,
    default=None,
    help="Items per wave (default 100 for sync, 1000 for batch; the cap is applied per wave).",
)
@click.option(
    "--max-attempts",
    type=int,
    default=1,
    show_default=True,
    help="Total tries per item; each retry uses a fresh shortlist without already-shown idioms. "
    "Retries raised the kept rate by ~6 points for ~1.75x the calls per item (pilot).",
)
@click.option(
    "--examples/--no-examples",
    default=False,
    show_default=True,
    help="Show a usage example per idiom in the pick prompt (about doubles its tokens).",
)
@click.option(
    "--retrieval/--no-retrieval",
    default=True,
    show_default=True,
    help="Shortlist idioms by meaning of model-proposed spans (needs an LLM call per item; "
    "dry-run always shows random shortlists).",
)
@click.option(
    "--template",
    default=TEMPLATE_FILE,
    show_default=True,
    help="Pick-prompt template in augmentation/prompts/ (part of the prompt hash).",
)
@click.option(
    "--spans-template",
    default=SPANS_TEMPLATE_FILE,
    show_default=True,
    help="Call-1 (span proposal) template in augmentation/prompts/ (part of its cache namespace).",
)
@click.option(
    "--skip-empty/--random-fallback",
    default=True,
    show_default=True,
    help="Items for which call 1 proposes no phrase get no call 2 (recorded as no_candidate).",
)
@click.option(
    "--spans-only",
    is_flag=True,
    help="Run only call 1 and write its proposals to <out>/span_proposals.jsonl (screen tests).",
)
@click.option(
    "--suggest/--no-suggest",
    default=False,
    show_default=True,
    help="Call 1 also names an idiom per span (spans prompt v3); suggestions found in MIDAS are "
    "added to call 2's shortlist (Kai 4a: the model's own choice, verified against MIDAS).",
)
@click.option(
    "--span-thinking",
    type=click.Choice(["minimal", "low", "medium", "high"]),
    default="minimal",
    show_default=True,
    help="Gemini thinking level for call 1 (span proposals); thought tokens are billed as output. "
    "MINIMAL (default since 2026-10-18) removed ~626 thought tokens per call with the same kept rows.",
)
@click.option(
    "--pick-thinking",
    type=click.Choice(["minimal", "low", "medium", "high"]),
    default="low",
    show_default=True,
    help="Gemini thinking level for call 2 (idiom pick).",
)
@click.option("--workers", type=int, default=8, show_default=True)
@click.option(
    "--cross-fraction",
    type=float,
    default=0.5,
    show_default=True,
    help="AG News: share of cross-domain items.",
)
@click.option("--temperature", type=float, default=0.5, show_default=True)
@click.option(
    "--pick-temperature",
    type=float,
    default=0.0,
    show_default=True,
    help="Temperature of call 2 only. 0 (the frozen setup used by the first waves, 2026-10-22) makes "
    "picks reproducible; it is part of every cached request, so keep it fixed within a run.",
)
@click.option("--max-output-tokens", type=int, default=1536, show_default=True)
@click.option(
    "--pools",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("datasets_out"),
)
@click.option(
    "--inventory",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_FINAL_PATH,
)
@click.option(
    "--extra-inventory",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    default=[DEFAULT_MAGPIE_EXT_PATH],
    show_default=True,
    help="Further inventory tier(s) with the same columns; default the unnormed MAGPIE tier.",
)
@click.option(
    "--normed-only",
    is_flag=True,
    help="Use only the normed inventory (ignore --extra-inventory).",
)
@click.option("--cache-dir", type=click.Path(path_type=Path), default=Path(".hf_cache/augment_v2"))
def main(
    dataset: str,
    n: int,
    where: tuple[str, ...],
    max_items: int,
    target_kept: int,
    pool: Path | None,
    seed: int,
    mode: str,
    provider: str,
    model: str,
    k: int,
    cap: int,
    wave: int | None,
    max_attempts: int,
    examples: bool,
    retrieval: bool,
    template: str,
    spans_template: str,
    skip_empty: bool,
    spans_only: bool,
    suggest: bool,
    span_thinking: str,
    pick_thinking: str,
    workers: int,
    cross_fraction: float,
    temperature: float,
    pick_temperature: float | None,
    max_output_tokens: int,
    pools: Path,
    inventory: Path,
    extra_inventory: tuple[Path, ...],
    normed_only: bool,
    cache_dir: Path,
) -> None:
    """Generate idiomatic and span-literal variants into datasets_out/{dataset}_v2/."""
    extra_inventory = () if normed_only else extra_inventory
    _check_token_budget(max_output_tokens, span_thinking, pick_thinking)
    items_all, pool_rows = load_pool(pool or pools / dataset / "original.csv", dataset)
    if n and max_items:
        raise click.UsageError("--max-items is for the full-run order (--n 0); --n draws a sample")
    items_all = _filter_items(items_all, where)
    items = stratified_sample(items_all, n, seed) if n else list(items_all)
    if max_items:
        items = items[:max_items]
    idioms, inv_rows = load_inventory(
        inventory,
        impli_csv=_INV / "impli_gloss_seed.csv",
        domains_csv=[_INV / "idiom_domains.csv", _INV / "idiom_domains_magpie.csv"],
        extra_csvs=extra_inventory,
    )
    out_dir = pools / f"{dataset}_v2"
    p_hash = prompt_hash(dataset, examples, template)
    wave = wave or (1000 if mode == "batch" else 100)
    model_tag = f"{provider}/{model}"
    if dataset == "agnews" and not any(i.domain for i in idioms):
        click.echo(
            "warning: no idiom_domains.csv; AG News shortlists are unconstrained (run idiom-tag-domains)"
        )

    if mode == "dry-run":
        prepared = prepare(
            items,
            idioms,
            k=k,
            cap=cap,
            seed=seed,
            usage=Counter(),
            cross_fraction=cross_fraction,
            examples=examples,
            template=template,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "dry_run_prompts.jsonl").open("w", encoding="utf-8") as f:
            for p in prepared:
                f.write(
                    json.dumps(
                        {
                            "id": p.item.id,
                            "condition": p.condition,
                            "shortlist": [i.idiom_id for i in p.shortlist],
                            "prompt": p.prompt,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        click.echo(
            f"dry run: {len(prepared)} prompts for {dataset}, prompt_hash={p_hash}; wrote {out_dir / 'dry_run_prompts.jsonl'}"
        )
        click.echo(
            f"mean prompt length: {sum(len(p.prompt) for p in prepared) / len(prepared):.0f} characters"
        )
        click.echo("--- first prompt ---\n" + prepared[0].prompt)
        return

    index = IdiomIndex(inv_rows, min_tokens=2, min_content=1)
    retriever = (
        IdiomRetriever(
            [i.idiom_id for i in idioms],
            [f"{i.name}: {' / '.join(i.definitions)}" for i in idioms],
            bge_embedder(),
            cache_dir / "embeddings",
        )
        if retrieval
        else None
    )
    cache = ResponseCache(cache_dir)
    usage: dict[str, Counter[str]] = {"span": Counter(), "pick": Counter()}

    def namespace(base: str, level: str) -> str:
        # LOW was the only level before 2026-10-18, so its cache entries keep the bare namespace.
        return base if level == "low" else f"{base}:thinking={level}"

    def build_complete(stage: str, base: str, level: str, temp: float) -> CompleteFn:
        ns = namespace(base, level)
        if mode == "sync":
            client = build_client(provider, model, thinking_level=level)
            client_usage: object = getattr(client, "usage", None)
            if isinstance(client_usage, Counter):
                usage[stage] = cast("Counter[str]", client_usage)
            return sync_completer(
                client,
                cache,
                ns,
                temperature=temp,
                max_output_tokens=max_output_tokens,
                workers=workers,
            )
        from augmentation.providers.gemini_batch import run_batch  # noqa: PLC0415

        def batch_run(prompts: Sequence[str]) -> list[str | None]:
            return run_batch(
                prompts,
                model=model,
                temperature=temp,
                max_output_tokens=max_output_tokens,
                state_dir=cache_dir / "batch_jobs",
                thinking_level=level,
                usage=usage[stage],
            )

        return cached_completer(
            batch_run,
            cache,
            ns,
            model_tag,
            temperature=temp,
            max_output_tokens=max_output_tokens,
        )

    spans_template = "idiom_spans_v3.txt" if suggest else spans_template
    pick_t = temperature if pick_temperature is None else pick_temperature
    complete, complete_spans = (
        build_complete("pick", p_hash, pick_thinking, pick_t),
        build_complete(
            "span", span_prompt_hash(dataset, spans_template), span_thinking, temperature
        ),
    )
    suggester = Suggester(load_midas(), idioms) if suggest else None
    if spans_only:
        return _write_proposals(out_dir, items, complete_spans, spans_template, usage)

    outcomes = generate(
        items,
        idioms,
        index,
        complete,
        model=model_tag,
        p_hash=p_hash,
        k=k,
        cap=cap,
        seed=seed,
        wave=wave,
        cross_fraction=cross_fraction,
        max_attempts=max_attempts,
        retriever=retriever,
        target_kept=target_kept,
        complete_spans=complete_spans,
        examples=examples,
        template=template,
        suggester=suggester,
        spans_template=spans_template,
        skip_empty=skip_empty,
    )
    counts = write_outputs(
        out_dir,
        outcomes,
        pool_rows,
        model=model_tag,
        p_hash=p_hash,
        meta={
            "dataset": dataset,
            "n_requested": len(items),
            "max_items": max_items,
            "where": list(where),
            "n_processed": len(outcomes),
            "target_kept": target_kept,
            "pool": str(pool or pools / dataset / "original.csv"),
            "seed": seed,
            "k": k,
            "cap": cap,
            "max_attempts": max_attempts,
            "examples": examples,
            "wave": wave,
            "retrieval": retrieval,
            "template": template,
            "span_thinking": span_thinking,
            "temperature": temperature,
            "pick_temperature": pick_t,
            "suggest": suggest,
            "spans_template": spans_template,
            "skip_empty": skip_empty,
            "pick_thinking": pick_thinking,
            "inventory": [str(inventory), *map(str, extra_inventory)],
            "n_idioms": len(idioms),
            "mode": mode,
        },
    )
    kept = sum(1 for o in outcomes if o.ok)
    click.echo(f"{dataset}: kept {kept} of {len(outcomes)} items; reasons: {counts}")
    _report_usage(out_dir, usage)
    click.echo(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
