"""v3 generation CLI: insert one bank idiom, then build the literal version (THESIS_PLAN_v3 §2, §7).

Dry run (free, no API call): renders the real prompts with their shortlists and estimates tokens.
Sync / batch modes call Gemini (paid; the author runs them). Output goes to --out-dir/<dataset>/:
original.csv, idiomatic.csv, span_literal.csv (the literal version; the variant keeps its v2 name so the
evaluation pipeline reads it unchanged), generation_log.csv, generation.meta.json.
"""

import csv
import json
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import click

from augmentation.cache import ResponseCache
from augmentation.idiom_generation import (
    Idiom,
    Item,
    cached_completer,
    load_inventory,
    load_pool,
    stratified_sample,
    sync_completer,
    write_outputs,
)
from augmentation.idiom_insertion import (
    EXAMPLE_IDS,
    N_FIT,
    N_RANDOM,
    TEMPLATE_FILE,
    bank_in_order,
    bank_order_index,
    generate_insert,
    insert_prompt_hash,
    item_rng,
    mixed_shortlist,
    render_insert_prompt,
    whole_bank,
)
from augmentation.idiom_retrieval import IdiomRetriever, bge_embedder
from augmentation.providers.base import build_client
from data.norms.idiom_text import IdiomIndex
from data.norms.inventory import DEFAULT_FINAL_PATH, DEFAULT_MAGPIE_EXT_PATH

_INV = DEFAULT_FINAL_PATH.parent
_POOLS = {
    "sst2": "sst2",
    "mnli": "mnli_wide",
    "mmlu": "mmlu_wide",
    "agnews": "agnews_wide",
    "siqa": "siqa",
}
_CHARS_PER_TOKEN = 4


def _retriever(idioms: Sequence[Idiom], cache_dir: Path) -> IdiomRetriever:
    return IdiomRetriever(
        [i.idiom_id for i in idioms],
        [f"{i.name}: {' / '.join(i.definitions)}" for i in idioms],
        bge_embedder(),
        cache_dir / "embeddings",
    )


def load_bank(cache_dir: Path) -> tuple[list[Idiom], IdiomIndex, IdiomRetriever]:
    """The idiom bank exactly as `main` loads it (inventory, string index, BGE retriever)."""
    idioms, inv_rows = load_inventory(
        DEFAULT_FINAL_PATH,
        impli_csv=_INV / "impli_gloss_seed.csv",
        domains_csv=[_INV / "idiom_domains.csv", _INV / "idiom_domains_magpie.csv"],
        extra_csvs=[DEFAULT_MAGPIE_EXT_PATH],
    )
    return idioms, IdiomIndex(inv_rows, min_tokens=2, min_content=1), _retriever(idioms, cache_dir)


def _items_from_logs(items_all: Sequence[Item], folders: Sequence[Path]) -> list[Item]:
    """Exactly the items an earlier run attempted (ids from its generation_log.csv, in order)."""
    by_id = {i.id: i for i in items_all}
    wanted: list[str] = []
    for folder in folders:
        with (folder / "generation_log.csv").open(encoding="utf-8", newline="") as f:
            wanted += [row["id"] for row in csv.DictReader(f) if row["id"] not in wanted]
    missing = [w for w in wanted if w not in by_id]
    if missing:
        raise click.ClickException(f"{len(missing)} ids not in the pool, e.g. {missing[:3]}")
    return [by_id[w] for w in wanted]


def _subject_set(subjects: str) -> set[str]:
    return {x.strip() for x in subjects.split(",") if x.strip()}


def _filter_candidates(
    items_all: Sequence[Item], used: set[str], min_words: int, subjects: set[str]
) -> list[Item]:
    """Items not used before, at least `min_words` long, and (MMLU) from the given subjects."""
    return [
        i
        for i in items_all
        if i.id not in used
        and len(i.x.split()) >= min_words
        and (not subjects or str(i.meta.get("subject", "")) in subjects)
    ]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--dataset", type=click.Choice(sorted(_POOLS)), required=True)
@click.option(
    "--n",
    type=int,
    default=20,
    show_default=True,
    help="Sentences to sample (stratified by label x length).",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--mode", type=click.Choice(["dry-run", "sync", "batch"]), default="dry-run", show_default=True
)
@click.option("--provider", default="gemini", show_default=True)
@click.option("--model", default="gemini-3.6-flash", show_default=True)
@click.option("--thinking", default="low", show_default=True, help="Gemini thinking level.")
@click.option("--temperature", type=float, default=0.0, show_default=True)
@click.option("--max-output-tokens", type=int, default=2048, show_default=True)
@click.option(
    "--template",
    type=click.Choice(
        [
            "idiom_insert_v1.txt",
            "idiom_insert_v2.txt",
            "idiom_insert_v3.txt",
            "idiom_insert_v4.txt",
            "idiom_insert_v5.txt",
            "idiom_insert_v6.txt",
        ]
    ),
    default=TEMPLATE_FILE,
    show_default=True,
    help="Prompt version (v1 = E1 2026-10-30; another version means new cache keys, i.e. a paid run).",
)
@click.option(
    "--exclude-ids-from",
    "exclude_from",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Earlier output folder(s) of this dataset; their item ids are not sampled again (repeatable).",
)
@click.option(
    "--ids-from",
    "ids_from",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Earlier output folder(s) of this dataset: re-run exactly their attempted items (all ids in their "
    "generation_log.csv, in order) instead of sampling; --n is ignored (POC-2, plan step 8d).",
)
@click.option(
    "--shortlist",
    type=click.Choice(["mixed", "all"]),
    default="mixed",
    show_default=True,
    help="mixed: 20 fit + 10 random idioms; all: the whole bank, shuffled per item (plan O41, POC-2).",
)
@click.option(
    "--min-words",
    type=int,
    default=0,
    show_default=True,
    help="Sample only texts with at least this many words (e.g. long MMLU scenario questions).",
)
@click.option(
    "--subjects",
    default="",
    help="MMLU only: comma-separated subjects to sample from (e.g. professional_law,professional_medicine).",
)
@click.option(
    "--n-fit", type=int, default=N_FIT, show_default=True, help="Shortlist: idioms by fit."
)
@click.option(
    "--n-random", type=int, default=N_RANDOM, show_default=True, help="Shortlist: random idioms."
)
@click.option(
    "--max-items",
    type=int,
    default=None,
    help="With --ids-from: use only the first N of those items (a small paid test before the full run).",
)
@click.option(
    "--bank-order",
    type=click.Choice(["fixed", "shuffled", "orders"]),
    default="fixed",
    show_default=True,
    help="With --shortlist all: fixed = one alphabetical order (cacheable, but position bias in POC-2); orders = 8 fixed random orders, one per item (bias spread, each order cacheable); shuffled = per item (no caching).",
)
@click.option(
    "--shortlist-seed",
    type=click.Choice(["item", "run"]),
    default="item",
    show_default=True,
    help="item: each item's shortlist depends only on its id (same prompt in any sample, cache reuse); "
    "run: one sequence for the whole run (how E1/E2 were made).",
)
@click.option("--cap", type=int, default=15, show_default=True, help="Max kept rows per idiom.")
@click.option("--workers", type=int, default=8, show_default=True)
@click.option(
    "--pools", type=click.Path(path_type=Path), default=Path("datasets_out/wide"), show_default=True
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("datasets_out/_exp_v3_e1"),
    show_default=True,
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=Path(".hf_cache/augment_v3"),
    show_default=True,
)
def main(
    dataset: str,
    n: int,
    seed: int,
    mode: str,
    provider: str,
    model: str,
    thinking: str,
    temperature: float,
    max_output_tokens: int,
    template: str,
    exclude_from: tuple[Path, ...],
    ids_from: tuple[Path, ...],
    shortlist: str,
    bank_order: str,
    max_items: int | None,
    min_words: int,
    subjects: str,
    n_fit: int,
    n_random: int,
    shortlist_seed: str,
    cap: int,
    workers: int,
    pools: Path,
    out_dir: Path,
    cache_dir: Path,
) -> None:
    """Insert an idiom and build its literal version for a stratified sample of one dataset."""
    pool_csv = pools / _POOLS[dataset] / "original.csv"
    items_all, pool_rows = load_pool(pool_csv, dataset)
    used = set(EXAMPLE_IDS)
    for folder in exclude_from:
        with (folder / "generation_log.csv").open(encoding="utf-8", newline="") as f:
            used |= {row["id"] for row in csv.DictReader(f)}
    candidates = _filter_candidates(items_all, used, min_words, _subject_set(subjects))
    if ids_from:
        items = _items_from_logs(items_all, ids_from)[:max_items]
        n = len(items)
    else:
        items = stratified_sample(candidates, n, seed)
    if len(items) < n:  # per-stratum rounding can fall short; top up deterministically
        picked = {i.id for i in items}
        rest = [i for i in candidates if i.id not in picked]
        random.Random(seed).shuffle(rest)
        items += rest[: n - len(items)]
    idioms, inv_rows = load_inventory(
        DEFAULT_FINAL_PATH,
        impli_csv=_INV / "impli_gloss_seed.csv",
        domains_csv=[_INV / "idiom_domains.csv", _INV / "idiom_domains_magpie.csv"],
        extra_csvs=[DEFAULT_MAGPIE_EXT_PATH],
    )
    index = IdiomIndex(inv_rows, min_tokens=2, min_content=1)
    retriever = _retriever(idioms, cache_dir)
    p_hash = insert_prompt_hash(dataset, template)
    target = out_dir / dataset
    model_tag = f"{provider}/{model}"
    ns = p_hash if thinking == "low" else f"{p_hash}:thinking={thinking}"

    if mode == "dry-run":
        rng = random.Random(seed)
        prompts = [
            render_insert_prompt(
                i,
                (
                    bank_in_order(idioms, bank_order_index(i.id, 1))
                    if bank_order == "orders"
                    else whole_bank(
                        idioms,
                        None
                        if bank_order == "fixed"
                        else (item_rng(seed, i.id, 1) if shortlist_seed == "item" else rng),
                    )
                )
                if shortlist == "all"
                else mixed_shortlist(
                    i,
                    idioms,
                    retriever,
                    item_rng(seed, i.id, 1) if shortlist_seed == "item" else rng,
                    n_fit=n_fit,
                    n_random=n_random,
                ),
                template,
            )
            for i in items
        ]
        target.mkdir(parents=True, exist_ok=True)
        with (target / "dry_run_prompts.jsonl").open("w", encoding="utf-8") as f:
            for item, prompt in zip(items, prompts, strict=True):
                f.write(json.dumps({"id": item.id, "prompt": prompt}, ensure_ascii=False) + "\n")
        mean_chars = sum(map(len, prompts)) / len(prompts)
        click.echo(
            f"dry run: {len(prompts)} prompts for {dataset} (template {template}, "
            f"{'whole bank' if shortlist == 'all' else f'{n_fit} fit + {n_random} random idioms'}), "
            f"prompt_hash={p_hash}; mean {mean_chars:.0f} characters "
            f"≈ {mean_chars / _CHARS_PER_TOKEN:.0f} input tokens (estimate); wrote {target / 'dry_run_prompts.jsonl'}"
        )
        click.echo("--- first prompt ---\n" + prompts[0])
        return

    cache = ResponseCache(cache_dir)
    usage: Counter[str] = Counter()
    if mode == "sync":
        client = build_client(provider, model, thinking_level=thinking)
        client_usage = getattr(client, "usage", None)
        if isinstance(client_usage, Counter):
            usage = client_usage
        complete = sync_completer(
            client,
            cache,
            ns,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            workers=workers,
        )
    else:
        from augmentation.providers.gemini_batch import run_batch  # noqa: PLC0415

        def batch_run(prompts: Sequence[str]) -> list[str | None]:
            return run_batch(
                prompts,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                state_dir=cache_dir / "batch_jobs",
                thinking_level=thinking,
                usage=usage,
            )

        complete = cached_completer(
            batch_run,
            cache,
            ns,
            model_tag,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    outcomes = generate_insert(
        items,
        idioms,
        index,
        complete,
        retriever,
        model=model_tag,
        p_hash=p_hash,
        cap=cap,
        seed=seed,
        template=template,
        per_item_seed=shortlist_seed == "item",
        shortlist=shortlist,
        bank_order=bank_order,
        n_fit=n_fit,
        n_random=n_random,
    )
    counts = write_outputs(
        target,
        outcomes,
        pool_rows,
        model=model_tag,
        p_hash=p_hash,
        meta={
            "design": "v3 insertion (THESIS_PLAN_v3 §2)",
            "dataset": dataset,
            "pool": str(pool_csv),
            "n_requested": n,
            "n_processed": len(outcomes),
            "seed": seed,
            "template": template,
            "shortlist_seed": shortlist_seed,
            "excluded_ids_from": [str(p) for p in exclude_from],
            "ids_from": [str(p) for p in ids_from],
            "min_words": min_words,
            "subjects": sorted(_subject_set(subjects)),
            "shortlist": f"whole bank ({bank_order} order)"
            if shortlist == "all"
            else f"{n_fit} fit + {n_random} random",
            "cap": cap,
            "thinking": thinking,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "mode": mode,
            "n_idioms": len(idioms),
            "usage_this_run": dict(usage),
        },
    )
    kept = sum(o.ok for o in outcomes)
    click.echo(
        f"{dataset}: kept {kept} of {len(outcomes)}; reasons: {counts}; billed: {dict(usage)}"
    )
    click.echo(f"wrote {target}")


if __name__ == "__main__":
    main()
