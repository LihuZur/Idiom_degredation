"""Step-3 extras built on the final inventory: natural-idiom candidates, the IMPLI gloss seed and the
MAGPIE-attested extension tier."""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import click

from augmentation.providers.base import build_client
from data.norms import extension_tags
from data.norms.domains import IdiomLine, batches
from data.norms.idiom_text import IdiomIndex, gloss_for, read_pairs
from data.norms.inventory import (
    DEFAULT_FINAL_PATH,
    DEFAULT_MAGPIE_CANDIDATES_PATH,
    DEFAULT_MAGPIE_EXT_PATH,
    load_midas,
    write_magpie_extension,
)

_INV_DIR = DEFAULT_FINAL_PATH.parent
_IMPLI_ENTAILMENT = (
    "manual_e",
    "fig_context_magpie_e",
    "fig_context_pie_e",
    "fig_context_semeval_e",
)


def _index(final: Path) -> IdiomIndex:
    with final.open(encoding="utf-8", newline="") as f:
        return IdiomIndex(list(csv.DictReader(f)))


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Natural-idiom candidates (A7 input), IMPLI gloss seed list, MAGPIE extension tier."""


@main.command()
@click.option("--final", type=click.Path(exists=True, path_type=Path), default=DEFAULT_FINAL_PATH)
@click.option(
    "--scope",
    type=click.Choice(["midas", "inventory"]),
    default="midas",
    show_default=True,
    help="midas: any MIDAS idiom (no norms needed for this check); inventory: final inventory only",
)
@click.option("--datasets", default="sst2,mnli,mmlu", show_default=True)
@click.option(
    "--pools",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("datasets_out"),
)
@click.option(
    "--out", type=click.Path(path_type=Path), default=_INV_DIR / "natural_idiom_candidates.csv"
)
def natural(final: Path, scope: str, datasets: str, pools: Path, out: Path) -> None:
    """Rows of the cleaned pools whose input text already contains an inventory idiom."""
    with final.open(encoding="utf-8", newline="") as f:
        inventory_ids = {i for r in csv.DictReader(f) for i in r["midas_ids"].split("|")}
    if scope == "inventory":
        index = _index(final)
    else:
        index = IdiomIndex(
            [
                {
                    "norm_idiom": m.forms[0],
                    "midas_ids": m.midas_id,
                    "definitions": json.dumps(list(m.meanings)),
                    "midas_forms": json.dumps(list(m.forms)),
                }
                for m in load_midas()
            ]
        )
    rows: list[dict[str, str]] = []
    totals: Counter[str] = Counter()
    for ds in (d.strip() for d in datasets.split(",") if d.strip()):
        with (pools / ds / "original.csv").open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                totals[ds] += 1
                for h in index.find(r["x"]):
                    rows.append(
                        {
                            "dataset": ds,
                            "row_id": r["id"],
                            "text": r["x"],
                            "idiom": h.ref.idiom,
                            "midas_ids": h.ref.midas_ids,
                            "in_inventory": str(
                                set(h.ref.midas_ids.split("|")) <= inventory_ids
                            ).lower(),
                            "matched_form": h.form,
                            "definition": h.ref.definition,
                            "human_check": "",  # A7: confirm figurative use and that a gloss fits the slot
                        }
                    )
    _write(out, rows)
    per = Counter(r["dataset"] for r in rows)
    rows_hit = {ds: len({r["row_id"] for r in rows if r["dataset"] == ds}) for ds in totals}
    for ds in totals:
        click.echo(
            f"{ds}: {rows_hit[ds]} of {totals[ds]} rows contain a {scope} idiom ({per[ds]} idiom hits)"
        )
    click.echo(f"distinct idioms found: {len({r['idiom'] for r in rows})}; wrote {out}")


@main.command()
@click.option("--final", type=click.Path(exists=True, path_type=Path), default=DEFAULT_FINAL_PATH)
@click.option(
    "--impli",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("/tmp/impli_data"),
)
@click.option("--out", type=click.Path(path_type=Path), default=_INV_DIR / "impli_gloss_seed.csv")
def impli(final: Path, impli: Path, out: Path) -> None:
    """Literal glosses IMPLI used for inventory idioms (entailment pairs, idiom side only)."""
    index = _index(final)
    glosses: dict[tuple[str, str], dict[str, str]] = {}
    counts: Counter[tuple[str, str]] = Counter()
    sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    for name in _IMPLI_ENTAILMENT:
        for idiom_s, literal_s in read_pairs(impli / f"{name}.tsv"):
            for h in index.find(idiom_s):
                gloss = gloss_for(idiom_s, literal_s, h)
                if not gloss:
                    continue
                key = (h.ref.idiom, gloss)
                counts[key] += 1
                sources[key].add(name)
                glosses.setdefault(
                    key,
                    {
                        "idiom": h.ref.idiom,
                        "midas_ids": h.ref.midas_ids,
                        "midas_definition": h.ref.definition,
                        "impli_gloss": gloss,
                        "example_idiom_sentence": idiom_s,
                        "example_literal_sentence": literal_s,
                    },
                )
    rows = [
        {**g, "n_pairs": str(counts[k]), "impli_files": json.dumps(sorted(sources[k]))}
        for k, g in sorted(glosses.items(), key=lambda kv: (kv[0][0], -counts[kv[0]]))
    ]
    _write(out, rows)
    click.echo(
        f"{len({r['idiom'] for r in rows})} inventory idioms with IMPLI glosses ({len(rows)} distinct glosses); wrote {out}"
    )


@main.command()
@click.option("--final", type=click.Path(exists=True, path_type=Path), default=DEFAULT_FINAL_PATH)
@click.option("--out", type=click.Path(path_type=Path), default=DEFAULT_MAGPIE_CANDIDATES_PATH)
@click.option(
    "--min-uses",
    type=int,
    default=5,
    show_default=True,
    help="Minimum idiomatic uses of the idiom in MAGPIE's BNC sample.",
)
@click.option("--min-confidence", type=float, default=0.75, show_default=True)
def magpie(final: Path, out: Path, min_uses: int, min_confidence: float) -> None:
    """Candidates for the unnormed tier: MIDAS idioms MAGPIE attests as common (then run magpie-tag)."""
    rows, counts = write_magpie_extension(
        final, out, min_uses=min_uses, min_confidence=min_confidence
    )
    click.echo(f"{len(rows)} candidates (min uses {min_uses}); counts: {dict(counts)}; wrote {out}")


@main.command("magpie-tag")
@click.option(
    "--candidates",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_MAGPIE_CANDIDATES_PATH,
)
@click.option("--out", type=click.Path(path_type=Path), default=DEFAULT_MAGPIE_EXT_PATH)
@click.option("--provider", default="gemini", show_default=True)
@click.option("--model", default="gemini-3.6-flash", show_default=True)
@click.option("--run", is_flag=True, help="Actually call the LLM (about 16 calls, a few cents).")
def magpie_tag(candidates: Path, out: Path, provider: str, model: str, run: bool) -> None:
    """Keep only figurative candidates (drop fixed phrases and obscure ones) and tag their domain.

    Writes the tier (`--out`), `idiom_domains_magpie.csv`, the rejected list and a 50-row spot-check.
    """
    with candidates.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    lines = [
        IdiomLine(r["midas_ids"], r["norm_idiom"], r["first_definition"].rstrip(".")) for r in rows
    ]
    click.echo(f"{len(lines)} candidates in {len(batches(lines))} calls")
    if not run:
        click.echo("dry run: pass --run to call the LLM")
        return
    client = build_client(provider, model)
    tags = extension_tags.tag_all(
        lines,
        lambda prompt: client.complete(
            system="", user=prompt, temperature=0.0, max_output_tokens=8192
        ),
    )
    tagged = [{**r, "kind": tags[r["midas_ids"]][0]} for r in rows if r["midas_ids"] in tags]
    # The tier itself carries no kind column: every row in it is figurative.
    kept = [
        {k: v for k, v in r.items() if k != "kind"} for r in tagged if r["kind"] == "figurative"
    ]
    _write(out, kept)
    _write(
        out.with_name(f"{out.stem}_rejected.csv"), [r for r in tagged if r["kind"] != "figurative"]
    )
    with out.with_name("idiom_domains_magpie.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idiom_id", "idiom", "domain"])
        for r in kept:
            w.writerow([r["midas_ids"], r["norm_idiom"], tags[r["midas_ids"]][1]])
    with out.with_name(f"{out.stem}_check.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idiom_id", "idiom", "definition", "kind", "domain", "author_ok"])
        for r in random.Random(0).sample(tagged, min(50, len(tagged))):
            kind, domain = tags[r["midas_ids"]]
            w.writerow([r["midas_ids"], r["norm_idiom"], r["first_definition"], kind, domain, ""])
    click.echo(
        f"tagged {len(tagged)} of {len(rows)}; kinds: {dict(Counter(r['kind'] for r in tagged))}; "
        f"kept {len(kept)} figurative; wrote {out}"
    )


if __name__ == "__main__":
    main()
