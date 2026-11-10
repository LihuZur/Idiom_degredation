"""Closed idiom inventory for the v2 generator: MIDAS-EN joined to human norms.

See paper_preparations/THESIS_PLAN_v2.md §3.1. MIDAS-EN (Kim et al., EMNLP 2025;
HF `HYU-NLP/MIDAS`, CC BY-SA 4.0) supplies a human-verified definition and form
variants per idiom; Bulkes & Tanner (2017) and Libben & Titone (2008) supply human
familiarity / opacity norms. An idiom is in the inventory only if a norm row matches
a MIDAS entry; every match carries a status so the author can review the uncertain
ones (annotation A1).
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import openpyxl

from data.norms.frequency import idiom_frequency, load_word_frequencies

_FILES = Path(__file__).parent / "_files"
DEFAULT_MIDAS_PATH = _FILES / "midas" / "EN_Idioms.json"
DEFAULT_BT_PATH = _FILES / "opacity" / "bulkes_tanner_2017.xlsx"
DEFAULT_LT_PATH = _FILES / "opacity" / "libben_titone_2008" / "LibbenTitone-Norms.txt"
DEFAULT_OUT_PATH = _FILES / "inventory" / "idiom_inventory.csv"
DEFAULT_FINAL_PATH = _FILES / "inventory" / "idiom_inventory_final.csv"

MatchStatus = Literal["exact", "key", "near", "ambiguous"]

# Slot words that differ between a norm's citation form and MIDAS's.
_DROP = frozenset(
    re.findall(
        r"\w+",
        "the a an to of be is are was were been i you he she it we they me him her us them "
        "my your his its our their someone something somebody anyone one ones oneself "
        "yourself himself herself itself themselves",
    )
)
_NEAR_JACCARD = 0.7
_NEAR_MIN_SHARED = 2
_MIN_LEN = 3

_IRREGULAR = dict(
    re.findall(
        r"(\w+):(\w+)",
        "took:take taken:take got:get gotten:get had:have went:go gone:go said:say saw:see "
        "seen:see came:come ran:run gave:give given:give made:make bit:bite bitten:bite "
        "broke:break broken:break threw:throw thrown:throw drew:draw drawn:draw fell:fall "
        "fallen:fall knew:know known:know sat:sit stood:stand held:hold kept:keep left:leave "
        "lost:lose paid:pay sold:sell told:tell thought:think brought:bring caught:catch "
        "bought:buy ate:eat eaten:eat spoke:speak spoken:speak wore:wear worn:wear won:win "
        "hung:hang hid:hide shot:shoot shook:shake rode:ride rang:ring sang:sing flew:fly "
        "grew:grow led:lead met:meet sent:send spent:spend struck:strike swept:sweep "
        "taught:teach tore:tear woke:wake wrote:write written:write dug:dig fed:feed "
        "felt:feel fought:fight found:find forgot:forget froze:freeze heard:hear laid:lay "
        "meant:mean rose:rise sank:sink slept:sleep stole:steal stuck:stick swore:swear "
        "sworn:swear",
    )
)

# (suffix, replacement, word must be longer than this); the first matching rule wins.
_SUFFIX_RULES = (
    ("ies", "y", 4),
    ("ied", "y", 3),
    ("ing", "", 5),
    ("ed", "", 4),
    ("ss", "ss", 0),
    ("ses", "s", 3),
    ("xes", "x", 3),
    ("zes", "z", 3),
    ("s", "", 3),
)


@dataclass(frozen=True, slots=True)
class MidasIdiom:
    """One MIDAS idiom (all senses sharing a base ID)."""

    midas_id: str
    forms: tuple[str, ...]
    meanings: tuple[str, ...]
    sentences: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NormRow:
    """One human-normed idiom; norms are None where the source lacks them."""

    idiom: str
    source: Literal["bulkes_tanner_2017", "libben_titone_2008"]
    familiarity: float | None
    meaningfulness: float | None
    opacity: float  # 0-1, higher = more opaque (1 - proportion rated decomposable)
    literal_plausibility: float | None


@dataclass(frozen=True, slots=True)
class Match:
    norm: NormRow
    midas_ids: tuple[str, ...]
    status: MatchStatus
    note: str = ""


def _stem(word: str) -> str:
    """Crude suffix stripper: enough to line up "buried"/"bury", "takes"/"take"."""
    w = _IRREGULAR.get(word, word)
    for suffix, repl, min_len in _SUFFIX_RULES:
        if w.endswith(suffix) and len(w) > min_len:
            w = w[: -len(suffix)] + repl
            break
    if len(w) > _MIN_LEN and w[-1] == w[-2] and w[-1] not in "sl":
        w = w[:-1]
    if w.endswith("e") and len(w) >= _MIN_LEN:
        w = w[:-1]
    if w.endswith("y") and len(w) >= _MIN_LEN:
        w = w[:-1] + "i"
    return w


stem_word = _stem

_LIGHT = frozenset(_stem(w) for w in re.findall(r"\w+", "do have get go take make put give"))


def surface_form(text: str) -> str:
    """Lowercased, punctuation-light surface form for exact comparison."""
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def content_key(text: str) -> frozenset[str]:
    """Stemmed content-word set; possessives, pronouns, articles and 'be' are dropped."""
    t = re.sub(
        r"\b\w+'s\b", lambda m: "" if m.group(0)[:-2] in _DROP else m.group(0)[:-2], text.lower()
    )
    return frozenset(_stem(w) for w in re.findall(r"[a-z0-9]+", t) if w not in _DROP)


def load_midas(path: Path = DEFAULT_MIDAS_PATH) -> list[MidasIdiom]:
    """Group MIDAS senses (IDs like '7-1', '7-2') into one record per idiom."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    forms: dict[str, list[str]] = defaultdict(list)
    meanings: dict[str, list[str]] = defaultdict(list)
    sentences: dict[str, list[str]] = defaultdict(list)
    for rec in raw:
        base = str(rec["ID"]).split("-")[0]
        for f in rec["Idiom"]:
            if f not in forms[base]:
                forms[base].append(f)
        if rec["Meaning"] not in meanings[base]:
            meanings[base].append(rec["Meaning"])
        for s in rec["Sentence"]:
            if s not in sentences[base]:
                sentences[base].append(s)
    return [MidasIdiom(i, tuple(forms[i]), tuple(meanings[i]), tuple(sentences[i])) for i in forms]


def _num(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _bt_sheet(wb: openpyxl.Workbook, name: str, col: int) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for row in wb[name].iter_rows(min_row=2, values_only=True):
        if row[0] is not None:
            out[str(row[0]).strip()] = _num(row[col])
    return out


def load_norm_rows(
    bt_path: Path = DEFAULT_BT_PATH, lt_path: Path = DEFAULT_LT_PATH
) -> list[NormRow]:
    """Bulkes & Tanner rows first; Libben & Titone rows only for idioms B&T lacks."""
    wb = openpyxl.load_workbook(bt_path, read_only=True)
    fam = _bt_sheet(wb, "FAMILIARITY", 2)
    mean = _bt_sheet(wb, "MEANINGFULNESS", 2)
    decomp = _bt_sheet(wb, "GLOBAL DECOMPOSABILITY", 2)
    lit = _bt_sheet(wb, "LITERAL PLAUSIBILITY", 2)
    rows: list[NormRow] = []
    seen: set[str] = set()
    for idiom, d in decomp.items():
        if d is None:
            continue
        seen.add(surface_form(idiom))
        rows.append(
            NormRow(
                idiom,
                "bulkes_tanner_2017",
                fam.get(idiom),
                mean.get(idiom),
                1.0 - d,
                lit.get(idiom),
            )
        )
    with lt_path.open(encoding="utf-8") as f:
        for rec in csv.DictReader(f, delimiter="\t"):
            idiom = rec["Idiom"].strip()
            if surface_form(idiom) in seen:
                continue

            def col(prefix: str, rec: dict[str, str] = rec) -> float | None:
                key = next((k for k in rec if k.startswith(prefix)), None)
                return float(rec[key]) if key and rec[key] else None

            d = col("Global Decomposability Proportion")
            if d is None:
                continue
            rows.append(
                NormRow(
                    idiom,
                    "libben_titone_2008",
                    col("Familiarity (1"),
                    col("Meaningfulness (1"),
                    1.0 - d,
                    col("Literal Plausibility (1"),
                )
            )
    return rows


def _similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard of two content keys; 0.9 when one only adds light verbs to the other."""
    shared = len(a & b)
    if shared < _NEAR_MIN_SHARED:
        return 0.0
    if (a <= b and b - a <= _LIGHT) or (b <= a and a - b <= _LIGHT):
        return 0.9
    return shared / len(a | b)


def match_norms(norms: list[NormRow], midas: list[MidasIdiom]) -> list[Match]:
    """Match each norm row to MIDAS idioms: exact form, stemmed key, then near (Jaccard)."""
    by_form: dict[str, set[str]] = defaultdict(set)
    by_key: dict[frozenset[str], set[str]] = defaultdict(set)
    keys_by_id: dict[str, list[frozenset[str]]] = defaultdict(list)
    for m in midas:
        for f in m.forms:
            by_form[surface_form(f)].add(m.midas_id)
            k = content_key(f)
            if k:
                by_key[k].add(m.midas_id)
                keys_by_id[m.midas_id].append(k)
    matches: list[Match] = []
    for n in norms:
        ids = by_form.get(surface_form(n.idiom))
        status: MatchStatus = "exact"
        note = ""
        if not ids:
            k = content_key(n.idiom)
            ids = by_key.get(k) if k else None
            status = "key"
        if not ids and len(content_key(n.idiom)) >= _NEAR_MIN_SHARED:
            k = content_key(n.idiom)
            scored: dict[str, float] = {}
            for mid, mkeys in keys_by_id.items():
                best = max(
                    (_similarity(k, mk) for mk in mkeys),
                    default=0.0,
                )
                if best >= _NEAR_JACCARD:
                    scored[mid] = best
            if scored:
                top = max(scored.values())
                ids = {mid for mid, s in scored.items() if s == top}
                status = "near"
                note = f"jaccard={top:.2f}"
        if not ids:
            continue
        if len(ids) > 1:
            status, note = "ambiguous", f"{len(ids)} candidate MIDAS ids"
        matches.append(Match(n, tuple(sorted(ids, key=int)), status, note))
    return matches


def build_rows(
    matches: list[Match], midas: list[MidasIdiom], word_freq: dict[str, int]
) -> list[dict[str, str]]:
    """One CSV row per matched norm idiom; list-valued fields are JSON-encoded."""
    by_id = {m.midas_id: m for m in midas}
    rows: list[dict[str, str]] = []
    for mt in matches:
        recs = [by_id[i] for i in mt.midas_ids]
        forms = [f for r in recs for f in r.forms]
        meanings = [x for r in recs for x in r.meanings]
        sentences = [s for r in recs for s in r.sentences]
        freq = idiom_frequency(mt.norm.idiom, word_freq)
        n = mt.norm
        rows.append(
            {
                "match_status": mt.status,
                # A1: author fills accept / reject / fix; exact matches are pre-accepted.
                "match_checked": "accept (auto)" if mt.status == "exact" else "",
                "norm_idiom": n.idiom,
                "candidates": " || ".join(
                    f"{r.midas_id}: {r.forms[0]} - {r.meanings[0]}" for r in recs
                ),
                "match_note": mt.note,
                "norm_source": n.source,
                "midas_ids": "|".join(mt.midas_ids),
                "midas_primary_form": recs[0].forms[0],
                "first_definition": meanings[0],
                "midas_forms": json.dumps(forms, ensure_ascii=False),
                "definitions": json.dumps(meanings, ensure_ascii=False),
                "n_senses": str(len(meanings)),
                "example_sentences": json.dumps(sentences[:5], ensure_ascii=False),
                "n_example_sentences": str(len(sentences)),
                "opacity": f"{n.opacity:.4f}",
                "familiarity": "" if n.familiarity is None else f"{n.familiarity:.3f}",
                "meaningfulness": "" if n.meaningfulness is None else f"{n.meaningfulness:.3f}",
                "literal_plausibility": ""
                if n.literal_plausibility is None
                else f"{n.literal_plausibility:.3f}",
                "min_word_count_enwiki": "" if freq is None else str(freq),
                "n_words": str(len(n.idiom.split())),
            }
        )
    return rows


_STATUS_ORDER = {"key": 0, "near": 1, "ambiguous": 2, "exact": 3}


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_inventory(
    out_path: Path = DEFAULT_OUT_PATH,
) -> tuple[list[dict[str, str]], list[NormRow]]:
    """Write the inventory (riskiest matches first) and a list of norm idioms with no match."""
    midas = load_midas()
    norms = load_norm_rows()
    matches = match_norms(norms, midas)
    matches.sort(key=lambda m: (_STATUS_ORDER[m.status], m.norm.idiom.lower()))
    rows = build_rows(matches, midas, load_word_frequencies())
    _write_csv(out_path, rows)
    matched = {m.norm.idiom for m in matches}
    unmatched = [n for n in norms if n.idiom not in matched]
    _write_csv(
        out_path.with_name(out_path.stem + "_unmatched.csv"),
        [
            {"norm_idiom": n.idiom, "norm_source": n.source, "opacity": f"{n.opacity:.4f}"}
            for n in unmatched
        ],
    )
    return rows, unmatched


def _opt(value: str) -> float | None:
    return float(value) if value.strip() else None


def _decision(raw: str) -> str:
    text = raw.strip().lower()
    if not text:
        return "unreviewed"
    for word in ("accept", "reject", "fix"):
        if text.startswith(word):
            return word
    raise ValueError(f"unknown match_checked value: {raw!r}")


def apply_review(
    review_rows: list[dict[str, str]], midas: list[MidasIdiom], word_freq: dict[str, int]
) -> tuple[list[dict[str, str]], Counter[tuple[str, str]], list[str]]:
    """Apply the A1 decisions: accept keeps all candidate ids, fix keeps the ids in match_note,
    reject drops the row, and blank (unreviewed) rows are excluded and returned by name."""
    known = {m.midas_id for m in midas}
    counts: Counter[tuple[str, str]] = Counter()
    unreviewed: list[str] = []
    kept: list[tuple[Match, str]] = []
    for r in review_rows:
        decision = _decision(r["match_checked"])
        counts[(r["match_status"], decision)] += 1
        if decision == "unreviewed":
            unreviewed.append(r["norm_idiom"])
        if decision in ("reject", "unreviewed"):
            continue
        ids = tuple(r["midas_ids"].split("|"))
        note = r["match_note"]
        if decision == "fix":
            # ids may be typed in match_checked ("fix with 547") or match_note; ignore matcher boilerplate
            boilerplate = re.sub(
                r"\d+ candidate MIDAS ids|jaccard=[\d.]+|fixed to [\d|]+", "", note
            )
            typed = re.findall(r"\d+", r["match_checked"]) or re.findall(r"\d+", boilerplate)
            ids = tuple(sorted(set(typed), key=int))
            if not ids or not set(ids) <= known:
                raise ValueError(
                    f"fix needs valid MIDAS ids in match_note: {r['norm_idiom']!r}: {note!r}"
                )
            note = f"fixed to {'|'.join(ids)}"
        norm = NormRow(
            r["norm_idiom"],
            "bulkes_tanner_2017"
            if r["norm_source"] == "bulkes_tanner_2017"
            else "libben_titone_2008",
            _opt(r["familiarity"]),
            _opt(r["meaningfulness"]),
            float(r["opacity"]),
            _opt(r["literal_plausibility"]),
        )
        status = cast("MatchStatus", r["match_status"])
        kept.append((Match(norm, ids, status, note), decision))
    rows = build_rows([m for m, _ in kept], midas, word_freq)
    for row, (_, decision) in zip(rows, kept, strict=True):
        row["match_checked"] = decision
    return rows, counts, unreviewed


def dedupe_shared_ids(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Keep one row per MIDAS idiom: rows sharing any id are merged into the Bulkes & Tanner row
    (else the first); the dropped norm rows are recorded in `also_normed_as`."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        ids = r["midas_ids"].split("|")
        for i in ids[1:]:
            parent[find(i)] = find(ids[0])
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        groups[find(r["midas_ids"].split("|")[0])].append(r)
    kept: list[dict[str, str]] = []
    dropped = 0
    for members in groups.values():
        members.sort(key=lambda r: r["norm_source"] != "bulkes_tanner_2017")
        keep, rest = members[0], members[1:]
        also = [
            {k: m[k] for k in ("norm_idiom", "norm_source", "opacity", "familiarity")} for m in rest
        ]
        kept.append({**keep, "also_normed_as": json.dumps(also, ensure_ascii=False)})
        dropped += len(rest)
    order = {r["norm_idiom"]: n for n, r in enumerate(rows)}
    kept.sort(key=lambda r: order[r["norm_idiom"]])
    return kept, dropped


def write_final_inventory(
    review_path: Path = DEFAULT_OUT_PATH, out_path: Path = DEFAULT_FINAL_PATH
) -> tuple[list[dict[str, str]], Counter[tuple[str, str]], list[str], int]:
    with review_path.open(encoding="utf-8-sig", newline="") as f:
        review_rows = list(csv.DictReader(f))
    rows, counts, unreviewed = apply_review(review_rows, load_midas(), load_word_frequencies())
    rows, merged = dedupe_shared_ids(rows)
    _write_csv(out_path, rows)
    return rows, counts, unreviewed, merged


# ---------------------------------------------------------------- MAGPIE-attested extension tier

DEFAULT_MAGPIE_PATH = _FILES / "magpie" / "MAGPIE_unfiltered.jsonl"
DEFAULT_MAGPIE_CANDIDATES_PATH = _FILES / "inventory" / "idiom_inventory_magpie_candidates.csv"
DEFAULT_MAGPIE_EXT_PATH = _FILES / "inventory" / "idiom_inventory_magpie.csv"
MAGPIE_SOURCE = "magpie_bnc"
_MIN_FORM_WORDS = 2


def magpie_idiomatic_uses(
    path: Path = DEFAULT_MAGPIE_PATH, min_confidence: float = 0.75
) -> Counter[str]:
    """Idiomatic ("i") uses per MAGPIE idiom type in its BNC sample, at or above `min_confidence`."""
    uses: Counter[str] = Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["label"] == "i" and float(rec["confidence"]) >= min_confidence:
                uses[str(rec["idiom"])] += 1
    return uses


def magpie_extension_rows(
    midas: list[MidasIdiom],
    uses: Counter[str],
    word_freq: dict[str, int],
    *,
    exclude_ids: set[str],
    exclude_names: set[str],
    min_uses: int,
) -> tuple[list[dict[str, str]], Counter[str]]:
    """Candidate MIDAS idioms without norms that MAGPIE attests as common (no match review needed).

    MAGPIE only decides *which* MIDAS idioms enter; form and definition come from MIDAS itself,
    so a wrong match can admit a less common idiom but never pairs an idiom with a wrong
    definition or wrong norms. A MAGPIE type matches a MIDAS idiom when its stemmed content key
    equals the key of one of the MIDAS forms (the "key" tier of `match_norms`); a key shared by
    several MIDAS idioms is skipped as ambiguous, and an idiom with no form of at least two words
    is skipped because the generator's idiom check could never find it. Fixed phrases ("in the long
    run") still pass here; `extension_tags` filters them before the tier is used.
    """
    by_key: dict[frozenset[str], set[str]] = defaultdict(set)
    for m in midas:
        for f in m.forms:
            if key := content_key(f):
                by_key[key].add(m.midas_id)
    counts: Counter[str] = Counter()
    hits: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for name, n in uses.items():
        ids = by_key.get(content_key(name), set())
        if not ids:
            counts["magpie_type_unmatched"] += 1
        elif len(ids) > 1:
            counts["magpie_type_ambiguous"] += 1
        else:
            hits[next(iter(ids))].append((name, n))
    by_id = {m.midas_id: m for m in midas}
    rows: list[dict[str, str]] = []
    for mid, types in hits.items():
        total = sum(n for _, n in types)
        m = by_id[mid]
        reason = (
            "already_in_inventory"
            if mid in exclude_ids or surface_form(m.forms[0]) in exclude_names
            else "single_word"
            if all(len(surface_form(f).split()) < _MIN_FORM_WORDS for f in m.forms)
            else "below_min_uses"
            if total < min_uses
            else "added"
        )
        counts[reason] += 1
        if reason != "added":
            continue
        freq = idiom_frequency(m.forms[0], word_freq)
        rows.append(
            {
                "match_status": "magpie_key",
                "match_checked": "accept (auto, extension tier)",
                "norm_idiom": m.forms[0],
                "candidates": f"{mid}: {m.forms[0]} - {m.meanings[0]}",
                "match_note": "",
                "norm_source": MAGPIE_SOURCE,
                "midas_ids": mid,
                "midas_primary_form": m.forms[0],
                "first_definition": m.meanings[0],
                "midas_forms": json.dumps(list(m.forms), ensure_ascii=False),
                "definitions": json.dumps(list(m.meanings), ensure_ascii=False),
                "n_senses": str(len(m.meanings)),
                "example_sentences": json.dumps(list(m.sentences[:5]), ensure_ascii=False),
                "n_example_sentences": str(len(m.sentences)),
                "opacity": "",
                "familiarity": "",
                "meaningfulness": "",
                "literal_plausibility": "",
                "min_word_count_enwiki": "" if freq is None else str(freq),
                "n_words": str(len(m.forms[0].split())),
                "also_normed_as": "[]",
                "magpie_types": json.dumps(sorted(t for t, _ in types), ensure_ascii=False),
                "magpie_idiomatic_uses": str(total),
            }
        )
    rows.sort(key=lambda r: -int(r["magpie_idiomatic_uses"]))
    return rows, counts


def write_magpie_extension(
    final_path: Path = DEFAULT_FINAL_PATH,
    out_path: Path = DEFAULT_MAGPIE_CANDIDATES_PATH,
    *,
    min_uses: int = 5,
    min_confidence: float = 0.75,
) -> tuple[list[dict[str, str]], Counter[str]]:
    with final_path.open(encoding="utf-8", newline="") as f:
        final = list(csv.DictReader(f))
    rows, counts = magpie_extension_rows(
        load_midas(),
        magpie_idiomatic_uses(min_confidence=min_confidence),
        load_word_frequencies(),
        exclude_ids={i for r in final for i in r["midas_ids"].split("|")},
        exclude_names={surface_form(r["norm_idiom"]) for r in final}
        | {surface_form(r["midas_primary_form"]) for r in final},
        min_uses=min_uses,
    )
    _write_csv(out_path, rows)
    return rows, counts
