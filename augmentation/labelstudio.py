"""Label Studio tooling for the human validation (THESIS_PLAN_v2 §5, annotations A2 and A6).

Gold set: annotators see the original, the idiomatic and the span-literal text side by side with the idiom,
span, definition and gold label, answer yes/no per criterion (meaning kept, label unchanged, fluent; idiom
present and used figuratively for the idiomatic text; no idiom and gloss fits the slot for the span-literal
text) and may correct a text. Blind relabel: one rewritten text at a time, without the original or its
label, and the annotator assigns the label ("not sure" allowed); this also gives a human accuracy per
variant. The importer flattens a Label Studio JSON export and reports Cohen's kappa per criterion.

The v2 functions above are kept to reproduce the first waves. The v3 section at the end (THESIS_PLAN_v3 §3,
`paper_preparations/annotation_guidelines.md`) is the one in use.
"""

import csv
import difflib
import json
import random
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from html import escape
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr

from analysis.stats import cohen_kappa

_YES_NO = ("yes", "no")
_ACTIONS = ("valid", "corrected", "invalid")
_TERCILES = 3
BLIND_LABELS: dict[str, tuple[tuple[str, str], ...]] = {
    "sst2": (("positive", "1"), ("negative", "0")),
    "mnli": (("entailment", "0"), ("neutral", "1"), ("contradiction", "2")),
    "agnews": (("World", "0"), ("Sports", "1"), ("Business", "2"), ("Sci/Tech", "3")),
}
GOLD_CRITERIA = (
    "idiomatic_meaning",
    "idiomatic_label",
    "idiomatic_fluent",
    "idiomatic_figurative",
    "span_literal_meaning",
    "span_literal_label",
    "span_literal_fluent",
    "span_literal_no_idiom",
    "span_literal_slot_fit",
)


def _choices(name: str, to: str, options: Sequence[str]) -> str:
    opts = "".join(f"<Choice value={quoteattr(o)}/>" for o in options)
    return f'<Choices name="{name}" toName="{to}" choice="single" required="true">{opts}</Choices>'


def gold_config_xml() -> str:
    def block(variant: str, extra: Sequence[str]) -> str:
        parts = [
            f'<Header value="{variant.replace("_", " ")}"/>',
            f'<Text name="{variant}" value="${variant}"/>',
        ]
        for crit in ("meaning", "label", "fluent", *extra):
            parts.append(f'<Header value={quoteattr(crit.replace("_", " ") + "?")} size="5"/>')
            parts.append(_choices(f"{variant}_{crit}", variant, _YES_NO))
        parts.append(_choices(f"{variant}_action", variant, _ACTIONS))
        parts.append(
            f'<TextArea name="{variant}_correction" toName="{variant}" rows="2" placeholder="corrected text (only if corrected)"/>'
        )
        return "".join(parts)

    return (
        "<View>"
        '<Header value="Original"/><Text name="original" value="$original"/>'
        '<Text name="info" value="Gold label: $label | idiom: $idiom | span: $span | meaning: $definition"/>'
        + block("idiomatic", ("figurative",))
        + block("span_literal", ("no_idiom", "slot_fit"))
        + "</View>"
    )


def blind_config_xml(dataset: str) -> str:
    labels = [name for name, _ in BLIND_LABELS[dataset]] + ["not sure"]
    hyp = (
        '<Header value="Hypothesis"/><Text name="hypothesis" value="$hypothesis"/>'
        if dataset == "mnli"
        else ""
    )
    return (
        "<View>"
        '<Header value="Text"/><Text name="text" value="$text"/>'
        + hyp
        + _choices("label", "text", labels)
        + "</View>"
    )


def _read(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def load_generated(out_dir: Path) -> list[dict[str, Any]]:
    """One record per generated item: the three texts, label and the idiom record."""
    orig, idi, lit = (
        _read(out_dir / f"{v}.csv") for v in ("original", "idiomatic", "span_literal")
    )
    items: list[dict[str, Any]] = []
    for id_, o in orig.items():
        if id_ not in idi or id_ not in lit:
            continue
        meta: dict[str, Any] = json.loads(idi[id_]["meta"])
        items.append(
            {
                "id": id_,
                "label": o["y"],
                "original": o["x"],
                "idiomatic": idi[id_]["x"],
                "span_literal": lit[id_]["x"],
                "meta": meta,
            }
        )
    return items


def stratified_gold_sample(
    items: Sequence[dict[str, Any]], n: int, seed: int
) -> list[dict[str, Any]]:
    """n items spread over label x length tercile x opacity tercile, drawn from raw generator output.

    Idioms of the unnormed tier (opacity None) form their own opacity stratum.
    """
    if n >= len(items):
        return list(items)
    lengths = sorted(len(i["original"].split()) for i in items)
    ops = sorted(float(o) for i in items if (o := i["meta"]["idiom"]["opacity"]) is not None) or [
        0.0
    ]
    lc = (lengths[len(lengths) // _TERCILES], lengths[2 * len(lengths) // _TERCILES])
    oc = (ops[len(ops) // _TERCILES], ops[2 * len(ops) // _TERCILES])

    def tercile(v: float, cuts: tuple[float, float]) -> int:
        return 0 if v < cuts[0] else 1 if v < cuts[1] else 2

    def opacity_stratum(op: object) -> int:
        return _TERCILES if op is None else tercile(float(str(op)), oc)

    strata: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for i in items:
        key = (
            i["label"],
            tercile(len(i["original"].split()), lc),
            opacity_stratum(i["meta"]["idiom"]["opacity"]),
        )
        strata[key].append(i)
    rng = random.Random(seed)
    for g in strata.values():
        rng.shuffle(g)
    out: list[dict[str, Any]] = []
    total = len(items)
    for g in strata.values():
        out.extend(g[: max(1, round(n * len(g) / total))])
    rng.shuffle(out)
    return out[:n]


def gold_tasks(sample: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for i in sample:
        rec = i["meta"]["idiom"]
        tasks.append(
            {
                "data": {
                    "id": i["id"],
                    "label": i["meta"].get("label_name", i["label"]),
                    "original": i["original"],
                    "idiomatic": i["idiomatic"],
                    "span_literal": i["span_literal"],
                    "idiom": rec["idiom"],
                    "span": rec["span"],
                    "definition": rec["definition"],
                }
            }
        )
    return tasks


def blind_tasks(dataset: str, sample: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Two tasks per item (idiomatic and span-literal text), shuffled; no original, label or variant shown."""
    tasks: list[dict[str, Any]] = []
    for i in sample:
        for variant in ("idiomatic", "span_literal"):
            data = {
                "id": i["id"],
                "variant": variant,
                "gold": i["label"],
                "text": i[variant],
                "hypothesis": i["meta"].get("hypothesis", ""),
            }
            tasks.append({"data": data})
    random.Random(seed).shuffle(tasks)
    return tasks


def flatten_export(export: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Label Studio JSON export -> one row per (item, variant, annotator, criterion)."""
    rows: list[dict[str, str]] = []
    for task in export:
        data = task.get("data", {})
        for ann in task.get("annotations", []):
            who = ann.get("completed_by")
            annotator = str(who.get("id") if isinstance(who, dict) else who)  # pyright: ignore[reportUnknownMemberType]
            for res in ann.get("result", []):
                value = res.get("value", {})
                text = value.get("choices", value.get("text", [""]))
                rows.append(
                    {
                        "item_id": str(data.get("id", "")),
                        "variant": str(data.get("variant", "")),
                        "gold": str(data.get("gold", "")),
                        "annotator": annotator,
                        "criterion": str(res.get("from_name", "")),
                        "value": " ".join(text) if isinstance(text, list) else str(text),  # pyright: ignore[reportUnknownArgumentType]
                    }
                )
    return rows


def agreement(rows: Sequence[dict[str, str]]) -> list[dict[str, str | int | float]]:
    """Raw agreement and Cohen's kappa per yes/no criterion for exactly two annotators."""
    by: dict[str, dict[tuple[str, str], str]] = defaultdict(dict)
    for r in rows:
        by[r["criterion"]][(r["item_id"], r["annotator"])] = r["value"]
    out: list[dict[str, str | int | float]] = []
    for crit, vals in sorted(by.items()):
        annotators = sorted({a for _, a in vals})
        if len(annotators) != 2 or crit.endswith("_correction"):  # noqa: PLR2004
            continue
        a, b = annotators
        shared = sorted(
            {i for i, _ in vals}
            & {i for i, ann in vals if ann == a}
            & {i for i, ann in vals if ann == b}
        )
        va = [vals[(i, a)] for i in shared]
        vb = [vals[(i, b)] for i in shared]
        agree = (
            sum(x == y for x, y in zip(va, vb, strict=True)) / len(shared)
            if shared
            else float("nan")
        )
        kappa = (
            cohen_kappa([x == "yes" for x in va], [x == "yes" for x in vb])
            if shared and set(va + vb) <= set(_YES_NO)
            else float("nan")
        )
        out.append(
            {"criterion": crit, "n_items": len(shared), "agreement": agree, "cohen_kappa": kappa}
        )
    return out


def blind_accuracy(
    rows: Sequence[dict[str, str]], dataset: str
) -> list[dict[str, str | int | float]]:
    """Human accuracy per annotator and variant; 'not sure' counts as wrong."""
    code = dict(BLIND_LABELS[dataset])
    tally: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in rows:
        if r["criterion"] == "label":
            tally[(r["annotator"], r["variant"])].append(code.get(r["value"]) == r["gold"])
    return [
        {"annotator": a, "variant": v, "n": len(x), "accuracy": sum(x) / len(x)}
        for (a, v), x in sorted(tally.items())
    ]


# ---------------------------------------------------------------------------------------------------------
# v3 (THESIS_PLAN_v3 §3, paper_preparations/annotation_guidelines.md). Label Studio is the tool ID10M-JAM used
# for its human validation (§3.4). Annotators see the original, the idiomatic and the literal version with the
# changes highlighted, plus the label and fixed context; never any AI verdict (the generator's `self_check` is
# dropped here). Everything runs locally: no model is called while building tasks, annotating or importing.

V3_STEPS = ("step1_label", "step2_idiom", "step3_literal", "step4_rest")
V3_PASS_FAIL = ("pass", "fail")
V3_NATURALNESS = ("3", "2", "1", "n/a")
V3_ACTIONS = ("valid", "fix", "invalid")
V3_REASONS = (
    "label_changed",
    "answer_hint",
    "criterion_shift",
    "wrong_sense",
    "meaningless",
    "literal_mismatch",
    "grammar",
    "detail_dropped",
    "new_name_or_number",
    "literal_clash",
    "other",
)
V3_FIELDS = (*V3_STEPS, "naturalness", "action", "reason", "fix_idiomatic", "fix_literal", "note")
_SST2_NAMES = {"0": "negative", "1": "positive"}
_AG_NAMES = {"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"}
_LETTERS_V3 = "ABCDEFGH"
_TOKEN = re.compile(r"\S+")


def _label_name(dataset: str, y: str, meta: dict[str, Any]) -> str:
    if dataset == "sst2":
        return _SST2_NAMES.get(y, y)
    if dataset == "agnews":
        return _AG_NAMES.get(y, y)
    if dataset == "mmlu":
        choices = list(meta.get("choices", []))
        idx = int(meta.get("answer_index", y))
        return f"{_LETTERS_V3[idx]}) {choices[idx]}" if idx < len(choices) else y
    return str(meta.get("label_name", y))


def load_v3(out_dir: Path, dataset: str) -> list[dict[str, Any]]:
    """One record per kept v3 pair, with everything an annotator or the AI checker needs.

    The generator's own verdict (`self_check`) is left out on purpose (guidelines §2: no AI verdicts).
    """
    orig, idi, lit = (
        _read(out_dir / f"{v}.csv") for v in ("original", "idiomatic", "span_literal")
    )
    items: list[dict[str, Any]] = []
    for id_, o in orig.items():
        if id_ not in idi or id_ not in lit:
            continue
        meta: dict[str, Any] = json.loads(idi[id_]["meta"])
        rec = meta["idiom"]
        items.append(
            {
                "key": f"{dataset}:{id_}",
                "dataset": dataset,
                "id": id_,
                "y": o["y"],
                "label": _label_name(dataset, o["y"], meta),
                "original": o["x"],
                "idiomatic": idi[id_]["x"],
                "literal": lit[id_]["x"],
                "idiom": rec["idiom"],
                "definition": rec["definition"],
                "idiom_text": rec["idiom_text"],
                "literal_text": rec["literal_text"],
                "idiomatic_span": list(rec["idiomatic_span"]),
                "literal_span": list(rec["literal_span"]),
                "hypothesis": str(meta.get("hypothesis", "")),
                "choices": list(meta.get("choices", [])),
                "answer_index": meta.get("answer_index"),
                "idiom_id": str(rec["midas_ids"]).split("|")[0],
                "original_meta": json.loads(o["meta"]) if o.get("meta") else {},
                "source_dir": str(out_dir),
            }
        )
    return items


def _changed_tokens(a: str, b: str) -> tuple[set[int], set[int]]:
    """Token indices of `a` that are missing from `b`, and of `b` that are new compared with `a`."""
    ta = [m.group().lower() for m in _TOKEN.finditer(a)]
    tb = [m.group().lower() for m in _TOKEN.finditer(b)]
    gone: set[int] = set()
    new: set[int] = set()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ta, b=tb, autojunk=False).get_opcodes():
        if op in ("delete", "replace"):
            gone.update(range(i1, i2))
        if op in ("insert", "replace"):
            new.update(range(j1, j2))
    return gone, new


def _html(text: str, span: tuple[int, int] | None, marked: set[int], tag: str) -> str:
    """Escape `text`; wrap the idiom / literal span in <mark>, other `marked` tokens in `tag`."""
    out: list[str] = []
    last = 0
    for n, m in enumerate(_TOKEN.finditer(text)):
        out.append(escape(text[last : m.start()]))
        word = escape(m.group())
        in_span = span is not None and m.start() < span[1] and m.end() > span[0]
        if in_span:
            word = f"<mark>{word}</mark>"
        elif n in marked:
            word = f"<{tag}>{word}</{tag}>"
        out.append(word)
        last = m.end()
    out.append(escape(text[last:]))
    return "".join(out)


def highlight_v3(item: dict[str, Any]) -> dict[str, str]:
    """HTML for the three versions: idiom / literal phrase in <mark>, other inserted words underlined,
    words of the original that are missing from the idiomatic version struck through."""
    gone, new_i = _changed_tokens(item["original"], item["idiomatic"])
    _, new_l = _changed_tokens(item["original"], item["literal"])
    i_span = tuple(item["idiomatic_span"])
    l_span = tuple(item["literal_span"])
    return {
        "original_html": _html(item["original"], None, gone, "s"),
        "idiomatic_html": _html(item["idiomatic"], (i_span[0], i_span[1]), new_i, "u"),
        "literal_html": _html(item["literal"], (l_span[0], l_span[1]), new_l, "u"),
    }


def _context_html(item: dict[str, Any]) -> str:
    parts = [f"<b>Correct label:</b> {escape(item['label'])}"]
    if item["dataset"] == "mnli" and item["hypothesis"]:
        parts.append(f"<b>Hypothesis (never changed):</b> {escape(item['hypothesis'])}")
    if item["dataset"] == "mmlu" and item["choices"]:
        opts = "<br/>".join(
            f"{_LETTERS_V3[n]}) {escape(str(c))}" for n, c in enumerate(item["choices"])
        )
        parts.append(f"<b>Answer choices (never changed):</b><br/>{opts}")
    parts.append(
        f"<b>Idiom:</b> {escape(item['idiom'])} — {escape(item['definition'])}"
        f"<br/><b>As used:</b> “{escape(item['idiom_text'])}” → literal: “{escape(item['literal_text'])}”"
    )
    return "<br/>".join(parts)


def v3_tasks(items: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Label Studio tasks in a random order fixed by `seed` (one seed per annotator)."""
    tasks = [
        {
            "data": {
                "key": i["key"],
                "dataset": i["dataset"],
                "id": i["id"],
                "context_html": _context_html(i),
                **highlight_v3(i),
            }
        }
        for i in items
    ]
    random.Random(seed).shuffle(tasks)
    return tasks


def v3_config_xml() -> str:
    """The annotation form: guidelines steps 1-5 plus the overall action (guidelines §3-§4)."""

    def choices(name: str, options: Sequence[str], header: str, *, required: bool = True) -> str:
        opts = "".join(f"<Choice value={quoteattr(o)}/>" for o in options)
        req = "true" if required else "false"
        return (
            f'<Header value={quoteattr(header)} size="5"/>'
            f'<Choices name="{name}" toName="idiomatic" choice="single" required="{req}" '
            f'showInline="true">{opts}</Choices>'
        )

    when = 'visibleWhen="choice-selected" whenTagName="action"'
    return (
        "<View>"
        "<Style>mark{background:#ffe08a} u{text-decoration-color:#1f6feb} s{color:#b42318}</Style>"
        '<Header value="Context"/><HyperText name="context" value="$context_html"/>'
        '<Header value="Original (struck through: words missing from the idiomatic version)"/>'
        '<HyperText name="original" value="$original_html"/>'
        '<Header value="Idiomatic (highlighted: idiom; underlined: other inserted words)"/>'
        '<HyperText name="idiomatic" value="$idiomatic_html"/>'
        '<Header value="Literal (highlighted: plain-meaning phrase)"/>'
        '<HyperText name="literal" value="$literal_html"/>'
        + choices(
            "step1_label",
            V3_PASS_FAIL,
            "1. Label: still correct for both versions, not given away?",
        )
        + choices("step2_idiom", V3_PASS_FAIL, "2. Idiom: figurative, makes sense where it stands?")
        + choices(
            "step3_literal",
            V3_PASS_FAIL,
            "3. Literal: same meaning and strength, fits grammatically, no idiom?",
        )
        + choices(
            "step4_rest",
            V3_PASS_FAIL,
            "4. Rest: nothing dropped, no new name/number, no clash, no new grammar error?",
        )
        + choices("naturalness", V3_NATURALNESS, "5. Naturalness (n/a if a step failed)")
        + choices("action", V3_ACTIONS, "Overall action")
        + choices("reason", V3_REASONS, "Primary reason (only for fix or invalid)", required=False)
        + f'<View {when} whenChoiceValue="fix">'
        '<Header value="Corrected idiomatic version" size="5"/>'
        '<TextArea name="fix_idiomatic" toName="idiomatic" rows="3" maxSubmissions="1"/>'
        '<Header value="Corrected literal version (must differ only in the idiom)" size="5"/>'
        '<TextArea name="fix_literal" toName="idiomatic" rows="3" maxSubmissions="1"/>'
        "</View>"
        '<Header value="Note (optional)" size="5"/>'
        '<TextArea name="note" toName="idiomatic" rows="2" maxSubmissions="1"/>'
        "</View>"
    )


def flatten_v3_export(export: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Label Studio JSON export -> one row per (item, annotator) with one column per form field."""
    rows: list[dict[str, str]] = []
    for task in export:
        data = task.get("data", {})
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            who = ann.get("completed_by")
            row = dict.fromkeys(V3_FIELDS, "")
            row |= {
                "key": str(data.get("key", "")),
                "dataset": str(data.get("dataset", "")),
                "id": str(data.get("id", "")),
                "annotator": str(who.get("id") if isinstance(who, dict) else who),  # pyright: ignore[reportUnknownMemberType]
            }
            for res in ann.get("result", []):
                name = str(res.get("from_name", ""))
                value = res.get("value", {})
                got = value.get("choices", value.get("text", [""]))
                if name in row:
                    row[name] = " ".join(got) if isinstance(got, list) else str(got)  # pyright: ignore[reportUnknownArgumentType]
            rows.append(row)
    return rows


def nominal_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa for two raters' paired categorical judgments (any number of categories)."""
    if len(a) != len(b) or not a:
        raise ValueError("nominal_kappa needs two non-empty sequences of equal length")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe >= 1.0 else (po - pe) / (1 - pe)


def v3_agreement(rows: Sequence[dict[str, str]]) -> list[dict[str, str | int | float]]:
    """Agreement and Cohen's kappa per field for exactly two annotators, plus the derived 'bad' (fix or
    invalid) decision the AI checker is scored on."""
    annotators = sorted({r["annotator"] for r in rows})
    if len(annotators) != 2:  # noqa: PLR2004
        return []
    a, b = annotators
    by = {(r["key"], r["annotator"]): r for r in rows}
    shared = sorted({r["key"] for r in rows if (r["key"], a) in by and (r["key"], b) in by})
    out: list[dict[str, str | int | float]] = []
    fields: list[tuple[str, Callable[[dict[str, str]], str]]] = [
        (f, lambda r, f=f: r[f]) for f in (*V3_STEPS, "naturalness", "action", "reason")
    ]
    fields.append(("bad", lambda r: str(r["action"] in ("fix", "invalid"))))
    for name, get in fields:
        va = [get(by[(k, a)]) for k in shared]
        vb = [get(by[(k, b)]) for k in shared]
        if not shared:
            continue
        out.append(
            {
                "field": name,
                "n_items": len(shared),
                "agreement": sum(x == y for x, y in zip(va, vb, strict=True)) / len(shared),
                "cohen_kappa": nominal_kappa(va, vb),
            }
        )
    return out
