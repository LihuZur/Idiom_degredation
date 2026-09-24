"""LLM opacity-rating pass — fills the human-norms coverage gap (RQ2 plan §3/§8).

`data/norms/opacity.py` only covers idioms in Bulkes & Tanner (2017) /
Libben & Titone (2008) (~1,080 idiom types, citation form). Real idioms
extracted from dataset text (`analysis/idiom_annotation.py`) mostly aren't in
either list, even after fuzzy content-word matching
(`data.norms.opacity.build_fuzzy_opacity_index`). This module rates the
remaining idioms directly with an LLM, once per distinct idiom string across
all datasets, and persists the result via `data.norms.llm_opacity` so the
cost is paid once, not once per dataset.

Design mirrors `analysis/idiom_annotation.py`: per-idiom response cache
(keyed by idiom string instead of row id), retry-then-skip, pydantic config.
Simpler in one respect — one call per distinct idiom type, not per row.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from analysis.idiom_annotation import CacheCfg, DecodingCfg, ExtractorSpec, RetryCfg, check_env_key
from analysis.prompts.loader import load_prompt
from augmentation.cache import ResponseCache
from augmentation.providers.base import EmptyResponseError, LLMClient, LLMError, build_client
from cleaning.hashing import prompt_hash
from data.norms.llm_opacity import append_llm_opacity_ratings, load_llm_opacity_ratings
from data.norms.opacity import build_fuzzy_opacity_index, content_words, load_opacity_norms


class OpacityRatingConfig(BaseModel, extra="forbid"):
    """RQ2 opacity-rating config, parsed from `configs/annotate/opacity_rating.yaml`."""

    prompt_file: str = "opacity_rating_v1.txt"
    extractor: ExtractorSpec
    decoding: DecodingCfg = Field(default_factory=DecodingCfg)
    retry: RetryCfg = Field(default_factory=RetryCfg)
    cache: CacheCfg = Field(default_factory=CacheCfg)
    output_dir: str = "datasets_out"


def load_config(path: Path) -> OpacityRatingConfig:
    """Parse and validate a `configs/annotate/opacity_rating.yaml`-shaped file."""
    raw: object = yaml.safe_load(path.read_text())
    return OpacityRatingConfig.model_validate(raw)


@dataclass(slots=True)
class RatingStats:
    """Tallies for one `run_rating` invocation."""

    total_idioms: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    skipped: int = 0


def collect_extracted_idioms(dataset_dir: Path) -> set[str]:
    """Return the set of distinct idiom surface strings extracted for one dataset."""
    path = dataset_dir / "idiom_extractions.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path, keep_default_na=False, dtype={"id": str})
    out: set[str] = set()
    for raw in df["idioms_json"]:
        idioms_json = str(raw)
        if not idioms_json:
            continue
        idioms: list[str] = json.loads(idioms_json)
        for idiom in idioms:
            out.add(idiom.strip().lower())
    return out


def unresolved_idioms(idioms: set[str]) -> set[str]:
    """Idioms in `idioms` not already covered by an exact, fuzzy, or cached LLM rating."""
    opacity_table = load_opacity_norms()
    fuzzy_index = build_fuzzy_opacity_index(opacity_table)
    llm_table = load_llm_opacity_ratings()
    unresolved: set[str] = set()
    for idiom in idioms:
        if idiom in opacity_table or idiom in llm_table:
            continue
        if content_words(idiom) in fuzzy_index:
            continue
        unresolved.add(idiom)
    return unresolved


def parse_rating(raw: str) -> float:
    """Parse the model's single-number response into an opacity value in [0, 1].

    Raises:
        EmptyResponseError: If the response is blank, unparseable, or the
            parsed value falls outside [0, 1].
    """
    text = raw.strip()
    if not text:
        raise EmptyResponseError("opacity rater returned empty output")
    first_line = text.splitlines()[0].strip()
    try:
        value = float(first_line)
    except ValueError as exc:
        raise EmptyResponseError(
            f"opacity rater returned unparseable output: {first_line!r}"
        ) from exc
    if not (0.0 <= value <= 1.0):
        raise EmptyResponseError(f"opacity rater returned out-of-range value: {value}")
    return value


def _rate_once(
    *,
    client: LLMClient,
    idiom: str,
    template: str,
    decoding: DecodingCfg,
    retry: RetryCfg,
) -> float:
    """Call the rater on one idiom with retry-on-transient; raises `LLMError` on failure."""
    user_prompt = template.format(idiom=idiom)
    last_exc: BaseException | None = None
    for attempt in range(1, retry.max_attempts + 1):
        try:
            raw = client.complete(
                system="",
                user=user_prompt,
                temperature=decoding.temperature,
                max_output_tokens=decoding.max_output_tokens,
            )
            return parse_rating(raw)
        except EmptyResponseError as exc:
            last_exc = exc
            if attempt < retry.max_attempts and retry.backoff_seconds > 0:
                time.sleep(retry.backoff_seconds * (2 ** (attempt - 1)))
        except LLMError as exc:
            last_exc = exc
            if attempt < retry.max_attempts and retry.backoff_seconds > 0:
                time.sleep(retry.backoff_seconds * (2 ** (attempt - 1)))
    raise LLMError(
        str(last_exc) if last_exc else f"rating failed after {retry.max_attempts} attempts"
    )


def run_rating(cfg: OpacityRatingConfig, dataset_names: list[str]) -> RatingStats:
    """Rate every not-yet-covered idiom across `dataset_names`, persisting each new rating.

    Idempotent/resumable: idioms already covered by the exact/fuzzy human-norm
    tiers, or already in the persistent LLM-ratings cache, cost nothing.
    """
    check_env_key(cfg.extractor.provider)

    idioms: set[str] = set()
    for name in dataset_names:
        idioms |= collect_extracted_idioms(Path(cfg.output_dir) / name)

    todo = sorted(unresolved_idioms(idioms))
    stats = RatingStats(total_idioms=len(todo))

    template = load_prompt(cfg.prompt_file)
    template_hash_value = prompt_hash(template)
    cache = ResponseCache(Path(cfg.cache.dir), enabled=cfg.cache.enabled)
    client = build_client(cfg.extractor.provider, cfg.extractor.model)
    label = cfg.extractor.label()

    decoding = cfg.decoding
    if cfg.extractor.temperature is not None:
        decoding = decoding.model_copy(update={"temperature": cfg.extractor.temperature})

    for idiom in tqdm(todo, desc="rate opacity", unit="idiom"):
        cached = cache.get(template_hash_value, label, idiom)
        if cached is not None:
            stats.cache_hits += 1
            append_llm_opacity_ratings({idiom: float(cached["opacity"])}, model=label)
            continue
        try:
            opacity = _rate_once(
                client=client, idiom=idiom, template=template, decoding=decoding, retry=cfg.retry
            )
        except LLMError as exc:
            tqdm.write(f"[skip] {idiom!r} — {exc}")
            stats.skipped += 1
            continue
        cache.put(template_hash_value, label, idiom, {"opacity": opacity})
        stats.api_calls += 1
        append_llm_opacity_ratings({idiom: opacity}, model=label)

    return stats
