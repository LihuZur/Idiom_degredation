"""Meaning-based shortlists: ask for replaceable spans, then retrieve inventory idioms by definition.

A random shortlist almost never contains an idiom that fits the sentence (1 of 12 items on every
dataset in the first live test), so each item first gets a few proposed spans with a plain-language
meaning; idioms whose definition is closest to that meaning form the shortlist.
"""

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_MAX_LEN = 128
_BATCH = 64

Matrix = npt.NDArray[np.float32]
EmbedFn = Callable[[Sequence[str], bool], Matrix]


@dataclass(frozen=True, slots=True)
class SpanProposal:
    span: str
    meaning: str
    idiom: str = ""  # idiom the model itself suggests for the span (spans prompt v3), if any


def parse_proposals(raw: str | None, x: str, limit: int = 6) -> list[SpanProposal]:
    """Valid proposals only: the span occurs exactly once in `x` and a meaning is given."""
    if not raw:
        return []
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[SpanProposal] = []
    for entry in data:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(entry, dict):
            continue
        span = str(entry.get("span", ""))  # pyright: ignore[reportUnknownMemberType]
        meaning = str(entry.get("meaning", "")).strip()  # pyright: ignore[reportUnknownMemberType]
        idiom = str(entry.get("idiom", "") or "").strip()  # pyright: ignore[reportUnknownMemberType]
        if span and meaning and x.count(span) == 1:
            out.append(SpanProposal(span, meaning, idiom))
    return out[:limit]


def bge_embedder() -> EmbedFn:
    """Local BGE embeddings (CLS pooling, L2-normalised); the model loads on first use."""
    state: dict[str, Any] = {}

    def embed(texts: Sequence[str], query: bool) -> Matrix:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        if not state:
            state["tok"] = AutoTokenizer.from_pretrained(EMBED_MODEL)
            state["model"] = AutoModel.from_pretrained(EMBED_MODEL).eval()
        rows: list[Matrix] = []
        for start in range(0, len(texts), _BATCH):
            chunk = [(_QUERY_PREFIX if query else "") + t for t in texts[start : start + _BATCH]]
            batch = state["tok"](
                chunk, padding=True, truncation=True, max_length=_MAX_LEN, return_tensors="pt"
            )
            with torch.no_grad():
                cls = state["model"](**batch).last_hidden_state[:, 0]
            rows.append(torch.nn.functional.normalize(cls, dim=-1).numpy().astype(np.float32))
        return np.concatenate(rows) if rows else np.zeros((0, 1), dtype=np.float32)

    return embed


class IdiomRetriever:
    """Ranks inventory idioms by how close their definition is to a given meaning."""

    def __init__(
        self,
        idiom_ids: Sequence[str],
        documents: Sequence[str],
        embed: EmbedFn,
        cache_dir: Path | None = None,
    ) -> None:
        self._ids = list(idiom_ids)
        self._embed = embed
        self._docs = self._document_matrix(documents, cache_dir)

    def _document_matrix(self, documents: Sequence[str], cache_dir: Path | None) -> Matrix:
        if cache_dir is None:
            return self._embed(documents, False)
        digest = hashlib.sha256(
            json.dumps([EMBED_MODEL, list(documents)]).encode("utf-8")
        ).hexdigest()[:20]
        path = cache_dir / f"idiom_docs_{digest}.npy"
        if path.is_file():
            return np.load(path)
        matrix = self._embed(documents, False)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix)
        return matrix

    def rank(self, meaning: str) -> list[str]:
        """All idiom ids, best match first."""
        sims = self._docs @ self._embed([meaning], True)[0]
        return [self._ids[j] for j in np.argsort(-sims, kind="stable")]
