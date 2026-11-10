"""Gemini Batch API (google-genai) for large generation waves at roughly half the price.

Used live for the first waves (SST-2, MNLI, 2026-10-22): `batches.create(model=, src=[inlined
requests])`, `batches.get(name=)`, job `dest.inlined_responses`. A reply cut off at
max_output_tokens is returned as None, like the sync client's MAX_TOKENS guard, so it is never cached
(before 2026-10-29 batch mode stored such replies; they show up as `bad_json`).
"""

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from google import genai
from google.genai import types

from augmentation.providers.base import LLMError
from augmentation.providers.env import load_env

_API_KEY_ENV = "GEMINI_API_KEY"
_RUNNING = {
    "JOB_STATE_UNSPECIFIED",
    "JOB_STATE_QUEUED",
    "JOB_STATE_PENDING",
    "JOB_STATE_RUNNING",
    "JOB_STATE_UPDATING",
}
_DONE = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}


def run_batch(
    prompts: Sequence[str],
    *,
    model: str,
    temperature: float,
    max_output_tokens: int,
    poll_seconds: float = 30.0,
    display_name: str = "idiom-v2",
    state_dir: Path | None = None,
    thinking_level: str = "low",
    usage: Counter[str] | None = None,
) -> list[str | None]:
    """Submit one batch job, wait for it, and return one text (or None) per prompt, in order.

    With `state_dir` the job name is saved under a hash of the request, so a rerun after a crash
    re-attaches to the same job instead of paying for a new one. The thinking level is set
    explicitly (as in the sync client) so batch replies match sync ones; billed tokens are added
    to `usage` if given.
    """
    load_env()
    api_key = os.environ.get(_API_KEY_ENV)
    if not api_key:
        raise LLMError(f"missing {_API_KEY_ENV} environment variable")
    client = genai.Client(api_key=api_key)
    requests = [
        {
            "contents": [{"role": "user", "parts": [{"text": p}]}],
            "config": {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "thinking_config": {"thinking_level": thinking_level.upper()},
            },
        }
        for p in prompts
    ]
    digest = hashlib.sha256(
        json.dumps([model, temperature, max_output_tokens, thinking_level, list(prompts)]).encode()
    ).hexdigest()[:24]
    state_file = state_dir / f"{digest}.json" if state_dir is not None else None
    job = None
    if state_file is not None and state_file.is_file():
        try:
            job = client.batches.get(name=json.loads(state_file.read_text())["name"])
        except Exception:  # expired or unknown job -> submit a new one
            job = None
    if job is None:
        job = client.batches.create(
            model=model,
            src=requests,  # type: ignore[arg-type]
            config={"display_name": display_name},
        )
        if state_file is not None:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(json.dumps({"name": job.name}))
    name = job.name or ""
    while job.state is not None and job.state.name in _RUNNING:
        time.sleep(poll_seconds)
        job = client.batches.get(name=name)
    state = job.state.name if job.state is not None else "unknown"
    if state not in _DONE:
        raise LLMError(f"gemini batch {name} ended in state {state}")
    responses = job.dest.inlined_responses if job.dest is not None else None
    if not responses or len(responses) != len(prompts):
        raise LLMError(
            f"gemini batch {name} returned {len(responses or [])} responses for {len(prompts)} prompts"
        )
    out: list[str | None] = []
    for r in responses:
        if usage is not None and r.response is not None:
            meta = r.response.usage_metadata
            usage["calls"] += 1
            if meta is not None:
                usage["input_tokens"] += meta.prompt_token_count or 0
                usage["output_tokens"] += meta.candidates_token_count or 0
                usage["thinking_tokens"] += meta.thoughts_token_count or 0
        out.append(reply_text(r.response))
    return out


def reply_text(response: "types.GenerateContentResponse | None") -> str | None:
    """Stripped reply text, or None if empty or cut off at max_output_tokens (never cached)."""
    if response is None:
        return None
    candidates = getattr(response, "candidates", None) or []
    if candidates and candidates[0].finish_reason == types.FinishReason.MAX_TOKENS:
        return None
    text = response.text
    return text.strip() if text and text.strip() else None
