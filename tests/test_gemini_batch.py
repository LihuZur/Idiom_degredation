"""Batch replies cut off at max_output_tokens must not reach the cache (sync client already guards)."""

from google.genai import types

from augmentation.providers.gemini_batch import reply_text


def _response(text: str | None, finish: types.FinishReason) -> types.GenerateContentResponse:
    parts = [types.Part(text=text)] if text is not None else []
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(parts=parts, role="model"), finish_reason=finish)
        ]
    )


def test_complete_reply_is_returned_stripped():
    assert (
        reply_text(_response('  {"choice": "NONE"} ', types.FinishReason.STOP))
        == '{"choice": "NONE"}'
    )


def test_cut_off_reply_is_dropped():
    assert reply_text(_response('{"span": "in the run', types.FinishReason.MAX_TOKENS)) is None


def test_missing_or_empty_reply_is_none():
    assert reply_text(None) is None
    assert reply_text(_response("   ", types.FinishReason.STOP)) is None
