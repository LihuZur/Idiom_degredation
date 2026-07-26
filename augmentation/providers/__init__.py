"""Hosted-LLM provider clients for the Stage 2 augmenter / judge (README §9.7).

Each client wraps one vendor SDK behind the `LLMClient` seam so the augmenter
and the LLM-judge validators share a single provider-agnostic code path. The
concrete SDK imports live in the per-provider modules, which are imported
lazily by `build_client` so that only the selected provider's SDK is loaded at
run time (and none are imported in the mock-backed offline tests).
"""
