# Idiom Degradation — Investigating the Effect of Idiomatic Language on NLP Tasks

## 1. Motivation & Hypothesis

In NLP it is commonly assumed that idiomatic language is harder for language
models to understand than direct / verbatim language. This project sets out to
test that assumption empirically.

**Central question:** Does phrasing task inputs in idiomatic language cause a
measurable degradation in model performance, over and above the degradation
that any paraphrasing causes?

To separate the effect of *paraphrasing itself* from the effect of *idiomatic
phrasing*, every original example `x` will be augmented into two variants:

- `x_paraphrase` — rephrased in different words, **without** idioms (control).
- `x_idiomatic` — rephrased **with** idiomatic expressions.

Labels `y` are preserved. Models are then evaluated on the three variants
(original, paraphrase, idiomatic) using the metrics defined in each dataset's
original paper.

We report two deltas per (model, dataset) pair:

- Δ<sub>paraphrase</sub> = metric(paraphrase) − metric(original)
- Δ<sub>idiom</sub> = metric(idiomatic) − metric(paraphrase)

The paraphrase variant is the **control group**: it isolates the incremental
effect of idiomatic phrasing from the baseline effect of rewording.

## 2. Scope

- **Datasets**: 2 in the initial run (SST-2, MMLU); admin cap of 3 leaves
  room for one more later.
- **3 variants** per dataset: original / paraphrase / idiomatic.
- **Active model track (this phase):**
  - **Decoder track** — instruction-tuned generative LLMs, zero/few-shot.
- **Deferred to future work:**
  - **Encoder track** — fine-tuned classifier models. All **training and
    fine-tuning is out of scope for the current phase**; it is documented
    in §4.1 for a later phase.
- **Metrics**: paper-standard per dataset + paired deltas across variants.

## 3. Datasets

Datasets selected for the initial run.

| # | Dataset | Task | Metric (paper) |
|---|---------|------|----------------|
| 1 | SST-2 (Stanford Sentiment Treebank) | Binary sentiment classification | Accuracy |
| 2 | MMLU (Massive Multitask Language Understanding) | Multi-choice knowledge QA | Accuracy |

## 4. Models

Per administrator feedback:

- Very large models (32B, 8x7B) are hard to run — the decoder list is
  **trimmed** to a smaller subset.
- **Encoder** models will be added later; they require fine-tuning, which is
  **deferred to future work** (see §4.1).
- The **newer** model families **Gemma 4** and **Qwen 3.5** should be
  included in place of older equivalents.

### 4.1 Encoder track (fine-tuned) — *future work, deferred*

> **Not part of the current phase.** All training / fine-tuning is out of
> scope for now; this section captures the plan for a later phase.

Once enabled, encoder models will be fine-tuned on the original training
split of each classification-style dataset, then evaluated on all three test
variants:

- BERT-base
- RoBERTa-base
- DeBERTa-v3-base (and optionally DeBERTa-v3-large)

> The exact scope of encoder fine-tuning (original-only vs. also on
> paraphrased / idiomatic training data) is a future-work decision.

### 4.2 Decoder track (zero / few-shot) — *active track for this phase*

Trimmed subset of the original proposal, with new families added:

- Qwen3.5-1.5B-Instruct
- Qwen3.5-7B-Instruct
- Gemma-4-9B-Instruct *(latest available Gemma 4 variant)*
- Mistral-7B-Instruct-v0.3
- Phi-4-mini-instruct
- DeepSeek-R1-Distill-Qwen-7B
- SmolLM2-1.7B-Instruct *(ungated, different pretraining lineage; small-model comparison point)*
- Llama-3.2-1B-Instruct *(gated on HF, pending manual review; smallest model in the roster)*
- Qwen3.5-0.5B-Instruct *(ungated; smaller sibling of Qwen3.5-1.5B-Instruct, testing whether its idiomatic-degradation signal scales with size within the Qwen family)*
- Qwen3.5-3B-Instruct *(ungated; fills the capability gap between Qwen3.5-1.5B-Instruct, the only model with a significant mmlu idiomatic-degradation result, and Qwen3.5-7B-Instruct, which shows a much weaker effect)*
- Gemma-4-2B-Instruct *(gated on HF, same license as Gemma-4-9B-Instruct; smaller sibling to test the same capability-sweet-spot hypothesis)*
- StableLM-2-1.6B-Chat *(ungated; different architecture/lineage at a similar size to the sweet-spot models above)*

## 5. Pipeline Overview

The flow is split into four explicit, independently runnable stages. Each
stage consumes files on disk and produces new files on disk, so any stage can
be re-run without repeating the others.

```
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
│ Stage 1: Clean   │   │ Stage 2: Augment │   │ Stage 3: Evaluate    │
│                  │   │                  │   │                      │
│ raw dataset      │──►│ cleaned CSV      │──►│ variant triple       │
│      ↓           │   │  + augmenter     │   │  + evaluated model   │
│ cleaned CSV      │   │  model           │   │      ↓               │
│ (original tasks) │   │      ↓           │   │ per-variant metrics  │
│                  │   │ paraphrase.csv   │   │ + paired per-task    │
│                  │   │ idiomatic.csv    │   │ predictions          │
└──────────────────┘   └──────────────────┘   └──────────┬───────────┘
                                                          │
                                                          ▼
                                              ┌──────────────────────┐
                                              │ Stage 4: Visualize    │
                                              │                      │
                                              │ result JSONs         │
                                              │      ↓               │
                                              │ deltas, comparison   │
                                              │ tables, Plotly       │
                                              │ figures              │
                                              └──────────────────────┘
```

Stage 4 is a cross-run step: it aggregates result files across models and
datasets, computes deltas / significance, and emits comparison tables and
Plotly figures. It reads only Stage 3 outputs, so it can be re-run on its own.

### Stage 1 — Clean

Loads a raw dataset and produces a single CSV of task rows that are
well-suited to being rephrased and idiomatized:

- filter out rows outside a configured input-length band — very short inputs
  (e.g. one-word sentences) leave no room for idioms, very long inputs waste
  augmentation budget and dilute the effect being measured;
- strip markup, control characters, and URLs; normalize whitespace and quoting- same as original paper;
- deduplicate; no need to shuffle; optionally cap to a target row count
  for compute budget;
- attach a stable `id` to every retained row so all downstream variants align.

**Output:** `datasets_out/{dataset}/original.csv` (schema in
[ARCHITECTURE.md](ARCHITECTURE.md) §3).

### Stage 2 — Augment

Takes **(cleaned CSV, augmenter-model id)** as arguments and produces the two
augmented CSVs using two fixed prompt templates plus the validators:

- `datasets_out/{dataset}/paraphrase.csv` — reworded, **no idioms**.
- `datasets_out/{dataset}/idiomatic.csv` — reworded, **with idioms**.

Validators filter augmented rows for (a) semantic similarity to the original,
(b) label preservation, and (c) idiom presence / absence for the correct
variant. `id` and `y` are preserved so all three CSVs align row-for-row.

The augmentation prompt should also contain the label `y` so that the rephrasing does not change the task results.

### Stage 3 — Evaluate

Takes **(variant triple, evaluated-model id)** as arguments and runs
inference on all three CSVs with the **same** dataset-specific prompt and
instructions — only `x` changes between runs. Results are joined by `id`, so
every original task yields one matched (`original`, `paraphrase`,
`idiomatic`) tuple of predictions per model, enabling paired comparisons.

Encoder models are first fine-tuned (see phases below); decoders run
zero / few-shot with dataset-specific prompt formats. Both tracks share the
same Stage 3 entrypoint.

### Stage 4 — Visualize

Aggregate result files across models and datasets, compute
Δ<sub>paraphrase</sub> and Δ<sub>idiom</sub>, run paired bootstrap
significance tests, and produce comparison tables and Plotly figures
(`analysis/tables/`, `analysis/figures/`). Runs on Stage 3 outputs only, so it
is re-runnable independently of the earlier stages.

## 6. Metrics

- Each dataset uses the metric defined in its original paper (accuracy for
  SST-2 and MMLU).
- Cross-variant reporting: Δ<sub>paraphrase</sub>, Δ<sub>idiom</sub>.
- **Significance**: delta confidence intervals from a paired bootstrap over
  examples, and delta p-values from McNemar's exact test (`scipy`) on the
  paired per-example correctness — comparing variants within the same model.

## 7. Phases & Milestones

| Phase | Goal | Key outputs |
|-------|------|-------------|
| 0. Setup | Repo scaffolding, env, config schema, dataset loaders | `data/`, `configs/`, dev env |
| 1. Augmentation | Paraphrase + idiomatize pipelines, validators, cached datasets | `datasets_out/{ds}_{variant}.parquet` |
| 2. Decoder evaluation | Zero/few-shot runs across trimmed decoder list | per-run metric JSONs |
| 3. Analysis & writeup | Aggregation, deltas, significance, plots, report | tables, figures, final report |
| *F. Encoder fine-tuning — future work* | Train encoder classifiers, checkpoint per dataset | fine-tuned model checkpoints |

## 8. Repository Layout

```
Idiom_degredation/
├── README.md                  # this file
├── ARCHITECTURE.md            # module design & data contract
├── pyproject.toml             # uv-managed project, deps, ruff & pyright config
├── uv.lock                    # uv lockfile (committed)
├── .python-version            # pinned Python version (uv-managed)
├── .pre-commit-config.yaml    # ruff + pyright hooks
├── configs/                   # YAML per stage / experiment
├── data/                      # raw dataset loaders (HF → Example iterator)
├── cleaning/                  # Stage 1: filter & normalize raw → cleaned CSV
├── augmentation/              # Stage 2: cleaned CSV → paraphrase + idiomatic
├── datasets_out/              # persisted CSVs: {dataset}_{variant}.csv
├── models/                    # model registry (encoder + decoder loaders)
├── eval/                      # Stage 3: run one model on a variant triple
├── analysis/                  # Stage 4: cross-run aggregation, deltas, Plotly figures
├── scripts/                   # CLI entrypoints (one per stage)
└── tests/                     # unit tests for pipelines & metrics
```

## 9. Engineering Standards

These standards apply to **all** code in this repository and are enforced by
tooling; CI and pre-commit will fail on violations.

### 9.1 Language & typing

- **Python** only, managed by **uv** (see §9.2). Use latest python version, supporting the required packages.
- **Strict typing** enforced by **pyright** in `strict` mode
  (`typeCheckingMode = "strict"` in `pyproject.toml`).
  - Public functions and class attributes must be fully annotated.
  - No implicit `Any`; no untyped `dict` / `list` in public interfaces — use
    `TypedDict`, `dataclass`, or `pydantic` models.
  - `# type: ignore` requires a specific error code and a short comment.
- **Formatting & linting** by **ruff** (both `ruff format` and `ruff check`).
  Ruff is the single source of truth — no `black`, no `isort`, no `flake8`.

### 9.2 Packaging & environments

- **uv** is the only supported package manager and environment tool.
- `pyproject.toml` declares dependencies; `uv.lock` is committed and
  authoritative.
- Standard commands:
  ```bash
  uv sync              # create / update the local .venv from uv.lock
  uv run <cmd>         # run a command inside the project env
  uv add <pkg>         # add a runtime dependency
  uv add --dev <pkg>   # add a dev-only dependency
  ```
- All CLI examples in §10 assume `uv run` in front (omitted for brevity).

### 9.3 Pre-commit hooks

`.pre-commit-config.yaml` runs on every commit and in CI:

1. `ruff format` — auto-format.
2. `ruff check --fix` — lint, auto-fix where safe.
3. `pyright` — strict type-check on staged files.

Setup:

```bash
uv run pre-commit install          # one-time, per clone
uv run pre-commit run --all-files  # run against the whole repo
```

### 9.4 GPU compatibility

- All model code must **auto-detect** the compute device via a single helper
  (`models.device.select_device()`) that returns, in order of preference:
  1. `cuda` if `torch.cuda.is_available()`,
  2. `mps` if available (Apple Silicon),
  3. `cpu` otherwise.
- Model loading, tensor placement, and inference batches must all honor the
  selected device — **never** hard-code `"cuda"`.
- Batch sizes and precision (`fp16` / `bf16` on GPU, `fp32` on CPU) are
  configurable per model in its registry entry with sensible defaults.
- Code must run to completion on CPU-only machines (for unit tests and
  smoke-testing), just slower.

### 9.5 Modularity & extensibility

Adding a new dataset or model must not require touching Stage 3 code. The
project is built around three **registries** (see [ARCHITECTURE.md](ARCHITECTURE.md) §2):

- `data/registry.py`         — `@register_dataset("sst2")`
- `models/registry.py`       — `@register_model("qwen3.5-7b-instruct", kind="decoder")`
- `augmentation/registry.py` — `@register_augmenter("gemini")` (also
  `"anthropic"`, `"openai"`)

A new dataset or model is added by:

1. Creating a new file in the relevant package that defines and
   `@register_*`‑decorates the class.
2. Adding a YAML config under `configs/` referencing it by name.
3. (Nothing else — no edits to runners, scripts, or existing modules.)

Removal is symmetric: delete the file and any configs that reference it.

### 9.6 Visualisation

- **Plotly only** for all charts, tables, and dashboards (`plotly.express`
  or `plotly.graph_objects`). **matplotlib / seaborn are not permitted** in
  this repo; a ruff rule bans the import.
- Figures are saved as interactive `.html` under `analysis/figures/`.
  **HTML only for this phase** — static `.png` export (via `plotly` +
  `kaleido`) is deferred to future work (§13).

### 9.7 Models & datasets sourcing

- **Hugging Face first.** All models and datasets are loaded via the
  Hugging Face ecosystem (`transformers`, `datasets`, `huggingface_hub`)
  whenever available.
- Any component that **cannot** come from Hugging Face must be **explicitly
  called out** — both in its registry entry (`source="non-hf"` with a
  reason string) and in this README under "Known non-HF components".
- **Known non-HF components (current):**
  - The **augmenter LLM** used in Stage 2 is a hosted API, not an HF model.
    Three providers are implemented and selectable via config
    (`augmenter` + `augmenter_model`): `gemini` (default), `anthropic`
    (Claude), and `openai` (GPT). This is intentional — the augmenter must
    be strictly stronger than the models under evaluation.
- Any future addition to this list requires a note here explaining why HF
  was insufficient.

## 10. How to Run *(placeholder — implemented in later phases)*

All commands assume the project env is active (`uv sync` once, then either
`source .venv/bin/activate` or prefix each command with `uv run`).

The project is installed in editable mode (`pyproject.toml [project.scripts]`),
so each stage runs as a short `uv run idiom-<stage>` console command.

```bash
# Stage 1 — Clean a raw dataset into an aligned CSV of task rows
uv run idiom-clean --config configs/clean/sst2.yaml
# → datasets_out/sst2_original.csv

# Stage 2 — Given a config (dataset, augmenter id, prompts, validators), produce both variants
uv run idiom-augment --config configs/augment/sst2.yaml
# → datasets_out/sst2/paraphrase.csv, datasets_out/sst2/idiomatic.csv
#   (+ paraphrase.meta.json / idiomatic.meta.json sidecars)

# Or run every registered dataset in one pass (requires configs/augment/{dataset}.yaml for each)
uv run idiom-augment --all

# (Future work — encoder track, deferred) Fine-tune before Stage 3
# uv run python scripts/train_encoder.py --config configs/encoder/sst2_bert_base.yaml

# Stage 3 — Run one model on the full (original + paraphrase + idiomatic) triple
uv run idiom-eval \
    --dataset sst2 \
    --model   <decoder-model-id>
# → results/sst2/<model-id>.json  (per-variant metrics + paired per-task rows)

# Stage 4 — Visualize: three flags-only CLIs, no --config file.
# Bootstrap params (--n-resamples, --ci, --seed) are shared across all three.

# Aggregate every result file into one summary table (tables only, no figures)
uv run idiom-analyze \
    --results results/ \
    --out-dir analysis/tables/ \
    --n-resamples 10000 \
    --ci 0.95 \
    --seed 0 \
    --force
# → analysis/tables/summary.{csv,md} (+ summary.meta.json)

# Per-dataset cross-model chart — compare every model that ran on `sst2`
# across its original / paraphrase / idiomatic variants
uv run idiom-plot-dataset \
    --dataset sst2 \
    --results results/ \
    --out-dir analysis/figures/ \
    --n-resamples 10000 \
    --ci 0.95 \
    --seed 0 \
    --force
# → analysis/figures/sst2_cross_model.html
#   analysis/tables/sst2_cross_model.{csv,md} (+ .meta.json)

# Or every dataset with a result folder, in one pass
uv run idiom-plot-dataset --all --results results/ --out-dir analysis/figures/

# Cross-dataset summary chart — grouped delta bars across all datasets & models
uv run idiom-plot-summary \
    --results results/ \
    --out-dir analysis/figures/ \
    --n-resamples 10000 \
    --ci 0.95 \
    --seed 0 \
    --force
# → analysis/figures/cross_dataset_summary.html
#   analysis/tables/cross_dataset_summary.{csv,md} (+ .meta.json)
```

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Augmentation LLM changes semantics or flips labels | Validators: semantic-similarity threshold + label-preservation check + prompt contains the labels as well |
| "Idiomatic" rewrite contains no real idioms | Idiom-presence check (idiom list and/or LLM judge) |
| "Paraphrase" rewrite accidentally uses idioms | Idiom-absence check on control variant |
| Compute budget for larger decoders | Trimmed model list per admin; batching, vLLM, quantization where safe |
| Prompt leakage across variants | Frozen prompt templates + hash logged with each augmented row |
| Non-determinism in decoder outputs | Fixed seed, temperature=0 for eval, model revisions pinned |

## 12. Open Decisions (pending approval)

The following items are intentionally **not** decided in this document. They
will be proposed and confirmed separately before implementation:

1. ~~Choice of the **paraphraser LLM** provider / version.~~ **Resolved:**
   all three (`gemini`, `anthropic`, `openai`) are registered, with `gemini`
   as the default; provider + model are chosen purely via config.
2. Exact model revisions / checkpoints for each decoder (especially Gemma 4
   and Qwen 3.5 variants, subject to availability).
3. Pinned **Python version** (≥ 3.11 assumed; exact minor to be set in
   `.python-version`).

## 13. Future Work (deferred, not in current phase)

Explicitly out of scope for the current phase; recorded here so the docs
stay consistent when we come back to them:

- **Encoder track (§4.1)** — fine-tuning of BERT / RoBERTa / DeBERTa
  classifiers. All training / fine-tuning code, configs, scripts, and
  checkpoints are deferred. Sub-decision when picked up: fine-tuning scope
  (original-only vs. also on paraphrase / idiomatic training data).
- **Third dataset** — one additional dataset may be added later, up to the
  admin cap of 3.
- **PNG export for figures (kaleido)** — Stage 4 figures are HTML-only for
  this phase; static `.png` export via `plotly` + `kaleido` is deferred.
