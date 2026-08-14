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

- **Datasets**: 3 — SST-2, MMLU, and MNLI. This fills the admin cap of 3.
- **3 variants** per dataset: original / paraphrase / idiomatic.
- **Active model track (this phase):**
  - **Decoder track** — instruction-tuned generative LLMs, zero/few-shot.
- **Deferred to future work:**
  - **Encoder track** — fine-tuned classifier models. All **training and
    fine-tuning is out of scope for the current phase**; it is documented
    in §4.1 for a later phase.
- **Metrics**: paper-standard per dataset + paired deltas across variants.

## 3. Datasets

| # | Dataset | Task | Metric (paper) | Rewritten text |
|---|---------|------|----------------|----------------|
| 1 | SST-2 (Stanford Sentiment Treebank) | Binary sentiment classification | Accuracy | the sentence |
| 2 | MMLU (Massive Multitask Language Understanding) | Multi-choice knowledge QA | Accuracy | the question stem (`choices` untouched) |
| 3 | MNLI (MultiNLI) | 3-way natural language inference | Accuracy | the **premise** (`hypothesis` untouched) |

MNLI was added last, and deliberately: on the other two datasets the surface
manipulation is weakly coupled to the decision. Sentiment survives rewording
(English idioms are overwhelmingly evaluative, so idiomatizing a review can even
*sharpen* the signal), and an MMLU answer still hinges on knowledge sitting in
the untouched `choices`. An entailment label, by contrast, turns on a *literal*
reading of the premise, so a figurative surface forces a resolution step that can
fail — the failure mode documented in IMPLI (Stowe et al., ACL 2022) and
Figurative-NLI.

**MNLI specifics.** Source is `nyu-mll/multi_nli`, both validation splits
(`validation_matched` + `validation_mismatched`, 19,647 pairs), keeping the
native 3-way label space (`entailment` / `neutral` / `contradiction`; chance is
33%). MNLI pairs roughly three hypotheses with each premise; the loader emits
**one row per premise** — the first pair per `promptID` — so no premise appears
twice and Stage 4's paired test keeps its independence assumption. Example `id`
is the source `pairID`.

### 3.1 Built dataset sizes

Rows surviving to the aligned variant triple, per dataset:

| Dataset | Stage 1 written | Aligned triple | Retention |
|---------|-----------------|----------------|-----------|
| SST-2 | 2000 | 1548 | 77.4% |
| MMLU | 1107 | 1085 | 98.0% |
| MNLI | 5127 | **4709** | **91.8%** |

MNLI's funnel: 19,647 pairs → 6,661 after one-row-per-premise selection → 5,127
after the 10/100-token length filter → 4,709 aligned. Label balance in the final
set is 1,693 entailment / 1,401 neutral / 1,615 contradiction, across all 10
genres.

The three retention figures differ for a reason that is itself informative.
SST-2's movie-review text is idiom-dense, so `idiom_absence` rejects ~22% of
attempted paraphrases; MMLU's academic phrasing almost never contains idioms, so
nearly everything survives. MNLI sits between them, and its dominant failure is
the *other* gate: 2.0% of premises could not be de-idiomatized, while 6.2% could
not have a natural idiom inserted while preserving the entailment relation.

> **Known limitation — genre confound.** `validation_mismatched` contributes five
> genres held out of MNLI's training data, so the merged set mixes in-domain and
> out-of-domain text. The length filter also prunes unevenly across genres —
> measured retention ranged from 50% (fiction) to 89% (oup), since short
> spoken/fiction premises are cut hardest by the 10-token floor. `meta.genre` is
> recorded on every row so this can be checked, but genre-stratified analysis is
> not in scope.

> **Known limitation — MNLI uses a different augmenter.** SST-2 and MMLU were
> built with `gemini-3.6-flash`; MNLI uses `gemini-3.5-flash-lite`. MNLI is the
> only uncapped dataset (~5.1k rows × 6 calls vs MMLU's 1.1k), so Stage 2 runtime
> mattered here and nowhere else. Flash-lite was faster per call and produced
> *deeper* rewrites: median token-set overlap with the original is 0.33 on MNLI
> against 0.39 on SST-2 and 0.54 on MMLU, though dataset and augmenter vary
> together and cannot be separated post hoc. Rebuilding the other two datasets
> on it would have invalidated their warm caches for no research benefit. The
> asymmetry is nonetheless real: cross-dataset comparisons carry an augmenter
> difference as well as a task difference.

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

The evaluated panel is 14 models spanning 0.5B–7B across 10 families. These are
exactly the models in the registry, in `results/`, and in the report — the three
lists are kept identical, so a model dropped from one is dropped from all:

- Qwen3.5-0.5B-Instruct, Qwen3.5-1.5B-Instruct, Qwen3.5-7B-Instruct
  *(three sizes in one family, separating size-driven from family-driven effects)*
- Falcon3-3B-Instruct, Falcon3-7B-Instruct
- OLMo-2-1B-Instruct, OLMo-2-7B-Instruct
- Gemma-4-2B-Instruct *(gated on HF)*
- Mistral-7B-Instruct-v0.3
- Phi-4-mini-instruct
- Granite-3.1-2B-Instruct
- H2O-Danube3-4B-Chat
- Yi-1.5-6B-Chat
- SmolLM2-1.7B-Instruct *(ungated, different pretraining lineage)*

Note the registry ids are offset from the published names: `qwen3.5-*` is really
Qwen2.5 and `gemma-4-*` is really Gemma-2 (`hf_repo` in each result JSON is
authoritative). The report uses the published names.

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

Validators filter augmented rows for (a) label preservation and (b) idiom
presence / absence for the correct variant. `id` and `y` are preserved so all
three CSVs align row-for-row.

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
  SST-2, MMLU, and MNLI — for MNLI this is 3-way accuracy over
  `entailment`/`neutral`/`contradiction`, so the chance floor is 33% rather
  than SST-2's 50%).
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

## 10. How to Run

All commands assume the project env is active (`uv sync` once, then either
`source .venv/bin/activate` or prefix each command with `uv run`).

The project is installed in editable mode (`pyproject.toml [project.scripts]`),
so each stage runs as a short `uv run idiom-<stage>` console command.

```bash
# Stage 1 — Clean a raw dataset into an aligned CSV of task rows
uv run idiom-clean --config configs/clean/sst2.yaml
# → datasets_out/sst2_original.csv
# Same shape for the other two: configs/clean/{mmlu,mnli}.yaml

# Stage 2 — Given a config (dataset, augmenter id, prompts, validators), produce both variants
uv run idiom-augment --config configs/augment/sst2.yaml
# → datasets_out/sst2/paraphrase.csv, datasets_out/sst2/idiomatic.csv
#   (+ paraphrase.meta.json / idiomatic.meta.json sidecars)
# Requires the provider key in the env, e.g. `export GEMINI_API_KEY=...`
# (GOOGLE_API_KEY is not read). MNLI is uncapped, so read
# datasets_out/mnli/original.meta.json `row_counts.written` first — Stage 2
# makes about 6 calls per row (1 rewrite + 2 judges, per variant).

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
# --out-dir is the *root*: each CLI appends `tables/` and `figures/` itself,
# so pass `analysis/` — not `analysis/tables/` or `analysis/figures/`.

# Aggregate every result file into one summary table (tables only, no figures)
uv run idiom-analyze \
    --results results/ \
    --out-dir analysis/ \
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
    --out-dir analysis/ \
    --n-resamples 10000 \
    --ci 0.95 \
    --seed 0 \
    --force
# → analysis/figures/sst2_cross_model.html
#   analysis/tables/sst2_cross_model.{csv,md} (+ .meta.json)
#   analysis/figures/sst2_cross_model_significant.html      (filtered cut, see below)
#   analysis/tables/sst2_cross_model_significant.{csv,md} (+ .meta.json)

# Or every dataset with a result folder, in one pass
uv run idiom-plot-dataset --all --results results/ --out-dir analysis/

# Cross-dataset summary chart — grouped delta bars across all datasets & models
uv run idiom-plot-summary \
    --results results/ \
    --out-dir analysis/ \
    --n-resamples 10000 \
    --ci 0.95 \
    --seed 0 \
    --force
# → analysis/figures/cross_dataset_summary.html
#   analysis/tables/cross_dataset_summary.{csv,md} (+ .meta.json)
#   analysis/figures/cross_dataset_summary_significant.html  (filtered cut, see below)
#   analysis/tables/cross_dataset_summary_significant.{csv,md} (+ .meta.json)

# Report derivations — every number in the report that is not in a Stage 4 table
# comes from one of these three. All read committed artifacts only; each needs
# analysis/tables/summary.csv (idiom-analyze above) to exist first.

# The control-free contrast: what a study with no paraphrase arm would have
# reported. Asserts its recomputed accuracies against summary.csv, so it fails
# loudly rather than printing stale numbers. (Report §4 opener.)
uv run idiom-naive-contrast

# Per-arm rewrite depth — median token-set overlap with the original for each
# arm. Shows the two arms are NOT matched on depth; the idiomatic arm is the
# shallower edit on all three datasets. Diagnostic only — the report no longer
# quotes these figures. Needs datasets_out/.
uv run idiom-arm-overlap

# Regenerate the two machine-generated LaTeX tables (tab:summary, tab:permodel)
# → analysis/tables/t_summary.tex, analysis/tables/t_permodel.tex
# Paste each over the matching table block in main.tex, matched by \label.
# Never hand-edit those blocks in main.tex — edit this generator instead.
uv run idiom-report-tables
```

### The `_significant` cut

Both plotting CLIs emit a second, filtered copy of every figure and companion
table alongside the full one. The filter keeps a run only when
Δ<sub>paraphrase</sub> **or** Δ<sub>idiom</sub> has a bootstrap confidence
interval that does **not** cross zero — i.e. the sign of the effect is
resolved by the data rather than being consistent with "no change". Runs where
both intervals straddle zero are dropped.

The cut is written only if at least one run survives; when nothing does, the
CLI says so and no `_significant` files are produced (so a missing file means
"nothing resolved", never "this step did not run"). On the current results:

| Dataset | Models | Survive the cut |
|---------|-------:|----------------:|
| sst2    | 17     | 8               |
| mmlu    | 17     | 1               |
| mnli    | 17     | 13              |

In the **unfiltered** figures the same distinction is drawn rather than
filtered: a delta bar whose CI clears zero is solid and marked `*`, one that
straddles zero is faded.

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Augmentation LLM changes semantics or flips labels | Validators: label-preservation check + prompt contains the labels as well. **Not** mitigated: semantic drift that preserves the label — there is no similarity gate (§13) |
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
- ~~**Third dataset** — one additional dataset may be added later, up to the
  admin cap of 3.~~ **Resolved:** MNLI (§3) fills the third and final slot.
- **Genre-stratified MNLI analysis** — `meta.genre` is recorded on every MNLI
  row, but breaking the deltas down by genre (matched vs. mismatched) is
  deferred.
- **A real semantic-similarity gate** — `semantic_similarity` is still an
  always-pass stub, so each variant has two effective validators, not three.
  It would be worth most on MNLI, where a rewrite that shifts meaning changes
  the entailment relation outright; enabling it needs an embedding-model
  decision.
- **Splitting the Stage 2 judge onto a stronger model** — one client currently
  both writes the rewrite and grades whether it preserved the label.
- **Transient API failures are misrecorded as validation failures** — a row whose
  every retry died on a 5xx/429 is written to the durable `{variant}.skipped.json`
  manifest as `validation_failed`, which resume consults specifically so it is
  never retried. Nothing was judged and rejected; the calls never completed. Each
  such row also costs its partner in the other variant, since reconciliation drops
  the id from both. Three rows hit this during the MNLI run (504 / 503 / 429) and
  were recovered by hand; the classification itself is unfixed. Stage 2 should
  distinguish "no verdict" from "failed the verdict".
- **Paraphrase no-ops** — 40 of 4,709 MNLI paraphrase rows (0.85%) are
  byte-identical to their original premise, so they contribute nothing as a
  control. The `idiom_absence` judge passes them because a literal sentence
  trivially contains no idioms. A minimum-edit-distance check would catch them.
- **PNG export for figures (kaleido)** — Stage 4 figures are HTML-only for
  this phase; static `.png` export via `plotly` + `kaleido` is deferred.
