# Artifact Registry

Every artifact below follows the same story: **the pipeline was executed to create the
artifact. The artifact was then frozen and reused for evaluation so that every comparison
referred to the same experimental run.** None of these files are hand-written or
unexplained — each one traces back to a real, tested function originally implemented in
`newstart_ai_benchmark/src/newstart_ai/` (vendored into `MVP/newstart_ai_mvp/` as a
self-contained copy), and each one is loaded (never regenerated) by the safe, default-mode
`newstart_ai_mvp` commands and the `MVP/notebooks/`.

Paths below are relative to the frozen-artifact root (`newstart_ai_benchmark/`, or
`NEWSTART_BENCHMARK_ROOT` if set — see `MVP/README.md`).

---

## Language audit
**Contains:** one row per collected document — detected language, confidence, and an
English/non-English/uncertain status.
**Files:** `artifacts/family_aware/reports/language_audit_v1.csv` (754 rows) +
`language_audit_manifest_v1.json`.
**Created by:** `newstart_ai.data.build_language_audit` / `save_language_audit`, via
`python -m newstart_ai_mvp.stage1_validate_and_audit --run`.
**Config values:** `settings.family_aware.language_filter` (detector name/thresholds).
**How to load:** `newstart_ai_mvp.artifact_report.describe_language_audit(settings)`, or
`pd.read_csv(...)` directly.
**Used by:** `MVP/notebooks/01_frozen_split_and_chunking.ipynb`.
**Regeneration stability:** deterministic (py3langid is not randomized) — exact re-run.

## Family audit
**Contains:** document-family grouping (form + instructions + supplement + translations),
manual override findings, per-document `final_modeling_eligibility`, and 8 category reports
(non-singleton families, cross-agency code conflicts, cross-language families, etc.).
**Files:** `artifacts/family_aware/reports/family_audit_v1.csv` (754 rows), 8×
`family_report_*.csv`, 2× `family_duplicate_candidates_{exact,near}.csv` +
`artifacts/family_aware/manifests/family_audit_manifest_v1.json`,
`family_overrides_{v1,v2}.json` (versioned, never overwritten in place).
**Created by:** `newstart_ai.data.build_full_family_audit` / `save_family_audit`, via
`python -m newstart_ai_mvp.stage1_validate_and_audit --run`.
**How to load:** `artifact_report.describe_family_audit(settings)`.
**Used by:** `MVP/notebooks/01_frozen_split_and_chunking.ipynb`.
**Regeneration stability:** deterministic except for the hand-confirmed manual findings
baked into `family_audit.py` — a rerun reproduces the same audit exactly.

## Family-aware split
**Contains:** the frozen, zero-family-overlap train/validation/test partition — 461/99/99
documents.
**Files:** `data/family_aware_splits/{train,validation,test}.csv` +
`family_split_manifest_v1.json`.
**Created by:** `newstart_ai.data.create_family_aware_split` / `save_family_split` (with 4
leakage-proof assertions run before saving), via `python -m newstart_ai_mvp.stage2_build_split --run`.
**Config values:** `settings.family_aware.split` (`random_seed: 42`, ratios).
**Used by:** every downstream notebook and CLI stage.
**Regeneration stability:** deterministic (seeded hash-based family assignment) — exact re-run.

## Chunks
**Contains:** overlapping 512-token windows (128-token overlap) for every eligible document,
reproducing exactly what BERT's tokenizer sees.
**Files:** `data/family_aware_chunks/{train,validation,test}_chunks.csv` (4300/790/820
chunks) + `artifacts/family_aware/manifests/chunk_manifest_v1.json`.
**Created by:** `newstart_ai.data.build_all_split_chunks` / `save_family_aware_chunks` (9
invariant assertions), via `python -m newstart_ai_mvp.stage3_build_chunks --run`.
**Config values:** `settings.family_aware.chunking` (`max_seq_length`, `chunk_overlap_tokens: 128`).
**Regeneration stability:** deterministic given a fixed tokenizer revision — exact re-run.

## Masked documents/chunks
**Contains:** the masked derivative of every document and chunk — agency names, form
numbers, OMB numbers, and URLs replaced with placeholders.
**Files:** `data/family_aware_masked/{split}_masked_{documents,chunks}.csv` +
`artifacts/family_aware/manifests/masking_policy_v1.json`.
**Created by:** `newstart_ai.data.build_masked_documents` / `build_masked_chunks` (real,
tested logic) + a small persistence helper added in `newstart_ai_mvp/stage4_build_masked.py`
(this module never had a `save_*()` function of its own in `src/newstart_ai`), via
`python -m newstart_ai_mvp.stage4_build_masked --run`.
**Regeneration stability:** deterministic (regex-based rule matching) — exact re-run.

## Partial-input selections
**Contains:** which chunk index(es) represent "beginning", "middle", "end", and
"beginning_middle_end" for every document.
**Files:** `data/family_aware_conditions/{partial_input_selections,test_partial_input_selections}.csv`
+ `artifacts/family_aware/manifests/partial_input_policy_v1.json`.
**Created by:** `newstart_ai.data.build_partial_input_selections` (no `save_*()` of its own —
persistence added in `newstart_ai_mvp/stage5_build_conditions.py`), via
`python -m newstart_ai_mvp.stage5_build_conditions --run`.
**Note:** `PartialInputConfig` has no `output_dir` field of its own — selections share
`family_aware.conditions.output_dir`, matching the real on-disk layout.
**Regeneration stability:** deterministic — exact re-run.

## Condition registry
**Contains:** the shared source of truth mapping every (document, condition) pair to its
exact text — all 3 methods (BERT/LLM/LLM+RAG) read from this registry, never from a
per-method copy, guaranteeing identical input across methods.
**Files:** `data/family_aware_conditions/condition_registry_{train_validation,test}.csv`
(5,600 / 990 rows — 10 conditions × document count) + `manifests/condition_registry{,_test}_v1.json`.
**Created by:** `newstart_ai.data.build_condition_registry`, via
`python -m newstart_ai_mvp.stage5_build_conditions --run`.
**Regeneration stability:** deterministic — exact re-run.

## Agency class weights + document balancing
**Contains:** inverse-frequency class weights (corrects unequal document counts per agency)
and inverse-chunk-count weights (corrects unequal chunk multiplicity per document — the
519-chunk outlier document contributes the same total training weight as a 1-chunk document).
**Files:** `artifacts/family_aware/manifests/agency_class_weights_v1.json`,
`document_balancing_v1.json`, `document_balancing_verification_v1.json`.
**Created by:** `newstart_ai.models.bert.agency_class_weights` / `document_balancing`, inside
`python -m newstart_ai_mvp.train_bert --run-training`.
**Regeneration stability:** deterministic given a fixed train split.

## Selected BERT checkpoint + metadata
**Contains:** the fine-tuned `bert-base-uncased` weights (`checkpoint/`) and a full
provenance record (`metadata.json`): per-epoch train/validation loss and macro F1, class
weights, torch/CUDA versions, determinism warnings, selected epoch (2, by the
highest-validation-macro-F1/ earliest-tie rule).
**Files:** `artifacts/family_aware/models/<artifact_id>/{checkpoint/,metadata.json}`.
**Created by:** `newstart_ai.models.bert.family_aware_training.train_family_aware_bert` +
`family_aware_artifact.save_family_aware_artifact`, via `python -m newstart_ai_mvp.train_bert --run-training`.
**Config values:** `settings.family_aware.training` (max_epochs: 6, batch_size: 8,
learning_rate: 2e-5, random_seed: 42, checkpoint_selection_metric: validation_macro_f1).
**Used by:** `MVP/notebooks/02_bert_training_summary.ipynb`, `06_bert_error_attribution.ipynb`.
**Regeneration stability:** NOT guaranteed bit-identical — GPU nondeterminism is real and
already documented inside the frozen `metadata.json`'s own `deterministic_algorithms_warnings`
field (cuBLAS/attention-backward operations PyTorch cannot fully pin down even with
`torch.use_deterministic_algorithms(True)`). Same seed/config, small possible numeric drift.

## Condition evaluation (validation, 10-condition sweep)
**Contains:** the same 10-condition robustness sweep computed on the validation split, used
to confirm the retrieval/aggregation policy before it was ever applied to test.
**Files:** `artifacts/family_aware/manifests/condition_evaluation_v1.json`, `diagnostics_v1.json`.
**Created by:** `newstart_ai.models.bert.condition_evaluation.evaluate_all_conditions`, inside
`python -m newstart_ai_mvp.evaluate_bert`.

## BERT test predictions
**Contains:** all 990 (99 documents × 10 conditions) BERT test-set predictions, the primary
(complete_unmasked) result, error analysis, and a one-time freeze record proving nothing
about the test split influenced any earlier decision.
**Files:** `artifacts/family_aware/reports/checkpoint8_test_predictions.csv` +
`manifests/checkpoint8_{pre_test_freeze,primary_test_result,test_condition_sweep,test_error_analysis,test_integrity_proof,historical_comparison,test_reproducibility}_v1.json`.
**Created by:** `newstart_ai.models.bert.test_evaluation.*`, via
`python -m newstart_ai_mvp.evaluate_bert --run --i-understand-this-is-the-frozen-test-set`.
**Used by:** `MVP/notebooks/03_bert_test_evaluation.ipynb`, `06_bert_error_attribution.ipynb`.
**Regeneration stability:** same GPU-nondeterminism caveat as the checkpoint itself.

## Embedding config + corpus manifests
**Contains:** the Gemini embedding configuration fingerprint and per-corpus (masked/
unmasked) indexing manifests — chunk counts, fingerprints, embedding usage/cost.
**Files:** `artifacts/family_aware/manifests/checkpoint9_{embedding_config,corpus_manifest_masked,corpus_manifest_unmasked,embedding_usage_masked,embedding_usage_unmasked}_v1.json`.
**Created by:** `newstart_ai.rag.family_aware_embeddings` / `family_aware_index.build_family_aware_corpus_index`,
via `python -m newstart_ai_mvp.build_rag_index --rebuild-embeddings --rebuild-index`.

## Chroma vector stores
**Contains:** two separate collections (masked, unmasked), each holding 4,300 embedded
training chunks — **binary Chroma stores, not human-readable.**
**Files:** `artifacts/family_aware/vector_stores/routing_index_{masked,unmasked}/`.
**Created by:** `build_family_aware_corpus_index`.
**⚠ Rebuild warning:** this function calls `client.delete_collection()` on the target
collection name before recreating it. `newstart_ai_mvp.build_rag_index`'s expensive mode
always runs inside `redirect_frozen_outputs` for exactly this reason — never point a rebuild
at the real `persist_dir`.
**Regeneration stability:** embeddings are cached by `(task_type, sha256(text))`; a full
rebuild against the same chunks and embedding model produces the same vectors, but Gemini's
hosted embedding model is not guaranteed stable indefinitely (see Gemini predictions below).

## Retrieval diagnostics
**Contains:** retrieval-only quality metrics (top-k agency hit rate, mean reciprocal rank)
computed on validation data before the retrieval policy was frozen, and a diversification
before/after effect report.
**Files:** `artifacts/family_aware/reports/checkpoint9_validation_retrieval_results.csv` +
`manifests/checkpoint9_{validation_retrieval_diagnostics,diversification_policy,diversification_effect,rag_integrity_proof,cost_runtime_report}_v1.json`.
**Created by:** `newstart_ai.rag.family_aware_diagnostics`.

## Gemini (no-RAG) predictions
**Contains:** 990 plain-LLM classification results — model, prompt version, truncation flag,
tokens, latency, cost per case.
**Files:** `artifacts/family_aware/reports/checkpoint10_llm_predictions.jsonl` +
`artifacts/family_aware/llm_eval_cache/llm/*.json` (per-case cache).
**Created by:** `newstart_ai.models.llm.family_aware_evaluation.run_plain_llm_case`, via
`python -m newstart_ai_mvp.evaluate_llm --run-api`.
**Used by:** `MVP/notebooks/04_llm_evaluation_summary.ipynb`.
**Regeneration stability:** NOT guaranteed identical — Gemini is a hosted model; the exact
model version and its behavior can change even with an unchanged model name and prompt.

## Gemini+RAG predictions
**Contains:** 990 LLM+RAG classification results, plus per-case retrieved-chunk provenance
(rank, similarity, parent document — never shown to the model itself, which only ever sees
retrieved text with no labels or IDs).
**Files:** `artifacts/family_aware/reports/checkpoint10_llm_rag_predictions.jsonl` +
`.../llm_eval_cache/llm_rag/*.json`.
**Created by:** `run_llm_rag_case`, via `python -m newstart_ai_mvp.evaluate_rag --run-api`.
**Used by:** `MVP/notebooks/05_llm_rag_evaluation_summary.ipynb`.
**Regeneration stability:** same Gemini-hosted-model caveat as above.

## Method/condition metrics + statistical comparison
**Contains:** per-(method, condition) macro F1/accuracy/confusion matrices, the primary
paired comparison, the full 10-condition robustness comparison, and bootstrap/McNemar
statistical uncertainty estimates.
**Files:** `artifacts/family_aware/manifests/checkpoint10_{method_condition_metrics,primary_paired_comparison,robustness_comparison,statistical_uncertainty,cost_runtime_report,evaluation_integrity_proof,pre_evaluation_freeze}_v1.json`.
**Created by:** `newstart_ai.models.llm.family_aware_metrics` / `family_aware_integrity`,
inside `evaluate_llm.py` / `evaluate_rag.py`.
**Used by:** `python -m newstart_ai_mvp.compare_models`, `MVP/notebooks/07_final_comparison.ipynb`.

## Integrated Gradients outputs
**Contains:** word-level attribution scores explaining each of BERT's 6 test-set errors,
computed against the frozen checkpoint's own embedding layer.
**Files:** none — **there is no separate frozen IG artifact file.** Attributions are
computed fresh, deterministically, each time `06_bert_error_attribution.ipynb` runs, from the
frozen checkpoint and frozen predictions. This is intentional (IG is cheap local inference,
not worth caching) — do not read the absence of a file as a missing artifact.
**Created by:** `captum.attr.LayerIntegratedGradients` applied to the loaded, frozen
checkpoint inside the notebook itself; every prediction the notebook explains is
independently reproduced and checked against the real saved prediction before any
attribution is trusted (see the notebook's "Reproduce every prediction" section).
