# The Three-Stage Model

This project did not begin with pre-existing results. During the original experiment the
team implemented and executed 14 computational steps, once, end to end:

1. Loaded the processed dataset (`data/processed/final_dataset.csv`).
2. Validated, language-filtered, and family-grouped the documents.
3. Built the frozen family-aware split.
4. Divided documents into tokenizer-aware overlapping chunks.
5. Built masked/unmasked representations and the ten registered conditions.
6. Trained the family-aware BERT model.
7. Used validation performance to select the earliest checkpoint with the highest macro F1.
8. Saved the selected BERT checkpoint.
9. Ran Gemini classification.
10. Generated Gemini embeddings.
11. Built the masked and unmasked training-only Chroma indexes.
12. Ran Gemini+RAG classification.
13. Saved predictions, retrieval information, metrics, and manifests.
14. Analyzed the frozen outputs to produce the report and presentation.

Every one of those 14 steps is real code, still present, still executable, and still the
same code that produced the artifacts this MVP loads. Nothing here is a substitute for
missing implementation. `MVP/` organizes this lifecycle into three stages so it's possible to
see, for any given step, exactly which code ran, what it produced, and how to rerun it.

## Stage 1 — Artifact creation

The real implementation of every step above, vendored into `MVP/newstart_ai_mvp/` as its own
self-contained copy — simplified and reorganized from the original `newstart_ai_benchmark/src/newstart_ai/`
research code, but preserving its behavior and methodology. `MVP/` does not import executable
code from `newstart_ai_benchmark/` at runtime; the original package remains read-only reference
material documenting where this logic came from. Every stage module's expensive mode is real,
executable code — never pseudocode, a placeholder, or a hardcoded output — but this cleanup
task never triggers any of it.

| Lifecycle step | CLI module | Real functions (originally `newstart_ai_benchmark/src/newstart_ai/...`) |
|---|---|---|
| 2. Validate / filter / group | `stage1_validate_and_audit.py` | `data.validate_dataset`, `data.build_language_audit`, `data.build_full_family_audit` |
| 3. Frozen split | `stage2_build_split.py` | `data.create_family_aware_split`, 4 leakage assertions |
| 4. Chunking | `stage3_build_chunks.py` | `data.build_all_split_chunks`, 9 invariant assertions |
| 5. Masking + conditions | `stage4_build_masked.py`, `stage5_build_conditions.py` | `data.build_masked_documents`, `data.build_condition_registry` |
| 6-8. BERT training + selection + save | `train_bert.py` | `models.bert.family_aware_training.train_family_aware_bert`, `family_aware_artifact.save_family_aware_artifact` |
| 9. Gemini classification | `evaluate_llm.py` | `models.llm.family_aware_evaluation.run_plain_llm_case` |
| 10-11. Embeddings + Chroma | `build_rag_index.py` | `rag.family_aware_embeddings`, `rag.family_aware_index.build_family_aware_corpus_index` |
| 12. Gemini+RAG classification | `evaluate_rag.py` | `models.llm.family_aware_evaluation.run_llm_rag_case` |
| 13. Metrics/manifests | `evaluate_llm.py` / `evaluate_rag.py` | `models.llm.family_aware_metrics` |

## Stage 2 — Frozen research evaluation

The primary way this MVP is actually used. Every CLI module's *default* mode, and every
notebook in `MVP/notebooks/`, loads the real frozen artifacts (see `docs/ARTIFACTS.md`),
verifies their schemas/counts/provenance, and recomputes metrics directly from raw
predictions with sklearn — printed next to the value already in the saved manifest, so
agreement is checked every time, not assumed. This never retrains BERT, never calls Gemini,
never generates embeddings, never rebuilds Chroma, and never overwrites a saved prediction.

## Stage 3 — Optional future reproduction

Every CLI module also has an *expensive* mode, gated behind an explicit flag
(`--run`, `--run-training`, `--run-api`, `--rebuild-embeddings --rebuild-index`). It reuses
the same seed, split, condition registry, label mapping, base model, class-weight
calculation, document-balancing method, optimizer, learning rate, batch size, max epochs,
validation metric, and earliest-best-epoch rule as the original run — nothing about the
methodology changes for a rerun. Output always lands under `MVP/runs/<run-id>/`, never in
`artifacts/family_aware/` or `data/family_aware_*`. See `docs/RERUN_GUIDE.md`.
