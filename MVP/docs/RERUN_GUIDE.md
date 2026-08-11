# Rerunning a Stage

## Reproducing the submitted results (the default, always-safe path)

Every `newstart_ai_mvp` command, run with no flags, only loads and describes the real,
frozen artifacts under `artifacts/family_aware/` and `data/family_aware_*` — the same files
the report and presentation were built from. This is the authoritative results path:

```bash
python -m newstart_ai_mvp.compare_models
python -m newstart_ai_mvp.evaluate_llm      # recomputes macro F1 from the real predictions
python -m newstart_ai_mvp.train_bert         # reads the real training history
```

No flag here ever trains a model, calls an API, generates an embedding, or rebuilds a
Chroma index. See `docs/ARTIFACTS.md` for what each command reads.

## Repeating the experiment (a future, deliberate rerun)

A future user can explicitly rerun any stage. Every expensive command requires its own flag
and writes exclusively under `MVP/runs/<run-id>/` — the submitted experiment is never
touched.

```bash
python -m newstart_ai_mvp.prepare_data --run                    # stages 1-5, one run_id
python -m newstart_ai_mvp.train_bert --run-training
python -m newstart_ai_mvp.evaluate_bert --run --i-understand-this-is-the-frozen-test-set
python -m newstart_ai_mvp.build_rag_index --rebuild-embeddings --rebuild-index
python -m newstart_ai_mvp.evaluate_llm --run-api
python -m newstart_ai_mvp.evaluate_rag --run-api --use-frozen-index
```

Pass `--run-id <id>` to any command to reuse a specific run directory instead of a fresh
timestamp; pass `--input-run-id <id>` to a downstream stage to chain it onto a previous
stage's own output from the same run, rather than reading the frozen upstream artifacts
(`prepare_data --run` does this automatically for stages 1-5).

### What's guaranteed to reproduce exactly

The deterministic, CPU-only, non-hosted-model stages — split, chunking, masking, condition
registry — use fixed seeds and no external service, so a rerun reproduces byte-identical
output.

### What's not guaranteed to reproduce exactly

- **BERT training**: same seed, split, class weights, optimizer, learning rate, batch size,
  epochs, and checkpoint-selection rule as the frozen run — but GPU operations (cuBLAS,
  attention backward) can introduce small numerical differences even with
  `torch.use_deterministic_algorithms(True)`. The frozen checkpoint's own `metadata.json`
  already records which specific operations PyTorch warned about
  (`deterministic_algorithms_warnings`).
- **Gemini / Gemini+RAG**: hosted models. The model name and prompt stay fixed, but the
  API's actual behavior is not guaranteed identical over time. A rerun is a new execution of
  the same methodology, not a replacement for the frozen, submitted predictions.
- **`evaluate_llm --run-api` / `evaluate_rag --run-api` start with an empty cache** under the
  run directory — they never silently reuse the frozen, submitted cache. This is
  intentional: reusing it would make a "rerun" partly the old run in disguise.

### Never overwritten

`redirect_frozen_outputs()` (`newstart_ai_mvp/run_scope.py`) patches every write path for
the duration of one CLI invocation so it's structurally impossible for an expensive command
to write into `artifacts/family_aware/` or `data/family_aware_*` — including the two cases
that would otherwise silently do so: `save_family_audit`'s hardcoded path literals, and
`build_family_aware_corpus_index`'s `delete_collection()` call on the target Chroma
collection.
