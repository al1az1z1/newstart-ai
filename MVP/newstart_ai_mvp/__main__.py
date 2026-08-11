"""`python -m newstart_ai_mvp` -- prints the command menu. Does nothing expensive."""

from __future__ import annotations

MENU = """
newstart_ai_mvp -- CLI entry points for the NewStart AI family-aware research pipeline.

Every command below is SAFE by default: it only describes the real, frozen, submitted
artifacts (no training, no API calls, no file writes). Pass the stage's expensive flag to
actually (re)run it -- output always lands under MVP/runs/<run-id>/, never overwriting the
submitted experiment.

  Data preparation (Checkpoints 2-6):
    python -m newstart_ai_mvp.stage1_validate_and_audit   [--run]
    python -m newstart_ai_mvp.stage2_build_split           [--run]
    python -m newstart_ai_mvp.stage3_build_chunks          [--run]
    python -m newstart_ai_mvp.stage4_build_masked          [--run]
    python -m newstart_ai_mvp.stage5_build_conditions      [--run]
    python -m newstart_ai_mvp.prepare_data                 [--run]   (runs stage1-5 as one)

  BERT (Checkpoint 7-8):
    python -m newstart_ai_mvp.train_bert                   [--run-training]
    python -m newstart_ai_mvp.evaluate_bert                [--run --artifact-id ID --i-understand-this-is-the-frozen-test-set]

  RAG + Gemini (Checkpoints 9-10):
    python -m newstart_ai_mvp.build_rag_index              [--rebuild-embeddings --rebuild-index]
    python -m newstart_ai_mvp.evaluate_llm                 [--run-api]
    python -m newstart_ai_mvp.evaluate_rag                 [--run-api --use-frozen-index | --index-run-id ID]

  Comparison (always safe):
    python -m newstart_ai_mvp.compare_models               [--out csv_path]

See MVP/docs/STAGES.md for what each command produces and MVP/docs/ARTIFACTS.md for every
frozen artifact's provenance.
"""

if __name__ == "__main__":
    print(MENU)
