"""newstart_ai_mvp -- a self-contained CLI package for the NewStart AI family-aware research
pipeline (Checkpoints 2-10): dataset validation, language filtering, family grouping, the
family-aware split, chunking, masking, condition registry, BERT training/evaluation, and
Gemini/Gemini+RAG evaluation.

Every research function this package needs lives inside it (config.py, data_pipeline.py,
bert_pipeline.py, llm_pipeline.py, rag_pipeline.py) -- nothing here imports from the original
project's `src/newstart_ai` package. It may still read the documented source dataset and
frozen research artifacts one directory up (configs/, data/, artifacts/), since those are
research inputs/outputs, not executable code.
"""

from __future__ import annotations
