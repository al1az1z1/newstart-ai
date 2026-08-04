"""Loads configs/*.yaml and .env into one typed Settings object.

This is the single entry point notebooks, the API, and services use to read configuration.
Nothing outside this module should open a YAML file directly -- that keeps dataset paths,
hyperparameters, and provider settings consistent everywhere they're used.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_yaml(relative_path: str) -> dict:
    path = PROJECT_ROOT / relative_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class DatasetConfig(BaseModel):
    path: str
    id_column: str
    text_column: str
    label_column: str


class SplitConfig(BaseModel):
    train: float
    validation: float
    test: float
    random_seed: int
    output_dir: str


class BeginningMiddleEndConfig(BaseModel):
    max_chunks: int
    chunk_overlap_tokens: int


class LongDocumentStrategyConfig(BaseModel):
    default: Literal["first_512", "beginning_middle_end"]
    max_tokens: int
    beginning_middle_end: BeginningMiddleEndConfig


class DemoConfig(BaseModel):
    # Set only after notebooks 04/06/08 compare validation macro F1 -- see
    # docs/BLUEPRINT.md Section 10. None means "not yet decided".
    default_routing_method: Literal["bert", "llm", "llm_rag"] | None = None


class BaseSettings(BaseModel):
    dataset: DatasetConfig
    labels: list[str]
    split: SplitConfig
    long_document_strategy: LongDocumentStrategyConfig
    demo: DemoConfig


class ImbalanceConfig(BaseModel):
    weighted_loss_threshold: float


class BertSettings(BaseModel):
    base_model: str
    max_epochs: int
    batch_size: int
    learning_rate: float
    checkpoint_selection_metric: str
    imbalance: ImbalanceConfig


class ValidationPassConfig(BaseModel):
    enabled: bool = True


class LlmSettings(BaseModel):
    provider: str
    model: str
    api_key_env: str
    endpoint: str | None = None
    prompt_version: str
    classification_prompt_path: str
    rag_classification_prompt_path: str
    guidance_prompt_dir: str
    validation_pass: ValidationPassConfig

    def resolve_api_key(self) -> str:
        """Reads the real API key from the env var named by api_key_env.

        Only the *name* of the variable is ever read from configuration or logged --
        never the key value.
        """
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env} is not set. "
                "Copy .env.example to .env and set it before calling the LLM provider."
            )
        return key


class RagSettings(BaseModel):
    vector_store: str
    persist_dir: str
    embedding_provider: str
    embedding_model: str
    top_k: int
    prompt_version: str
    classification_prompt_path: str


class LanguageFilterSettings(BaseModel):
    detector_name: str
    target_language: str
    min_text_length: int
    min_alphabetic_ratio: float
    confident_english_threshold: float
    confident_non_english_threshold: float
    mixed_language_window_count: int


class ChunkingConfig(BaseModel):
    tokenizer_revision: str
    max_seq_length: int
    num_special_tokens: int
    chunk_overlap_tokens: int
    chunking_policy_version: str
    output_dir: str


class AggregationConfig(BaseModel):
    default_method: Literal["mean_logits", "mean_probabilities", "majority_vote", "max_confidence"]
    candidate_methods: list[str]
    tie_breaker: str
    policy_version: str


class DocumentBalancingConfig(BaseModel):
    method: Literal["inverse_chunk_count_weight"]
    policy_version: str


class PartialInputConfig(BaseModel):
    chunks_per_region: int
    policy_version: str


class MaskingConfig(BaseModel):
    policy_version: str
    agency_name_placeholder: str
    form_number_placeholder: str
    omb_number_placeholder: str
    url_placeholder: str
    agency_identifier_phrases: dict[str, list[str]]
    output_dir: str


class ConditionRegistryConfig(BaseModel):
    policy_version: str
    names: list[str]
    output_dir: str


class TrainingConfig(BaseModel):
    max_epochs: int
    batch_size: int
    learning_rate: float
    random_seed: int
    checkpoint_selection_metric: str
    checkpoint_selection_aggregation_method: Literal["mean_logits", "mean_probabilities", "majority_vote", "max_confidence"]
    imbalance: ImbalanceConfig
    output_dir: str


class RetrievalPolicyConfig(BaseModel):
    candidate_pool_size: int
    top_k: int
    similarity_metric: str
    max_chunks_per_parent_document: int
    max_results_per_effective_family: int
    minimum_similarity_threshold: float | None
    tie_breaker: str
    policy_version: str


class FamilyAwareRagConfig(BaseModel):
    document_task_type: str
    query_task_type: str
    output_dimensionality: int | None
    embedding_batch_size: int
    max_retries: int
    retry_backoff_seconds: float
    cache_dir: str
    persist_dir_unmasked: str
    persist_dir_masked: str
    policy_version: str
    retrieval: RetrievalPolicyConfig


class EvaluationConfig(BaseModel):
    family_aware_rag_classification_prompt_path: str
    max_attempts: int
    retry_backoff_seconds: float
    retryable_error_substrings: list[str]
    cache_dir: str
    output_dir: str
    policy_version: str


class FamilyAwareSettings(BaseModel):
    """Version 6 family-aware chunked BERT robustness research (configs/family_aware.yaml).

    Additive and separate from `base`/`bert`/`llm`/`rag` -- nothing here ever changes the
    historical MVP's dataset, split, model, or RAG configuration. `language_filter` was
    added in Checkpoint 2, `split` in Checkpoint 4, `chunking` in Checkpoint 5,
    `aggregation`/`document_balancing`/`partial_input`/`masking`/`conditions` in Checkpoint 6.
    """

    language_filter: LanguageFilterSettings
    split: SplitConfig  # reuses the same shape as base.split -- train/validation/test/seed/output_dir
    chunking: ChunkingConfig
    aggregation: AggregationConfig
    document_balancing: DocumentBalancingConfig
    partial_input: PartialInputConfig
    masking: MaskingConfig
    conditions: ConditionRegistryConfig
    training: TrainingConfig
    rag: FamilyAwareRagConfig
    evaluation: EvaluationConfig


class Settings(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    project_root: Path
    base: BaseSettings
    bert: BertSettings
    llm: LlmSettings
    rag: RagSettings
    family_aware: FamilyAwareSettings

    def resolve_path(self, relative_path: str) -> Path:
        """Resolves a config-relative path (e.g. base.dataset.path) against the project root."""
        return self.project_root / relative_path


def load_settings() -> Settings:
    """Loads configs/*.yaml into one typed Settings object.

    Also loads .env (if present) so provider settings can read API keys via os.environ.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings(
        project_root=PROJECT_ROOT,
        base=BaseSettings(**_load_yaml("configs/base.yaml")),
        bert=BertSettings(**_load_yaml("configs/bert.yaml")),
        llm=LlmSettings(**_load_yaml("configs/llm.yaml")),
        rag=RagSettings(**_load_yaml("configs/rag.yaml")),
        family_aware=FamilyAwareSettings(**_load_yaml("configs/family_aware.yaml")),
    )
