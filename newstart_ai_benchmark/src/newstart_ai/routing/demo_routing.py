"""Resolves which method's prediction routes the Random Form Routing Demo to a
GuidanceAgent. Pure wiring around a configuration value -- no classification or evaluation
logic lives here (docs/BLUEPRINT.md Section 10).
"""

from __future__ import annotations

from newstart_ai.config.settings import Settings
from newstart_ai.schemas.classification import AgencyLabel, Method


def resolve_default_routing_method(settings: Settings) -> Method:
    """Returns the demo's configured default routing method.

    This value is resolved by comparing BERT/LLM/LLM+RAG validation macro F1
    (`08_llm_rag_evaluation.ipynb`) -- it is never hard-coded to a particular method here.
    """
    method = settings.base.demo.default_routing_method
    if method is None:
        raise RuntimeError(
            "configs/base.yaml: demo.default_routing_method is not set. Run notebooks "
            "04 through 08 first so it can be resolved from validation macro F1."
        )
    return method


def select_routed_agency(
    predictions: dict[Method, AgencyLabel], settings: Settings
) -> tuple[Method, AgencyLabel]:
    """Given each method's predicted label, returns (chosen_method, routed_agency) using
    the configured default routing method."""
    method = resolve_default_routing_method(settings)
    return method, predictions[method]
