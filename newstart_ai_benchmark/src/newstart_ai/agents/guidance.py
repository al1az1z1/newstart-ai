"""Simple per-agency guidance agents for the Random Form Routing Demo.

This is explicitly a demonstration, not the research question (docs/BLUEPRINT.md Section
10): answer quality is not evaluated here, and the routing RAG index is never reused as an
answer-generation knowledge base. Classification and guidance generation stay
architecturally separate -- this module never decides which agency a document belongs to,
it only responds once that decision has already been made elsewhere.
"""

from __future__ import annotations

from newstart_ai.config.settings import Settings
from newstart_ai.models.llm.prompts import load_prompt
from newstart_ai.models.llm.provider import LLMProvider
from newstart_ai.schemas.classification import AgencyLabel, GuidanceResult

# One prompt file per agency, using the same minimal pattern with only the agency name
# changed (docs/BLUEPRINT.md Section 10).
_AGENCY_PROMPT_FILES: dict[str, str] = {
    "USCIS": "uscis.yaml",
    "DMV": "dmv.yaml",
    "SSA": "ssa.yaml",
    "IRS": "irs.yaml",
}


def load_guidance_prompt(settings: Settings, agency: AgencyLabel):
    directory = settings.resolve_path(settings.llm.guidance_prompt_dir)
    return load_prompt(directory / _AGENCY_PROMPT_FILES[agency])


class GuidanceAgent:
    """Wraps an LLMProvider with the four per-agency prompts to produce one short demo
    answer. Never used for routing/classification -- only called after a routing method has
    already produced a predicted agency."""

    def __init__(self, llm_provider: LLMProvider, settings: Settings):
        self.llm_provider = llm_provider
        self.settings = settings

    def respond(self, text: str, agency: AgencyLabel) -> GuidanceResult:
        prompt = load_guidance_prompt(self.settings, agency)
        return self.llm_provider.generate_guidance(text, agency, prompt)
