"""Loads versioned prompt files (docs/BLUEPRINT.md Section 7; prompts/ directory).

Prompts live outside application code so they can be versioned, reviewed, and frozen
independently of the services that use them.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from newstart_ai.config.settings import Settings


class PromptTemplate(BaseModel):
    version: str
    created_date: str
    purpose: str
    system_prompt: str
    user_template: str
    # Classification-only fields -- guidance prompts (free-text answers, not a label
    # decision) leave these unset.
    allowed_labels: list[str] | None = None
    response_schema: dict | None = None
    change_notes: str = ""

    def render(self, **placeholders: str) -> str:
        """Fills named placeholders (e.g. {text}, {context}) using plain replace, not
        str.format -- document text may itself contain literal curly braces (form field
        examples, etc.), which would otherwise break str.format."""
        message = self.user_template
        for key, value in placeholders.items():
            message = message.replace("{" + key + "}", value)
        return message

    def render_user_message(self, text: str) -> str:
        return self.render(text=text)


def load_prompt(path: Path) -> PromptTemplate:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PromptTemplate(**raw)


def load_classification_prompt(settings: Settings) -> PromptTemplate:
    return load_prompt(settings.resolve_path(settings.llm.classification_prompt_path))


def load_rag_classification_prompt(settings: Settings) -> PromptTemplate:
    return load_prompt(settings.resolve_path(settings.llm.rag_classification_prompt_path))


def load_family_aware_rag_classification_prompt(settings: Settings) -> PromptTemplate:
    """Version 6 Checkpoint 10's RAG prompt -- retrieved excerpts carry no agency label,
    unlike the historical rag_classification/v1.yaml. Kept fully separate."""
    return load_prompt(settings.resolve_path(settings.family_aware.evaluation.family_aware_rag_classification_prompt_path))
