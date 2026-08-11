"""Shared argparse scaffolding used by every newstart_ai_mvp stage module.

Each stage module still owns its own dispatch logic (what "expensive mode" actually does
differs per stage), but they all share the same --run-id flag, the same safe/expensive
banners, and the same "load settings once" pattern.
"""

from __future__ import annotations

import argparse

from newstart_ai_mvp.config import Settings, load_settings


def build_base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Reuse an existing MVP/runs/<run-id> directory instead of creating a new one.",
    )
    parser.add_argument(
        "--input-run-id",
        default=None,
        help=(
            "Read this stage's upstream input from a previous run's own output "
            "(MVP/runs/<input-run-id>/...) instead of the frozen artifacts. "
            "Omit to read the frozen, submitted artifacts."
        ),
    )
    return parser


def print_safe_mode_banner(stage_name: str) -> None:
    print(f"[{stage_name}] Safe mode (default): describing the real, frozen artifact only.")
    print(f"[{stage_name}] No file is written, no model is loaded, no API is called.")
    print(f"[{stage_name}] Pass the stage's expensive flag to actually (re)run this stage.\n")


def print_expensive_mode_banner(stage_name: str, run_id: str) -> None:
    print(f"[{stage_name}] Expensive mode: running the real pipeline stage.")
    print(f"[{stage_name}] Output will be written under MVP/runs/{run_id}/ -- the frozen,")
    print(f"[{stage_name}] submitted artifacts under artifacts/family_aware/ and data/family_aware_*")
    print(f"[{stage_name}] are never touched.\n")


def get_settings() -> Settings:
    return load_settings()
