"""Mechanically enforces the self-contained requirement: recursively scans every MVP/ Python
file (excluding this test itself, which legitimately mentions the banned patterns as string
literals) and every notebook code cell for a runtime dependency on the original project's
`src/newstart_ai` package -- `import newstart_ai`/`from newstart_ai import ...` (not
`newstart_ai_mvp`, which is this package itself), or `sys.path` manipulation used to expose
it. Fails with the exact file and line if either is found.

Reading the documented source dataset or frozen research artifacts (configs/, data/,
artifacts/ -- plain data files, not executable code) from one directory above MVP/ is fine
and expected; this test only flags imports of executable code and path-injection tricks.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

BANNED_IMPORT_MODULE_PREFIX = "newstart_ai"  # but NOT "newstart_ai_mvp" -- see _is_banned_import


def _is_banned_import(module_name: str | None) -> bool:
    if module_name is None:
        return False
    return module_name == BANNED_IMPORT_MODULE_PREFIX or module_name.startswith(BANNED_IMPORT_MODULE_PREFIX + ".")


def _scan_source_for_violations(source: str, label: str) -> list[str]:
    violations: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [f"{label}: could not parse as Python (syntax error) -- skipped"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_banned_import(alias.name):
                    violations.append(f"{label}:{node.lineno}: `import {alias.name}` -- runtime dependency on the original source")
        elif isinstance(node, ast.ImportFrom):
            if _is_banned_import(node.module):
                violations.append(f"{label}:{node.lineno}: `from {node.module} import ...` -- runtime dependency on the original source")

    # sys.path manipulation: flag any attribute access on sys.path (insert/append/extend/etc.)
    # -- the specific pattern this refactor was asked to eliminate everywhere in MVP/.
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name) and node.value.value.id == "sys" and node.value.attr == "path":
                violations.append(f"{label}:{node.lineno}: `sys.path.{node.attr}(...)` -- path injection is not allowed in MVP/")

    return violations


def _mvp_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _python_files() -> list[Path]:
    root = _mvp_root()
    return [
        p for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and p != Path(__file__).resolve()
    ]


def _notebook_files() -> list[Path]:
    return list((_mvp_root() / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(_mvp_root())))
def test_python_file_has_no_original_source_dependency(path: Path):
    source = path.read_text(encoding="utf-8")
    violations = _scan_source_for_violations(source, str(path.relative_to(_mvp_root())))
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("path", _notebook_files(), ids=lambda p: p.name)
def test_notebook_code_cells_have_no_original_source_dependency(path: Path):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    violations: list[str] = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        violations.extend(_scan_source_for_violations(source, f"{path.name} cell {i}"))

    assert not violations, "\n".join(violations)


def test_banned_pattern_detector_actually_catches_violations():
    """Proof the scanner itself works -- not a vacuously-passing check."""
    assert _scan_source_for_violations("import newstart_ai", "x")
    assert _scan_source_for_violations("from newstart_ai.config import settings", "x")
    assert _scan_source_for_violations("from newstart_ai import data", "x")
    assert _scan_source_for_violations("import sys\nsys.path.insert(0, 'x')", "x")
    assert not _scan_source_for_violations("import newstart_ai_mvp\nfrom newstart_ai_mvp import config", "x")
    assert not _scan_source_for_violations("import sys\nprint(sys.argv)", "x")
