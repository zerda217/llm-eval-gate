"""Read an eval dataset (JSON Lines) and render prompts from it.

Each line::

    {"id": "c001", "input": {"conversation": "..."}, "expected": {...}, "tags": ["relative_date"]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PLACEHOLDER


class DatasetError(Exception):
    pass


@dataclass(frozen=True)
class Case:
    id: str
    input: dict[str, Any]
    expected: Any
    tags: list[str] = field(default_factory=list)


def load_cases(path: Path) -> list[Case]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise DatasetError(f"dataset not found: {path}") from None

    cases, seen = [], set()
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{path.name}:{lineno}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{where}: invalid JSON ({exc.msg})") from None
        if not isinstance(row, dict) or "id" not in row or "input" not in row or "expected" not in row:
            raise DatasetError(f"{where}: each case needs 'id', 'input' and 'expected'")
        if not isinstance(row["input"], dict):
            raise DatasetError(f"{where}: 'input' must be an object")
        case_id = str(row["id"])
        if case_id in seen:
            raise DatasetError(f"{where}: duplicate case id '{case_id}'")
        seen.add(case_id)
        tags = row.get("tags") or []
        if not isinstance(tags, list):
            raise DatasetError(f"{where}: 'tags' must be a list")
        cases.append(Case(id=case_id, input=row["input"], expected=row["expected"], tags=[str(t) for t in tags]))

    if not cases:
        raise DatasetError(f"dataset is empty: {path}")
    return cases


def check_placeholders(cases: list[Case], placeholders: set[str]) -> None:
    """Fail before any model call if a case lacks a field the prompt needs."""
    for case in cases:
        missing = sorted(placeholders - set(case.input))
        if missing:
            raise DatasetError(f"case '{case.id}' is missing input field(s) used by the prompt: {', '.join(missing)}")


def render(template: str, values: dict[str, Any]) -> str:
    def sub(match) -> str:
        value = values[match.group(1)]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    return PLACEHOLDER.sub(sub, template)
