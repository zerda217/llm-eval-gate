"""Built-in metrics.

All metrics return a value in [0, 1] where higher is better, so every gate
rule reads the same way: ``value >= min`` and ``value >= baseline - max_regression``.

Invalid JSON never crashes a metric: it is scored as an empty answer (fail closed
for recall-type metrics) and reported once by ``json_valid_rate``.
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from .registry import MetricContext, ParamError, register
from .result import CaseRecord, MetricOutcome

_MISSING = object()


# ---------------------------------------------------------------- helpers


def get_path(obj: Any, path: str) -> Any:
    """Read 'a.b.0.c' from nested dicts/lists. Returns _MISSING when absent."""
    if path in ("", "."):
        return obj
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return _MISSING
    return cur


def _norm(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    return value


def _key_set(container: Any, path: str, keys: list[str]) -> set[tuple]:
    items = get_path(container, path)
    if not isinstance(items, list):
        return set()
    out = set()
    for item in items:
        if isinstance(item, dict):
            out.add(tuple(_norm(item.get(k)) for k in keys))
    return out


def _fmt(key: tuple) -> str:
    return "(" + ", ".join("null" if v is None else str(v) for v in key) + ")"


def _ratio(num: int, den: int) -> float:
    return 1.0 if den == 0 else num / den


def _validate_set_params(params: dict[str, Any]) -> None:
    keys = params["keys"]
    if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys):
        raise ParamError("'keys' must be a non-empty list of field names")
    if not isinstance(params["path"], str):
        raise ParamError("'path' must be a string such as 'events'")


# ---------------------------------------------------------------- metrics


@register("json_valid_rate")
def json_valid_rate(records: list[CaseRecord], params: dict[str, Any], ctx: MetricContext) -> MetricOutcome:
    """Share of outputs that parse as JSON."""
    problems = {r.id: [f"invalid JSON: {r.parse_error}"] for r in records if not r.valid_json}
    ok = len(records) - len(problems)
    return MetricOutcome(_ratio(ok, len(records)), problems, f"{ok}/{len(records)} parsed")


@register("schema_valid_rate", needs_schema=True)
def schema_valid_rate(records: list[CaseRecord], params: dict[str, Any], ctx: MetricContext) -> MetricOutcome:
    """Share of outputs that match task.output_schema (invalid JSON counts as invalid)."""
    validator = Draft202012Validator(ctx.output_schema or {})
    problems: dict[str, list[str]] = {}
    for r in records:
        if not r.valid_json:
            continue  # already reported by json_valid_rate; still counted as invalid below
        errors = sorted(validator.iter_errors(r.parsed), key=lambda e: list(e.path))
        if errors:
            e = errors[0]
            where = "/".join(map(str, e.path)) or "<root>"
            problems[r.id] = [f"schema: {where}: {e.message}"]
    ok = sum(1 for r in records if r.valid_json and r.id not in problems)
    return MetricOutcome(_ratio(ok, len(records)), problems, f"{ok}/{len(records)} valid")


@register("exact_match_rate", required=("path",))
def exact_match_rate(records: list[CaseRecord], params: dict[str, Any], ctx: MetricContext) -> MetricOutcome:
    """Share of cases where the value at 'path' equals the expected value (classification tasks)."""
    path = params["path"]
    problems = {}
    for r in records:
        got = get_path(r.parsed, path) if r.valid_json else _MISSING
        want = get_path(r.expected, path)
        if _norm(got) != _norm(want) or got is _MISSING:
            shown = "<missing>" if got is _MISSING else repr(got)
            problems[r.id] = [f"{path}: expected {want!r}, got {shown}"]
    ok = len(records) - len(problems)
    return MetricOutcome(_ratio(ok, len(records)), problems, f"{ok}/{len(records)} matched")


@register("set_precision", required=("path", "keys"), validate=_validate_set_params)
def set_precision(records: list[CaseRecord], params: dict[str, Any], ctx: MetricContext) -> MetricOutcome:
    """Of all predicted items (identified by 'keys'), the share that are expected. Micro-averaged."""
    tp = fp = 0
    problems: dict[str, list[str]] = {}
    for r in records:
        pred = _key_set(r.parsed, params["path"], params["keys"]) if r.valid_json else set()
        gold = _key_set(r.expected, params["path"], params["keys"])
        extra = pred - gold
        tp += len(pred & gold)
        fp += len(extra)
        if extra:
            problems[r.id] = [f"unexpected {_fmt(k)}" for k in sorted(extra, key=str)]
    return MetricOutcome(_ratio(tp, tp + fp), problems, f"{tp} correct / {tp + fp} predicted")


@register("set_recall", required=("path", "keys"), validate=_validate_set_params)
def set_recall(records: list[CaseRecord], params: dict[str, Any], ctx: MetricContext) -> MetricOutcome:
    """Of all expected items (identified by 'keys'), the share that were predicted. Micro-averaged."""
    tp = fn = 0
    problems: dict[str, list[str]] = {}
    for r in records:
        pred = _key_set(r.parsed, params["path"], params["keys"]) if r.valid_json else set()
        gold = _key_set(r.expected, params["path"], params["keys"])
        missed = gold - pred
        tp += len(pred & gold)
        fn += len(missed)
        if missed:
            problems[r.id] = [f"missed {_fmt(k)}" for k in sorted(missed, key=str)]
    return MetricOutcome(_ratio(tp, tp + fn), problems, f"{tp} found / {tp + fn} expected")


@register("empty_when_expected_rate", required=("path",))
def empty_when_expected_rate(records: list[CaseRecord], params: dict[str, Any], ctx: MetricContext) -> MetricOutcome:
    """Among cases whose expected list at 'path' is empty, the share where the output is also empty."""
    path = params["path"]
    targets = [r for r in records if get_path(r.expected, path) == []]
    problems = {}
    for r in targets:
        got = get_path(r.parsed, path) if r.valid_json else _MISSING
        if got != []:
            problems[r.id] = ["expected no items, got " + ("<invalid>" if got is _MISSING else f"{len(got) if isinstance(got, list) else got!r}")]
    ok = len(targets) - len(problems)
    return MetricOutcome(_ratio(ok, len(targets)), problems, f"{ok}/{len(targets)} correctly empty")
