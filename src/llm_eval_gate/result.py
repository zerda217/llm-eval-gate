"""Result types shared by metrics, the runner and reporters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    ERROR = "error"   # blocks the gate
    WARN = "warn"     # reported, blocks only with --fail-on-warn


class ExitCode:
    """Process exit codes, kept identical to pipeguard so pipelines treat both the same way."""

    OK = 0             # every error-level metric passed
    GATE_FAILED = 1    # an error-level metric is below its minimum or regressed
    CONFIG_ERROR = 2   # the YAML config (or a referenced file) is invalid
    RUNTIME_ERROR = 3  # dataset unreadable, model call failed, cache miss in replay mode


@dataclass(frozen=True)
class Generation:
    """One model response plus the usage data needed for cost and latency tracking."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0.0
    cached: bool = False
    finish_reason: str | None = None


@dataclass(frozen=True)
class CaseRecord:
    """Everything known about one evaluated case."""

    id: str
    tags: list[str]
    expected: Any
    output_text: str
    parsed: Any = None
    parse_error: str | None = None
    generation: Generation | None = None

    @property
    def valid_json(self) -> bool:
        return self.parse_error is None


@dataclass(frozen=True)
class MetricOutcome:
    """What a metric function returns: an aggregate value and per-case problems."""

    value: float
    problems: dict[str, list[str]] = field(default_factory=dict)  # case_id -> problem descriptions
    detail: str = ""


@dataclass(frozen=True)
class MetricResult:
    name: str
    metric_type: str
    severity: Severity
    value: float
    minimum: float | None
    baseline: float | None
    max_regression: float | None
    problems: dict[str, list[str]] = field(default_factory=dict)
    detail: str = ""

    @property
    def below_min(self) -> bool:
        return self.minimum is not None and self.value < self.minimum

    @property
    def regressed(self) -> bool:
        if self.baseline is None or self.max_regression is None:
            return False
        return self.value < self.baseline - self.max_regression - 1e-9

    @property
    def passed(self) -> bool:
        return not (self.below_min or self.regressed)

    @property
    def status(self) -> str:
        if self.passed:
            return "pass"
        return "fail" if self.severity is Severity.ERROR else "warn"

    @property
    def reasons(self) -> list[str]:
        out = []
        if self.below_min:
            out.append(f"below min {self.minimum:.3f}")
        if self.regressed:
            out.append(f"regressed from baseline {self.baseline:.3f} (allowed -{self.max_regression:.3f})")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.metric_type,
            "severity": self.severity.value,
            "status": self.status,
            "value": round(self.value, 6),
            "min": self.minimum,
            "baseline": self.baseline,
            "max_regression": self.max_regression,
            "reasons": self.reasons,
            "detail": self.detail,
            "failed_cases": len(self.problems),
        }


@dataclass(frozen=True)
class RunResult:
    task: str
    records: list[CaseRecord]
    metrics: list[MetricResult]
    fail_on_warn: bool = False

    @property
    def errors(self) -> list[MetricResult]:
        return [m for m in self.metrics if m.status == "fail"]

    @property
    def warnings(self) -> list[MetricResult]:
        return [m for m in self.metrics if m.status == "warn"]

    @property
    def passed(self) -> bool:
        if self.errors:
            return False
        return not (self.fail_on_warn and self.warnings)

    @property
    def exit_code(self) -> int:
        return ExitCode.OK if self.passed else ExitCode.GATE_FAILED

    def case_problems(self) -> dict[str, list[str]]:
        """Merge problems from all metrics, keeping each distinct message once, in case order."""
        merged: dict[str, list[str]] = {}
        for metric in self.metrics:
            for case_id, msgs in metric.problems.items():
                bucket = merged.setdefault(case_id, [])
                bucket.extend(m for m in msgs if m not in bucket)
        order = {r.id: i for i, r in enumerate(self.records)}
        return dict(sorted(merged.items(), key=lambda kv: order.get(kv[0], 0)))

    def usage(self) -> dict[str, Any]:
        gens = [r.generation for r in self.records if r.generation is not None]
        live = [g for g in gens if not g.cached]
        latencies = sorted(g.latency_ms for g in live)
        return {
            "calls": len(gens),
            "cache_hits": len(gens) - len(live),
            "input_tokens": sum(g.input_tokens or 0 for g in gens),
            "output_tokens": sum(g.output_tokens or 0 for g in gens),
            "latency_p50_ms": _percentile(latencies, 50),
            "latency_p95_ms": _percentile(latencies, 95),
        }

    def by_tag(self) -> dict[str, tuple[int, int]]:
        """tag -> (cases with problems, total cases)."""
        problems = self.case_problems()
        out: dict[str, list[int]] = {}
        for r in self.records:
            for tag in r.tags:
                bucket = out.setdefault(tag, [0, 0])
                bucket[1] += 1
                bucket[0] += r.id in problems
        return {k: (v[0], v[1]) for k, v in sorted(out.items())}

    def metric_values(self) -> dict[str, float]:
        return {m.name: round(m.value, 6) for m in self.metrics}

    def to_dict(self) -> dict[str, Any]:
        problems = self.case_problems()
        return {
            "task": self.task,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "cases": len(self.records),
            "metrics": [m.to_dict() for m in self.metrics],
            "usage": self.usage(),
            "by_tag": {k: {"failed": f, "total": t} for k, (f, t) in self.by_tag().items()},
            "failed_cases": [
                {"id": r.id, "tags": r.tags, "problems": problems[r.id], "output": r.output_text, "expected": r.expected}
                for r in self.records
                if r.id in problems
            ],
        }


def _percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    k = max(0, min(len(values) - 1, round(pct / 100 * (len(values) - 1))))
    return round(values[k], 1)
