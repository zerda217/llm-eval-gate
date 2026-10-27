"""Run a task against a provider and score it."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import registry
from .config import Config, ConfigError, ProviderConfig
from .dataset import Case, check_placeholders, load_cases, render
from .providers import CachedProvider, FileProvider, GeminiProvider, Provider
from .registry import MetricContext
from .result import CaseRecord, MetricOutcome, MetricResult, RunResult

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def build_provider(cfg: ProviderConfig, mode: str = "cached") -> Provider:
    if cfg.type == "file":
        return FileProvider(cfg.outputs)  # type: ignore[arg-type]
    provider: Provider = GeminiProvider(cfg.model, temperature=cfg.temperature, rpm=cfg.rpm, json_mode=cfg.json_mode)  # type: ignore[arg-type]
    if cfg.cache_dir is not None:
        provider = CachedProvider(provider, cfg.cache_dir, mode)
    elif mode == "replay":
        raise ConfigError("--mode replay needs provider.cache_dir")
    return provider


def parse_output(text: str) -> tuple[Any, str | None]:
    """Parse model text as JSON, tolerating a ```json fence. Returns (value, error)."""
    match = _FENCE.match(text)
    candidate = match.group(1) if match else text
    if not candidate.strip():
        return None, "empty output"
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as exc:
        return None, f"{exc.msg} (line {exc.lineno}, col {exc.colno})"


def evaluate_cases(cases: list[Case], config: Config, provider: Provider) -> list[CaseRecord]:
    task = config.task
    check_placeholders(cases, task.placeholders)
    records = []
    for case in cases:
        prompt = render(task.prompt_template, case.input)
        gen = provider.generate(case.id, task.system_prompt, prompt)
        parsed, error = parse_output(gen.text)
        records.append(
            CaseRecord(id=case.id, tags=case.tags, expected=case.expected, output_text=gen.text, parsed=parsed, parse_error=error, generation=gen)
        )
    return records


def score(
    records: list[CaseRecord],
    config: Config,
    *,
    baseline: dict[str, float] | None = None,
    fail_on_warn: bool = False,
) -> RunResult:
    ctx = MetricContext(output_schema=config.task.output_schema)
    results = []
    for m in config.metrics:
        spec = registry.get(m.type)
        try:
            outcome = spec.fn(records, m.params, ctx)
        except Exception as exc:  # noqa: BLE001 - a crashing metric must not pass silently
            outcome = MetricOutcome(value=0.0, detail=f"metric raised {type(exc).__name__}: {exc}")
        results.append(
            MetricResult(
                name=m.name,
                metric_type=m.type,
                severity=m.severity,
                value=outcome.value,
                minimum=m.minimum,
                baseline=(baseline or {}).get(m.name),
                max_regression=config.gate.max_regression,
                problems=outcome.problems,
                detail=outcome.detail,
            )
        )
    return RunResult(task=config.task.name, records=records, metrics=results, fail_on_warn=fail_on_warn)


def run(config: Config, provider: Provider, *, baseline: dict[str, float] | None = None, fail_on_warn: bool = False) -> RunResult:
    cases = load_cases(config.task.dataset)
    records = evaluate_cases(cases, config, provider)
    return score(records, config, baseline=baseline, fail_on_warn=fail_on_warn)


def load_baseline(path: Path | None) -> dict[str, float] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"baseline {path} is not valid JSON: {exc}") from None
    metrics = data.get("metrics") if isinstance(data, dict) else None
    if not isinstance(metrics, dict):
        raise ConfigError(f"baseline {path} must look like {{\"metrics\": {{name: value}}}}")
    return {str(k): float(v) for k, v in metrics.items()}


def save_baseline(path: Path, run_result: RunResult, provider: Provider) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"task": run_result.task, "provider": provider.describe(), "metrics": run_result.metric_values()}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
