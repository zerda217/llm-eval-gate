"""Load and validate an eval config.

Everything that can be checked without calling a model is checked here, so a
typo costs milliseconds and zero API quota (exit code 2).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from . import metrics as _builtin_metrics  # noqa: F401  (registers built-in metrics)
from . import registry
from .result import Severity

PROVIDERS = {"gemini", "file"}
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_RESERVED = {"name", "type", "severity", "min", "description"}


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class TaskConfig:
    name: str
    dataset: Path
    prompt_template: str
    system_prompt: str | None
    output_schema: dict[str, Any] | None

    @property
    def placeholders(self) -> set[str]:
        return set(PLACEHOLDER.findall(self.prompt_template))


@dataclass(frozen=True)
class ProviderConfig:
    type: str
    model: str | None = None
    temperature: float = 0.0
    rpm: float | None = None
    json_mode: bool = True
    cache_dir: Path | None = None
    outputs: Path | None = None


@dataclass(frozen=True)
class MetricConfig:
    name: str
    type: str
    severity: Severity
    minimum: float | None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateConfig:
    baseline: Path | None = None
    max_regression: float | None = None


@dataclass(frozen=True)
class Config:
    task: TaskConfig
    provider: ProviderConfig
    metrics: list[MetricConfig]
    gate: GateConfig


def load_config(path: str | Path) -> Config:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from None
    return parse_config(raw, base_dir=path.parent)


def parse_config(raw: Any, base_dir: Path = Path(".")) -> Config:
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping with 'task', 'provider' and 'metrics'")
    if raw.get("version", 1) != 1:
        raise ConfigError(f"unsupported config version {raw.get('version')!r}")

    task = _parse_task(_mapping(raw, "task"), base_dir)
    provider = _parse_provider(_mapping(raw, "provider"), base_dir)
    metrics = _parse_metrics(raw.get("metrics"), task)
    gate = _parse_gate(raw.get("gate") or {}, base_dir)
    return Config(task=task, provider=provider, metrics=metrics, gate=gate)


def _mapping(raw: dict, key: str) -> dict:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping")
    return value


def _resolve(base_dir: Path, value: Any, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"'{where}' must be a file path")
    p = Path(value)
    return p if p.is_absolute() else base_dir / p


def _read_text(path: Path, where: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"'{where}' file not found: {path}") from None


def _parse_task(raw: dict, base_dir: Path) -> TaskConfig:
    if "dataset" not in raw or "prompt" not in raw:
        raise ConfigError("'task.dataset' and 'task.prompt' are required")
    dataset = _resolve(base_dir, raw["dataset"], "task.dataset")
    template = _read_text(_resolve(base_dir, raw["prompt"], "task.prompt"), "task.prompt")
    if not PLACEHOLDER.search(template):
        raise ConfigError("task.prompt has no {{placeholder}}; every case would get the same prompt")

    system = None
    if raw.get("system"):
        system = _read_text(_resolve(base_dir, raw["system"], "task.system"), "task.system")

    schema = None
    if raw.get("output_schema"):
        schema_path = _resolve(base_dir, raw["output_schema"], "task.output_schema")
        try:
            schema = json.loads(_read_text(schema_path, "task.output_schema"))
            Draft202012Validator.check_schema(schema)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"task.output_schema is not valid JSON: {exc}") from None
        except SchemaError as exc:
            raise ConfigError(f"task.output_schema is not a valid JSON Schema: {exc.message}") from None

    return TaskConfig(
        name=str(raw.get("name") or dataset.stem),
        dataset=dataset,
        prompt_template=template,
        system_prompt=system,
        output_schema=schema,
    )


def _parse_provider(raw: dict, base_dir: Path) -> ProviderConfig:
    ptype = raw.get("type")
    if ptype not in PROVIDERS:
        raise ConfigError(f"provider.type must be one of {sorted(PROVIDERS)}, got {ptype!r}")

    known = {"type", "model", "temperature", "rpm", "json_mode", "cache_dir", "outputs"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"provider: unknown key(s): {', '.join(unknown)}")

    temperature = raw.get("temperature", 0.0)
    if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise ConfigError("provider.temperature must be a number between 0 and 2")
    rpm = raw.get("rpm")
    if rpm is not None and (not isinstance(rpm, (int, float)) or rpm <= 0):
        raise ConfigError("provider.rpm must be a positive number")

    model = raw.get("model")
    outputs = None
    if ptype == "gemini" and not model:
        raise ConfigError("provider.model is required for gemini")
    if ptype == "file":
        outputs = _resolve(base_dir, raw.get("outputs"), "provider.outputs")

    cache_dir = _resolve(base_dir, raw["cache_dir"], "provider.cache_dir") if raw.get("cache_dir") else None
    return ProviderConfig(
        type=ptype,
        model=model,
        temperature=float(temperature),
        rpm=float(rpm) if rpm else None,
        json_mode=bool(raw.get("json_mode", True)),
        cache_dir=cache_dir,
        outputs=outputs,
    )


def _parse_metrics(raw: Any, task: TaskConfig) -> list[MetricConfig]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError("'metrics' must be a non-empty list")
    out, seen = [], set()
    for i, item in enumerate(raw):
        where = f"metrics[{i}]"
        if not isinstance(item, dict) or not item.get("type"):
            raise ConfigError(f"{where}: each metric needs a 'type'")
        try:
            spec = registry.get(item["type"])
        except registry.RegistryError as exc:
            raise ConfigError(f"{where}: {exc}") from None

        name = str(item.get("name") or item["type"])
        where = f"{where} ('{name}')"
        if name in seen:
            raise ConfigError(f"duplicate metric name '{name}'")
        seen.add(name)

        try:
            severity = Severity(item.get("severity", "error"))
        except ValueError:
            raise ConfigError(f"{where}: severity must be 'error' or 'warn'") from None

        minimum = item.get("min")
        if minimum is not None and (not isinstance(minimum, (int, float)) or not 0 <= minimum <= 1):
            raise ConfigError(f"{where}: 'min' must be between 0 and 1")

        params = {k: v for k, v in item.items() if k not in _RESERVED}
        missing = [p for p in spec.required if p not in params]
        if missing:
            raise ConfigError(f"{where}: missing required parameter(s): {', '.join(missing)}")
        unknown = sorted(set(params) - spec.allowed)
        if unknown:
            raise ConfigError(f"{where}: unknown parameter(s): {', '.join(unknown)}")
        if spec.validate:
            try:
                spec.validate(params)
            except registry.ParamError as exc:
                raise ConfigError(f"{where}: {exc}") from None
        if spec.needs_schema and task.output_schema is None:
            raise ConfigError(f"{where}: requires task.output_schema")

        out.append(MetricConfig(name=name, type=item["type"], severity=severity, minimum=minimum, params=params))
    return out


def _parse_gate(raw: Any, base_dir: Path) -> GateConfig:
    if not isinstance(raw, dict):
        raise ConfigError("'gate' must be a mapping")
    baseline = _resolve(base_dir, raw["baseline"], "gate.baseline") if raw.get("baseline") else None
    max_reg = raw.get("max_regression")
    if max_reg is not None and (not isinstance(max_reg, (int, float)) or not 0 <= max_reg <= 1):
        raise ConfigError("gate.max_regression must be between 0 and 1")
    if max_reg is not None and baseline is None:
        raise ConfigError("gate.max_regression needs gate.baseline")
    return GateConfig(baseline=baseline, max_regression=max_reg)
