"""Plugin registry for metrics.

A metric is a function ``(records, params, ctx) -> MetricOutcome`` that turns
all case records into one score in [0, 1] plus per-case problem notes.
Parameters are declared at registration so configs are validated before any
model call is made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .result import CaseRecord, MetricOutcome


@dataclass(frozen=True)
class MetricContext:
    output_schema: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


MetricFn = Callable[[list[CaseRecord], dict[str, Any], MetricContext], MetricOutcome]
Validator = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    fn: MetricFn
    required: tuple[str, ...]
    optional: tuple[str, ...]
    validate: Validator | None
    needs_schema: bool
    description: str

    @property
    def allowed(self) -> set[str]:
        return set(self.required) | set(self.optional)


class RegistryError(Exception):
    pass


class ParamError(ValueError):
    pass


_REGISTRY: dict[str, MetricSpec] = {}


def register(
    name: str,
    *,
    required: tuple[str, ...] = (),
    optional: tuple[str, ...] = (),
    validate: Validator | None = None,
    needs_schema: bool = False,
) -> Callable[[MetricFn], MetricFn]:
    def decorator(fn: MetricFn) -> MetricFn:
        if name in _REGISTRY:
            raise RegistryError(f"metric type '{name}' is already registered")
        doc = (fn.__doc__ or "").strip().splitlines()
        _REGISTRY[name] = MetricSpec(name, fn, tuple(required), tuple(optional), validate, needs_schema, doc[0] if doc else "")
        return fn

    return decorator


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def get(name: str) -> MetricSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise RegistryError(f"unknown metric type '{name}' (available: {known})") from None


def available() -> list[MetricSpec]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]
