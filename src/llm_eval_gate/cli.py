"""Command line interface.

    llm-eval-gate run CONFIG [--mode cached|replay|live] [--outputs FILE]
                             [--json-out FILE] [--md-out FILE]
                             [--fail-on-warn] [--update-baseline]
    llm-eval-gate validate CONFIG
    llm-eval-gate list-metrics
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from . import __version__, registry
from .config import ConfigError, load_config
from .dataset import DatasetError, check_placeholders, load_cases
from .providers import CACHE_MODES, ProviderError
from .report import format_json, format_markdown, format_text
from .result import ExitCode
from .runner import build_provider, load_baseline, run, save_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-eval-gate", description="Evaluation gate for LLM features.")
    parser.add_argument("--version", action="version", version=f"llm-eval-gate {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="evaluate and exit non-zero when the gate fails")
    r.add_argument("config", type=Path)
    r.add_argument("--mode", choices=sorted(CACHE_MODES), default="cached", help="response cache mode (default: cached)")
    r.add_argument("--outputs", type=Path, help="score pre-recorded outputs (JSONL) instead of calling the model")
    r.add_argument("--json-out", type=Path, help="write the full result as JSON")
    r.add_argument("--md-out", type=Path, help="write a Markdown summary (for PR comments / job summary)")
    r.add_argument("--fail-on-warn", action="store_true", help="treat warn-level failures as blocking")
    r.add_argument("--update-baseline", action="store_true", help="save this run's metric values as the new baseline")
    r.add_argument("--no-baseline", action="store_true", help="ignore the baseline (no regression check)")

    v = sub.add_parser("validate", help="validate config and dataset without calling a model")
    v.add_argument("config", type=Path)

    sub.add_parser("list-metrics", help="list available metric types")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list-metrics":
        for spec in registry.available():
            params = [*spec.required, *(f"[{p}]" for p in spec.optional)]
            print(f"{spec.name:<26} {' '.join(params):<16} {spec.description}")
        return ExitCode.OK
    if args.command == "validate":
        return _validate(args.config)
    return _run(args)


def _err(kind: str, exc: Exception) -> None:
    print(f"{kind}: {exc}", file=sys.stderr)


def _validate(path: Path) -> int:
    try:
        config = load_config(path)
        cases = load_cases(config.task.dataset)
        check_placeholders(cases, config.task.placeholders)
    except ConfigError as exc:
        _err("config error", exc)
        return ExitCode.CONFIG_ERROR
    except DatasetError as exc:
        _err("dataset error", exc)
        return ExitCode.RUNTIME_ERROR
    print(f"config OK: {len(cases)} cases, {len(config.metrics)} metrics, provider {config.provider.type}")
    return ExitCode.OK


def _run(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        if args.outputs:
            config = replace(config, provider=replace(config.provider, type="file", outputs=args.outputs))
        provider = build_provider(config.provider, args.mode)
        baseline = None if args.no_baseline else load_baseline(config.gate.baseline)
    except ConfigError as exc:
        _err("config error", exc)
        return ExitCode.CONFIG_ERROR
    except ProviderError as exc:
        _err("provider error", exc)
        return ExitCode.RUNTIME_ERROR

    try:
        result = run(config, provider, baseline=baseline, fail_on_warn=args.fail_on_warn)
    except DatasetError as exc:
        _err("dataset error", exc)
        return ExitCode.RUNTIME_ERROR
    except ProviderError as exc:
        _err("provider error", exc)
        return ExitCode.RUNTIME_ERROR

    print(format_text(result))
    for target, fmt in ((args.json_out, format_json), (args.md_out, format_markdown)):
        if target:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(fmt(result), encoding="utf-8")

    if args.update_baseline:
        if config.gate.baseline is None:
            _err("config error", ConfigError("--update-baseline needs gate.baseline in the config"))
            return ExitCode.CONFIG_ERROR
        save_baseline(config.gate.baseline, result, provider)
        print(f"baseline updated: {config.gate.baseline}")
    return result.exit_code
