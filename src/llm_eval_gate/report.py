"""Render a RunResult as console text, Markdown (PR comment / job summary) or JSON."""

from __future__ import annotations

import json

from .result import RunResult

_LABEL = {"pass": "PASS", "fail": "FAIL", "warn": "WARN"}
_ICON = {"pass": "✅", "fail": "❌", "warn": "⚠️"}
MAX_CASES = 15


def _num(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def format_text(run: RunResult) -> str:
    lines = [f"llm-eval-gate · {run.task} · {len(run.records)} cases", ""]
    width = max(len(m.name) for m in run.metrics)
    for m in run.metrics:
        line = f"  {_LABEL[m.status]}  {m.name:<{width}}  {m.value:.3f}  (min {_num(m.minimum)}, baseline {_num(m.baseline)})  {m.detail}"
        if m.reasons:
            line += "  <- " + "; ".join(m.reasons)
        lines.append(line)

    problems = run.case_problems()
    if problems:
        lines += ["", f"Cases with problems ({len(problems)}):"]
        tags = {r.id: r.tags for r in run.records}
        for case_id, msgs in list(problems.items())[:MAX_CASES]:
            tag = f" [{', '.join(tags.get(case_id, []))}]" if tags.get(case_id) else ""
            lines.append(f"  - {case_id}{tag}: {'; '.join(msgs)}")
        if len(problems) > MAX_CASES:
            lines.append(f"  ... and {len(problems) - MAX_CASES} more (see --json-out)")

    by_tag = run.by_tag()
    if by_tag:
        lines += ["", "By tag (cases with problems / total):"]
        lines += [f"  {tag:<16} {f}/{t}" for tag, (f, t) in by_tag.items()]

    u = run.usage()
    lines += ["", f"Usage: {u['calls']} calls ({u['cache_hits']} cached), {u['input_tokens']:,} in / {u['output_tokens']:,} out tokens, "
                  f"latency p50 {u['latency_p50_ms'] or '-'} ms, p95 {u['latency_p95_ms'] or '-'} ms"]

    verdict = "PASSED" if run.passed else "FAILED"
    detail = f"{len(run.errors)} error(s), {len(run.warnings)} warning(s)"
    if run.fail_on_warn and run.warnings:
        detail += ", warnings treated as errors"
    lines += ["", f"Result: {verdict} ({detail}) -> exit {run.exit_code}"]
    return "\n".join(lines)


def format_markdown(run: RunResult) -> str:
    verdict = "✅ Passed" if run.passed else "❌ Blocked"
    out = [f"### llm-eval-gate · {run.task} — {verdict}", "", "| | Metric | Value | Min | Baseline | Note |", "|---|---|---|---|---|---|"]
    for m in run.metrics:
        out.append(f"| {_ICON[m.status]} | `{m.name}` | {m.value:.3f} | {_num(m.minimum)} | {_num(m.baseline)} | {'; '.join(m.reasons) or m.detail} |")

    by_tag = run.by_tag()
    if by_tag:
        out += ["", "| Tag | Failed / Total |", "|---|---|"]
        out += [f"| {tag} | {f} / {t} |" for tag, (f, t) in by_tag.items()]

    problems = run.case_problems()
    if problems:
        out += ["", f"<details><summary>{len(problems)} case(s) with problems</summary>", ""]
        out += [f"- `{cid}`: {'; '.join(msgs)}" for cid, msgs in list(problems.items())[: MAX_CASES * 2]]
        out += ["", "</details>"]

    u = run.usage()
    out += ["", f"{len(run.records)} cases · {u['calls']} calls ({u['cache_hits']} cached) · {u['input_tokens']:,} in / {u['output_tokens']:,} out tokens"]
    return "\n".join(out) + "\n"


def format_json(run: RunResult) -> str:
    return json.dumps(run.to_dict(), ensure_ascii=False, indent=2, default=str)
