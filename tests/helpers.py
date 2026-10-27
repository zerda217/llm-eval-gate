import json
from pathlib import Path

from llm_eval_gate.registry import MetricContext, get
from llm_eval_gate.result import CaseRecord, Generation
from llm_eval_gate.runner import parse_output

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "schedule"


def rec(case_id: str, output, expected, tags=None, gen: Generation | None = None) -> CaseRecord:
    """Build a CaseRecord from a Python value (serialised) or raw text output."""
    text = output if isinstance(output, str) else json.dumps(output)
    parsed, error = parse_output(text)
    return CaseRecord(id=case_id, tags=tags or [], expected=expected, output_text=text, parsed=parsed, parse_error=error, generation=gen)


def metric(name: str, records, schema=None, **params):
    return get(name).fn(records, params, MetricContext(output_schema=schema))


def ev(date: str, start: str | None = "19:00", title: str = "t", location=None) -> dict:
    return {"title": title, "date": date, "start_time": start, "location": location}


def write_project(tmp: Path, *, outputs: list[dict], cases: list[dict], metrics_yaml: str, gate_yaml: str = "") -> Path:
    """Create a minimal self-contained eval project using the file provider."""
    (tmp / "prompt.txt").write_text("Q: {{question}}", encoding="utf-8")
    (tmp / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")
    (tmp / "outputs.jsonl").write_text("".join(json.dumps(o) + "\n" for o in outputs), encoding="utf-8")
    cfg = tmp / "eval.yaml"
    cfg.write_text(
        "version: 1\n"
        "task: {dataset: cases.jsonl, prompt: prompt.txt}\n"
        "provider: {type: file, outputs: outputs.jsonl}\n"
        f"metrics:\n{metrics_yaml}\n{gate_yaml}",
        encoding="utf-8",
    )
    return cfg
