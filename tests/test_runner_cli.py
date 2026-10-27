import json

import pytest

from llm_eval_gate.cli import main
from llm_eval_gate.dataset import DatasetError, load_cases, render
from llm_eval_gate.result import ExitCode
from helpers import EXAMPLE, write_project

CONFIG = str(EXAMPLE / "eval.yaml")
REFERENCE = str(EXAMPLE / "fixtures" / "reference_outputs.jsonl")
BROKEN = str(EXAMPLE / "fixtures" / "broken_outputs.jsonl")

CASES = [
    {"id": "a", "input": {"question": "q1"}, "expected": {"label": "x"}, "tags": ["easy"]},
    {"id": "b", "input": {"question": "q2"}, "expected": {"label": "y"}, "tags": ["hard"]},
]
EXACT = "  - {type: exact_match_rate, path: label, min: 0.9}"


# ----------------------------------------------------------------- dataset


def test_render_substitutes_and_serialises():
    assert render("A {{x}} B {{ y }}", {"x": "1", "y": {"k": "값"}}) == 'A 1 B {"k": "값"}'


@pytest.mark.parametrize(
    "content,match",
    [
        ("", "empty"),
        ("{bad\n", "invalid JSON"),
        ('{"id": "a", "input": {}}\n', "need"),
        ('{"id": "a", "input": [], "expected": 1}\n', "'input' must be an object"),
        ('{"id": "a", "input": {}, "expected": 1}\n{"id": "a", "input": {}, "expected": 1}\n', "duplicate"),
        ('{"id": "a", "input": {}, "expected": 1, "tags": "x"}\n', "tags"),
    ],
)
def test_dataset_errors(tmp_path, content, match):
    path = tmp_path / "c.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetError, match=match):
        load_cases(path)


def test_example_dataset_is_well_formed():
    cases = load_cases(EXAMPLE / "cases.jsonl")
    assert len(cases) == 40
    assert all("conversation" in c.input and isinstance(c.expected["events"], list) for c in cases)
    assert sum(len(c.expected["events"]) == 0 for c in cases) == 12


# ------------------------------------------------------------- example task


def test_reference_outputs_pass_the_gate(capsys):
    assert main(["run", CONFIG, "--outputs", REFERENCE, "--no-baseline"]) == ExitCode.OK
    assert "Result: PASSED" in capsys.readouterr().out


def test_broken_outputs_are_blocked(tmp_path, capsys):
    json_out, md_out = tmp_path / "r.json", tmp_path / "r.md"
    code = main(["run", CONFIG, "--outputs", BROKEN, "--no-baseline", "--json-out", str(json_out), "--md-out", str(md_out)])
    assert code == ExitCode.GATE_FAILED

    report = json.loads(json_out.read_text(encoding="utf-8"))
    status = {m["name"]: m["status"] for m in report["metrics"]}
    assert status == {"json_valid": "fail", "schema_valid": "pass", "event_precision": "fail", "no_false_plans": "fail", "event_recall": "warn"}
    failed_ids = {c["id"] for c in report["failed_cases"]}
    assert {"s012", "s026", "s030"} <= failed_ids  # cross-midnight date, cancelled kept, unanswered registered
    assert report["by_tag"]["cross_midnight"] == {"failed": 1, "total": 1}
    assert "❌ Blocked" in md_out.read_text(encoding="utf-8")


def test_validate_example(capsys):
    assert main(["validate", CONFIG]) == ExitCode.OK
    assert "40 cases" in capsys.readouterr().out


def test_list_metrics(capsys):
    assert main(["list-metrics"]) == ExitCode.OK
    assert "set_precision" in capsys.readouterr().out


# --------------------------------------------------------- gate semantics


def test_warn_does_not_block_unless_asked(tmp_path):
    cfg = write_project(
        tmp_path,
        cases=CASES,
        outputs=[{"id": "a", "output": {"label": "x"}}, {"id": "b", "output": {"label": "wrong"}}],
        metrics_yaml="  - {type: exact_match_rate, path: label, min: 0.9, severity: warn}",
    )
    assert main(["run", str(cfg)]) == ExitCode.OK
    assert main(["run", str(cfg), "--fail-on-warn"]) == ExitCode.GATE_FAILED


def test_baseline_regression_blocks_even_above_min(tmp_path, capsys):
    outputs = [{"id": "a", "output": {"label": "x"}}, {"id": "b", "output": {"label": "y"}}]
    cfg = write_project(
        tmp_path,
        cases=CASES,
        outputs=outputs,
        metrics_yaml="  - {type: exact_match_rate, path: label, min: 0.4}",
        gate_yaml="gate: {baseline: baseline.json, max_regression: 0.1}",
    )
    # first run records the baseline (1.0)
    assert main(["run", str(cfg), "--update-baseline"]) == ExitCode.OK
    saved = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    assert saved["metrics"] == {"exact_match_rate": 1.0}

    # a prompt change drops accuracy to 0.5: still above min 0.4, but a 0.5 regression
    (tmp_path / "outputs.jsonl").write_text(
        json.dumps(outputs[0]) + "\n" + json.dumps({"id": "b", "output": {"label": "no"}}) + "\n", encoding="utf-8"
    )
    capsys.readouterr()
    assert main(["run", str(cfg)]) == ExitCode.GATE_FAILED
    assert "regressed from baseline 1.000" in capsys.readouterr().out
    assert main(["run", str(cfg), "--no-baseline"]) == ExitCode.OK


def test_crashing_metric_fails_closed(tmp_path):
    from llm_eval_gate.registry import register, unregister

    @register("explodes")
    def explodes(records, params, ctx):
        """Always raises."""
        raise RuntimeError("boom")

    try:
        cfg = write_project(tmp_path, cases=CASES, outputs=[{"id": "a", "output": {}}, {"id": "b", "output": {}}], metrics_yaml="  - {type: explodes, min: 0.5}")
        assert main(["run", str(cfg)]) == ExitCode.GATE_FAILED
    finally:
        unregister("explodes")


# ------------------------------------------------------------- exit codes


def test_config_error_exit(tmp_path, capsys):
    cfg = tmp_path / "eval.yaml"
    cfg.write_text("version: 1\ntask: {}\n", encoding="utf-8")
    assert main(["run", str(cfg)]) == ExitCode.CONFIG_ERROR
    assert "config error" in capsys.readouterr().err


def test_missing_output_is_runtime_error(tmp_path, capsys):
    cfg = write_project(tmp_path, cases=CASES, outputs=[{"id": "a", "output": {}}], metrics_yaml=EXACT)
    assert main(["run", str(cfg)]) == ExitCode.RUNTIME_ERROR
    assert "no output for case 'b'" in capsys.readouterr().err


def test_prompt_field_missing_in_case(tmp_path, capsys):
    cases = [{"id": "a", "input": {"other": "x"}, "expected": {}}]
    cfg = write_project(tmp_path, cases=cases, outputs=[{"id": "a", "output": {}}], metrics_yaml=EXACT)
    assert main(["validate", str(cfg)]) == ExitCode.RUNTIME_ERROR
    assert "missing input field(s) used by the prompt: question" in capsys.readouterr().err


def test_missing_api_key(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    (tmp_path / "p.txt").write_text("{{question}}", encoding="utf-8")
    (tmp_path / "cases.jsonl").write_text(json.dumps(CASES[0]) + "\n", encoding="utf-8")
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(
        "task: {dataset: cases.jsonl, prompt: p.txt}\nprovider: {type: gemini, model: m}\nmetrics: [{type: json_valid_rate}]\n",
        encoding="utf-8",
    )
    assert main(["run", str(cfg)]) == ExitCode.RUNTIME_ERROR
    assert "GEMINI_API_KEY" in capsys.readouterr().err
