import json
from pathlib import Path

import pytest

from llm_eval_gate.config import ConfigError, load_config, parse_config
from llm_eval_gate.result import Severity
from helpers import EXAMPLE


@pytest.fixture
def base(tmp_path):
    (tmp_path / "p.txt").write_text("Q: {{question}}", encoding="utf-8")
    (tmp_path / "s.json").write_text(json.dumps({"type": "object"}), encoding="utf-8")
    return tmp_path


def raw(**over):
    cfg = {
        "version": 1,
        "task": {"dataset": "cases.jsonl", "prompt": "p.txt"},
        "provider": {"type": "gemini", "model": "m"},
        "metrics": [{"type": "json_valid_rate", "min": 0.9}],
    }
    for key, value in over.items():
        cfg[key] = value
    return cfg


def test_minimal_config(base):
    cfg = parse_config(raw(), base)
    assert cfg.task.placeholders == {"question"}
    assert cfg.task.dataset == base / "cases.jsonl"
    assert cfg.provider.temperature == 0.0 and cfg.provider.json_mode
    m = cfg.metrics[0]
    assert (m.name, m.severity, m.minimum) == ("json_valid_rate", Severity.ERROR, 0.9)
    assert cfg.gate.baseline is None


def test_example_config_loads():
    cfg = load_config(EXAMPLE / "eval.yaml")
    assert cfg.task.name == "telegram-appointments"
    assert cfg.task.output_schema is not None and cfg.task.system_prompt
    assert [m.name for m in cfg.metrics] == ["json_valid", "schema_valid", "event_precision", "no_false_plans", "event_recall"]
    assert cfg.metrics[-1].severity is Severity.WARN
    assert cfg.provider.cache_dir == EXAMPLE / "cache"


@pytest.mark.parametrize(
    "override,match",
    [
        ({"version": 2}, "version"),
        ({"task": {"prompt": "p.txt"}}, "task.dataset"),
        ({"task": {"dataset": "c.jsonl", "prompt": "missing.txt"}}, "not found"),
        ({"provider": {"type": "openai"}}, "provider.type"),
        ({"provider": {"type": "gemini"}}, "model is required"),
        ({"provider": {"type": "gemini", "model": "m", "temperature": 5}}, "temperature"),
        ({"provider": {"type": "gemini", "model": "m", "rpm": 0}}, "rpm"),
        ({"provider": {"type": "gemini", "model": "m", "modle": "x"}}, "unknown key"),
        ({"provider": {"type": "file"}}, "provider.outputs"),
        ({"metrics": []}, "non-empty"),
        ({"metrics": [{"type": "bleu"}]}, "unknown metric"),
        ({"metrics": [{"type": "json_valid_rate", "min": 90}]}, "between 0 and 1"),
        ({"metrics": [{"type": "json_valid_rate", "severity": "fatal"}]}, "severity"),
        ({"metrics": [{"type": "set_recall", "path": "events"}]}, "missing required parameter"),
        ({"metrics": [{"type": "set_recall", "path": "e", "keys": ["d"], "key": 1}]}, "unknown parameter"),
        ({"metrics": [{"type": "set_recall", "path": "e", "keys": []}]}, "non-empty list"),
        ({"metrics": [{"type": "schema_valid_rate"}]}, "requires task.output_schema"),
        ({"metrics": [{"type": "json_valid_rate"}, {"type": "json_valid_rate"}]}, "duplicate"),
        ({"gate": {"max_regression": 0.05}}, "needs gate.baseline"),
        ({"gate": {"baseline": "b.json", "max_regression": 2}}, "between 0 and 1"),
    ],
)
def test_invalid_configs(base, override, match):
    with pytest.raises(ConfigError, match=match):
        parse_config(raw(**override), base)


def test_prompt_without_placeholder_is_rejected(base):
    (base / "p.txt").write_text("static prompt", encoding="utf-8")
    with pytest.raises(ConfigError, match="placeholder"):
        parse_config(raw(), base)


def test_invalid_output_schema(base):
    (base / "bad.json").write_text('{"type": "not-a-type"}', encoding="utf-8")
    with pytest.raises(ConfigError, match="not a valid JSON Schema"):
        parse_config(raw(task={"dataset": "c.jsonl", "prompt": "p.txt", "output_schema": "bad.json"}), base)


def test_load_config_errors(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("task: [", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad)
