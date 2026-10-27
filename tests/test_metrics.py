import pytest

from llm_eval_gate.metrics import get_path, _MISSING
from llm_eval_gate.registry import ParamError, get
from llm_eval_gate.runner import parse_output
from helpers import ev, metric, rec

SCHEMA = {
    "type": "object",
    "required": ["events"],
    "properties": {"events": {"type": "array", "items": {"type": "object", "required": ["date"]}}},
}


# ------------------------------------------------------------ parse_output


@pytest.mark.parametrize(
    "text,value",
    [('{"a": 1}', {"a": 1}), ('```json\n{"a": 1}\n```', {"a": 1}), ("```\n[1, 2]\n```", [1, 2]), ("  true ", True)],
)
def test_parse_output_accepts_json_and_fences(text, value):
    assert parse_output(text) == (value, None)


@pytest.mark.parametrize("text", ["", "   ", "{'a': 1}", '{"a": 1', "Sure! here it is: {}"])
def test_parse_output_rejects(text):
    value, error = parse_output(text)
    assert value is None and error


# ---------------------------------------------------------------- get_path


def test_get_path():
    obj = {"a": {"b": [10, {"c": 3}]}}
    assert get_path(obj, "a.b.1.c") == 3
    assert get_path(obj, "a.b.0") == 10
    assert get_path(obj, "a.x") is _MISSING
    assert get_path(obj, "a.b.9") is _MISSING
    assert get_path(obj, "") is obj


# ------------------------------------------------------------ json / schema


def test_json_valid_rate():
    records = [rec("a", {"events": []}, {}), rec("b", "not json", {})]
    out = metric("json_valid_rate", records)
    assert out.value == 0.5
    assert list(out.problems) == ["b"]


def test_schema_valid_rate_counts_invalid_json_as_invalid():
    records = [
        rec("ok", {"events": [{"date": "2026-11-01"}]}, {}),
        rec("bad", {"events": [{"title": "x"}]}, {}),
        rec("nojson", "oops", {}),
    ]
    out = metric("schema_valid_rate", records, schema=SCHEMA)
    assert out.value == pytest.approx(1 / 3)
    assert "date" in out.problems["bad"][0]
    assert "nojson" not in out.problems  # reported once, by json_valid_rate


# --------------------------------------------------------- set precision/recall

GOLD = {"events": [ev("2026-11-01"), ev("2026-11-02", None)]}


def test_set_metrics_perfect():
    records = [rec("a", GOLD, GOLD)]
    assert metric("set_precision", records, path="events", keys=["date", "start_time"]).value == 1.0
    assert metric("set_recall", records, path="events", keys=["date", "start_time"]).value == 1.0


def test_set_metrics_wrong_time_is_both_fp_and_fn():
    out = {"events": [ev("2026-11-01", "07:00"), ev("2026-11-02", None)]}
    records = [rec("a", out, GOLD)]
    p = metric("set_precision", records, path="events", keys=["date", "start_time"])
    r = metric("set_recall", records, path="events", keys=["date", "start_time"])
    assert p.value == 0.5 and r.value == 0.5
    assert p.problems["a"] == ["unexpected (2026-11-01, 07:00)"]
    assert r.problems["a"] == ["missed (2026-11-01, 19:00)"]


def test_set_metrics_are_micro_averaged():
    records = [
        rec("a", {"events": [ev("2026-11-01"), ev("2026-11-03")]}, {"events": [ev("2026-11-01")]}),  # 1 tp, 1 fp
        rec("b", {"events": [ev("2026-11-05")]}, {"events": [ev("2026-11-05")]}),  # 1 tp
    ]
    assert metric("set_precision", records, path="events", keys=["date"]).value == pytest.approx(2 / 3)


def test_set_metrics_ignore_non_key_fields_and_whitespace():
    out = {"events": [ev(" 2026-11-01 ", title="different title", location="elsewhere")]}
    assert metric("set_recall", [rec("a", out, GOLD)], path="events", keys=["date", "start_time"]).value == 0.5


def test_set_metrics_invalid_json_counts_as_no_prediction():
    records = [rec("a", "garbage", GOLD)]
    assert metric("set_recall", records, path="events", keys=["date"]).value == 0.0
    assert metric("set_precision", records, path="events", keys=["date"]).value == 1.0  # nothing predicted


def test_set_metrics_tolerate_wrong_shapes():
    records = [rec("a", {"events": "none"}, GOLD), rec("b", {"events": [1, "x", ev("2026-11-01")]}, GOLD)]
    assert metric("set_precision", records, path="events", keys=["date", "start_time"]).value == 1.0


@pytest.mark.parametrize("params", [{"path": "events", "keys": []}, {"path": "events", "keys": "date"}, {"path": 1, "keys": ["d"]}])
def test_set_params_validation(params):
    with pytest.raises(ParamError):
        get("set_precision").validate(params)


# ------------------------------------------------------- empty / exact match


def test_empty_when_expected_rate():
    records = [
        rec("quiet", {"events": []}, {"events": []}),
        rec("hallucinated", {"events": [ev("2026-11-01")]}, {"events": []}),
        rec("broken", "nope", {"events": []}),
        rec("ignored", {"events": []}, GOLD),  # expected non-empty -> not counted
    ]
    out = metric("empty_when_expected_rate", records, path="events")
    assert out.value == pytest.approx(1 / 3)
    assert set(out.problems) == {"hallucinated", "broken"}


def test_exact_match_rate():
    records = [
        rec("a", {"label": "spam"}, {"label": "spam"}),
        rec("b", {"label": "ham"}, {"label": "spam"}),
        rec("c", {}, {"label": "spam"}),
    ]
    out = metric("exact_match_rate", records, path="label")
    assert out.value == pytest.approx(1 / 3)
    assert "<missing>" in out.problems["c"][0]


def test_empty_dataset_scores_one():
    assert metric("set_recall", [], path="events", keys=["date"]).value == 1.0
