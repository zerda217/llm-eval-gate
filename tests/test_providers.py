import json

import pytest

from llm_eval_gate.providers import CachedProvider, FileProvider, GeminiProvider, ProviderError
from llm_eval_gate.result import Generation

OK_RESPONSE = {
    "candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": '{"events": []}'}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 9},
}


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": headers, "body": json.loads(body)})
        return self.responses.pop(0)


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def gemini(responses, **kw):
    clock = FakeClock()
    transport = FakeTransport(responses)
    p = GeminiProvider("gemini-test", api_key="k", transport=transport, sleep=clock.sleep, clock=clock.time, **kw)
    return p, transport, clock


# ------------------------------------------------------------------ gemini


def test_gemini_request_shape_and_parsing():
    p, transport, _ = gemini([(200, OK_RESPONSE)])
    gen = p.generate("c1", "SYSTEM", "PROMPT")
    call = transport.calls[0]
    assert call["url"].endswith("/models/gemini-test:generateContent")
    assert call["headers"]["x-goog-api-key"] == "k"
    assert call["body"]["systemInstruction"] == {"parts": [{"text": "SYSTEM"}]}
    assert call["body"]["contents"][0]["parts"][0]["text"] == "PROMPT"
    assert call["body"]["generationConfig"] == {"temperature": 0.0, "responseMimeType": "application/json"}
    assert gen.text == '{"events": []}'  # thought parts are dropped
    assert (gen.input_tokens, gen.output_tokens, gen.finish_reason) == (120, 9, "STOP")


def test_gemini_json_mode_off_and_no_system():
    p, transport, _ = gemini([(200, OK_RESPONSE)], json_mode=False)
    p.generate("c1", None, "PROMPT")
    body = transport.calls[0]["body"]
    assert "systemInstruction" not in body
    assert "responseMimeType" not in body["generationConfig"]


def test_gemini_retries_rate_limits_with_backoff():
    p, transport, clock = gemini([(429, {"error": {"message": "quota"}}), (503, {}), (200, OK_RESPONSE)])
    assert p.generate("c1", None, "P").text == '{"events": []}'
    assert len(transport.calls) == 3
    assert clock.sleeps == [2.0, 4.0]


def test_gemini_gives_up_after_max_retries():
    p, _, _ = gemini([(429, {"error": {"message": "quota"}})] * 3, max_retries=2)
    with pytest.raises(ProviderError, match="429.*quota"):
        p.generate("c1", None, "P")


def test_gemini_does_not_retry_client_errors():
    p, transport, _ = gemini([(400, {"error": {"message": "model not found"}})])
    with pytest.raises(ProviderError, match="model not found"):
        p.generate("c1", None, "P")
    assert len(transport.calls) == 1


def test_gemini_blocked_prompt_returns_empty_text():
    p, _, _ = gemini([(200, {"promptFeedback": {"blockReason": "SAFETY"}})])
    gen = p.generate("c1", None, "P")
    assert gen.text == "" and gen.finish_reason == "SAFETY"


def test_gemini_throttles_to_rpm():
    p, _, clock = gemini([(200, OK_RESPONSE)] * 3, rpm=30)  # one call every 2s
    for _ in range(3):
        p.generate("c", None, "P")
    assert clock.sleeps == [2.0, 2.0]


def test_gemini_requires_key():
    p = GeminiProvider("m", api_key="placeholder", transport=FakeTransport([]))
    p.api_key = None  # simulate an environment without GEMINI_API_KEY
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        p.generate("c", None, "P")


# -------------------------------------------------------------------- file


def test_file_provider(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text('{"id": "a", "output": {"x": 1}}\n{"id": "b", "output": "raw text"}\n', encoding="utf-8")
    p = FileProvider(path)
    assert json.loads(p.generate("a", None, "").text) == {"x": 1}
    assert p.generate("b", None, "").text == "raw text"
    with pytest.raises(ProviderError, match="no output for case 'zzz'"):
        p.generate("zzz", None, "")


@pytest.mark.parametrize("content", ['{"id": "a"}\n', "not json\n"])
def test_file_provider_rejects_bad_lines(tmp_path, content):
    path = tmp_path / "out.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ProviderError, match="out.jsonl:1"):
        FileProvider(path)


def test_file_provider_missing_file(tmp_path):
    with pytest.raises(ProviderError, match="not found"):
        FileProvider(tmp_path / "nope.jsonl")


# ------------------------------------------------------------------- cache


class CountingProvider:
    def __init__(self, model="m"):
        self.model, self.calls = model, 0

    def describe(self):
        return {"type": "fake", "model": self.model}

    def generate(self, case_id, system, prompt):
        self.calls += 1
        return Generation(text=f"answer to {prompt}", input_tokens=5, output_tokens=2, latency_ms=10.0)


def test_cache_stores_then_hits(tmp_path):
    inner = CountingProvider()
    cache = CachedProvider(inner, tmp_path)
    first = cache.generate("c1", "S", "P")
    second = cache.generate("c1", "S", "P")
    assert inner.calls == 1
    assert not first.cached and second.cached
    assert second.text == first.text and second.input_tokens == 5
    assert (cache.hits, cache.misses) == (1, 1)


def test_cache_key_changes_with_prompt_system_and_model(tmp_path):
    a = CachedProvider(CountingProvider("m1"), tmp_path)
    b = CachedProvider(CountingProvider("m2"), tmp_path)
    keys = {a.key("S", "P"), a.key("S", "P2"), a.key("S2", "P"), b.key("S", "P")}
    assert len(keys) == 4


def test_replay_mode_never_calls_the_model(tmp_path):
    CachedProvider(CountingProvider(), tmp_path).generate("c1", None, "P")
    inner = CountingProvider()
    replay = CachedProvider(inner, tmp_path, mode="replay")
    assert replay.generate("c1", None, "P").cached
    with pytest.raises(ProviderError, match="cache miss"):
        replay.generate("c2", None, "new prompt")
    assert inner.calls == 0


def test_live_mode_refreshes_cache(tmp_path):
    inner = CountingProvider()
    live = CachedProvider(inner, tmp_path, mode="live")
    live.generate("c1", None, "P")
    live.generate("c1", None, "P")
    assert inner.calls == 2


def test_invalid_cache_mode(tmp_path):
    with pytest.raises(ValueError):
        CachedProvider(CountingProvider(), tmp_path, mode="sometimes")
