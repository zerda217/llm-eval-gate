"""Model providers, response cache and rate limiting.

- GeminiProvider calls the Gemini REST API with the standard library only.
- FileProvider replays outputs produced elsewhere (offline eval of logs, fixtures).
- CachedProvider wraps any provider. A committed cache lets CI re-run the eval
  without an API key, and keeps free-tier quota for prompts that actually changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Protocol

from .result import Generation

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY_STATUS = {429, 500, 502, 503, 504}
CACHE_MODES = {"cached", "replay", "live"}


class ProviderError(Exception):
    """A model call failed in a way that retrying will not fix."""


class Provider(Protocol):
    def describe(self) -> dict[str, Any]: ...

    def generate(self, case_id: str, system: str | None, prompt: str) -> Generation: ...


# ------------------------------------------------------------------ file


class FileProvider:
    """Serve outputs from a JSONL file of {"id": ..., "output": <string or JSON value>}."""

    def __init__(self, path: Path):
        self.path = path
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            raise ProviderError(f"outputs file not found: {path}") from None
        self._outputs: dict[str, str] = {}
        for lineno, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                out = row["output"]
                self._outputs[str(row["id"])] = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
            except (json.JSONDecodeError, KeyError, TypeError):
                raise ProviderError(f"{path.name}:{lineno}: expected {{'id': ..., 'output': ...}}") from None

    def describe(self) -> dict[str, Any]:
        return {"type": "file", "outputs": str(self.path)}

    def generate(self, case_id: str, system: str | None, prompt: str) -> Generation:
        if case_id not in self._outputs:
            raise ProviderError(f"no output for case '{case_id}' in {self.path.name}")
        return Generation(text=self._outputs[case_id])


# ---------------------------------------------------------------- gemini

Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, dict[str, Any]]]


def _urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except (ValueError, OSError):
            payload = {}
        return exc.code, payload
    except (urllib.error.URLError, TimeoutError) as exc:
        return 0, {"error": {"message": str(exc)}}


class GeminiProvider:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        rpm: float | None = None,
        json_mode: bool = True,
        max_retries: int = 5,
        timeout: float = 60.0,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.temperature = temperature
        self.json_mode = json_mode
        self.max_retries = max_retries
        self.timeout = timeout
        self.min_interval = 60.0 / rpm if rpm else 0.0
        self._transport = transport or _urllib_transport
        self._sleep = sleep
        self._clock = clock
        self._last_call: float | None = None

    def describe(self) -> dict[str, Any]:
        return {"type": "gemini", "model": self.model, "temperature": self.temperature, "json_mode": self.json_mode}

    def generate(self, case_id: str, system: str | None, prompt: str) -> Generation:
        if not self.api_key:
            raise ProviderError("GEMINI_API_KEY is not set (get a free key at aistudio.google.com)")

        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        if self.json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        url = GEMINI_URL.format(model=self.model)
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        payload = json.dumps(body).encode("utf-8")

        for attempt in range(self.max_retries + 1):
            self._throttle()
            started = self._clock()
            status, data = self._transport(url, headers, payload, self.timeout)
            latency_ms = (self._clock() - started) * 1000
            if status == 200:
                return _parse_gemini(data, latency_ms)
            message = (data.get("error") or {}).get("message", "") if isinstance(data, dict) else ""
            if (status in RETRY_STATUS or status == 0) and attempt < self.max_retries:
                self._sleep(min(60.0, 2.0 * 2**attempt))
                continue
            raise ProviderError(f"gemini HTTP {status} for case '{case_id}': {message or 'no message'}")
        raise ProviderError("unreachable")  # pragma: no cover

    def _throttle(self) -> None:
        if self.min_interval and self._last_call is not None:
            wait = self.min_interval - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = self._clock()


def _parse_gemini(data: dict[str, Any], latency_ms: float) -> Generation:
    usage = data.get("usageMetadata") or {}
    candidates = data.get("candidates") or []
    text, finish = "", None
    if candidates:
        cand = candidates[0]
        finish = cand.get("finishReason")
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    else:
        finish = (data.get("promptFeedback") or {}).get("blockReason", "NO_CANDIDATES")
    return Generation(
        text=text,
        input_tokens=usage.get("promptTokenCount"),
        output_tokens=usage.get("candidatesTokenCount"),
        latency_ms=round(latency_ms, 1),
        finish_reason=finish,
    )


# ----------------------------------------------------------------- cache


class CachedProvider:
    """Content-addressed response cache.

    Modes:
      cached  read cache, call the model on a miss and store the answer (default)
      replay  cache only; a miss is an error. Use in CI without an API key.
      live    always call the model, overwrite the cache
    """

    def __init__(self, inner: Provider, cache_dir: Path, mode: str = "cached"):
        if mode not in CACHE_MODES:
            raise ValueError(f"mode must be one of {sorted(CACHE_MODES)}")
        self.inner, self.cache_dir, self.mode = inner, cache_dir, mode
        self.hits = self.misses = 0

    def describe(self) -> dict[str, Any]:
        return {**self.inner.describe(), "cache": str(self.cache_dir), "mode": self.mode}

    def key(self, system: str | None, prompt: str) -> str:
        material = json.dumps({"provider": self.inner.describe(), "system": system, "prompt": prompt}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def generate(self, case_id: str, system: str | None, prompt: str) -> Generation:
        key = self.key(system, prompt)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if self.mode != "live" and path.exists():
            self.hits += 1
            data = json.loads(path.read_text(encoding="utf-8"))
            return Generation(**{**data["generation"], "cached": True})
        if self.mode == "replay":
            raise ProviderError(f"cache miss for case '{case_id}' in replay mode (prompt or model changed? run with --mode cached)")
        self.misses += 1
        gen = self.inner.generate(case_id, system, prompt)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"case_id": case_id, "provider": self.inner.describe(), "generation": {**asdict(gen), "cached": False}}
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return gen
