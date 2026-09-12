"""Thin LLM client: token/cost accounting and a disk cache keyed on (prompt hash, model).

Only the OpenAI-compatible provider is wired (NeuralWatt). Cached replies cost
nothing and are counted separately so a results row never claims tokens it
did not buy.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from arc_harness.env import PROJECT_ROOT
from config.models import EFFORTS, MAX_TOKENS, ModelSpec

CACHE_DIR = PROJECT_ROOT / ".llm_cache"
RATE_LIMIT_MAX_WAIT_S = 600.0  # total back-off before a 429 becomes an error


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0  # served from the provider's prefix cache (billed at the cached rate)
    reasoning_tokens: int = 0  # subset of output_tokens
    calls: int = 0
    cache_hits: int = 0  # local disk cache; no tokens bought

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.calls += other.calls
        self.cache_hits += other.cache_hits

    def cost_usd(self, model: ModelSpec | None) -> float:
        if model is None:
            return 0.0
        return model.cost_usd(self.input_tokens, self.output_tokens, self.cached_input_tokens)


@dataclass
class LLMResponse:
    text: str
    finish_reason: str
    usage: Usage
    from_cache: bool
    latency_s: float

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class CostCapExceeded(RuntimeError):
    """Raised before a call that would push this client's spend past its budget."""


class LLMClient:
    def __init__(self, model: ModelSpec, effort: str = "none", use_cache: bool = True,
                 max_tokens: int = MAX_TOKENS, temperature: float = 0.0, budget_usd: float | None = None) -> None:
        if model.provider != "openai-compatible":
            raise NotImplementedError(f"provider {model.provider!r} is not wired yet; use an openai-compatible model")
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}")
        load_dotenv(PROJECT_ROOT / ".env")
        key = os.environ.get(model.api_key_env)
        if not key:
            raise RuntimeError(f"{model.api_key_env} is not set; put it in {PROJECT_ROOT / '.env'}")
        from openai import OpenAI

        self.model = model
        self.effort = effort
        self.use_cache = use_cache
        # Reasoning shares max_tokens with the answer; leave room for both. (thinking_token_budget
        # is rejected by this model's backend, so max_tokens is the only ceiling.)
        self.max_tokens = max_tokens if effort == "none" else max(max_tokens, 16384)
        self.temperature = temperature
        self.usage = Usage()
        self.budget_usd = budget_usd
        self._client = OpenAI(base_url=model.base_url, api_key=key, max_retries=4, timeout=180)

    def _create_with_backoff(self, kwargs: dict):
        """The provider caps concurrent requests per model; a 429 means wait, not fail."""
        from openai import RateLimitError

        delay, waited = 1.0, 0.0
        while True:
            try:
                return self._client.chat.completions.create(**kwargs)
            except RateLimitError as e:
                if waited >= RATE_LIMIT_MAX_WAIT_S:
                    raise
                time.sleep(delay)
                waited += delay
                delay = min(delay * 2, 30.0)

    def _cache_path(self, payload: dict) -> Path:
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return CACHE_DIR / self.model.name / f"{digest}.json"

    def chat(self, messages: list[dict], json_mode: bool = True, temperature: float | None = None) -> LLMResponse:
        temperature = self.temperature if temperature is None else temperature
        payload = {
            "model": self.model.name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "reasoning_effort": self.effort,
            "json_mode": json_mode,
        }
        path = self._cache_path(payload)
        if self.use_cache and path.exists():
            data = json.loads(path.read_text())
            self.usage.add(Usage(cache_hits=1))
            return LLMResponse(data["text"], data["finish_reason"], Usage(cache_hits=1), True, 0.0)

        spent = self.usage.cost_usd(self.model)
        if self.budget_usd is not None and spent >= self.budget_usd:
            raise CostCapExceeded(f"spent ${spent:.2f} of the ${self.budget_usd:.2f} run budget")

        kwargs: dict = {
            "model": self.model.name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "reasoning_effort": self.effort,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        resp = self._create_with_backoff(kwargs)
        latency = time.perf_counter() - t0

        choice = resp.choices[0]
        text = choice.message.content or ""
        u = resp.usage
        details_p = getattr(u, "prompt_tokens_details", None)
        details_c = getattr(u, "completion_tokens_details", None)
        usage = Usage(
            input_tokens=u.prompt_tokens,
            output_tokens=u.completion_tokens,
            cached_input_tokens=getattr(details_p, "cached_tokens", 0) or 0,
            reasoning_tokens=getattr(details_c, "reasoning_tokens", 0) or 0,
            calls=1,
        )
        self.usage.add(usage)
        out = LLMResponse(text, choice.finish_reason or "stop", usage, False, latency)
        if self.use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": text, "finish_reason": out.finish_reason, "usage": usage.__dict__}))
        return out
