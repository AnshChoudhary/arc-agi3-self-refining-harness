"""Model names, pricing, and the cost cap.

Rule 6 (CLAUDE.md): the cheapest available model is the default. A stronger
model is only ever selected via an explicit CLI flag.
Prices are USD per million tokens (Anthropic first-party API, checked 2026-09).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_usd_per_mtok
            + output_tokens * self.output_usd_per_mtok
        ) / 1_000_000


MODELS: dict[str, ModelSpec] = {
    "haiku": ModelSpec("claude-haiku-4-5", 1.00, 5.00),
    "sonnet": ModelSpec("claude-sonnet-5", 2.00, 10.00),
    "opus": ModelSpec("claude-opus-5", 5.00, 25.00),
}

# Cheapest first-party model. Do not change this default; pass --model instead.
DEFAULT_MODEL: str = "haiku"

# scripts/eval.py prints the projected cost and aborts above this.
COST_CAP_USD: float = 5.00


def get_model(key: str = DEFAULT_MODEL) -> ModelSpec:
    if key in MODELS:
        return MODELS[key]
    raise KeyError(f"unknown model alias {key!r}; choose from {sorted(MODELS)}")
