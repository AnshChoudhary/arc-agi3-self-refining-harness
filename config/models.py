"""Model names, pricing, cost cap, and the referee-fixed student knobs.

Rule 6 (CLAUDE.md): the cheapest available model is the default. A stronger
model or deeper reasoning is only ever selected via an explicit CLI flag.
Prices are USD per million tokens, read from each provider (checked 2026-09-12).
"""

from dataclasses import dataclass

NEURALWATT_BASE_URL = "https://api.neuralwatt.com/v1"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str  # "openai-compatible" or "anthropic"
    base_url: str | None
    api_key_env: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cached_input_usd_per_mtok: float
    context_tokens: int

    def cost_usd(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
        fresh = max(input_tokens - cached_input_tokens, 0)
        return (
            fresh * self.input_usd_per_mtok
            + cached_input_tokens * self.cached_input_usd_per_mtok
            + output_tokens * self.output_usd_per_mtok
        ) / 1_000_000


MODELS: dict[str, ModelSpec] = {
    # NeuralWatt (OpenAI-compatible). Reasoning is off by default on this model; thinking tokens bill as output.
    "deepseek-flash": ModelSpec("deepseek-v4-flash", "openai-compatible", NEURALWATT_BASE_URL,
                                "NEURALWATT_API_KEY", 0.14, 0.28, 0.028, 1_048_560),
    "gemma": ModelSpec("gemma-4-31b", "openai-compatible", NEURALWATT_BASE_URL,
                       "NEURALWATT_API_KEY", 0.144, 0.42, 0.0144, 262_128),
    # Anthropic first-party (client not wired yet; listed so the cost table is complete).
    "haiku": ModelSpec("claude-haiku-4-5", "anthropic", None, "ANTHROPIC_API_KEY", 1.00, 5.00, 0.10, 200_000),
    "sonnet": ModelSpec("claude-sonnet-5", "anthropic", None, "ANTHROPIC_API_KEY", 2.00, 10.00, 0.20, 1_000_000),
    "opus": ModelSpec("claude-opus-5", "anthropic", None, "ANTHROPIC_API_KEY", 5.00, 25.00, 0.50, 1_000_000),
}

# Cheapest model and cheapest reasoning setting. Do not change these defaults; pass --model / --effort.
DEFAULT_MODEL: str = "deepseek-flash"
DEFAULT_EFFORT: str = "none"
EFFORTS = ("none", "high", "max")

# scripts/eval.py prints the projected cost and aborts above this.
COST_CAP_USD: float = 10.00

# Student loop knobs. These belong to the referee (fixed across runs), not to the harness.
MAX_TOKENS: int = 2048  # JSON replies are short; raised automatically when reasoning is on
MAX_ANALYSIS_CALLS_PER_ACTION: int = 4  # free REPL calls before the student must act
MAX_BAD_REPLIES: int = 3  # unparseable / invalid / repeated replies in a row before the episode is abandoned
MAX_CALLS_PER_ACTION: int = MAX_ANALYSIS_CALLS_PER_ACTION + MAX_BAD_REPLIES + 1  # hard stop per env step
HISTORY_WINDOW: int = 6  # past env steps kept verbatim in the prompt
RETRY_TEMPERATURE: float = 0.7  # after a bad reply; breaks deterministic repeat loops

# Cost projection per budgeted action. Calibrated on the first full-budget student run
# (ls20 level 1, effort none: 139 calls / 110 actions, 6.4k in + 145 out tokens per call);
# a 1.5x margin is applied. Reasoning effort adds output tokens on top.
EST_CALLS_PER_ACTION: float = 1.9
EST_INPUT_TOKENS_PER_CALL: int = 6_500
EST_OUTPUT_TOKENS_PER_CALL: int = 250


def get_model(key: str = DEFAULT_MODEL) -> ModelSpec:
    if key in MODELS:
        return MODELS[key]
    raise KeyError(f"unknown model alias {key!r}; choose from {sorted(MODELS)}")
