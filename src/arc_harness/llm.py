"""Token accounting shared by every agent. The Anthropic client itself lands in M1."""

from __future__ import annotations

from dataclasses import dataclass

from config.models import ModelSpec


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens

    def cost_usd(self, model: ModelSpec | None) -> float:
        return model.cost_usd(self.input_tokens, self.output_tokens) if model else 0.0
