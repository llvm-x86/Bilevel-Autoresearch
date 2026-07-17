"""Search configuration for gpu_bench inner loop (Level 1.5 control surface)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchConfig:
    """Controls how the inner loop searches hyperparameters."""

    editable_params: list[str] = field(
        default_factory=lambda: ["LR", "WEIGHT_DECAY", "BATCH_SIZE", "HIDDEN_DIM"]
    )
    frozen_params: list[str] = field(default_factory=list)
    strategy: str = "explore"
    guidance: str = ""
    inner_budget: int = 5

    @property
    def active_params(self) -> list[str]:
        return [p for p in self.editable_params if p not in self.frozen_params]
