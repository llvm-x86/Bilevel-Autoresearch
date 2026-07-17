"""Configuration surface for Level-2/Level-3 mechanism research."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MechanismResearchConfig:
    """Mutable config that Level 3 may tune via schedule patches."""

    level2_interval: int = 2
    level3_interval: int = 2
    max_code_retries: int = 3
    tabu_max_size: int = 20
    tabu_default_tenure: int = 3
    enable_level3: bool = False
    enable_tabu: bool = True
    enable_adaptive_schedule: bool = True
    validation_strict: bool = True
    extra: dict = field(default_factory=dict)

    def apply_patch(self, patch: dict) -> None:
        """Apply a schedule/config patch from an L3 session."""
        for key, value in patch.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.extra[key] = value

    def to_dict(self) -> dict:
        return {
            "level2_interval": self.level2_interval,
            "level3_interval": self.level3_interval,
            "max_code_retries": self.max_code_retries,
            "tabu_max_size": self.tabu_max_size,
            "tabu_default_tenure": self.tabu_default_tenure,
            "enable_level3": self.enable_level3,
            "enable_tabu": self.enable_tabu,
            "enable_adaptive_schedule": self.enable_adaptive_schedule,
            "validation_strict": self.validation_strict,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MechanismResearchConfig:
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data and k != "extra"}
        extra = data.get("extra", {})
        return cls(**known, extra=extra)
