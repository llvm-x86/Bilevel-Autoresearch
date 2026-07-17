"""Adaptive scheduling for Level-2 and Level-3 mechanism research."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from trilevel_research.config import MechanismResearchConfig
from trilevel_research.core.mechanism_session_trace import MechanismSessionRecord

__all__ = [
    "AdaptiveMechanismSchedule",
    "ScheduleDecision",
]


@dataclass
class ScheduleDecision:
    fire_level2: bool
    fire_level3: bool
    reason: str
    batch_size: int | None = None


@dataclass
class AdaptiveMechanismSchedule:
    """Decide when to fire L2/L3 based on inner trace and L2 session history."""

    level2_interval: int = 2
    level3_interval: int = 2
    discard_rate_threshold: float = 0.70
    revert_rate_threshold: float = 0.50
    lookback_iters: int = 10

    def decide(
        self,
        inner_trace: list[dict],
        l2_sessions: list[MechanismSessionRecord],
        completed_outer_cycles: int,
        config: MechanismResearchConfig | None = None,
    ) -> ScheduleDecision:
        interval = config.level2_interval if config else self.level2_interval
        if not inner_trace or len(inner_trace) < 2:
            self.fire_level2 = False
            self.fire_level3 = False
            return ScheduleDecision.STAY
        recent_configs = [entry.get("config", {}) for entry in inner_trace[-self.window:]]
        recent_metrics = [entry.get("metric", float('inf')) for entry in inner_trace[-self.window:] if entry.get("metric") is not None]
        best_metric = min(recent_metrics) if recent_metrics else float('inf')
        cascade_result = self.detect_value_cascade(recent_configs, recent_metrics, best_metric)
        if cascade_result.is_cascade:
            self.fire_level2 = True
            self.fire_level3 = False
            return ScheduleDecision.LEVEL2
        self.fire_level2 = False
        self.fire_level3 = False
        return ScheduleDecision.STAY

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "level2_interval": self.level2_interval,
            "level3_interval": self.level3_interval,
            "discard_rate_threshold": self.discard_rate_threshold,
            "revert_rate_threshold": self.revert_rate_threshold,
            "lookback_iters": self.lookback_iters,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> AdaptiveMechanismSchedule:
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{k: data[k] for k in data if k in cls.__dataclass_fields__})

    def stats(self) -> dict:
        return {
            "level2_interval": self.level2_interval,
            "level3_interval": self.level3_interval,
        }
