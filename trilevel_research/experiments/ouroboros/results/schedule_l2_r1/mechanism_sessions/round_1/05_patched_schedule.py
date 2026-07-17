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

    import math

    class AdaptiveMechanismSchedule:
        def __init__(self, volatility_window: int = 5, volatility_threshold: float = 0.3, research_budget: int = 2, **kwargs):
            super().__init__(**kwargs)
            self.volatility_window = volatility_window
            self.volatility_threshold = volatility_threshold
            self.research_budget = research_budget
            self._volatility_gate_open = True
            self._consecutive_l2_after_gate = 0

        def _calculate_volatility(self, recent_scores: list[float]) -> float:
            clean_scores = [s for s in recent_scores if s is not None and not (isinstance(s, float) and math.isnan(s))]
            if len(clean_scores) < 2:
                return 0.0
            mean = sum(clean_scores) / len(clean_scores)
            if mean == 0:
                return 0.0
            variance = sum((x - mean) ** 2 for x in clean_scores) / len(clean_scores)
            std_dev = math.sqrt(variance)
            return std_dev / abs(mean)

        def decide(self, inner_trace, l2_sessions, completed_outer_cycles, config=None):
            interval = config.level2_interval if config else self.level2_interval
            l3_interval = config.level3_interval if config else self.level3_interval
            batch_size = interval

            fire_l2 = True
            fire_l3 = False
            reasons = []

            recent = inner_trace[-self.lookback_iters:] if inner_trace else []
            if recent:
                n = len(recent)
                discards = sum(1 for r in recent if r.get("status") == "discard")
                keeps = sum(1 for r in recent if r.get("status") == "keep")
                discard_rate = discards / n
                if discard_rate > self.discard_rate_threshold:
                    reasons.append(f"high discard rate ({discard_rate:.0%})")
                    fire_l2 = True
                elif keeps == 0 and n >= 3:
                    reasons.append("zero keeps in lookback window")
                    fire_l2 = True
                elif completed_outer_cycles % interval != 0 and discard_rate < 0.5:
                    fire_l2 = False
                    reasons.append("inner loop improving; defer L2")

            # Volatility gate logic (placed after existing reasons but before fixed-interval override)
            recent_scores = [r.get("score") for r in recent if r.get("score") is not None]
            cv = self._calculate_volatility(recent_scores)
            volatility_gate_open = cv >= self.volatility_threshold or len(recent_scores) < 2

            if not self._volatility_gate_open and volatility_gate_open:
                self._consecutive_l2_after_gate = 0

            if volatility_gate_open:
                self._consecutive_l2_after_gate += 1
                if self._consecutive_l2_after_gate > self.research_budget:
                    fire_l2 = False
                    reasons.append(f"volatility budget exhausted ({self._consecutive_l2_after_gate} > {self.research_budget})")
                    self._volatility_gate_open = False
                    self._consecutive_l2_after_gate = 0
            else:
                fire_l2 = False
                reasons.append(f"low volatility (CV={cv:.2f} < {self.volatility_threshold}); suppressing L2")

            self._volatility_gate_open = volatility_gate_open

            # Fixed-interval override (modified to respect volatility gate)
            if completed_outer_cycles % interval == 0:
                if not any("volatility" in r for r in reasons):
                    fire_l2 = True
                    if "defer L2" in " ".join(reasons):
                        reasons = [f"fixed interval ({interval} cycles) overrides defer"]
                    elif not reasons:
                        reasons.append(f"fixed L2 interval ({interval} cycles)")
                else:
                    reasons.append(f"volatility gate overrides fixed interval; L2 suppressed")

            if l2_sessions:
                attempted = [s for s in l2_sessions if not s.blocked_by_tabu]
                if attempted:
                    reverts = sum(1 for s in attempted if s.applied and s.validated is False)
                    revert_rate = reverts / len(attempted)
                    if revert_rate >= self.revert_rate_threshold:
                        fire_l2 = True
                        reasons.append(f"high revert rate ({revert_rate:.0%})")

            from trilevel_research.core.mechanism_research_config import MechanismResearchConfig
            return ScheduleDecision(fire_l2=fire_l2, fire_l3=fire_l3, reasons=reasons)

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
