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
        l2_sessions: list,
        completed_outer_cycles: int,
        config=None,
    ):
        interval = config.level2_interval if config else self.level2_interval
        l3_interval = config.level3_interval if config else self.level3_interval
        batch_size = interval
        discard_threshold = 0.55

        fire_l2 = True
        fire_l3 = False
        reasons: list[str] = []

        recent = inner_trace[-self.lookback_iters :] if inner_trace else []
        if recent:
            n = len(recent)
            discards = sum(1 for r in recent if r.get("status") == "discard")
            keeps = sum(1 for r in recent if r.get("status") == "keep")
            discard_rate = discards / n

            keeps_streak = 0
            for record in reversed(recent):
                if record.get("status") == "keep":
                    keeps_streak += 1
                else:
                    break

            if discard_rate > discard_threshold:
                reasons.append(f"high discard rate ({discard_rate:.0%})")
                fire_l2 = True
            elif keeps == 0 and n >= 3:
                reasons.append("zero keeps in lookback window")
                fire_l2 = True
            elif keeps_streak >= 2 and completed_outer_cycles % interval != 0:
                fire_l2 = False
                reasons.append("improving streak; defer L2")
            elif completed_outer_cycles % interval != 0 and discard_rate < 0.5:
                fire_l2 = False
                reasons.append("inner loop improving; defer L2")

        if completed_outer_cycles % interval == 0:
            fire_l2 = True
            if "defer L2" in " ".join(reasons):
                reasons = [f"fixed interval ({interval} cycles) overrides defer"]
            elif not reasons:
                reasons.append(f"fixed L2 interval ({interval} cycles)")

        if l2_sessions:
            attempted = [s for s in l2_sessions if not getattr(s, "blocked_by_tabu", False)]
            if attempted:
                reverts = sum(
                    1 for s in attempted
                    if getattr(s, "applied", False) and getattr(s, "validated", None) is False
                )
                revert_rate = reverts / len(attempted)
                if revert_rate >= self.revert_rate_threshold:
                    fire_l3 = True
                    reasons.append(f"L2 revert rate {revert_rate:.0%}")

            consecutive_fail = 0
            for s in reversed(l2_sessions):
                if getattr(s, "applied", False):
                    break
                consecutive_fail += 1
            if consecutive_fail >= 2:
                fire_l3 = True
                reasons.append(f"{consecutive_fail} consecutive L2 failures")

            names = [getattr(s, "mechanism_name", "") for s in l2_sessions[-3:]]
            if len(names) >= 2 and names[-1] == names[-2]:
                fire_l3 = True
                reasons.append("duplicate mechanism name in consecutive rounds")

        l2_rounds = len(l2_sessions)
        if l2_rounds > 0 and l2_rounds % l3_interval == 0:
            fire_l3 = True
            reasons.append(f"L3 interval ({l3_interval} L2 rounds)")

        if not reasons:
            reasons.append("default schedule")

        return ScheduleDecision(
            fire_level2=fire_l2,
            fire_level3=fire_l3,
            reason="; ".join(reasons),
            batch_size=batch_size,
        )

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
