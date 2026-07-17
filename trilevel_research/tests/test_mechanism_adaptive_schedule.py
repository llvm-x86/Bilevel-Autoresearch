"""Tests for core/adaptive_mechanism_schedule.py."""
from __future__ import annotations

from trilevel_research.config import MechanismResearchConfig
from trilevel_research.core.adaptive_mechanism_schedule import (
    AdaptiveMechanismSchedule,
    ScheduleDecision,
)
from trilevel_research.core.mechanism_session_trace import MechanismSessionRecord


def _session(
    *,
    round: int = 1,
    name: str = "mech_a",
    applied: bool = True,
    validated: bool | None = True,
    blocked: bool = False,
) -> MechanismSessionRecord:
    return MechanismSessionRecord(
        round=round,
        session_id=f"round_{round}",
        mechanism_name=name,
        implementation_strategy="new_helper_class",
        target=f"Target_{round}",
        applied=applied,
        validated=validated,
        blocked_by_tabu=blocked,
    )


class TestAdaptiveMechanismSchedule:
    def test_default_schedule_no_history(self):
        sched = AdaptiveMechanismSchedule()
        decision = sched.decide([], [], completed_outer_cycles=1)
        assert isinstance(decision, ScheduleDecision)
        assert decision.fire_level2 is True
        assert decision.fire_level3 is False
        assert "default schedule" in decision.reason

    def test_high_discard_rate_triggers_l2(self):
        sched = AdaptiveMechanismSchedule(discard_rate_threshold=0.5)
        inner = [{"status": "discard"}] * 8 + [{"status": "keep"}] * 2
        decision = sched.decide(inner, [], completed_outer_cycles=1)
        assert decision.fire_level2 is True
        assert "high discard rate" in decision.reason

    def test_zero_keeps_in_lookback(self):
        sched = AdaptiveMechanismSchedule()
        inner = [{"status": "crash"}] * 5
        decision = sched.decide(inner, [], completed_outer_cycles=1)
        assert "zero keeps" in decision.reason

    def test_defer_l2_when_improving(self):
        sched = AdaptiveMechanismSchedule(level2_interval=3)
        inner = [{"status": "keep"}] * 6 + [{"status": "discard"}]
        decision = sched.decide(inner, [], completed_outer_cycles=1)
        assert decision.fire_level2 is False
        assert "defer L2" in decision.reason

    def test_fixed_interval_overrides_defer(self):
        sched = AdaptiveMechanismSchedule(level2_interval=2)
        inner = [{"status": "keep"}] * 6
        decision = sched.decide(inner, [], completed_outer_cycles=2)
        assert decision.fire_level2 is True
        assert "fixed" in decision.reason.lower()

    def test_fixed_interval_at_cycle_zero(self):
        sched = AdaptiveMechanismSchedule(level2_interval=2)
        decision = sched.decide([], [], completed_outer_cycles=2)
        assert decision.fire_level2 is True
        assert "fixed L2 interval" in decision.reason

    def test_revert_rate_triggers_l3(self):
        sched = AdaptiveMechanismSchedule(revert_rate_threshold=0.5)
        sessions = [
            _session(round=1, validated=False),
            _session(round=2, validated=False),
        ]
        decision = sched.decide([], sessions, completed_outer_cycles=2)
        assert decision.fire_level3 is True
        assert "revert rate" in decision.reason

    def test_blocked_sessions_excluded_from_revert_rate(self):
        sched = AdaptiveMechanismSchedule(revert_rate_threshold=0.5)
        sessions = [
            _session(round=1, validated=False, blocked=True),
        ]
        decision = sched.decide([], sessions, completed_outer_cycles=2)
        assert decision.fire_level3 is False

    def test_consecutive_l2_failures_trigger_l3(self):
        sched = AdaptiveMechanismSchedule()
        sessions = [
            _session(round=1, applied=False),
            _session(round=2, applied=False),
        ]
        decision = sched.decide([], sessions, completed_outer_cycles=2)
        assert decision.fire_level3 is True
        assert "consecutive L2 failure" in decision.reason

    def test_duplicate_mechanism_name_triggers_l3(self):
        sched = AdaptiveMechanismSchedule()
        sessions = [
            _session(round=1, name="same_mech"),
            _session(round=2, name="same_mech"),
        ]
        decision = sched.decide([], sessions, completed_outer_cycles=2)
        assert decision.fire_level3 is True
        assert "duplicate mechanism" in decision.reason

    def test_l3_interval_on_l2_round_count(self):
        sched = AdaptiveMechanismSchedule(level3_interval=2)
        sessions = [_session(round=1), _session(round=2)]
        decision = sched.decide([], sessions, completed_outer_cycles=2)
        assert decision.fire_level3 is True
        assert "L3 interval" in decision.reason

    def test_config_overrides_intervals(self):
        sched = AdaptiveMechanismSchedule(level2_interval=5)
        cfg = MechanismResearchConfig(level2_interval=2, level3_interval=1)
        decision = sched.decide([], [], completed_outer_cycles=2, config=cfg)
        assert decision.batch_size == 2

    def test_save_and_load_roundtrip(self, tmp_path):
        sched = AdaptiveMechanismSchedule(
            level2_interval=4,
            level3_interval=3,
            discard_rate_threshold=0.8,
        )
        path = tmp_path / "schedule.json"
        sched.save(path)

        loaded = AdaptiveMechanismSchedule.load(path)
        assert loaded.level2_interval == 4
        assert loaded.level3_interval == 3
        assert loaded.discard_rate_threshold == 0.8

    def test_load_missing_returns_defaults(self, tmp_path):
        loaded = AdaptiveMechanismSchedule.load(tmp_path / "missing.json")
        assert loaded.level2_interval == 2

    def test_stats(self):
        sched = AdaptiveMechanismSchedule(level2_interval=7, level3_interval=5)
        assert sched.stats() == {"level2_interval": 7, "level3_interval": 5}
