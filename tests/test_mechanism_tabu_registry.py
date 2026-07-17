"""Tests for core/mechanism_tabu_registry.py."""
from __future__ import annotations

import json
from pathlib import Path

from core.mechanism_session_trace import MechanismSessionRecord, MechanismSessionTrace
from core.mechanism_tabu_registry import MechanismTabuRegistry


def _session(name: str, target: str, round: int = 1) -> MechanismSessionRecord:
    return MechanismSessionRecord(
        round=round,
        session_id="test",
        mechanism_name=name,
        implementation_strategy="new_helper_class",
        target=target,
        code_retries=0,
        hypothesis="test",
    )


class TestMechanismTabuRegistry:
    def test_not_tabu_initially(self):
        reg = MechanismTabuRegistry()
        blocked, reason = reg.is_tabu("tabu_search_manager", "TabuSearchManager", round_num=1)
        assert blocked is False
        assert reason == ""

    def test_record_failure_blocks_reproposal(self):
        reg = MechanismTabuRegistry(default_tenure=3)
        session = _session("tabu_search_manager", "TabuSearchManager")
        reg.record_failure(session, round_num=1, reason="import_fail")

        blocked, reason = reg.is_tabu("tabu_search_manager", "TabuSearchManager", round_num=2)
        assert blocked is True
        assert "import_fail" in reason

    def test_tabu_expires_after_tenure(self):
        reg = MechanismTabuRegistry(default_tenure=2)
        session = _session("gp_regressor", "GPRegressor")
        reg.record_failure(session, round_num=1, reason="missing_sklearn")

        assert reg.is_tabu("gp_regressor", "GPRegressor", round_num=2)[0] is True
        assert reg.is_tabu("gp_regressor", "GPRegressor", round_num=4)[0] is False

    def test_record_success_short_tenure(self):
        reg = MechanismTabuRegistry(default_tenure=4)
        session = _session("elite_pool_v2", "ElitePoolV2")
        reg.record_success(session, round_num=5)

        blocked, _ = reg.is_tabu("elite_pool_v2", "ElitePoolV2", round_num=6)
        assert blocked is True
        assert reg.is_tabu("elite_pool_v2", "ElitePoolV2", round_num=7)[0] is False

    def test_to_prompt_block_lists_entries(self):
        reg = MechanismTabuRegistry()
        reg.record_failure(_session("foo", "FooClass"), round_num=1, reason="duplicate")
        block = reg.to_prompt_block()
        assert "foo" in block
        assert "do not" in block.lower()

    def test_save_and_load_roundtrip(self, tmp_path):
        reg = MechanismTabuRegistry(max_size=5, default_tenure=3)
        reg.record_failure(_session("a", "A"), round_num=1, reason="fail")
        reg.record_failure(_session("b", "B"), round_num=2, reason="fail")

        path = tmp_path / "tabu.json"
        reg.save(path)

        loaded = MechanismTabuRegistry.load(path)
        assert loaded.max_size == 5
        assert len(loaded.entries) == 2
        assert loaded.is_tabu("a", "A", round_num=2)[0] is True

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert "entries" in raw
        assert raw["entries"][0]["mechanism_name"] == "a"

    def test_load_missing_file_returns_empty(self, tmp_path):
        reg = MechanismTabuRegistry.load(tmp_path / "missing.json")
        assert reg.entries == []

    def test_max_size_evicts_oldest(self):
        reg = MechanismTabuRegistry(max_size=2, default_tenure=10)
        reg.record_failure(_session("old", "Old"), round_num=1)
        reg.record_failure(_session("mid", "Mid"), round_num=2)
        reg.record_failure(_session("new", "New"), round_num=3)

        assert len(reg.entries) == 2
        assert reg.is_tabu("old", "Old", round_num=4)[0] is False
        assert reg.is_tabu("new", "New", round_num=4)[0] is True
