"""Tests for domains/train_opt/mechanism_research_config.py."""
from __future__ import annotations

from trilevel_research.config import MechanismResearchConfig


class TestMechanismResearchConfig:
    def test_apply_patch_known_fields(self):
        cfg = MechanismResearchConfig()
        cfg.apply_patch({"level2_interval": 7, "enable_tabu": False})
        assert cfg.level2_interval == 7
        assert cfg.enable_tabu is False

    def test_apply_patch_unknown_goes_to_extra(self):
        cfg = MechanismResearchConfig()
        cfg.apply_patch({"custom_flag": True})
        assert cfg.extra["custom_flag"] is True

    def test_to_dict_roundtrip(self):
        cfg = MechanismResearchConfig(level3_interval=5, extra={"x": 1})
        data = cfg.to_dict()
        assert data["level3_interval"] == 5
        assert data["extra"] == {"x": 1}

    def test_from_dict(self):
        cfg = MechanismResearchConfig.from_dict(
            {"level2_interval": 3, "extra": {"note": "test"}}
        )
        assert cfg.level2_interval == 3
        assert cfg.extra["note"] == "test"
