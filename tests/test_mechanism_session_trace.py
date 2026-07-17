"""Tests for core/mechanism_session_trace.py using C1 fixture sessions."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.mechanism_session_trace import (
    MechanismSessionTrace,
    MechanismSessionTraceBuilder,
    parse_session_dir,
)

FIXTURE_ROOT = (
    Path(__file__).resolve().parent.parent
    / "experiments/ablations/paper_ablation/run2_results/results_C/C1"
)
SESSIONS_ROOT = FIXTURE_ROOT / "mechanism_sessions"
REPORT_PATH = FIXTURE_ROOT / "report.json"


@pytest.fixture
def report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


class TestParseSessionDir:
    def test_round_1_summary_fields(self, report):
        session_dir = SESSIONS_ROOT / "round_1"
        record = parse_session_dir(
            session_dir,
            round=1,
            report_entry=report["level2_sessions"][0],
        )

        assert record.session_id == "20260323_115346"
        assert record.mechanism_name == "generated_mechanism_20260323_115346"
        assert record.implementation_strategy == "new_helper_class"
        assert record.target == "GeneratedMechanism_20260323_115346"
        assert record.code_retries == 0
        assert record.round == 1
        assert len(record.exploration) > 100
        assert len(record.critique) > 100
        assert "tabu_search_manager" in record.spec.lower() or "Tabu" in record.spec
        assert len(record.code) > 50
        assert record.applied is True
        assert record.validated is False

    def test_round_2_parses(self, report):
        session_dir = SESSIONS_ROOT / "round_2"
        record = parse_session_dir(
            session_dir,
            round=2,
            report_entry=report["level2_sessions"][1],
        )
        assert record.session_id == "20260323_130143"
        assert record.round == 2
        assert record.patched_artifact  # 05_patched_runner.py exists

    def test_missing_summary_raises(self, tmp_path):
        empty = tmp_path / "empty_session"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            parse_session_dir(empty, round=1)


class TestMechanismSessionTraceBuilder:
    def test_build_all_rounds(self, report):
        trace = MechanismSessionTrace(
            sessions=MechanismSessionTraceBuilder.build(SESSIONS_ROOT, report=report)
        )
        assert len(trace.sessions) == 2
        assert trace.sessions[0].round == 1
        assert trace.sessions[1].round == 2

    def test_to_text_includes_mechanism_names(self, report):
        trace = MechanismSessionTrace(
            sessions=MechanismSessionTraceBuilder.build(SESSIONS_ROOT, report=report)
        )
        text = trace.to_text()
        assert "generated_mechanism_20260323_115346" in text
        assert "L2 Session Trace" in text

    def test_revert_rate_from_report(self, report):
        trace = MechanismSessionTrace(
            sessions=MechanismSessionTraceBuilder.build(SESSIONS_ROOT, report=report)
        )
        # Both sessions applied=True, validated=False → 100% revert rate
        assert trace.revert_rate() == 1.0
        assert trace.apply_rate() == 1.0

    def test_mechanism_names(self, report):
        trace = MechanismSessionTrace(
            sessions=MechanismSessionTraceBuilder.build(SESSIONS_ROOT, report=report)
        )
        names = trace.mechanism_names()
        assert len(names) == 2
        assert all(n.startswith("generated_mechanism_") for n in names)
