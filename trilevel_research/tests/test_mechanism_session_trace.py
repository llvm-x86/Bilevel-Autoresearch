"""Tests for core/mechanism_session_trace.py using C1 fixture sessions."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trilevel_research.core.mechanism_session_trace import (
    MechanismSessionRecord,
    MechanismSessionTrace,
    MechanismSessionTraceBuilder,
    parse_session_dir,
)

FIXTURE_ROOT = (
    Path(__file__).resolve().parent.parent.parent
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

    def test_to_dict(self, report):
        record = parse_session_dir(SESSIONS_ROOT / "round_1", round=1)
        data = record.to_dict()
        assert data["round"] == 1
        assert "mechanism_name" in data
        assert "session_dir" in data

    def test_empty_trace_rates(self):
        trace = MechanismSessionTrace()
        assert trace.revert_rate() == 0.0
        assert trace.apply_rate() == 0.0
        assert trace.mechanism_names() == []

    def test_parse_round_from_dir_name(self, report):
        record = parse_session_dir(SESSIONS_ROOT / "round_2", report_entry=None)
        assert record.round == 2

    def test_code_from_final_when_no_attempts(self, tmp_path):
        session = tmp_path / "round_99"
        session.mkdir()
        (session / "06_summary.json").write_text(
            json.dumps({"session_id": "x", "mechanism_name": "m"}),
            encoding="utf-8",
        )
        (session / "04_code_final.py").write_text("x = 42\n", encoding="utf-8")
        record = parse_session_dir(session)
        assert record.code.strip() == "x = 42"

    def test_patched_mechanism_research_fallback(self, tmp_path):
        session = tmp_path / "round_1"
        session.mkdir()
        (session / "06_summary.json").write_text("{}", encoding="utf-8")
        (session / "05_patched_mechanism_research.py").write_text(
            "# patched mech\n", encoding="utf-8"
        )
        record = parse_session_dir(session)
        assert "patched mech" in record.patched_artifact

    def test_build_trace_wrapper(self, report):
        trace = MechanismSessionTraceBuilder.build_trace(SESSIONS_ROOT, report=report)
        assert len(trace.sessions) == 2

    def test_report_only_session_without_dir(self):
        records = MechanismSessionTraceBuilder.build(
            Path("/nonexistent/sessions"),
            report_sessions=[
                {
                    "round": 5,
                    "mechanism_name": "orphan_mech",
                    "strategy": "replace_method",
                    "target": "Orphan",
                    "applied": False,
                    "blocked_by_tabu": True,
                    "error": "tabu hit",
                }
            ],
        )
        assert len(records) == 1
        assert records[0].mechanism_name == "orphan_mech"
        assert records[0].blocked_by_tabu is True

    def test_to_text_status_variants(self):
        records = [
            MechanismSessionRecord(
                round=1,
                session_id="s1",
                mechanism_name="m1",
                implementation_strategy="new_helper_class",
                target="T1",
                applied=False,
                validated=True,
                blocked_by_tabu=True,
            ),
            MechanismSessionRecord(
                round=2,
                session_id="s2",
                mechanism_name="m2",
                implementation_strategy="new_helper_class",
                target="T2",
                applied=True,
                validated=False,
                error="boom",
            ),
        ]
        text = MechanismSessionTraceBuilder.to_text(records)
        assert "blocked_by_tabu" in text
        assert "error: boom" in text

    def test_to_text_with_inner_trace(self):
        text = MechanismSessionTraceBuilder.to_text(
            [],
            inner_trace=[
                {"status": "keep"},
                {"status": "discard"},
                {"status": "crash"},
            ],
        )
        assert "Inner loop stats" in text
        assert "keeps: 1" in text
        assert "no Level-2 sessions" in text

    def test_import_failed_status_without_error(self):
        records = [
            MechanismSessionRecord(
                round=1,
                session_id="s1",
                mechanism_name="m1",
                implementation_strategy="new_helper_class",
                target="T1",
                applied=True,
                validated=False,
            ),
        ]
        text = MechanismSessionTraceBuilder.to_text(records)
        assert "import_failed" in text

    def test_invalid_round_dir_name_uses_fallback(self, tmp_path):
        bad_dir = tmp_path / "round_x"
        bad_dir.mkdir()
        (bad_dir / "06_summary.json").write_text("{}", encoding="utf-8")
        records = MechanismSessionTraceBuilder.build(tmp_path)
        assert len(records) == 1
        assert records[0].round == 1
