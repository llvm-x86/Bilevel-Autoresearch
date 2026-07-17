"""Tests for schedule_mechanism_research.py (ouroboros L2 schedule patches)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from trilevel_research.core.adaptive_mechanism_schedule import (
    AdaptiveMechanismSchedule,
)
from trilevel_research.domains.gpu_bench_opt.schedule_mechanism_research import (
    BOOTSTRAP_DECIDE_CODE,
    ScheduleMechanismResearcher,
    ScheduleMechanismResult,
    VALIDATE_FIXTURE_TRACE,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CANONICAL_SCHEDULE = (
    REPO_ROOT / "trilevel_research" / "core" / "adaptive_mechanism_schedule.py"
)


class TestScheduleValidate:
    def setup_method(self):
        self.researcher = ScheduleMechanismResearcher(api_key="mock")

    def test_validate_canonical_schedule(self):
        assert self.researcher.validate(CANONICAL_SCHEDULE) is True

    def test_validate_fixture_trace_returns_bools(self, tmp_path):
        schedule = tmp_path / "schedule.py"
        schedule.write_text(CANONICAL_SCHEDULE.read_text(encoding="utf-8"), encoding="utf-8")
        assert self.researcher.validate(schedule) is True


class TestExtractMethodBody:
    def setup_method(self):
        self.researcher = ScheduleMechanismResearcher(api_key="mock")

    def test_extracts_top_level_method(self):
        code = textwrap.dedent(
            '''\
            def decide(self, inner_trace, l2_sessions, completed_outer_cycles, config=None):
                return ScheduleDecision(fire_level2=True, fire_level3=False, reason="ok")
            '''
        )
        body = self.researcher._extract_method_body(code, "decide")
        assert body.startswith("def decide(")
        assert "fire_level2" in body

    def test_extracts_decide_from_helper_class_fragment(self):
        """LLM often emits helper class + decide; extract decide only."""
        code = textwrap.dedent(
            '''\
            class ValueCascadeAnalyzer:
                def __init__(self):
                    self.threshold = 0.85

                def detect(self):
                    return False

            def decide(self, inner_trace, l2_sessions, completed_outer_cycles, config=None):
                fire_level2 = True
                fire_level3 = False
                return ScheduleDecision(
                    fire_level2=fire_level2,
                    fire_level3=fire_level3,
                    reason="ok",
                    batch_size=2,
                )
            '''
        )
        body = self.researcher._extract_method_body(code, "decide")
        assert body.startswith("def decide(")
        assert "ValueCascadeAnalyzer" not in body
        assert "fire_level2" in body

    def test_rejects_full_class_without_target_method(self):
        code = textwrap.dedent(
            '''\
            class AdaptiveMechanismSchedule:
                def other(self):
                    pass
            '''
        )
        with pytest.raises(ValueError, match="redefinition must not"):
            self.researcher._extract_method_body(code, "decide")

    def test_extracts_method_from_nested_class(self):
        code = textwrap.dedent(
            '''\
            class AdaptiveMechanismSchedule:
                def decide(self, a, b, c, config=None):
                    fire_level2 = True
                    fire_level3 = False
                    return ScheduleDecision(fire_level2=fire_level2, fire_level3=fire_level3, reason="x")
            '''
        )
        body = self.researcher._extract_method_body(code, "decide")
        assert "fire_level2" in body
        assert "class AdaptiveMechanismSchedule" not in body.split("def decide")[0]


class TestCodegenSchema:
    def setup_method(self):
        self.researcher = ScheduleMechanismResearcher(api_key="mock")

    def test_valid_decide_passes_schema(self):
        code = textwrap.dedent(
            '''\
            def decide(self, inner_trace, l2_sessions, completed_outer_cycles, config=None):
                return ScheduleDecision(fire_level2=True, fire_level3=False, reason="ok")
            '''
        )
        assert self.researcher._validate_codegen_schema(code, "replace_method", "decide") is None

    def test_missing_fire_fields_fails_schema(self):
        code = "def decide(self, a, b, c, config=None):\n    return ScheduleDecision(reason='x')\n"
        err = self.researcher._validate_codegen_schema(code, "replace_method", "decide")
        assert err is not None
        assert "fire_level2" in err or "fire_level3" in err


class TestBootstrapPatch:
    def setup_method(self):
        self.researcher = ScheduleMechanismResearcher(api_key="mock")

    def test_bootstrap_schema_valid(self):
        err = self.researcher._validate_codegen_schema(
            BOOTSTRAP_DECIDE_CODE, "replace_method", "decide"
        )
        assert err is None

    def test_bootstrap_apply_and_validate(self, tmp_path):
        schedule = tmp_path / "adaptive_mechanism_schedule.py"
        schedule.write_text(CANONICAL_SCHEDULE.read_text(encoding="utf-8"), encoding="utf-8")
        result = ScheduleMechanismResult(
            session_id="bootstrap_test",
            hypothesis="bootstrap",
            mechanism_name="bootstrap_improving_streak_defer",
            implementation_strategy="replace_method",
            target="decide",
            spec="",
            code=BOOTSTRAP_DECIDE_CODE,
            session_dir=tmp_path / "session",
        )
        result.session_dir.mkdir(parents=True, exist_ok=True)
        assert self.researcher.apply(schedule, result) is True
        assert self.researcher.validate(schedule) is True

    def test_bootstrap_defers_l2_on_improving_streak(self, tmp_path):
        schedule = tmp_path / "adaptive_mechanism_schedule.py"
        schedule.write_text(CANONICAL_SCHEDULE.read_text(encoding="utf-8"), encoding="utf-8")
        result = ScheduleMechanismResult(
            session_id="bootstrap_test",
            hypothesis="bootstrap",
            mechanism_name="bootstrap_improving_streak_defer",
            implementation_strategy="replace_method",
            target="decide",
            spec="",
            code=BOOTSTRAP_DECIDE_CODE,
            session_dir=tmp_path / "session",
        )
        result.session_dir.mkdir(parents=True, exist_ok=True)
        self.researcher.apply(schedule, result)
        assert self.researcher.validate(schedule) is True

        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("_boot_sched", schedule)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        inst = mod.AdaptiveMechanismSchedule()
        decision = inst.decide(VALIDATE_FIXTURE_TRACE, [], completed_outer_cycles=3)
        assert decision.fire_level2 is False
        assert "improving streak" in decision.reason.lower()


class TestNormalizeCodegen:
    def setup_method(self):
        self.researcher = ScheduleMechanismResearcher(api_key="mock")

    def test_normalize_strips_class_wrapper(self):
        wrapped = textwrap.dedent(
            '''\
            class AdaptiveMechanismSchedule:
                def decide(self, inner_trace, l2_sessions, completed_outer_cycles, config=None):
                    return ScheduleDecision(fire_level2=True, fire_level3=False, reason="ok")
            '''
        )
        normalized = self.researcher._normalize_codegen(wrapped, "replace_method", "decide")
        assert normalized.startswith("def decide(")
        assert "class AdaptiveMechanismSchedule" not in normalized


class TestSanitizeDecideCode:
    def setup_method(self):
        self.researcher = ScheduleMechanismResearcher(api_key="mock")

    def test_sanitize_replaces_unknown_self_attr(self):
        code = textwrap.dedent(
            '''\
            def decide(self, inner_trace, l2_sessions, completed_outer_cycles, config=None):
                fire_level2 = True
                fire_level3 = False
                low = self.min_discard_rate_threshold
                high = self.discard_rate_threshold
                if low > high:
                    fire_level2 = False
                return ScheduleDecision(
                    fire_level2=fire_level2,
                    fire_level3=fire_level3,
                    reason="test",
                    batch_size=self.level2_interval,
                )
            '''
        )
        sanitized = self.researcher._sanitize_decide_code(code)
        assert "min_discard_rate_threshold" not in sanitized
        assert "self.discard_rate_threshold" in sanitized

    def test_sanitize_wraps_l2_session_attr_access(self):
        code = textwrap.dedent(
            '''\
            def decide(self, inner_trace, l2_sessions, completed_outer_cycles, config=None):
                fire_level2 = True
                fire_level3 = False
                for s in l2_sessions:
                    if s.applied and s.validated is False:
                        fire_level3 = True
                return ScheduleDecision(
                    fire_level2=fire_level2,
                    fire_level3=fire_level3,
                    reason="test",
                )
            '''
        )
        sanitized = self.researcher._sanitize_decide_code(code)
        assert "getattr(s, 'applied'" in sanitized
        assert "getattr(s, 'validated'" in sanitized


ATTEMPT2_CODE_PATH = (
    REPO_ROOT
    / "trilevel_research"
    / "experiments"
    / "bilevel_improves_trilevel"
    / "results"
    / "ouroboros"
    / "schedule_l2_attempt_2"
    / "mechanism_sessions"
    / "round_1"
    / "04_code_attempt_1.py"
)


class TestAttempt2SamplePatch:
    def setup_method(self):
        self.researcher = ScheduleMechanismResearcher(api_key="mock")

    @pytest.mark.skipif(not ATTEMPT2_CODE_PATH.exists(), reason="attempt 2 fixture missing")
    def test_attempt2_code_validates_after_fixes(self, tmp_path):
        raw_code = ATTEMPT2_CODE_PATH.read_text(encoding="utf-8")
        schedule = tmp_path / "adaptive_mechanism_schedule.py"
        schedule.write_text(CANONICAL_SCHEDULE.read_text(encoding="utf-8"), encoding="utf-8")
        result = ScheduleMechanismResult(
            session_id="attempt2_test",
            hypothesis="attempt2 fixture",
            mechanism_name="schedule_mechanism_attempt2",
            implementation_strategy="replace_method",
            target="decide",
            spec="",
            code=raw_code,
            session_dir=tmp_path / "session",
        )
        result.session_dir.mkdir(parents=True, exist_ok=True)
        assert self.researcher.apply(schedule, result) is True
        assert self.researcher.validate(schedule, result) is True
        assert result.validation_error == ""
        assert "import_fail" not in (result.validation_error or "")

    @pytest.mark.skipif(not ATTEMPT2_CODE_PATH.exists(), reason="attempt 2 fixture missing")
    def test_attempt2_validate_failure_is_explicit_not_import_fail(self, tmp_path):
        """If validation fails, error must be validate_error not generic import_fail."""
        raw_code = ATTEMPT2_CODE_PATH.read_text(encoding="utf-8")
        schedule = tmp_path / "adaptive_mechanism_schedule.py"
        schedule.write_text(CANONICAL_SCHEDULE.read_text(encoding="utf-8"), encoding="utf-8")
        result = ScheduleMechanismResult(
            session_id="attempt2_explicit",
            hypothesis="attempt2 fixture",
            mechanism_name="schedule_mechanism_attempt2",
            implementation_strategy="replace_method",
            target="decide",
            spec="",
            code=raw_code,
            session_dir=tmp_path / "session",
        )
        result.session_dir.mkdir(parents=True, exist_ok=True)
        self.researcher.apply(schedule, result)
        self.researcher.validate(schedule, result)
        if not result.validated:
            assert result.validation_error.startswith("validate_error:")
            assert result.validation_error != "import_fail"
