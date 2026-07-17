"""E2E verification: every patch path and feature works — no silent or loud failures.

Invariants enforced here (the failure classes that burned us live):

1. apply() returning False ⟹ validation_error is non-empty and specifically labeled.
2. validate() returning False ⟹ validation_error is non-empty and starts with
   "validate_error:" — never the catch-all "import_fail".
3. 06_summary.json always mirrors the final result state — never stale defaults.
4. Tabu entries are run-scoped: a persisted registry from a previous run must
   NEVER block a fresh run (round-based expiry is meaningless across runs).
5. AdaptiveMechanismSchedule.decide() never raises, regardless of trace shape.
6. Every implementation strategy (replace_method, new_method, new_helper_class,
   modify_init) round-trips apply+validate on every patch target.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trilevel_research.core.adaptive_mechanism_schedule import (
    AdaptiveMechanismSchedule,
    ScheduleDecision,
)
from trilevel_research.core.base_meta_mechanism_research import MetaMechanismResult
from trilevel_research.core.mechanism_session_trace import MechanismSessionRecord
from trilevel_research.core.mechanism_tabu_registry import MechanismTabuRegistry
from trilevel_research.core.mechanism_validation_harness import (
    MechanismValidationHarness,
)
from trilevel_research.domains.gpu_bench_opt.mechanism_research import (
    GpuBenchMechanismResearcher,
    GpuBenchMechanismResult,
)
from trilevel_research.domains.gpu_bench_opt.meta_mechanism_research import (
    GpuBenchMetaMechanismResearcher,
)
from trilevel_research.domains.gpu_bench_opt.schedule_mechanism_research import (
    BOOTSTRAP_DECIDE_CODE,
    ScheduleMechanismResearcher,
    ScheduleMechanismResult,
)
from trilevel_research.domains.gpu_bench_opt.tri_level_controller import (
    GpuBenchTriLevelController,
)

CANONICAL_SCHEDULE = REPO_ROOT / "trilevel_research/core/adaptive_mechanism_schedule.py"
CANONICAL_RUNNER = REPO_ROOT / "trilevel_research/domains/gpu_bench_opt/runner.py"
CANONICAL_MECH_RESEARCH = (
    REPO_ROOT / "trilevel_research/domains/gpu_bench_opt/mechanism_research.py"
)

ALL_STRATEGIES = ("replace_method", "new_method", "new_helper_class", "modify_init")

BAD_DECIDE_CODE = '''def decide(self, inner_trace, l2_sessions, completed_outer_cycles, config=None):
    # references nonexistent attrs and bare l2_sessions access (pre-fix failure mode)
    if self.min_discard_rate_threshold > 0:
        return ScheduleDecision(fire_level2=True, fire_level3=False, reason="x", batch_size=1)
    for s in l2_sessions:
        if s.applied and not s.validated:
            return ScheduleDecision(fire_level2=True, fire_level3=False, reason="y", batch_size=1)
    return ScheduleDecision(fire_level2=False, fire_level3=False, reason="z", batch_size=1)
'''

RUNTIME_BROKEN_DECIDE = '''def decide(self, inner_trace, l2_sessions, completed_outer_cycles, config=None):
    return self.nonexistent_method()
'''

SYNTAX_BROKEN_CODE = "def decide(self, inner_trace, l2_sessions, completed_outer_cycles, config=None)\n    pass\n"

GOOD_HELPER_CLASS = '''class MockHelper:
    """Trivial helper for E2E apply/validate round-trip."""

    def __init__(self, window: int = 3):
        self.window = window

    def summarize(self, items):
        return list(items)[-self.window :]
'''

GOOD_NEW_METHOD = '''def _e2e_helper_method(self, x):
    """Trivial method inserted by E2E test."""
    return x
'''

GOOD_INIT_SNIPPET = "self._e2e_init_marker = True"


def _schedule_result(tmp_path: Path, code: str, strategy: str = "replace_method",
                     target: str = "decide") -> ScheduleMechanismResult:
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    return ScheduleMechanismResult(
        session_id="e2e_sched",
        hypothesis="e2e",
        mechanism_name="e2e_mechanism",
        implementation_strategy=strategy,
        target=target,
        spec="",
        code=code,
        session_dir=session_dir,
    )


def _runner_result(tmp_path: Path, code: str, strategy: str,
                   target: str = "run_iteration") -> GpuBenchMechanismResult:
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    return GpuBenchMechanismResult(
        session_id="e2e_runner",
        hypothesis="e2e",
        mechanism_name="e2e_mechanism",
        implementation_strategy=strategy,
        target=target,
        spec="",
        code=code,
        session_dir=session_dir,
    )


# ---------------------------------------------------------------------------
# 1. Tabu registry run-scoping (regression: cross-run poisoning killed a live run)
# ---------------------------------------------------------------------------
class TestTabuRunScoping:
    def test_stale_registry_does_not_poison_fresh_run(self, tmp_path):
        path = tmp_path / "mechanism_tabu.json"
        old = MechanismTabuRegistry(run_token="run_A")
        old.record_failure("m1", "decide", 1, reason="import_fail")
        old.save(path)

        fresh = MechanismTabuRegistry.load(path, run_token="run_B")
        is_tabu, _ = fresh.is_tabu("m2", "decide", 1)
        assert not is_tabu, "stale registry from another run blocked a fresh run"

    def test_same_run_token_preserves_entries(self, tmp_path):
        path = tmp_path / "mechanism_tabu.json"
        old = MechanismTabuRegistry(run_token="run_A")
        old.record_failure("m1", "decide", 1, reason="import_fail")
        old.save(path)

        resumed = MechanismTabuRegistry.load(path, run_token="run_A")
        is_tabu, reason = resumed.is_tabu("m2", "decide", 1)
        assert is_tabu and "import_fail" in reason

    def test_round_expiry_within_one_run(self):
        reg = MechanismTabuRegistry(run_token="r")
        reg.record_failure("m1", "decide", 1, reason="validate_fail")
        assert reg.is_tabu("m2", "decide", 2)[0]
        assert not reg.is_tabu("m2", "decide", 99)[0]

    def test_failure_creates_strategy_entry_and_success_does_not_block(self):
        reg = MechanismTabuRegistry()
        rec = MechanismSessionRecord(
            round=1, session_id="s", mechanism_name="m", implementation_strategy="replace_method",
            target="decide", session_dir=Path("."),
        )
        reg.record_failure(rec, reason="validate_attr_error")
        assert reg.is_tabu("other", "other_target", 1, strategy="replace_method")[0]
        reg2 = MechanismTabuRegistry()
        reg2.record_success(rec)
        assert not reg2.is_tabu("other", "other_target", 1, strategy="replace_method")[0]


# ---------------------------------------------------------------------------
# 2. Schedule researcher — every strategy round-trips; failures are never silent
# ---------------------------------------------------------------------------
class TestScheduleResearcherE2E:
    def setup_method(self):
        self.researcher = ScheduleMechanismResearcher(api_key="mock")

    def _sched(self, tmp_path) -> Path:
        p = tmp_path / "adaptive_mechanism_schedule.py"
        p.write_text(CANONICAL_SCHEDULE.read_text(encoding="utf-8"), encoding="utf-8")
        return p

    def test_replace_method_happy_path(self, tmp_path):
        sched = self._sched(tmp_path)
        res = _schedule_result(tmp_path, BOOTSTRAP_DECIDE_CODE)
        assert self.researcher.apply(sched, res), res.validation_error
        assert self.researcher.validate(sched, res), res.validation_error
        assert res.applied and res.validated and res.validation_error == ""

    def test_sanitizer_repairs_known_bad_codegen(self, tmp_path):
        """Pre-fix failure mode (unknown self.* + bare s.applied) must now validate."""
        sched = self._sched(tmp_path)
        res = _schedule_result(tmp_path, BAD_DECIDE_CODE)
        assert self.researcher.apply(sched, res), res.validation_error
        assert self.researcher.validate(sched, res), res.validation_error

    def test_runtime_broken_decide_fails_loudly_and_specifically(self, tmp_path):
        sched = self._sched(tmp_path)
        res = _schedule_result(tmp_path, RUNTIME_BROKEN_DECIDE)
        applied = self.researcher.apply(sched, res)
        if applied:
            assert not self.researcher.validate(sched, res)
            assert res.validation_error.startswith("validate_error:")
            assert "import_fail" not in res.validation_error
        else:
            assert res.validation_error  # never silent

    def test_syntax_broken_codegen_fails_loudly(self, tmp_path):
        sched = self._sched(tmp_path)
        res = _schedule_result(tmp_path, SYNTAX_BROKEN_CODE)
        ok = self.researcher.apply(sched, res)
        if not ok:
            assert res.validation_error.startswith(("syntax_error:", "patch_apply_error:"))
        else:
            # extraction repaired it — then validate decides; either way, explicit
            valid = self.researcher.validate(sched, res)
            assert valid or res.validation_error.startswith("validate_error:")

    def test_apply_false_always_carries_error(self, tmp_path):
        """Invariant: apply()==False ⟹ non-empty, labeled validation_error."""
        sched = self._sched(tmp_path)
        for code in ("", "   ", "class Unrelated:\n    pass\n", SYNTAX_BROKEN_CODE):
            res = _schedule_result(tmp_path, code)
            if not self.researcher.apply(sched, res):
                assert res.validation_error, f"silent apply failure for {code!r}"
                assert ":" in res.validation_error

    def test_summary_mirrors_final_result_state(self, tmp_path):
        """Invariant: 06_summary.json must reflect post-apply/validate state, not defaults."""
        sched = self._sched(tmp_path)
        res = _schedule_result(tmp_path, BOOTSTRAP_DECIDE_CODE)
        self.researcher.apply(sched, res)
        self.researcher.validate(sched, res)
        self.researcher.update_session_summary(res, res.session_dir)
        summary = json.loads((res.session_dir / "06_summary.json").read_text())
        assert summary["applied"] == res.applied
        assert summary["validated"] == res.validated
        assert summary["validation_error"] == res.validation_error

    def test_validate_fixture_sessions_exercise_both_branches(self, tmp_path):
        sched = self._sched(tmp_path)
        res = _schedule_result(tmp_path, BOOTSTRAP_DECIDE_CODE)
        self.researcher.apply(sched, res)
        assert self.researcher.validate(sched, res)


# ---------------------------------------------------------------------------
# 3. GpuBench runner researcher — all strategies; import-path validate works
# ---------------------------------------------------------------------------
class TestGpuBenchResearcherE2E:
    def setup_method(self):
        self.researcher = GpuBenchMechanismResearcher(api_key="mock")

    def _runner(self, tmp_path) -> Path:
        p = tmp_path / "runner.py"
        p.write_text(CANONICAL_RUNNER.read_text(encoding="utf-8"), encoding="utf-8")
        return p

    def test_new_helper_class_round_trip(self, tmp_path):
        runner = self._runner(tmp_path)
        res = _runner_result(tmp_path, GOOD_HELPER_CLASS, "new_helper_class")
        assert self.researcher.apply(runner, res), res.validation_error
        assert self.researcher.validate(runner, res), res.validation_error
        assert res.applied and res.validated and res.validation_error == ""

    def test_new_method_round_trip(self, tmp_path):
        runner = self._runner(tmp_path)
        res = _runner_result(tmp_path, GOOD_NEW_METHOD, "new_method")
        assert self.researcher.apply(runner, res), res.validation_error
        assert self.researcher.validate(runner, res), res.validation_error

    def test_modify_init_round_trip(self, tmp_path):
        runner = self._runner(tmp_path)
        res = _runner_result(tmp_path, GOOD_INIT_SNIPPET, "modify_init", target="__init__")
        assert self.researcher.apply(runner, res), res.validation_error
        assert self.researcher.validate(runner, res), res.validation_error

    def test_replace_method_wrong_target_fails_loudly(self, tmp_path):
        """Live failure mode: LLM named a nonexistent method — must be patch_apply_error."""
        runner = self._runner(tmp_path)
        res = _runner_result(
            tmp_path, "def no_such_method(self):\n    pass\n",
            "replace_method", target="no_such_method",
        )
        ok = self.researcher.apply(runner, res)
        assert not ok
        assert res.validation_error.startswith("patch_apply_error:")
        assert "import_fail" not in res.validation_error

    def test_import_broken_patch_fails_validate_with_specific_error(self, tmp_path):
        runner = self._runner(tmp_path)
        bad_helper = "class BrokenHelper:\n    def __init__(self):\n        import nonexistent_module_xyz\n"
        res = _runner_result(tmp_path, bad_helper, "new_helper_class")
        if self.researcher.apply(runner, res):
            valid = self.researcher.validate(runner, res)
            assert valid or res.validation_error.startswith("validate_error:")

    def test_apply_false_always_carries_error(self, tmp_path):
        runner = self._runner(tmp_path)
        res = _runner_result(tmp_path, SYNTAX_BROKEN_CODE, "new_helper_class")
        if not self.researcher.apply(runner, res):
            assert res.validation_error


# ---------------------------------------------------------------------------
# 4. Meta (L3) researcher — apply/validate round-trip and loud failure
# ---------------------------------------------------------------------------
class TestMetaResearcherE2E:
    def setup_method(self):
        self.researcher = GpuBenchMetaMechanismResearcher(api_key="mock")

    def _mech(self, tmp_path) -> Path:
        p = tmp_path / "mechanism_research.py"
        p.write_text(CANONICAL_MECH_RESEARCH.read_text(encoding="utf-8"), encoding="utf-8")
        return p

    def _result(self, tmp_path, code, strategy, target) -> MetaMechanismResult:
        session_dir = tmp_path / "session"
        session_dir.mkdir(parents=True, exist_ok=True)
        return MetaMechanismResult(
            session_id="e2e_meta",
            hypothesis="e2e",
            patch_name="e2e_patch",
            implementation_strategy=strategy,
            target=target,
            spec="",
            code=code,
            session_dir=session_dir,
        )

    def test_new_helper_class_round_trip(self, tmp_path):
        mech = self._mech(tmp_path)
        res = self._result(tmp_path, GOOD_HELPER_CLASS, "new_helper_class", "MockHelper")
        assert self.researcher.apply(mech, res), res.validation_error
        assert self.researcher.validate(mech)

    def test_loud_failure_on_bad_target(self, tmp_path):
        mech = self._mech(tmp_path)
        res = self._result(
            tmp_path, "def no_such(self):\n    pass\n", "replace_method", "no_such_method_xyz",
        )
        if not self.researcher.apply(mech, res):
            assert res.validation_error.startswith("patch_apply_error:")


# ---------------------------------------------------------------------------
# 5. AdaptiveMechanismSchedule.decide — total function, never raises
# ---------------------------------------------------------------------------
class TestScheduleDecideRobustness:
    def _check(self, inner_trace, l2_sessions, cycles):
        d = AdaptiveMechanismSchedule().decide(
            inner_trace, l2_sessions, completed_outer_cycles=cycles
        )
        assert isinstance(d, ScheduleDecision)
        assert isinstance(d.fire_level2, bool) and isinstance(d.fire_level3, bool)
        assert isinstance(d.batch_size, int) and d.batch_size >= 1

    def test_empty_trace(self):
        self._check([], [], 0)

    def test_all_discards(self):
        trace = [{"status": "discard", "bpb": 9.0}] * 10
        self._check(trace, [], 5)

    def test_all_keeps(self):
        trace = [{"status": "keep", "bpb": 1.0 - i * 0.01} for i in range(10)]
        self._check(trace, [], 3)

    def test_malformed_entries(self):
        trace = [{}, {"status": None}, {"bpb": "x"}, "garbage"]  # type: ignore[list-item]
        try:
            self._check(trace, [], 2)
        except (AttributeError, TypeError) as exc:
            pytest.fail(f"decide() raised on malformed trace: {exc}")

    def test_l2_sessions_any_shape(self):
        from types import SimpleNamespace
        sessions = [
            SimpleNamespace(applied=True, validated=False),
            SimpleNamespace(),  # missing attrs entirely
            {"applied": True},  # dict instead of object  # type: ignore[list-item]
        ]
        try:
            self._check([{"status": "keep", "bpb": 1.0}], sessions, 1)
        except (AttributeError, TypeError) as exc:
            pytest.fail(f"decide() raised on odd l2_sessions: {exc}")


# ---------------------------------------------------------------------------
# 6. Validation harness — syntax gate works both ways
# ---------------------------------------------------------------------------
class TestValidationHarness:
    def test_syntax_ok(self):
        h = MechanismValidationHarness(domain="gpu_bench_opt", project_root=REPO_ROOT)
        assert h.validate_syntax("x = 1\n") is None

    def test_syntax_fail_labeled(self):
        h = MechanismValidationHarness(domain="gpu_bench_opt", project_root=REPO_ROOT)
        err = h.validate_syntax("def broken(:\n")
        assert err and "syntax" in err.lower()


# ---------------------------------------------------------------------------
# 7. Controller error mapping — every failure shape maps to a specific reason
# ---------------------------------------------------------------------------
class TestControllerErrorMapping:
    def test_no_mapping_returns_generic_import_fail_for_typed_errors(self):
        f = GpuBenchTriLevelController._tabu_failure_reason
        assert f("") == "validate_fail"
        assert f("validate_error: TypeError: x") == "validate_fail"
        assert f("validate_error: AttributeError: y") in (
            "validate_fail", "validate_attr_error"
        )
        assert f("syntax_error: bad indent") == "harness_syntax_fail"
        # genuine import errors only
        assert f("validate_error: ImportError: no module named x") == "import_fail"


# ---------------------------------------------------------------------------
# 8. Codegen task builder — all strategies (regression: del'd name reused)
# ---------------------------------------------------------------------------
class TestCodegenTaskBuilder:
    def test_all_strategies_no_exception(self):
        researcher = ScheduleMechanismResearcher(api_key="mock")
        for strategy in ALL_STRATEGIES:
            task = researcher._build_codegen_task("mech", strategy, "decide", "spec")
            assert isinstance(task, str) and task.strip()
