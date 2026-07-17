"""Level 3 smoke tests — offline, no API keys or GPU."""
from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

from core.adaptive_mechanism_schedule import AdaptiveMechanismSchedule, MechanismResearchConfig
from core.base_meta_mechanism_research import BaseMetaMechanismResearcher, MetaMechanismResult
from core.mechanism_session_trace import MechanismSessionTrace, MechanismSessionTraceBuilder
from core.mechanism_tabu_registry import MechanismTabuRegistry
from core.mechanism_validation_harness import MechanismValidationHarness

FIXTURE_ROOT = (
    Path(__file__).resolve().parent.parent
    / "experiments/ablations/paper_ablation/run2_results/results_C/C1"
)


class _SmokeMetaResearcher(BaseMetaMechanismResearcher):
    def _get_explore_prompt(self, **kwargs):
        return (
            f"L2 trace:\n{kwargs.get('l2_trace_summary', '')}\n"
            f"Bottleneck: {kwargs.get('bottleneck', '')}",
            "system",
        )

    def _get_specify_prompt(self, selected_hypothesis, critique, **kwargs):
        return ("specify L3 patch", "system")

    def _get_codegen_prompt(self, spec, reference_code, **kwargs):
        return "generate helper class"

    def _get_reference_code(self, **kwargs):
        return "class ExampleHelper:\n    pass\n"

    def _parse_spec_metadata(self, spec, session_id):
        return ("l3_patch", "new_helper_class", "L3SmokeHelper")

    def _get_researcher_class_name(self) -> str:
        return "SmokeMechanismResearcher"

    def _get_mechanism_research_path(self) -> Path:
        return Path(__file__).parent / "nonexistent_mech.py"


MINIMAL_MECH_RESEARCH = textwrap.dedent(
    '''\
    """Minimal mechanism_research stub for L3 smoke tests."""
    from __future__ import annotations


    class SmokeMechanismResearcher:
        def __init__(self, api_key: str = ""):
            self.api_key = api_key

        def research(self, trace_summary: str, runner_code: str, session_dir, bottleneck: str = ""):
            return {"ok": True}
    '''
)


class TestLevel3Smoke:
    def test_session_trace_tabu_schedule_pipeline(self):
        report = json.loads((FIXTURE_ROOT / "report.json").read_text(encoding="utf-8"))
        trace = MechanismSessionTrace(
            sessions=MechanismSessionTraceBuilder.build(
                FIXTURE_ROOT / "mechanism_sessions",
                report=report,
            )
        )
        assert len(trace.sessions) >= 2

        tabu = MechanismTabuRegistry()
        for rec in trace.sessions:
            if rec.validated is False:
                tabu.record_failure(rec, round_num=rec.round or 1, reason="import_fail")

        blocked, _ = tabu.is_tabu(
            trace.sessions[0].mechanism_name,
            trace.sessions[0].target,
            round_num=3,
        )
        assert blocked is True

        schedule = AdaptiveMechanismSchedule()
        schedule.level2_interval = 2
        schedule.level3_interval = 1
        decision = schedule.decide(
            inner_trace=report["trace"][-10:],
            l2_sessions=trace.sessions,
            completed_outer_cycles=6,
            config=MechanismResearchConfig(level2_interval=2, level3_interval=1),
        )
        assert decision.batch_size == 2
        # High revert rate from fixtures should escalate L3
        assert decision.fire_level3 is True

    def test_meta_researcher_apply_and_validate(self, tmp_path):
        mech_path = tmp_path / "mechanism_research.py"
        mech_path.write_text(MINIMAL_MECH_RESEARCH, encoding="utf-8")

        researcher = _SmokeMetaResearcher(api_key="mock")
        result = MetaMechanismResult(
            session_id="smoke001",
            hypothesis="add helper",
            patch_name="l3_smoke_helper",
            implementation_strategy="new_helper_class",
            target="L3SmokeHelper",
            spec="spec",
            code=textwrap.dedent(
                '''\
                class L3SmokeHelper:
                    """Injected by L3 smoke test."""
                    value = 1
                '''
            ),
            session_dir=tmp_path / "session",
        )
        (result.session_dir).mkdir(parents=True, exist_ok=True)

        assert researcher.apply(mech_path, result) is True
        patched = mech_path.read_text(encoding="utf-8")
        assert "class L3SmokeHelper" in patched
        assert "class SmokeMechanismResearcher" in patched
        assert researcher.validate(mech_path) is True

    def test_meta_researcher_dry_run_with_mock_llm(self, tmp_path):
        harness = MechanismValidationHarness()
        responses = {
            "exploration": "## Hypothesis 1\nAdd tabu registry to L2.\n",
            "critique": "**Selected**: 1 — highest impact.\n",
            "spec": (
                "1. **Mechanism name**: tabu_hook\n"
                "2. **Implementation strategy**: new_helper_class\n"
                "3. **Target**: TabuHook\n"
            ),
            "codegen": "class TabuHook:\n    pass\n",
        }
        result = harness.validate_dry_run(
            researcher_cls=_SmokeMetaResearcher,
            mock_trace="L2 revert rate high",
            mock_llm_responses=responses,
            session_dir=tmp_path / "l3_session",
            research_kwargs={
                "l2_trace_summary": "2 sessions, import failures",
                "mechanism_research_code": MINIMAL_MECH_RESEARCH,
            },
        )
        assert result["ok"] is True

    def test_replace_method_strategy(self, tmp_path):
        mech_path = tmp_path / "mechanism_research.py"
        mech_path.write_text(MINIMAL_MECH_RESEARCH, encoding="utf-8")
        researcher = _SmokeMetaResearcher(api_key="mock")

        new_research = textwrap.dedent(
            '''\
            def research(self, trace_summary: str, runner_code: str, session_dir, bottleneck: str = ""):
                return {"patched": True}
            '''
        )
        result = MetaMechanismResult(
            session_id="smoke002",
            hypothesis="patch research",
            patch_name="research_patch",
            implementation_strategy="replace_method",
            target="research",
            spec="spec",
            code=new_research,
            session_dir=tmp_path / "session2",
        )
        result.session_dir.mkdir(parents=True, exist_ok=True)

        assert researcher.apply(mech_path, result) is True
        patched = mech_path.read_text(encoding="utf-8")
        assert '"patched": True' in patched
        assert researcher.validate(mech_path) is True

    def test_modify_init_strategy(self, tmp_path):
        mech_path = tmp_path / "mechanism_research.py"
        mech_path.write_text(
            MINIMAL_MECH_RESEARCH.replace(
                "self.api_key = api_key",
                "self.api_key = api_key\n        self.tabu_registry = None",
            ),
            encoding="utf-8",
        )
        researcher = _SmokeMetaResearcher(api_key="mock")
        result = MetaMechanismResult(
            session_id="smoke003",
            hypothesis="wire tabu",
            patch_name="init_tabu",
            implementation_strategy="modify_init",
            target="__init__",
            spec="spec",
            code="self.l3_flag = True",
            session_dir=tmp_path / "session3",
        )
        result.session_dir.mkdir(parents=True, exist_ok=True)

        assert researcher.apply(mech_path, result) is True
        assert "self.l3_flag = True" in mech_path.read_text(encoding="utf-8")

    def test_c1_fixture_trace_text_non_empty(self):
        report = json.loads((FIXTURE_ROOT / "report.json").read_text(encoding="utf-8"))
        trace = MechanismSessionTrace(
            sessions=MechanismSessionTraceBuilder.build(
                FIXTURE_ROOT / "mechanism_sessions",
                report=report,
            )
        )
        text = trace.to_text()
        assert "round 1" in text.lower() or "Round 1" in text
        assert len(text) > 50
