"""Tests for core/mechanism_validation_harness.py."""
from __future__ import annotations

import textwrap
from pathlib import Path

from core.base_mechanism_research import BaseMechanismResearcher
from core.mechanism_tabu_registry import MechanismTabuRegistry
from core.mechanism_validation_harness import MechanismValidationHarness, MockLLMClient


class _HarnessResearcher(BaseMechanismResearcher):
    def _get_explore_prompt(self, **kwargs):
        return ("explore " + kwargs.get("trace_summary", ""), "system")

    def _get_specify_prompt(self, selected_hypothesis, critique, **kwargs):
        return ("specify", "system")

    def _get_codegen_prompt(self, spec, reference_code, **kwargs):
        return "codegen"

    def _get_reference_code(self, **kwargs):
        return "# ref"

    def _parse_spec_metadata(self, spec, session_id):
        return ("test_mech", "new_helper_class", "TestHelper")


class TestValidateSyntax:
    def setup_method(self):
        self.harness = MechanismValidationHarness()

    def test_valid_code(self):
        assert self.harness.validate_syntax("x = 1\n") is None

    def test_invalid_code(self):
        err = self.harness.validate_syntax("def broken(\n")
        assert err is not None
        assert "SyntaxError" in err


class TestValidateImport:
    def setup_method(self):
        self.harness = MechanismValidationHarness()

    def test_imports_valid_module(self, tmp_path):
        mod = tmp_path / "sample_mod.py"
        mod.write_text("VALUE = 42\n", encoding="utf-8")
        assert self.harness.validate_import(mod, domain="generic") is True

    def test_rejects_syntax_error_module(self, tmp_path):
        mod = tmp_path / "bad_mod.py"
        mod.write_text("def x(\n", encoding="utf-8")
        assert self.harness.validate_import(mod) is False


class TestMockLLMClient:
    def test_returns_canned_responses_by_keyword(self):
        client = MockLLMClient({"exploration": "EXPLORE_TEXT"})
        out = client.call("Write exploration hypotheses", system="")
        assert out == "EXPLORE_TEXT"
        assert len(client.calls) == 1


class TestValidateDryRun:
    def test_dry_run_completes_four_rounds(self, tmp_path):
        harness = MechanismValidationHarness()
        responses = {
            "exploration": "## Hypothesis 1\nImprove search.\n",
            "critique": "**Selected**: 1 — feasible.\n",
            "spec": (
                "1. **Mechanism name**: test_mech\n"
                "2. **Implementation strategy**: new_helper_class\n"
                "3. **Target**: TestHelper\n"
            ),
            "codegen": "class TestHelper:\n    pass\n",
        }
        result = harness.validate_dry_run(
            researcher_cls=_HarnessResearcher,
            mock_trace="iter 0 keep bpb=1.1",
            mock_llm_responses=responses,
            session_dir=tmp_path / "session",
        )
        assert result["exploration_ok"] is True
        assert result["spec_metadata_ok"] is True
        assert result["codegen_ok"] is True
        assert result["ok"] is True
        assert (tmp_path / "session" / "01_exploration.md").is_file()


class TestValidateReplaySessions:
    def test_replay_with_tabu(self):
        from core.mechanism_session_trace import parse_session_dir

        fixture_root = (
            Path(__file__).resolve().parent.parent
            / "experiments/ablations/paper_ablation/run2_results/results_C/C1/mechanism_sessions"
        )
        session_dir = fixture_root / "round_1"
        record = parse_session_dir(session_dir, round=1)

        reg = MechanismTabuRegistry()
        reg.record_failure(record, round_num=1, reason="import_fail")

        harness = MechanismValidationHarness()
        result = harness.validate_replay_sessions(
            tabu_check=reg.is_tabu,
            fixture_sessions=[session_dir],
        )
        assert result["ok"] is True
        assert result["parsed"] == 1
        assert result["blocked"] == 1
