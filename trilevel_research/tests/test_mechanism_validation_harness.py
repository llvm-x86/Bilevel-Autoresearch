"""Tests for core/mechanism_validation_harness.py."""
from __future__ import annotations

from pathlib import Path

from core.base_mechanism_research import BaseMechanismResearcher
from trilevel_research.core.mechanism_tabu_registry import MechanismTabuRegistry
from trilevel_research.core.mechanism_validation_harness import (
    MechanismValidationHarness,
    MockLLMClient,
)


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

    def test_critique_and_spec_defaults(self):
        client = MockLLMClient()
        assert "Selected" in client.call("Write critique failure modes", system="")
        assert "Mechanism name" in client.call("Write implementation specification", system="")
        assert "class" in client.call("fix failed with SyntaxError", system="")

    def test_codegen_default(self):
        client = MockLLMClient()
        out = client.call("generate code please", system="")
        assert "class" in out


class TestValidateImportDomains:
    def test_train_opt_requires_researcher_class(self, tmp_path):
        harness = MechanismValidationHarness()
        mod = tmp_path / "train_mech.py"
        mod.write_text("class TrainMechanismResearcher:\n    pass\n", encoding="utf-8")
        assert harness.validate_import(mod, domain="train_opt") is True

    def test_article_opt_requires_mechanism_researcher(self, tmp_path):
        harness = MechanismValidationHarness()
        mod = tmp_path / "article_mech.py"
        mod.write_text("class MechanismResearcher:\n    pass\n", encoding="utf-8")
        assert harness.validate_import(mod, domain="article_opt") is True

    def test_missing_class_returns_false(self, tmp_path):
        harness = MechanismValidationHarness()
        mod = tmp_path / "empty.py"
        mod.write_text("x = 1\n", encoding="utf-8")
        assert harness.validate_import(mod, domain="train_opt") is False


class TestValidateDryRunErrors:
    def test_research_exception_returns_error(self, tmp_path):
        class _BrokenResearcher(_HarnessResearcher):
            def research(self, **kwargs):
                raise RuntimeError("boom")

        harness = MechanismValidationHarness()
        result = harness.validate_dry_run(
            researcher_cls=_BrokenResearcher,
            mock_trace="trace",
            session_dir=tmp_path / "broken",
        )
        assert result["ok"] is False
        assert "boom" in result["error"]

    def test_incomplete_dry_run_sets_error(self, tmp_path):
        class _NoOpResearcher:
            def __init__(self, api_key=""):
                self.client = None

            def research(self, **kwargs):
                return {"code": "def broken(\n"}

        harness = MechanismValidationHarness()
        result = harness.validate_dry_run(
            researcher_cls=_NoOpResearcher,
            mock_trace="trace",
            session_dir=tmp_path / "incomplete",
        )
        assert result["ok"] is False
        assert result["error"]


class TestValidateDryRunSessionFallback:
    def test_run_session_fallback_on_type_error(self, tmp_path):
        class _SessionResearcher(_HarnessResearcher):
            def research(self, **kwargs):
                raise TypeError("wrong signature")

        harness = MechanismValidationHarness()
        result = harness.validate_dry_run(
            researcher_cls=_SessionResearcher,
            mock_trace="trace",
            session_dir=tmp_path / "session_fb",
            mock_llm_responses={
                "exploration": "Hypothesis 1\n",
                "critique": "**Selected**: 1\n",
                "spec": "1. **Mechanism name**: m\n2. **Implementation strategy**: new_helper_class\n3. **Target**: T\n",
                "codegen": "class T:\n    pass\n",
            },
        )
        assert result["ok"] is True


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
        from trilevel_research.core.mechanism_session_trace import parse_session_dir

        fixture_root = (
            Path(__file__).resolve().parent.parent.parent
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

    def test_empty_fixture_list_not_ok(self):
        harness = MechanismValidationHarness()
        reg = MechanismTabuRegistry()
        result = harness.validate_replay_sessions(
            tabu_check=reg.is_tabu,
            fixture_sessions=[],
        )
        assert result["ok"] is False
        assert result["parsed"] == 0
