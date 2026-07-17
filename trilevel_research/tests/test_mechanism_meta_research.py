"""Tests for core/base_meta_mechanism_research.py."""
from __future__ import annotations

import textwrap
from pathlib import Path

from trilevel_research.core.base_meta_mechanism_research import (
    BaseMetaMechanismResearcher,
    MetaMechanismResult,
)
from trilevel_research.core.mechanism_validation_harness import MockLLMClient

MINIMAL_MECH = textwrap.dedent(
    '''\
    """Stub mechanism_research for L3 unit tests."""
    class TestMechanismResearcher:
        def __init__(self, api_key: str = ""):
            self.api_key = api_key

        def research(self, trace_summary: str, runner_code: str, session_dir, bottleneck: str = ""):
            return None
    '''
)


class _UnitMetaResearcher(BaseMetaMechanismResearcher):
    def _get_explore_prompt(self, **kwargs):
        return (f"explore {kwargs.get('bottleneck', '')}", "system")

    def _get_specify_prompt(self, selected_hypothesis, critique, **kwargs):
        return ("specify", "system")

    def _get_codegen_prompt(self, spec, reference_code, **kwargs):
        return "codegen"

    def _get_reference_code(self, **kwargs):
        return kwargs.get("mechanism_research_code", "")[:500]

    def _parse_spec_metadata(self, spec, session_id):
        return ("unit_patch", "new_helper_class", "UnitHelper")

    def _get_researcher_class_name(self) -> str:
        return "TestMechanismResearcher"

    def _get_mechanism_research_path(self) -> Path:
        return Path("mechanism_research.py")


class TestMetaMechanismResult:
    def test_schedule_patch_defaults_none(self):
        result = MetaMechanismResult(
            session_id="s1",
            hypothesis="h",
            patch_name="p",
            implementation_strategy="new_helper_class",
            target="T",
            spec="",
            code="",
        )
        assert result.schedule_patch is None


class TestParseL3Metadata:
    def setup_method(self):
        self.researcher = _UnitMetaResearcher(api_key="mock")

    def test_parses_spec_fields(self):
        spec = textwrap.dedent(
            """\
            1. **Patch name**: tabu_wiring
            2. **Implementation strategy**: replace_method
            3. **Target**: research
            """
        )
        name, strategy, target = self.researcher._parse_l3_spec_metadata(spec, "sid")
        assert name == "tabu_wiring"
        assert strategy == "replace_method"
        assert target == "research"

    def test_defaults_when_spec_sparse(self):
        name, strategy, target = self.researcher._parse_l3_spec_metadata("", "abc123")
        assert name == "meta_patch_abc123"
        assert strategy == "new_helper_class"
        assert target == "MetaHelper_abc123"


class TestInferBottleneck:
    def setup_method(self):
        self.researcher = _UnitMetaResearcher(api_key="mock")

    def test_import_failure_bottleneck(self):
        text = "Round 1: foo [import_failed]"
        assert "import" in self.researcher._infer_l2_bottleneck(text).lower()

    def test_duplicate_bottleneck(self):
        assert "repeated" in self.researcher._infer_l2_bottleneck("duplicate proposals").lower()

    def test_default_bottleneck(self):
        assert "efficiency" in self.researcher._infer_l2_bottleneck("normal trace").lower()


class TestSummarizeAndExtract:
    def setup_method(self):
        self.researcher = _UnitMetaResearcher(api_key="mock")

    def test_summarize_mech_research(self):
        code = "class Foo:\n    pass\n" + "# comment\n" * 30
        summary = self.researcher._summarize_mech_research(code)
        assert summary.startswith("class Foo:")

    def test_summarize_falls_back_to_prefix(self):
        summary = self.researcher._summarize_mech_research("# only comments\n" * 50)
        assert len(summary) <= 500

    def test_extract_mech_section(self):
        code = "header\nclass TestMechanismResearcher:\n    x = 1\n" + "y\n" * 100
        section = self.researcher._extract_mech_section(code)
        assert section.startswith("class TestMechanismResearcher:")

    def test_extract_mech_section_fallback(self):
        section = self.researcher._extract_mech_section("no class here\n" * 100)
        assert len(section) <= 2000


class TestResearchFlow:
    def test_research_writes_session_artifacts(self, tmp_path):
        researcher = _UnitMetaResearcher(api_key="mock", max_code_retries=1)
        researcher.client = MockLLMClient(
            {
                "exploration": "Hypothesis 1: improve tabu\n**Domain**: opt",
                "critique": "**Selected**: 1 — best.\n",
                "spec": (
                    "1. **Patch name**: l3_test\n"
                    "2. **Implementation strategy**: new_helper_class\n"
                    "3. **Target**: L3Helper\n"
                ),
                "codegen": "class L3Helper:\n    pass\n",
            }
        )
        session_dir = tmp_path / "l3_session"
        result = researcher.research(
            l2_trace_summary="2 sessions",
            mechanism_research_code=MINIMAL_MECH,
            session_dir=session_dir,
        )
        assert result.patch_name == "l3_test"
        assert (session_dir / "06_summary.json").is_file()
        assert (session_dir / "01_exploration.md").is_file()


class TestApplyStrategies:
    def setup_method(self):
        self.researcher = _UnitMetaResearcher(api_key="mock")

    def _result(self, strategy: str, code: str, target: str = "research") -> MetaMechanismResult:
        return MetaMechanismResult(
            session_id="apply001",
            hypothesis="test",
            patch_name="patch",
            implementation_strategy=strategy,
            target=target,
            spec="spec",
            code=code,
            session_dir=Path("."),
        )

    def test_new_helper_class(self, tmp_path):
        mech = tmp_path / "mechanism_research.py"
        mech.write_text(MINIMAL_MECH, encoding="utf-8")
        result = self._result(
            "new_helper_class",
            "class InjectedHelper:\n    value = 1\n",
        )
        (tmp_path / "session").mkdir()
        result.session_dir = tmp_path / "session"
        assert self.researcher.apply(mech, result) is True
        assert "InjectedHelper" in mech.read_text(encoding="utf-8")

    def test_new_method(self, tmp_path):
        mech = tmp_path / "mechanism_research.py"
        mech.write_text(MINIMAL_MECH, encoding="utf-8")
        code = textwrap.dedent(
            '''\
            def extra_hook(self):
                return True
            '''
        )
        result = self._result("new_method", code)
        result.session_dir = tmp_path / "session"
        result.session_dir.mkdir()
        assert self.researcher.apply(mech, result) is True
        assert "extra_hook" in mech.read_text(encoding="utf-8")

    def test_unknown_strategy_falls_back_to_helper(self, tmp_path):
        mech = tmp_path / "mechanism_research.py"
        mech.write_text(MINIMAL_MECH, encoding="utf-8")
        result = self._result("unknown_strategy", "class Fallback:\n    pass\n")
        result.session_dir = tmp_path / "session"
        result.session_dir.mkdir()
        assert self.researcher.apply(mech, result) is True
        assert "Fallback" in mech.read_text(encoding="utf-8")

    def test_apply_syntax_error_restores_backup(self, tmp_path):
        mech = tmp_path / "mechanism_research.py"
        mech.write_text(MINIMAL_MECH, encoding="utf-8")
        result = self._result("new_helper_class", "class Broken(\n")
        result.session_dir = tmp_path / "session"
        result.session_dir.mkdir()
        assert self.researcher.apply(mech, result) is False
        assert "syntax_error" in result.validation_error
        assert mech.read_text(encoding="utf-8") == MINIMAL_MECH

    def test_apply_missing_anchor_raises(self, tmp_path):
        mech = tmp_path / "mechanism_research.py"
        mech.write_text("class WrongName:\n    pass\n", encoding="utf-8")
        result = self._result("new_helper_class", "class X:\n    pass\n")
        result.session_dir = tmp_path / "session"
        result.session_dir.mkdir()
        assert self.researcher.apply(mech, result) is False
        assert "patch_apply_error" in result.validation_error

    def test_replace_method_regex_fallback(self, tmp_path):
        mech = tmp_path / "mechanism_research.py"
        mech.write_text(MINIMAL_MECH, encoding="utf-8")
        new_code = textwrap.dedent(
            '''\
            def research(self, trace_summary: str, runner_code: str, session_dir, bottleneck: str = ""):
                return {"replaced": True}
            '''
        )
        result = self._result("replace_method", new_code, target="research")
        result.session_dir = tmp_path / "session"
        result.session_dir.mkdir()
        assert self.researcher.apply(mech, result) is True
        assert '"replaced": True' in mech.read_text(encoding="utf-8")


class TestValidate:
    def setup_method(self):
        self.researcher = _UnitMetaResearcher(api_key="mock")

    def test_validate_success(self, tmp_path):
        mech = tmp_path / "mechanism_research.py"
        mech.write_text(MINIMAL_MECH, encoding="utf-8")
        assert self.researcher.validate(mech) is True

    def test_validate_missing_class(self, tmp_path):
        mech = tmp_path / "mechanism_research.py"
        mech.write_text("class Other:\n    pass\n", encoding="utf-8")
        assert self.researcher.validate(mech) is False

    def test_validate_import_fail_exception_path(self, tmp_path):
        mech = tmp_path / "mechanism_research.py"
        mech.write_text("raise RuntimeError('nope')\n", encoding="utf-8")
        assert self.researcher.validate(mech) is False

    def test_ast_replace_method_direct(self, tmp_path):
        mech = tmp_path / "mechanism_research.py"
        mech.write_text(MINIMAL_MECH, encoding="utf-8")
        new_code = textwrap.dedent(
            '''\
            def research(self, trace_summary: str, runner_code: str, session_dir, bottleneck: str = ""):
                return {"ast": True}
            '''
        )
        patched = self.researcher._ast_replace_method(
            MINIMAL_MECH, "TestMechanismResearcher", "research", new_code
        )
        assert '"ast": True' in patched


class TestModifyInitFallback:
    def setup_method(self):
        self.researcher = _UnitMetaResearcher(api_key="mock")

    def test_modify_init_regex_fallback(self, tmp_path):
        mech = tmp_path / "mechanism_research.py"
        source = MINIMAL_MECH.replace(
            "self.api_key = api_key",
            "self.api_key = api_key\n        self.extra = 0",
        )
        mech.write_text(source, encoding="utf-8")
        result = MetaMechanismResult(
            session_id="init_fb",
            hypothesis="test",
            patch_name="patch",
            implementation_strategy="modify_init",
            target="__init__",
            spec="spec",
            code="self.fallback_flag = True",
            session_dir=tmp_path / "session",
        )
        result.session_dir.mkdir()
        assert self.researcher.apply(mech, result) is True
        assert "self.fallback_flag = True" in mech.read_text(encoding="utf-8")
