"""Offline validation for mechanism_research.py patches."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock


class MockLLMClient:
    """Deterministic LLM stand-in for dry-run validation."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.calls: list[tuple[str, dict]] = []

    def call(self, prompt: str, system: str = "", max_tokens: int = 4000) -> str:
        self.calls.append((prompt, {"system": system, "max_tokens": max_tokens}))
        for key, value in self.responses.items():
            if key in prompt or key in system:
                return value
        if "exploration" in prompt.lower() or "hypothes" in prompt.lower():
            return self.responses.get(
                "exploration",
                "Hypothesis 1: tabu filter\n**Domain**: optimization",
            )
        if "critique" in prompt.lower() or "failure mode" in prompt.lower():
            return self.responses.get("critique", "**Selected**: 1 — best option.")
        if "specification" in prompt.lower() or "implementation spec" in prompt.lower():
            return self.responses.get(
                "spec",
                (
                    "1. **Mechanism name**: tabu_filter\n"
                    "2. **Implementation strategy**: new_helper_class\n"
                    "3. **Target**: TabuFilter\n"
                ),
            )
        if "fix" in prompt.lower() or "failed with" in prompt.lower():
            return self.responses.get("fix", "class TabuFilter:\n    pass\n")
        return self.responses.get("codegen", "class TabuFilter:\n    pass\n")


class MechanismValidationHarness:
    """Validate syntax, import, and dry-run behavior of mechanism researchers."""

    def __init__(self, domain: str = "train_opt", project_root: Path | None = None):
        self.domain = domain
        self.project_root = project_root or self._find_project_root()

    def validate_syntax(self, code: str) -> str | None:
        try:
            compile(code, "<generated>", "exec")
            return None
        except SyntaxError as e:
            return f"SyntaxError: {e}"

    def validate_import(self, mech_path: Path, domain: str | None = None) -> bool:
        mech_path = Path(mech_path)
        domain = domain or self.domain
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        spec = importlib.util.spec_from_file_location(
            f"_harness_mech_{mech_path.stat().st_mtime_ns}", mech_path
        )
        if spec is None or spec.loader is None:
            return False

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            return False

        if domain == "train_opt":
            return hasattr(module, "TrainMechanismResearcher")
        if domain == "article_opt":
            return hasattr(module, "MechanismResearcher")
        return True

    def validate_dry_run(
        self,
        researcher_cls: type,
        mock_trace: str,
        mock_llm_responses: dict[str, str] | None = None,
        runner_code: str = "# mock runner",
        session_dir: Path | None = None,
        research_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run research() with a mocked LLM client."""
        session_dir = session_dir or Path(tempfile.mkdtemp(prefix="mech_harness_"))
        session_dir.mkdir(parents=True, exist_ok=True)

        responses = mock_llm_responses or {
            "exploration": "Hypothesis 1: tabu filter\n**Domain**: optimization",
            "critique": "**Selected**: 1 — best option.",
            "spec": (
                "1. **Mechanism name**: tabu_filter\n"
                "2. **Implementation strategy**: new_helper_class\n"
                "3. **Target**: TabuFilter\n"
            ),
            "codegen": "class TabuFilter:\n    pass\n",
        }

        researcher = researcher_cls(api_key="test")
        mock_client = MockLLMClient(responses)
        researcher.client = mock_client

        kwargs: dict[str, Any] = {
            "trace_summary": mock_trace,
            "runner_code": runner_code,
            "session_dir": session_dir,
            "bottleneck": "test bottleneck",
        }
        if research_kwargs:
            kwargs.update(research_kwargs)

        result_payload: dict[str, Any] = {
            "ok": False,
            "exploration_ok": False,
            "spec_metadata_ok": False,
            "codegen_ok": False,
            "error": "",
        }

        try:
            if "l2_trace_summary" in kwargs:
                mech_result = researcher.research(
                    l2_trace_summary=kwargs.pop("l2_trace_summary"),
                    mechanism_research_code=kwargs.pop(
                        "mechanism_research_code", runner_code
                    ),
                    session_dir=session_dir,
                    bottleneck=kwargs.pop("bottleneck", ""),
                )
            elif hasattr(researcher, "research"):
                mech_result = researcher.research(**kwargs)
            else:
                raise AttributeError("no research()")
        except (TypeError, AttributeError):
            # BaseMechanismResearcher subclasses used via _run_session in tests
            try:
                session = researcher._run_session(
                    session_dir=session_dir,
                    session_id="dry_run_test",
                    log_prefix="DryRun",
                    explore_kwargs=kwargs,
                    specify_kwargs=kwargs,
                    codegen_kwargs=kwargs,
                )
                mech_result = session
            except Exception as e:
                result_payload["error"] = str(e)
                return result_payload
        except Exception as e:
            result_payload["error"] = str(e)
            return result_payload

        exploration_path = session_dir / "01_exploration.md"
        result_payload["exploration_ok"] = (
            exploration_path.is_file() and exploration_path.stat().st_size > 0
        )
        result_payload["spec_metadata_ok"] = (session_dir / "03_spec.md").is_file()

        code_files = list(session_dir.glob("04_code_attempt_*.py"))
        if code_files:
            code = code_files[-1].read_text(encoding="utf-8")
            result_payload["codegen_ok"] = self.validate_syntax(code) is None
        elif hasattr(mech_result, "code"):
            result_payload["codegen_ok"] = self.validate_syntax(mech_result.code) is None
        elif isinstance(mech_result, dict) and mech_result.get("code"):
            result_payload["codegen_ok"] = self.validate_syntax(mech_result["code"]) is None

        if hasattr(mech_result, "mechanism_name"):
            result_payload["spec_metadata_ok"] = bool(mech_result.mechanism_name)
            result_payload["mechanism_name"] = mech_result.mechanism_name
        elif hasattr(mech_result, "patch_name"):
            result_payload["spec_metadata_ok"] = bool(mech_result.patch_name)

        result_payload["ok"] = (
            result_payload["exploration_ok"]
            and result_payload["spec_metadata_ok"]
            and result_payload["codegen_ok"]
        )
        if not result_payload["ok"] and not result_payload["error"]:
            result_payload["error"] = "dry-run checks incomplete"

        return result_payload

    def validate_replay_sessions(
        self,
        tabu_check: Callable[[str, str, int], tuple[bool, str]],
        fixture_sessions: list[Path],
    ) -> dict[str, Any]:
        from core.mechanism_session_trace import parse_session_dir

        blocked = 0
        parsed = 0
        for session_dir in fixture_sessions:
            record = parse_session_dir(session_dir)
            parsed += 1
            rnd = record.round or parsed
            is_tabu, _ = tabu_check(record.mechanism_name, record.target, rnd + 1)
            if is_tabu:
                blocked += 1

        return {"ok": parsed > 0, "parsed": parsed, "blocked": blocked}

    @staticmethod
    def _find_project_root() -> Path:
        here = Path(__file__).resolve().parent
        return here.parent
