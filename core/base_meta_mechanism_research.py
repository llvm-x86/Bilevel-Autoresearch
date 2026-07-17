"""Base class for Level 3 meta-mechanism research (patches mechanism_research.py)."""
from __future__ import annotations

import ast
import importlib.util
import json
import logging
import re
import sys
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from core.base_mechanism_research import BaseMechanismResearcher

logger = logging.getLogger(__name__)

VALID_L3_STRATEGIES = frozenset({
    "replace_method",
    "new_helper_class",
    "modify_init",
    "new_method",
})


@dataclass
class MetaMechanismResult:
    session_id: str
    hypothesis: str
    patch_name: str
    implementation_strategy: str
    target: str
    spec: str
    code: str
    exploration: str = ""
    critique: str = ""
    code_retries: int = 0
    applied: bool = False
    validated: bool = False
    validation_error: str = ""
    session_dir: Path = field(default_factory=lambda: Path("."))


class BaseMetaMechanismResearcher(BaseMechanismResearcher, ABC):
    """
    Level-3 protocol: Explore → Critique → Specify → Generate patches for
    mechanism_research.py (not runner.py).

    Subclasses supply domain-specific prompts and reference code extraction.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        provider: str = "deepseek",
        api_key: str = "",
        max_code_retries: int = 3,
        artifacts_base: Path | None = None,
    ):
        super().__init__(
            model=model,
            provider=provider,
            api_key=api_key,
            max_code_retries=max_code_retries,
            artifacts_base=artifacts_base or Path("artifacts/meta_mechanism_research"),
        )

    @abstractmethod
    def _get_researcher_class_name(self) -> str:
        """Return the main researcher class name in mechanism_research.py."""
        ...

    @abstractmethod
    def _get_mechanism_research_path(self) -> Path:
        """Return canonical mechanism_research.py path for this domain."""
        ...

    def research(
        self,
        l2_trace_summary: str,
        mechanism_research_code: str,
        session_dir: Path,
        bottleneck: str = "",
    ) -> MetaMechanismResult:
        """Run one L3 session targeting mechanism_research.py."""
        session_dir = Path(session_dir)
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"[MetaMechResearch {session_id}] Starting L3 session")

        explore_kwargs = {
            "l2_trace_summary": l2_trace_summary,
            "mechanism_research_code": mechanism_research_code,
            "bottleneck": bottleneck or self._infer_l2_bottleneck(l2_trace_summary),
            "mech_code_summary": self._summarize_mech_research(mechanism_research_code),
        }
        specify_kwargs = {
            "mechanism_research_section": self._extract_mech_section(
                mechanism_research_code
            )[:3000],
        }
        codegen_kwargs = {
            "mechanism_research_code": mechanism_research_code,
            "researcher_class": self._get_researcher_class_name(),
        }

        session = self._run_session(
            session_dir=session_dir,
            session_id=session_id,
            log_prefix="MetaMechResearch",
            explore_kwargs=explore_kwargs,
            specify_kwargs=specify_kwargs,
            codegen_kwargs=codegen_kwargs,
        )

        patch_name, impl_strategy, target = self._parse_l3_spec_metadata(
            session["spec"], session_id
        )

        result = MetaMechanismResult(
            session_id=session_id,
            hypothesis=session["selected_hypothesis"],
            patch_name=patch_name,
            implementation_strategy=impl_strategy,
            target=target,
            spec=session["spec"],
            code=session["code"],
            exploration=session["exploration"],
            critique=session["critique"],
            code_retries=session["retries"],
            session_dir=session_dir,
        )
        self._save_summary(result, session_dir)
        return result

    def apply(self, mech_path: Path, result: MetaMechanismResult) -> bool:
        """
        Apply generated patch to mechanism_research.py.

        Supported strategies: replace_method, new_helper_class, modify_init, new_method.
        """
        mech_path = Path(mech_path)
        original = mech_path.read_text(encoding="utf-8")
        backup_path = mech_path.with_suffix(f".py.bak_{result.session_id}")
        backup_path.write_text(original, encoding="utf-8")

        strategy = result.implementation_strategy
        code = result.code.strip()
        class_name = self._get_researcher_class_name()

        try:
            if strategy == "new_helper_class":
                patched = self._insert_helper_class(original, code, class_name)
            elif strategy == "replace_method":
                patched = self._replace_method(original, class_name, result.target, code)
            elif strategy == "new_method":
                patched = self._insert_method(original, class_name, code)
            elif strategy == "modify_init":
                patched = self._append_to_init(original, class_name, code)
            else:
                logger.warning(f"[Apply L3] Unknown strategy '{strategy}', using new_helper_class")
                patched = self._insert_helper_class(original, code, class_name)
        except Exception as e:
            logger.error(f"[Apply L3] Patch failed: {e}")
            result.validation_error = f"patch_apply_error: {e}"
            return False

        error = self._syntax_check(patched)
        if error:
            result.validation_error = f"syntax_error: {error}"
            mech_path.write_text(original, encoding="utf-8")
            return False

        mech_path.write_text(patched, encoding="utf-8")
        (result.session_dir / "05_patched_mechanism_research.py").write_text(
            patched, encoding="utf-8"
        )
        result.applied = True
        return True

    def validate(self, mech_path: Path) -> bool:
        """Import-check the patched mechanism_research.py via importlib."""
        mech_path = Path(mech_path).resolve()
        project_root = mech_path
        for _ in range(10):
            if (project_root / "core").is_dir():
                break
            project_root = project_root.parent

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        module_name = f"_meta_mech_validate_{mech_path.stem}_{id(mech_path)}"
        spec = importlib.util.spec_from_file_location(module_name, mech_path)
        if spec is None or spec.loader is None:
            return False

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            class_name = self._get_researcher_class_name()
            if not hasattr(module, class_name):
                logger.error(f"[Validate L3] Missing class {class_name}")
                return False
            return True
        except Exception as e:
            logger.error(f"[Validate L3] Import failed: {e}")
            return False

    # ------------------------------------------------------------------
    # L3 parsing / summarization
    # ------------------------------------------------------------------

    def _parse_l3_spec_metadata(self, spec: str, session_id: str) -> tuple[str, str, str]:
        patch_name = f"meta_patch_{session_id}"
        impl_strategy = "new_helper_class"
        target = f"MetaHelper_{session_id}"

        for line in spec.splitlines():
            low = line.lower()
            if "mechanism name" in low or "patch name" in low:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    candidate = parts[1].strip().strip("`").strip("*").strip()
                    if candidate and " " not in candidate:
                        patch_name = re.sub(r"[^a-zA-Z0-9_]", "", candidate)
            if "implementation strategy" in low:
                for strat in VALID_L3_STRATEGIES:
                    if strat in low.replace(" ", "_") or strat in low:
                        impl_strategy = strat
                        break
            if low.strip().startswith("3.") and "target" in low:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    target = parts[1].strip().strip("`").strip("*").strip()

        return patch_name, impl_strategy, target

    def _infer_l2_bottleneck(self, l2_trace_summary: str) -> str:
        low = l2_trace_summary.lower()
        if "import_failed" in low or "validated" in low and "false" in low:
            return "high L2 revert / import failure rate"
        if "duplicate" in low:
            return "repeated mechanism proposals"
        return "L2 sessions not improving inner loop efficiency"

    def _summarize_mech_research(self, code: str) -> str:
        lines = []
        for line in code.splitlines():
            if line.startswith("class ") or line.startswith("def "):
                lines.append(line.strip())
            if len(lines) >= 20:
                break
        return "\n".join(lines) if lines else code[:500]

    def _extract_mech_section(self, code: str) -> str:
        class_name = self._get_researcher_class_name()
        marker = f"class {class_name}:"
        idx = code.find(marker)
        if idx != -1:
            return code[idx : idx + 2000]
        return code[:2000]

    def _save_summary(self, result: MetaMechanismResult, session_dir: Path) -> None:
        summary = {
            "session_id": result.session_id,
            "patch_name": result.patch_name,
            "implementation_strategy": result.implementation_strategy,
            "target": result.target,
            "code_retries": result.code_retries,
            "hypothesis": result.hypothesis[:300],
        }
        (session_dir / "06_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Patch helpers (mechanism_research.py targets)
    # ------------------------------------------------------------------

    def _insert_helper_class(self, original: str, new_class_code: str, anchor_class: str) -> str:
        marker = f"\nclass {anchor_class}:"
        idx = original.find(marker)
        if idx == -1:
            marker = f"class {anchor_class}:"
            idx = original.find(marker)
        if idx == -1:
            raise ValueError(f"Could not find 'class {anchor_class}:' in mechanism_research.py")

        separator = "\n\n\n# ---------------------------------------------------------------------------\n"
        insert_block = f"{separator}{new_class_code.strip()}\n\n\n"
        return original[:idx] + insert_block + original[idx:]

    def _replace_method(
        self,
        original: str,
        class_name: str,
        method_name: str,
        new_method_code: str,
    ) -> str:
        try:
            return self._ast_replace_method(original, class_name, method_name, new_method_code)
        except Exception as e:
            logger.warning(f"[Replace L3] AST failed ({e}), using regex fallback")

        pattern = rf'(    def {re.escape(method_name)}\(.*?\n)((?:(?!    def |\nclass ).)*)'
        lines = new_method_code.splitlines()
        if lines and not lines[0].startswith("    "):
            new_block = textwrap.indent(new_method_code.strip(), "    ") + "\n"
        else:
            new_block = new_method_code.rstrip() + "\n"
        patched, count = re.subn(pattern, new_block, original, count=1, flags=re.DOTALL)
        if count == 0:
            raise ValueError(f"Could not find method '{method_name}' for replacement")
        return patched

    def _ast_replace_method(
        self,
        original: str,
        class_name: str,
        method_name: str,
        new_method_code: str,
    ) -> str:
        tree = ast.parse(original)
        lines = original.splitlines(keepends=True)

        target_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                target_class = node
                break
        if target_class is None:
            raise ValueError(f"Class {class_name} not found")

        target_node = None
        for node in target_class.body:
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                target_node = node
                break
        if target_node is None:
            raise ValueError(f"Method '{method_name}' not found in {class_name}")

        start_line = target_node.lineno - 1
        end_line = target_node.end_lineno
        new_lines = new_method_code.rstrip().splitlines()
        if new_lines and not new_lines[0].startswith("    "):
            new_method_indented = textwrap.indent(new_method_code.strip(), "    ") + "\n"
        else:
            new_method_indented = new_method_code.rstrip() + "\n"

        return "".join(lines[:start_line] + [new_method_indented] + lines[end_line:])

    def _insert_method(self, original: str, class_name: str, new_method_code: str) -> str:
        marker = f"class {class_name}:"
        idx = original.find(marker)
        if idx == -1:
            raise ValueError(f"Class {class_name} not found")

        insert_at = original.find("\n    def ", idx)
        if insert_at == -1:
            insert_at = len(original)

        lines = new_method_code.rstrip().splitlines()
        if lines and not lines[0].startswith("    "):
            insert_block = "\n" + textwrap.indent(new_method_code.strip(), "    ") + "\n"
        else:
            insert_block = "\n" + new_method_code.strip() + "\n"
        return original[:insert_at] + insert_block + original[insert_at:]

    def _append_to_init(self, original: str, class_name: str, new_init_code: str) -> str:
        try:
            return self._ast_append_to_init(original, class_name, new_init_code)
        except Exception as e:
            logger.warning(f"[AppendInit L3] AST failed ({e})")

        init_pattern = rf'(class {re.escape(class_name)}.*?    def __init__\(.*?\n(?:(?!    def ).)*)'
        match = re.search(init_pattern, original, re.DOTALL)
        if not match:
            raise ValueError(f"Could not find __init__ in {class_name}")
        insert_at = match.end()
        snippet = textwrap.indent(new_init_code.strip(), "        ") + "\n"
        return original[:insert_at] + snippet + original[insert_at:]

    def _ast_append_to_init(
        self,
        original: str,
        class_name: str,
        new_code: str,
    ) -> str:
        tree = ast.parse(original)
        lines = original.splitlines(keepends=True)

        target_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                target_class = node
                break
        if target_class is None:
            raise ValueError(f"Class {class_name} not found")

        init_node = None
        for node in target_class.body:
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                init_node = node
                break
        if init_node is None:
            raise ValueError(f"__init__ not found in {class_name}")

        end_line = init_node.end_lineno
        snippet = textwrap.indent(new_code.strip(), "        ") + "\n"
        return "".join(lines[:end_line] + [snippet] + lines[end_line:])
