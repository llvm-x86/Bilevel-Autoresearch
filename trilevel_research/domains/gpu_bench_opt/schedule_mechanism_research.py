"""Level 2 mechanism research — patches AdaptiveMechanismSchedule (ouroboros target)."""
from __future__ import annotations

import ast
import importlib.util
import json
import logging
import re
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from core.base_mechanism_research import BaseMechanismResearcher

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

EXPLORE_SYSTEM = """You are a meta-researcher specializing in adaptive mechanism scheduling.
Propose concrete changes to AdaptiveMechanismSchedule.decide() to improve when Level-2
and Level-3 mechanism research fires during bilevel GPU hyperparameter search."""

EXPLORE_PROMPT = """\
## Inner Loop Trace Analysis

### Iteration trace (most recent {n_iters} iterations):
{trace_summary}

### Current adaptive_mechanism_schedule.py:
{schedule_summary}

### Bottleneck:
{bottleneck}

---

Propose 3-4 mechanism improvements to AdaptiveMechanismSchedule (decide() logic).

For EACH hypothesis:
1. **Domain**: inspiration field
2. **Core idea**: one sentence
3. **Implementation target**: method/class in adaptive_mechanism_schedule.py
4. **Why it helps**: causal argument
5. **Implementation complexity**: 1–5
6. **Risk**: low / medium / high
"""

SPECIFY_SYSTEM = """You are a senior engineer writing a patch spec for AdaptiveMechanismSchedule."""

SPECIFY_PROMPT = """\
## Selected Hypothesis
{selected_hypothesis}

## Critique notes
{critique_notes}

## Current schedule section
```python
{schedule_section}
```

Write an implementation specification:

1. **Mechanism name** (snake_case):
2. **Implementation strategy**: new_method | replace_method | new_helper_class | modify_init
3. **Target**: method or class name (prefer decide for schedule logic)
4. **Step-by-step logic**
5. **Integration points**
"""

CODEGEN_PROMPT = """\
## Implementation Specification
{spec}

## Reference code
```python
{reference_code}
```

## Task
{codegen_task}

Write ONLY the Python fragment. Raw Python, no markdown fences.
The fragment patches trilevel_research/core/adaptive_mechanism_schedule.py
(AdaptiveMechanismSchedule class).

For replace_method on decide: output ONLY the decide method (def decide(...): ...).
Use fire_level2/fire_level3 and return ScheduleDecision(reason=..., batch_size=...).
Do NOT redefine the class or import nonexistent modules.
"""


@dataclass
class ScheduleMechanismResult:
    session_id: str
    hypothesis: str
    mechanism_name: str
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


class ScheduleMechanismResearcher(BaseMechanismResearcher):
    """Level-2 researcher targeting AdaptiveMechanismSchedule (ouroboros)."""

    TARGET_CLASS = "AdaptiveMechanismSchedule"

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str = "",
        provider: str = "deepseek",
        max_code_retries: int = 3,
    ):
        super().__init__(
            model=model,
            provider=provider,
            api_key=api_key,
            max_code_retries=max_code_retries,
        )
        self.tabu_registry = None

    def _get_explore_prompt(self, **kwargs) -> tuple[str, str]:
        return (
            EXPLORE_PROMPT.format(
                n_iters=kwargs["n_iters"],
                trace_summary=kwargs["trace_summary"],
                schedule_summary=kwargs["schedule_summary"],
                bottleneck=kwargs["bottleneck"],
            ),
            EXPLORE_SYSTEM,
        )

    def _get_specify_prompt(self, selected_hypothesis: str, critique: str, **kwargs) -> tuple[str, str]:
        return (
            SPECIFY_PROMPT.format(
                selected_hypothesis=selected_hypothesis,
                critique_notes=critique[-2000:],
                schedule_section=kwargs.get("schedule_section", "")[:3000],
            ),
            SPECIFY_SYSTEM,
        )

    def _get_codegen_prompt(self, spec: str, reference_code: str, **kwargs) -> str:
        return CODEGEN_PROMPT.format(
            spec=spec,
            reference_code=reference_code,
            codegen_task=kwargs.get("codegen_task", ""),
        )

    def _get_reference_code(self, **kwargs) -> str:
        schedule_code = kwargs.get("schedule_code", "")
        impl_strategy = kwargs.get("impl_strategy", "replace_method")
        return self._read_reference_code(schedule_code, impl_strategy)

    def research(
        self,
        trace_summary: str,
        schedule_code: str,
        session_dir: Path,
        bottleneck: str = "",
        *,
        runner_code: str = "",
    ) -> ScheduleMechanismResult:
        """Research a schedule patch. Accepts runner_code alias for controller compatibility."""
        del runner_code
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        schedule_summary = self._summarize_existing_mechanisms(schedule_code)
        n_iters = trace_summary.count("iter ") or trace_summary.count("\n")
        if not bottleneck:
            bottleneck = self._infer_bottleneck(trace_summary)

        explore_kwargs = {
            "n_iters": n_iters,
            "trace_summary": trace_summary,
            "schedule_summary": schedule_summary,
            "bottleneck": bottleneck,
        }
        if self.tabu_registry is not None:
            explore_kwargs["tabu_block"] = self.tabu_registry.to_prompt_block()

        session = self._run_session(
            session_dir=session_dir,
            session_id=session_id,
            log_prefix="ScheduleMechResearch",
            explore_kwargs=explore_kwargs,
            specify_kwargs={
                "schedule_section": self._extract_schedule_section(schedule_code),
                "schedule_code": schedule_code,
            },
            codegen_kwargs={
                "schedule_code": schedule_code,
                "impl_strategy": "replace_method",
            },
        )

        mechanism_name, impl_strategy, target = session["spec_metadata"]
        result = ScheduleMechanismResult(
            session_id=session_id,
            hypothesis=session["selected_hypothesis"],
            mechanism_name=mechanism_name,
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

    def apply(self, schedule_path: Path, result: ScheduleMechanismResult) -> bool:
        schedule_path = Path(schedule_path)
        original = schedule_path.read_text(encoding="utf-8")
        backup_path = schedule_path.with_suffix(f".py.bak_{result.session_id}")
        backup_path.write_text(original, encoding="utf-8")

        strategy = result.implementation_strategy
        code = self._normalize_codegen(result.code.strip(), strategy, result.target)
        try:
            if strategy == "new_helper_class":
                patched = self._insert_helper_class(original, code)
            elif strategy == "replace_method":
                patched = self._replace_method(original, result.target, code)
            elif strategy == "new_method":
                patched = self._insert_method(original, code)
            elif strategy == "modify_init":
                patched = self._append_to_init(original, code)
            else:
                patched = self._replace_method(original, "decide", code)
        except Exception as exc:
            result.validation_error = f"patch_apply_error: {exc}"
            return False

        error = self._syntax_check(patched)
        if error:
            schedule_path.write_text(original, encoding="utf-8")
            result.validation_error = f"syntax_error: {error}"
            return False

        schedule_path.write_text(patched, encoding="utf-8")
        (result.session_dir / "05_patched_schedule.py").write_text(patched, encoding="utf-8")
        result.applied = True
        return True

    def validate(self, schedule_path: Path) -> bool:
        schedule_path = Path(schedule_path)
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        try:
            spec = importlib.util.spec_from_file_location(
                f"_sched_validate_{schedule_path.stat().st_mtime_ns}",
                schedule_path,
            )
            if spec is None or spec.loader is None:
                return False
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            cls = getattr(module, self.TARGET_CLASS, None)
            if cls is None:
                return False
            instance = cls()
            decision = instance.decide([], [], completed_outer_cycles=1)
            return hasattr(decision, "fire_level2") and hasattr(decision, "fire_level3")
        except Exception as exc:
            logger.debug("Schedule validate failed: %s", exc)
            return False

    def _normalize_codegen(self, code: str, strategy: str, target: str) -> str:
        code = code.strip()
        if strategy != "replace_method":
            return code
        if code.startswith(f"def {target}("):
            return code
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == target:
                    lines = code.splitlines(keepends=True)
                    return "".join(lines[item.lineno - 1 : item.end_lineno])
        match = re.search(rf"(    def {re.escape(target)}\(.*)", code, flags=re.DOTALL)
        if match:
            return match.group(1).rstrip()
        return code

    def _insert_helper_class(self, original: str, new_class_code: str) -> str:
        marker = "\nclass AdaptiveMechanismSchedule:"
        idx = original.find(marker)
        if idx == -1:
            marker = "class AdaptiveMechanismSchedule:"
            idx = original.find(marker)
        if idx == -1:
            raise ValueError("Could not find AdaptiveMechanismSchedule class")
        insert_block = f"\n\n{new_class_code.strip()}\n\n"
        return original[:idx] + insert_block + original[idx:]

    def _replace_method(self, original: str, method_name: str, new_method_code: str) -> str:
        try:
            return self._ast_replace_method(original, method_name, new_method_code)
        except Exception:
            pattern = rf"(    def {re.escape(method_name)}\(.*?\n)((?:(?!    def |\nclass ).)*)"
            new_block = textwrap.indent(new_method_code.strip(), "    ") + "\n"
            patched, count = re.subn(pattern, new_block, original, count=1, flags=re.DOTALL)
            if count == 0:
                raise ValueError(f"Method '{method_name}' not found")
            return patched

    def _ast_replace_method(self, original: str, method_name: str, new_method_code: str) -> str:
        tree = ast.parse(original)
        lines = original.splitlines(keepends=True)
        target_class = next(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == self.TARGET_CLASS
            ),
            None,
        )
        if target_class is None:
            raise ValueError("AdaptiveMechanismSchedule not found")
        target_node = next(
            (
                n
                for n in target_class.body
                if isinstance(n, ast.FunctionDef) and n.name == method_name
            ),
            None,
        )
        if target_node is None:
            raise ValueError(f"Method '{method_name}' not found")
        new_method_indented = (
            textwrap.indent(new_method_code.strip(), "    ") + "\n"
            if not new_method_code.strip().startswith("    ")
            else new_method_code.rstrip() + "\n"
        )
        return "".join(
            lines[: target_node.lineno - 1]
            + [new_method_indented]
            + lines[target_node.end_lineno :]
        )

    def _insert_method(self, original: str, new_method_code: str) -> str:
        anchor = "    def decide("
        idx = original.find(anchor)
        if idx == -1:
            return original.rstrip() + "\n" + textwrap.indent(new_method_code.strip(), "    ") + "\n"
        insert_text = "\n" + textwrap.indent(new_method_code.strip(), "    ") + "\n\n"
        return original[:idx] + insert_text + original[idx:]

    def _append_to_init(self, original: str, new_init_code: str) -> str:
        tree = ast.parse(original)
        lines = original.splitlines(keepends=True)
        target_class = next(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == self.TARGET_CLASS
            ),
            None,
        )
        if target_class is None:
            raise ValueError("AdaptiveMechanismSchedule not found")
        init_node = next(
            (n for n in target_class.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
            None,
        )
        if init_node is None:
            raise ValueError("__init__ not found")
        snippet = textwrap.indent(new_init_code.strip(), "        ") + "\n"
        return "".join(lines[: init_node.end_lineno] + [snippet] + lines[init_node.end_lineno :])

    def _parse_spec_metadata(self, spec: str, session_id: str) -> tuple[str, str, str]:
        mechanism_name = f"schedule_mechanism_{session_id}"
        impl_strategy = "replace_method"
        target = "decide"
        valid = {"new_method", "replace_method", "new_helper_class", "modify_init"}
        for line in spec.splitlines():
            low = line.lower()
            if "mechanism name" in low and ":" in line:
                candidate = line.split(":", 1)[1].strip().strip("`").strip("*")
                if candidate and " " not in candidate:
                    mechanism_name = re.sub(r"[^a-zA-Z0-9_]", "", candidate)
            if "implementation strategy" in low:
                for strat in valid:
                    if strat in low.replace(" ", "_"):
                        impl_strategy = strat
                        break
            if "target" in low and ":" in line:
                candidate = line.split(":", 1)[1].strip().strip("`").strip("*")
                if candidate and len(candidate) < 80:
                    target = candidate
        return mechanism_name, impl_strategy, target

    def _summarize_existing_mechanisms(self, schedule_code: str) -> str:
        idx = schedule_code.find('"""')
        if idx == -1:
            return schedule_code[:1200]
        end = schedule_code.find('"""', idx + 3)
        if end == -1:
            return schedule_code[idx : idx + 800]
        decide_idx = schedule_code.find("    def decide(", end)
        if decide_idx != -1:
            return schedule_code[idx : decide_idx + 1500]
        return schedule_code[idx : end + 3]

    def _extract_schedule_section(self, schedule_code: str) -> str:
        idx = schedule_code.find("class AdaptiveMechanismSchedule:")
        return schedule_code[idx : idx + 3500] if idx != -1 else schedule_code[:3500]

    def _infer_bottleneck(self, trace_summary: str) -> str:
        low = trace_summary.lower()
        if low.count("discard") > 5:
            return "Too many discards — L2 may fire too often or at wrong times."
        if "defer l2" in low:
            return "L2 deferred while search stagnates — schedule too conservative."
        return "L2/L3 cadence may not match inner-loop dynamics."

    def _build_codegen_task(self, mechanism_name: str, impl_strategy: str, target: str, spec: str) -> str:
        del spec
        if impl_strategy == "replace_method":
            return f"Write a REPLACEMENT for AdaptiveMechanismSchedule.{target}."
        if impl_strategy == "modify_init":
            return "Write statements to append to AdaptiveMechanismSchedule.__init__ (8-space indent)."
        if impl_strategy == "new_helper_class":
            return f"Implement helper class '{mechanism_name}' used by AdaptiveMechanismSchedule."
        return f"Implement new method '{target}' on AdaptiveMechanismSchedule."

    def _read_reference_code(self, schedule_code: str, impl_strategy: str) -> str:
        if impl_strategy == "replace_method":
            idx = schedule_code.find("    def decide(")
            if idx != -1:
                return schedule_code[idx : idx + 2000]
        idx = schedule_code.find("class AdaptiveMechanismSchedule:")
        return schedule_code[idx : idx + 2000] if idx != -1 else schedule_code[:1500]

    def _save_summary(self, result: ScheduleMechanismResult, session_dir: Path) -> None:
        summary = {
            "session_id": result.session_id,
            "mechanism_name": result.mechanism_name,
            "implementation_strategy": result.implementation_strategy,
            "target": result.target,
            "code_retries": result.code_retries,
            "patch_target": "schedule",
        }
        (session_dir / "06_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
