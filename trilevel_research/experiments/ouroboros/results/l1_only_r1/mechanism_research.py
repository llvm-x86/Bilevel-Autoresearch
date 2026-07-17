"""Level 2 mechanism research for gpu_bench — patches GpuBenchRunner in runner.py."""
from __future__ import annotations

import ast
import json
import logging
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from core.base_mechanism_research import (
    BaseMechanismResearcher,
)

logger = logging.getLogger(__name__)

EXPLORE_SYSTEM = """You are a meta-researcher specializing in GPU hyperparameter search.
Propose concrete mechanism changes to improve GpuBenchRunner's ability to find
lower val_bpb on an AMD RX 580 HIP MLP benchmark."""

EXPLORE_PROMPT = """\
## Inner Loop Trace Analysis

### Iteration trace (most recent {n_iters} iterations):
{trace_summary}

### Current runner.py mechanisms:
{runner_summary}

### Bottleneck:
{bottleneck}

---

Propose 3-4 mechanism improvements to GpuBenchRunner (runner.py).

For EACH hypothesis:
1. **Domain**: inspiration field
2. **Core idea**: one sentence
3. **Implementation target**: method/class in runner.py
4. **Why it helps**: causal argument
5. **Implementation complexity**: 1–5
6. **Risk**: low / medium / high
"""

SPECIFY_SYSTEM = """You are a senior engineer writing a patch spec for GpuBenchRunner."""

SPECIFY_PROMPT = """\
## Selected Hypothesis
{selected_hypothesis}

## Critique notes
{critique_notes}

## Current runner.py section
```python
{runner_section}
```

Write an implementation specification:

1. **Mechanism name** (snake_case):
2. **Implementation strategy**: new_method | replace_method | new_helper_class | modify_init
3. **Target**: method or class name
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
The fragment patches domains/gpu_bench_opt/runner.py (GpuBenchRunner class).
"""


@dataclass
class GpuBenchMechanismResult:
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


class GpuBenchMechanismResearcher(BaseMechanismResearcher):
    """Level-2 researcher for gpu_bench — patches runner.py."""

    RUNNER_CLASS = "GpuBenchRunner"

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
                runner_summary=kwargs["runner_summary"],
                bottleneck=kwargs["bottleneck"],
            ),
            EXPLORE_SYSTEM,
        )

    def _get_specify_prompt(self, selected_hypothesis: str, critique: str, **kwargs) -> tuple[str, str]:
        return (
            SPECIFY_PROMPT.format(
                selected_hypothesis=selected_hypothesis,
                critique_notes=critique[-2000:],
                runner_section=kwargs.get("runner_section", "")[:3000],
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
        runner_code = kwargs.get("runner_code", "")
        impl_strategy = kwargs.get("impl_strategy", "new_helper_class")
        return self._read_reference_code(runner_code, impl_strategy)

    def research(
        self,
        trace_summary: str,
        runner_code: str,
        session_dir: Path,
        bottleneck: str = "",
    ) -> GpuBenchMechanismResult:
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        runner_summary = self._summarize_existing_mechanisms(runner_code)
        n_iters = trace_summary.count("iter ") or trace_summary.count("\n")
        if not bottleneck:
            bottleneck = self._infer_bottleneck(trace_summary)

        explore_kwargs = {
            "n_iters": n_iters,
            "trace_summary": trace_summary,
            "runner_summary": runner_summary,
            "bottleneck": bottleneck,
        }
        if self.tabu_registry is not None:
            explore_kwargs["tabu_block"] = self.tabu_registry.to_prompt_block()

        session = self._run_session(
            session_dir=session_dir,
            session_id=session_id,
            log_prefix="GpuBenchMechResearch",
            explore_kwargs=explore_kwargs,
            specify_kwargs={
                "runner_section": self._extract_runner_section(runner_code),
                "runner_code": runner_code,
            },
            codegen_kwargs={
                "runner_code": runner_code,
                "impl_strategy": "new_helper_class",
            },
        )

        mechanism_name, impl_strategy, target = session["spec_metadata"]
        result = GpuBenchMechanismResult(
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

    def apply(self, runner_path: Path, result: GpuBenchMechanismResult) -> bool:
        runner_path = Path(runner_path)
        original = runner_path.read_text(encoding="utf-8")
        backup_path = runner_path.with_suffix(f".py.bak_{result.session_id}")
        backup_path.write_text(original, encoding="utf-8")

        strategy = result.implementation_strategy
        code = result.code.strip()
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
                patched = self._insert_helper_class(original, code)
        except Exception as exc:
            result.validation_error = f"patch_apply_error: {exc}"
            return False

        error = self._syntax_check(patched)
        if error:
            runner_path.write_text(original, encoding="utf-8")
            result.validation_error = f"syntax_error: {error}"
            return False

        runner_path.write_text(patched, encoding="utf-8")
        (result.session_dir / "05_patched_runner.py").write_text(patched, encoding="utf-8")
        result.applied = True
        return True

    def validate(self, runner_path: Path) -> bool:
        import subprocess

        runner_path = Path(runner_path)
        project_root = runner_path.parent.parent.parent
        module_path = (
            runner_path.relative_to(project_root)
            .with_suffix("")
            .as_posix()
            .replace("/", ".")
        )
        cmd = [
            "python",
            "-c",
            f"import sys; sys.path.insert(0, '{project_root}'); "
            f"import {module_path}; print('IMPORT_OK')",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return "IMPORT_OK" in proc.stdout
        except Exception:
            return False

    def _insert_helper_class(self, original: str, new_class_code: str) -> str:
        marker = "\nclass GpuBenchRunner:"
        idx = original.find(marker)
        if idx == -1:
            marker = "class GpuBenchRunner:"
            idx = original.find(marker)
        if idx == -1:
            raise ValueError("Could not find GpuBenchRunner class")
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
        runner_class = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == self.RUNNER_CLASS),
            None,
        )
        if runner_class is None:
            raise ValueError("GpuBenchRunner not found")
        target_node = next(
            (n for n in runner_class.body if isinstance(n, ast.FunctionDef) and n.name == method_name),
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
        anchor = "    def _run_trial("
        idx = original.find(anchor)
        if idx == -1:
            return original.rstrip() + "\n" + textwrap.indent(new_method_code.strip(), "    ") + "\n"
        insert_text = "\n" + textwrap.indent(new_method_code.strip(), "    ") + "\n\n"
        return original[:idx] + insert_text + original[idx:]

    def _append_to_init(self, original: str, new_init_code: str) -> str:
        tree = ast.parse(original)
        lines = original.splitlines(keepends=True)
        runner_class = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == self.RUNNER_CLASS),
            None,
        )
        if runner_class is None:
            raise ValueError("GpuBenchRunner not found")
        init_node = next(
            (n for n in runner_class.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
            None,
        )
        if init_node is None:
            raise ValueError("__init__ not found")
        snippet = textwrap.indent(new_init_code.strip(), "        ") + "\n"
        return "".join(lines[: init_node.end_lineno] + [snippet] + lines[init_node.end_lineno :])

    def _parse_spec_metadata(self, spec: str, session_id: str) -> tuple[str, str, str]:
        mechanism_name = f"gpu_bench_mechanism_{session_id}"
        impl_strategy = "new_helper_class"
        target = f"GpuBenchHelper_{session_id}"
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

    def _summarize_existing_mechanisms(self, runner_code: str) -> str:
        idx = runner_code.find('"""')
        if idx == -1:
            return "(no docstring — mechanisms unknown)"
        end = runner_code.find('"""', idx + 3)
        if end == -1:
            return runner_code[idx : idx + 800]
        return runner_code[idx : end + 3]

    def _extract_runner_section(self, runner_code: str) -> str:
        idx = runner_code.find("class GpuBenchRunner:")
        return runner_code[idx : idx + 3000] if idx != -1 else runner_code[:3000]

    def _infer_bottleneck(self, trace_summary: str) -> str:
        low = trace_summary.lower()
        if low.count("crash") > 2:
            return "High crash rate — need better proposal validation."
        if low.count("discard") > 5:
            return "Too many discards — proposals not improving val_bpb."
        return "Search converging slowly — need better exploration."

    def _build_codegen_task(self, mechanism_name: str, impl_strategy: str, target: str, spec: str) -> str:
        if impl_strategy == "replace_method":
            return f"Write a REPLACEMENT for GpuBenchRunner.{target}."
        if impl_strategy == "modify_init":
            return "Write statements to append to GpuBenchRunner.__init__ (8-space indent)."
        return f"Implement mechanism '{mechanism_name}' for GpuBenchRunner."

    def _read_reference_code(self, runner_code: str, impl_strategy: str) -> str:
        if impl_strategy == "replace_method":
            idx = runner_code.find("    def run_iteration(")
            if idx != -1:
                return runner_code[idx : idx + 1500]
        idx = runner_code.find("class GpuBenchRunner:")
        return runner_code[idx : idx + 1500] if idx != -1 else runner_code[:1000]

    def _save_summary(self, result: GpuBenchMechanismResult, session_dir: Path) -> None:
        summary = {
            "session_id": result.session_id,
            "mechanism_name": result.mechanism_name,
            "implementation_strategy": result.implementation_strategy,
            "target": result.target,
            "code_retries": result.code_retries,
        }
        (session_dir / "06_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
