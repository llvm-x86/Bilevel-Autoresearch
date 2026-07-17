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
from types import SimpleNamespace

from core.base_mechanism_research import CODEGEN_SYSTEM, BaseMechanismResearcher

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Fixture inner trace for validate() — improving streak after early discards.
VALIDATE_FIXTURE_TRACE: list[dict] = [
    {"status": "discard", "val_bpb": 2.0},
    {"status": "discard", "val_bpb": 1.95},
    {"status": "keep", "val_bpb": 1.85},
    {"status": "keep", "val_bpb": 1.80},
    {"status": "keep", "val_bpb": 1.78},
]

BOOTSTRAP_MECHANISM_NAME = "bootstrap_improving_streak_defer"

BOOTSTRAP_DECIDE_CODE = textwrap.dedent(
    '''\
    def decide(
        self,
        inner_trace: list[dict],
        l2_sessions: list,
        completed_outer_cycles: int,
        config=None,
    ):
        interval = config.level2_interval if config else self.level2_interval
        l3_interval = config.level3_interval if config else self.level3_interval
        batch_size = interval
        discard_threshold = 0.55

        fire_l2 = True
        fire_l3 = False
        reasons: list[str] = []

        recent = inner_trace[-self.lookback_iters :] if inner_trace else []
        if recent:
            n = len(recent)
            discards = sum(1 for r in recent if r.get("status") == "discard")
            keeps = sum(1 for r in recent if r.get("status") == "keep")
            discard_rate = discards / n

            keeps_streak = 0
            for record in reversed(recent):
                if record.get("status") == "keep":
                    keeps_streak += 1
                else:
                    break

            if discard_rate > discard_threshold:
                reasons.append(f"high discard rate ({discard_rate:.0%})")
                fire_l2 = True
            elif keeps == 0 and n >= 3:
                reasons.append("zero keeps in lookback window")
                fire_l2 = True
            elif keeps_streak >= 2 and completed_outer_cycles % interval != 0:
                fire_l2 = False
                reasons.append("improving streak; defer L2")
            elif completed_outer_cycles % interval != 0 and discard_rate < 0.5:
                fire_l2 = False
                reasons.append("inner loop improving; defer L2")

        if completed_outer_cycles % interval == 0:
            fire_l2 = True
            if "defer L2" in " ".join(reasons):
                reasons = [f"fixed interval ({interval} cycles) overrides defer"]
            elif not reasons:
                reasons.append(f"fixed L2 interval ({interval} cycles)")

        if l2_sessions:
            attempted = [s for s in l2_sessions if not getattr(s, "blocked_by_tabu", False)]
            if attempted:
                reverts = sum(
                    1 for s in attempted
                    if getattr(s, "applied", False) and getattr(s, "validated", None) is False
                )
                revert_rate = reverts / len(attempted)
                if revert_rate >= self.revert_rate_threshold:
                    fire_l3 = True
                    reasons.append(f"L2 revert rate {revert_rate:.0%}")

            consecutive_fail = 0
            for s in reversed(l2_sessions):
                if getattr(s, "applied", False):
                    break
                consecutive_fail += 1
            if consecutive_fail >= 2:
                fire_l3 = True
                reasons.append(f"{consecutive_fail} consecutive L2 failures")

            names = [getattr(s, "mechanism_name", "") for s in l2_sessions[-3:]]
            if len(names) >= 2 and names[-1] == names[-2]:
                fire_l3 = True
                reasons.append("duplicate mechanism name in consecutive rounds")

        l2_rounds = len(l2_sessions)
        if l2_rounds > 0 and l2_rounds % l3_interval == 0:
            fire_l3 = True
            reasons.append(f"L3 interval ({l3_interval} L2 rounds)")

        if not reasons:
            reasons.append("default schedule")

        return ScheduleDecision(
            fire_level2=fire_l2,
            fire_level3=fire_l3,
            reason="; ".join(reasons),
            batch_size=batch_size,
        )
    '''
)

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

SCHEDULE_FIX_PROMPT = """\
The decide() method you generated failed validation:

```
{error}
```

Here is the code that failed:

```python
{code}
```

Fix the error. Return ONLY the corrected `def decide(self, inner_trace, l2_sessions, completed_outer_cycles, config=None):` method body.
Rules:
- NO new classes, NO imports, NO helper classes.
- Use ONLY these self fields: level2_interval, level3_interval, discard_rate_threshold, revert_rate_threshold, lookback_iters.
- Access l2_sessions items with getattr(s, 'applied', False), getattr(s, 'validated', None), getattr(s, 'blocked_by_tabu', False), getattr(s, 'mechanism_name', '').
- Set local vars fire_level2 and fire_level3, then return ScheduleDecision(fire_level2=..., fire_level3=..., reason=..., batch_size=...).
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

For replace_method on decide: output ONLY this method signature and body:
  def decide(self, inner_trace, l2_sessions, completed_outer_cycles, config=None):

STRICT RULES:
- NO new classes, NO imports, NO helper classes — only the decide() method.
- Use ONLY these self fields: level2_interval, level3_interval, discard_rate_threshold, revert_rate_threshold, lookback_iters.
- Access l2_sessions items with getattr(s, 'applied', False), getattr(s, 'validated', None), getattr(s, 'blocked_by_tabu', False), getattr(s, 'mechanism_name', '').
- Set local vars fire_level2 and fire_level3, then return ScheduleDecision(fire_level2=..., fire_level3=..., reason=..., batch_size=...).
- Do NOT reference self.min_discard_rate_threshold or any other self.* field not listed above.
"""

ALLOWED_SELF_ATTRS = frozenset({
    "level2_interval",
    "level3_interval",
    "discard_rate_threshold",
    "revert_rate_threshold",
    "lookback_iters",
})

SELF_ATTR_REPLACEMENTS = {
    "min_discard_rate_threshold": "0.0",
    "max_discard_rate_threshold": "self.discard_rate_threshold",
    "discard_threshold": "self.discard_rate_threshold",
}

L2_SESSION_ATTR_DEFAULTS = {
    "applied": "False",
    "validated": "None",
    "blocked_by_tabu": "False",
    "mechanism_name": "''",
}


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

    def validate(
        self,
        schedule_path: Path,
        result: ScheduleMechanismResult | None = None,
    ) -> bool:
        schedule_path = Path(schedule_path)
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        module_name = f"_sched_validate_{schedule_path.stat().st_mtime_ns}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, schedule_path)
            if spec is None or spec.loader is None:
                msg = "validate_error: could not load schedule module spec"
                if result is not None:
                    result.validation_error = msg
                return False
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            cls = getattr(module, self.TARGET_CLASS, None)
            if cls is None:
                msg = f"validate_error: {self.TARGET_CLASS} not found in patched module"
                if result is not None:
                    result.validation_error = msg
                return False
            instance = cls()
            l2_sessions = [
                SimpleNamespace(
                    applied=True,
                    validated=False,
                    blocked_by_tabu=False,
                    mechanism_name="fixture_mech",
                ),
                SimpleNamespace(
                    applied=False,
                    validated=None,
                    blocked_by_tabu=True,
                    mechanism_name="tabu_mech",
                ),
            ]
            decision = instance.decide(
                VALIDATE_FIXTURE_TRACE,
                l2_sessions,
                completed_outer_cycles=3,
            )
            schedule_decision = getattr(module, "ScheduleDecision", None)
            if schedule_decision is not None and not isinstance(decision, schedule_decision):
                msg = "validate_error: decide() did not return ScheduleDecision"
                if result is not None:
                    result.validation_error = msg
                return False
            if not (
                isinstance(getattr(decision, "fire_level2", None), bool)
                and isinstance(getattr(decision, "fire_level3", None), bool)
            ):
                msg = "validate_error: decide() must return bool fire_level2 and fire_level3"
                if result is not None:
                    result.validation_error = msg
                return False
            if result is not None:
                result.validated = True
                result.validation_error = ""
            return True
        except Exception as exc:
            msg = f"validate_error: {type(exc).__name__}: {exc}"
            if result is not None:
                result.validation_error = msg
            logger.debug("Schedule validate failed: %s", exc)
            return False
        finally:
            sys.modules.pop(module_name, None)

    def _extract_method_body(self, code: str, target: str) -> str:
        """Extract a single method body; ignore unrelated helper classes."""
        code = code.strip()
        if not code:
            raise ValueError("empty codegen fragment")

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise ValueError(f"syntax error in codegen: {exc}") from exc

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if node.name == self.TARGET_CLASS:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == target:
                            lines = code.splitlines(keepends=True)
                            extracted = "".join(
                                lines[item.lineno - 1 : item.end_lineno]
                            ).rstrip()
                            return textwrap.dedent(extracted)
                    raise ValueError(
                        f"class {self.TARGET_CLASS} redefinition must not be used for replace_method"
                    )
                continue

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == target:
                lines = code.splitlines(keepends=True)
                return "".join(lines[node.lineno - 1 : node.end_lineno]).rstrip()

        match = re.search(rf"(    def {re.escape(target)}\(.*)", code, flags=re.DOTALL)
        if match:
            return match.group(1).rstrip()

        top_level = re.search(rf"(^def {re.escape(target)}\(.*)", code, flags=re.MULTILINE | re.DOTALL)
        if top_level:
            start = top_level.start(1)
            return code[start:].rstrip()

        raise ValueError(f"could not extract def {target}(...) from codegen fragment")

    def _sanitize_decide_code(self, code: str) -> str:
        """Replace unknown self.* attrs and bare l2_sessions attribute access."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        class Sanitizer(ast.NodeTransformer):
            def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
                self.generic_visit(node)
                if isinstance(node.value, ast.Name) and node.value.id == "self":
                    attr = node.attr
                    if attr in ALLOWED_SELF_ATTRS:
                        return node
                    if attr in SELF_ATTR_REPLACEMENTS:
                        replacement = SELF_ATTR_REPLACEMENTS[attr]
                        if replacement.startswith("self."):
                            sub_attr = replacement.split(".", 1)[1]
                            return ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr=sub_attr,
                                ctx=node.ctx,
                            )
                        return ast.Constant(value=ast.literal_eval(replacement))
                    return ast.Call(
                        func=ast.Name(id="getattr", ctx=ast.Load()),
                        args=[
                            ast.Name(id="self", ctx=ast.Load()),
                            ast.Constant(value=attr),
                            ast.Constant(value=None),
                        ],
                        keywords=[],
                    )
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id in {"s", "session", "rec", "record"}
                    and node.attr in L2_SESSION_ATTR_DEFAULTS
                ):
                    default = L2_SESSION_ATTR_DEFAULTS[node.attr]
                    return ast.Call(
                        func=ast.Name(id="getattr", ctx=ast.Load()),
                        args=[
                            node.value,
                            ast.Constant(value=node.attr),
                            ast.Constant(value=ast.literal_eval(default)),
                        ],
                        keywords=[],
                    )
                return node

        sanitized_tree = Sanitizer().visit(tree)
        ast.fix_missing_locations(sanitized_tree)
        return ast.unparse(sanitized_tree)

    def _validate_codegen_schema(self, code: str, strategy: str, target: str) -> str | None:
        """Return an error string if codegen violates replace_method constraints."""
        if strategy != "replace_method" or target != "decide":
            return None
        try:
            body = self._extract_method_body(code, target)
        except ValueError as exc:
            return str(exc)
        if "fire_level2" not in body or "fire_level3" not in body:
            return "decide() must set fire_level2 and fire_level3"
        if "ScheduleDecision" not in body:
            return "decide() must return ScheduleDecision(...)"
        if re.search(rf"class\s+{re.escape(self.TARGET_CLASS)}\b", body):
            return f"decide fragment must not redefine {self.TARGET_CLASS}"
        return None

    def _bootstrap_patch_code(self) -> str:
        return BOOTSTRAP_DECIDE_CODE.strip()

    def _generate_with_retries(
        self,
        spec: str,
        reference_code: str,
        session_dir: Path,
        codegen_kwargs: dict,
    ) -> tuple[str, int]:
        """Generate code with schema checks; fall back to known-good bootstrap patch."""
        impl_strategy = codegen_kwargs.get("impl_strategy", "replace_method")
        _, _, target = self._parse_spec_metadata(spec, "codegen")
        codegen_prompt = self._get_codegen_prompt(
            spec=spec,
            reference_code=reference_code,
            **codegen_kwargs,
        )
        code = self.client.call(
            codegen_prompt,
            system=CODEGEN_SYSTEM,
            max_tokens=6000,
        )
        code = self._strip_fences(code)

        for attempt in range(self.max_code_retries):
            (session_dir / f"04_code_attempt_{attempt + 1}.py").write_text(
                code, encoding="utf-8"
            )
            syntax_error = self._syntax_check(code)
            schema_error = self._validate_codegen_schema(code, impl_strategy, target)
            if syntax_error is None and schema_error is None:
                logger.info("[CodeGen] Code OK on attempt %d", attempt + 1)
                return code, attempt

            error = syntax_error or schema_error or "unknown error"
            logger.warning("[CodeGen] Validation failed attempt %d: %s", attempt + 1, error[:200])
            (session_dir / f"04_error_{attempt + 1}.txt").write_text(error, encoding="utf-8")

            code = self.client.call(
                SCHEDULE_FIX_PROMPT.format(error=error[:2000], code=code[:4000]),
                system=CODEGEN_SYSTEM,
                max_tokens=6000,
            )
            code = self._strip_fences(code)

        bootstrap = self._bootstrap_patch_code()
        schema_error = self._validate_codegen_schema(bootstrap, impl_strategy, target)
        if schema_error:
            raise RuntimeError(f"Bootstrap patch failed schema check: {schema_error}")

        (session_dir / "04_code_bootstrap.py").write_text(bootstrap, encoding="utf-8")
        logger.warning(
            "[CodeGen] LLM failed after %d retries; using bootstrap patch",
            self.max_code_retries,
        )
        return bootstrap, self.max_code_retries

    def _normalize_codegen(self, code: str, strategy: str, target: str) -> str:
        code = code.strip()
        if strategy != "replace_method":
            return code
        try:
            extracted = self._extract_method_body(code, target)
        except ValueError:
            if code.startswith(f"def {target}("):
                return code
            raise
        if strategy == "replace_method" and target == "decide":
            return self._sanitize_decide_code(extracted)
        return extracted

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
        del spec  # task text is strategy-dependent only
        if impl_strategy == "replace_method":
            return (
                "Write a REPLACEMENT for AdaptiveMechanismSchedule.decide(). "
                "Output ONLY def decide(self, inner_trace, l2_sessions, completed_outer_cycles, config=None): "
                "with NO helper classes and NO imports."
            )
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
            "applied": result.applied,
            "validated": result.validated,
            "validation_error": result.validation_error,
        }
        (session_dir / "06_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    def update_session_summary(self, result: ScheduleMechanismResult, session_dir: Path) -> None:
        """Rewrite 06_summary.json after apply/validate in the controller."""
        self._save_summary(result, session_dir)
