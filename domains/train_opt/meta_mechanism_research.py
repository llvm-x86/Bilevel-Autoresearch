"""Level 3 meta-mechanism research for the training domain.

Patches mechanism_research.py to improve Level-2's research process.
"""
from __future__ import annotations

import re

from core.base_meta_mechanism_research import BaseMetaMechanismResearcher

EXPLORE_SYSTEM = """You are a meta-meta-researcher optimizing how an AI system discovers
new hyperparameter search mechanisms. Your job is to improve mechanism_research.py —
the Level-2 code that generates patches to runner.py.

Focus on: mechanism tabu registry, adaptive L2 scheduling, validation harness,
and explore-prompt enrichment."""

EXPLORE_PROMPT = """\
## Level-2 Session Trace

{l2_trace_summary}

### Current mechanism_research.py overview:
{mech_summary}

### Identified Level-2 bottleneck:
{bottleneck}

---

## Your Task

Propose 3-4 structural improvements to mechanism_research.py (Level 2).

For EACH hypothesis:
1. **Target**: which method/class to add or modify (e.g. research(), __init__, tabu wiring)
2. **Core idea**: one sentence
3. **Why it helps Level 2**: causal argument from the session trace
4. **Implementation complexity**: 1–5
5. **Risk**: low / medium / high

Allowed targets ONLY in mechanism_research.py — do NOT propose runner.py changes.
"""

SPECIFY_SYSTEM = """You are a senior engineer writing a precise patch spec for mechanism_research.py."""

SPECIFY_PROMPT = """\
## Selected Hypothesis

{selected_hypothesis}

## Critique notes

{critique_notes}

## Current TrainMechanismResearcher section

```python
{mech_section}
```

---

Write an implementation specification:

1. **Patch name** (snake_case):
2. **Implementation strategy**: new_helper_class | replace_method | modify_init
3. **Target**: method or class name
4. **Step-by-step logic**
5. **Optional schedule patch** (JSON): e.g. {{"level2_interval": 3}}
"""

CODEGEN_PROMPT = """\
## Implementation Specification

{spec}

---

## Reference code (style guide)

```python
{reference_code}
```

## Task

{codegen_task}

Write ONLY the Python fragment to insert. Raw Python, no markdown fences.
The fragment patches domains/train_opt/mechanism_research.py.
"""


class TrainMetaMechanismResearcher(BaseMetaMechanismResearcher):
    """Level-3 researcher for the training domain."""

    RESEARCHER_CLASS_NAME = "TrainMechanismResearcher"

    def _get_explore_prompt(self, **kwargs) -> tuple[str, str]:
        return (
            EXPLORE_PROMPT.format(
                l2_trace_summary=kwargs["l2_trace_summary"],
                mech_summary=kwargs["mech_summary"],
                bottleneck=kwargs["bottleneck"],
            ),
            EXPLORE_SYSTEM,
        )

    def _get_specify_prompt(
        self,
        selected_hypothesis: str,
        critique: str,
        **kwargs,
    ) -> tuple[str, str]:
        return (
            SPECIFY_PROMPT.format(
                selected_hypothesis=selected_hypothesis,
                critique_notes=critique[-2000:],
                mech_section=kwargs.get("mech_section", "")[:3000],
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
        code = kwargs.get("mechanism_research_code", "")
        impl_strategy = kwargs.get("impl_strategy", "new_helper_class")
        if impl_strategy == "replace_method":
            idx = code.find("    def research(")
            if idx != -1:
                return code[idx : idx + 2000]
        idx = code.find("class TrainMechanismResearcher:")
        if idx != -1:
            return code[idx : idx + 1500]
        return code[:1000]

    def _parse_spec_metadata(self, spec: str, session_id: str) -> tuple[str, str, str]:
        patch_name = f"meta_patch_{session_id}"
        impl_strategy = "modify_init"
        target = "__init__"
        valid_strategies = {"new_method", "replace_method", "new_helper_class", "modify_init"}

        for line in spec.splitlines():
            low = line.lower()
            if "patch name" in low and ":" in line:
                candidate = line.split(":", 1)[1].strip().strip("`").strip("*")
                if candidate and " " not in candidate:
                    patch_name = re.sub(r"[^a-zA-Z0-9_]", "", candidate)
            if "implementation strategy" in low:
                for strat in valid_strategies:
                    if strat in low.replace(" ", "_"):
                        impl_strategy = strat
                        break
            if "target" in low and ":" in line:
                candidate = line.split(":", 1)[1].strip().strip("`").strip("*")
                if candidate and len(candidate) < 80:
                    target = candidate

        return patch_name, impl_strategy, target
