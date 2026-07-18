from __future__ import annotations
import ast, importlib.util, json, logging, re, sys, textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from core.base_mechanism_research import (
    BaseMechanismResearcher,
    ResearchIteration,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class GpuBenchMechanismResearcher(BaseMechanismResearcher):
    """Researches mechanisms to improve GpuBenchRunner's hyperparameter search.

    Detects research bottleneck levels (1–5) from trace history and
    produces structured feedback targeting the specific level.
    """

    research_level: int = 1
    level2_interval: int = 3
    feedback_ratelimit: int = 2
    escalate_after: int = 5

    _last_level: int = 1
    _consecutive_level_calls: int = 0

    def __post_init__(self) -> None:
        """Detect current research level based on repeated failures in trace."""
        # Level detection: scan trace_summary for consecutive failures
        if not self.trace_summary:
            self.research_level = 1
            return

        # Split trace_summary into lines, find [import_failed]
        lines = self.trace_summary.split('\n')
        import_fails = []
        no_progress = 0
        max_consec_import = 0
        current_import_run = 0

        for line in lines:
            if '[import_failed]' in line.lower():
                current_import_run += 1
                max_consec_import = max(max_consec_import, current_import_run)
            else:
                if 'no_progress' in line.lower() or 'no improvement' in line.lower() or 'stuck' in line.lower():
                    no_progress += 1
                current_import_run = 0

        # Detect level
        if max_consec_import >= 2:
            self.research_level = 2
        elif no_progress >= 3:
            self.research_level = 3
        else:
            self.research_level = 1

        self._last_level = self.research_level
        self._consecutive_level_calls = 1

    def generate_feedback(self) -> str:
        """Generate structured feedback based on detected level and trace history."""
        level = self.research_level
        trace = self.trace_summary or ""

        # Base feedback intro
        feedback = f"### Research Feedback (Level {level})\n\n"

        if level == 2:
            constraints = self.extract_structural_constraints()
            if constraints:
                feedback += "**Structural Constraints Detected:**\n"
                for c in constraints:
                    feedback += f"- {c}\n"
                feedback += "\n"
            feedback += (
                "Focus on structural constraints: verify required exports (class/method signatures), "
                "file locations, and import paths. Ensure all dependencies are resolved before "
                "mechanism changes take effect.\n"
            )
        elif level == 3:
            feedback += (
                "Algorithmic correctness suspected. Verify expected outputs match actual behaviour. "
                "Add assertions, check data flow, and watch for silent failures.\n"
            )
        else:  # Level 1
            feedback += (
                "Encourage creativity and exploration. The current mechanism may be too predictable. "
                "Try alternative search strategies, new hyperparameters, or different sensitivity approaches.\n"
            )

        # Append trace insight
        if trace:
            feedback += "\n**Trace Insight:**\n" + trace[:200] + "...\n"

        return feedback

    def extract_structural_constraints(self) -> list[str]:
        """Parse current runner.py for class/method signatures and required exports.

        Returns a list of constraint descriptions (e.g., 'Expected export: GpuBenchRunner').
        """
        constraints: list[str] = []
        runner_path = REPO_ROOT / "runner.py"
        if not runner_path.exists():
            constraints.append("runner.py not found — cannot verify structure.")
            return constraints

        try:
            source = runner_path.read_text()
        except OSError:
            constraints.append("Cannot read runner.py.")
            return constraints

        tree = ast.parse(source)

        # Find classes and their methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                constraints.append(f"Class {node.name} is defined.")
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.FunctionDef):
                        args = [a.arg for a in child.args.args if a.arg != 'self']
                        sig = ', '.join(args)
                        constraints.append(
                            f"Method {node.name}.{child.name}({sig}) expected."
                        )

        # Detect 'if __name__' blocks indicating entry points
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and hasattr(node.test, 'id') and node.test.id == '__name__':
                constraints.append("Entry point (if __name__) pattern detected.")

        return constraints

    def _prepare_explore_kwargs(self) -> dict:
        """Prepare keyword arguments for the explore prompt, with level-appropriate additions."""
        base_kwargs = super()._prepare_explore_kwargs() if hasattr(super(), '_prepare_explore_kwargs') else {}

        # Detect current level (if not done by post_init)
        self.__post_init__()

        level = self.research_level

        # Structural constraints section for Level 2
        if level == 2:
            constraints = self.extract_structural_constraints()
            if constraints:
                structure_block = "\n## Structural Constraints\n"
                structure_block += "\n".join("- " + c for c in constraints)
                base_kwargs.setdefault("extra_sections", []).append(structure_block)

        # Expected Behavior section for Level 3
        if level == 3:
            behavior_block = (
                "\n## Expected Behavior\n"
                "Verify the mechanism produces correct outputs. Compare against baselines. "
                "Add checkpoints and validation steps."
            )
            base_kwargs.setdefault("extra_sections", []).append(behavior_block)

        # Rate-limit feedback
        if self._consecutive_level_calls >= self.feedback_ratelimit:
            if level >= 2:
                feedback = self.generate_feedback()
                base_kwargs.setdefault("extra_sections", []).append(feedback)
            self._consecutive_level_calls = 0
        else:
            self._consecutive_level_calls += 1

        # Escalation
        if self._consecutive_level_calls >= self.escalate_after:
            self.research_level = min(5, self.research_level + 1)
            self._consecutive_level_calls = 0

        return base_kwargs