"""Level-2 mechanism researcher with tabu registry support for tri-level runs."""
from __future__ import annotations

from domains.train_opt.mechanism_research import (
    EXPLORE_PROMPT,
    EXPLORE_SYSTEM,
    TrainMechanismResearcher,
)


class TriLevelTrainMechanismResearcher(TrainMechanismResearcher):
    """TrainMechanismResearcher with optional tabu prompt injection."""

    def __init__(self, *args, tabu_registry=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tabu_registry = tabu_registry

    def research(self, trace_summary, runner_code, session_dir, bottleneck=""):
        if self.tabu_registry is None:
            return super().research(
                trace_summary=trace_summary,
                runner_code=runner_code,
                session_dir=session_dir,
                bottleneck=bottleneck,
            )

        session_dir = __import__("pathlib").Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        runner_summary = self._summarize_existing_mechanisms(runner_code)
        n_iters = trace_summary.count("iter ") or trace_summary.count("\n")

        if not bottleneck:
            bottleneck = self._infer_bottleneck(trace_summary)

        explore_prompt = EXPLORE_PROMPT.format(
            n_iters=n_iters,
            trace_summary=trace_summary,
            runner_summary=runner_summary,
            bottleneck=bottleneck,
        )
        explore_prompt += (
            "\n\n## Forbidden mechanisms (tabu registry)\n"
            + self.tabu_registry.to_prompt_block()
        )
        exploration = self.client.call(
            explore_prompt,
            system=EXPLORE_SYSTEM,
            max_tokens=4000,
        )
        (session_dir / "01_exploration.md").write_text(exploration, encoding="utf-8")

        return self._finish_research_from_exploration(
            exploration=exploration,
            runner_code=runner_code,
            session_dir=session_dir,
            bottleneck=bottleneck,
            session_id=session_id,
        )

    def _finish_research_from_exploration(
        self,
        *,
        exploration: str,
        runner_code: str,
        session_dir,
        bottleneck: str,
        session_id: str,
    ):
        """Continue research rounds after a tabu-augmented exploration prompt."""
        import logging

        from domains.train_opt.mechanism_research import (
            CRITIQUE_PROMPT,
            CRITIQUE_SYSTEM,
            SPECIFY_PROMPT,
            SPECIFY_SYSTEM,
            TrainMechanismResult,
        )

        logger = logging.getLogger(__name__)

        logger.info(f"[TrainMechResearch {session_id}] Round 2: Critique")
        critique = self.client.call(
            CRITIQUE_PROMPT.format(exploration=exploration),
            system=CRITIQUE_SYSTEM,
            max_tokens=3000,
        )
        (session_dir / "02_critique.md").write_text(critique, encoding="utf-8")

        logger.info(f"[TrainMechResearch {session_id}] Round 3: Specification")
        selected_hypothesis = self._extract_selected(exploration, critique)
        runner_section = self._extract_runner_section(runner_code)

        spec = self.client.call(
            SPECIFY_PROMPT.format(
                selected_hypothesis=selected_hypothesis,
                critique_notes=critique[-2000:],
                runner_section=runner_section[:3000],
            ),
            system=SPECIFY_SYSTEM,
            max_tokens=2500,
        )
        (session_dir / "03_spec.md").write_text(spec, encoding="utf-8")

        mechanism_name, impl_strategy, target = self._parse_spec_metadata(spec, session_id)
        reference_code = self._read_reference_code(runner_code, impl_strategy)
        codegen_task = self._build_codegen_task(mechanism_name, impl_strategy, target, spec)

        code, retries = self._generate_with_retries(
            spec=spec,
            reference_code=reference_code,
            codegen_task=codegen_task,
            session_dir=session_dir,
        )

        result = TrainMechanismResult(
            session_id=session_id,
            hypothesis=selected_hypothesis,
            mechanism_name=mechanism_name,
            implementation_strategy=impl_strategy,
            target=target,
            spec=spec,
            code=code,
            exploration=exploration,
            critique=critique,
            code_retries=retries,
            session_dir=session_dir,
        )
        self._save_summary(result, session_dir)
        return result
