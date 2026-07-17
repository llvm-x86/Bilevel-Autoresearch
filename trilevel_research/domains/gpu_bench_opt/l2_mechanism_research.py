"""Level-2 mechanism researcher with tabu registry support for gpu_bench tri-level runs."""
from __future__ import annotations

from pathlib import Path

from trilevel_research.domains.gpu_bench_opt.mechanism_research import (
    EXPLORE_PROMPT,
    EXPLORE_SYSTEM,
    GpuBenchMechanismResearcher,
    GpuBenchMechanismResult,
)


class TriLevelGpuBenchMechanismResearcher(GpuBenchMechanismResearcher):
    """GpuBenchMechanismResearcher with optional tabu prompt injection."""

    def __init__(self, *args, tabu_registry=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tabu_registry = tabu_registry

    def research(
        self,
        trace_summary: str,
        runner_code: str,
        session_dir: Path,
        bottleneck: str = "",
    ) -> GpuBenchMechanismResult:
        if self.tabu_registry is None:
            return super().research(
                trace_summary=trace_summary,
                runner_code=runner_code,
                session_dir=session_dir,
                bottleneck=bottleneck,
            )

        from datetime import datetime

        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
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
            session_id=session_id,
        )

    def _finish_research_from_exploration(
        self,
        *,
        exploration: str,
        runner_code: str,
        session_dir: Path,
        session_id: str,
    ) -> GpuBenchMechanismResult:
        from core.base_mechanism_research import CRITIQUE_PROMPT, CRITIQUE_SYSTEM

        from trilevel_research.domains.gpu_bench_opt.mechanism_research import (
            SPECIFY_PROMPT,
            SPECIFY_SYSTEM,
        )

        critique = self.client.call(
            CRITIQUE_PROMPT.format(exploration=exploration),
            system=CRITIQUE_SYSTEM,
            max_tokens=3000,
        )
        (session_dir / "02_critique.md").write_text(critique, encoding="utf-8")

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
            session_dir=session_dir,
            codegen_kwargs={
                "runner_code": runner_code,
                "impl_strategy": impl_strategy,
                "codegen_task": codegen_task,
            },
        )

        result = GpuBenchMechanismResult(
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
