"""GpuBenchTriLevelController with L2 patching adaptive_mechanism_schedule.py (ouroboros)."""
from __future__ import annotations

import importlib.util
import logging
import shutil
import sys
import traceback as tb
from pathlib import Path

from trilevel_research.core.adaptive_mechanism_schedule import AdaptiveMechanismSchedule
from trilevel_research.core.mechanism_session_trace import MechanismSessionRecord
from trilevel_research.domains.gpu_bench_opt.tri_level_controller import (
    REPO_ROOT,
    GpuBenchTriLevelController,
)

logger = logging.getLogger(__name__)


class OuroborosScheduleController(GpuBenchTriLevelController):
    """Group C driver: L1 tunes gpu_bench; L2 patches AdaptiveMechanismSchedule."""

    def __init__(
        self,
        *args,
        enable_l2: bool = True,
        canonical_schedule_py: Path | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.enable_l2 = enable_l2
        self.canonical_schedule_py = canonical_schedule_py or (
            REPO_ROOT / "trilevel_research" / "core" / "adaptive_mechanism_schedule.py"
        )
        self.run_schedule_py = self.run_dir / "adaptive_mechanism_schedule.py"

    def run(self):
        shutil.copy2(self.canonical_schedule_py, self.run_schedule_py)
        report = super().run()
        shutil.copy2(self.run_schedule_py, self.run_dir / "adaptive_mechanism_schedule_final.py")
        return report

    def _run_level2(
        self,
        runner,
        l2_round: int,
        batch_size: int,
    ) -> MechanismSessionRecord:
        if not self.enable_l2:
            return MechanismSessionRecord(
                round=l2_round,
                session_id=f"round_{l2_round}",
                mechanism_name="disabled",
                implementation_strategy="none",
                target="none",
                session_dir=self.mech_sessions_dir / f"round_{l2_round}",
                error="l2_disabled",
            )

        MechResearcher = self._load_schedule_researcher_class()
        researcher = MechResearcher(
            model=self.client._model,
            api_key=self.client._api_key,
            provider=getattr(self.client, "_provider", "deepseek"),
            max_code_retries=self.mech_config.max_code_retries,
        )
        if self.mech_config.enable_tabu:
            researcher.tabu_registry = self.tabu

        session_dir = self.mech_sessions_dir / f"round_{l2_round}"
        session_dir.mkdir(parents=True, exist_ok=True)
        trace_text = runner.trace.summary(last_n=self.inner_budget * batch_size + 5)
        schedule_code = self.run_schedule_py.read_text(encoding="utf-8")

        record = MechanismSessionRecord(
            round=l2_round,
            session_id=f"round_{l2_round}",
            mechanism_name="unknown",
            implementation_strategy="unknown",
            target="unknown",
            session_dir=session_dir,
        )

        try:
            result = researcher.research(
                trace_summary=trace_text,
                schedule_code=schedule_code,
                session_dir=session_dir,
                bottleneck="",
            )
            record.session_id = result.session_id
            record.mechanism_name = result.mechanism_name
            record.implementation_strategy = result.implementation_strategy
            record.target = result.target
            record.hypothesis = result.hypothesis
            record.code_retries = result.code_retries

            if self.mech_config.enable_tabu:
                is_tabu, reason = self.tabu.is_tabu(
                    result.mechanism_name, result.target, l2_round
                )
                if is_tabu:
                    record.blocked_by_tabu = True
                    record.error = reason
                    return record

            applied = researcher.apply(self.run_schedule_py, result)
            valid = False
            if applied:
                valid = researcher.validate(self.run_schedule_py, result)
                if not valid:
                    self._restore_backup(self.run_schedule_py, result.session_id)
                    applied = False
                    if self.mech_config.enable_tabu:
                        self.tabu.record_failure(
                            result.mechanism_name,
                            result.target,
                            l2_round,
                            reason=self._tabu_failure_reason(result.validation_error),
                        )
                    record.error = result.validation_error or "validate_fail"
                else:
                    self.schedule = self._reload_schedule(self.run_schedule_py)
                    self.schedule.save(self.run_dir / "schedule_config.json")
                    if self.mech_config.enable_tabu:
                        self.tabu.record_success(
                            result.mechanism_name, result.target, l2_round
                        )

            record.applied = applied
            record.validated = valid
            researcher.update_session_summary(result, session_dir)
            logger.info(
                "L2 round %d (schedule): %s applied=%s validated=%s",
                l2_round,
                result.mechanism_name,
                applied,
                valid,
            )

            shutil.copy2(
                self.run_schedule_py,
                self.run_dir / f"schedule_after_l2_round_{l2_round}.py",
            )

        except Exception as exc:
            err_text = tb.format_exc()
            logger.error("L2 round %d (schedule) failed:\n%s", l2_round, err_text)
            record.error = str(exc)

        return record

    def _load_schedule_researcher_class(self):
        from trilevel_research.domains.gpu_bench_opt.schedule_mechanism_research import (
            ScheduleMechanismResearcher,
        )

        return ScheduleMechanismResearcher

    def _reload_schedule(self, schedule_py: Path) -> AdaptiveMechanismSchedule:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        spec = importlib.util.spec_from_file_location(
            f"_ouroboros_sched_{schedule_py.stat().st_mtime_ns}",
            schedule_py,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.AdaptiveMechanismSchedule.load(self.run_dir / "schedule_config.json")
