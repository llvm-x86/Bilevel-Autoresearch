"""Tri-level controller: orchestrates L1 + L1.5 + L2 (+ optional L3).

Extracted from paper ablation Group C/F logic into a reusable driver.
"""
from __future__ import annotations

import importlib.util
import logging
import shutil
import sys
import traceback as tb
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from core.llm_client import LLMClient
from domains.train_opt.config import SearchConfig
from domains.train_opt.outer import TrainOuterLoop
from domains.train_opt.runner import TrainRunner
from trilevel_research.config import MechanismResearchConfig
from trilevel_research.core.adaptive_mechanism_schedule import AdaptiveMechanismSchedule
from trilevel_research.core.mechanism_session_trace import (
    MechanismSessionRecord,
    MechanismSessionTraceBuilder,
)
from trilevel_research.core.mechanism_tabu_registry import MechanismTabuRegistry
from trilevel_research.core.mechanism_validation_harness import (
    MechanismValidationHarness,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class TriLevelReport:
    group: str
    baseline_bpb: float | None = None
    best_val_bpb: float = float("inf")
    best_iteration: int = 0
    improvement: float = 0.0
    total_iterations: int = 0
    outer_cycles: int = 0
    level2_rounds: int = 0
    level3_rounds: int = 0
    level2_sessions: list[dict] = field(default_factory=list)
    level3_sessions: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    outer_trace: list[dict] = field(default_factory=list)
    tabu_stats: dict = field(default_factory=dict)
    schedule_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "baseline_bpb": self.baseline_bpb,
            "best_val_bpb": self.best_val_bpb,
            "best_iteration": self.best_iteration,
            "improvement": self.improvement,
            "total_iterations": self.total_iterations,
            "outer_cycles": self.outer_cycles,
            "level2_rounds": self.level2_rounds,
            "level3_rounds": self.level3_rounds,
            "level2_sessions": self.level2_sessions,
            "level3_sessions": self.level3_sessions,
            "trace": self.trace,
            "outer_trace": self.outer_trace,
            "tabu_stats": self.tabu_stats,
            "schedule_stats": self.schedule_stats,
        }


class TriLevelController:
    """Orchestrate bilevel (C) or tri-level (F) training experiments."""

    def __init__(
        self,
        run_dir: Path,
        train_py: Path,
        work_dir: Path,
        llm_client: LLMClient,
        inner_budget: int = 5,
        outer_cycles: int = 6,
        time_budget: int = 300,
        enable_level3: bool = False,
        mech_config: MechanismResearchConfig | None = None,
        canonical_runner_py: Path | None = None,
        canonical_mech_research_py: Path | None = None,
        search_config: SearchConfig | None = None,
        forbidden_params: set[str] | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.train_py = Path(train_py)
        self.work_dir = Path(work_dir)
        self.client = llm_client
        self.inner_budget = inner_budget
        self.outer_cycles = outer_cycles
        self.time_budget = time_budget
        self.enable_level3 = enable_level3
        self.mech_config = mech_config or MechanismResearchConfig(
            enable_level3=enable_level3
        )
        self.canonical_runner_py = canonical_runner_py or (
            REPO_ROOT / "domains" / "train_opt" / "runner.py"
        )
        self.canonical_mech_research_py = canonical_mech_research_py or (
            REPO_ROOT / "domains" / "train_opt" / "mechanism_research.py"
        )
        self.search_config_template = search_config
        self.forbidden_params = forbidden_params or set()

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_runner_py = self.run_dir / "runner.py"
        self.run_mech_py = self.run_dir / "mechanism_research.py"
        self.mech_sessions_dir = self.run_dir / "mechanism_sessions"
        self.meta_sessions_dir = self.run_dir / "meta_mechanism_sessions"

        self.tabu = MechanismTabuRegistry.load(
            self.run_dir / "mechanism_tabu.json", run_token=uuid.uuid4().hex
        )
        self.schedule = AdaptiveMechanismSchedule.load(self.run_dir / "schedule_config.json")
        self.harness = MechanismValidationHarness(domain="train_opt", project_root=REPO_ROOT)

    def run(self) -> TriLevelReport:
        group = "F" if self.enable_level3 else "C"
        logger.info(f"=== TriLevelController group={group} outer_cycles={self.outer_cycles} ===")

        shutil.copy2(self.canonical_runner_py, self.run_runner_py)
        shutil.copy2(self.canonical_mech_research_py, self.run_mech_py)
        self.mech_sessions_dir.mkdir(parents=True, exist_ok=True)

        report = TriLevelReport(group=group, outer_cycles=self.outer_cycles)
        l2_session_records: list[MechanismSessionRecord] = []
        completed_cycles = 0
        l2_round = 0
        l3_round = 0

        while completed_cycles < self.outer_cycles:
            decision = self.schedule.decide(
                inner_trace=report.trace,
                l2_sessions=l2_session_records,
                completed_outer_cycles=completed_cycles,
                config=self.mech_config,
            )
            batch_size = min(
                decision.batch_size or self.mech_config.level2_interval,
                self.outer_cycles - completed_cycles,
            )
            batch_start = completed_cycles + 1
            batch_end = completed_cycles + batch_size

            logger.info(
                f"Batch cycles {batch_start}–{batch_end} (schedule: {decision.reason})"
            )

            PatchedTrainRunner = self._load_runner_class(self.run_runner_py)
            config = self._make_search_config()
            batch_artifacts = self.run_dir / "artifacts" / f"batch_{batch_start}_{batch_end}"

            runner = PatchedTrainRunner(
                train_py=self.train_py,
                work_dir=self.work_dir,
                llm_client=self.client,
                search_config=config,
                artifacts_dir=batch_artifacts,
                simple_mode=True,
            )

            outer = TrainOuterLoop(
                runner=runner,
                llm_client=self.client,
                max_outer_cycles=batch_size,
                artifacts_dir=batch_artifacts / "outer",
            )
            batch_report = outer.run()

            if report.baseline_bpb is None:
                report.baseline_bpb = batch_report.get("baseline_bpb")

            batch_best = batch_report.get("best_val_bpb", float("inf"))
            if batch_best < report.best_val_bpb:
                report.best_val_bpb = batch_best
                report.best_iteration = batch_report.get("best_iteration", 0)

            report.trace.extend(batch_report.get("trace", []))
            report.outer_trace.extend(batch_report.get("outer_trace", []))
            completed_cycles += batch_size

            shutil.copy2(
                self.run_runner_py,
                self.run_dir / f"runner_after_cycles_{batch_start}_{batch_end}.py",
            )

            if completed_cycles >= self.outer_cycles:
                break

            if not decision.fire_level2:
                logger.info(f"Skipping L2: {decision.reason}")
                continue

            l2_round += 1
            l2_record = self._run_level2(
                runner=runner,
                l2_round=l2_round,
                batch_size=batch_size,
            )
            l2_session_records.append(l2_record)
            report.level2_sessions.append(l2_record.to_dict())
            report.level2_rounds = l2_round
            self.tabu.save(self.run_dir / "mechanism_tabu.json")

            fire_l3 = self.enable_level3 and (
                decision.fire_level3
                or l2_round % self.mech_config.level3_interval == 0
            )
            if fire_l3:
                l3_round += 1
                l3_info = self._run_level3(l3_round=l3_round, inner_trace=report.trace)
                report.level3_sessions.append(l3_info)
                report.level3_rounds = l3_round
                shutil.copy2(
                    self.run_mech_py,
                    self.run_dir / f"mechanism_research_after_l3_round_{l3_round}.py",
                )

        report.total_iterations = len(report.trace)
        if report.baseline_bpb is not None:
            report.improvement = report.baseline_bpb - report.best_val_bpb
        report.tabu_stats = self.tabu.stats()
        report.schedule_stats = self.schedule.stats()

        shutil.copy2(self.run_runner_py, self.run_dir / "runner_final.py")
        return report

    def _run_level2(
        self,
        runner: TrainRunner,
        l2_round: int,
        batch_size: int,
    ) -> MechanismSessionRecord:
        ResearcherClass = self._load_mechanism_researcher_class()
        tabu = self.tabu if self.mech_config.enable_tabu else None
        researcher = ResearcherClass(
            model=self.client._model,
            api_key=self.client._api_key,
            provider=getattr(self.client, "_provider", "deepseek"),
            max_code_retries=self.mech_config.max_code_retries,
            tabu_registry=tabu,
        )

        session_dir = self.mech_sessions_dir / f"round_{l2_round}"
        session_dir.mkdir(parents=True, exist_ok=True)
        trace_text = runner.trace.summary(last_n=self.inner_budget * batch_size + 5)
        runner_code = self.run_runner_py.read_text(encoding="utf-8")

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
                runner_code=runner_code,
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
                    logger.warning(f"L2 round {l2_round} blocked by tabu: {reason}")
                    record.blocked_by_tabu = True
                    record.error = reason
                    return record

            applied = researcher.apply(self.run_runner_py, result)
            valid = False
            if applied:
                valid = researcher.validate(self.run_runner_py)
                if not valid:
                    self._restore_backup(self.run_runner_py, result.session_id)
                    applied = False
                    if self.mech_config.enable_tabu:
                        self.tabu.record_failure(
                            result.mechanism_name,
                            result.target,
                            l2_round,
                            reason="import_fail",
                        )
                    record.error = result.validation_error or "import_fail"
                else:
                    if self.mech_config.enable_tabu:
                        self.tabu.record_success(
                            result.mechanism_name, result.target, l2_round
                        )

            record.applied = applied
            record.validated = valid
            logger.info(
                f"L2 round {l2_round}: {result.mechanism_name} applied={applied}"
            )

        except Exception as exc:
            err_text = tb.format_exc()
            logger.error(f"L2 round {l2_round} failed:\n{err_text}")
            record.error = str(exc)

        return record

    def _run_level3(self, l3_round: int, inner_trace: list[dict]) -> dict:
        from .meta_mechanism_research import TrainMetaMechanismResearcher

        l2_records = MechanismSessionTraceBuilder.build(
            self.mech_sessions_dir,
            inner_trace=inner_trace,
        )
        l2_trace_text = MechanismSessionTraceBuilder.to_text(l2_records, inner_trace)
        mech_code = self.run_mech_py.read_text(encoding="utf-8")

        meta = TrainMetaMechanismResearcher(
            model=self.client._model,
            api_key=self.client._api_key,
            provider=getattr(self.client, "_provider", "deepseek"),
        )
        session_dir = self.meta_sessions_dir / f"round_{l3_round}"
        session_dir.mkdir(parents=True, exist_ok=True)

        info: dict = {"round": l3_round, "applied": False}

        try:
            result = meta.research(
                l2_trace_summary=l2_trace_text,
                mechanism_research_code=mech_code,
                session_dir=session_dir,
                bottleneck="",
            )
            info.update({
                "session_id": result.session_id,
                "patch_name": result.patch_name,
                "strategy": result.implementation_strategy,
                "target": result.target,
            })

            if result.schedule_patch:
                self.mech_config.apply_patch(result.schedule_patch)
                self.schedule.level2_interval = self.mech_config.level2_interval
                self.schedule.level3_interval = self.mech_config.level3_interval
                self.schedule.save(self.run_dir / "schedule_config.json")

            applied = meta.apply(self.run_mech_py, result)
            if applied and self.mech_config.validation_strict:
                valid = meta.validate(self.run_mech_py)
                if not valid:
                    self._restore_backup(self.run_mech_py, result.session_id)
                    applied = False
                    info["error"] = result.validation_error or "import_fail"

            info["applied"] = applied
            logger.info(f"L3 round {l3_round}: patch={result.patch_name} applied={applied}")

        except Exception as exc:
            err_text = tb.format_exc()
            logger.error(f"L3 round {l3_round} failed:\n{err_text}")
            info["error"] = str(exc)

        return info

    def _make_search_config(self) -> SearchConfig:
        if self.search_config_template:
            cfg = SearchConfig(
                inner_budget=self.search_config_template.inner_budget,
                time_budget=self.search_config_template.time_budget,
                strategy=self.search_config_template.strategy,
                guidance=self.search_config_template.guidance,
                frozen_params=list(self.search_config_template.frozen_params),
                editable_params=list(self.search_config_template.editable_params),
            )
        else:
            cfg = SearchConfig(inner_budget=self.inner_budget, time_budget=self.time_budget)
        if self.forbidden_params:
            cfg.editable_params = [
                p for p in cfg.editable_params if p not in self.forbidden_params
            ]
        return cfg

    def _load_runner_class(self, runner_py: Path):
        module = self._load_module_from_path(runner_py, prefix="tri_runner")
        return module.TrainRunner

    def _load_mechanism_researcher_class(self):
        from trilevel_research.domains.train_opt.l2_mechanism_research import (
            TriLevelTrainMechanismResearcher,
        )

        module = self._load_module_from_path(self.run_mech_py, prefix="tri_mech")
        loaded = getattr(module, "TrainMechanismResearcher", None)
        if loaded is not None and loaded.__name__ == "TrainMechanismResearcher":
            return TriLevelTrainMechanismResearcher
        return loaded or TriLevelTrainMechanismResearcher

    def _load_module_from_path(self, py_path: Path, prefix: str):
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        import core.llm_client  # noqa: F401
        import domains.train_opt.config  # noqa: F401

        code = py_path.read_text(encoding="utf-8")
        if "from .config import" in code:
            code = code.replace(
                "from .config import", "from domains.train_opt.config import"
            )
            py_path.write_text(code, encoding="utf-8")

        spec = importlib.util.spec_from_file_location(
            f"_{prefix}_{py_path.stat().st_mtime_ns}", py_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _restore_backup(target: Path, session_id: str) -> None:
        backup = target.with_suffix(f".py.bak_{session_id}")
        if backup.exists():
            shutil.copy2(backup, target)
            logger.info(f"Restored {target} from backup")
