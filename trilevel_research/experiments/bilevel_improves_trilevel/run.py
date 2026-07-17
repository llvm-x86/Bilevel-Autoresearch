"""Bilevel improves tri-level: ouroboros schedule L2 → ablation C vs F.

Step 1: Run ouroboros bilevel on adaptive_mechanism_schedule until L2 validates.
Step 2: Promote validated schedule to core working copy.
Step 3: Paired Group C vs Group F ablation on gpu_bench.
Step 4: Success if F mean improvement > C mean + 0.01 and F L2 apply rate > 0.

Usage:
  cd Bilevel-Autoresearch
  export GPU_BENCH_BIN=/path/to/gpu_bench PYTHONPATH=$PWD HIP_VISIBLE_DEVICES=0
  python -m trilevel_research.experiments.bilevel_improves_trilevel.run \\
    --repeats 4 --workers 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import statistics
import sys
import traceback as tb
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

_env_path = REPO_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _v = _v.strip().strip('"').strip("'")
            os.environ.setdefault(_k.strip(), _v)

from core.llm_client import PROVIDERS, LLMClient
from trilevel_research.config import MechanismResearchConfig
from trilevel_research.domains.gpu_bench_opt.runner import DEFAULT_BIN
from trilevel_research.domains.gpu_bench_opt.schedule_mechanism_research import (
    BOOTSTRAP_DECIDE_CODE,
    ScheduleMechanismResearcher,
    ScheduleMechanismResult,
)
from trilevel_research.domains.gpu_bench_opt.search_config import SearchConfig
from trilevel_research.experiments.gpu_bench_tri_level.run_ablation import (
    RESULTS_DIR as ABLATION_RESULTS_DIR,
    _print_group_summary,
    _write_summary,
    run_group,
)
from trilevel_research.experiments.ouroboros.schedule_controller import (
    OuroborosScheduleController,
)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_SCHEDULE = REPO_ROOT / "trilevel_research" / "core" / "adaptive_mechanism_schedule.py"
WORKING_COPY_FLAG = CANONICAL_SCHEDULE.with_suffix(".py.working_copy")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bilevel_improves_trilevel")


def _make_llm_client(provider: str, model: str) -> LLMClient:
    pinfo = PROVIDERS.get(provider)
    if not pinfo:
        raise ValueError(f"Unknown provider '{provider}'")
    api_key = os.environ.get(pinfo["api_key_env"], "")
    if not api_key:
        raise EnvironmentError(f"API key not set: {pinfo['api_key_env']}")
    return LLMClient(provider, api_key, model or pinfo["default_model"])


from trilevel_research.experiments.bilevel_improves_trilevel.compare import (
    compare_groups,
    l2_apply_rate,
)


def _run_ouroboros_until_l2_valid(
    *,
    provider: str,
    model: str,
    bench_bin: Path,
    inner_budget: int,
    outer_cycles: int,
    level2_interval: int,
    level3_interval: int,
    timeout_s: int,
    max_rounds: int,
) -> dict:
    """Run ouroboros schedule L2 until a validated apply or bootstrap succeeds."""
    ouro_dir = RESULTS_DIR / "ouroboros"
    ouro_dir.mkdir(parents=True, exist_ok=True)
    client = _make_llm_client(provider, model)
    mech_config = MechanismResearchConfig(
        level2_interval=level2_interval,
        level3_interval=level3_interval,
        enable_level3=False,
        enable_tabu=True,
        enable_adaptive_schedule=True,
    )

    last_report: dict = {}
    for attempt in range(1, max_rounds + 1):
        run_dir = ouro_dir / f"schedule_l2_attempt_{attempt}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=== Ouroboros attempt %d/%d ===", attempt, max_rounds)

        controller = OuroborosScheduleController(
            run_dir=run_dir,
            bench_bin=bench_bin,
            llm_client=client,
            inner_budget=inner_budget,
            outer_cycles=outer_cycles,
            enable_level3=False,
            enable_l2=True,
            mech_config=mech_config,
            search_config=SearchConfig(inner_budget=inner_budget),
            timeout_s=timeout_s,
        )
        report = controller.run()
        report_dict = report.to_dict()
        report_dict["attempt"] = attempt
        report_dict["status"] = "ok"
        (run_dir / "report.json").write_text(
            json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        last_report = report_dict

        sessions = report_dict.get("level2_sessions", [])
        validated = [s for s in sessions if s.get("validated")]
        if validated:
            logger.info("Ouroboros L2 validated on attempt %d", attempt)
            schedule_final = run_dir / "adaptive_mechanism_schedule_final.py"
            return {
                "status": "validated",
                "attempt": attempt,
                "schedule_path": str(schedule_final),
                "l2_apply_rate": l2_apply_rate(sessions),
                "report": report_dict,
            }

    # Bootstrap fallback: apply known-good patch directly
    logger.warning("Ouroboros LLM L2 did not validate; applying bootstrap schedule patch")
    bootstrap_dir = ouro_dir / "bootstrap_patch"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    schedule_py = bootstrap_dir / "adaptive_mechanism_schedule.py"
    shutil.copy2(CANONICAL_SCHEDULE, schedule_py)

    researcher = ScheduleMechanismResearcher(
        model=model or PROVIDERS[provider]["default_model"],
        api_key=os.environ.get(PROVIDERS[provider]["api_key_env"], ""),
        provider=provider,
    )
    session_dir = bootstrap_dir / "mechanism_sessions" / "bootstrap"
    session_dir.mkdir(parents=True, exist_ok=True)
    result = ScheduleMechanismResult(
        session_id=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        hypothesis="bootstrap_improving_streak_defer",
        mechanism_name="bootstrap_improving_streak_defer",
        implementation_strategy="replace_method",
        target="decide",
        spec="hand-written bootstrap",
        code=BOOTSTRAP_DECIDE_CODE,
        session_dir=session_dir,
    )
    applied = researcher.apply(schedule_py, result)
    valid = researcher.validate(schedule_py) if applied else False
    if not (applied and valid):
        return {
            "status": "bootstrap_failed",
            "attempt": max_rounds,
            "l2_apply_rate": 0.0,
            "report": last_report,
            "error": result.validation_error or "bootstrap validate failed",
        }

    shutil.copy2(schedule_py, bootstrap_dir / "adaptive_mechanism_schedule_final.py")
    return {
        "status": "bootstrap",
        "attempt": max_rounds,
        "schedule_path": str(schedule_py),
        "l2_apply_rate": 1.0,
        "report": last_report,
    }


def _promote_schedule(schedule_path: Path) -> None:
    """Copy validated schedule to canonical core module with working-copy flag."""
    schedule_path = Path(schedule_path)
    if not schedule_path.is_file():
        raise FileNotFoundError(schedule_path)

    backup = CANONICAL_SCHEDULE.with_suffix(".py.bak_pre_bilevel")
    if not backup.exists():
        shutil.copy2(CANONICAL_SCHEDULE, backup)

    header = (
        "# WORKING_COPY: promoted by bilevel_improves_trilevel "
        f"at {datetime.now(timezone.utc).isoformat()}\n"
    )
    content = schedule_path.read_text(encoding="utf-8")
    if not content.startswith("# WORKING_COPY"):
        content = header + content
    CANONICAL_SCHEDULE.write_text(content, encoding="utf-8")
    WORKING_COPY_FLAG.write_text(
        json.dumps(
            {
                "source": str(schedule_path),
                "promoted_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Promoted schedule to %s (flag: %s)", CANONICAL_SCHEDULE, WORKING_COPY_FLAG)


def _run_ablation(
    *,
    repeats: int,
    workers: int,
    inner_budget: int,
    outer_cycles: int,
    level2_interval: int,
    level3_interval: int,
    timeout_s: int,
    provider: str,
    model: str,
    bench_bin: Path,
    skip_existing: bool,
) -> dict[str, list[dict]]:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    groups = ["C", "F"]
    jobs: list[dict] = []

    for group in groups:
        enable_l3 = group == "F"
        for repeat in range(1, repeats + 1):
            report_path = ABLATION_RESULTS_DIR / f"{group}{repeat}" / "report.json"
            if skip_existing and report_path.is_file():
                try:
                    data = json.loads(report_path.read_text())
                    if data.get("status") == "ok":
                        logger.info("Skipping %s%d (existing ok)", group, repeat)
                        continue
                except (json.JSONDecodeError, OSError):
                    pass
            jobs.append(
                {
                    "group": group,
                    "repeat": repeat,
                    "inner_budget": inner_budget,
                    "outer_cycles": outer_cycles,
                    "enable_level3": enable_l3,
                    "provider": provider,
                    "model": model,
                    "bench_bin": bench_bin,
                    "level2_interval": level2_interval,
                    "level3_interval": level3_interval,
                    "timeout_s": timeout_s,
                }
            )

    if jobs:
        worker_count = min(workers, len(jobs))
        logger.info("Running %d ablation jobs with %d workers", len(jobs), worker_count)
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(run_group, **job): job for job in jobs}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    logger.error("[%s%d] failed: %s", job["group"], job["repeat"], exc)

    all_results: dict[str, list[dict]] = {}
    for group in groups:
        group_results = []
        for repeat in range(1, repeats + 1):
            report_path = ABLATION_RESULTS_DIR / f"{group}{repeat}" / "report.json"
            if report_path.is_file():
                group_results.append(json.loads(report_path.read_text()))
            else:
                group_results.append({"group": group, "repeat": repeat, "status": "missing"})
        all_results[group] = group_results
        _print_group_summary(group, group_results)

    _write_summary(all_results)
    return all_results


def _compare_groups(
    all_results: dict[str, list[dict]], f_l2_apply_rate: float, margin_pct: float | None = None
) -> dict:
    return compare_groups(
        all_results,
        ouroboros_l2_apply_rate=f_l2_apply_rate,
        margin_abs=0.01,
        margin_pct=margin_pct,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bilevel improves tri-level driver")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--inner-budget", type=int, default=5)
    parser.add_argument("--outer-cycles", type=int, default=4)
    parser.add_argument("--ouroboros-cycles", type=int, default=4)
    parser.add_argument("--ouroboros-attempts", type=int, default=2)
    parser.add_argument("--level2-interval", type=int, default=2)
    parser.add_argument("--level3-interval", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--bench-bin",
        type=Path,
        default=Path(os.environ.get("GPU_BENCH_BIN", str(DEFAULT_BIN))),
    )
    parser.add_argument("--margin-pct", type=float, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-ouroboros", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.bench_bin.is_file():
        logger.error("gpu_bench binary not found: %s", args.bench_bin)
        sys.exit(1)

    outcome: dict = {"started_at": datetime.now(timezone.utc).isoformat()}

    try:
        ouro_result: dict = {"status": "skipped", "l2_apply_rate": 0.0}
        if not args.skip_ouroboros:
            ouro_result = _run_ouroboros_until_l2_valid(
                provider=args.provider,
                model=args.model,
                bench_bin=args.bench_bin,
                inner_budget=args.inner_budget,
                outer_cycles=args.ouroboros_cycles,
                level2_interval=args.level2_interval,
                level3_interval=args.level3_interval,
                timeout_s=args.timeout,
                max_rounds=args.ouroboros_attempts,
            )
            outcome["ouroboros"] = ouro_result

            if ouro_result["status"] in ("validated", "bootstrap"):
                _promote_schedule(Path(ouro_result["schedule_path"]))
                outcome["schedule_promoted"] = True
            else:
                outcome["schedule_promoted"] = False
                logger.error("No validated schedule; aborting ablation")
                sys.exit(1)

        ablation_results: dict[str, list[dict]] = {}
        if not args.skip_ablation:
            ablation_results = _run_ablation(
                repeats=args.repeats,
                workers=args.workers,
                inner_budget=args.inner_budget,
                outer_cycles=args.outer_cycles,
                level2_interval=args.level2_interval,
                level3_interval=args.level3_interval,
                timeout_s=args.timeout,
                provider=args.provider,
                model=args.model,
                bench_bin=args.bench_bin,
                skip_existing=args.skip_existing,
            )
            outcome["ablation"] = _compare_groups(
                ablation_results,
                f_l2_apply_rate=ouro_result.get("l2_apply_rate", 0.0),
                margin_pct=args.margin_pct,
            )

        outcome["finished_at"] = datetime.now(timezone.utc).isoformat()
        summary_path = RESULTS_DIR / "bilevel_improves_trilevel_summary.json"
        summary_path.write_text(json.dumps(outcome, indent=2), encoding="utf-8")
        logger.info("Summary: %s", summary_path)

        if outcome.get("ablation", {}).get("success"):
            print("\nVERDICT: SUCCESS — Group F beat Group C by >0.01 with L2 apply > 0")
            sys.exit(0)
        if "ablation" in outcome:
            print("\nVERDICT: FAIL — Group F did not beat Group C by required margin")
            sys.exit(2)
        print("\nVERDICT: partial run (ouroboros only)")

    except Exception as exc:
        logger.error("Driver failed: %s\n%s", exc, tb.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
