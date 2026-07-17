"""Iterative bilevel loop: ouroboros improves tri-level until F beats C by target %.

Each iteration:
  1. Bilevel ouroboros patches adaptive_mechanism_schedule.py (LLM → bootstrap)
  2. Promote validated schedule to core
  3. Fresh Group C vs F gpu_bench ablation
  4. Stop when (F_mean - C_mean) / C_mean * 100 >= --target-margin-pct

Usage:
  cd Bilevel-Autoresearch
  export GPU_BENCH_BIN=... PYTHONPATH=$PWD HIP_VISIBLE_DEVICES=0
  python -m trilevel_research.experiments.bilevel_improves_trilevel.run_iterative \\
    --target-margin-pct 20 --max-iterations 8 --repeats 6 --workers 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
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

from trilevel_research.domains.gpu_bench_opt.runner import DEFAULT_BIN
from trilevel_research.experiments.bilevel_improves_trilevel.compare import compare_groups
from trilevel_research.experiments.bilevel_improves_trilevel.run import (
    EXPERIMENT_DIR,
    RESULTS_DIR,
    _promote_schedule,
    _run_ablation,
    _run_ouroboros_until_l2_valid,
)
from trilevel_research.experiments.gpu_bench_tri_level.run_ablation import (
    RESULTS_DIR as ABLATION_RESULTS_DIR,
)

LOCK_FILE = EXPERIMENT_DIR / ".run.lock"
ITERATIVE_SUMMARY = RESULTS_DIR / "iterative_summary.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bilevel_improves_iterative")


def _acquire_lock() -> None:
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            os.kill(pid, 0)
            raise RuntimeError(
                f"Another driver holds lock (PID {pid}). See MANAGER.md."
            )
        except (ProcessLookupError, ValueError):
            pass
    LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")


def _release_lock() -> None:
    if LOCK_FILE.exists():
        try:
            if int(LOCK_FILE.read_text().strip()) == os.getpid():
                LOCK_FILE.unlink()
        except (ValueError, OSError):
            pass


def _clear_ablation_results(repeats: int) -> None:
    for group in ("C", "F"):
        for repeat in range(1, repeats + 1):
            d = ABLATION_RESULTS_DIR / f"{group}{repeat}"
            if d.is_dir():
                shutil.rmtree(d)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative bilevel loop until tri-level beats bi-level by target %"
    )
    parser.add_argument("--target-margin-pct", type=float, default=20.0)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--inner-budget", type=int, default=5)
    parser.add_argument("--outer-cycles", type=int, default=6)
    parser.add_argument("--ouroboros-cycles", type=int, default=4)
    parser.add_argument("--ouroboros-attempts", type=int, default=2)
    parser.add_argument("--level2-interval", type=int, default=2)
    parser.add_argument("--level3-interval", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--bench-bin",
        type=Path,
        default=Path(os.environ.get("GPU_BENCH_BIN", str(DEFAULT_BIN))),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.bench_bin.is_file():
        logger.error("gpu_bench binary not found: %s", args.bench_bin)
        sys.exit(1)

    _acquire_lock()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    outcome: dict = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "target_margin_pct": args.target_margin_pct,
        "max_iterations": args.max_iterations,
        "iterations": [],
    }

    try:
        for iteration in range(1, args.max_iterations + 1):
            logger.info(
                "=== Iteration %d/%d (target +%.0f%%) ===",
                iteration,
                args.max_iterations,
                args.target_margin_pct,
            )
            iter_dir = RESULTS_DIR / f"iter_{iteration}"
            iter_dir.mkdir(parents=True, exist_ok=True)

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
            if ouro_result["status"] not in ("validated", "bootstrap"):
                logger.error("Iteration %d: ouroboros failed", iteration)
                outcome["iterations"].append(
                    {"iteration": iteration, "status": "ouroboros_failed", "ouroboros": ouro_result}
                )
                continue

            _promote_schedule(Path(ouro_result["schedule_path"]))
            _clear_ablation_results(args.repeats)

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
                skip_existing=False,
            )
            comparison = compare_groups(
                ablation_results,
                ouroboros_l2_apply_rate=ouro_result.get("l2_apply_rate", 0.0),
                margin_pct=args.target_margin_pct,
            )

            iter_record = {
                "iteration": iteration,
                "ouroboros_status": ouro_result["status"],
                "ouroboros_l2_apply_rate": ouro_result.get("l2_apply_rate"),
                "ablation": comparison,
            }
            outcome["iterations"].append(iter_record)
            (iter_dir / "comparison.json").write_text(
                json.dumps(iter_record, indent=2), encoding="utf-8"
            )

            margin_pct = comparison.get("margin_pct")
            logger.info(
                "Iteration %d: C=%.4f F=%.4f margin=%+.2f%% (need +%.0f%%)",
                iteration,
                comparison.get("c_mean_improvement") or 0,
                comparison.get("f_mean_improvement") or 0,
                margin_pct or 0,
                args.target_margin_pct,
            )

            if comparison.get("success"):
                outcome["success"] = True
                outcome["winning_iteration"] = iteration
                outcome["final_ablation"] = comparison
                outcome["finished_at"] = datetime.now(timezone.utc).isoformat()
                ITERATIVE_SUMMARY.write_text(
                    json.dumps(outcome, indent=2), encoding="utf-8"
                )
                print(
                    f"\nVERDICT: SUCCESS — F beat C by {margin_pct:.1f}% "
                    f"(target {args.target_margin_pct}%) on iteration {iteration}"
                )
                sys.exit(0)

        outcome["success"] = False
        outcome["finished_at"] = datetime.now(timezone.utc).isoformat()
        if outcome["iterations"]:
            outcome["final_ablation"] = outcome["iterations"][-1].get("ablation")
        ITERATIVE_SUMMARY.write_text(json.dumps(outcome, indent=2), encoding="utf-8")
        last = outcome["iterations"][-1]["ablation"] if outcome["iterations"] else {}
        print(
            f"\nVERDICT: FAIL — best margin {last.get('margin_pct', 0):.1f}% "
            f"< target {args.target_margin_pct}% after {args.max_iterations} iterations"
        )
        sys.exit(2)

    except Exception as exc:
        logger.error("Iterative driver failed: %s\n%s", exc, tb.format_exc())
        sys.exit(1)
    finally:
        _release_lock()


if __name__ == "__main__":
    main()
