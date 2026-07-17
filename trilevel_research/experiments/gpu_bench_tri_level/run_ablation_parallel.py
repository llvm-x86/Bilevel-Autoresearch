"""Parallel gpu_bench tri-level ablation — one (group, repeat) per worker.

LLM API latency is the bottleneck; GPU evals are ~0.14s. Workers run
independently with separate result dirs and skip already-completed runs.

Usage:
  python -m trilevel_research.experiments.gpu_bench_tri_level.run_ablation_parallel \\
    --group all --repeats 8 --workers 8 --skip-existing
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from trilevel_research.experiments.gpu_bench_tri_level.run_ablation import (  # noqa: E402
    RESULTS_DIR,
    _print_group_summary,
    _write_summary,
    run_group,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gpu_bench_parallel")


def _is_complete(group: str, repeat: int) -> bool:
    report_path = RESULTS_DIR / f"{group}{repeat}" / "report.json"
    if not report_path.is_file():
        return False
    try:
        data = json.loads(report_path.read_text())
        return data.get("status") == "ok"
    except (json.JSONDecodeError, OSError):
        return False


def _worker(job: dict) -> dict:
    """Process-pool entry: run one ablation repeat."""
    return run_group(**job)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel Group C vs F ablation")
    parser.add_argument("--group", default="all", choices=["C", "F", "all"])
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--inner-budget", type=int, default=10)
    parser.add_argument("--outer-cycles", type=int, default=6)
    parser.add_argument("--level2-interval", type=int, default=2)
    parser.add_argument("--level3-interval", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--bench-bin",
        type=Path,
        default=Path(os.environ.get("GPU_BENCH_BIN", "")),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip repeats with status=ok report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bench_bin = args.bench_bin
    if not bench_bin or not bench_bin.is_file():
        from trilevel_research.domains.gpu_bench_opt.runner import DEFAULT_BIN

        bench_bin = Path(os.environ.get("GPU_BENCH_BIN", str(DEFAULT_BIN)))
    if not bench_bin.is_file():
        logger.error("gpu_bench binary not found: %s", bench_bin)
        sys.exit(1)

    groups = ["C", "F"] if args.group == "all" else [args.group]
    jobs: list[dict] = []

    for group in groups:
        enable_l3 = group == "F"
        for repeat in range(1, args.repeats + 1):
            if args.skip_existing and _is_complete(group, repeat):
                logger.info("Skipping %s%d (already complete)", group, repeat)
                continue
            jobs.append(
                {
                    "group": group,
                    "repeat": repeat,
                    "inner_budget": args.inner_budget,
                    "outer_cycles": args.outer_cycles,
                    "enable_level3": enable_l3,
                    "provider": args.provider,
                    "model": args.model,
                    "bench_bin": bench_bin,
                    "level2_interval": args.level2_interval,
                    "level3_interval": args.level3_interval,
                    "timeout_s": args.timeout,
                }
            )

    if not jobs:
        logger.info("All repeats complete — writing summary only.")
    else:
        workers = min(args.workers, len(jobs))
        logger.info(
            "Launching %d jobs with %d workers (parallel LLM-bound runs)",
            len(jobs),
            workers,
        )
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, job): job for job in jobs}
            for fut in as_completed(futures):
                job = futures[fut]
                tag = f"{job['group']}{job['repeat']}"
                try:
                    result = fut.result()
                    logger.info(
                        "[%s] finished status=%s improvement=%s",
                        tag,
                        result.get("status"),
                        result.get("improvement"),
                    )
                except Exception as exc:
                    logger.error("[%s] worker exception: %s", tag, exc)

    all_results: dict[str, list[dict]] = {}
    for group in groups:
        group_results = []
        for repeat in range(1, args.repeats + 1):
            report_path = RESULTS_DIR / f"{group}{repeat}" / "report.json"
            if report_path.is_file():
                group_results.append(json.loads(report_path.read_text()))
            else:
                group_results.append(
                    {"group": group, "repeat": repeat, "status": "missing"}
                )
        all_results[group] = group_results
        _print_group_summary(group, group_results)

    summary_path = _write_summary(all_results)
    print(f"\nSummary written to: {summary_path}")
    logger.info("Parallel ablation complete.")


if __name__ == "__main__":
    main()
