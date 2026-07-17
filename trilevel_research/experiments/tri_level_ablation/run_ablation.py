"""Tri-level ablation: Group C (L1+L1.5+L2) vs Group F (L1+L1.5+L2+L3).

Usage:
  cd Bilevel-Autoresearch
  python -m trilevel_research.experiments.tri_level_ablation.run_ablation --group C --repeats 3
  python -m trilevel_research.experiments.tri_level_ablation.run_ablation --group F --repeats 3
  python -m trilevel_research.experiments.tri_level_ablation.run_ablation --group all --repeats 3
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import traceback as tb
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

from experiments.ablations.paper_ablation.run_ablation import (  # noqa: E402
    FORBIDDEN_PARAMS,
    TIME_BUDGET,
    _build_run_dir,
    _get_train_py,
    _make_llm_client,
    _make_search_config,
    _print_group_summary,
    _save_report,
    _setup_run_logging,
    _teardown_run_logging,
    run_group_c,
)
from trilevel_research.config import MechanismResearchConfig  # noqa: E402
from trilevel_research.domains.train_opt.tri_level_controller import (
    TriLevelController,  # noqa: E402
)

ABLATION_DIR = Path(__file__).parent
RESULTS_DIR = ABLATION_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tri_level_ablation")


def run_group_f(
    repeat: int,
    iterations: int,
    outer_cycles: int,
    time_budget: int,
    provider: str,
    model: str,
    autoresearch_dir: Path,
    level3_interval: int = 2,
) -> dict:
    """Group F: L1 + L1.5 + L2 + L3 via TriLevelController."""
    run_dir = _build_run_dir("F", repeat)
    # Override results dir to tri_level results
    run_dir = RESULTS_DIR / f"F{repeat}"
    run_dir.mkdir(parents=True, exist_ok=True)

    fh = _setup_run_logging(run_dir)
    inner_per_cycle = max(1, iterations // outer_cycles)
    logger.info(
        f"=== Group F | Repeat {repeat} | {outer_cycles} outer cycles × "
        f"{inner_per_cycle} inner iters/cycle | L3 enabled ==="
    )

    try:
        train_py = _get_train_py(autoresearch_dir)
        client = _make_llm_client(provider, model)
        search_cfg = _make_search_config(inner_per_cycle, time_budget)

        mech_config = MechanismResearchConfig(
            level2_interval=2,
            level3_interval=level3_interval,
            enable_level3=True,
            enable_tabu=True,
            enable_adaptive_schedule=True,
        )

        controller = TriLevelController(
            run_dir=run_dir,
            train_py=train_py,
            work_dir=autoresearch_dir,
            llm_client=client,
            inner_budget=inner_per_cycle,
            outer_cycles=outer_cycles,
            time_budget=time_budget,
            enable_level3=True,
            mech_config=mech_config,
            search_config=search_cfg,
            forbidden_params=FORBIDDEN_PARAMS,
        )

        report = controller.run()
        report_dict = report.to_dict()
        report_dict.update({"group": "F", "repeat": repeat, "status": "ok"})
        _save_report(run_dir, report_dict)

        logger.info(
            f"[F{repeat}] Done. baseline={report_dict['baseline_bpb']:.6f}  "
            f"best={report_dict['best_val_bpb']:.6f}  "
            f"improvement={report_dict['improvement']:.6f}  "
            f"L2={report_dict['level2_rounds']} L3={report_dict['level3_rounds']}"
        )
        return report_dict

    except Exception as exc:
        err_text = tb.format_exc()
        logger.error(f"[F{repeat}] FAILED:\n{err_text}")
        error_report = {
            "group": "F", "repeat": repeat, "status": "error",
            "error": str(exc), "traceback": err_text,
        }
        _save_report(run_dir, error_report)
        return error_report

    finally:
        _teardown_run_logging(fh)


def run_group_c_tri(
    repeat: int,
    iterations: int,
    outer_cycles: int,
    time_budget: int,
    provider: str,
    model: str,
    autoresearch_dir: Path,
) -> dict:
    """Group C via TriLevelController (L3 disabled) for fair comparison with F."""
    run_dir = RESULTS_DIR / f"C{repeat}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = _setup_run_logging(run_dir)

    inner_per_cycle = max(1, iterations // outer_cycles)
    logger.info(
        f"=== Group C (tri-level driver) | Repeat {repeat} | "
        f"{outer_cycles} outer cycles × {inner_per_cycle} inner iters/cycle ==="
    )

    try:
        train_py = _get_train_py(autoresearch_dir)
        client = _make_llm_client(provider, model)
        search_cfg = _make_search_config(inner_per_cycle, time_budget)

        mech_config = MechanismResearchConfig(
            level2_interval=2,
            enable_level3=False,
            enable_tabu=True,
            enable_adaptive_schedule=True,
        )

        controller = TriLevelController(
            run_dir=run_dir,
            train_py=train_py,
            work_dir=autoresearch_dir,
            llm_client=client,
            inner_budget=inner_per_cycle,
            outer_cycles=outer_cycles,
            time_budget=time_budget,
            enable_level3=False,
            mech_config=mech_config,
            search_config=search_cfg,
            forbidden_params=FORBIDDEN_PARAMS,
        )

        report = controller.run()
        report_dict = report.to_dict()
        report_dict.update({"group": "C", "repeat": repeat, "status": "ok"})
        _save_report(run_dir, report_dict)
        return report_dict

    except Exception as exc:
        err_text = tb.format_exc()
        error_report = {
            "group": "C", "repeat": repeat, "status": "error",
            "error": str(exc), "traceback": err_text,
        }
        _save_report(run_dir, error_report)
        return error_report

    finally:
        _teardown_run_logging(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tri-level ablation: Group C (L1+L1.5+L2) vs F (+L3)",
    )
    parser.add_argument("--group", default="all", choices=["C", "F", "all"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--outer-cycles", type=int, default=6)
    parser.add_argument("--level3-interval", type=int, default=2)
    parser.add_argument("--time-budget", type=int, default=TIME_BUDGET)
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--autoresearch-dir",
        type=Path,
        default=Path.home() / "karpathy_autoresearch",
    )
    parser.add_argument(
        "--use-legacy-group-c",
        action="store_true",
        help="Use original paper_ablation run_group_c instead of TriLevelController",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = ["C", "F"] if args.group == "all" else [args.group]
    all_results: dict[str, list[dict]] = {}

    original_train_py = args.autoresearch_dir / "train.py"
    train_py_backup = args.autoresearch_dir / "train.py.original_backup"
    if original_train_py.exists() and not train_py_backup.exists():
        shutil.copy2(original_train_py, train_py_backup)

    for group in groups:
        logger.info(f"\n{'='*70}\nStarting Group {group} ({args.repeats} repeats)\n{'='*70}")
        group_results: list[dict] = []

        for repeat in range(1, args.repeats + 1):
            if train_py_backup.exists():
                shutil.copy2(train_py_backup, original_train_py)

            if group == "C":
                if args.use_legacy_group_c:
                    # Legacy path writes to paper_ablation/results/
                    result = run_group_c(
                        repeat=repeat,
                        iterations=args.iterations,
                        outer_cycles=args.outer_cycles,
                        time_budget=args.time_budget,
                        provider=args.provider,
                        model=args.model,
                        autoresearch_dir=args.autoresearch_dir,
                    )
                else:
                    result = run_group_c_tri(
                        repeat=repeat,
                        iterations=args.iterations,
                        outer_cycles=args.outer_cycles,
                        time_budget=args.time_budget,
                        provider=args.provider,
                        model=args.model,
                        autoresearch_dir=args.autoresearch_dir,
                    )
            else:
                result = run_group_f(
                    repeat=repeat,
                    iterations=args.iterations,
                    outer_cycles=args.outer_cycles,
                    time_budget=args.time_budget,
                    provider=args.provider,
                    model=args.model,
                    autoresearch_dir=args.autoresearch_dir,
                    level3_interval=args.level3_interval,
                )

            group_results.append(result)

        all_results[group] = group_results
        _print_group_summary(group, group_results)

    if len(groups) > 1:
        print(f"\n{'='*70}\nTri-Level Ablation Summary\n{'='*70}")
        for grp in groups:
            _print_group_summary(grp, all_results[grp])

    logger.info("Tri-level ablation complete.")


if __name__ == "__main__":
    main()
