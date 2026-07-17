"""GPU bench tri-level ablation: Group C (L1+L1.5+L2) vs Group F (+L3).

Usage:
  cd Bilevel-Autoresearch
  python -m trilevel_research.experiments.gpu_bench_tri_level.run_ablation --group all --repeats 2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
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

from core.llm_client import PROVIDERS, LLMClient
from trilevel_research.config import MechanismResearchConfig
from trilevel_research.domains.gpu_bench_opt.runner import DEFAULT_BIN
from trilevel_research.domains.gpu_bench_opt.search_config import SearchConfig
from trilevel_research.domains.gpu_bench_opt.tri_level_controller import (
    GpuBenchTriLevelController,
)

ABLATION_DIR = Path(__file__).parent
RESULTS_DIR = ABLATION_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gpu_bench_tri_level_ablation")


def _make_llm_client(provider: str, model: str) -> LLMClient:
    pinfo = PROVIDERS.get(provider)
    if not pinfo:
        raise ValueError(f"Unknown provider '{provider}'")
    api_key = os.environ.get(pinfo["api_key_env"], "")
    if not api_key:
        raise EnvironmentError(f"API key not set: {pinfo['api_key_env']}")
    return LLMClient(provider, api_key, model or pinfo["default_model"])


def _setup_run_logging(run_dir: Path):
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    logging.getLogger().addHandler(fh)
    return fh


def _teardown_run_logging(fh) -> None:
    logging.getLogger().removeHandler(fh)
    fh.close()


def _save_report(run_dir: Path, report: dict) -> None:
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _print_group_summary(group: str, results: list[dict]) -> None:
    ok = [r for r in results if r.get("status") == "ok"]
    print(f"\n{'='*60}\nGroup {group} Summary ({len(ok)}/{len(results)} ok)\n{'='*60}")
    if not ok:
        print("  No successful runs.")
        return
    improvements = [r["improvement"] for r in ok if r.get("improvement") is not None]
    bests = [r["best_val_bpb"] for r in ok if r.get("best_val_bpb") != float("inf")]
    if improvements:
        print(f"  Improvement: mean={statistics.mean(improvements):.6f} "
              f"stdev={statistics.pstdev(improvements) if len(improvements) > 1 else 0:.6f}")
    if bests:
        print(f"  Best val_bpb: mean={statistics.mean(bests):.6f}")
    for r in results:
        tag = r.get("status", "?")
        rep = r.get("repeat", "?")
        imp = r.get("improvement", "N/A")
        l2 = r.get("level2_rounds", 0)
        l3 = r.get("level3_rounds", 0)
        print(f"  {group}{rep}: [{tag}] improvement={imp} L2={l2} L3={l3}")


def run_group(
    group: str,
    repeat: int,
    inner_budget: int,
    outer_cycles: int,
    enable_level3: bool,
    provider: str,
    model: str,
    bench_bin: Path,
    level2_interval: int = 2,
    level3_interval: int = 2,
    timeout_s: int = 120,
) -> dict:
    run_dir = RESULTS_DIR / f"{group}{repeat}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = _setup_run_logging(run_dir)

    logger.info(
        "=== Group %s | Repeat %d | %d outer × %d inner | L3=%s ===",
        group,
        repeat,
        outer_cycles,
        inner_budget,
        enable_level3,
    )

    try:
        client = _make_llm_client(provider, model)
        mech_config = MechanismResearchConfig(
            level2_interval=level2_interval,
            level3_interval=level3_interval,
            enable_level3=enable_level3,
            enable_tabu=True,
            enable_adaptive_schedule=True,
        )
        controller = GpuBenchTriLevelController(
            run_dir=run_dir,
            bench_bin=bench_bin,
            llm_client=client,
            inner_budget=inner_budget,
            outer_cycles=outer_cycles,
            enable_level3=enable_level3,
            mech_config=mech_config,
            search_config=SearchConfig(inner_budget=inner_budget),
            timeout_s=timeout_s,
        )
        report = controller.run()
        report_dict = report.to_dict()
        report_dict.update({"group": group, "repeat": repeat, "status": "ok"})
        _save_report(run_dir, report_dict)
        logger.info(
            "[%s%d] Done baseline=%.6f best=%.6f improvement=%.6f L2=%d L3=%d",
            group,
            repeat,
            report_dict.get("baseline_bpb") or 0,
            report_dict.get("best_val_bpb") or 0,
            report_dict.get("improvement") or 0,
            report_dict.get("level2_rounds", 0),
            report_dict.get("level3_rounds", 0),
        )
        return report_dict

    except Exception as exc:
        err_text = tb.format_exc()
        logger.error("[%s%d] FAILED:\n%s", group, repeat, err_text)
        error_report = {
            "group": group,
            "repeat": repeat,
            "status": "error",
            "error": str(exc),
            "traceback": err_text,
        }
        _save_report(run_dir, error_report)
        return error_report

    finally:
        _teardown_run_logging(fh)


def _write_summary(all_results: dict[str, list[dict]]) -> Path:
    summary_path = RESULTS_DIR / "ablation_summary.json"
    summary = {}
    for group, results in all_results.items():
        ok = [r for r in results if r.get("status") == "ok"]
        summary[group] = {
            "repeats": len(results),
            "successful": len(ok),
            "improvements": [r.get("improvement") for r in ok],
            "best_val_bpb": [r.get("best_val_bpb") for r in ok],
            "level2_rounds": [r.get("level2_rounds") for r in ok],
            "level3_rounds": [r.get("level3_rounds") for r in ok],
        }
        if ok:
            imps = [r["improvement"] for r in ok if r.get("improvement") is not None]
            bests = [r["best_val_bpb"] for r in ok if r.get("best_val_bpb") != float("inf")]
            summary[group]["mean_improvement"] = statistics.mean(imps) if imps else None
            summary[group]["mean_best_val_bpb"] = statistics.mean(bests) if bests else None

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU bench tri-level ablation: Group C vs F",
    )
    parser.add_argument("--group", default="all", choices=["C", "F", "all"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--inner-budget", type=int, default=5)
    parser.add_argument("--outer-cycles", type=int, default=4)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.bench_bin.is_file():
        logger.error("gpu_bench binary not found: %s", args.bench_bin)
        sys.exit(1)

    groups = ["C", "F"] if args.group == "all" else [args.group]
    all_results: dict[str, list[dict]] = {}

    for group in groups:
        enable_l3 = group == "F"
        logger.info("\n%s\nStarting Group %s (%d repeats)\n%s", "=" * 70, group, args.repeats, "=" * 70)
        group_results = []
        for repeat in range(1, args.repeats + 1):
            result = run_group(
                group=group,
                repeat=repeat,
                inner_budget=args.inner_budget,
                outer_cycles=args.outer_cycles,
                enable_level3=enable_l3,
                provider=args.provider,
                model=args.model,
                bench_bin=args.bench_bin,
                level2_interval=args.level2_interval,
                level3_interval=args.level3_interval,
                timeout_s=args.timeout,
            )
            group_results.append(result)
        all_results[group] = group_results
        _print_group_summary(group, group_results)

    summary_path = _write_summary(all_results)
    if len(groups) > 1:
        print(f"\n{'='*70}\nGPU Bench Tri-Level Ablation Summary\n{'='*70}")
        for grp in groups:
            _print_group_summary(grp, all_results[grp])
    print(f"\nSummary written to: {summary_path}")
    logger.info("GPU bench tri-level ablation complete.")


if __name__ == "__main__":
    main()
