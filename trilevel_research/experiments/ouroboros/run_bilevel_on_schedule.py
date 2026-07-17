"""Ouroboros experiment: bilevel Group C with L2 patching adaptive_mechanism_schedule.py.

L1+L1.5+L2 (no L3): inner gpu_bench hyperparam search + L2 schedule patches.

Usage:
  cd Bilevel-Autoresearch
  export GPU_BENCH_BIN=/path/to/gpu_bench PYTHONPATH=$PWD HIP_VISIBLE_DEVICES=0
  python -m trilevel_research.experiments.ouroboros.run_bilevel_on_schedule \\
    --repeats 1 --inner-budget 5 --outer-cycles 4 --level2-interval 2 --provider deepseek
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
from trilevel_research.core.adaptive_mechanism_schedule import AdaptiveMechanismSchedule
from trilevel_research.core.mechanism_session_trace import MechanismSessionRecord
from trilevel_research.domains.gpu_bench_opt.runner import DEFAULT_BIN
from trilevel_research.domains.gpu_bench_opt.search_config import SearchConfig
from trilevel_research.experiments.ouroboros.schedule_controller import (
    OuroborosScheduleController,
)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ouroboros_bilevel_schedule")


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


def _l2_stats(sessions: list[dict]) -> dict:
    if not sessions:
        return {"rounds": 0, "apply_rate": 0.0, "validate_rate": 0.0, "import_fail": 0}
    attempted = [s for s in sessions if not s.get("blocked_by_tabu")]
    applied = [s for s in attempted if s.get("applied")]
    validated = [s for s in applied if s.get("validated")]
    import_fail = sum(
        1 for s in attempted if (s.get("error") or "").startswith("import_fail") or s.get("error") == "import_fail"
    )
    n = len(attempted) or 1
    return {
        "rounds": len(sessions),
        "attempted": len(attempted),
        "applied": len(applied),
        "validated": len(validated),
        "apply_rate": len(applied) / n,
        "validate_rate": len(validated) / n,
        "import_fail": import_fail,
    }


def _mini_f_run_decisions(
    schedule_py: Path,
    inner_trace: list[dict],
    l2_sessions: list[dict],
    mech_config: MechanismResearchConfig,
) -> list[dict]:
    """Replay schedule.decide() with enable_level3=True to observe L2/L3 firing."""
    schedule = AdaptiveMechanismSchedule.load(schedule_py.parent / "schedule_config.json")
    if schedule_py.exists():
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("_mini_f", schedule_py)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            schedule = mod.AdaptiveMechanismSchedule.load(schedule_py.parent / "schedule_config.json")
        except Exception:
            pass

    records = [
        MechanismSessionRecord(
            round=s.get("round", i + 1),
            session_id=s.get("session_id", f"round_{i+1}"),
            mechanism_name=s.get("mechanism_name", "unknown"),
            implementation_strategy=s.get("implementation_strategy", "unknown"),
            target=s.get("target", "unknown"),
            applied=s.get("applied", False),
            validated=s.get("validated"),
            blocked_by_tabu=s.get("blocked_by_tabu", False),
        )
        for i, s in enumerate(l2_sessions)
    ]

    cfg = MechanismResearchConfig(
        level2_interval=mech_config.level2_interval,
        level3_interval=mech_config.level3_interval,
        enable_level3=True,
    )
    decisions = []
    for cycle in range(1, 5):
        d = schedule.decide(
            inner_trace=inner_trace,
            l2_sessions=records,
            completed_outer_cycles=cycle,
            config=cfg,
        )
        decisions.append({
            "cycle": cycle,
            "fire_level2": d.fire_level2,
            "fire_level3": d.fire_level3,
            "reason": d.reason,
        })
    return decisions


def run_repeat(
    repeat: int,
    *,
    inner_budget: int,
    outer_cycles: int,
    enable_l2: bool,
    provider: str,
    model: str,
    bench_bin: Path,
    level2_interval: int,
    level3_interval: int,
    timeout_s: int,
) -> dict:
    tag = "schedule_l2" if enable_l2 else "l1_only"
    run_dir = RESULTS_DIR / f"{tag}_r{repeat}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = _setup_run_logging(run_dir)

    logger.info(
        "=== Ouroboros %s | Repeat %d | outer=%d inner=%d L2=%s ===",
        tag,
        repeat,
        outer_cycles,
        inner_budget,
        enable_l2,
    )

    try:
        client = _make_llm_client(provider, model)
        mech_config = MechanismResearchConfig(
            level2_interval=level2_interval,
            level3_interval=level3_interval,
            enable_level3=False,
            enable_tabu=True,
            enable_adaptive_schedule=True,
        )
        controller = OuroborosScheduleController(
            run_dir=run_dir,
            bench_bin=bench_bin,
            llm_client=client,
            inner_budget=inner_budget,
            outer_cycles=outer_cycles,
            enable_level3=False,
            enable_l2=enable_l2,
            mech_config=mech_config,
            search_config=SearchConfig(inner_budget=inner_budget),
            timeout_s=timeout_s,
        )
        report = controller.run()
        report_dict = report.to_dict()
        l2_stats = _l2_stats(report_dict.get("level2_sessions", []))

        schedule_py = run_dir / "adaptive_mechanism_schedule.py"
        mini_f = _mini_f_run_decisions(
            schedule_py,
            report_dict.get("trace", []),
            report_dict.get("level2_sessions", []),
            mech_config,
        )

        report_dict.update({
            "mode": tag,
            "repeat": repeat,
            "status": "ok",
            "l2_stats": l2_stats,
            "mini_f_decisions": mini_f,
        })
        (run_dir / "report.json").write_text(
            json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "[%s r%d] improvement=%.6f L2 apply=%.0f%% validate=%.0f%%",
            tag,
            repeat,
            report_dict.get("improvement") or 0,
            100 * l2_stats["apply_rate"],
            100 * l2_stats["validate_rate"],
        )
        return report_dict

    except Exception as exc:
        err_text = tb.format_exc()
        logger.error("[%s r%d] FAILED:\n%s", tag, repeat, err_text)
        error_report = {
            "mode": tag,
            "repeat": repeat,
            "status": "error",
            "error": str(exc),
            "traceback": err_text,
        }
        (run_dir / "report.json").write_text(
            json.dumps(error_report, indent=2), encoding="utf-8"
        )
        return error_report

    finally:
        _teardown_run_logging(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ouroboros: bilevel Group C with L2 schedule patches",
    )
    parser.add_argument("--repeats", type=int, default=1)
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
    parser.add_argument(
        "--baseline-l1-only",
        action="store_true",
        help="Also run L1+L1.5 only (L2 disabled) for comparison",
    )
    parser.add_argument(
        "--disable-l2",
        action="store_true",
        help="Skip schedule L2; run L1+L1.5 only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.bench_bin.is_file():
        logger.error("gpu_bench binary not found: %s", args.bench_bin)
        sys.exit(1)

    all_results: list[dict] = []

    if not args.disable_l2:
        for repeat in range(1, args.repeats + 1):
            all_results.append(
                run_repeat(
                    repeat,
                    inner_budget=args.inner_budget,
                    outer_cycles=args.outer_cycles,
                    enable_l2=True,
                    provider=args.provider,
                    model=args.model,
                    bench_bin=args.bench_bin,
                    level2_interval=args.level2_interval,
                    level3_interval=args.level3_interval,
                    timeout_s=args.timeout,
                )
            )

    if args.baseline_l1_only or args.disable_l2:
        for repeat in range(1, args.repeats + 1):
            all_results.append(
                run_repeat(
                    repeat,
                    inner_budget=args.inner_budget,
                    outer_cycles=args.outer_cycles,
                    enable_l2=False,
                    provider=args.provider,
                    model=args.model,
                    bench_bin=args.bench_bin,
                    level2_interval=args.level2_interval,
                    level3_interval=args.level3_interval,
                    timeout_s=args.timeout,
                )
            )

    summary = {}
    for mode in ("schedule_l2", "l1_only"):
        mode_results = [r for r in all_results if r.get("mode") == mode and r.get("status") == "ok"]
        if not mode_results:
            continue
        apply_rates = [r["l2_stats"]["apply_rate"] for r in mode_results]
        validate_rates = [r["l2_stats"]["validate_rate"] for r in mode_results]
        summary[mode] = {
            "runs": len(mode_results),
            "mean_apply_rate": statistics.mean(apply_rates) if apply_rates else 0,
            "mean_validate_rate": statistics.mean(validate_rates) if validate_rates else 0,
            "mini_f_decisions": mode_results[-1].get("mini_f_decisions", []),
        }

    summary_path = RESULTS_DIR / "ouroboros_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nOuroboros summary: {summary_path}")
    for mode, stats in summary.items():
        print(
            f"  {mode}: apply={stats['mean_apply_rate']:.0%} "
            f"validate={stats['mean_validate_rate']:.0%}"
        )


if __name__ == "__main__":
    main()
