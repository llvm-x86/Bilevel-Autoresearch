"""CPU-only counterfactual simulation: Group C fixtures vs tri-level (Group F) policies.

Replays published paper-ablation Group C traces (no GPU, no LLM) and estimates
whether Level 3 policies (tabu registry, adaptive schedule, validation harness)
would have improved L2 efficiency.

Usage:
  python -m trilevel_research.experiments.tri_level_ablation.simulate_from_fixtures
  python -m trilevel_research.experiments.tri_level_ablation.simulate_from_fixtures --write-report
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIXTURE_ROOT = (
    REPO_ROOT / "experiments/ablations/paper_ablation/run2_results/results_C"
)
ABLATION_DIR = Path(__file__).parent
RESULTS_DIR = ABLATION_DIR / "simulation_results"

import sys

sys.path.insert(0, str(REPO_ROOT))

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


@dataclass
class SimulatedL2Session:
    original: MechanismSessionRecord
    blocked_by_tabu: bool = False
    blocked_by_harness: bool = False
    would_apply: bool = False
    would_validate: bool = False
    tabu_reason: str = ""


@dataclass
class RepeatSimulation:
    repeat: str
    baseline_improvement: float
    best_val_bpb: float
    baseline_bpb: float
    original_l2: list[SimulatedL2Session] = field(default_factory=list)
    simulated_l2: list[SimulatedL2Session] = field(default_factory=list)
    schedule_decisions: list[dict] = field(default_factory=list)
    l3_fires: int = 0
    tabu_blocks: int = 0
    harness_blocks: int = 0

    def l2_metrics(self, sessions: list[SimulatedL2Session]) -> dict:
        attempted = len(sessions)
        if attempted == 0:
            return {
                "attempted": 0,
                "apply_rate": 0.0,
                "revert_rate": 0.0,
                "validated_rate": 0.0,
                "blocked_rate": 0.0,
            }
        applied = sum(1 for s in sessions if s.would_apply)
        validated = sum(1 for s in sessions if s.would_validate)
        blocked = sum(
            1 for s in sessions if s.blocked_by_tabu or s.blocked_by_harness
        )
        reverts = sum(
            1 for s in sessions if s.would_apply and not s.would_validate
        )
        return {
            "attempted": attempted,
            "apply_rate": applied / attempted,
            "revert_rate": reverts / max(applied, 1),
            "validated_rate": validated / attempted,
            "blocked_rate": blocked / attempted,
        }


def _load_report(repeat_dir: Path) -> dict:
    return json.loads((repeat_dir / "report.json").read_text())


def _sessions_from_report(report: dict, repeat_dir: Path) -> list[MechanismSessionRecord]:
    mech_root = repeat_dir / "mechanism_sessions"
    if mech_root.is_dir():
        return MechanismSessionTraceBuilder.build(mech_root, report=report)

    records: list[MechanismSessionRecord] = []
    for entry in report.get("level2_sessions", []):
        records.append(
            MechanismSessionRecord(
                round=entry.get("round", 0),
                session_id=entry.get("session_id", ""),
                mechanism_name=entry.get("mechanism_name", "unknown"),
                implementation_strategy=entry.get("strategy", ""),
                target=entry.get("target", ""),
                code_retries=entry.get("code_retries", 0),
                applied=entry.get("applied", False),
                validated=entry.get("validated", False),
                error=entry.get("error", ""),
            )
        )
    return records


def _simulate_repeat(repeat_dir: Path) -> RepeatSimulation:
    report = _load_report(repeat_dir)
    repeat = repeat_dir.name
    sessions = _sessions_from_report(report, repeat_dir)
    trace = report.get("trace", [])
    outer_cycles = report.get("outer_cycles", 6)

    sim = RepeatSimulation(
        repeat=repeat,
        baseline_improvement=report.get("improvement", 0.0),
        best_val_bpb=report.get("best_val_bpb", 0.0),
        baseline_bpb=report.get("baseline_bpb", 0.0),
    )

    tabu = MechanismTabuRegistry(max_size=20, default_tenure=3)
    schedule = AdaptiveMechanismSchedule()
    harness = MechanismValidationHarness(domain="train_opt")
    config = MechanismResearchConfig(
        level2_interval=2,
        level3_interval=2,
        enable_level3=True,
        enable_tabu=True,
        enable_adaptive_schedule=True,
    )

    l2_round = 0
    simulated: list[SimulatedL2Session] = []

    for session in sessions:
        l2_round += 1
        orig = SimulatedL2Session(
            original=session,
            would_apply=session.applied,
            would_validate=session.validated,
        )
        sim.original_l2.append(orig)

        blocked_tabu, tabu_reason = tabu.is_tabu(
            session.mechanism_name,
            session.target,
            l2_round,
            strategy=session.implementation_strategy,
        )
        blocked_harness = False

        if blocked_tabu:
            sim.tabu_blocks += 1
            sim_l2 = SimulatedL2Session(
                original=session,
                blocked_by_tabu=True,
                would_apply=False,
                would_validate=False,
                tabu_reason=tabu_reason,
            )
        else:
            # Harness catches import/syntax failures before GPU apply
            if session.applied and not session.validated:
                code_path = session.session_dir / "04_code_attempt_1.py"
                if code_path.is_file():
                    err = harness.validate_syntax(code_path.read_text())
                    if err:
                        blocked_harness = True
                        sim.harness_blocks += 1

            if blocked_harness:
                sim_l2 = SimulatedL2Session(
                    original=session,
                    blocked_by_harness=True,
                    would_apply=False,
                    would_validate=False,
                )
                tabu.record_failure(session, l2_round, reason="harness_syntax_fail")
            elif session.applied:
                if session.validated:
                    sim_l2 = SimulatedL2Session(
                        original=session,
                        would_apply=True,
                        would_validate=True,
                    )
                    tabu.record_success(session, l2_round)
                else:
                    sim_l2 = SimulatedL2Session(
                        original=session,
                        would_apply=True,
                        would_validate=False,
                    )
                    tabu.record_failure(session, l2_round, reason="import_fail")
            else:
                sim_l2 = SimulatedL2Session(
                    original=session,
                    would_apply=False,
                    would_validate=False,
                )
                tabu.record_failure(session, l2_round, reason="not_applied")

        simulated.append(sim_l2)

        # Schedule decisions at each outer-cycle boundary
        partial_trace = trace[: max(1, l2_round * 5)]
        decision = schedule.decide(
            inner_trace=partial_trace,
            l2_sessions=[s.original for s in simulated],
            completed_outer_cycles=min(l2_round * 2, outer_cycles),
            config=config,
        )
        sim.schedule_decisions.append(
            {
                "l2_round": l2_round,
                "fire_level2": decision.fire_level2,
                "fire_level3": decision.fire_level3,
                "reason": decision.reason,
            }
        )
        if decision.fire_level3:
            sim.l3_fires += 1

    sim.simulated_l2 = simulated
    return sim


def run_simulation() -> dict:
    repeats = sorted(FIXTURE_ROOT.glob("C*"))
    if not repeats:
        raise FileNotFoundError(f"No C-group fixtures under {FIXTURE_ROOT}")

    all_sims = [_simulate_repeat(d) for d in repeats]

    def mean_metric(key: str, group: str) -> float:
        vals = []
        for s in all_sims:
            metrics = (
                s.l2_metrics(s.original_l2)
                if group == "C"
                else s.l2_metrics(s.simulated_l2)
            )
            vals.append(metrics[key])
        return statistics.mean(vals) if vals else 0.0

    c_apply = mean_metric("apply_rate", "C")
    f_apply = mean_metric("apply_rate", "F")
    c_revert = mean_metric("revert_rate", "C")
    f_revert = mean_metric("revert_rate", "F")
    c_valid = mean_metric("validated_rate", "C")
    f_valid = mean_metric("validated_rate", "F")

    improvements = [s.baseline_improvement for s in all_sims]

    summary = {
        "fixture_source": str(FIXTURE_ROOT),
        "repeats": [s.repeat for s in all_sims],
        "group_C_actual": {
            "mean_delta_val_bpb": statistics.mean(improvements),
            "std_delta_val_bpb": (
                statistics.stdev(improvements) if len(improvements) > 1 else 0.0
            ),
            "l2_apply_rate": c_apply,
            "l2_revert_rate": c_revert,
            "l2_validated_rate": c_valid,
        },
        "group_F_simulated_policies": {
            "l2_apply_rate": f_apply,
            "l2_revert_rate": f_revert,
            "l2_validated_rate": f_valid,
            "tabu_blocks_total": sum(s.tabu_blocks for s in all_sims),
            "harness_blocks_total": sum(s.harness_blocks for s in all_sims),
            "l3_fire_decisions": sum(s.l3_fires for s in all_sims),
        },
        "per_repeat": [],
    }

    for s in all_sims:
        summary["per_repeat"].append(
            {
                "repeat": s.repeat,
                "delta_val_bpb": s.baseline_improvement,
                "C_l2_metrics": s.l2_metrics(s.original_l2),
                "F_l2_metrics": s.l2_metrics(s.simulated_l2),
                "tabu_blocks": s.tabu_blocks,
                "harness_blocks": s.harness_blocks,
                "l3_fires": s.l3_fires,
                "schedule": s.schedule_decisions,
            }
        )

    # Counterfactual task metric: if L2 reverts blocked, inner loop keeps searching
    # instead of running broken patches — estimate modest gain when revert rate drops
    revert_reduction = max(0.0, c_revert - f_revert)
    summary["counterfactual_estimate"] = {
        "revert_rate_reduction": revert_reduction,
        "estimated_task_gain_from_l3": revert_reduction * 0.02,
        "note": (
            "Conservative heuristic: each avoided L2 revert saves ~1 wasted outer "
            "batch; 0.02 val_bpb per revert based on C1 batch-3 breakthrough pattern."
        ),
    }

    return summary


def write_report(summary: dict, path: Path) -> None:
    c = summary["group_C_actual"]
    f = summary["group_F_simulated_policies"]
    est = summary["counterfactual_estimate"]

    lines = [
        "# Tri-Level CPU Simulation Report",
        "",
        "CPU-only counterfactual replay of paper ablation **Group C** fixtures",
        f"(`{summary['fixture_source']}`).",
        "",
        "No GPU or LLM calls. Simulates Level 3 policies (tabu registry, adaptive",
        "schedule, validation harness) against historical L2 session artifacts.",
        "",
        "## Summary",
        "",
        "| Metric | Group C (actual) | Group F (L3 policies simulated) |",
        "|--------|------------------|----------------------------------|",
        f"| Mean Δval_bpb | {c['mean_delta_val_bpb']:.4f} ± {c['std_delta_val_bpb']:.4f} | (same task trace — meta-layer only) |",
        f"| L2 apply rate | {c['l2_apply_rate']:.0%} | {f['l2_apply_rate']:.0%} |",
        f"| L2 revert rate | {c['l2_revert_rate']:.0%} | {f['l2_revert_rate']:.0%} |",
        f"| L2 validated rate | {c['l2_validated_rate']:.0%} | {f['l2_validated_rate']:.0%} |",
        f"| Tabu blocks | — | {f['tabu_blocks_total']} |",
        f"| Harness blocks | — | {f['harness_blocks_total']} |",
        f"| L3 fire decisions | — | {f['l3_fire_decisions']} |",
        "",
        "## Interpretation",
        "",
    ]

    if f["l2_revert_rate"] < c["l2_revert_rate"]:
        lines.append(
            f"- **L2 revert rate drops** from {c['l2_revert_rate']:.0%} to "
            f"{f['l2_revert_rate']:.0%} — tabu + harness prevent re-applying "
            "broken mechanisms."
        )
    if f["tabu_blocks_total"] > 0:
        lines.append(
            f"- **Tabu registry** would have blocked {f['tabu_blocks_total']} "
            "duplicate/failed mechanism proposals across repeats."
        )
    if f["l3_fire_decisions"] > 0:
        lines.append(
            f"- **Adaptive schedule** triggered {f['l3_fire_decisions']} Level 3 "
            "escalations (high L2 revert rate in all C repeats)."
        )

    lines.extend(
        [
            "",
            "## Counterfactual task estimate",
            "",
            f"- Revert rate reduction: {est['revert_rate_reduction']:.0%}",
            f"- Estimated additional Δval_bpb from avoided bad L2 patches: "
            f"**+{est['estimated_task_gain_from_l3']:.4f}**",
            f"- {est['note']}",
            "",
            "## Per-repeat detail",
            "",
        ]
    )

    for pr in summary["per_repeat"]:
        lines.append(f"### {pr['repeat']}")
        lines.append(f"- Δval_bpb: {pr['delta_val_bpb']:.4f}")
        lines.append(
            f"- C revert rate: {pr['C_l2_metrics']['revert_rate']:.0%} → "
            f"F: {pr['F_l2_metrics']['revert_rate']:.0%}"
        )
        lines.append(
            f"- Tabu blocks: {pr['tabu_blocks']}, harness blocks: {pr['harness_blocks']}, "
            f"L3 fires: {pr['l3_fires']}"
        )
        lines.append("")

    lines.extend(
        [
            "## Limitations",
            "",
            "- CPU simulation replays historical traces; does not re-run training.",
            "- Full GPU ablation (Group F live) requires RTX 5090 + DeepSeek API.",
            "- Counterfactual task gain is a heuristic, not measured val_bpb.",
            "",
            "## Next steps",
            "",
            "```bash",
            "python -m trilevel_research.experiments.tri_level_ablation.run_ablation \\",
            "  --group all --repeats 3 --iterations 30 --outer-cycles 6",
            "```",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU fixture replay: C vs F policies")
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write REPORT.md and simulation_summary.json",
    )
    args = parser.parse_args()

    summary = run_simulation()
    print(json.dumps(summary, indent=2))

    if args.write_report:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = RESULTS_DIR / "simulation_summary.json"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        report_path = ABLATION_DIR / "REPORT.md"
        write_report(summary, report_path)
        print(f"\nWrote {json_path}")
        print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
