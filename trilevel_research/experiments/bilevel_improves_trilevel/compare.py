"""Shared C vs F comparison metrics for bilevel_improves_trilevel drivers."""
from __future__ import annotations

import statistics
from typing import Any


def l2_apply_rate(sessions: list[dict]) -> float:
    attempted = [s for s in sessions if not s.get("blocked_by_tabu")]
    if not attempted:
        return 0.0
    applied = sum(1 for s in attempted if s.get("applied"))
    return applied / len(attempted)


def compare_groups(
    all_results: dict[str, list[dict]],
    *,
    ouroboros_l2_apply_rate: float = 0.0,
    margin_abs: float = 0.01,
    margin_pct: float | None = None,
) -> dict[str, Any]:
    """Compare Group C (bi-level) vs Group F (tri-level).

    Success when F beats C by margin_abs OR by margin_pct relative to C mean:
      (F_mean - C_mean) / C_mean * 100 >= margin_pct
    Also requires ouroboros_l2_apply_rate > 0 when margin_pct is set (schedule layer).
    """
    summary: dict[str, Any] = {}
    for group in ("C", "F"):
        ok = [r for r in all_results.get(group, []) if r.get("status") == "ok"]
        imps = [r["improvement"] for r in ok if r.get("improvement") is not None]
        summary[group] = {
            "successful": len(ok),
            "mean_improvement": statistics.mean(imps) if imps else None,
            "stdev_improvement": statistics.stdev(imps) if len(imps) > 1 else 0.0,
            "improvements": imps,
            "l2_apply_rates": [l2_apply_rate(r.get("level2_sessions", [])) for r in ok],
            "l3_rounds": [r.get("level3_rounds", 0) for r in ok],
        }

    c_mean = summary.get("C", {}).get("mean_improvement")
    f_mean = summary.get("F", {}).get("mean_improvement")
    margin = (f_mean - c_mean) if (c_mean is not None and f_mean is not None) else None
    margin_pct_actual = (margin / c_mean * 100.0) if (margin is not None and c_mean) else None

    success = False
    if c_mean is not None and f_mean is not None:
        abs_ok = f_mean > c_mean + margin_abs
        pct_ok = margin_pct_actual is not None and margin_pct_actual >= (margin_pct or 0.0)
        if margin_pct is not None:
            success = pct_ok and ouroboros_l2_apply_rate > 0
        else:
            success = abs_ok and ouroboros_l2_apply_rate > 0

    return {
        "group_summary": summary,
        "c_mean_improvement": c_mean,
        "f_mean_improvement": f_mean,
        "margin": margin,
        "margin_pct": margin_pct_actual,
        "margin_pct_required": margin_pct,
        "margin_abs_required": margin_abs,
        "ouroboros_l2_apply_rate": ouroboros_l2_apply_rate,
        "success": success,
    }
