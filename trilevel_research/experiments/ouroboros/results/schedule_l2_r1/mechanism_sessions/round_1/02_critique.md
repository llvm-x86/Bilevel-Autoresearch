## Critique of Hypotheses

### Hypothesis 1: Volatility-Gated Research Scheduling

**Most likely failure mode**: False stability trap. The system will *create* its own low-volatility periods by suppressing research exactly when volatility is high, then fire L2 during deceptively quiet periods that are actually plateaus. The 0.6 threshold is arbitrary and will cause oscillation — high volatility → suppress → low volatility → fire L2 → generate noise → high volatility → suppress → repeat. This turns a bursty problem into a periodic one.

**Implementation trap**: The volatility metric conflates *status changes* (discard/keep flip-flopping) with *exploration noise*. A legitimate hill-climbing search will naturally produce alternating discard/keep patterns as it overshoots and corrects. The code as written cannot distinguish between "thrashing" and "healthy gradient descent with momentum."

**Evidence from trace**: Partially supports. Iterations 5-11 show 6 status changes in 7 transitions (85% volatility), so the gate would fire. But iteration 4 → 5 shows a keep→discard transition after a genuine improvement — this single transition would create 100% volatility in a 3-window, blocking L2 research that might have diagnosed *why* iteration 5 was bad.

**Score**: impact 3 × feasibility 4 ÷ complexity 2 = **6.0**

### Hypothesis 2: Performance-Parameter Correlation Thresholding

**Most likely failure mode**: Silent starvation. The correlation calculation only considers *improvements* and counts parameter change *counts* (not which parameters changed). If the search is making single-parameter changes that consistently produce 5% improvements, correlation is high. But if it's making 3-parameter changes that randomly produce 70x performance swings, correlation is zero. This masks the exact scenario in the trace — high-impact changes mixed with catastrophic failures. The threshold will block research precisely when it's most needed to understand the catastrophic failures.

**Implementation trap**: Numerical instability in correlation calculation. `trace[i].get("val_bpb", float('inf'))` returns infinity for discarded configurations, making `(prev_perf - curr_perf) / prev_perf` = -inf for *any* discard. Correlation with infinite values is undefined. The code silently drops these (only considers improvements), so the correlation only sees half the data — the good half. This creates survivorship bias in the research trigger.

**Evidence from trace**: Contradicts. The trace shows correlation between parameter changes and performance is *extremely high* — changing parameters causes val_bpb to jump from 0.0998 to 7.0+. The hypothesized correlation filter would show STRONG correlation and fire L2 *more*, not less. This is the opposite of the intended effect.

**Score**: impact 1 × feasibility 2 ÷ complexity 4 = **0.5**

### Hypothesis 3: Temporal Reward-Consolidation Window

**Most likely failure mode**: Hysteresis death spiral. After finding a local optimum (iteration 4), the consolidation window blocks L2 for 3 cycles. During those 3 cycles, the inner loop explores degenerate configurations (iterations 5-7). When the window expires at iteration 7, the inner loop is in a terrible state (val_bpb=6.97). L2 fires into this bad state, produces negative recommendations that poison the search further, and the system never recovers because the "last best" is anchored at iteration 4 while the actual state is at iteration 7+.

**Implementation trap**: The code checks `last_iter.get("is_new_best")` but the trace doesn't show this field. The trace only shows `status: best` at iteration 4. If no explicit `is_new_best` flag exists, the consolidation window *never activates*. The window is implemented as a dead letter if the trace schema doesn't match assumptions.

**Evidence from trace**: Strongly supports. Iteration 4 is clearly the best, followed by 7 iterations of garbage. A 3-cycle window would block L2 from iterations 5-7, which would prevent L2 from analyzing iteration 5's discard. But the trace shows 4 consecutive discards (5-8) followed by 3 discards in 10-12 — the window would expire and fire L2 into iteration 8 or 11, both terrible states.

**Score**: impact 4 × feasibility 4 ÷ complexity 3 = **5.33**

### Hypothesis 4: Gradient-Aware Research Scheduling

**Most likely failure mode**: Gradient sign reversal trap. The code computes `slope = (recent[-1] - recent[0]) / len(recent)`. For iteration 4→5, recent[-1]=7.0, recent[0]=0.0998, slope = +1.38 — positive (bad). But iteration 4 was the BEST result. The gradient is *always positive* when leaving an optimum, because you're getting worse. This means the system suppresses L2 *exactly when it's leaving the best-found configuration* — the most critical time to understand what changed. The system creates a dead zone around every local optimum where research is suppressed during the most informative exploration.

**Implementation trap**: Outlier handling. `recent = [x for x in recent if x < float('inf')]` is a no-op because float('inf') < float('inf') is false, but what about NaN? The trace doesn't show NaN, but many systems return NaN for invalid configurations. NaN comparisons always return false, so `recent = [x for x in recent if x < float('inf')]` would *include* NaN values. Then `recent[-1] - recent[0]` returns NaN, and `NaN > 0.5` is false, so the gate silently opens for NaN-dominated windows.

**Evidence from trace**: Ambiguous. The trace shows positive gradient from iteration 4 onward (+1.38 per iteration initially), which would suppress L2. But the hypothesized benefit ("defer L2 during bad regions") requires L2 to fire *during improvement* — but the trace has zero improvement after iteration 4. The system would become permanently locked out of L2 research because the gradient is never negative.

**Score**: impact 2 × feasibility 3 ÷ complexity 3 = **2.0**

---

## Summary Table

| Hypothesis | Impact | Feasibility | Complexity | Score | Trace Support |
|------------|--------|-------------|------------|-------|---------------|
| 1 (Volatility) | 3 | 4 | 2 | 6.0 | Partial |
| 2 (Correlation) | 1 | 2 | 4 | 0.5 | Contradicts |
| 3 (Consolidation) | 4 | 4 | 3 | 5.33 | Strong |
| 4 (Gradient) | 2 | 3 | 3 | 2.0 | Ambiguous |

**Selected**: Hypothesis 1 — the volatility gate is the least bad option because its false stability trap is self-correcting (suppression → lower volatility → gate opens), while the others have self-reinforcing failure modes (dead zones, survivorship bias, permanent lockout).