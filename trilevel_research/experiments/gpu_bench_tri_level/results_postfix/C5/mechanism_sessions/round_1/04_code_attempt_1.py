from dataclasses import dataclass, field
from typing import List, Optional
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchTrace:
    """Extension of existing BenchTrace to include plausibility tracking."""
    baseline_val_bpb: float = float('inf')
    validation_events: List[str] = field(default_factory=list)


class PlausibilityChecker:
    """Small helper class to validate result plausibility for GpuBenchRunner."""

    MIN_PLAUSIBLE_BPB = 0.5
    MAX_IMPROVEMENT_RATIO = 0.1  # 10x better than best_bpb is suspicious

    def __init__(self, runner: 'GpuBenchRunner'):
        self.runner = runner

    def is_plausible(self, result: 'BenchResult', iteration: int) -> bool:
        """Check if a result is physically plausible given context and history."""
        # Rule 1: Absolute minimum bpb
        if result.val_bpb < self.MIN_PLAUSIBLE_BPB:
            self.runner.trace.validation_events.append(
                f"iter{iteration}: val_bpb={result.val_bpb:.4f} < ABS_MIN={self.MIN_PLAUSIBLE_BPB}"
            )
            return False

        # Rule 2: NaN or infinity
        if not math.isfinite(result.val_bpb):
            self.runner.trace.validation_events.append(
                f"iter{iteration}: val_bpb={result.val_bpb} is NaN or Inf"
            )
            return False

        # Rule 3: Unbelievable improvement over best
        best_bpb = self.runner.trace.best_bpb
        if best_bpb > 0 and best_bpb < float('inf'):
            improvement_ratio = result.val_bpb / best_bpb
            if result.val_bpb < best_bpb * self.MAX_IMPROVEMENT_RATIO:
                self.runner.trace.validation_events.append(
                    f"iter{iteration}: val_bpb={result.val_bpb:.4f} is {improvement_ratio:.2e}x better than best={best_bpb:.4f}"
                )
                return False

        # Rule 4: Context length based lower bound (optional, inferred from config)
        ctx_len = 1024
        if self.runner.config is not None:
            ctx_len = getattr(self.runner.config, 'CONTEXT_LENGTH', 1024)
        # No strict lower bound from ctx_len alone, but could be expanded

        return True