"""Inner-loop runner: subprocess to HIP gpu_bench binary."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.llm_client import LLMClient, parse_json_response
from trilevel_research.domains.gpu_bench_opt.config import GpuBenchConfig
from trilevel_research.domains.gpu_bench_opt.search_config import SearchConfig

logger = logging.getLogger(__name__)

DEFAULT_BIN = Path.home() / "gpu-bench" / "build" / "gpu_bench"

PROPOSE_PROMPT = """\
You optimize a small GPU MLP byte-predictor (HIP on AMD RX 580).
Metric: val_bpb — lower is better.

Current best config:
{config}

Recent trace:
{trace}

Search strategy: {strategy}
Active parameters (you may change ONLY these): {active_params}
Frozen parameters (do NOT change): {frozen_params}
{guidance}

Respond JSON only:
{{"changes": {{"LR": 0.003}}, "hypothesis": "one sentence why"}}
"""


@dataclass
class BenchResult:
    val_bpb: float
    train_bpb: float
    elapsed_s: float
    status: str
    changes: dict = field(default_factory=dict)
    hypothesis: str = ""
    iteration: int = 0
    accepted: bool = False


@dataclass
class BenchTrace:
    results: list[BenchResult] = field(default_factory=list)
    best_val_bpb: float = float("inf")
    best_bpb: float = float("inf")
    best_iteration: int = 0
    best_config: GpuBenchConfig | None = None

    def record(self, result: BenchResult) -> None:
        if result.status == "keep" and result.val_bpb < self.best_bpb:
            self.best_bpb = result.val_bpb
            self.best_val_bpb = result.val_bpb
            self.best_iteration = result.iteration

    def summary(self, last_n: int = 10) -> str:
        lines = [f"Best val_bpb: {self.best_bpb:.6f} (iter {self.best_iteration})"]
        for r in self.results[-last_n:]:
            lines.append(
                f"  iter {r.iteration} [{r.status}] val_bpb={r.val_bpb:.4f} "
                f"changes={r.changes} — {r.hypothesis[:80]}"
            )
        return "\n".join(lines)



class NelderMeadOptimizer:
    """Nelder-Mead simplex optimization for hyperparameter tuning."""

    def __init__(
        self,
        search_config,
        param_ranges,
        alpha=1.0,
        gamma=2.0,
        rho=0.5,
        sigma=0.5,
    ):
        self.active_params = sorted(search_config.active_params)
        self.param_ranges = param_ranges
        self.n = len(self.active_params)
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.simplex = []
        self.phase = "initialize"
        self._last_requested_index = None
        self._last_reflected = None
        self.best_point = None
        self.best_loss = float("inf")

    def normalize(self, params):
        normalized = {}
        for param, value in params.items():
            lo, hi = self.param_ranges.get(param, (0.0, 1.0))
            normalized[param] = (value - lo) / (hi - lo)
        return normalized

    def denormalize(self, normalized):
        denormalized = {}
        for param, value in normalized.items():
            lo, hi = self.param_ranges.get(param, (0.0, 1.0))
            denormalized[param] = value * (hi - lo) + lo
        return denormalized

    def initialize(self, best_config):
        normalized_best = self.normalize(best_config)
        self.simplex = [(best_config.copy(), None)]
        for i in range(self.n):
            new_point = best_config.copy()
            param = self.active_params[i]
            lo, hi = self.param_ranges.get(param, (0.0, 1.0))
            step = 0.05 * (hi - lo)
            nudged = normalized_best[param] + 0.05
            if nudged > 1.0:
                nudged = normalized_best[param] - 0.05
            actual_value = nudged * (hi - lo) + lo
            new_point[param] = max(lo, min(hi, actual_value))
            self.simplex.append((new_point, None))
        self.phase = "reflect"
        return self.simplex[1][0]

    def get_next_point(self, loss=None):
        if loss is not None and self._last_requested_index is not None:
            old_point = self.simplex[self._last_requested_index][0]
            self.simplex[self._last_requested_index] = (old_point, loss)
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_point = old_point
            self._last_requested_index = None
        if self.phase == "initialize":
            for i, (p, l) in enumerate(self.simplex):
                if l is None:
                    return p
            self.phase = "reflect"
        if self.phase == "reflect":
            return self._reflect()
        if self.phase == "expand":
            return self._expand()
        if self.phase == "contract":
            return self._contract()
        if self.phase == "shrink":
            return self._shrink()
        return None

    def _centroid(self, exclude_last=1):
        points = self.simplex[:-exclude_last] if exclude_last else self.simplex
        centroid = {}
        for param in self.active_params:
            centroid[param] = sum(p[0][param] for p in points) / len(points)
        return centroid

    def _reflect(self):
        self.simplex.sort(key=lambda x: x[1] if x[1] is not None else float("inf"), reverse=True)
        centroid = self._centroid(exclude_last=1)
        worst = self.simplex[-1][0]
        c_norm = self.normalize(centroid)
        w_norm = self.normalize(worst)
        r_norm = {}
        for p in self.active_params:
            r_norm[p] = max(0.0, min(1.0, c_norm[p] + self.alpha * (c_norm[p] - w_norm[p])))
        reflected = self.denormalize(r_norm)
        self._last_reflected = reflected
        self._last_requested_index = -1
        self.phase = "evaluate_reflection"
        return reflected

    def _expand(self):
        centroid = self._centroid(exclude_last=1)
        c_norm = self.normalize(centroid)
        r_norm = self.normalize(self._last_reflected)
        e_norm = {}
        for p in self.active_params:
            e_norm[p] = max(0.0, min(1.0, c_norm[p] + self.gamma * (r_norm[p] - c_norm[p])))
        expanded = self.denormalize(e_norm)
        self._last_requested_index = -1
        self.phase = "reflect"
        return expanded

    def _contract(self):
        centroid = self._centroid(exclude_last=1)
        c_norm = self.normalize(centroid)
        w_norm = self.normalize(self.simplex[-1][0])
        ct_norm = {}
        for p in self.active_params:
            ct_norm[p] = max(0.0, min(1.0, c_norm[p] + self.rho * (w_norm[p] - c_norm[p])))
        contracted = self.denormalize(ct_norm)
        self._last_requested_index = -1
        self.phase = "evaluate_contraction"
        return contracted

    def _shrink(self):
        self.simplex.sort(key=lambda x: x[1] if x[1] is not None else float("inf"))
        best = self.simplex[0][0]
        b_norm = self.normalize(best)
        for i in range(1, len(self.simplex)):
            p_norm = self.normalize(self.simplex[i][0])
            s_norm = {}
            for p in self.active_params:
                s_norm[p] = max(0.0, min(1.0, b_norm[p] + self.sigma * (p_norm[p] - b_norm[p])))
            self.simplex[i] = (self.denormalize(s_norm), None)
        self.phase = "reflect"
        return self.simplex[1][0] if len(self.simplex) > 1 else None


class GpuBenchRunner:
    def __init__(
        self,
        bench_bin: Path | None = None,
        timeout_s: int = 120,
        llm_client: LLMClient | None = None,
        search_config: SearchConfig | None = None,
        artifacts_dir: Path | None = None,
        simple_mode: bool = True,
    ) -> None:
        env_bin = os.environ.get("GPU_BENCH_BIN")
        self.bench_bin = Path(env_bin) if env_bin else (bench_bin or DEFAULT_BIN)
        self.timeout_s = timeout_s
        self.client = llm_client
        self.search_config = search_config or SearchConfig()
        self.artifacts_dir = artifacts_dir or Path("artifacts/gpu_bench_opt")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.simple_mode = simple_mode
        self.config = GpuBenchConfig()
        self.trace = BenchTrace()

    def run_baseline(self) -> BenchResult:
        result = self._run_trial(self.config, iteration=0, hypothesis="baseline")
        self.trace.best_val_bpb = result.val_bpb
        self.trace.best_bpb = result.val_bpb
        self.trace.best_iteration = 0
        self.trace.best_config = GpuBenchConfig()
        result.status = "keep"
        result.accepted = True
        self.trace.results.append(result)
        return result

    def run_iteration(
        self,
        iteration: int,
        *,
        changes: dict | None = None,
        hypothesis: str = "",
    ) -> BenchResult:
        """Run one inner iteration. Uses LLM when changes is None."""
        if changes is None:
            if self.client is None:
                raise ValueError("LLM client required when changes not provided")
            changes, hypothesis = self._propose(iteration)

        active = set(self.search_config.active_params)
        filtered = {k: v for k, v in changes.items() if k.upper() in active}
        if not filtered:
            return BenchResult(
                val_bpb=self.trace.best_bpb,
                train_bpb=self.trace.best_bpb,
                elapsed_s=0.0,
                status="discard",
                changes={},
                hypothesis=hypothesis or "no active param changes",
                iteration=iteration,
            )

        trial = self._trial_config(iteration)
        trial.apply_changes(filtered)
        result = self._run_trial(trial, iteration=iteration, hypothesis=hypothesis)
        result.changes = filtered

        if result.status == "crash":
            result.accepted = False
        elif result.val_bpb < self.trace.best_bpb:
            result.status = "keep"
            result.accepted = True
            self.trace.best_val_bpb = result.val_bpb
            self.trace.best_bpb = result.val_bpb
            self.trace.best_iteration = iteration
            self.trace.best_config = trial
            self.config = trial
        else:
            result.status = "discard"
            result.accepted = False

        self.trace.record(result)
        self.trace.results.append(result)
        return result

    def _trial_config(self, iteration: int) -> GpuBenchConfig:
        base = self.trace.best_config or self.config
        return GpuBenchConfig(
            lr=base.lr,
            weight_decay=base.weight_decay,
            batch_size=base.batch_size,
            hidden_dim=base.hidden_dim,
            train_steps=base.train_steps,
            seed=self.config.seed + iteration,
        )

    def _propose(self, iteration: int) -> tuple[dict, str]:
        prompt = PROPOSE_PROMPT.format(
            config=json.dumps(
                (self.trace.best_config or self.config).snapshot(), indent=2
            ),
            trace=self.trace.summary(last_n=5),
            strategy=self.search_config.strategy,
            active_params=", ".join(self.search_config.active_params),
            frozen_params=", ".join(self.search_config.frozen_params) or "(none)",
            guidance=self.search_config.guidance or "",
        )
        resp = self.client.call(prompt, system="You are a hyperparameter researcher.")
        parsed = parse_json_response(resp)
        changes = parsed.get("changes", {}) if isinstance(parsed, dict) else {}
        hypothesis = parsed.get("hypothesis", "") if isinstance(parsed, dict) else ""
        return changes, hypothesis

    def _run_trial(
        self,
        cfg: GpuBenchConfig,
        iteration: int,
        hypothesis: str,
    ) -> BenchResult:
        if not self.bench_bin.is_file():
            raise FileNotFoundError(f"gpu_bench not found: {self.bench_bin}")

        cmd = [str(self.bench_bin), "--json"]
        for k, v in cfg.as_cli_args().items():
            cmd.extend([f"--{k}", str(v)])

        logger.info("Running: %s", " ".join(cmd))
        t0 = time.monotonic()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
        )
        wall = time.monotonic() - t0

        if proc.returncode != 0:
            logger.error("gpu_bench failed rc=%s stderr=%s", proc.returncode, proc.stderr)
            return BenchResult(
                val_bpb=99.0,
                train_bpb=99.0,
                elapsed_s=wall,
                status="crash",
                hypothesis=hypothesis,
                iteration=iteration,
            )

        line = proc.stdout.strip().splitlines()[-1]
        payload = json.loads(line)
        return BenchResult(
            val_bpb=float(payload["val_bpb"]),
            train_bpb=float(payload.get("train_bpb", payload["val_bpb"])),
            elapsed_s=float(payload.get("elapsed_s", wall)),
            status=payload.get("status", "ok"),
            hypothesis=hypothesis,
            iteration=iteration,
        )
