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
        from dataclasses import dataclass
        import math
        import random
        from typing import Dict, Optional, Set, Tuple

        @dataclass
        class ExplorationState:
            dimension_attempts: Dict[str, int] = None
            dimension_successes: Dict[str, int] = None
            last_direction: Dict[str, float] = None
            consecutive_failures: int = 0
            stagnation_counter: int = 0
            exploration_budget: int = 10
            last_improvement_iteration: int = 0

            def __post_init__(self):
                if self.dimension_attempts is None:
                    self.dimension_attempts = {}
                if self.dimension_successes is None:
                    self.dimension_successes = {}
                if self.last_direction is None:
                    self.last_direction = {}

        class StructuredRandomExplorer:
            def __init__(self, config, trace):
                self.config = config
                self.trace = trace
                self.state = ExplorationState()
                self.param_bounds = self._get_param_bounds()

            def _get_param_bounds(self):
                bounds = {}
                if hasattr(self.config, 'lr_range'):
                    bounds['lr'] = self.config.lr_range
                if hasattr(self.config, 'batch_size_range'):
                    bounds['batch_size'] = self.config.batch_size_range
                if hasattr(self.config, 'hidden_dim_range'):
                    bounds['hidden_dim'] = self.config.hidden_dim_range
                return bounds

            def propose_exploration(self, iteration, best_config):
                active_params = set(self.config.active_params)
                if iteration - self.state.last_improvement_iteration > 5:
                    return self._propose_reset_exploration(active_params, best_config)
                least_explored = self._find_least_explored_param(active_params)
                if least_explored:
                    return self._probe_parameter(least_explored, best_config, active_params)
                return self._propose_local_perturbation(active_params, best_config)

            def _find_least_explored_param(self, active_params):
                if not active_params:
                    return None
                attempts = {p: self.state.dimension_attempts.get(p, 0) for p in active_params}
                min_attempts = min(attempts.values())
                least_explored = [p for p, a in attempts.items() if a == min_attempts]
                return least_explored[0] if least_explored else None

            def _propose_reset_exploration(self, active_params, best_config):
                param = self._find_least_explored_param(active_params)
                if not param:
                    param = next(iter(active_params))
                changes = {}
                if param == 'lr' and 'lr' in self.param_bounds:
                    min_lr, max_lr = self.param_bounds['lr']
                    changes['lr'] = 10 ** ((math.log10(min_lr) + math.log10(max_lr)) / 2)
                elif param == 'batch_size' and 'batch_size' in self.param_bounds:
                    min_bs, max_bs = self.param_bounds['batch_size']
                    changes['batch_size'] = (min_bs + max_bs) // 2
                elif param == 'hidden_dim' and 'hidden_dim' in self.param_bounds:
                    min_hd, max_hd = self.param_bounds['hidden_dim']
                    changes['hidden_dim'] = (min_hd + max_hd) // 2
                return changes, f"reset_exploration:{param}:midpoint"

            def _probe_parameter(self, param, best_config, active_params):
                if param not in self.param_bounds:
                    return self._propose_local_perturbation(active_params, best_config)
                min_val, max_val = self.param_bounds[param]
                current_val = getattr(best_config, param, None)
                if current_val is None:
                    return self._propose_local_perturbation(active_params, best_config)
                last_dir = self.state.last_direction.get(param, 1)
                failures_in_dim = self.state.dimension_attempts.get(param, 0) - self.state.dimension_successes.get(param, 0)
                direction = -last_dir if failures_in_dim > 2 else last_dir
                if param == 'lr':
                    step_factor = 1.5 if direction > 0 else 1/1.5
                    new_val = current_val * step_factor
                elif param == 'batch_size':
                    step = 16 * direction
                    new_val = max(16, min(current_val + step, max_val))
                elif param == 'hidden_dim':
                    step = 32 * direction
                    new_val = max(32, min(current_val + step, max_val))
                new_val = max(min_val, min(new_val, max_val))
                self.state.last_direction[param] = direction
                changes = {param: new_val}
                return changes, f"probe:{param}:dir={direction}:val={new_val}"

            def _propose_local_perturbation(self, active_params, best_config):
                param = random.choice(list(active_params))
                if param == 'lr':
                    last_dir = self.state.last_direction.get('lr', 1)
                    bias = 0.2 * last_dir
                    noise = random.gauss(bias, 0.3)
                    changes = {'lr': best_config.lr * (10 ** noise)}
                elif param == 'batch_size':
                    step = random.choice([-16, -8, 8, 16])
                    changes = {'batch_size': max(16, min(best_config.batch_size + step, 256))}
                elif param == 'hidden_dim':
                    step = random.choice([-32, -16, 16, 32])
                    changes = {'hidden_dim': max(32, min(best_config.hidden_dim + step, 1024))}
                else:
                    changes = {}
                return changes, f"local_perturbation:{param}:multi_modal"

            def update_state(self, changes, was_improvement):
                for param in changes:
                    self.state.dimension_attempts[param] = self.state.dimension_attempts.get(param, 0) + 1
                    if was_improvement:
                        self.state.dimension_successes[param] = self.state.dimension_successes.get(param, 0) + 1
                if was_improvement:
                    self.state.consecutive_failures = 0
                    if self.trace.results:
                        self.state.last_improvement_iteration = self.trace.results[-1].iteration
                else:
                    self.state.consecutive_failures += 1

        # Append to __init__:
                self.explorer = StructuredRandomExplorer(
                    config=self.search_config,
                    trace=self.trace
                )
        class MultiVariatePerturbationExplorer:
            """Joint perturbation explorer for continuous hyperparameters.
    
            Learns correlation structure from successful configurations to propose
            correlated multi-parameter perturbations.
            """
            def __init__(self, param_names, param_ranges):
                self.param_names = param_names
                self.param_ranges = param_ranges
                self.success_history = []
                self.covariance_matrix = None
                self.mean_vector = None
                self.min_samples = 5
                self.scale_factor = 0.1
        
            def _normalize_config(self, config_dict):
                arr = []
                for name in self.param_names:
                    lo, hi = self.param_ranges[name]
                    val = config_dict[name]
                    normalized = (val - lo) / (hi - lo) if hi != lo else 0.5
                    arr.append(normalized)
                return np.array(arr)
    
            def _denormalize_config(self, normalized):
                config = {}
                for i, name in enumerate(self.param_names):
                    lo, hi = self.param_ranges[name]
                    val = normalized[i] * (hi - lo) + lo
                    config[name] = val
                return config
    
            def add_observation(self, config_dict, loss):
                self.success_history.append((config_dict.copy(), loss))
                if len(self.success_history) >= self.min_samples:
                    self._update_covariance()
            
            def _update_covariance(self):
                sorted_history = sorted(self.success_history, key=lambda x: x[1])
                best_n = max(len(sorted_history) // 2, self.min_samples)
                best_configs = sorted_history[:best_n]
                normalized_configs = np.array([
                    self._normalize_config(c) for c, _ in best_configs
                ])
                self.mean_vector = np.mean(normalized_configs, axis=0)
                centered = normalized_configs - self.mean_vector
                if len(normalized_configs) > 1:
                    self.covariance_matrix = np.cov(centered, rowvar=False)
                    self.covariance_matrix += np.eye(len(self.param_names)) * 0.01
                else:
                    self.covariance_matrix = np.eye(len(self.param_names)) * 0.01
        
            def propose_perturbation(self, current_config):
                if len(self.success_history) < self.min_samples:
                    return self._propose_univariate_exploration(current_config)
                normalized_current = self._normalize_config(current_config)
                try:
                    if self.covariance_matrix is not None:
                        perturbation = np.random.multivariate_normal(
                            mean=np.zeros(len(self.param_names)),
                            cov=self.covariance_matrix * self.scale_factor
                        )
                    else:
                        perturbation = np.random.uniform(-0.1, 0.1, size=len(self.param_names))
                except np.linalg.LinAlgError:
                    perturbation = np.random.uniform(-0.1, 0.1, size=len(self.param_names))
                new_normalized = normalized_current + perturbation
                new_normalized = np.clip(new_normalized, 0.0, 1.0)
                return self._denormalize_config(new_normalized)
    
            def _propose_univariate_exploration(self, current_config):
                proposed = current_config.copy()
                for name in self.param_names:
                    lo, hi = self.param_ranges[name]
                    val = current_config[name]
                    lo = lo if lo < hi else hi
                    hi = hi if hi > lo else lo
                    perturbation = np.random.uniform(-0.1, 0.1) * (hi - lo)
                    proposed[name] = np.clip(val + perturbation, lo, hi)
                return proposed

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
