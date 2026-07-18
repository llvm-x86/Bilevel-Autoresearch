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



class PerturbationDecayTracker:
    """Tracks global iteration count and computes decay-based perturbation scaling."""
    
    def __init__(self, decay_rate=0.95, min_factor=0.1, max_factor=2.0):
        self.global_iteration_count = 0
        self.decay_rate = decay_rate
        self.min_perturbation_factor = min_factor
        self.max_perturbation_factor = max_factor
    
    def record_perturbation(self):
        self.global_iteration_count += 1
    
    def compute_perturbation_factor(self, best_config_last_updated_iteration):
        iterations_since_best_update = self.global_iteration_count - best_config_last_updated_iteration
        decay_factor = self.decay_rate ** iterations_since_best_update
        factor = (self.max_perturbation_factor * (1 - decay_factor) +
                  self.min_perturbation_factor * decay_factor)
        return max(self.min_perturbation_factor, min(self.max_perturbation_factor, factor))
    
    def should_be_exploratory(self, decay_factor):
        return decay_factor <= 0.5
    
    def get_param_weights(self, iteration):
        if iteration < 10:
            return {'LR': 0.4, 'BATCH_SIZE': 0.4, 'OPTIMIZER': 0.2}
        elif iteration < 30:
            return {'LR': 0.3, 'BATCH_SIZE': 0.3, 'OPTIMIZER': 0.4}
        else:
            return {'LR': 0.2, 'BATCH_SIZE': 0.2, 'OPTIMIZER': 0.6}


class PerturbedHyperparameterEnginePatch:
    """Small helper class that provides the replace_method generate_perturbation."""
    
    def __init__(self, decay_tracker: PerturbationDecayTracker):
        self.decay_tracker = decay_tracker
    
    def generate_perturbation(self, current_config, best_config, iteration):
        import random
        import numpy as np
        
        factor = self.decay_tracker.compute_perturbation_factor(
            getattr(best_config, 'last_updated_iteration', 0)
        )
        
        param_weights = self.decay_tracker.get_param_weights(iteration)
        param = random.choices(list(param_weights.keys()),
                               weights=list(param_weights.values()))[0]
        
        changes = {}
        
        if param == 'LR':
            log_lr = np.log10(current_config.lr)
            perturbation_width = np.log10(factor)
            new_log_lr = log_lr + random.uniform(-perturbation_width, perturbation_width)
            new_log_lr = max(np.log10(1e-6), min(np.log10(1.0), new_log_lr))
            changes['LR'] = 10.0 ** new_log_lr
        
        elif param == 'BATCH_SIZE':
            log2_bs = np.log2(current_config.batch_size)
            max_shift = max(1, int(abs(np.log2(factor)) + 0.5))
            shift = random.randint(-max_shift, max_shift)
            new_log2_bs = int(log2_bs + shift)
            new_log2_bs = max(2, min(10, new_log2_bs))
            changes['BATCH_SIZE'] = 2 ** new_log2_bs
        
        elif param == 'OPTIMIZER':
            decay_factor = self.decay_tracker.decay_rate ** (
                self.decay_tracker.global_iteration_count -
                getattr(best_config, 'last_updated_iteration', 0)
            )
            if decay_factor > 0.5:
                changes.update(self._conservative_optimizer_perturb(current_config))
            else:
                changes.update(self._exploratory_optimizer_perturb(current_config))
        
        return changes
    
    def _conservative_optimizer_perturb(self, current_config):
        import random
        changes = {}
        opt = current_config.optimizer
        if opt in ('adam', 'adamw'):
            alternatives = {'adam': ['adamw', 'adam_amsgrad'],
                            'adamw': ['adam', 'adam_amsgrad']}
            changes['OPTIMIZER'] = random.choice(alternatives.get(opt, [opt]))
        elif opt == 'sgd':
            changes['OPTIMIZER'] = 'sgd_momentum'
        return changes
    
    def _exploratory_optimizer_perturb(self, current_config):
        import random
        all_opts = ['adam', 'adamw', 'sgd', 'adagrad', 'rmsprop']
        alternatives = [o for o in all_opts if o != current_config.optimizer]
        return {'OPTIMIZER': random.choice(alternatives)}


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
        class HyperparameterPerturbationEngine:
            def __init__(self, config: dict, recent_changes: dict):
                self.config = config
                self.recent_changes = recent_changes

            def generate_perturbation(self, current_config, best_config,
                                      iteration: int) -> dict:
                import random
                import numpy as np

                strategy = random.choices(
                    ['large_jump', 'significant_perturb', 'neglected_param'],
                    weights=[0.3, 0.4, 0.3]
                )[0]

                changes = {}

                if strategy == 'large_jump':
                    param = random.choice(list(self.config.keys()))
                    if param == 'LR':
                        changes['LR'] = self._log_uniform_sample(
                            current_config.lr / 10,
                            current_config.lr * 10
                        )
                    elif param == 'BATCH_SIZE':
                        factors = [0.125, 0.25, 0.5, 2.0, 4.0, 8.0]
                        factor = random.choice(factors)
                        new_bs = int(current_config.batch_size * factor)
                        changes['BATCH_SIZE'] = 2 ** int(round(np.log2(new_bs)))
                    elif param == 'OPTIMIZER':
                        current_opt = current_config.optimizer
                        alternatives = [opt for opt in self.config['OPTIMIZER']['alternatives']
                                        if opt != current_opt]
                        changes['OPTIMIZER'] = random.choice(alternatives)

                elif strategy == 'significant_perturb':
                    param = random.choice(['LR', 'BATCH_SIZE'])
                    if param == 'LR':
                        factor = random.choice([0.1, 0.2, 0.33, 3.0, 5.0, 10.0])
                        changes['LR'] = current_config.lr * factor
                    else:
                        factors = [0.25, 0.5, 2.0, 4.0]
                        factor = random.choice(factors)
                        new_bs = int(current_config.batch_size * factor)
                        changes['BATCH_SIZE'] = 2 ** int(round(np.log2(new_bs)))

                else:  # neglected_param
                    param_ages = {}
                    for param in self.config.keys():
                        if param in ['random_seed', 'OPTIMIZER']:
                            continue
                        history = self.recent_changes.get(param, [])
                        if not history:
                            param_ages[param] = 999
                        else:
                            param_ages[param] = iteration - max(history)

                    neglected = max(param_ages, key=param_ages.get)

                    if neglected == 'LR':
                        changes['LR'] = current_config.lr * random.choice([0.2, 0.33, 3.0, 5.0])
                    else:
                        changes['BATCH_SIZE'] = 2 ** int(round(np.log2(
                            current_config.batch_size * random.choice([0.25, 0.5, 2.0, 4.0])
                        )))

                return changes

            def _log_uniform_sample(self, low, high):
                import numpy as np
                log_low = np.log(low)
                log_high = np.log(high)
                return np.exp(np.random.uniform(log_low, log_high))

        self.perturbation_config = {
            'LR': {'min_factor': 0.1, 'max_factor': 10.0},
            'BATCH_SIZE': {
                'min_factor': 0.25,
                'max_factor': 4.0
            },
            'OPTIMIZER': {
                'alternatives': ['AdamW', 'Adam', 'SGD'],
                'must_switch': False
            },
            'random_seed': {
                'strategy': 'log_uniform',
                're_seed_probability': 0.3
            }
        }
        self.recent_changes = {'LR': [], 'BATCH_SIZE': [], 'OPTIMIZER': []}
        self.iteration_counter = 0

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
