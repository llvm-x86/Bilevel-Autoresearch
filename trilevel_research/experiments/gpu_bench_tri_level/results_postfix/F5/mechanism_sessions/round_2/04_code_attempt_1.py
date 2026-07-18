import random
import logging
from copy import deepcopy as deepest_copy
from typing import Optional

class StagnationRestartController:
    def __init__(
        self,
        window_size: int = 5,
        stagnation_threshold_bpb: float = 0.001,
        top_k: int = 3,
        perturbation_std: float = 0.01,
        restart_cooldown: int = 3,
    ) -> None:
        self.window_size = window_size
        self.stagnation_threshold_bpb = stagnation_threshold_bpb
        self.top_k = top_k
        self.perturbation_std = perturbation_std
        self.restart_cooldown = restart_cooldown

        self.best_bpb_history: list[float] = []
        self.best_configs: list[tuple[float, 'GpuBenchConfig']] = []
        self.restart_count: int = 0
        self.last_restart_iteration: int = -restart_cooldown
        self.stagnant_iterations: int = 0
        self.restart_active: bool = False

    def update(self, iteration: int, config: 'GpuBenchConfig', val_bpb: float) -> bool:
        if not self.best_bpb_history or val_bpb < self.best_bpb_history[-1]:
            self.best_bpb_history.append(val_bpb)
        else:
            self.best_bpb_history.append(self.best_bpb_history[-1])

        if len(self.best_bpb_history) > self.window_size:
            self.best_bpb_history.pop(0)

        self.best_configs.append((val_bpb, config))
        self.best_configs.sort(key=lambda x: x[0])
        self.best_configs = self.best_configs[:self.top_k]

        if iteration - self.last_restart_iteration < self.restart_cooldown:
            return False

        if len(self.best_bpb_history) < self.window_size:
            return False

        improvement = self.best_bpb_history[-1] - self.best_bpb_history[0]
        if improvement < self.stagnation_threshold_bpb:
            self.stagnant_iterations += 1
            if self.stagnant_iterations >= 2:
                self.restart_count += 1
                self.last_restart_iteration = iteration
                self.stagnant_iterations = 0
                return True
        else:
            self.stagnant_iterations = 0

        return False

    def generate_restart_config(self) -> 'GpuBenchConfig':
        if not self.best_configs:
            from domains.gpu_bench_opt.runner import GpuBenchConfig
            return GpuBenchConfig()

        best_val, best_config = self.best_configs[0]
        restart_config = deepest_copy(best_config)

        for param_name, param_value in vars(restart_config).items():
            if isinstance(param_value, float):
                noise = random.gauss(0, self.perturbation_std)
                new_value = param_value * (1.0 + noise)
                setattr(restart_config, param_name, new_value)

        return restart_config