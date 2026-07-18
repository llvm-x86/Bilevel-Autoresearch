@dataclass
class CrossoverConfig:
    """Configuration for crossover search space expansion."""
    enabled: bool = True
    pool_size: int = 4
    crossover_population: int = 2
    mutation_rate: float = 0.2
    scale_factor: float = 1.5
    max_model_params: int = 100_000_000


class CrossoverPool:
    """
    Maintains a pool of elite configurations and generates new
    configurations via crossover (mixing parameters from two parents).
    """
    def __init__(self, config: CrossoverConfig):
        self.config = config
        self.elite_pool: list[tuple[float, 'GpuBenchConfig']] = []

    def add_result(self, val_bpb: float, config: 'GpuBenchConfig') -> None:
        self.elite_pool.append((val_bpb, config))
        self.elite_pool.sort(key=lambda x: x[0])
        self.elite_pool = self.elite_pool[:self.config.pool_size]

    def generate_crossover_configs(self, base_config: 'GpuBenchConfig') -> list['GpuBenchConfig']:
        if len(self.elite_pool) < 2:
            return []
        new_configs = []
        elite_configs = [c for _, c in self.elite_pool]
        for _ in range(self.config.crossover_population):
            parent1 = random.choice(elite_configs)
            if random.random() < 0.5 and len(elite_configs) > 1:
                parent2 = random.choice([c for c in elite_configs if c is not parent1])
            else:
                parent2 = base_config
            child = self._crossover(parent1, parent2)
            new_configs.append(child)
        return new_configs

    def _crossover(self, parent1: 'GpuBenchConfig', parent2: 'GpuBenchConfig') -> 'GpuBenchConfig':
        child = copy.deepcopy(parent1)
        crossover_params = ['N_LAYER', 'N_HEAD', 'N_EMBD', 'BLOCK_SIZE', 'VOCAB_SIZE', 'N_EMBD_HEAD', 'COMPILE']
        p1d = parent1.to_dict()
        p2d = parent2.to_dict()
        cd = child.to_dict()
        for p in crossover_params:
            if p in p1d and p in p2d:
                if random.random() < 0.5:
                    cd[p] = p2d[p]
        if random.random() < self.config.mutation_rate:
            for sp in ['N_LAYER', 'N_EMBD']:
                if sp in cd and random.random() < 0.3:
                    cd[sp] = int(cd[sp] * self.config.scale_factor)
        cd['N_LAYER'] = min(cd.get('N_LAYER', 12), 48)
        cd['N_EMBD'] = min(cd.get('N_EMBD', 768), 2048)
        child = GpuBenchConfig.from_dict(cd, check_constraints=False)
        total_params = child.total_params()
        if total_params > self.config.max_model_params:
            scale = self.config.max_model_params / total_params
            cd['N_LAYER'] = max(4, int(cd['N_LAYER'] * scale ** 0.5))
            cd['N_EMBD'] = max(64, int(cd['N_EMBD'] * scale ** 0.5))
            child = GpuBenchConfig.from_dict(cd, check_constraints=False)
        return child