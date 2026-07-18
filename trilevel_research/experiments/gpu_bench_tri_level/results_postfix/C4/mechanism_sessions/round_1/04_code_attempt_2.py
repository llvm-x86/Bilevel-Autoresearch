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