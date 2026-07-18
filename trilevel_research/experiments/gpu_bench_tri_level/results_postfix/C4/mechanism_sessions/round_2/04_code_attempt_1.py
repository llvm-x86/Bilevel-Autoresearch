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