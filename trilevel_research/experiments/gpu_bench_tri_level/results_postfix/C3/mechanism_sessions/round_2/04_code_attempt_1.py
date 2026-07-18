# Initialize multivariate perturbation explorer for continuous parameter tuning
        if not self.simple_mode:
            param_names = []
            param_ranges = {}
            if hasattr(self.config, 'learning_rate_range'):
                param_names.append('lr')
                param_ranges['lr'] = self.config.learning_rate_range
            if hasattr(self.config, 'lr_scaler_range'):
                param_names.append('lr_scaler')
                param_ranges['lr_scaler'] = self.config.lr_scaler_range
            if hasattr(self.config, 'batch_size_range'):
                param_names.append('batch_size')
                param_ranges['batch_size'] = self.config.batch_size_range
            if hasattr(self.config, 'hidden_dim_range'):
                param_names.append('hidden_dim')
                param_ranges['hidden_dim'] = self.config.hidden_dim_range
            
            self.multivariate_explorer = MultiVariatePerturbationExplorer(
                param_names=param_names,
                param_ranges=param_ranges
            )
        else:
            self.multivariate_explorer = None

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
        new_normalized = np.clip(new_normalized, 0, 1)
        return self._denormalize_config(new_normalized)
        
    def _propose_univariate_exploration(self, current_config):
        param = np.random.choice(self.param_names)
        lo, hi = self.param_ranges[param]
        current_val = current_config[param]
        step = (hi - lo) * np.random.uniform(-0.2, 0.2)
        new_val = np.clip(current_val + step, lo, hi)
        new_config = current_config.copy()
        new_config[param] = new_val
        return new_config