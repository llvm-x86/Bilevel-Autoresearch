# Adaptive perturbation controller initialization
        self.perturbation_controller = PerturbationController(
            base_magnitude=search_config.perturb_base_magnitude if hasattr(search_config, 'perturb_base_magnitude') else 0.1,
            min_magnitude=search_config.perturb_min_magnitude if hasattr(search_config, 'perturb_min_magnitude') else 0.01,
            max_magnitude=search_config.perturb_max_magnitude if hasattr(search_config, 'perturb_max_magnitude') else 0.5,
        )