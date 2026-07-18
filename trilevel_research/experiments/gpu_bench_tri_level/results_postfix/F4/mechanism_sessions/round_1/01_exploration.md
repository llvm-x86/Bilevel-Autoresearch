Based on the trace, the runner is stuck in a local optimum around val_bpb=0.017 with LR=0.005, WD=0.0001, BS=512. The proposals are too conservative and fail to escape the basin. Here are four mechanism changes:

---

### 1. **Domain**: Bayesian Optimization (BO) / Gaussian Process-based sampling

**Core idea**: Replace random perturbation with a lightweight Gaussian Process (GP) surrogate model to propose hyperparameter configurations that balance exploration (high uncertainty) and exploitation (low predicted val_bpb).

**Implementation target**: Modify `HyperparamSearcher.perturb()` or add a new method `propose_via_gp()` in `runner.py`.

**Implementation sketch**:
- Maintain a history of evaluated configs and their val_bpb scores.
- Fit a simple GP (e.g., using `sklearn.gaussian_process.GaussianProcessRegressor` with Matern kernel) on the last ~20-50 evaluations.
- At each iteration, use Expected Improvement (EI) or Upper Confidence Bound (UCB) as acquisition function to propose the next config.
- Fall back to random perturbation if GP fitting fails or returns NaN.

**Why it helps**: The current random perturbations are blind — they don't leverage the information from previous evaluations. A GP model captures the response surface (e.g., LR vs BS vs WD trade-offs) and can propose configurations that are likely to improve, even if they differ significantly from the current best. This directly addresses the "too many discards" bottleneck.

**Implementation complexity**: 4 (requires adding sklearn dependency or a lightweight GP implementation, maintaining a history buffer, and implementing acquisition function logic).

**Risk**: Medium — GP can be slow to fit on many points, and its performance depends on kernel choice. If the response surface is non-smooth, GP might propose degenerate points. Fallback to random ensures safety.

---

### 2. **Domain**: Simulated Annealing / Temperature-based exploration

**Core idea**: Introduce a "temperature" parameter that controls the magnitude of hyperparameter perturbations, decaying over time to transition from broad exploration to fine exploitation.

**Implementation target**: Modify `HyperparamSearcher.__init__()` and `perturb()` to accept/expose a `temperature` parameter.

**Implementation sketch**:
- In `__init__`, set `self.temperature = 1.0` initial.
- After each evaluation, decay `self.temperature *= 0.95` (or adaptive decay based on consecutive discards).
- In `perturb()`, scale perturbation amount by `self.temperature`:
  - Multiplicative parameters (LR, WD): multiply by `random.uniform(1-temp*0.5, 1+temp*0.5)`.
  - Categorical parameters (batch size): wider sampling range when temp is high.
- Reset temperature when a new best is found to re-energize exploration.

**Why it helps**: The current perturbations are constant over time. The runner often explores too locally (small LR changes) and gets stuck. Temperature decay forces the runner to try bolder moves early on (e.g., trying LR=0.01 or BS=1024) and gradually focus on fine-tuning as iterations progress. This mimics the successful strategy of "warm-up then fine-tune" seen in many optimization pipelines.

**Implementation complexity**: 2 (simple parameter addition, decay logic, and perturbation scaling).

**Risk**: Low — temperature decay is a well-known strategy. The worst case is slower convergence if decay is too fast, but adaptive reset mitigates this.

---

### 3. **Domain**: Evolutionary Strategies / Population-based search

**Core idea**: Maintain a population of the top-K configurations (e.g., K=5) instead of a single best, and propose new configs by crossover and mutation from the population.

**Implementation target**: Modify `GpuBenchRunner._inner_loop()` to manage a population of `(config, val_bpb)` entries, and replace the single "perturb from best" logic with "recombine two random elites".

**Implementation sketch**:
- Keep `self.elite_population = []` sorted by val_bpb (size ≤ K).
- After each completed evaluation, if val_bpb is better than the worst elite, insert it (evict worst).
- To propose a new config:
  1. Select two parents randomly from the top 3 elites.
  2. For each hyperparameter, with probability 0.5 take parent1's value, else parent2's; with 0.1 probability mutate (random perturbation).
  3. Ensure batch size remains a power of 2.
- Continue evaluating and updating the population.

**Why it helps**: The single-best strategy leads to premature convergence — all proposals are near the current best. A population maintains diversity (e.g., one elite may use BS=512, another BS=128, another LR=0.003). Crossover recombines promising traits, e.g., combining LR from elite1 with BS from elite2 can produce a config that outperforms either parent. This directly counters the stagnation seen after iteration 17.

**Implementation complexity**: 3 (requires population management, crossover logic, and handling of mixed parameter types). No external dependencies.

**Risk**: Medium — if crossover produces many invalid configs (e.g., batch size not divisible by 32), the runner may waste iterations. Need careful constraints handling. Also, population might homogenize over time; mutation helps but may not be enough.

---

### 4. **Domain**: Gradient-free optimization with Lévy flight / heavy-tailed perturbations

**Core idea**: Replace uniform perturbations with heavy-tailed (e.g., Cauchy or Lévy) distributions that occasionally take large jumps, enabling escape from local optima while still performing local search most of the time.

**Implementation target**: Modify the perturbation generation in `HyperparamSearcher.perturb()`.

**Implementation sketch**:
- For each continuous hyperparameter (LR, WD):
  - Sample `delta` from Cauchy(0, gamma) where gamma = 0.2 (scale parameter).
  - New value = old_value * exp(delta)  [multiplicative perturbation].
- For categorical (batch size):
  - With probability 0.8: sample from `bs_candidates = [32, 64, 128, 256, 512, 1024]` uniformly.
  - With probability 0.2: keep current.
- Optionally, adapt `gamma` based on recent improvement rate (increase gamma if many discards in a row).

**Why it helps**: The current perturb uses small, bounded changes (e.g., ±0.001 on LR). Cauchy-distributed perturbations have infinite variance — they produce small changes most of the time but occasionally large jumps. This is critical when the runner is stuck at a local optimum: a large LR jump (e.g., from 0.005 to 0.02) can move it to a different loss basin. This is inspired by the success of Lévy flights in evolutionary optimization for rugged landscapes.

**Implementation complexity**: 1 (just change the sampling distribution). No new data structures.

**Risk**: Low — Cauchy perturbation is a simple drop-in replacement. If it causes instability, revert to uniform. The occasional large jump may waste an iteration, but the cost is low (each iteration is fast on GPU).

---

### Priority recommendation:

1. **Implement #2 (Simulated Annealing)** first — simplest, lowest risk, immediate effect on exploration.
2. If still stuck, add **#4 (Lévy flight perturbations)** on top — tiny code change, can be combined with #2.
3. If rich evaluation budget (>50 iterations), implement **#1 (GP search)** for principled exploitation.
4. Consider **#3 (population crossover)** only if batch size and LR combinations are highly interdependent (suggests by trace: BS=512 works, BS=256 fails — hidden interaction).