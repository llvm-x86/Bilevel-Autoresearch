Based on the trace analysis, I can see that the search is getting stuck in a local optimum around HIDDEN_DIM=1024 with LR=0.003, and most perturbations away from this configuration lead to significantly worse val_bpb. Here are my proposed mechanisms:

## Hypothesis 1: **Adaptive Perturbation Decay**

**Domain**: Simulated Annealing / Bayesian Optimization  
**Core idea**: Gradually reduce perturbation magnitude over iterations to focus search radius around the best configuration as exploitation increases.  
**Implementation target**: `GpuBenchRunner._generate_candidate_config()` (new method) that scales perturbation standard deviation by exp(-k * (iter - best_iter)) where k=0.15.  
**Why it helps**: Currently, large perturbations from iter 15's best config (e.g., reducing HIDDEN_DIM from 1024 to 512 in iter 16) cause catastrophic val_bpb degradation. Focusing search radius prevents wasteful exploration far from the optimum.  
**Complexity**: 2  
**Risk**: low  

## Hypothesis 2: **Directional Gradient-Guided Perturbation**

**Domain**: Gradient-based optimization (Adam optimizer concept)  
**Core idea**: Track which hyperparameter changes previously improved val_bpb and bias future perturbations in those directions using momentum.  
**Implementation target**: `GpuBenchRunner._update_direction_momentum()` (new method) maintaining exponential moving average of successful delta vectors, used to skew random perturbation selection.  
**Why it helps**: The trace shows HIDDEN_DIM increases consistently improved val_bpb (768→0.9252, 1024→0.0052), while LR changes were harmful. The runner should learn to preferentially increase HIDDEN_DIM rather than randomly trying LR, batch_size, or weight_decay.  
**Complexity**: 4  
**Risk**: medium  

## Hypothesis 3: **Exploration Budget with Forced Random Restarts**

**Domain**: Evolutionary Strategies / Population-based training  
**Core idea**: After N consecutive discards (currently ~71% discard rate), force a small random restart with parameters significantly different from the current best to escape local minima.  
**Implementation target**: `GpuBenchRunner._check_exploration_stagnation()` (new method) monitoring consecutive discard streak, triggering `_generate_restart_config()` when threshold > 5.  
**Why it helps**: The system is stuck—iter 16-21 all discard despite only small perturbations. A restart at distinct LR (e.g., 0.001) and HIDDEN_DIM (e.g., 512) might discover alternative basins, especially given the model seems sensitive to capacity.  
**Complexity**: 3  
**Risk**: low