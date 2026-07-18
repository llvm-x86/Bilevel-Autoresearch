1. **Mechanism name** (snake_case): `dedup_config_proposal`

2. **Implementation strategy**: `new_helper_class` + `modify_init`

3. **Target**: `GpuBenchRunner.__init__` and new class `ConfigDedupValidator`

4. **Step-by-step logic**:

   **Step 1: Create new helper class `ConfigDedupValidator`**
   - Constructor takes `ConfigDedupParams` with fields:
     - `min_hamming_distance: int = 2` (at least 2 hyperparams must differ)
     - `param_rounding: dict[str, float]` = `{'LR': 0.001, 'WEIGHT_DECAY': 1e-6, 'BATCH_SIZE': 8, 'HIDDEN_DIM': 128}` (quantization/precision floor)
     - `jitter_std: float = 0.05` (fraction of parameter range for random jitter fallback)
   - Internal state:
     - `self._seen_configs: list[dict[str, float]]` — stores normalized (rounded) param dicts
   
   **Step 2: Implement `normalize_config(config: dict) -> dict`**
   - Apply rounding: for each param key, `round(value / param_rounding[key]) * param_rounding[key]`
   - Return a sorted tuple of (key, rounded_value) for deterministic comparison
   
   **Step 3: Implement `is_novel(self, config: dict) -> tuple[bool, dict]`**
   - Normalize the proposed config
   - Compare against all stored normalized configs using Hamming distance:
     - Hamming distance = count of params where `abs(current_val - stored_val) > param_rounding[key]` (i.e., the values differ after rounding)
   - If distance < `min_hamming_distance` for any stored config → return `(False, None)`
   - Otherwise → add normalized version to `self._seen_configs`, return `(True, normalized_config)`
   
   **Step 4: Implement `fallback_jitter(self, best_config: GpuBenchConfig) -> dict`**
   - Take `best_config` (the current best `GpuBenchConfig` that corresponds to `self.trace.best_config`)
   - For each active param, generate a random perturbation:
     - `LR`: `best_config.lr * (1 + np.random.normal(0, jitter_std))`
     - `WEIGHT_DECAY`: `best_config.weight_decay * (1 + np.random.normal(0, jitter_std))`
     - `BATCH_SIZE`: `best_config.batch_size * (1 + np.random.normal(0, jitter_std))` → round to multiple of 8
     - `HIDDEN_DIM`: choose randomly from `[128, 256, 384, 512]` if current best not already tried
   - Return the perturbed config dictionary

   **Step 5: Modify `GpuBenchRunner.__init__`**
   - Add: `self.dedup_validator = ConfigDedupValidator()`
   - Import `ConfigDedupParams` at top if not already present; can be a simple dataclass or inline dict

   **Step 6: Modify `_propose()` method (or add a wrapper around it)**
   - After LLM proposes `changes` dict:
     ```python
     novel, normalized = self.dedup_validator.is_novel(changes)
     if not novel:
         # Fallback: generate jitter from best config
         changes = self.dedup_validator.fallback_jitter(self.trace.best_config)
         hypothesis = f"dedup_fallback_jitter_from_iter_{self.trace.best_iteration}"
         # Re-check novelty after jitter (to avoid infinite loop — if still not novel, just force it)
         novel2, _ = self.dedup_validator.is_novel(changes)
         if not novel2:
             # Force acceptance anyway (worst-case: redundant run, but rare)
             pass
     ```
   - Ensure `is_novel` is called *after* rounding but *before* we record the config as "seen" — the recording happens inside `is_novel` if novel.

   **Step 7: Integration with `run_iteration()`**
   - No changes needed in `run_iteration()` itself. The dedup check runs inside `_propose()` before the config is returned.
   - The `filtered` changes dict will be the deduplicated/novel one, so the rest of `run_iteration()` remains untouched.

   **Step 8: Edge-case handling**
   - If `self.trace.best_config` is not yet set (baseline not run), fallback jitter uses the default `GpuBenchConfig()`.
   - If no active params are in the proposal after dedup, the existing `if not filtered: return discard` handles it gracefully.
   - The dedup validator is reset each run (not persisted across runs) since `GpuBenchRunner` is instantiated fresh per optimization session.

5. **Integration points**:
   - **New import**: `ConfigDedupValidator`, `ConfigDedupParams` (or defined within file)
   - **Modified method**: `GpuBenchRunner.__init__` — add `self.dedup_validator = ...`
   - **Modified internal flow**: `_propose()` method — add dedup check and fallback jitter between LLM proposal generation and return
   - **No changes to**: `run_baseline()`, `run_iteration()`, `_run_trial()`, or any of the `GpuBenchConfig` fields
   - **Configurable**: `min_hamming_distance` and `param_rounding` can be exposed via `SearchConfig` if desired, but default to sensible values as given in Step 1