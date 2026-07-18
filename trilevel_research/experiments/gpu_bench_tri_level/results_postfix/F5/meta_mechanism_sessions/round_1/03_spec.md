## Implementation Specification for Mechanism Research Hypothesis 3: Staged Import Validation

1. **Patch name**: `staged_import_validation_researcher`

2. **Implementation strategy**: `new_helper_class` (with modification to `GpuBenchMechanismResearcher.explore` and `specify` methods)

3. **Target**: `GpuBenchMechanismResearcher` class in `mechanism_research.py`

4. **Step-by-step logic**:

   a. Create a new dataclass `ImportValidationResult` with fields: `module_name: str`, `is_importable: bool`, `error_message: Optional[str]`, and `module_path: Optional[Path]`.

   b. Create a new helper class `RunnerImportValidator` with methods:
      - `validate_import(module_name: str, base_path: Path) -> ImportValidationResult`: Attempts to import a module by checking if it exists in the expected location under `base_path`, then tries `importlib.import_module`. Returns the validation result without crashing.
      - `validate_all_runner_dependencies(runner_path: Path) -> List[ImportValidationResult]`: Parses `runner.py` to extract all top-level `import` and `from ... import` statements, then validates each one using `validate_import`.
      - `format_validation_report(results: List[ImportValidationResult]) -> str`: Formats the results into a structured log message showing importable vs broken imports.

   c. Modify `GpuBenchMechanismResearcher.__init__` to accept an optional `import_validator` parameter (defaults to `RunnerImportValidator()`).

   d. Before `explore()` generates the trace summary, call `self.import_validator.validate_all_runner_dependencies(...)` on the runner path. Log the validation report as WARNING level. This becomes a structured field in the exploration context that can be injected into prompts.

   e. In the `EXPLORE_PROMPT` template, add a new section:
      ```
      ### Import validation report:
      {import_report}
      ```
      
   f. When `import_report` is empty or all imports valid, suppress the section entirely. When broken imports are found, they appear as actionable diagnostics in the prompt.

   g. Implement gracefully: if `validate_all_runner_dependencies` raises an unexpected error, catch it, log `"Import validation failed: {error}"` at ERROR level, and continue with an empty import report (fail-open, not fail-closed).

5. **Integration points**:

   - `analysis/gpu/mechanism_research.py`: New classes and init modification
   - `GpuBenchMechanismResearcher.explore()`: Injection point for report generation
   - `EXPLORE_PROMPT`: Template augmentation (field extraction and suppression logic)
   - No changes to `runner.py` or any other domain file

6. **Optional schedule patch** (JSON): None required — this runs synchronously during each `explore()` call as a cheap diagnostic step. If the import validation takes too long, consider caching results per file hash in a separate PR.