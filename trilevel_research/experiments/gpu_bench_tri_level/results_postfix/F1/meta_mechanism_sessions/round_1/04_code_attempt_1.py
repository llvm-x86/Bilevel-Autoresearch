def run_mechanism_iteration(
    self,
    researcher: BaseMechanismResearcher,
    iteration_num: int,
    config: dict,
) -> MechanismResult:
    """Run a single iteration with corrected error classification.

    Changes: Any exception from researcher.run_iteration() is classified
    as a RUNTIME failure (not import failure), except for syntax errors
    which are genuinely import-level failures.
    """
    try:
        result = researcher.run_iteration(iteration_num=iteration_num, config=config)
        return result
    except SyntaxError as e:
        # Genuine import failure - syntax error in module-level code
        self.logger.warning(
            f"Iteration {iteration_num}: import failure (syntax error) in researcher code: {e}"
        )
        return MechanismResult(
            success=False,
            failure_reason="import_failure",
            details={"error": str(e), "iteration": iteration_num}
        )
    except Exception as e:
        # Runtime failure - any other exception from run_iteration
        # This prevents false "import failure" rate inflation
        self.logger.warning(
            f"Iteration {iteration_num}: runtime failure in researcher code: {e}"
        )
        return MechanismResult(
            success=False,
            failure_reason="runtime_failure",
            details={"error": str(e), "iteration": iteration_num}
        )