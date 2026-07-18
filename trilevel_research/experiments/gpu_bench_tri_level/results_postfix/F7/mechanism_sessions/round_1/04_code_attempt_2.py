class DeduplicateProposals:
    """Helper class for deduplicating LLM-proposed changes in GpuBenchRunner."""

    def __init__(self, trace):
        self.trace = trace

    def deduplicate(self, proposed_changes: dict, iteration: int) -> dict:
        """
        Remove changes that have been tried before with similar or worse results.

        Args:
            proposed_changes: dict of {param: value} from LLM proposal
            iteration: current iteration number for logging

        Returns:
            Filtered dict with duplicates removed, or empty dict if all are duplicates
        """
        if not proposed_changes or not self.trace.results:
            return proposed_changes

        filtered_changes = {}
        rejected_params = []

        for param, value in proposed_changes.items():
            is_duplicate = False
            for result in self.trace.results:
                if result.changes and param in result.changes:
                    existing_value = result.changes[param]
                    try:
                        if float(existing_value) == float(value):
                            is_duplicate = True
                            break
                    except (ValueError, TypeError):
                        if str(existing_value) == str(value):
                            is_duplicate = True
                            break

            if is_duplicate:
                rejected_params.append(param)
            else:
                filtered_changes[param] = value

        if rejected_params:
            import logging
            logging.info(
                f"Iteration {iteration}: Rejected duplicate proposals: "
                f"{', '.join(rejected_params)}"
            )

        return filtered_changes

    def is_all_duplicates(self, proposed_changes: dict, iteration: int) -> bool:
        """Check if all proposed changes are duplicates (returns True if all rejected)."""
        if not proposed_changes or not self.trace.results:
            return False
        deduped = self.deduplicate(proposed_changes, iteration)
        return not deduped