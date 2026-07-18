class ContextRollbackManager:
    """Manages versioned snapshots of importable state to enable safe rollback on import failure."""

    def __init__(self, max_rollbacks: int = 3):
        self._snapshots = []          # stack of saved state dicts
        self._rollback_count = 0
        self._max_rollbacks = max_rollbacks
        self._last_known_good = None  # version marker (timestamp or index)
        self._version = 0

    def save_state(self, importable_classes: dict) -> dict:
        """Deep-copy the importable_classes dict and push snapshot."""
        import copy
        snapshot = {
            'importable_classes': copy.deepcopy(importable_classes),
            'version': self._version,
        }
        self._snapshots.append(snapshot)
        self._last_known_good = snapshot['version']
        return snapshot

    def restore_state(self) -> dict | None:
        """Pop the last snapshot and return it, or None if no snapshots exist."""
        if not self._snapshots:
            return None
        snapshot = self._snapshots.pop()
        self._rollback_count += 1
        return snapshot

    def snapshot_available(self) -> bool:
        return len(self._snapshots) > 0

    def rollback_exceeded(self) -> bool:
        return self._rollback_count >= self._max_rollbacks

    def reset_rollback_count(self):
        self._rollback_count = 0

    def increment_version(self):
        self._version += 1


def _try_import_with_rollback(self, code: str, context: dict) -> dict | None:
    """Attempt to import generated module; revert to saved state on failure."""
    import copy
    # Save current context
    snapshot = self._rollback_manager.save_state(self.importable_classes)

    try:
        result = self.try_import_generated_module(code, context)
    except Exception as exc:
        logger.warning(
            "Rollback triggered: import failed for code hash %s with error: %s",
            hash(code), exc
        )
        restored = self._rollback_manager.restore_state()
        if restored is not None:
            self.importable_classes = copy.deepcopy(restored['importable_classes'])
        # Reraise so caller sees failure
        raise
    else:
        if result is not None:
            self._rollback_manager.reset_rollback_count()
            self._rollback_manager.save_state(self.importable_classes)
        return result
    # If import succeeded but result is None (e.g. no class extracted), keep snapshot
    finally:
        self._rollback_manager.increment_version()