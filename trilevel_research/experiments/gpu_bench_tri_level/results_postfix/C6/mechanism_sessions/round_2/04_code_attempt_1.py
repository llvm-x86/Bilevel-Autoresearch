from typing import Any

class MomentumTracker:
    """Tracks momentum across iterations to guide LLM proposals."""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.direction: dict[str, int] = {}          # +1 or -1 per parameter
        self.strength: dict[str, int] = {}           # consecutive improvements
        self._prev_changes: dict[str, int] = {}      # last direction per param

    def update(self, changes: dict[str, int], accepted: bool) -> None:
        """Update momentum based on whether the iteration was accepted."""
        if accepted:
            for param, change in changes.items():
                prev = self._prev_changes.get(param)
                if prev is not None and change == prev:
                    self.strength[param] = self.strength.get(param, 0) + 1
                else:
                    self.strength[param] = 1
                self.direction[param] = change
                self._prev_changes[param] = change
        else:
            # Reset consecutive counts for parameters that changed
            for param in changes:
                self.strength[param] = 0
            # Keep direction for potential contradiction logging

    def get_hints(self, min_strength: int = 2) -> dict[str, tuple[int, int]]:
        """Return {param: (direction, strength)} for strong momentum."""
        return {
            p: (self.direction[p], self.strength[p])
            for p in self.strength
            if self.strength[p] >= min_strength
        }

    def compose_prompt(self, param: str, direction: int, strength: int) -> str:
        """Return a prompt snippet for a single parameter."""
        dir_word = "increase" if direction > 0 else "decrease"
        return (
            f"The previous {strength} improvements used {dir_word} for {param}. "
            f"Continue in the same direction with a similar magnitude."
        )