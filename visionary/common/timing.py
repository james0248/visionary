import contextlib
import logging
import time


class PhaseTimer:
    """
    Accumulate wall time per labeled phase via `with timer("phase"):`.
    flush(steps) returns per-step averages plus "wall" and "sps", then resets.
    """

    def __init__(self):
        self._acc: dict[str, float] = {}

    @contextlib.contextmanager
    def __call__(self, phase: str):
        start = time.monotonic()
        try:
            yield
        finally:
            self._acc[phase] = self._acc.get(phase, 0.0) + time.monotonic() - start

    def flush(self, steps: int) -> dict[str, float]:
        steps = max(int(steps), 1)
        total = sum(self._acc.values())
        stats = {phase: acc / steps for phase, acc in self._acc.items()}
        stats["wall"] = total / steps
        stats["sps"] = steps / total if total else 0.0
        self._acc = {}
        return stats

    def log(self, logger: logging.Logger, step: int, steps: int, **extra: float) -> dict[str, float]:
        stats = self.flush(steps)
        parts = ", ".join(f"{k}: {v:.2f}" if k == "sps" else f"{k}: {v:.3f}s" for k, v in {**stats, **extra}.items())
        logger.info("Step %d - %s", step, parts)
        return stats
