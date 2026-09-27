from __future__ import annotations

from collections.abc import Callable

ProgressCallback = Callable[[float | None, str], None]


class GenerationProgress:
    """One progress timeline across loading, inference, finishing, and saving."""

    def __init__(self, callback: ProgressCallback | None = None) -> None:
        self._callback = callback

    def update(self, fraction: float | None, message: str) -> None:
        if self._callback is not None:
            self._callback(None if fraction is None else max(0.0, min(1.0, fraction)), message)

    def inference(self, output_index: int, count: int, step: int, steps: int) -> None:
        done = (output_index + step / max(1, steps)) / max(1, count)
        self.update(
            0.15 + 0.7 * done,
            f"Generating image {output_index + 1}/{count} · step {step}/{steps}",
        )