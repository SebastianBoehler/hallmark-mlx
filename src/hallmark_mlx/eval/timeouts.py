"""Thin timeout helpers for long-running benchmark calls."""

from __future__ import annotations

import signal
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")


class EvaluationTimeoutError(TimeoutError):
    """Raised when a benchmark call exceeds its outer time budget."""


@contextmanager
def _alarm_timeout(seconds: int) -> Iterator[None]:
    if seconds <= 0:
        yield
        return

    def _raise_timeout(signum: int, frame: object) -> None:
        raise EvaluationTimeoutError(f"Timed out after {seconds} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def run_with_timeout(seconds: int, fn: Callable[[], T]) -> T:
    """Run ``fn`` with a wall-clock timeout on Unix-like systems."""

    with _alarm_timeout(seconds):
        return fn()
