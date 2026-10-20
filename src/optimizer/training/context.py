"""Async-safe context that exposes the active trajectory buffer to the executor tool.

``contextvars`` (rather than thread-locals) keeps concurrent rollouts launched with
``asyncio.gather`` isolated from each other.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.optimizer.training.trajectory import TrajectoryCollector

_collector_var: "ContextVar[Optional[TrajectoryCollector]]" = ContextVar("trajectory_collector", default=None)


class TrajectoryContext:
    @classmethod
    def get_collector(cls) -> Optional["TrajectoryCollector"]:
        """Return the collector of the current async context (``None`` outside training rollouts)."""
        return _collector_var.get()

    @classmethod
    def set_collector(cls, collector: "TrajectoryCollector") -> None:
        _collector_var.set(collector)

    @classmethod
    def clear_collector(cls) -> None:
        _collector_var.set(None)


@contextmanager
def trajectory_collection_context(collector: "TrajectoryCollector"):
    """Within this context, every ``executor_tool`` call records its decision into ``collector``.

    Example:
        >>> with trajectory_collection_context(collector):
        ...     await planner.run(task)
    """
    TrajectoryContext.set_collector(collector)
    try:
        yield
    finally:
        TrajectoryContext.clear_collector()
