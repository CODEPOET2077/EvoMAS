"""Optional SwanLab experiment tracking for Workflow Adapter training.

If ``swanlab`` is not installed or tracking is disabled, every call is a no-op.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import swanlab

    SWANLAB_AVAILABLE = True
except ImportError:
    swanlab = None
    SWANLAB_AVAILABLE = False


class SwanLabLogger:
    """Thin wrapper around ``swanlab.init / log / finish`` with graceful degradation.

    Args:
        enabled: Turn tracking on or off.
        run_id: Resume an existing SwanLab experiment with this id (``resume='allow'``).
        **init_kwargs: Forwarded to ``swanlab.init`` (``project``, ``experiment_name``, ``config`` ...).
    """

    def __init__(self, enabled: bool = True, run_id: Optional[str] = None, **init_kwargs):
        self.enabled = enabled and SWANLAB_AVAILABLE
        self.run_id = run_id
        if not self.enabled:
            if enabled and not SWANLAB_AVAILABLE:
                logger.warning("SwanLab is not installed (pip install swanlab); tracking disabled.")
            return

        params = {"project": "EvoMAS", "description": "REINFORCE training of the Workflow Adapter", **init_kwargs}
        if run_id:
            params.update(id=run_id, resume="allow")
        try:
            run = swanlab.init(**params)
            if self.run_id is None:
                self.run_id = getattr(run, "id", None)
        except Exception as e:
            logger.error(f"Failed to initialize SwanLab: {e}")
            self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        try:
            swanlab.log(metrics, step=step) if step is not None else swanlab.log(metrics)
        except Exception as e:
            logger.warning(f"Failed to log metrics to SwanLab: {e}")

    def log_config(self, config: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            swanlab.config.update(config)
        except Exception as e:
            logger.warning(f"Failed to log config to SwanLab: {e}")

    def get_run_id(self) -> Optional[str]:
        return self.run_id if self.enabled else None

    def finish(self) -> None:
        if not self.enabled:
            return
        try:
            swanlab.finish()
        except Exception as e:
            logger.warning(f"Failed to finish SwanLab run: {e}")


_global_logger: Optional[SwanLabLogger] = None


def init_logger(enabled: bool = True, **init_kwargs) -> SwanLabLogger:
    """Create the process-wide experiment logger."""
    global _global_logger
    _global_logger = SwanLabLogger(enabled=enabled, **init_kwargs)
    return _global_logger


def get_logger() -> Optional[SwanLabLogger]:
    return _global_logger


__all__ = ["SwanLabLogger", "init_logger", "get_logger", "SWANLAB_AVAILABLE"]
