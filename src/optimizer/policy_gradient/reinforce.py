"""REINFORCE objective for the Workflow Adapter (paper Section 3.3, Eq. 14-16).

    loss = - sum_t  log P_Theta(G_t | s~_t) * (G_t - cost_weight * C_t)
    log P_Theta(G_t | s~_t) = sum_l log P_Theta(i_t^(l) | s~_t)

``log_prob_mode`` selects how a layer's sampled set is scored:
* ``sequence`` (default, paper Eq. 28): exact log-probability of the ordered sample drawn
  without replacement under cumulative-mass soft sampling.
* ``mean``: mean of the selected agents' log-probabilities (a simpler surrogate).
"""

from typing import TYPE_CHECKING, List

import torch

from src.logger import logger
from src.workflow_adapter.sampling import sequence_log_prob

if TYPE_CHECKING:
    from src.optimizer.training.trajectory import Trajectory

LOG_PROB_MODES = ("sequence", "mean")


class ReinforceLoss:
    def __init__(self, log_prob_mode: str = "sequence", cost_weight: float = 0.0):
        if log_prob_mode not in LOG_PROB_MODES:
            raise ValueError(f"log_prob_mode must be one of {LOG_PROB_MODES}, got {log_prob_mode!r}")
        self.log_prob_mode = log_prob_mode
        self.cost_weight = cost_weight

    def _layer_log_prob(self, log_probs: torch.Tensor, indices: List[int]) -> torch.Tensor:
        if self.log_prob_mode == "sequence":
            return sequence_log_prob(log_probs, indices)
        log_probs_1d = log_probs[0] if log_probs.dim() > 1 else log_probs
        return log_probs_1d[indices].mean()

    def workflow_log_prob(self, trajectory: "Trajectory", step: int) -> torch.Tensor:
        """``log P_Theta(G_t | s~_t)`` summed over the workflow layers (Eq. 14)."""
        total = None
        for layer_log_probs, indices in zip(trajectory.log_probs[step], trajectory.sampled_indices[step]):
            if layer_log_probs is None or not indices:
                continue
            term = self._layer_log_prob(layer_log_probs, indices)
            total = term if total is None else total + term
        return total

    def trajectory_loss(self, trajectory: "Trajectory", returns: List[float]) -> torch.Tensor:
        if len(returns) != trajectory.num_steps:
            raise ValueError(
                f"Trajectory {trajectory.episode_id}: {trajectory.num_steps} steps but {len(returns)} returns."
            )
        costs = trajectory.costs if len(trajectory.costs) == trajectory.num_steps else [0.0] * trajectory.num_steps
        loss = None
        for t in range(trajectory.num_steps):
            log_prob = self.workflow_log_prob(trajectory, t)
            if log_prob is None:
                continue
            utility = returns[t] - self.cost_weight * costs[t]
            term = -log_prob * utility
            loss = term if loss is None else loss + term
        return loss

    def batch_loss(self, trajectories: List["Trajectory"], returns_batch: List[List[float]]) -> torch.Tensor:
        """Average REINFORCE loss over the non-empty trajectories of a batch."""
        losses = []
        for trajectory, returns in zip(trajectories, returns_batch):
            if trajectory.num_steps == 0 or not returns:
                continue
            loss = self.trajectory_loss(trajectory, returns)
            if loss is not None:
                losses.append(loss)
        if not losses:
            logger.warning("No valid trajectories in this batch; skipping the policy update.")
            return None
        return torch.stack(losses).mean()
