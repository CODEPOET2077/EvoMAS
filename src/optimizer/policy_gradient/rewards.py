"""Per-step returns used by the policy-gradient update.

* ``terminal`` (main setting, paper Section 3.3 / Eq. 17): every meta-step receives the sparse,
  verifiable terminal utility ``U(tau)``. Evaluator feedback only enters the task state.
* ``process`` (PRM variant, paper Section 4.4): Evaluator rewards act as dense intermediate
  rewards and the terminal utility is the reward of the final step; returns are discounted,
  ``G_t = r_t + gamma * G_{t+1}``. Useful when terminal success is extremely sparse.
"""

from typing import TYPE_CHECKING, List

from src.logger import logger

if TYPE_CHECKING:
    from src.optimizer.training.trajectory import Trajectory

REWARD_MODES = ("terminal", "process")


class RewardProcessor:
    def __init__(self, reward_mode: str = "terminal", gamma: float = 0.99):
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
        self.reward_mode = reward_mode
        self.gamma = gamma

    def compute_returns(self, trajectory: "Trajectory") -> List[float]:
        num_steps = trajectory.num_steps
        if num_steps == 0:
            return []

        if self.reward_mode == "terminal":
            return [trajectory.terminal_reward] * num_steps

        rewards = list(trajectory.step_rewards[:num_steps])
        rewards += [0.0] * (num_steps - len(rewards))
        rewards[-1] = trajectory.terminal_reward
        returns, running = [], 0.0
        for reward in reversed(rewards):
            running = reward + self.gamma * running
            returns.insert(0, running)
        logger.debug(f"Process-reward returns for {trajectory.episode_id}: {returns}")
        return returns

    def compute_returns_batch(self, trajectories: List["Trajectory"]) -> List[List[float]]:
        return [self.compute_returns(trajectory) for trajectory in trajectories]
