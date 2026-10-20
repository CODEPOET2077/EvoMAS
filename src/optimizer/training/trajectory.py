"""Trajectory buffer ``D`` of one task episode (paper Algorithm 1).

Each meta-step ``t`` (one ``executor_tool`` call) stores the task state ``s~_{t-1}``, the sampled
workflow ``G_t`` (ordered index sequence and distribution per layer) and the Evaluator reward.
The terminal utility ``U(tau)`` is attached once the final answer has been scored.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class Trajectory:
    episode_id: str = ""
    task_description: str = ""
    # Task state s~_{t-1} at every meta-step.
    states: List[Dict[str, Any]] = field(default_factory=list)
    # Ordered sampled indices i_t^(l): [step][layer][rank].
    sampled_indices: List[List[List[int]]] = field(default_factory=list)
    # Log-probabilities over the agent pool: [step][layer] -> Tensor[1, N] (with autograd graph).
    log_probs: List[List[Optional[torch.Tensor]]] = field(default_factory=list)
    # Probabilities over the agent pool: [step][layer] -> Tensor[1, N].
    probs: List[List[Optional[torch.Tensor]]] = field(default_factory=list)
    # Names of the agents actually executed: [step][layer][agent].
    executed_agents: List[List[List[str]]] = field(default_factory=list)
    # Evaluator reward per meta-step (dense signal for the process-reward variant).
    step_rewards: List[float] = field(default_factory=list)
    # USD cost per meta-step.
    costs: List[float] = field(default_factory=list)
    # Verifiable terminal utility U(tau) from the benchmark evaluator.
    terminal_reward: float = 0.0

    @property
    def num_steps(self) -> int:
        return len(self.sampled_indices)


class TrajectoryCollector:
    """Accumulates the meta-step decisions of one episode."""

    def __init__(self, task_description: str, episode_id: Optional[str] = None):
        self.trajectory = Trajectory(
            episode_id=episode_id or f"ep_{uuid.uuid4().hex[:8]}",
            task_description=task_description,
        )

    def add_state(self, state: Dict[str, Any]) -> None:
        self.trajectory.states.append(state)

    def add_workflow_decision(
        self,
        sampled_indices: List[List[int]],
        log_probs: List[Optional[torch.Tensor]],
        probs: List[Optional[torch.Tensor]],
        executed_names: List[List[str]],
    ) -> None:
        self.trajectory.sampled_indices.append(sampled_indices)
        self.trajectory.log_probs.append(log_probs)
        self.trajectory.probs.append(probs)
        self.trajectory.executed_agents.append(executed_names)

    def add_step_reward(self, reward: float) -> None:
        self.trajectory.step_rewards.append(reward)

    def add_cost(self, cost: float) -> None:
        self.trajectory.costs.append(cost)

    def set_terminal_reward(self, reward: float) -> None:
        self.trajectory.terminal_reward = float(reward)

    def get_trajectory(self) -> Trajectory:
        return self.trajectory
