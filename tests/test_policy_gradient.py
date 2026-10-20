"""Returns and REINFORCE loss (paper Section 3.3 and Section 4.4)."""

import pytest
import torch

from src.optimizer.policy_gradient import ReinforceLoss, RewardProcessor
from src.optimizer.training.trajectory import TrajectoryCollector


def _trajectory(logits, terminal_reward, step_rewards=(1.0, 0.0, 1.0)):
    collector = TrajectoryCollector("toy task")
    for reward in step_rewards:
        log_probs = torch.log_softmax(logits, dim=0).unsqueeze(0)
        collector.add_workflow_decision([[0, 2], [1]], [log_probs, log_probs], [log_probs.exp()] * 2, [[], []])
        collector.add_step_reward(reward)
        collector.add_cost(0.0)
    collector.set_terminal_reward(terminal_reward)
    return collector.get_trajectory()


def test_terminal_returns_assign_utility_to_every_step():
    trajectory = _trajectory(torch.zeros(4), terminal_reward=1.0)
    assert RewardProcessor("terminal").compute_returns(trajectory) == [1.0, 1.0, 1.0]


def test_process_returns_are_discounted():
    trajectory = _trajectory(torch.zeros(4), terminal_reward=1.0, step_rewards=(1.0, 0.0, 0.0))
    returns = RewardProcessor("process", gamma=0.5).compute_returns(trajectory)
    assert returns == pytest.approx([1.0 + 0.5 * (0.0 + 0.5 * 1.0), 0.5, 1.0])


@pytest.mark.parametrize("mode", ["sequence", "mean"])
def test_gradient_step_increases_log_prob_of_successful_workflows(mode):
    logits = torch.zeros(4, requires_grad=True)
    loss_fn = ReinforceLoss(log_prob_mode=mode)

    def total_log_prob():
        trajectory = _trajectory(logits, terminal_reward=1.0)
        return sum(loss_fn.workflow_log_prob(trajectory, t) for t in range(trajectory.num_steps))

    before = total_log_prob().item()
    trajectory = _trajectory(logits, terminal_reward=1.0)
    loss_fn.batch_loss([trajectory], [RewardProcessor().compute_returns(trajectory)]).backward()
    with torch.no_grad():
        logits -= 0.1 * logits.grad
    assert total_log_prob().item() > before


def test_failed_trajectories_give_zero_gradient_under_terminal_reward():
    logits = torch.zeros(4, requires_grad=True)
    trajectory = _trajectory(logits, terminal_reward=0.0)
    loss = ReinforceLoss().batch_loss([trajectory], [RewardProcessor().compute_returns(trajectory)])
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0
