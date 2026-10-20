"""Cumulative-mass soft sampling (paper Eq. 11-12) and its sequence log-probability (Eq. 28)."""

import itertools
import math

import torch

from src.workflow_adapter.sampling import AgentSampler, sequence_log_prob, soft_sampling


def test_soft_sampling_reaches_threshold_without_repeats():
    torch.manual_seed(0)
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    for _ in range(200):
        idx = soft_sampling(probs, threshold=0.5).tolist()
        assert len(idx) == len(set(idx))
        mass = probs[idx].sum().item()
        assert mass >= 0.5 - 1e-6
        # Stopping time: the mass before the last draw is still below the threshold (Eq. 29).
        assert probs[idx[:-1]].sum().item() < 0.5


def test_soft_sampling_threshold_one_selects_all():
    idx = soft_sampling(torch.tensor([0.25, 0.25, 0.5]), threshold=1.0).tolist()
    assert sorted(idx) == [0, 1, 2]


def test_sequence_log_prob_matches_closed_form():
    probs = torch.tensor([0.5, 0.3, 0.2])
    expected = math.log(0.3) + math.log(0.5 / (1 - 0.3))
    assert math.isclose(sequence_log_prob(probs.log(), [1, 0]).item(), expected, rel_tol=1e-5)


def test_sequence_probabilities_sum_to_one_under_soft_stopping_rule():
    """Summing Eq. 28 over every possible soft-sampling outcome must give 1."""
    probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
    rho, total = 0.55, 0.0
    for k in range(1, len(probs) + 1):
        for seq in itertools.permutations(range(len(probs)), k):
            prefix_mass = probs[list(seq[:-1])].sum().item() if k > 1 else 0.0
            if prefix_mass < rho <= probs[list(seq)].sum().item():
                total += sequence_log_prob(probs.log(), list(seq)).exp().item()
    assert math.isclose(total, 1.0, rel_tol=1e-5)


def test_sequence_log_prob_is_differentiable():
    logits = torch.randn(5, requires_grad=True)
    sequence_log_prob(torch.log_softmax(logits, dim=0), [3, 1]).backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_agent_sampler_methods():
    probs = torch.tensor([0.1, 0.6, 0.3])
    assert AgentSampler("top_k", {"k": 2})(probs).tolist() == [1, 2]
    assert AgentSampler("greedy")(probs).tolist() == [1]
    assert len(AgentSampler("multinomial", {"num_samples": 2})(probs)) == 2
