"""Layer-wise agent sampling strategies for the Workflow Adapter (paper Section 3.2).

Given the categorical distribution ``p_t^(l)`` over the candidate agent pool at layer ``l``,
a sampler returns the *ordered* index sequence ``i_t^(l)`` of selected agents.

* ``soft``        - cumulative-mass soft sampling (paper Eq. 11-12, default): sample agents
                    without replacement until the selected probability mass reaches ``threshold``
                    (``rho`` in the paper).
* ``top_k``       - deterministic top-k selection.
* ``multinomial`` - sample a fixed number of agents without replacement (optional temperature).
* ``greedy``      - alias of top-k with ``num_samples`` agents.

``sequence_log_prob`` implements the exact log-probability of an ordered sample drawn without
replacement (paper Eq. 28), which is the quantity used by REINFORCE (paper Eq. 14 and 16).
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import torch

from src.logger import logger


class SamplingMethod(str, Enum):
    SOFT = "soft"
    TOP_K = "top_k"
    MULTINOMIAL = "multinomial"
    GREEDY = "greedy"


def soft_sampling(probs: torch.Tensor, threshold: float = 0.25, **kwargs) -> torch.Tensor:
    """Cumulative-mass soft sampling (paper Eq. 11-12).

    Agents are drawn one by one without replacement from ``probs`` (renormalised over the
    remaining agents) until the probability mass of the selected set reaches ``threshold``.
    The number of selected agents is therefore adaptive: a peaked distribution selects few
    agents, a flat distribution selects more.

    Args:
        probs: Normalised probability vector of shape ``[N]``.
        threshold: Cumulative probability threshold ``rho`` in ``(0, 1]``.

    Returns:
        Ordered index tensor of shape ``[K]`` (``K`` is the random stopping time).
    """
    device = probs.device
    probs = probs.detach()
    num_items = probs.size(0)
    if num_items == 0:
        return torch.tensor([], dtype=torch.long, device=device)

    selected: List[int] = []
    cumulative = 0.0
    remaining = torch.arange(num_items, device=device)

    while cumulative < threshold and remaining.numel() > 0:
        # torch.multinomial renormalises the remaining mass internally.
        pos = torch.multinomial(probs[remaining], num_samples=1).item()
        idx = remaining[pos].item()
        selected.append(idx)
        cumulative += probs[idx].item()
        remaining = torch.cat([remaining[:pos], remaining[pos + 1:]])

    # Fallback for numerical / degenerate edge cases (paper Appendix D.1).
    if not selected:
        selected = [int(probs.argmax().item())]

    logger.debug(f"Soft sampling: selected {len(selected)} agents, cumulative mass {cumulative:.3f}")
    return torch.tensor(selected, dtype=torch.long, device=device)


def top_k_sampling(probs: torch.Tensor, k: int = 3, **kwargs) -> torch.Tensor:
    """Select the ``k`` most probable agents (sorted by descending probability)."""
    probs = probs.detach()
    if probs.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=probs.device)
    _, top_indices = torch.topk(probs, k=min(k, probs.size(0)))
    return top_indices


def multinomial_sampling(
    probs: torch.Tensor, num_samples: int = 3, temperature: float = 1.0, **kwargs
) -> torch.Tensor:
    """Sample ``num_samples`` agents without replacement, optionally re-tempered."""
    probs = probs.detach()
    if probs.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=probs.device)
    if temperature != 1.0:
        probs = torch.softmax(torch.log(probs + 1e-10) / temperature, dim=0)
    return torch.multinomial(probs, num_samples=min(num_samples, probs.size(0)), replacement=False)


def greedy_sampling(probs: torch.Tensor, num_samples: int = 1, **kwargs) -> torch.Tensor:
    """Greedy selection, equivalent to top-k with ``k = num_samples``."""
    return top_k_sampling(probs, k=num_samples)


_SAMPLING_FUNCTIONS = {
    SamplingMethod.SOFT.value: soft_sampling,
    SamplingMethod.TOP_K.value: top_k_sampling,
    SamplingMethod.MULTINOMIAL.value: multinomial_sampling,
    SamplingMethod.GREEDY.value: greedy_sampling,
}

# Default parameters per sampling method (user parameters take precedence).
DEFAULT_SAMPLING_PARAMS: Dict[str, Dict[str, Any]] = {
    "soft": {"threshold": 0.25},
    "top_k": {"k": 2},
    "multinomial": {"num_samples": 2, "temperature": 1.0},
    "greedy": {"num_samples": 1},
}


class AgentSampler:
    """Unified, configurable interface over the sampling strategies above.

    Example:
        >>> sampler = AgentSampler(method="soft", params={"threshold": 0.25})
        >>> indices = sampler(probs)
    """

    def __init__(self, method: str = "soft", params: Optional[Dict[str, Any]] = None):
        self.method = method.lower()
        if self.method not in _SAMPLING_FUNCTIONS:
            raise ValueError(
                f"Invalid sampling method: {self.method}. Valid methods: {list(_SAMPLING_FUNCTIONS)}"
            )
        self.params = {**DEFAULT_SAMPLING_PARAMS.get(self.method, {}), **(params or {})}
        self._fn = _SAMPLING_FUNCTIONS[self.method]

    def sample(self, probs: torch.Tensor, **override_params) -> torch.Tensor:
        return self._fn(probs, **{**self.params, **override_params})

    def __call__(self, probs: torch.Tensor, **override_params) -> torch.Tensor:
        return self.sample(probs, **override_params)

    def __repr__(self) -> str:
        return f"AgentSampler(method='{self.method}', params={self.params})"


def sequence_log_prob(log_probs: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    """Log-probability of an ordered index sequence sampled without replacement (paper Eq. 28).

        log P(i | s) = sum_r [ log p_{i_r} - log(1 - sum_{q<r} p_{i_q}) ]

    Args:
        log_probs: Log-probabilities over the agent pool, shape ``[N]`` or ``[1, N]``.
        indices: Ordered sampled indices ``(i_1, ..., i_K)``.

    Returns:
        Scalar tensor (differentiable w.r.t. ``log_probs``).
    """
    if log_probs.dim() > 1:
        log_probs = log_probs[0]
    if len(indices) == 0:
        return log_probs.new_zeros(())

    idx = torch.as_tensor(list(indices), dtype=torch.long, device=log_probs.device)
    selected_log_p = log_probs[idx]
    selected_p = selected_log_p.exp()
    # Mass already removed before each draw: [0, p_1, p_1 + p_2, ...].
    consumed = torch.cumsum(selected_p, dim=0) - selected_p
    remaining = (1.0 - consumed).clamp_min(1e-8)
    return (selected_log_p - torch.log(remaining)).sum()


__all__ = [
    "SamplingMethod",
    "AgentSampler",
    "DEFAULT_SAMPLING_PARAMS",
    "soft_sampling",
    "top_k_sampling",
    "multinomial_sampling",
    "greedy_sampling",
    "sequence_log_prob",
]
