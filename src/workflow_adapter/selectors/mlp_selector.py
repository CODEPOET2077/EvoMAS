"""MLP layer selector: a lightweight alternative to the cross-attention selector.

The task state is encoded as a single sentence vector. A state encoder and an agent encoder
(both two-layer MLPs) project into a shared space, and agents are scored by cosine similarity
followed by a tempered softmax, as in paper Eq. 10-11. For non-first layers, every candidate
agent embedding is concatenated with the mean embedding of the agents selected upstream.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.logger import logger


def _mlp(in_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
    )


class LayerAgentSelector(nn.Module):
    """Args:
        input_dim: Dimension of the sentence embeddings (384 for MiniLM).
        hidden_dim: Dimension of the shared scoring space.
        temperature: Softmax temperature ``lambda`` in paper Eq. 11.
        device: Torch device (defaults to CUDA if available).
        is_first_layer: Non-first layers concatenate the upstream-agent context to each agent.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
        is_first_layer: bool = False,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.is_first_layer = is_first_layer

        self.state_encoder = _mlp(input_dim, hidden_dim)
        self.agent_encoder = _mlp(input_dim if is_first_layer else input_dim * 2, hidden_dim)
        self.to(self.device)

    def forward(
        self,
        state_embedding: torch.Tensor,
        agents_embeddings: torch.Tensor,
        prev_agents_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            state_embedding: Sentence embedding of the task state, ``[d]`` or ``[1, d]``.
            agents_embeddings: Candidate agent embeddings, ``[N, d]``.
            prev_agents_embeddings: Embeddings of agents selected upstream, ``[K, d]``.

        Returns:
            ``(log_probs, probs)``, each of shape ``[1, N]``.
        """
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)
        state_embedding = state_embedding.to(self.device)
        agents_embeddings = agents_embeddings.to(self.device)

        state_encoded = self.state_encoder(state_embedding)

        if self.is_first_layer:
            agents_input = agents_embeddings
        else:
            if prev_agents_embeddings is None or prev_agents_embeddings.numel() == 0:
                raise ValueError("Non-first layers require the embeddings of the agents selected upstream.")
            context = prev_agents_embeddings.to(self.device).mean(dim=0)
            context = context.unsqueeze(0).expand(agents_embeddings.size(0), -1)
            agents_input = torch.cat([agents_embeddings, context], dim=1)
        agents_encoded = self.agent_encoder(agents_input)

        scores = F.normalize(state_encoded, p=2, dim=1) @ F.normalize(agents_encoded, p=2, dim=1).T
        scores = scores / self.temperature
        return F.log_softmax(scores, dim=1), F.softmax(scores, dim=1)

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "temperature": self.temperature,
                "is_first_layer": self.is_first_layer,
            },
            path,
        )
        logger.info(f"Saved LayerAgentSelector to {path}")

    def load(self, path: str, map_location: Optional[torch.device] = None) -> None:
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=True)
        for key in ("input_dim", "hidden_dim", "is_first_layer"):
            if key in checkpoint and checkpoint[key] != getattr(self, key):
                raise ValueError(
                    f"Checkpoint {path} has {key}={checkpoint[key]}, expected {getattr(self, key)}."
                )
        self.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded LayerAgentSelector from {path}")

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"LayerAgentSelector(input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"temperature={self.temperature}, is_first_layer={self.is_first_layer}, "
            f"params={self.get_num_parameters():,})"
        )


__all__ = ["LayerAgentSelector"]
