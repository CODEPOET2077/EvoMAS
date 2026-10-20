"""Cross-attention layer selector (paper Section 3.2 and Appendix D.1).

For workflow layer ``l`` at meta-step ``t``:

    c_t^(l)  = MeanPool({e_i | a_i in V_t^(<l)})            (Eq. 22, c_t^(1) = 0 / omitted)
    X~_t^(l) = [X_t ; c_t^(l)]                               (Eq. 23)
    H_t^(l)  = Attn(X~_t^(l) W_Q, E W_K, E W_V)              (Eq. 24-25)
    h_t^(l)  = Proj(MeanPool(FFN(H_t^(l))))                  (Eq. 26)
    alpha_ti = cos(h_t^(l), e_i)                             (Eq. 10)
    p_t^(l)  = softmax(alpha_t^(l) / lambda)                 (Eq. 11)

The residual LayerNorm/FFN block and the projection head are standard transformer refinements
on top of the cross-attention described in the paper.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.logger import logger


class AttentionLayerAgentSelector(nn.Module):
    """Scores every candidate agent for one workflow layer via state-to-agent cross-attention.

    Args:
        d_model: Embedding dimension of task-state tokens and agent embeddings.
        num_heads: Number of attention heads (``d_model`` must be divisible by it).
        dropout: Dropout rate inside attention and FFN.
        temperature: Softmax temperature ``lambda`` in paper Eq. 11.
        device: Torch device (defaults to CUDA if available).
        is_first_layer: The first layer has no upstream context token ``c_t^(l)``.
    """

    def __init__(
        self,
        d_model: int = 384,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
        is_first_layer: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.temperature = temperature
        self.is_first_layer = is_first_layer

        # Query: task-state tokens (+ upstream context token); Key/Value: agent embeddings.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        # Maps the pooled state representation to the layer-specific query vector h_t^(l).
        self.agent_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.to(self.device)

    def _build_query_tokens(
        self, state_tokens: torch.Tensor, prev_agents_embeddings: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Return ``X~_t^(l)``: state tokens, with the pooled upstream-agent token appended for l > 1."""
        if self.is_first_layer:
            return state_tokens
        if prev_agents_embeddings is None or prev_agents_embeddings.numel() == 0:
            raise ValueError("Non-first layers require the embeddings of the agents selected upstream.")
        context = prev_agents_embeddings.to(self.device).mean(dim=0)  # c_t^(l), Eq. 22
        context = context.view(1, 1, -1).expand(state_tokens.size(0), -1, -1)
        return torch.cat([state_tokens, context], dim=1)  # Eq. 23

    def forward(
        self,
        state_embedding: torch.Tensor,
        agents_embeddings: torch.Tensor,
        prev_agents_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the layer-wise agent selection distribution.

        Args:
            state_embedding: Task-state token embeddings ``X_t``, shape ``[M, d]`` or ``[B, M, d]``.
            agents_embeddings: Candidate agent embeddings ``E``, shape ``[N, d]``.
            prev_agents_embeddings: Embeddings of agents selected in previous layers, ``[K, d]``.

        Returns:
            ``(log_probs, probs)``, each of shape ``[B, N]``.
        """
        if state_embedding.dim() == 2:
            state_embedding = state_embedding.unsqueeze(0)
        state_tokens = state_embedding.to(self.device)
        agents_embeddings = agents_embeddings.to(self.device)
        batch_size = state_tokens.size(0)

        query = self._build_query_tokens(state_tokens, prev_agents_embeddings)
        agents = agents_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        attn_output, _ = self.cross_attention(query=query, key=agents, value=agents, need_weights=False)
        hidden = self.norm1(query + attn_output)
        hidden = self.norm2(hidden + self.ffn(hidden))

        h = self.agent_projection(hidden.mean(dim=1))  # layer-specific query vector, [B, d]

        # Cosine similarity between h_t^(l) and every agent embedding e_i (Eq. 10).
        scores = F.normalize(h, p=2, dim=1) @ F.normalize(agents_embeddings, p=2, dim=1).T
        scores = scores / self.temperature
        return F.log_softmax(scores, dim=1), F.softmax(scores, dim=1)

    @torch.no_grad()
    def get_attention_weights(
        self,
        state_tokens: torch.Tensor,
        agents_embeddings: torch.Tensor,
        prev_agents_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-head attention weights ``[B, num_heads, M(+1), N]`` (for visualisation)."""
        if state_tokens.dim() == 2:
            state_tokens = state_tokens.unsqueeze(0)
        state_tokens = state_tokens.to(self.device)
        agents_embeddings = agents_embeddings.to(self.device)
        query = self._build_query_tokens(state_tokens, prev_agents_embeddings)
        agents = agents_embeddings.unsqueeze(0).expand(state_tokens.size(0), -1, -1)
        _, attn_weights = self.cross_attention(
            query=query, key=agents, value=agents, need_weights=True, average_attn_weights=False
        )
        return attn_weights

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "temperature": self.temperature,
                "is_first_layer": self.is_first_layer,
            },
            path,
        )
        logger.info(f"Saved AttentionLayerAgentSelector to {path}")

    def load(self, path: str, map_location: Optional[torch.device] = None) -> None:
        """Load parameters in place (the controller owns the selector instances)."""
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=True)
        if checkpoint.get("is_first_layer", self.is_first_layer) != self.is_first_layer:
            raise ValueError(f"Checkpoint {path} was saved for a different layer position.")
        self.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded AttentionLayerAgentSelector from {path}")

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"AttentionLayerAgentSelector(d_model={self.d_model}, num_heads={self.num_heads}, "
            f"temperature={self.temperature}, is_first_layer={self.is_first_layer}, "
            f"params={self.get_num_parameters():,})"
        )


__all__ = ["AttentionLayerAgentSelector"]
