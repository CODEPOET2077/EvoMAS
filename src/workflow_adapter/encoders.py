"""Frozen text encoders used by the Workflow Adapter (paper Section 3.2).

* ``TokenEncoder``    - encodes the compact task state into token embeddings ``X_t in R^{M x d}``
                        (used by the cross-attention selector).
* ``SentenceEncoder`` - encodes text into a single pooled vector ``e in R^d``; used for the
                        candidate-agent embeddings ``e_i`` and for the MLP selector state.

Both encoders wrap a frozen SentenceTransformer (MiniLM by default, ``d = 384``); only the
Workflow Adapter on top of them is trained.
"""

from typing import List, Union

import torch
from sentence_transformers import SentenceTransformer

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


class _FrozenEncoder(torch.nn.Module):
    def __init__(self, model_name: str = DEFAULT_ENCODER):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()


class SentenceEncoder(_FrozenEncoder):
    """Sentence-level encoder: ``str -> [d]`` and ``List[str] -> [N, d]``."""

    def forward(self, sentence: Union[str, List[str]]) -> torch.Tensor:
        embeddings = self.model.encode(sentence, convert_to_tensor=True, show_progress_bar=False)
        if isinstance(sentence, str) and embeddings.dim() == 2:
            embeddings = embeddings.squeeze(0)
        return embeddings

    def encode_batch(self, sentences: List[str]) -> torch.Tensor:
        return self.model.encode(
            sentences, convert_to_tensor=True, show_progress_bar=False, batch_size=32
        )


class TokenEncoder(_FrozenEncoder):
    """Token-level encoder: ``str -> [M, d]`` and ``List[str] -> [B, M, d]``.

    The output contains all tokens of the (truncated) sequence, including special tokens.
    """

    def forward(self, sentence: Union[str, List[str]]) -> torch.Tensor:
        embeddings = self.model.encode(
            sentence,
            output_value="token_embeddings",
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        if isinstance(sentence, str) and embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)
        return embeddings

    def encode_batch(self, sentences: List[str]) -> torch.Tensor:
        return self.model.encode(
            sentences,
            output_value="token_embeddings",
            convert_to_tensor=True,
            show_progress_bar=False,
            batch_size=32,
        )


__all__ = ["SentenceEncoder", "TokenEncoder", "DEFAULT_ENCODER"]
