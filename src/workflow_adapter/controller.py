"""Trainable part of the Workflow Adapter: one selector per workflow layer plus a shared sampler.

The controller holds the policy parameters ``Theta`` optimised by REINFORCE (paper Section 3.3).
Each layer has its own selector, so layer ``l`` produces its own distribution ``p_t^(l)``.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.logger import logger
from src.workflow_adapter.sampling import AgentSampler
from src.workflow_adapter.selectors import AttentionLayerAgentSelector, LayerAgentSelector


@dataclass
class LayerOutput:
    """Output of one workflow layer.

    Attributes:
        selected_indices: Ordered sampled index sequence ``i_t^(l)``.
        log_probs: Log-probabilities over the agent pool, ``[1, N]`` (keeps the graph when training).
        probs: Probabilities over the agent pool, ``[1, N]``.
    """

    selected_indices: List[int]
    log_probs: torch.Tensor
    probs: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowAdapterController:
    """Manages the per-layer selectors and the shared agent sampler.

    Args:
        config: ``workflow_adapter_config`` dictionary. Relevant keys:
            ``selector_type`` ('attention' | 'mlp'), ``num_layers``, ``sampling`` (method/params),
            ``temperature``, ``d_model``, ``num_heads``, ``dropout`` (attention), ``hidden_dim`` (mlp).
        training: Keep the autograd graph of the layer distributions for policy-gradient updates.
    """

    def __init__(self, config: Dict[str, Any], training: bool):
        self.config = dict(config)
        self.selector_type = config["selector_type"]
        self.num_layers = config.get("num_layers", 3)
        self.training = training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layer_selectors = self._create_layer_selectors()

        sampling_config = config.get("sampling") or {"method": "soft", "params": {"threshold": 0.25}}
        self.sampler = AgentSampler(
            method=sampling_config.get("method", "soft"),
            params=sampling_config.get("params", {}),
        )
        logger.info(
            f"WorkflowAdapterController: {self.num_layers} layers, selector={self.selector_type}, "
            f"sampler={self.sampler}"
        )

    def _create_layer_selectors(self) -> nn.ModuleList:
        selectors = nn.ModuleList()
        temperature = self.config.get("temperature", 1.0)
        for layer_idx in range(self.num_layers):
            is_first = layer_idx == 0
            if self.selector_type == "attention":
                selector = AttentionLayerAgentSelector(
                    d_model=self.config.get("d_model", 384),
                    num_heads=self.config.get("num_heads", 4),
                    dropout=self.config.get("dropout", 0.1),
                    temperature=temperature,
                    device=self.device,
                    is_first_layer=is_first,
                )
            elif self.selector_type == "mlp":
                selector = LayerAgentSelector(
                    input_dim=self.config.get("d_model", 384),
                    hidden_dim=self.config.get("hidden_dim", 128),
                    temperature=temperature,
                    device=self.device,
                    is_first_layer=is_first,
                )
            else:
                raise ValueError(f"Invalid selector_type: {self.selector_type}. Must be 'attention' or 'mlp'.")
            selectors.append(selector)
        return selectors

    def parameters(self):
        """All trainable parameters ``Theta`` across layers."""
        for selector in self.layer_selectors:
            yield from selector.parameters()

    def forward_layer(
        self,
        layer_idx: int,
        state_embedding: torch.Tensor,
        agents_embeddings: torch.Tensor,
        prev_agents_embeddings: Optional[torch.Tensor] = None,
        num_agents_to_select: Optional[int] = None,
    ) -> LayerOutput:
        """Score the agent pool for one layer and sample the layer's agent set.

        Args:
            layer_idx: Workflow layer index ``l - 1`` (0-based).
            state_embedding: Encoded task state (token sequence for attention, vector for MLP).
            agents_embeddings: Candidate agent embeddings ``[N, d]``.
            prev_agents_embeddings: Embeddings of agents selected in layers ``< l`` (required for l > 1).
            num_agents_to_select: Optional fixed layer width (ignored by soft sampling).
        """
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(f"layer_idx={layer_idx} out of range [0, {self.num_layers})")

        selector = self.layer_selectors[layer_idx]
        with torch.set_grad_enabled(self.training):
            log_probs, probs = selector(
                state_embedding=state_embedding,
                agents_embeddings=agents_embeddings,
                prev_agents_embeddings=prev_agents_embeddings,
            )

        probs_1d = probs[0] if probs.dim() > 1 else probs
        override = {}
        if num_agents_to_select is not None:
            if self.sampler.method == "top_k":
                override = {"k": num_agents_to_select}
            elif self.sampler.method in ("multinomial", "greedy"):
                override = {"num_samples": num_agents_to_select}
        selected = self.sampler(probs_1d, **override).tolist()
        if isinstance(selected, int):
            selected = [selected]

        return LayerOutput(
            selected_indices=selected,
            log_probs=log_probs,
            probs=probs,
            metadata={"layer_idx": layer_idx, "sampling_method": self.sampler.method},
        )

    def train_mode(self) -> None:
        for selector in self.layer_selectors:
            selector.train()

    def eval_mode(self) -> None:
        for selector in self.layer_selectors:
            selector.eval()

    def save(self, save_dir: str) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for layer_idx, selector in enumerate(self.layer_selectors):
            selector.save(str(save_dir / f"layer_{layer_idx}_selector.pth"))
        with open(save_dir / "controller_config.json", "w") as f:
            json.dump(self.config, f, indent=2, default=str)
        logger.info(f"Saved Workflow Adapter ({self.num_layers} layers) to {save_dir}")

    def load(self, save_dir: str) -> None:
        save_dir = Path(save_dir)
        for layer_idx, selector in enumerate(self.layer_selectors):
            selector.load(str(save_dir / f"layer_{layer_idx}_selector.pth"))
        logger.info(f"Loaded Workflow Adapter ({self.num_layers} layers) from {save_dir}")

    def get_num_parameters(self) -> Dict[str, Any]:
        per_layer = [selector.get_num_parameters() for selector in self.layer_selectors]
        return {"per_layer": per_layer, "total": sum(per_layer), "num_layers": self.num_layers}

    def __repr__(self) -> str:
        return (
            f"WorkflowAdapterController(num_layers={self.num_layers}, selector_type='{self.selector_type}', "
            f"sampler={self.sampler}, total_parameters={self.get_num_parameters()['total']:,})"
        )


__all__ = ["WorkflowAdapterController", "LayerOutput"]
