"""Workflow Adapter (paper Section 3.2).

Given the compact task state ``s~_t``, the adapter instantiates a stage-specific layered
workflow ``G_t = (V_t, E_t)`` from the fixed candidate agent pool ``P``:

1. Encode ``s~_t`` into token embeddings ``X_t`` (frozen MiniLM).
2. For each layer ``l = 1..L``, score all candidate agents with the layer selector and sample the
   layer's agent set ``V_t^(l)`` with cumulative-mass soft sampling (Eq. 10-12). The agents chosen
   in all earlier layers are mean-pooled and fed back as context (Eq. 22-23).
3. Consecutive layers are fully connected (Eq. 9); the workflow ends early if the Early-exit
   agent is selected.

The adapter only *routes* agents: agent roles, prompts and tool interfaces stay fixed.
"""

import asyncio
import concurrent.futures
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.logger import logger
from src.workflow_adapter.controller import LayerOutput, WorkflowAdapterController
from src.workflow_adapter.encoders import SentenceEncoder, TokenEncoder

EARLY_EXIT_AGENT = "early_exit_agent"


@dataclass
class WorkflowResult:
    """A sampled workflow ``G_t`` together with the quantities needed for policy-gradient training.

    Attributes:
        layers: Agent instances executed at each layer.
        agent_names_layers: Names of the executed agents at each layer.
        agent_indices_layers: Pool indices of the executed agents at each layer.
        sampled_indices_layers: Ordered index sequences ``i_t^(l)`` drawn by the sampler. These are
            the actions whose probability enters ``log P_Theta(G_t | s~_t)`` (Eq. 14).
        log_probs_layers: Per-layer log-probabilities over the pool (``None`` in non-learned modes).
        probs_layers: Per-layer probabilities over the pool (``None`` in non-learned modes).
        metadata: Extra information, e.g. ``early_exit`` and ``constraint_applied_layers``.
    """

    layers: List[List[Any]]
    agent_names_layers: List[List[str]]
    agent_indices_layers: List[List[int]]
    sampled_indices_layers: List[List[int]]
    log_probs_layers: List[Optional[torch.Tensor]]
    probs_layers: List[Optional[torch.Tensor]]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _run_coroutine_sync(coro_factory):
    """Run a coroutine from synchronous code, whether or not an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())
    # Inside a running loop (e.g. constructed from an async entry point): use a helper thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro_factory())).result()


class WorkflowAdapter:
    """State-conditioned workflow constructor over a fixed candidate agent pool.

    ``workflow_adapter_config`` keys:
        enabled: ``True`` (learned policy), ``False`` (every agent at every layer) or
            ``'random'`` (uniformly random agents, ``agents_per_layer`` per layer).
        available_agents: Names of the candidate agents (each needs a ``<name>_config``).
        num_layers: Maximum workflow depth ``L``.
        agents_per_layer: Optional fixed layer width (``None`` = adaptive soft sampling).
        selector_type / sampling / temperature / d_model / ...: see ``WorkflowAdapterController``.
        first_layer_constraints: Optional ``dict(forbidden=[...], required=[...])`` execution-validity
            rules for the first layer (e.g. aggregators need upstream inputs to aggregate).
        resume_path: Optional checkpoint directory to load at start-up.
    """

    def __init__(self, config: Any):
        self.config = config
        adapter_config = getattr(config, "workflow_adapter_config", None)
        if adapter_config is None:
            raise ValueError("Config must define `workflow_adapter_config`.")
        self._validate_config(adapter_config)
        self.adapter_config = adapter_config

        enabled = adapter_config.get("enabled", True)
        self.is_enabled = enabled is True
        self.is_random = enabled == "random"
        optimizer_config = getattr(config, "optimizer_config", None) or {}
        self.training = optimizer_config.get("training_mode", None) == "workflow_adapter"

        self.selector_type = adapter_config["selector_type"]
        self.num_layers = adapter_config.get("num_layers", 3)
        self.agents_per_layer = adapter_config.get("agents_per_layer", None)
        constraints = adapter_config.get("first_layer_constraints") or {}
        self.first_layer_forbidden = set(constraints.get("forbidden", []))
        self.first_layer_required = list(constraints.get("required", []))

        self.agents, self.agent_descriptions = self._build_agent_pool(adapter_config["available_agents"])

        if self.is_enabled:
            self.encoder = TokenEncoder() if self.selector_type == "attention" else SentenceEncoder()
            self.agents_embeddings = self._compute_agents_embeddings()
            self.controller = WorkflowAdapterController(adapter_config, training=self.training)
            resume_path = adapter_config.get("resume_path", None)
            if resume_path:
                self.controller.load(resume_path)
        else:
            self.encoder = None
            self.agents_embeddings = None
            self.controller = None

        mode = "learned" if self.is_enabled else ("random" if self.is_random else "all-agents")
        logger.info(
            f"WorkflowAdapter initialized: mode={mode}, {len(self.agents)} candidate agents "
            f"{[a.name for a in self.agents]}, {self.num_layers} layers, selector={self.selector_type}"
        )

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        for key in ("selector_type", "available_agents"):
            if key not in config:
                raise ValueError(f"Missing required workflow_adapter_config key: {key}")
        if not config["available_agents"]:
            raise ValueError("available_agents cannot be empty")
        if config.get("enabled", True) not in (True, False, "random"):
            raise ValueError("enabled must be True, False or 'random'")
        if config.get("num_layers", 3) < 1:
            raise ValueError("num_layers must be >= 1")
        agents_per_layer = config.get("agents_per_layer", None)
        if agents_per_layer is not None and agents_per_layer < 1:
            raise ValueError("agents_per_layer must be >= 1 or None")

    # ------------------------------------------------------------------ #
    # Agent pool
    # ------------------------------------------------------------------ #
    def _build_agent_pool(self, agent_names: Sequence[str]) -> Tuple[List[Any], List[str]]:
        """Instantiate the fixed candidate agent pool ``P`` from ``<name>_config`` entries."""
        from src.atomic_agents.agent import build_agent
        from src.registry import TOOL

        async def _build_one(name: str):
            config_name = f"{name}_config"
            if config_name not in self.config:
                logger.warning(f"Config '{config_name}' not found; skipping candidate agent '{name}'.")
                return None
            try:
                return await build_agent(config=self.config, agent_config=self.config[config_name], default_tools=TOOL)
            except Exception as e:
                logger.error(f"Failed to build candidate agent '{name}': {e}")
                logger.exception(e)
                return None

        async def _build_all():
            return await asyncio.gather(*[_build_one(name) for name in agent_names])

        agents = [agent for agent in _run_coroutine_sync(_build_all) if agent is not None]
        if not agents:
            raise ValueError("No candidate agents were successfully initialized.")
        descriptions = [self._describe_agent(agent) for agent in agents]
        return agents, descriptions

    @staticmethod
    def _describe_agent(agent: Any) -> str:
        """Textual agent specification (role, capabilities, tool interface) used to derive ``e_i``."""
        text = f"{agent.name}: {getattr(agent, 'description', '')}"
        tool_names = [name for name in getattr(agent, "tools", {}) if name != "final_answer_tool"]
        if tool_names:
            text += f" Tools: {', '.join(tool_names)}."
        return text

    def _compute_agents_embeddings(self) -> torch.Tensor:
        """Agent embeddings ``E = [e_1; ...; e_N]`` (always sentence-level vectors)."""
        with torch.no_grad():
            embeddings = SentenceEncoder().encode_batch(self.agent_descriptions)
        logger.info(f"Computed candidate-agent embeddings: {tuple(embeddings.shape)}")
        return embeddings

    def get_agent_by_name(self, agent_name: str) -> Optional[Any]:
        return next((agent for agent in self.agents if agent.name == agent_name), None)

    def _index_of(self, agent_name: str) -> Optional[int]:
        return next((i for i, agent in enumerate(self.agents) if agent.name == agent_name), None)

    # ------------------------------------------------------------------ #
    # Task-state encoding
    # ------------------------------------------------------------------ #
    @staticmethod
    def format_state_text(current_state: Dict[str, Any]) -> str:
        """Compact serialization of ``s~_t`` centred on the active subtask (paper Appendix B.3)."""
        parts = []
        if current_state.get("task_description"):
            parts.append(f"Task: {current_state['task_description']}")
        if current_state.get("current_step_description"):
            parts.append(f"Current Step: {current_state['current_step_description']}")
        if current_state.get("judge_feedback"):
            parts.append(f"Feedback: {current_state['judge_feedback']}")
        return " ".join(parts)

    def _encode_state(self, current_state: Dict[str, Any]) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(self.format_state_text(current_state))

    # ------------------------------------------------------------------ #
    # Workflow construction
    # ------------------------------------------------------------------ #
    def _apply_first_layer_constraints(
        self, indices: List[int], probs: Optional[torch.Tensor]
    ) -> Tuple[List[int], bool]:
        """Remove agents that cannot run first and add required ones; returns (indices, applied)."""
        if not self.first_layer_forbidden and not self.first_layer_required:
            return indices, False

        kept = [i for i in indices if self.agents[i].name not in self.first_layer_forbidden]
        applied = len(kept) != len(indices)
        for name in self.first_layer_required:
            idx = self._index_of(name)
            if idx is None:
                logger.warning(f"Required first-layer agent '{name}' is not in the candidate pool.")
            elif idx not in kept:
                kept.append(idx)
                applied = True

        if not kept:
            # Fall back to the most probable admissible agent.
            order = probs.flatten().argsort(descending=True).tolist() if probs is not None else range(len(self.agents))
            kept = [next(i for i in order if self.agents[i].name not in self.first_layer_forbidden)]
            applied = True
        return kept, applied

    def coordinate(self, current_state: Dict[str, Any]) -> WorkflowResult:
        """Sample the stage-specific workflow ``G_t ~ pi_wf(. | s~_t; Theta)`` (paper Eq. 13)."""
        if not self.is_enabled:
            return self._coordinate_static()

        state_embedding = self._encode_state(current_state)
        layers, names, executed, sampled, log_probs, probs, constrained = [], [], [], [], [], [], []
        prev_agents_embeddings = None
        upstream_indices: List[int] = []  # Agents selected in all layers < l (V_t^(<l), Eq. 22).
        early_exit = False

        for layer_idx in range(self.num_layers):
            output: LayerOutput = self.controller.forward_layer(
                layer_idx=layer_idx,
                state_embedding=state_embedding,
                agents_embeddings=self.agents_embeddings,
                prev_agents_embeddings=prev_agents_embeddings,
                num_agents_to_select=self.agents_per_layer,
            )
            indices = list(output.selected_indices)
            applied = False
            if layer_idx == 0:
                indices, applied = self._apply_first_layer_constraints(indices, output.probs)

            layer_agents = [self.agents[i] for i in indices]
            layers.append(layer_agents)
            names.append([agent.name for agent in layer_agents])
            executed.append(indices)
            sampled.append(list(output.selected_indices))
            log_probs.append(output.log_probs)
            probs.append(output.probs)
            constrained.append(applied)
            logger.info(f"Layer {layer_idx + 1}/{self.num_layers}: {names[-1]}")

            if EARLY_EXIT_AGENT in names[-1]:
                early_exit = True
                logger.info("Early-exit agent selected: stopping workflow construction.")
                break
            upstream_indices.extend(indices)
            prev_agents_embeddings = self.agents_embeddings[upstream_indices]

        return WorkflowResult(
            layers=layers,
            agent_names_layers=names,
            agent_indices_layers=executed,
            sampled_indices_layers=sampled,
            log_probs_layers=log_probs,
            probs_layers=probs,
            metadata={
                "mode": "learned",
                "num_layers": self.num_layers,
                "actual_num_layers": len(layers),
                "early_exit": early_exit,
                "constraint_applied_layers": constrained,
            },
        )

    def _coordinate_static(self) -> WorkflowResult:
        """Non-learned baselines: all agents at every layer, or uniformly random agents."""
        layers, names, indices = [], [], []
        for _ in range(self.num_layers):
            if self.is_random:
                k = min(self.agents_per_layer or 1, len(self.agents))
                layer_indices = sorted(random.sample(range(len(self.agents)), k))
            else:
                layer_indices = list(range(len(self.agents)))
            layers.append([self.agents[i] for i in layer_indices])
            names.append([self.agents[i].name for i in layer_indices])
            indices.append(layer_indices)

        return WorkflowResult(
            layers=layers,
            agent_names_layers=names,
            agent_indices_layers=indices,
            sampled_indices_layers=indices,
            log_probs_layers=[None] * len(layers),
            probs_layers=[None] * len(layers),
            metadata={
                "mode": "random" if self.is_random else "all-agents",
                "num_layers": self.num_layers,
                "actual_num_layers": len(layers),
                "early_exit": False,
                "constraint_applied_layers": [False] * len(layers),
            },
        )

    def get_agents_info(self) -> Dict[str, Any]:
        return {
            "num_agents": len(self.agents),
            "agent_names": [agent.name for agent in self.agents],
            "agent_descriptions": self.agent_descriptions,
            "selector_type": self.selector_type,
            "num_layers": self.num_layers,
            "agents_per_layer": self.agents_per_layer,
        }


__all__ = ["WorkflowAdapter", "WorkflowResult", "EARLY_EXIT_AGENT"]
