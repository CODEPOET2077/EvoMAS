"""Factories for building agents (candidate agents, Planner and Evaluator) from config entries."""

from typing import Any, Dict, Optional

from src.logger import logger
from src.models import model_manager
from src.registry import AGENT, TOOL


def _build_tool(config: Any, tool_name: str, pre_created_tools: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """Return a shared pre-created tool instance if available, otherwise build it from the registry.

    A tool-specific config ``<tool_name>_config`` (e.g. ``web_searcher_tool_config``) is used when
    present; otherwise the tool is built with default arguments.
    """
    if pre_created_tools and tool_name in pre_created_tools:
        return pre_created_tools[tool_name]
    if tool_name not in TOOL:
        logger.warning(f"Tool '{tool_name}' is not registered; skipping.")
        return None
    tool_config = config[f"{tool_name}_config"] if f"{tool_name}_config" in config else dict(type=tool_name)
    return TOOL.build(tool_config)


async def build_agent(
    config: Any,
    agent_config: Any,
    default_tools: Any = None,
    pre_created_tools: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build an agent from its config entry.

    Args:
        config: Global config (provides tool configs).
        agent_config: Agent config with ``type``, ``name``, ``model_id``, ``description``,
            ``template_path``, ``max_steps`` and optional ``tools``.
        default_tools: Tool registry. If ``None``, the agent is built without tools.
        pre_created_tools: Shared tool instances keyed by tool name, used instead of building new ones
            (e.g. the training ``executor_tool`` that wraps the shared Workflow Adapter).
    """
    tools = []
    if default_tools is not None:
        for tool_name in agent_config.get("tools", []):
            tool = _build_tool(config, tool_name, pre_created_tools)
            if tool is not None:
                tools.append(tool)
    if tools:
        logger.info(f"| {agent_config.name}: tools = {[tool.name for tool in tools]}")

    model_id = agent_config["model_id"]
    if model_id not in model_manager.registed_models:
        raise KeyError(f"Model '{model_id}' is not registered. Available: {list(model_manager.registed_models)}")

    return AGENT.build(
        dict(
            type=agent_config.type,
            config=agent_config,
            model=model_manager.registed_models[model_id],
            tools=tools,
            max_steps=agent_config.get("max_steps", 1),
            name=agent_config.name,
            description=agent_config.description,
            provide_run_summary=agent_config.get("provide_run_summary", False),
        )
    )


async def create_agent(config: Any, pre_created_tools: Optional[Dict[str, Any]] = None) -> Any:
    """Build the top-level agent (the Planner, ``config.agent_config``)."""
    return await build_agent(
        config, config.agent_config, default_tools=TOOL, pre_created_tools=pre_created_tools
    )


__all__ = ["build_agent", "create_agent"]
