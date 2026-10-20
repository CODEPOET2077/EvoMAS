"""Component registries.

Components are registered by name and built from config dictionaries, e.g.
``TOOL.build(dict(type="web_searcher_tool"))`` or ``AGENT.build(dict(type="io_agent", ...))``.
"""

from mmengine.registry import Registry

DATASET = Registry("dataset", locations=["src.dataset"])
TOOL = Registry("tool", locations=["src.tools"])
# Candidate agents (src.atomic_agents) plus the Planner / Evaluator (src.task_state).
AGENT = Registry("agent", locations=["src.atomic_agents", "src.task_state"])
