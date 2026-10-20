from typing import (
    Any,
)
import yaml
from src.atomic_agents.general_agent import GeneralAgent
from src.atomic_agents.base.async_atomic_multistep_agent import PromptTemplates

from src.memory import AgentMemory
from src.models import Model
from src.registry import AGENT
from src.utils import assemble_project_path

@AGENT.register_module(name="tool_agent", force=True)
class ToolAgent(GeneralAgent):
    """Auxiliary tool agent: a multi-step ReAct agent that solves the task with its (dynamically updatable) tool set."""
    def __init__(
            self,
            config,
            tools: list[Any],
            model: Model,
            prompt_templates: PromptTemplates | None = None,
            planning_interval: int | None = None,
            stream_outputs: bool = False,
            max_tool_threads: int | None = None,
            **kwargs,
    ):
        self.config = config

        super(ToolAgent, self).__init__(
            config=config,
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )

        template_path = assemble_project_path(self.config.template_path)
        with open(template_path, "r") as f:
            self.prompt_templates = yaml.safe_load(f)

        self.system_prompt = self.initialize_system_prompt()
        self.user_prompt = self.initialize_user_prompt()

        # Streaming setup
        self.stream_outputs = stream_outputs
        if self.stream_outputs and not hasattr(self.model, "generate_stream"):
            raise ValueError(
                "`stream_outputs` is set to True, but the model class implements no `generate_stream` method."
            )
        # Tool calling setup
        self.max_tool_threads = max_tool_threads

        self.memory = AgentMemory(
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
        )  # Records steps, user_prompt and system_prompt

    def should_use_steps(self) -> bool:
        """Use multi-step (ReAct) execution."""
        return True
    
    def update_tools(self, new_tools: list[Any]) -> None:
        """Replace the agent's tool set at runtime and regenerate the system prompt.

        Args:
            new_tools: The new list of tools.
        """
        from src.logger import logger
        
        # Rebuild the tools dict
        self.tools = {tool.name: tool for tool in new_tools}
        
        # Regenerate system_prompt from the new self.tools
        self.system_prompt = self.initialize_system_prompt()
        
        # Sync system_prompt into memory
        if hasattr(self, 'memory') and self.memory is not None:
            self.memory.system_prompt = self.system_prompt
        
        logger.debug(
            f"ToolAgent tools updated. Now has {len(self.tools)} tools: "
            f"{list(self.tools.keys())}"
        )