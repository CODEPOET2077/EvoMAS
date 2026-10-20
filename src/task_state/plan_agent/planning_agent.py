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

@AGENT.register_module(name="planning_agent", force=True)
class PlanningAgent(GeneralAgent):
    """Planner (paper Section 3.1): maintains subtask plan and execution status across execution stages."""
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
        
        # planning_interval is specific to the Planner and is not passed to the parent class
        self.planning_interval = planning_interval

        super(PlanningAgent, self).__init__(
            config=config,
            tools=tools,
            model=model,
            prompt_templates=prompt_templates, 
            max_tool_threads=max_tool_threads,
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
        )
    
    def should_use_steps(self) -> bool:
        """Use multi-step (ReAct) execution: create plan, dispatch execution, return result."""
        return True