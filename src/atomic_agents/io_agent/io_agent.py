from typing import (
    Any,
    List,
    Optional,
)
import yaml

from src.atomic_agents.general_agent import GeneralAgent
from src.atomic_agents.base.async_atomic_multistep_agent import PromptTemplates
from src.atomic_agents.base.async_atomic_multistep_agent import populate_template


from src.memory import AgentMemory
from src.models import Model
from src.registry import AGENT
from src.utils import assemble_project_path

@AGENT.register_module(name="io_agent", force=True)
class IOAgent(GeneralAgent):
    """I/O candidate agent: answers the task directly from the model's internal knowledge."""
    def __init__(
            self,
            config,
            tools: list[Any],
            model: Model,
            prompt_templates: PromptTemplates | None = None,
            stream_outputs: bool = False,
            max_tool_threads: int | None = None,
            **kwargs,
    ):
        self.config = config

        super(IOAgent, self).__init__(
            config=config,
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
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

    
    async def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: Optional[List[Any]] = None,
        layer_memory: Optional[str] = None,
        additional_args: Optional[dict] = None,
        max_steps: Optional[int] = None,
    ):
        """Answer the task directly with the model.

        Args:
            task: The problem/question to be solved.
            Other parameters are the same as the parent run() method.
        """
        self.problem = task
        return await super().run(
            task=task,
            stream=stream,
            reset=reset,
            images=images,
            layer_memory=layer_memory,
            additional_args=additional_args,
            max_steps=max_steps,
        )

    def initialize_task_instruction(self) -> str:
        """Initialize task instruction with the problem."""
        
        return populate_template(
            self.prompt_templates["task_instruction"],
            variables={
                "problem": self.problem or self.task,
            },
        )
