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

@AGENT.register_module(name="judge_agent", force=True)
class JudgeAgent(GeneralAgent):
    """Evaluator (paper Section 3.1): stage-level assessment (verdict, confidence, feedback)."""
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
        
        # planning_interval is agent-specific and is not passed to the parent class
        self.planning_interval = planning_interval

        # The Evaluator manages no sub-agents
        self.managed_agents = {}

        super(JudgeAgent, self).__init__(
            config=config,
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            stream_outputs=stream_outputs,
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
        )  # Records steps, user_prompt and system_prompt

    async def run(
        self,
        task: str,
        current_step_task: str,
        execution_result: str,
        stream: bool = False,
        reset: bool = True,
        images: Optional[List[Any]] = None,
        layer_memory: Optional[str] = None,
        additional_args: Optional[dict] = None,
        max_steps: Optional[int] = None,
    ):
        """Evaluate the execution result of the current step.

        Args:
            task: The overall task.
            current_step_task: Description of the step being evaluated.
            execution_result: The result produced for that step.
            Other parameters are the same as the parent run() method.
        """
        self.current_step_task = current_step_task
        self.execution_result = execution_result
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
        """Initialize task instruction with the overall task, current step, and its execution result."""
        
        return populate_template(
            self.prompt_templates["task_instruction"],
            variables={
                "overall_task": self.task,
                "current_step_description": self.current_step_task,
                "execution_result": self.execution_result,
            },
        )


    def should_use_steps(self) -> bool:
        """Use multi-step (ReAct) execution."""
        return True  