from typing import (
    Any,
)
import yaml
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import PIL.Image

from src.atomic_agents.general_agent import GeneralAgent
from src.atomic_agents.base.async_atomic_multistep_agent import PromptTemplates

from src.memory import AgentMemory
from src.models import Model
from src.registry import AGENT
from src.utils import assemble_project_path

@AGENT.register_module(name="early_exit_agent", force=True)
class EarlyExitAgent(GeneralAgent):
    """Early-exit candidate agent: when selected by the Workflow Adapter, signals that no further layers should be executed."""
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

        super(EarlyExitAgent, self).__init__(
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
        images: list["PIL.Image.Image"] | None = None,
        layer_memory: str | None = None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
    ) -> str:
        """Return the early-exit signal immediately without running the model.

        All arguments are accepted for interface compatibility and ignored.

        Returns:
            str: The early-exit signal message.
        """
        return "Early exit signal: Coordinator should stop processing subsequent layers."