"""
Lightweight base class for atomic agents.
Intended for simple, direct atomic agents without the full multi-step agent framework,
while keeping an API consistent with GeneralAgent.
"""
import inspect
import asyncio
import yaml
import time
import warnings
import re
from rich.live import Live
from rich.markdown import Markdown
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, TypedDict, Union, Literal, TypeAlias, List
from dataclasses import dataclass
from jinja2 import StrictUndefined, Template
from collections.abc import AsyncGenerator
from src.memory import (ActionStep,
                        AgentMemory,
                        FinalAnswerStep,
                        PlanningStep,
                        SystemPromptStep,
                        UserPromptStep,
                        LayerMemoryPromptStep,
                        TaskStep)
from src.models import (Model,
                        parse_json_if_needed,
                        agglomerate_stream_deltas,
                        ChatMessage,
                        ChatMessageStreamDelta)
from src.base import (ToolOutput,
                      ActionOutput,
                      StreamEvent) 
from src.utils import (
    is_valid_name,
    make_init_file,
    truncate_content,
    handle_agent_output_types,
)

from src.tools import Tool, AsyncTool
from src.exception import (
    AgentError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,

)
from src.logger import (
    AgentLogger,
    LogLevel,
    Monitor,
    Timing,
    TokenUsage,
)
from src.utils.agent_types import AgentAudio, AgentImage
from src.memory import ToolCall
from rich.panel import Panel
from rich.text import Text
import json
from src.utils import assemble_project_path

# Imported only for type checking, not at runtime.
if TYPE_CHECKING:
    import PIL.Image

from src.tools.final_answer import FinalAnswerTool
from src.tools.default_tools import TOOL_MAPPING

# Find all {{ ... }} placeholders in the template string.
def get_variable_names(self, template: str) -> set[str]:
    pattern = re.compile(r"\{\{([^{}]+)\}\}")
    return {match.group(1).strip() for match in pattern.finditer(template)}

# Render the template with the given variables.
def populate_template(template: str, variables: dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")

class PlanningPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        plan (`str`): Initial plan prompt.
        update_plan_pre_messages (`str`): Update plan pre-messages prompt.
        update_plan_post_messages (`str`): Update plan post-messages prompt.
    """

    initial_plan: str
    update_plan_pre_messages: str
    update_plan_post_messages: str

class ManagedAgentPromptTemplate(TypedDict):
    """
    Prompt templates for the managed agent.

    Args:
        task (`str`): Task prompt.
        report (`str`): Report prompt.
    """

    task: str
    report: str

class FinalAnswerPromptTemplate(TypedDict):
    """
    Prompt templates for the final answer.

    Args:
        pre_messages (`str`): Pre-messages prompt.
        post_messages (`str`): Post-messages prompt.
    """

    pre_messages: str
    post_messages: str

class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        managed_agent ([`~agents.ManagedAgentPromptTemplate`]): Managed agent prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
    """

    system_prompt: str
    user_prompt: str
    task_instruction: str
    planning: PlanningPromptTemplate
    managed_agent: ManagedAgentPromptTemplate
    final_answer: FinalAnswerPromptTemplate

# Empty prompt templates instance
EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    user_prompt="",
    task_instruction="",
    planning=PlanningPromptTemplate(
        initial_plan="",
        update_plan_pre_messages="",
        update_plan_post_messages="",
    ),
    managed_agent=ManagedAgentPromptTemplate(task="", report=""),
    final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
)

class AsyncAtomicMultiStepAgent(ABC):
    """Lightweight base class for atomic agents.
    Design goals: stability, robustness, and API alignment with the rest of the project.

    The set of tools available to an atomic agent is dynamic.

    All atomic agents are initialized at program startup, before the available tool set is determined.
    """
    
    def __init__(
        self,  
        tools: List[Tool], 
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        instructions: str | None = None,
        max_steps: int = 1,  # ReAct-style agents use multiple steps; CoT-style agents use one
        add_base_tools: bool = False,
        verbosity_level: LogLevel = LogLevel.INFO,
        grammar: dict[str, str] | None = None,
        step_callbacks: list[Callable] | None = None,
        planning_interval: int | None = None,
        name: str | None = None,
        description: str | None = None,
        provide_run_summary: bool = False,
        final_answer_checks: list[Callable] | None = None,
        return_full_result: bool = False,
        logger: AgentLogger | None = None,
        ):
        """
        Initialize the atomic agent.
        
        Args:
            tools: List of tools.
            prompt_templates: Prompt templates.
            model: Language model instance.
            agent_type: Agent type identifier.
            description: Description of the agent's function.
        """
        self.agent_name = self.__class__.__name__
        self.model = model
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        if prompt_templates is not None:
            missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(prompt_templates.keys())
            assert not missing_keys, (
                f"Some prompt templates are missing from your custom `prompt_templates`: {missing_keys}"
            )
            for key, value in EMPTY_PROMPT_TEMPLATES.items():
                if isinstance(value, dict):
                    for subkey in value.keys():
                        assert key in prompt_templates.keys() and (subkey in prompt_templates[key].keys()), (
                            f"Some prompt templates are missing from your custom `prompt_templates`: {subkey} under {key}"
                        )
        self.max_steps = max_steps
        self.step_number = 0
        if grammar is not None:
            warnings.warn(
                "Parameter 'grammar' is deprecated and will be removed in version 1.20.",
                FutureWarning,
            )
        self.grammar = grammar
        self.planning_interval = planning_interval
        self.state: dict[str, Any] = {}
        self.name = self._validate_name(name)
        self.description = description
        self.provide_run_summary = provide_run_summary
        self.final_answer_checks = final_answer_checks if final_answer_checks is not None else []
        self.return_full_result = return_full_result
        self.instructions = instructions
        self._setup_tools(tools, add_base_tools) 
        self._validate_tools(tools)

        self.system_prompt = self.initialize_system_prompt()
        
        self.task: str | None = None
        self.memory = AgentMemory(self.system_prompt)
        # Initialize logger
        if logger is None:
            self.logger = AgentLogger(level=verbosity_level)
        else:
            self.logger = logger

        self.monitor = Monitor(self.model, self.logger)
        self.step_callbacks = step_callbacks if step_callbacks is not None else []
        self.step_callbacks.append(self.monitor.update_metrics)
        self.stream_outputs = False

    def _validate_name(self, name: str | None) -> str | None:
        if name is not None and not is_valid_name(name):
            raise ValueError(f"Agent name '{name}' must be a valid Python identifier and not a reserved keyword.")
        return name

    def _setup_tools(self, tools, add_base_tools):
        assert all(isinstance(tool, AsyncTool) for tool in tools), "All elements must be instance of AsyncTool (or a subclass)"
        self.tools = {tool.name: tool for tool in tools}
        if add_base_tools:
            self.tools.update(
                {
                    name: cls()
                    for name, cls in TOOL_MAPPING.items()
                    if name != "python_interpreter_tool" or self.__class__.__name__ == "ToolCallingAgent"
                }
            )
        # self.tools.setdefault("final_answer_tool", FinalAnswerTool())

    def _validate_tools(self, tools):
        tool_names = [tool.name for tool in tools]
       
        if self.name:
            tool_names.append(self.name)
        if len(tool_names) != len(set(tool_names)):
            raise ValueError(
                "Each tool should have a unique name! You passed these duplicate names: "
                f"{[name for name in tool_names if tool_names.count(name) > 1]}"
            )

    async def run(
        self, 
        task: str,
        stream: bool = False, 
        reset: bool = True,
        images: list["PIL.Image.Image"] | None = None,
        layer_memory: str | None = None,  # memory from the previous layer
        additional_args: dict | None = None,
        max_steps: int | None = None,
    ):
        """Run the atomic agent.
        
        Args:
            max_steps: Maximum number of steps; defaults to self.max_steps if None.
        Returns:
            AsyncGenerator[ActionStep | ChatMessageStreamDelta]: Execution result.
        """
        max_steps = max_steps if max_steps is not None else self.max_steps
        self.task = task
        self.interrupt_switch = False
        
        # Store the previous layer's memory (empty string if None)
        self.layer_memory = layer_memory
        layer_memory_str = layer_memory if layer_memory is not None else ""
        self.memory.layer_memory = LayerMemoryPromptStep(layer_memory=layer_memory_str)
        # Tools should be set via set_tools() before run(); the system prompt is
        # regenerated below so it reflects the current tools.
        if additional_args is not None:
            self.state.update(additional_args)
            self.task += f"""
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{str(additional_args)}."""

        self.task = self.initialize_task_instruction()
        self.system_prompt = self.initialize_system_prompt()
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        self.user_prompt = self.initialize_user_prompt()
        self.memory.user_prompt = UserPromptStep(user_prompt=self.user_prompt)
        
        if reset:
            self.memory.reset()
            self.monitor.reset()

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )
        
        self.memory.steps.append(TaskStep(task=self.task, task_images=images))
        # Send variables and tools to the executor
        if getattr(self, "python_executor", None):
            self.python_executor.send_variables(variables=self.state)
            self.python_executor.send_tools(self.tools)

        if stream:  # streaming mode is not fully supported yet
            return self._run_stream(task=self.task, max_steps=max_steps, images=images)
        
        
        steps = [step async for step in self._run_stream(task=self.task, max_steps=max_steps, images=images)]
        assert isinstance(steps[-1], FinalAnswerStep)
        output = steps[-1].output

        return output  # output of the FinalAnswerStep
 
    async def _run_stream(
        self, task: str, max_steps: int, images: list["PIL.Image.Image"] | None = None
    ) -> AsyncGenerator[ActionStep | FinalAnswerStep | ChatMessageStreamDelta]:
        self.step_number = 1  # only meaningful for multi-step (ReAct) agents
        returned_final_answer = False

        if self.should_use_steps():  # multi-step agents (e.g. ReAct) loop
            final_answer = None  # avoid UnboundLocalError
            while not returned_final_answer and self.step_number <= max_steps:
                if self.interrupt_switch:
                    raise AgentError("Agent interrupted.", self.logger)
                action_step_start_time = time.time()
                action_step = ActionStep(
                    step_number=self.step_number,
                    timing=Timing(start_time=action_step_start_time),
                    observations_images=images,
                )
                # PlanningAgent logs "Env Step" in a different color
                if self.__class__.__name__ == "PlanningAgent":
                    self.logger.log_rule(f"Env Step {self.step_number}", level=LogLevel.INFO, style="cyan")
                else:
                    self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
                try:
                    async for output in self._step_stream(action_step):
                        if not isinstance(output, (ActionOutput, ToolOutput)): 
                            yield output

                        if isinstance(output, (ActionOutput, ToolOutput)) and output.is_final_answer:
                            if self.final_answer_checks:
                                self._validate_final_answer(output.output)
                            returned_final_answer = True
                            action_step.is_final_answer = True
                            final_answer = output.output
                except AgentGenerationError as e:
                    # Agent generation errors are not caused by a Model error but an implementation error: so we should raise them and exit.
                    raise e
                except AgentError as e:
                    # Other AgentError types are caused by the Model, so we should log them and iterate.
                    action_step.error = e
                finally:
                    self._finalize_step(action_step)
                    self.memory.steps.append(action_step) 
                    yield action_step
                    self.step_number += 1

            if not returned_final_answer and self.step_number == max_steps + 1:
                final_answer = await self._handle_max_steps_reached(task, images)
                yield action_step
            # Fall back to the last action_step's output if no final answer was produced
            if final_answer is None:
                final_answer = action_step.action_output if action_step.action_output is not None else ""
            yield FinalAnswerStep(handle_agent_output_types(final_answer))

        else:
            # Single-step agents (CoT, IO, Self-Refine, ...) run once without a loop
            action_step_start_time = time.time()
            action_step = ActionStep(
                step_number=self.step_number,
                timing=Timing(start_time=action_step_start_time),
                observations_images=images,
            )
            # PlanningAgent logs "Env Step" in a different color
            if self.__class__.__name__ == "PlanningAgent":
                self.logger.log_rule(f"Env Step {self.step_number}", level=LogLevel.INFO, style="cyan")
            else:
                self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
            final_answer = None  # avoid UnboundLocalError
            try:
                async for output in self._step_stream(action_step):
                    if not isinstance(output, (ActionOutput, ToolOutput)): 
                        yield output
                    if isinstance(output, (ActionOutput, ToolOutput)) and output.is_final_answer:
                        if self.final_answer_checks:
                            self._validate_final_answer(output.output)
                        returned_final_answer = True
                        action_step.is_final_answer = True
                        final_answer = output.output
            except AgentGenerationError as e:
                # Agent generation errors are not caused by a Model error but an implementation error: so we should raise them and exit.
                raise e
            except AgentError as e:
                # Other AgentError types are caused by the Model, so we should log them and iterate.
                action_step.error = e
            finally:
                self._finalize_step(action_step)
                self.memory.steps.append(action_step)
                yield action_step
        
            if not returned_final_answer:
                final_answer = await self._handle_max_steps_reached(task, images)
                yield action_step
            # Fall back to action_step's output if no final answer was produced
            if final_answer is None:
                final_answer = action_step.action_output if action_step.action_output is not None else ""
            yield FinalAnswerStep(handle_agent_output_types(final_answer))



    def _validate_final_answer(self, final_answer: Any):
        for check_function in self.final_answer_checks:
            try:
                assert check_function(final_answer, self.memory)
            except Exception as e:
                raise AgentError(f"Check {check_function.__name__} failed with error: {e}", self.logger)

    def _finalize_step(self, memory_step: ActionStep):
        memory_step.timing.end_time = time.time()
        for callback in self.step_callbacks:
            # For compatibility with old callbacks that don't take the agent as an argument
            callback(memory_step) if len(inspect.signature(callback).parameters) == 1 else callback(
                memory_step, agent=self
            )

    async def _handle_max_steps_reached(self, task: str, images: list["PIL.Image.Image"]) -> Any:
        action_step_start_time = time.time()
        final_answer = await self.provide_final_answer(task, images)
        final_memory_step = ActionStep(
            step_number=self.step_number,
            error=AgentMaxStepsError("Reached max steps.", self.logger),
            timing=Timing(start_time=action_step_start_time, end_time=time.time()),
            token_usage=final_answer.token_usage,
        )
        final_memory_step.action_output = final_answer.content
        self._finalize_step(final_memory_step)
        self.memory.steps.append(final_memory_step)
        return final_answer.content
    
    @property
    def _tools_(self):
        """Returns a combined list of tools."""
        return list(self.tools.values()) 

    @abstractmethod
    def initialize_system_prompt(self) -> str:
        """To be implemented in child classes"""
        ...

    @abstractmethod
    def initialize_user_prompt(self) -> str:
        """To be implemented in child classes"""
        ...

    def initialize_task_instruction(self, task: str) -> str:
        """Initialize the task instruction for the agent."""
        task_instruction = populate_template(
            self.prompt_templates["task_instruction"],
            variables={"task": task},
        )
        return task_instruction   


    def _substitute_state_variables(self, arguments: dict[str, str] | str) -> dict[str, Any] | str:
        """Replace string values in arguments with their corresponding state values if they exist."""
        # No state (or empty state): return arguments unchanged
        if not hasattr(self, 'state') or not self.state:
            return arguments
            
        # For dict arguments, substitute values that are state keys
        if isinstance(arguments, dict):
            return {
                key: self.state.get(value, value) if isinstance(value, str) else value
                for key, value in arguments.items()
            }  # fall back to the literal value if not a state key
        return arguments

    def get_available_tools(self) -> List[Tool]:
        """
        Get the list of available tools.
        
        Returns:
            List[Tool]: Available tools.
        """
        return list(self.tools.values())

    @abstractmethod
    def should_use_steps(self) -> bool:
        """
        Whether the agent runs in multiple steps (only ReAct-style agents do).
        
        Returns:
            bool: Whether to use steps.
        """
        pass

    async def write_memory_to_messages(
        self,
        summary_mode: bool = False,
        include_layer_memory: bool = True,
    ) -> list[ChatMessage]:
        """
        Convert memory into a list of messages, including layer_memory.
        
        Args:
            summary_mode: Whether to use summary mode.
            include_layer_memory: Whether to include the previous layer's memory.
            
        Returns:
            list[ChatMessage]: List of messages.
        """
        messages = []
        
        # 1. System prompt of this agent
        messages.extend(self.memory.system_prompt.to_messages(summary_mode=summary_mode))
        
        # 2. Step history of this agent.
        # All agent types include the TaskStep; ReAct agents also include ActionSteps.
        for memory_step in self.memory.steps:
            if self.should_use_steps():
                # ReAct: all steps (TaskStep + ActionStep)
                messages.extend(memory_step.to_messages(summary_mode=summary_mode))
            else:
                # CoT/SC/ToT: TaskStep only
                from src.memory import TaskStep
                if isinstance(memory_step, TaskStep):
                    messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        
        # 3. Previous layer's memory (only if enabled and non-empty)
        if include_layer_memory and hasattr(self.memory, 'layer_memory') and self.memory.layer_memory:
            if self.memory.layer_memory.layer_memory and self.memory.layer_memory.layer_memory.strip():
                messages.extend(self.memory.layer_memory.to_messages(summary_mode=summary_mode))
        
        # 4. User prompt (may be None)
        if hasattr(self.memory, 'user_prompt') and self.memory.user_prompt is not None:
            messages.extend(self.memory.user_prompt.to_messages(summary_mode=summary_mode))
        
        return messages




   
    