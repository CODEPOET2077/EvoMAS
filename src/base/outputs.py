"""Output types shared by the agent runtime (adapted from smolagents)."""

from dataclasses import dataclass
from typing import Any, TypeAlias, Union

from src.memory import ActionStep, FinalAnswerStep, PlanningStep
from src.models import ChatMessageStreamDelta, ChatMessageToolCall


@dataclass
class ActionOutput:
    output: Any
    is_final_answer: bool


@dataclass
class ToolOutput:
    output: Any
    is_final_answer: bool


StreamEvent: TypeAlias = Union[
    ChatMessageStreamDelta,
    ChatMessageToolCall,
    ActionOutput,
    ToolOutput,
    PlanningStep,
    ActionStep,
    FinalAnswerStep,
]
