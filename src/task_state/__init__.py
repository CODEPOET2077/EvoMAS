"""Task-state construction (paper Section 3.1).

* Planner   (``PlanningAgent``): maintains the subtask plan and the active subtask.
* Evaluator (``JudgeAgent``):    verifies whether the latest execution solved the active subtask.
* Updater:  deterministic state aggregation inside ``src/tools/executor_tool.py``.
"""

from .judge_agent import JudgeAgent
from .plan_agent import PlanningAgent

# Paper-level aliases.
Planner = PlanningAgent
Evaluator = JudgeAgent

__all__ = ["PlanningAgent", "JudgeAgent", "Planner", "Evaluator"]
