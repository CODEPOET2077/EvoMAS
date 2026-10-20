from typing import Dict, List, Optional
from typing_extensions import Literal

from src.registry import TOOL
from src.tools import AsyncTool, ToolResult
from src.logger import logger


_PLANNING_TOOL_DESCRIPTION = """A planning tool that allows the agent to create and manage plans for solving complex tasks. The tool provides functionality for creating plans, updating plan steps, and tracking progress.

Available actions:
1. `create`: Create a new plan with a unique plan_id, title, and list of steps. The newly created plan will automatically become the active plan.
2. `set_active`: Set a plan as the active plan (use this if you need to switch between multiple plans).
3. `mark_step`: Mark a step's status and add optional notes. This is CRITICAL for tracking progress:
   - Before executing a step, mark it as "in_progress" using: {"action": "mark_step", "step_index": N, "step_status": "in_progress"}
   - After completing a step successfully, mark it as "completed" using: {"action": "mark_step", "step_index": N, "step_status": "completed", "step_notes": "Brief summary of what was accomplished"}
   - If a step cannot proceed due to dependencies or errors, mark it as "blocked" with notes explaining why
4. `get`: Get the current plan details to check progress and step statuses.
5. `update`: Update an existing plan's title or steps.
6. `list`: List all available plans.
7. `delete`: Delete a plan.

IMPORTANT WORKFLOW:
- Step 1: Create a plan using `create` action with plan_id, title, and steps.
- Step 2: Before executing each step, use `mark_step` to set the step status to "in_progress".
- Step 3: Execute the step:
  - For steps that require action (e.g., search, analyze, compute): Execute using appropriate tools (e.g., executor_tool, web_searcher, etc.).
  - For steps that are informational or simple confirmations (e.g., "confirm understanding", "note that information is already available", "clarify requirement"): You can directly mark the step as "completed" with detailed notes explaining the result, WITHOUT needing to use tools. This is especially useful when:
    * You already have sufficient information from previous steps
    * The step is just a confirmation or clarification that can be resolved through reasoning
    * The step is a simple note or observation that doesn't require external tools
- Step 4: After completing the step, use `mark_step` to set the step status to "completed" with notes about what was accomplished.
- Step 5: Use `get` action periodically to review plan progress and determine next steps.
- Repeat Steps 2-5 for each step in the plan.

NOTES:
- When creating a plan, you must use intuitively meaningful terms (such as analyze, search, write code, etc.) to explicitly describe each step.
- You must solve the complex task in ≤ 5 steps. You should structure your plan with a reasonable number of steps: avoid overly granular breakdowns that fragment the workflow, but also avoid overly broad steps that obscure necessary subtasks. Strive for clarity and efficiency in step design.
- Always track your progress by marking steps as "in_progress" before execution and "completed" after successful completion.
- For simple steps that don't require tool execution (e.g., confirming information already available, making observations, clarifying requirements), you can directly use `mark_step` with "completed" status and detailed notes, without calling executor_tool or other tools. This saves time and avoids unnecessary tool calls.
- If you need to check the current plan status, use the `get` action (plan_id is optional if there's an active plan).
"""


@TOOL.register_module(name="planning_tool", force=True)
class PlanningTool(AsyncTool):
    """
    A tool that is used to plan tasks and manage plans.
    """

    name: str = "planning_tool"
    description: str = _PLANNING_TOOL_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "description": "The action to execute. Available actions: create, update, list, get, set_active, mark_step, delete.",
                "enum": [
                    "create",
                    "update",
                    "list",
                    "get",
                    "set_active",
                    "mark_step",
                    "delete",
                ],
                "type": "string",
            },
            "plan_id": {
                "description": "Unique identifier for the plan. Required for create, update, set_active, and delete actions. Optional for get and mark_step (uses active plan if not specified).",
                "type": "string",
                "nullable": True,
            },
            "title": {
                "description": "Title for the plan. Required for create action, optional for update action.",
                "type": "string",
                "nullable": True,
            },
            "steps": {
                "description": "List of plan steps. Required for create action, optional for update action.",
                "type": "array",
                "items": {"type": "string"},
                "nullable": True,
            },
            "step_index": {
                "description": "Index of the step to update (0-based). Required for mark_step action.",
                "type": "integer",
                "nullable": True,
            },
            "step_status": {
                "description": "Status to set for a step. Used with mark_step action.",
                "enum": ["not_started", "in_progress", "completed", "blocked"],
                "type": "string",
                "nullable": True,
            },
            "step_notes": {
                "description": "Additional notes for a step. Optional for mark_step action.",
                "type": "string",
                "nullable": True,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    output_type = "any"

    plans: dict = {}  # Dictionary to store plans by plan_id
    _current_plan_id: Optional[str] = None  # Track the current active plan

    def __init__(self):
        super(PlanningTool, self).__init__()

        # Initialize plans dictionary
        self.plans = {}
        self._current_plan_id = None

    async def _create_plan(
        self,
        plan_id: Optional[str],
        title: Optional[str],
        steps: Optional[List[str]]
    ):
        """Create a new plan with the given ID, title, and steps."""
        if not plan_id:
            res = "Parameter `plan_id` is required for action: create"
            logger.error(res)
            return ToolResult(
                output=None,
                error=res,
            )

        if plan_id in self.plans:
            res = f"A plan with ID '{plan_id}' already exists. Use 'update' to modify existing plans."
            logger.error(res)
            return ToolResult(
                output=None,
                error=res,
            )

        if not title:
            res = "Parameter `title` is required for action: create"
            logger.error(res)
            return ToolResult(
                output=None,
                error=res,
            )

        if (
            not steps
            or not isinstance(steps, list)
            or not all(isinstance(step, str) for step in steps)
        ):
            res = "Parameter `steps` must be a non-empty list of strings for action: create"
            logger.error(res)
            return ToolResult(
                output=None,
                error=res,
            )

        # Create a new plan with initialized step statuses
        plan = {
            "plan_id": plan_id,
            "title": title,
            "steps": steps,
            "step_statuses": ["not_started"] * len(steps),
            "step_notes": [""] * len(steps),
        }

        self.plans[plan_id] = plan
        self._current_plan_id = plan_id  # Set as active plan

        res = f"Plan created successfully with ID: {plan_id}\n\n{self._format_plan(plan)}"
        logger.info(res)

        return ToolResult(
            output=res,
            error=None,
        )

    async def _update_plan(
        self,
        plan_id: Optional[str],
        title: Optional[str],
        steps: Optional[List[str]]
    ) -> ToolResult:
        """Update an existing plan with new title or steps."""
        if not plan_id:
            res = "Parameter `plan_id` is required for action: update"
            logger.error(res)
            return ToolResult(
                output=None,
                error=res,
            )

        if plan_id not in self.plans:
            res = f"No plan found with ID: {plan_id}"
            logger.error(res)
            return ToolResult(
                output=None,
                error=res,
            )

        plan = self.plans[plan_id]

        if title:
            plan["title"] = title

        if steps:
            if not isinstance(steps, list) or not all(
                isinstance(step, str) for step in steps
            ):
                res = "Parameter `steps` must be a list of strings for action: update"
                logger.error(res)
                return ToolResult(
                    output=None,
                    error=res,
                )

            # Preserve existing step statuses for unchanged steps
            # Preserve existing step statuses for unchanged steps
            old_steps = plan["steps"]
            old_statuses = plan["step_statuses"]
            old_notes = plan["step_notes"]

            # Create new step statuses and notes
            new_statuses = []
            new_notes = []

            for i, step in enumerate(steps):
                # If the step exists at the same position in old steps, preserve status and notes
                # If the step exists at the same position in the old steps, keep its status and notes
                if i < len(old_steps) and step == old_steps[i]:
                    new_statuses.append(old_statuses[i])
                    new_notes.append(old_notes[i])
                else:
                    new_statuses.append("not_started")
                    new_notes.append("")

            plan["steps"] = steps
            plan["step_statuses"] = new_statuses
            plan["step_notes"] = new_notes

        res = f"Plan updated successfully: {plan_id}\n\n{self._format_plan(plan)}"
        logger.info(res)
        return ToolResult(
            output=res,
            error=None,
        )

    async def _list_plans(self) -> ToolResult:
        """List all available plans."""
        if not self.plans:
            res = "No plans available. Create a plan with the 'create' action."
            logger.error(res)
            return ToolResult(
                output=None,
                error=res,
            )

        output = "Available plans:\n"
        for plan_id, plan in self.plans.items():
            current_marker = " (active)" if plan_id == self._current_plan_id else ""
            completed = sum(
                1 for status in plan["step_statuses"] if status == "completed"
            )
            total = len(plan["steps"])
            progress = f"{completed}/{total} steps completed"
            output += f"• {plan_id}{current_marker}: {plan['title']} - {progress}\n"

        res = output
        logger.info(res)
        return ToolResult(
            output=res,
            error=None,
        )

    async def _get_plan(self, plan_id: Optional[str]) -> ToolResult:
        """Get details of a specific plan."""
        if not plan_id:
            # If no plan_id is provided, use the current active plan
            if not self._current_plan_id:
                res = "No active plan. Please specify a plan_id or set an active plan."
                logger.error(res)
                return ToolResult(
                    output=None,
                    error=res,
                )
            plan_id = self._current_plan_id

        if plan_id not in self.plans:
            res = f"No plan found with ID: {plan_id}"
            logger.error(res)
            return ToolResult(
                output=None,
                error=res,
            )

        plan = self.plans[plan_id]

        res = self._format_plan(plan)
        logger.info(res)
        return ToolResult(
            output=res,
            error=None,
        )

    async def _set_active_plan(self, plan_id: Optional[str]) -> ToolResult:
        """Set a plan as the active plan."""
        if not plan_id:
            res = "Parameter `plan_id` is required for action: set_active"
            logger.error(res)
            return ToolResult(
                output=None,
                error=res,
            )

        if plan_id not in self.plans:
            res = f"No plan found with ID: {plan_id}"
            logger.error(res)
            return ToolResult(
                output=None,
                error=res,
            )

        self._current_plan_id = plan_id

        res = f"Plan '{plan_id}' is now the active plan.\n\n{self._format_plan(self.plans[plan_id])}"
        logger.info(res)
        return ToolResult(
            output=res,
            error=None,
        )

    async def _mark_step(
        self,
        plan_id: Optional[str],
        step_index: Optional[int],
        step_status: Optional[str],
        step_notes: Optional[str],
    ) -> ToolResult:
        """Mark a step with a specific status and optional notes."""
        if not plan_id:
            # If no plan_id is provided, use the current active plan
            if not self._current_plan_id:
                res = "No active plan. Please specify a plan_id or set an active plan."
                logger.error(res)
                return ToolResult(
                    output=None,
                    error=res,
                )
            plan_id = self._current_plan_id

        if plan_id not in self.plans:
            res = f"No plan found with ID: {plan_id}"
            logger.error(res)
            return ToolResult(
                output=None,
                error=res,
            )

        if step_index is None:
            res = "Parameter `step_index` is required for action: mark_step"
            logger.error(res)
            return ToolResult(
                output=None,
                error=res,
            )

        plan = self.plans[plan_id]

        if step_index < 0 or step_index >= len(plan["steps"]):
            res = f"Invalid step_index: {step_index}. Valid indices range from 0 to {len(plan['steps'])-1}."
            logger.error(res)
            return ToolResult(
                output=None,
                error=res,
            )

        if step_status and step_status not in [
            "not_started",
            "in_progress",
            "completed",
            "blocked",
        ]:
            res = f"Invalid step_status: {step_status}. Valid statuses are: not_started, in_progress, completed, blocked"
            logger.error(res)
            return ToolResult(
                output=None,
                error=res,
            )

        if step_status:
            plan["step_statuses"][step_index] = step_status

        if step_notes:
            plan["step_notes"][step_index] = step_notes

        res = f"Step {step_index} updated successfully in plan '{plan_id}'.\n\n{self._format_plan(plan)}"
        logger.info(res)
        return ToolResult(
            output=res,
            error=None,
        )

    async def _delete_plan(self, plan_id: Optional[str]) -> ToolResult:
        """Delete a plan."""
        if not plan_id:
            res = "Parameter `plan_id` is required for action: delete"
            logger.error(res)
            return ToolResult(
                output=None,
                error=res,
            )

        if plan_id not in self.plans:
            res = f"No plan found with ID: {plan_id}"
            logger.error(res)
            return ToolResult(
                output=None,
                error=res,
            )

        del self.plans[plan_id]

        # If the deleted plan was the active plan, clear the active plan
        if self._current_plan_id == plan_id:
            self._current_plan_id = None

        res = f"Plan '{plan_id}' has been deleted."
        logger.info(res)
        return ToolResult(
            output=res,
            error=None,
        )

    def _format_plan(self, plan: Dict) -> str:
        """Format a plan for display."""
        output = f"Plan: {plan['title']} (ID: {plan['plan_id']})\n"
        output += "=" * len(output) + "\n\n"

        # Calculate progress statistics
        total_steps = len(plan["steps"])
        completed = sum(1 for status in plan["step_statuses"] if status == "completed")
        in_progress = sum(
            1 for status in plan["step_statuses"] if status == "in_progress"
        )
        blocked = sum(1 for status in plan["step_statuses"] if status == "blocked")
        not_started = sum(
            1 for status in plan["step_statuses"] if status == "not_started"
        )

        output += f"Progress: {completed}/{total_steps} steps completed "
        if total_steps > 0:
            percentage = (completed / total_steps) * 100
            output += f"({percentage:.1f}%)\n"
        else:
            output += "(0%)\n"

        output += f"Status: {completed} completed, {in_progress} in progress, {blocked} blocked, {not_started} not started\n\n"
        output += "Steps:\n"

        # Add each step with its status and notes
        for i, (step, status, notes) in enumerate(
            zip(plan["steps"], plan["step_statuses"], plan["step_notes"])
        ):
            status_symbol = {
                "not_started": "[ ]",
                "in_progress": "[→]",
                "completed": "[✓]",
                "blocked": "[!]",
            }.get(status, "[ ]")

            output += f"{i}. {status_symbol} {step}\n"
            if notes:
                output += f"   Notes: {notes}\n"

        return output

    async def forward(
        self,
        action: Literal[
            "create", "update", "list", "get", "set_active", "mark_step", "delete"
        ],
        plan_id: Optional[str] = None,
        title: Optional[str] = None,
        steps: Optional[List[str]] = None,
        step_index: Optional[int] = None,
        step_status: Optional[Literal["not_started", "in_progress", "completed", "blocked"]] = None,
        step_notes: Optional[str] = None,
    ):
        """
        Execute the planning tool with the given action and parameters.

        Parameters:
        - action: The operation to perform
        - plan_id: Unique identifier for the plan
        - title: Title for the plan (used with create action)
        - steps: List of steps for the plan (used with create action)
        - step_index: Index of the step to update (used with mark_step action)
        - step_status: Status to set for a step (used with mark_step action)
        - step_notes: Additional notes for a step (used with mark_step action)
        """

        if action == "create":
            return await self._create_plan(plan_id, title, steps)
        elif action == "update":
            return await self._update_plan(plan_id, title, steps)
        elif action == "list":
            return await self._list_plans()
        elif action == "get":
            return await self._get_plan(plan_id)
        elif action == "set_active":
            return await self._set_active_plan(plan_id)
        elif action == "mark_step":
            return await self._mark_step(plan_id, step_index, step_status, step_notes)
        elif action == "delete":
            return await self._delete_plan(plan_id)
        else:
            res = f"Unrecognized action: {action}. Allowed actions are: create, update, list, get, set_active, mark_step, delete"
            logger.error(res)
            return ToolResult(
                output=None,
                error=res,
            )