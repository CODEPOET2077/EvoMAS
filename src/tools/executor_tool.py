"""Executor tool: one EvoMAS meta-step (paper Algorithm 1, lines 5-13).

The Planner calls this tool once per execution stage with the current task state ``s~_{t-1}``.
The tool then

1. asks the Workflow Adapter for a stage-specific layered workflow ``G_t ~ pi_wf(. | s~_{t-1})``;
2. executes the workflow layer by layer to obtain the execution record ``xi_t``;
3. asks the Evaluator (JudgeAgent) for the stage-level assessment ``e_t`` (verdict + feedback);
4. acts as the **Updater** ``mu``: deterministically merges the plan state, execution outcome,
   evaluator feedback and a compact execution history into the record returned to the Planner,
   from which the next task state ``s~_t`` is built (paper Section 3.1 and Appendix C.3);
5. records ``(s~_{t-1}, G_t, e_t)`` in the trajectory buffer when training.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.config import config
from src.logger import logger
from src.registry import TOOL
from src.tools import AsyncTool, ToolResult
from src.utils.cost_utils import get_cost_manager
from src.workflow_adapter import EARLY_EXIT_AGENT, WorkflowAdapter, WorkflowResult

_EXECUTOR_TOOL_DESCRIPTION = """An execution engine that executes individual steps of a plan created by the planning_tool.
This tool is the primary execution mechanism for the planning agent.

ROLE:
- planning_tool creates and manages plans, while executor_tool executes the individual steps of those plans.
- After creating a plan with planning_tool, use executor_tool to execute each step sequentially.

PARAMETERS (inside `current_state`):
- task_description: The complete task to be executed. Do not modify it; keep it consistent across calls.
- current_step_description: The step from your current plan to execute.
- completed_steps_summary: A concise summary of previously completed steps.
- judge_feedback: (Optional) Feedback on the previous step execution, if available.

RETURN VALUE:
- step_status: "completed" (reward=1), "failed" (reward=0) or "unverified" (no evaluator verdict).
- current_step_execution_result: The result of the current step execution.
- judge_evaluation.feedback: Detailed feedback about the execution.
- judge_evaluation.reward: 0 (failed) or 1 (completed).
- next_stage_hint: Suggested next action (refine the step, continue, or finalize after an Early-exit).
"""

# Agents whose output replaces the current candidate set (they aggregate upstream solutions).
_AGGREGATOR_AGENTS = ("sc_ensemble_agent", "summary_agent")


@TOOL.register_module(name="executor_tool", force=True)
class ExecutorTool(AsyncTool):
    """Runs one execution stage: Workflow Adapter -> layered execution -> Evaluator -> Updater."""

    name: str = "executor_tool"
    description: str = _EXECUTOR_TOOL_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "current_state": {
                "type": "object",
                "description": "Task state at this execution stage: task description, current step, "
                               "completed steps summary and optional evaluator feedback.",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "The complete task to be executed. Do not modify it; keep it consistent.",
                    },
                    "current_step_description": {
                        "type": "string",
                        "description": "The step description from your current plan to execute.",
                    },
                    "completed_steps_summary": {
                        "type": "string",
                        "description": "A concise summary of previously completed steps.",
                        "nullable": True,
                    },
                    "judge_feedback": {
                        "type": "string",
                        "description": "Optional feedback from the evaluator on the previous step execution.",
                        "nullable": True,
                    },
                },
                "required": ["task_description", "current_step_description", "completed_steps_summary"],
            }
        },
        "required": ["current_state"],
    }
    output_type = "object"

    def __init__(
        self,
        workflow_adapter: Optional[WorkflowAdapter] = None,
        evaluator: Optional[Any] = None,
        use_evaluator: bool = True,
        **kwargs,
    ):
        """
        Args:
            workflow_adapter: Pre-built Workflow Adapter (injected by the trainer so that all episodes
                share the same policy parameters). Built lazily from the global config otherwise.
            evaluator: Pre-built Evaluator (JudgeAgent). Built lazily from ``config.evaluator_config``.
            use_evaluator: Whether to call the Evaluator after every stage. When disabled, every stage
                is returned as "unverified" without feedback.
        """
        super().__init__(**kwargs)
        self.workflow_adapter = workflow_adapter
        self.evaluator = evaluator
        self.use_evaluator = use_evaluator
        # Compact execution history of the current episode (part of the Updater state).
        self.execution_history: List[Dict[str, Any]] = []
        self._current_episode_id: Optional[str] = None

    async def _ensure_initialized(self) -> None:
        if self.workflow_adapter is None:
            self.workflow_adapter = WorkflowAdapter(config=config)
        if self.use_evaluator and self.evaluator is None:
            from src.atomic_agents.agent import build_agent

            self.evaluator = await build_agent(
                config=config, agent_config=config.evaluator_config, default_tools=TOOL
            )
            logger.info("Evaluator (JudgeAgent) initialized from config.evaluator_config")

    async def forward(self, current_state: Dict) -> ToolResult:
        """Execute one stage conditioned on the task state ``current_state`` (= ``s~_{t-1}``)."""
        await self._ensure_initialized()

        from src.optimizer.training.context import TrajectoryContext

        trajectory_collector = TrajectoryContext.get_collector()
        # The tool instance is shared across training episodes: reset the history per episode.
        if trajectory_collector is not None:
            episode_id = trajectory_collector.trajectory.episode_id
            if episode_id != self._current_episode_id:
                self.reset_execution_history()
                self._current_episode_id = episode_id

        try:
            # 1. Construct the stage-specific workflow G_t.
            workflow = self.workflow_adapter.coordinate(current_state)
            logger.info(f"Workflow G_t: {workflow.agent_names_layers}")

            # 2. Execute the workflow -> execution record xi_t.
            execution_result, step_cost = await self._execute_workflow(workflow, current_state)
            logger.info(f"Stage execution finished (cost=${step_cost:.4f})")

            # 3. Stage-level assessment e_t.
            assessment = await self._evaluate(current_state, execution_result)
            logger.info(f"Evaluator: reward={assessment.get('reward')}")

            # 4. Training: store (s~_{t-1}, G_t, e_t) in the trajectory buffer D.
            if trajectory_collector is not None:
                self._collect_trajectory_step(
                    trajectory_collector, current_state, workflow, assessment, step_cost
                )

            # 5. Updater: integrate the stage outcome into the next structured task state.
            self._append_execution_history(
                step_description=current_state.get("current_step_description", ""),
                execution_result=str(execution_result),
                feedback=assessment.get("feedback", ""),
                reward=assessment.get("reward"),
            )
            return ToolResult(
                output=self._update_task_state(current_state, workflow, execution_result, assessment),
                error=None,
            )
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            logger.exception(e)
            return ToolResult(output=None, error=f"Workflow execution failed: {e}")

    # ------------------------------------------------------------------ #
    # Workflow execution
    # ------------------------------------------------------------------ #
    async def _execute_workflow(self, workflow: WorkflowResult, current_state: Dict) -> Tuple[str, float]:
        """Execute ``G_t`` layer by layer (sequential across layers, in order within a layer).

        ``solutions`` holds the candidate answers produced so far; it is the aggregated message
        passed downstream. Each agent's prompt determines how it consumes this message:
        generators append to it, refiners refine the latest solution, and aggregators
        (Ensemble / Summary) collapse it into a single answer.
        """
        task = self._format_task_for_agents(current_state)
        history = self._format_execution_history()
        current_solution = ""
        solutions: List[str] = []
        total_cost = 0.0

        for layer_idx, layer_agents in enumerate(workflow.layers):
            logger.info(f"[Layer {layer_idx + 1}] Executing {[a.name for a in layer_agents]}")
            for agent in layer_agents:
                name = agent.name
                if name == EARLY_EXIT_AGENT:
                    # Pure termination signal: no execution.
                    continue

                if name == "sc_ensemble_agent":
                    candidates = list(solutions)
                    result = await agent.run(task, solutions=candidates, layer_memory=history)
                    new_solution = self._select_solution_by_letter(result, candidates)
                    solutions = [new_solution]
                elif name == "summary_agent":
                    new_solution = await agent.run(task, solutions=solutions)
                    solutions = [new_solution]
                elif name == "self_refine_agent":
                    new_solution = await agent.run(task, solution=current_solution)
                    solutions.append(new_solution)
                elif name == "multi_generate_agent":
                    result = await agent.run(task)
                    responses = result.get("response", []) if isinstance(result, dict) else [result]
                    solutions.extend(str(r) for r in responses)
                    new_solution = solutions[-1] if solutions else current_solution
                elif name == "generate_query_agent":
                    # Auxiliary agent: produces a follow-up query consumed by the next agent only.
                    new_solution = await agent.run(task, solutions=solutions, layer_memory=history)
                elif name == "file_read_agent":
                    new_solution = await agent.run(current_solution or task)
                    solutions.append(new_solution)
                elif name in ("web_search_agent", "web_browser_agent"):
                    new_solution = await agent.run(task)
                    solutions.append(new_solution)
                else:
                    # io_agent, code_agent, tool_agent and other generators.
                    new_solution = await agent.run(task, layer_memory=history)
                    solutions.append(new_solution)

                total_cost += self._calculate_agent_cost(agent)
                current_solution = new_solution

        # Aggregate remaining candidates if the workflow did not end with an aggregator.
        if len(solutions) > 1:
            for aggregator_name in _AGGREGATOR_AGENTS:
                aggregator = self.workflow_adapter.get_agent_by_name(aggregator_name)
                if aggregator is None:
                    continue
                result = await aggregator.run(task, solutions=solutions)
                if aggregator_name == "sc_ensemble_agent":
                    result = self._select_solution_by_letter(result, solutions)
                total_cost += self._calculate_agent_cost(aggregator)
                return result, total_cost
            logger.warning("No aggregator agent in the pool; returning the latest solution.")
        elif solutions:
            current_solution = solutions[0]
        return current_solution, total_cost

    @staticmethod
    def _format_task_for_agents(current_state: Dict) -> str:
        """Render the task state as the execution-time working context given to every agent."""
        task_description = (current_state.get("task_description") or "").strip()
        step_description = (current_state.get("current_step_description") or "").strip()
        completed_summary = (current_state.get("completed_steps_summary") or "").strip()

        parts = []
        if task_description and step_description:
            parts += [
                f"Context (for reference only): The overall task is: {task_description}",
                "",
                f"Your task: {step_description}",
                "",
                "IMPORTANT: You should only complete the current step above. The overall task is provided "
                "only as context to help you understand the background. Do not attempt to answer or "
                "complete the overall task.",
            ]
        elif step_description or task_description:
            parts.append(f"Your task: {step_description or task_description}")

        if completed_summary:
            parts += [
                "",
                "=== PREVIOUSLY COMPLETED STEPS ===",
                "The following is a summary of steps that have already been completed in this task:",
                "",
                completed_summary,
                "",
                "When working on your current task above, you may refer to the information from these "
                "completed steps if it is relevant to help you understand the context or build upon previous work.",
                "=== END OF PREVIOUSLY COMPLETED STEPS ===",
            ]

        formatted = "\n".join(parts).strip()
        if not formatted:
            logger.warning("Empty task_description and current_step_description in current_state")
            formatted = "No task description provided"
        return formatted

    @staticmethod
    def _calculate_agent_cost(agent: Any) -> float:
        """USD cost of the agent's last run, computed from its token usage and the price table."""
        try:
            monitor = getattr(agent, "monitor", None)
            if monitor is None:
                return 0.0
            usage = monitor.get_total_token_counts()
            model_id = getattr(getattr(agent, "model", None), "model_id", None)
            if model_id is None:
                return float(usage.total_tokens) / 1000.0
            return get_cost_manager().calculate_cost(
                model_id=model_id, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens
            )
        except Exception as e:
            logger.warning(f"Failed to calculate cost for agent {getattr(agent, 'name', agent)}: {e}")
            return 0.0

    @staticmethod
    def _select_solution_by_letter(result: Any, solutions: List[str]) -> Any:
        """Map the Ensemble agent's ``solution_letter`` (A, B, ...) back to the candidate solution."""
        if not isinstance(result, str) or not solutions:
            return result
        letter = None
        try:
            letter = json.loads(result).get("solution_letter")
        except (json.JSONDecodeError, AttributeError):
            pass
        if not letter:
            # Accept JSON blocks, Markdown and plain "solution_letter: A" formats.
            match = re.search(r"solution_letter[\"'\s:*]*([A-Za-z])\b", result)
            letter = match.group(1) if match else None
        if not letter:
            logger.warning("Could not parse solution_letter from the Ensemble output; keeping raw output.")
            return result
        index = ord(letter.strip().upper()) - ord("A")
        if 0 <= index < len(solutions):
            return solutions[index]
        logger.warning(f"solution_letter '{letter}' out of range for {len(solutions)} solutions.")
        return result

    # ------------------------------------------------------------------ #
    # Evaluator
    # ------------------------------------------------------------------ #
    async def _evaluate(self, current_state: Dict, execution_result: Any) -> Dict[str, Any]:
        """Stage-level assessment ``e_t = (verdict, feedback)`` from the Evaluator (paper Appendix B.2).

        The Evaluator returns ``{"feedback": str, "reward": 0 | 1}`` via ``final_judge_answer_tool``;
        reward 1 maps to a successful verdict and reward 0 to a failed / needs-refinement verdict.
        """
        if not self.use_evaluator or self.evaluator is None:
            return {"feedback": "", "reward": None, "evaluation_skipped": True}
        try:
            response = await self.evaluator.run(
                task=current_state.get("task_description", "No task description provided"),
                current_step_task=current_state.get("current_step_description", "No step description provided"),
                execution_result=str(execution_result),
            )
            if isinstance(response, dict) and "reward" in response:
                return response
            logger.warning(f"Unexpected Evaluator output: {str(response)[:200]}")
            return {"feedback": str(response), "reward": None}
        except Exception as e:
            logger.error(f"Evaluator failed: {e}")
            logger.exception(e)
            return {"feedback": f"Evaluation failed: {e}", "reward": None, "evaluation_error": str(e)}

    # ------------------------------------------------------------------ #
    # Updater and trajectory recording
    # ------------------------------------------------------------------ #
    @staticmethod
    def _update_task_state(
        current_state: Dict, workflow: WorkflowResult, execution_result: Any, assessment: Dict
    ) -> Dict[str, Any]:
        """Updater ``mu``: structured record from which the Planner forms ``s~_t`` (paper Eq. 7)."""
        reward = assessment.get("reward")
        step_status = {1: "completed", 0: "failed"}.get(reward, "unverified")
        early_exit = workflow.metadata.get("early_exit", False)
        if step_status == "failed":
            hint = "Refine or repair the current step using the evaluator feedback before moving on."
        elif early_exit:
            hint = ("The Workflow Adapter selected Early-exit: enough evidence has been gathered. "
                    "Move on to the next step or give the final answer.")
        else:
            hint = "Proceed to the next pending step of the plan."
        return {
            "step_status": step_status,
            "task_description": current_state.get("task_description"),
            "current_step_description": current_state.get("current_step_description"),
            "current_step_execution_result": execution_result,
            "judge_evaluation": {"feedback": assessment.get("feedback"), "reward": reward},
            "early_exit": early_exit,
            "next_stage_hint": hint,
        }

    @staticmethod
    def _collect_trajectory_step(
        collector: Any, current_state: Dict, workflow: WorkflowResult, assessment: Dict, cost: float
    ) -> None:
        """Store the stage decision for REINFORCE: state, sampled order per layer, distributions, reward."""
        collector.add_state(dict(current_state))
        collector.add_workflow_decision(
            sampled_indices=workflow.sampled_indices_layers,
            log_probs=workflow.log_probs_layers,
            probs=workflow.probs_layers,
            executed_names=workflow.agent_names_layers,
        )
        # Evaluator reward: only used as a dense signal in the process-reward (PRM) variant.
        reward = assessment.get("reward")
        collector.add_step_reward(float(reward) if reward is not None else 0.0)
        collector.add_cost(cost)

    def _format_execution_history(self) -> Optional[str]:
        """Compact execution history passed to agents as additional working context."""
        if not self.execution_history:
            return None
        lines = []
        for idx, step in enumerate(self.execution_history, 1):
            lines.append(f"Step {idx}: {step['step_description']}")
            if step["execution_result"]:
                lines.append(f"Result: {step['execution_result']}")
            lines.append("")
        return "\n".join(lines).strip()

    def _append_execution_history(
        self, step_description: str, execution_result: str, feedback: str, reward: Optional[int]
    ) -> None:
        self.execution_history.append(
            {
                "step_description": step_description,
                "execution_result": execution_result,
                "feedback": feedback,
                "reward": reward,
            }
        )

    def reset_execution_history(self) -> None:
        self.execution_history = []
        self._current_episode_id = None


__all__ = ["ExecutorTool"]
