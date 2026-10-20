"""Roll out EvoMAS on training tasks and collect trajectories (paper Algorithm 1).

One rollout runs the Planner on a task inside ``trajectory_collection_context``. Every
``executor_tool`` call records its workflow decision, and after the episode the final answer
is scored against the ground truth to obtain the terminal utility ``U(tau)``.
"""

import asyncio
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.atomic_agents import create_agent
from src.logger import logger
from src.metric import evaluate_answer
from src.optimizer.training.context import trajectory_collection_context
from src.optimizer.training.trajectory import Trajectory, TrajectoryCollector

_write_lock = threading.Lock()


def append_jsonl(entry: Dict[str, Any], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def build_question(example: Dict[str, Any]) -> str:
    """Task text given to the Planner (attached files are mentioned explicitly)."""
    question = example.get("question") or example.get("task", "")
    if example.get("file_name"):
        question += (
            "\n\nTo solve the task above, you will have to use these attached files:\n"
            f" - Attached file: {example['file_name']}"
        )
    return question


class DataCollector:
    """Args:
        config: Global config (uses ``optimizer_config.evaluation`` and ``save_path``).
        pre_created_tools: Tool instances shared by all rollouts, e.g. ``{"executor_tool": tool}``
            so every episode uses the same Workflow Adapter parameters.
    """

    def __init__(self, config: Any, pre_created_tools: Optional[Dict[str, Any]] = None):
        self.config = config
        self.pre_created_tools = pre_created_tools
        evaluation = (getattr(config, "optimizer_config", None) or {}).get("evaluation", {}) or {}
        self.eval_method = evaluation.get("method", "llm")
        self.eval_model = evaluation.get("model", "gpt-4o-mini")
        self.save_path = getattr(config, "save_path", None)
        # Candidate agents and the executor tool are shared across rollouts and keep per-run state,
        # so rollouts are sequential by default.
        self.rollout_concurrency = (getattr(config, "optimizer_config", None) or {}).get("rollout_concurrency", 1)

    async def collect_single_episode(self, example: Dict[str, Any]) -> Trajectory:
        question = example.get("question") or example.get("task", "")
        true_answer = example.get("true_answer")
        collector = TrajectoryCollector(task_description=question)
        episode_id = collector.trajectory.episode_id
        start_time = datetime.now().isoformat(timespec="seconds")

        final_answer, error = None, None
        planner = await create_agent(self.config, pre_created_tools=self.pre_created_tools)
        try:
            with trajectory_collection_context(collector):
                final_answer = await planner.run(task=build_question(example))
        except Exception as e:
            logger.error(f"Episode {episode_id} failed: {e}")
            logger.exception(e)
            error = str(e)

        trajectory = collector.get_trajectory()
        if trajectory.num_steps == 0:
            logger.warning(f"Episode {episode_id} made no executor_tool call; it will be skipped.")

        # Verifiable terminal utility U(tau).
        evaluation = {"reward": 0, "is_correct": False, "rationale": "No answer or no ground truth", "method": None}
        if final_answer is not None and true_answer is not None:
            evaluation = await evaluate_answer(
                question=question,
                pred_answer=final_answer,
                ground_truth=true_answer,
                method=self.eval_method,
                model=self.eval_model,
            )
        collector.set_terminal_reward(evaluation["reward"])
        logger.info(
            f"Episode {episode_id}: {trajectory.num_steps} meta-steps, U(tau)={evaluation['reward']} "
            f"({evaluation['rationale'][:120]})"
        )

        if self.save_path:
            append_jsonl(
                {
                    "task_id": example.get("task_id", episode_id),
                    "episode_id": episode_id,
                    "question": question,
                    "prediction": None if final_answer is None else str(final_answer),
                    "true_answer": true_answer,
                    "is_correct": evaluation["is_correct"],
                    "evaluation": evaluation,
                    "workflows": trajectory.executed_agents,
                    "step_rewards": trajectory.step_rewards,
                    "agent_error": error,
                    "start_time": start_time,
                    "end_time": datetime.now().isoformat(timespec="seconds"),
                },
                self.save_path,
            )
        return trajectory

    async def collect_batch(self, examples: List[Dict[str, Any]]) -> List[Trajectory]:
        """Collect one batch of rollouts (at most ``rollout_concurrency`` at a time)."""
        semaphore = asyncio.Semaphore(max(1, self.rollout_concurrency))

        async def _collect(example):
            async with semaphore:
                return await self.collect_single_episode(example)

        results = await asyncio.gather(*[_collect(example) for example in examples], return_exceptions=True)
        trajectories = []
        for example, result in zip(examples, results):
            if isinstance(result, Exception):
                logger.error(f"Rollout failed for task {example.get('task_id')}: {result}")
                trajectories.append(Trajectory(task_description=example.get("question", "")))
            else:
                trajectories.append(result)
        logger.info(
            f"Collected {len(trajectories)} trajectories "
            f"({sum(t.num_steps > 0 for t in trajectories)} non-empty)"
        )
        if self.save_path:
            self._save_trajectories(trajectories)
        return trajectories

    def _save_trajectories(self, trajectories: List[Trajectory]) -> None:
        """Append serialised trajectories (tensors converted to lists) to ``<save_path>_trajectories.jsonl``."""
        path = Path(self.save_path)
        path = path.with_name(path.stem + "_trajectories.jsonl")

        def to_list(x):
            return x.detach().cpu().tolist() if isinstance(x, torch.Tensor) else x

        for trajectory in trajectories:
            append_jsonl(
                {
                    "episode_id": trajectory.episode_id,
                    "task_description": trajectory.task_description,
                    "states": trajectory.states,
                    "sampled_indices": trajectory.sampled_indices,
                    "probs": [[to_list(p) for p in step] for step in trajectory.probs],
                    "executed_agents": trajectory.executed_agents,
                    "step_rewards": trajectory.step_rewards,
                    "terminal_reward": trajectory.terminal_reward,
                    "costs": trajectory.costs,
                },
                str(path),
            )
