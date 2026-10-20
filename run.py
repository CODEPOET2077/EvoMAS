"""Run EvoMAS on a single task or on a JSONL task file (inference, paper Algorithm 1).

Examples:
    python run.py --config configs/evomas.py --task "How many encoder layers does BERT-base have?"
    python run.py --config configs/evomas.py --dataset data/example_tasks.jsonl
    python run.py --config configs/evomas.py --dataset data/example_tasks.jsonl \
        --cfg-options workflow_adapter_config.resume_path=workdir/evomas_train/checkpoints/best
"""

import argparse
import asyncio
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

from mmengine import DictAction

warnings.simplefilter("ignore", DeprecationWarning)
ROOT = str(Path(__file__).resolve().parent)
sys.path.append(ROOT)

from src.atomic_agents import create_agent  # noqa: E402
from src.config import config  # noqa: E402
from src.logger import logger  # noqa: E402
from src.metric import evaluate_answer  # noqa: E402
from src.models import model_manager  # noqa: E402
from src.optimizer.training.data_collector import append_jsonl, build_question  # noqa: E402
from src.registry import DATASET  # noqa: E402
from src.tools.executor_tool import ExecutorTool  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run EvoMAS")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "evomas.py"), help="Config file path.")
    parser.add_argument("--task", default=None, help="Solve a single task given on the command line.")
    parser.add_argument("--dataset", default=None, help="JSONL task file (overrides `dataset.path`).")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N tasks.")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config entries, e.g. workflow_adapter_config.num_layers=2",
    )
    return parser.parse_args()


async def solve(example, executor_tool):
    """Run the Planner on one task; the shared executor tool holds the Workflow Adapter."""
    executor_tool.reset_execution_history()
    planner = await create_agent(config, pre_created_tools={"executor_tool": executor_tool})
    start_time = datetime.now().isoformat(timespec="seconds")
    try:
        prediction, error = await planner.run(task=build_question(example)), None
    except Exception as e:
        logger.exception(e)
        prediction, error = None, str(e)

    evaluation = None
    if example.get("true_answer") is not None and prediction is not None:
        evaluation = await evaluate_answer(example["question"], prediction, example["true_answer"])

    record = {
        "task_id": example.get("task_id"),
        "question": example["question"],
        "prediction": None if prediction is None else str(prediction),
        "true_answer": example.get("true_answer"),
        "evaluation": evaluation,
        "agent_error": error,
        "start_time": start_time,
        "end_time": datetime.now().isoformat(timespec="seconds"),
    }
    append_jsonl(record, config.save_path)
    logger.info(f"| Task {record['task_id']}: prediction={record['prediction']!r}")
    return record


async def main():
    args = parse_args()
    task, dataset_path, limit = args.task, args.dataset, args.limit
    args.task = args.dataset = args.limit = None  # Not config entries.
    config.init_config(args.config, args)
    logger.init_logger(log_path=config.log_path)
    logger.info(f"| Config:\n{config.pretty_text}")
    model_manager.init_models()

    if task is not None:
        examples = [{"task_id": "cli", "question": task, "true_answer": None, "file_name": ""}]
    else:
        dataset_cfg = dict(config.dataset)
        if dataset_path:
            dataset_cfg["path"] = dataset_path
        examples = list(DATASET.build(dataset_cfg).data)
    examples = examples[:limit] if limit else examples
    logger.info(f"| Running EvoMAS on {len(examples)} task(s)")

    # One executor tool (and thus one Workflow Adapter + Evaluator) is shared across tasks.
    executor_tool = ExecutorTool(use_evaluator=config.executor_tool_config.get("use_evaluator", True))
    records = [await solve(example, executor_tool) for example in examples]

    scored = [r for r in records if r["evaluation"] is not None]
    if scored:
        accuracy = sum(r["evaluation"]["is_correct"] for r in scored) / len(scored)
        logger.info(f"| Success rate: {accuracy:.4f} ({len(scored)} scored tasks)")
    logger.info(f"| Results saved to {config.save_path}")


if __name__ == "__main__":
    asyncio.run(main())
