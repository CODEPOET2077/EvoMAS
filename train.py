"""Train the Workflow Adapter with REINFORCE (paper Section 3.3).

Examples:
    python train.py --config configs/evomas_train.py
    python train.py --config configs/evomas_train.py --cfg-options optimizer_config.reward_mode=process
"""

import argparse
import asyncio
import os
import sys
import warnings
from pathlib import Path

from mmengine import DictAction

warnings.simplefilter("ignore", DeprecationWarning)
ROOT = str(Path(__file__).resolve().parent)
sys.path.append(ROOT)

from src.config import config  # noqa: E402
from src.logger import logger  # noqa: E402
from src.models import model_manager  # noqa: E402
from src.optimizer.training.swanlab_logger import init_logger  # noqa: E402
from src.optimizer.training.trainer import load_resume_run_id, train_workflow_adapter  # noqa: E402
from src.registry import DATASET  # noqa: E402
from src.tools.executor_tool import ExecutorTool  # noqa: E402
from src.workflow_adapter import WorkflowAdapter  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train the EvoMAS Workflow Adapter")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "evomas_train.py"), help="Config file path.")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config entries, e.g. optimizer_config.learning_rate=5e-5",
    )
    return parser.parse_args()


def load_tasks():
    """Tasks with a ground-truth answer, sorted by id for a reproducible split."""
    tasks = [t for t in DATASET.build(config.dataset).data if t.get("true_answer") is not None]
    if not tasks:
        raise ValueError("No training tasks with `true_answer` found in the dataset.")
    return sorted(tasks, key=lambda t: str(t.get("task_id", t["question"])))


async def main():
    args = parse_args()
    config.init_config(args.config, args)
    logger.init_logger(log_path=config.log_path)
    logger.info(f"| Config:\n{config.pretty_text}")
    model_manager.init_models()

    opt = config.optimizer_config
    if opt.get("training_mode") != "workflow_adapter":
        raise ValueError("Set optimizer_config.training_mode='workflow_adapter' to train the Workflow Adapter.")

    checkpoint_dir = Path(opt.get("checkpoint_dir") or Path(config.exp_path) / "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tracker = init_logger(
        enabled=opt.get("swanlab_enabled", False),
        project=opt.get("swanlab_project", "EvoMAS"),
        experiment_name=opt.get("swanlab_experiment_name"),
        config=dict(optimizer=dict(opt), workflow_adapter=dict(config.workflow_adapter_config)),
        run_id=opt.get("swanlab_run_id") or load_resume_run_id(checkpoint_dir, opt.get("resume_from_epoch")),
    )

    # One Workflow Adapter shared by every rollout; its controller holds the trainable parameters.
    workflow_adapter = WorkflowAdapter(config=config)
    executor_tool = ExecutorTool(
        workflow_adapter=workflow_adapter,
        use_evaluator=config.executor_tool_config.get("use_evaluator", True),
    )
    tasks = load_tasks()
    logger.info(f"| Loaded {len(tasks)} tasks")

    try:
        await train_workflow_adapter(
            config=config,
            tasks=tasks,
            workflow_adapter=workflow_adapter,
            pre_created_tools={"executor_tool": executor_tool},
            checkpoint_dir=checkpoint_dir,
        )
    finally:
        tracker.finish()


if __name__ == "__main__":
    asyncio.run(main())
