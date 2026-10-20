"""REINFORCE training loop for the Workflow Adapter (paper Section 3.3).

Each optimisation step: roll out a batch of training tasks with the current policy, compute the
per-step returns (terminal utility, or process reward), and update ``Theta`` with Eq. 16.
Validation success on the held-out split is tracked throughout training (paper Figure 3b).
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.logger import logger
from src.optimizer.policy_gradient import ReinforceLoss, RewardProcessor
from src.optimizer.training.data_collector import DataCollector
from src.optimizer.training.swanlab_logger import get_logger
from src.workflow_adapter import WorkflowAdapter


def split_train_val(tasks: List[Dict], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Random train / validation split (8:2 in the paper)."""
    shuffled = list(tasks)
    random.Random(seed).shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    logger.info(f"Dataset split: {split} train / {len(shuffled) - split} validation")
    return shuffled[:split], shuffled[split:]


class WorkflowAdapterTrainer:
    """Args:
        config: Global config; reads ``optimizer_config``.
        workflow_adapter: The shared Workflow Adapter whose controller parameters are optimised.
        pre_created_tools: Shared tool instances (the ``executor_tool`` wrapping ``workflow_adapter``).
    """

    def __init__(self, config: Any, workflow_adapter: WorkflowAdapter, pre_created_tools: Dict[str, Any]):
        opt = config.optimizer_config
        self.config = config
        self.workflow_adapter = workflow_adapter
        self.data_collector = DataCollector(config, pre_created_tools=pre_created_tools)
        self.reward_processor = RewardProcessor(
            reward_mode=opt.get("reward_mode", "terminal"), gamma=opt.get("gamma", 0.99)
        )
        self.loss_fn = ReinforceLoss(
            log_prob_mode=opt.get("log_prob_mode", "sequence"), cost_weight=opt.get("cost_weight", 0.0)
        )
        self.num_epochs = opt.get("num_epochs", 10)
        self.batch_size = opt.get("batch_size", 4)
        self.val_frequency = opt.get("val_frequency", 2)
        self.optimizer = torch.optim.Adam(
            list(workflow_adapter.controller.parameters()), lr=opt.get("learning_rate", 1e-4)
        )
        logger.info(
            f"WorkflowAdapterTrainer: reward_mode={self.reward_processor.reward_mode}, "
            f"log_prob_mode={self.loss_fn.log_prob_mode}, "
            f"parameters={workflow_adapter.controller.get_num_parameters()['total']:,}"
        )

    async def train_step(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """One REINFORCE update on a batch of rollouts."""
        trajectories = await self.data_collector.collect_batch(batch)
        returns = self.reward_processor.compute_returns_batch(trajectories)
        loss = self.loss_fn.batch_loss(trajectories, returns)
        if loss is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        valid = [t for t in trajectories if t.num_steps > 0]
        return {
            "loss": loss.item() if loss is not None else 0.0,
            "success_rate": sum(t.terminal_reward for t in valid) / len(valid) if valid else 0.0,
            "avg_meta_steps": sum(t.num_steps for t in valid) / len(valid) if valid else 0.0,
            "valid_ratio": len(valid) / len(trajectories) if trajectories else 0.0,
        }

    async def validate(self, val_tasks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Success rate on the held-out split (no parameter update)."""
        trajectories = await self.data_collector.collect_batch(val_tasks)
        num_correct = sum(t.terminal_reward > 0.5 for t in trajectories)
        accuracy = num_correct / len(trajectories) if trajectories else 0.0
        logger.info(f"Validation: {num_correct}/{len(trajectories)} correct (accuracy={accuracy:.4f})")
        return {"accuracy": accuracy, "num_correct": num_correct, "num_tasks": len(trajectories)}

    async def fit(
        self,
        train_tasks: List[Dict[str, Any]],
        val_tasks: List[Dict[str, Any]],
        checkpoint_dir: Path,
        start_epoch: int = 0,
    ) -> None:
        tracker = get_logger()
        num_batches = (len(train_tasks) + self.batch_size - 1) // self.batch_size
        global_step = start_epoch * num_batches
        best_val = -1.0

        if start_epoch == 0 and val_tasks:
            metrics = await self.validate(val_tasks)
            if tracker:
                tracker.log({f"val/{k}": v for k, v in metrics.items()}, step=0)

        for epoch in range(start_epoch, self.num_epochs):
            logger.info(f"===== Epoch {epoch + 1}/{self.num_epochs} =====")
            epoch_losses = []
            for batch_idx in range(num_batches):
                batch = train_tasks[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
                global_step += 1
                metrics = await self.train_step(batch)
                epoch_losses.append(metrics["loss"])
                logger.info(f"[step {global_step}] {metrics}")
                if tracker:
                    tracker.log({f"train/{k}": v for k, v in metrics.items()}, step=global_step)

                if val_tasks and global_step % self.val_frequency == 0:
                    val_metrics = await self.validate(val_tasks)
                    if tracker:
                        tracker.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)
                    if val_metrics["accuracy"] > best_val:
                        best_val = val_metrics["accuracy"]
                        self.workflow_adapter.controller.save(str(checkpoint_dir / "best"))
                        logger.info(f"New best validation accuracy {best_val:.4f}; checkpoint saved.")

            epoch_dir = checkpoint_dir / f"epoch_{epoch + 1}"
            self.workflow_adapter.controller.save(str(epoch_dir))
            with open(epoch_dir / "training_metadata.json", "w") as f:
                json.dump(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "train_loss": sum(epoch_losses) / max(len(epoch_losses), 1),
                        "swanlab_run_id": tracker.get_run_id() if tracker else None,
                    },
                    f,
                    indent=2,
                )


async def train_workflow_adapter(
    config: Any,
    tasks: List[Dict[str, Any]],
    workflow_adapter: WorkflowAdapter,
    pre_created_tools: Dict[str, Any],
    checkpoint_dir: Path,
) -> None:
    """Split tasks, optionally resume from a checkpoint, and run the training loop."""
    opt = config.optimizer_config
    train_tasks, val_tasks = split_train_val(
        tasks, train_ratio=opt.get("train_ratio", 0.8), seed=opt.get("random_seed", 42)
    )

    start_epoch = 0
    resume_from_epoch = opt.get("resume_from_epoch", None)
    if resume_from_epoch:
        resume_dir = checkpoint_dir / f"epoch_{resume_from_epoch}"
        if resume_dir.exists():
            workflow_adapter.controller.load(str(resume_dir))
            start_epoch = resume_from_epoch
        else:
            logger.warning(f"Resume checkpoint {resume_dir} not found; training from scratch.")

    trainer = WorkflowAdapterTrainer(config, workflow_adapter, pre_created_tools)
    await trainer.fit(train_tasks, val_tasks, checkpoint_dir, start_epoch=start_epoch)


def load_resume_run_id(checkpoint_dir: Path, resume_from_epoch: Optional[int]) -> Optional[str]:
    """SwanLab run id stored with an epoch checkpoint (to resume the same experiment)."""
    if not resume_from_epoch:
        return None
    metadata_file = checkpoint_dir / f"epoch_{resume_from_epoch}" / "training_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            return json.load(f).get("swanlab_run_id")
    return None
