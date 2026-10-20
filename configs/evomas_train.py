# REINFORCE training of the Workflow Adapter (paper Section 3.3).
#
#   python train.py --config configs/evomas_train.py
#   python train.py --config configs/evomas_train.py --cfg-options optimizer_config.reward_mode=process
from mmengine.config import read_base

with read_base():
    from .evomas import *  # noqa: F401,F403

tag = "evomas_train"
save_path = "rollouts.jsonl"

# Training data: JSONL with `question` and `true_answer` (split 8:2 into train / validation).
dataset = dict(type="jsonl_dataset", path="data/example_tasks.jsonl")

optimizer_config = dict(
    training_mode="workflow_adapter",
    # 'terminal': sparse verifiable reward U(tau) for every meta-step (main setting, Eq. 17).
    # 'process' : evaluator rewards as dense intermediate rewards (PRM analysis, Section 4.4).
    reward_mode="terminal",
    gamma=0.99,               # Discount factor, only used by reward_mode='process'.
    # 'sequence': exact log-probability of the ordered soft-sampling draw (Eq. 28).
    log_prob_mode="sequence",
    cost_weight=0.0,          # Optional penalty on the USD cost of each stage.
    # Terminal-utility scorer: 'llm' (semantic judge) or 'exact' (normalised exact match).
    evaluation=dict(method="llm", model="gpt-4o-mini"),
    num_epochs=4,
    batch_size=4,
    learning_rate=1e-4,
    train_ratio=0.8,
    random_seed=42,
    val_frequency=2,          # Validate every N optimisation steps.
    checkpoint_dir=None,      # Defaults to <workdir>/<tag>/checkpoints.
    resume_from_epoch=None,
    # Experiment tracking (requires `pip install swanlab`).
    swanlab_enabled=False,
    swanlab_project="EvoMAS",
    swanlab_experiment_name=None,
)
