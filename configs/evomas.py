# EvoMAS default configuration (paper Section 4): 7-agent candidate pool, 3-layer workflows.
#
#   python run.py --config configs/evomas.py --task "..."
#   python run.py --config configs/evomas.py --dataset data/example_tasks.jsonl
from mmengine.config import read_base

with read_base():
    from .base import *  # noqa: F401,F403  tool configs and candidate agents

# Planner / Evaluator backbone (paper Table 6).
PLANNER_MODEL = "gpt-4o-mini"
EVALUATOR_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
tag = "evomas"
workdir = "workdir"
log_path = "log.txt"
save_path = "results.jsonl"

dataset = dict(type="jsonl_dataset", path="data/example_tasks.jsonl")

# ---------------------------------------------------------------------------
# Task-state construction (paper Section 3.1)
# ---------------------------------------------------------------------------
# Planner: maintains the subtask plan and calls `executor_tool` once per execution stage.
planner_config = dict(
    type="planning_agent",
    name="planning_agent",
    model_id=PLANNER_MODEL,
    description="A planning agent that maintains the subtask plan and invokes executor_tool per step.",
    max_steps=20,
    template_path="src/task_state/plan_agent/prompts/planning_agent.yaml",
    provide_run_summary=True,
    tools=["planning_tool", "executor_tool", "final_answer_tool"],
)

# Evaluator: tool-augmented verification of the active subtask (paper Appendix C.2).
evaluator_config = dict(
    type="judge_agent",
    name="judge_agent",
    model_id=EVALUATOR_MODEL,
    description="An evaluation agent that assesses task execution quality and provides structured feedback.",
    max_steps=20,
    template_path="src/task_state/judge_agent/prompts/judge_agent.yaml",
    provide_run_summary=True,
    tools=["final_judge_answer_tool", "deep_researcher_tool", "deep_analyzer_tool"],
)

# Updater + workflow execution of one meta-step.
executor_tool_config = dict(
    type="executor_tool",
    use_evaluator=True,  # Call the Evaluator after every stage (its feedback enters the task state).
)

# The top-level agent that receives the task.
agent_config = planner_config

# ---------------------------------------------------------------------------
# Workflow Adapter (paper Section 3.2, Appendix D)
# ---------------------------------------------------------------------------
workflow_adapter_config = dict(
    # True: learned policy | False: every agent in every layer | 'random': random agents.
    enabled=True,
    # Candidate agent pool P (EvoMAS-7, paper Table 4).
    available_agents=[
        "io_agent",
        "sc_ensemble_agent",
        "self_refine_agent",
        "web_search_agent",
        "multi_generate_agent",
        "web_browser_agent",
        "early_exit_agent",
    ],
    num_layers=3,            # Maximum workflow depth L.
    agents_per_layer=None,   # None: adaptive layer width from soft sampling.
    selector_type="attention",  # 'attention' (cross-attention, default) | 'mlp'.
    sampling=dict(method="soft", params=dict(threshold=0.25)),  # Cumulative-mass threshold rho.
    temperature=1.0,         # Softmax temperature lambda (Eq. 11).
    d_model=384,             # MiniLM embedding size.
    num_heads=4,
    dropout=0.1,
    hidden_dim=128,          # Only used by the MLP selector.
    # Execution-validity rules for the first layer: aggregators / refiners need upstream outputs
    # and an Early-exit cannot be the first action of a stage.
    first_layer_constraints=dict(
        forbidden=["early_exit_agent", "self_refine_agent", "sc_ensemble_agent", "summary_agent"],
        required=["io_agent"],
    ),
    resume_path=None,        # Path to a trained checkpoint directory, e.g. workdir/evomas_train/checkpoints/best
)
