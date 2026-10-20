# EvoMAS-4: the 4-agent candidate pool of paper Table 1 / Table 4.
from mmengine.config import read_base

with read_base():
    from .evomas import *  # noqa: F401,F403

tag = "evomas_4agents"

workflow_adapter_config.update(
    available_agents=["io_agent", "early_exit_agent", "web_search_agent", "sc_ensemble_agent"],
)
