"""Candidate agent pool (paper Section 3.2, Appendix C.4 and D.2).

Agents used in the paper's pools:
    I/O (``io_agent``), Multi-generate (``multi_generate_agent``), Self-refine (``self_refine_agent``),
    Web-search (``web_search_agent``), Web-browser (``web_browser_agent``),
    Early-exit (``early_exit_agent``), Ensemble (``sc_ensemble_agent``).

Auxiliary agents (not part of the default 4-/7-agent pools):
    ``summary_agent`` (alternative aggregator), ``generate_query_agent``, ``file_read_agent``,
    ``code_agent``, ``tool_agent``.
"""

from src.atomic_agents.agent import build_agent, create_agent
from src.atomic_agents.code_agent import CodeAgent
from src.atomic_agents.early_exit_agent import EarlyExitAgent
from src.atomic_agents.file_read_agent import FileReadAgent
from src.atomic_agents.general_agent import GeneralAgent
from src.atomic_agents.generate_query_agent import GenerateQueryAgent
from src.atomic_agents.io_agent import IOAgent
from src.atomic_agents.multi_generate_agent import MultiGenerateAgent
from src.atomic_agents.reformulator import prepare_response
from src.atomic_agents.sc_ensemble_agent import SCEnsembleAgent
from src.atomic_agents.self_refine_agent import SelfRefineAgent
from src.atomic_agents.summary_agent import SummaryAgent
from src.atomic_agents.tool_agent import ToolAgent
from src.atomic_agents.web_browser_agent import WebBrowserAgent
from src.atomic_agents.web_search_agent import WebSearchAgent

__all__ = [
    "build_agent",
    "create_agent",
    "prepare_response",
    "GeneralAgent",
    "IOAgent",
    "MultiGenerateAgent",
    "SelfRefineAgent",
    "WebSearchAgent",
    "WebBrowserAgent",
    "EarlyExitAgent",
    "SCEnsembleAgent",
    "SummaryAgent",
    "GenerateQueryAgent",
    "FileReadAgent",
    "CodeAgent",
    "ToolAgent",
]
