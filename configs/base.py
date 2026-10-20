# Shared building blocks: tool configs and the candidate agent pool (paper Appendix C.4 / D.2).
# Task-level configs (configs/evomas*.py) inherit from this file.

# Backbone of the candidate (execution) agents. The main experiments use GPT-4o-mini;
# switching to "gpt-4o" reproduces the backbone-sensitivity setting (paper Table 2).
EXECUTION_MODEL = "gpt-4o-mini"
# Backbone of the analysis / research tools used by the Evaluator.
TOOL_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
browser = dict(
    use_proxy=False,
    proxy=dict(server="", username="", password=""),
    headless=True,
    disable_security=False,  # Keep browser web-security on when visiting arbitrary pages.
    extra_chromium_args=None,
    chrome_instance_path=None,
    wss_url=None,
    cdp_url=None,
    max_content_length=5000,
    browser_window_size=dict(width=1280, height=720),
    new_context_config=None,
)

web_fetcher_tool_config = dict(type="web_fetcher_tool")

web_searcher_tool_config = dict(
    type="web_searcher_tool",
    engine="Firecrawl",  # Options: "Firecrawl", "Google", "Bing", "DuckDuckGo", "Baidu"
    retry_delay=10,
    max_retries=3,
    lang="en",
    country="us",
    num_results=5,
    fetch_content=True,
    max_length=4096,
)

deep_analyzer_tool_config = dict(
    type="deep_analyzer_tool",
    analyzer_model_ids=[TOOL_MODEL],  # More than one model -> ensemble analysis.
    summarizer_model_id=TOOL_MODEL,
)

deep_researcher_tool_config = dict(
    type="deep_researcher_tool",
    model_id=TOOL_MODEL,
    max_depth=2,
    max_insights=20,
    time_limit_seconds=60,
    max_follow_ups=3,
)

file_reader_tool_config = dict(type="file_reader_tool")
python_interpreter_tool_config = dict(type="python_interpreter_tool")
custom_browser_tool_config = dict(type="custom_browser_tool")

# ---------------------------------------------------------------------------
# Candidate agents used in the paper's pools (Table 3 / Table 4)
# ---------------------------------------------------------------------------
io_agent_config = dict(
    type="io_agent",
    name="io_agent",
    model_id=EXECUTION_MODEL,
    description="A knowledge-based assistant that answers questions using model's internal knowledge.",
    max_steps=1,
    template_path="src/atomic_agents/io_agent/prompts/io_agent.yaml",
)

multi_generate_agent_config = dict(
    type="multi_generate_agent",
    name="multi_generate_agent",
    model_id=EXECUTION_MODEL,
    description="A multi-generate agent that can generate multiple responses for the same task.",
    max_steps=1,
    template_path="src/atomic_agents/multi_generate_agent/prompts/multi_generate_agent.yaml",
    model_temperature=0.7,
)

self_refine_agent_config = dict(
    type="self_refine_agent",
    name="self_refine_agent",
    model_id=EXECUTION_MODEL,
    description="A self-refinement agent that iteratively improves outputs through self-critique and revision.",
    max_steps=1,
    template_path="src/atomic_agents/self_refine_agent/prompts/self_refine_agent.yaml",
)

web_search_agent_config = dict(
    type="web_search_agent",
    name="web_search_agent",
    model_id=EXECUTION_MODEL,
    description="A web search agent that can search the web for information.",
    max_steps=5,
    template_path="src/atomic_agents/web_search_agent/prompts/web_search_agent.yaml",
    tools=["web_searcher_tool", "final_answer_tool"],
)

web_browser_agent_config = dict(
    type="web_browser_agent",
    name="web_browser_agent",
    model_id=EXECUTION_MODEL,
    description="A web browser agent that can browse the web and provide information.",
    max_steps=5,
    template_path="src/atomic_agents/web_browser_agent/prompts/web_browser_agent.yaml",
    tools=["custom_browser_tool", "final_answer_tool"],
)

early_exit_agent_config = dict(
    type="early_exit_agent",
    name="early_exit_agent",
    model_id=EXECUTION_MODEL,
    description="An agent that determines if a task can be completed early or needs further processing.",
    max_steps=1,
    template_path="src/atomic_agents/early_exit_agent/prompts/early_exit_agent.yaml",
)

sc_ensemble_agent_config = dict(
    type="sc_ensemble_agent",
    name="sc_ensemble_agent",
    model_id=EXECUTION_MODEL,
    description="A solution evaluation agent that can evaluate multiple solutions to the same problem and "
                "identify the most consistent answer.",
    max_steps=1,
    template_path="src/atomic_agents/sc_ensemble_agent/prompts/sc_ensemble_agent.yaml",
)

# ---------------------------------------------------------------------------
# Auxiliary agents (not part of the default pools; can be added to `available_agents`)
# ---------------------------------------------------------------------------
summary_agent_config = dict(
    type="summary_agent",
    name="summary_agent",
    model_id=EXECUTION_MODEL,
    description="A summarization agent that extracts key information and provides concise summaries.",
    max_steps=1,
    template_path="src/atomic_agents/summary_agent/prompts/summary_agent.yaml",
)

generate_query_agent_config = dict(
    type="generate_query_agent",
    name="generate_query_agent",
    model_id=EXECUTION_MODEL,
    description="A query generation agent that can generate follow-up queries for further exploration "
                "based on existing solutions",
    max_steps=1,
    template_path="src/atomic_agents/generate_query_agent/prompts/generate_query_agent.yaml",
)

file_read_agent_config = dict(
    type="file_read_agent",
    name="file_read_agent",
    model_id=EXECUTION_MODEL,
    description="A file read agent that can read files and provide information.",
    max_steps=3,
    template_path="src/atomic_agents/file_read_agent/prompts/file_read_agent.yaml",
    tools=["file_reader_tool", "final_answer_tool"],
)

code_agent_config = dict(
    type="code_agent",
    name="code_agent",
    model_id=EXECUTION_MODEL,
    description="A code agent that can solve any task using code blobs.",
    max_steps=3,
    template_path="src/atomic_agents/code_agent/prompts/code_agent.yaml",
    tools=["python_interpreter_tool", "final_answer_tool"],
)

tool_agent_config = dict(
    type="tool_agent",
    name="tool_agent",
    model_id=EXECUTION_MODEL,
    description="A tool-using agent that can execute various tools to accomplish tasks.",
    max_steps=10,
    template_path="src/atomic_agents/tool_agent/prompts/tool_agent.yaml",
    provide_run_summary=True,
    tools=["python_interpreter_tool", "final_answer_tool"],
)
