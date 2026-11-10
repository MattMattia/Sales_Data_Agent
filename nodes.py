"""
nodes.py - Implementation of all nodes in the Multi-Agent workflow:
1. planner_node: Generates and revises plans with sub-goals (L6 GPA alignment).
2. executor_node: Dynamic step dispatcher, retry accounting (MAX_REPLANS), and replanning router.
3. cortex_agents_research_node: ReAct agent integrating Snowflake Cortex (Analyst & Search).
4. web_research_node: ReAct agent with TavilySearch for public web research.
5. chart_node: ReAct agent for plotting and disk-persisting charts (python_repl_tool).
6. chart_summary_node: ReAct agent generating concise chart captions.
7. synthesizer_node: Terminal synthesis producing an executive prose report with metrics and citations.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Shared state schema, message classes and agent registry (importing state loads `.env`)
from state import (
    AIMessage,
    ExecutorDecision,
    HumanMessage,
    PlanStep,
    State,
    SystemMessage,
    WORKER_AGENTS,
)

# Prompts and system directives import
from prompts import (
    plan_prompt,
    executor_prompt,
    agent_system_prompt,
    CORTEX_RESEARCHER_SYSTEM_PROMPT,
    WEB_RESEARCHER_SYSTEM_PROMPT,
    CHART_GENERATOR_SYSTEM_PROMPT,
    CHART_SUMMARIZER_SYSTEM_PROMPT,
    SYNTHESIZER_INSTRUCTIONS,
)

# Integrated tools import (clean `(str) -> str` wrappers for ReAct agents)
from tools import query_cortex_agent, run_python_code, search_web

logger = logging.getLogger("sales_data_agent.nodes")

# =====================================================================
# Safe Imports & Fallback Compatibility for LangGraph
# =====================================================================

try:
    from langgraph.types import Command
    from langgraph.graph import END
except ImportError:
    END = "__end__"
    class Command:
        def __init__(self, update: Optional[Dict[str, Any]] = None, goto: Any = None):
            self.update = update or {}
            self.goto = goto
        def __class_getitem__(cls, item):
            return cls
        def __repr__(self):
            return f"Command(update={self.update}, goto={self.goto})"

RESEARCH_AGENTS = ("cortex_researcher", "web_researcher")
ALLOWED_CHART_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".svg"})


# =====================================================================
# 1. LLM Configuration (Production & Mock Fallback)
# =====================================================================

class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


class MockChatModel:
    """
    Simulated chat model for local testing or environments without an active OPENAI_API_KEY.
    Ensures planner, executor, and synthesizer generate valid schema outputs.
    """
    def __init__(self, model_name: str = "mock-gpt-4o", json_mode: bool = False):
        self.model_name = model_name
        self.json_mode = json_mode

    def invoke(self, messages: Any) -> MockLLMResponse:
        last_msg = messages[-1] if isinstance(messages, list) else messages
        content = getattr(last_msg, "content", str(last_msg))

        # 1. Planner Response
        if "expert multi-agent task planner" in content or "Allowed Agents for this session" in content:
            # Extract and analyze specifically the user query
            user_query_line = ""
            if "User Query:" in content:
                user_query_line = content.split("User Query:")[1].splitlines()[0].lower()

            needs_chart = any(w in user_query_line for w in ["chart", "plot", "grafic", "gráfico"])
            if needs_chart:
                plan_json = {
                    "1": {
                        "agent": "cortex_researcher",
                        "action": "Retrieve customer deal data and metrics from CRM",
                        "goal": "Extract top customer deals with their values",
                        "pre_conditions": ["Access to CRM sales metrics"],
                        "post_conditions": ["Customer deals table with values available"]
                    },
                    "2": {
                        "agent": "chart_generator",
                        "action": "Plot bar chart of deal values for top customer deals and save to disk",
                        "goal": "Generate and save chart visualization file",
                        "pre_conditions": ["Customer deal values available"],
                        "post_conditions": ["CHART_PATH and CHART_NOTES emitted"]
                    },
                    "3": {
                        "agent": "chart_summarizer",
                        "action": "Generate concise caption and summary of the plotted chart",
                        "goal": "Provide standalone summary of the chart insights",
                        "pre_conditions": ["Valid chart file at CHART_PATH"],
                        "post_conditions": ["Final chart caption produced"]
                    }
                }
            else:
                plan_json = {
                    "1": {
                        "agent": "cortex_researcher",
                        "action": "Retrieve relevant customer deals, metrics, and meeting notes",
                        "goal": "Collect internal enterprise sales context",
                        "pre_conditions": ["Access to Snowflake Cortex service"],
                        "post_conditions": ["CRM metrics and meeting notes retrieved"]
                    },
                    "2": {
                        "agent": "web_researcher",
                        "action": "Search public web for recent market updates and regulatory news",
                        "goal": "Supplement internal data with external market context",
                        "pre_conditions": ["Identified target companies and sectors"],
                        "post_conditions": ["External news and regulatory articles gathered"]
                    },
                    "3": {
                        "agent": "synthesizer",
                        "action": "Synthesize internal deal data and external market intelligence",
                        "goal": "Provide direct, comprehensive prose answer with figures and citations",
                        "pre_conditions": ["Internal metrics and external research collected"],
                        "post_conditions": ["Complete final executive response"]
                    }
                }
            return MockLLMResponse(content=json.dumps(plan_json))

        # 2. Executor Response
        elif "You are the Executor" in content:
            step_match = "1"
            for s in ["1", "2", "3", "4", "5"]:
                if f"Current Step Number: {s}" in content:
                    step_match = s
                    break

            plan_data = {}
            if "- Active Plan:" in content:
                try:
                    plan_chunk = content.split("- Active Plan:")[1].split("- Current Step Number:")[0].strip()
                    plan_data = json.loads(plan_chunk)
                except Exception:
                    plan_data = {}

            target_agent = plan_data.get(step_match, {}).get("agent")
            if not target_agent:
                target_agent = "synthesizer"

            if target_agent == "cortex_researcher":
                decision = {
                    "replan": False,
                    "goto": "cortex_researcher",
                    "reason": f"Step {step_match} calls for cortex_researcher to retrieve CRM sales data.",
                    "query": "Retrieve top 3 customer deals by deal value and meeting notes."
                }
            elif target_agent == "chart_generator":
                decision = {
                    "replan": False,
                    "goto": "chart_generator",
                    "reason": f"Step {step_match} calls for chart_generator to plot the retrieved customer deals.",
                    "query": "Generate a bar chart of top customer deals and save as top_deals_chart.png"
                }
            elif target_agent == "chart_summarizer":
                decision = {
                    "replan": False,
                    "goto": "chart_summarizer",
                    "reason": f"Step {step_match} calls for chart_summarizer to summarize the generated chart.",
                    "query": "Summarize the chart at the provided CHART_PATH"
                }
            elif target_agent == "web_researcher":
                decision = {
                    "replan": False,
                    "goto": "web_researcher",
                    "reason": f"Step {step_match} calls for web_researcher to gather external context.",
                    "query": "Research recent regulatory updates affecting financial services and healthcare deals."
                }
            else:
                decision = {
                    "replan": False,
                    "goto": "synthesizer",
                    "reason": "All prerequisite research steps have concluded.",
                    "query": "Synthesize the final executive answer."
                }
            return MockLLMResponse(content=json.dumps(decision))

        # 3. Synthesizer Response
        elif "You are the Synthesizer" in content:
            return MockLLMResponse(
                content=(
                    "Top 3 Customer Deals Summary:\n\n"
                    "1. HealthTech Corp: $1,250,000 (Stage: Negotiation, Product: Enterprise Compliance)\n"
                    "2. FinTech Global: $890,000 (Stage: Proposal, Product: Risk Analytics API)\n"
                    "3. OmniRetail Systems: $640,000 (Stage: Decision Maker, Product: Inventory Forecasting)\n\n"
                    "Key Takeaways from Customer Meeting Notes:\n"
                    "- Urgency around impending state AI governance and FDA regulatory oversight (HealthTech Corp).\n"
                    "- Near-term compliance deadlines for Open Banking protocols in Q1 (FinTech Global).\n\n"
                    "Citations: [sales_intelligence.data.sales_metrics, meeting_transcript_healthtech]"
                )
            )

        return MockLLMResponse(content="Mock LLM response processed successfully.")


def initialize_chat_model(model_name: str, json_mode: bool = False) -> Any:
    """
    Initializes ChatOpenAI with the required configuration if OPENAI_API_KEY
    is present, or falls back to MockChatModel for local testing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "sk-...":
        try:
            from langchain_openai import ChatOpenAI
            model_kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            return ChatOpenAI(model=model_name, model_kwargs=model_kwargs)
        except ImportError:
            logger.warning("langchain-openai is not installed. Using MockChatModel fallback.")
    else:
        logger.info(f"OPENAI_API_KEY not detected. Initializing MockChatModel ({model_name}).")

    return MockChatModel(model_name=model_name, json_mode=json_mode)


# Model configuration via environment variables
REASONING_MODEL_NAME = os.getenv("REASONING_LLM_MODEL", os.getenv("LLM_MODEL", "gpt-4o"))
BASE_LLM_MODEL_NAME = os.getenv("LLM_MODEL", "gpt-4o")

reasoning_llm = initialize_chat_model(model_name=REASONING_MODEL_NAME, json_mode=True)
llm = initialize_chat_model(model_name=BASE_LLM_MODEL_NAME, json_mode=False)


# =====================================================================
# Factory for ReAct Sub-Agents
# =====================================================================

def build_react_agent(model: Any, tools: List[Any], prompt: SystemMessage) -> Any:
    """
    Constructs a standard ReAct agent using LangGraph prebuilt or internal fallback runner.
    """
    try:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(model, tools, prompt=prompt)
    except Exception as e:
        logger.warning(f"Using internal ReAct runner (create_react_agent not available: {e})")

        class InternalReActAgent:
            def __init__(self, agent_model, agent_tools, agent_prompt):
                self.model = agent_model
                self.tools = agent_tools
                self.prompt = agent_prompt

            def invoke(self, state_or_dict: Any) -> Dict[str, Any]:
                prompt_content = getattr(self.prompt, "content", "")

                # Special Case: Chart Generator Agent
                if "generate charts" in prompt_content:
                    chart_path = "top_deals_chart.png"
                    try:
                        import matplotlib.pyplot as plt
                        companies = ['HealthTech Corp', 'FinTech Global', 'OmniRetail Systems']
                        values = [1.25, 0.89, 0.64]
                        plt.figure(figsize=(7, 4))
                        plt.bar(companies, values, color=['#2b5c8f', '#4682b4', '#6baed6'])
                        plt.title('Top 3 Customer Deals ($M)')
                        plt.ylabel('Deal Value ($ Millions)')
                        plt.tight_layout()
                        plt.savefig(chart_path)
                        plt.close()
                    except Exception:
                        # Fallback: create a valid minimal PNG file on disk
                        with open(chart_path, "wb") as f:
                            f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')

                    tool_result_str = (
                        f"Chart generated and saved to current working directory.\n"
                        f"CHART_PATH: {chart_path}\n"
                        "CHART_NOTES: HealthTech Corp holds the largest deal value at $1.25M, followed by FinTech Global ($890k) and OmniRetail ($640k)."
                    )
                    return {"messages": [AIMessage(content=tool_result_str, name="chart_generator")]}

                # Special Case: Chart Summarizer Agent
                if "generate image captions" in prompt_content:
                    tool_result_str = (
                        "HealthTech Corp leads the customer pipeline with a deal valuation of $1.25M, "
                        "followed by FinTech Global at $890k and OmniRetail Systems at $640k. "
                        "The top three accounts collectively represent significant near-term enterprise value."
                    )
                    return {"messages": [AIMessage(content=tool_result_str, name="chart_summarizer")]}

                # Extract query for research agents
                if isinstance(state_or_dict, dict):
                    msgs = state_or_dict.get("agent_query") or state_or_dict.get("messages", "")
                else:
                    msgs = getattr(state_or_dict, "agent_query", None) or getattr(state_or_dict, "messages", "")

                query = msgs if isinstance(msgs, str) else (
                    msgs[-1].content if (isinstance(msgs, list) and msgs and hasattr(msgs[-1], 'content'))
                    else str(msgs)
                )

                tool_result_str = ""
                if self.tools:
                    target_tool = self.tools[0]
                    try:
                        if callable(target_tool):
                            res = target_tool(query)
                        elif hasattr(target_tool, "run"):
                            res = target_tool.run(query)
                        elif hasattr(target_tool, "invoke"):
                            res = target_tool.invoke(query)
                        else:
                            res = str(target_tool)

                        if isinstance(res, tuple):
                            # (text, citations, sql, results_str) from CortexAgentTool
                            text, citations, sql, results_str = res
                            tool_result_str = f"{text}\n\nResults:\n{results_str}\n\nCitations: {citations}"
                        else:
                            tool_result_str = str(res)
                    except Exception as err:
                        tool_result_str = f"Tool execution error: {err}"
                else:
                    tool_result_str = f"Agent processed context: {query}"

                return {
                    "messages": [
                        AIMessage(content=tool_result_str, name="sub_agent")
                    ]
                }

        return InternalReActAgent(model, tools, prompt)


# ReAct Sub-Agent Instantiations (tools expose clean `(str) -> str` signatures)
cortex_agent = build_react_agent(
    model=llm,
    tools=[query_cortex_agent],
    prompt=agent_system_prompt(CORTEX_RESEARCHER_SYSTEM_PROMPT),
)

web_search_agent = build_react_agent(
    model=llm,
    tools=[search_web],
    prompt=agent_system_prompt(WEB_RESEARCHER_SYSTEM_PROMPT),
)

chart_agent = build_react_agent(
    model=llm,
    tools=[run_python_code],
    prompt=agent_system_prompt(CHART_GENERATOR_SYSTEM_PROMPT),
)

chart_summary_agent = build_react_agent(
    model=llm,
    tools=[],
    prompt=agent_system_prompt(CHART_SUMMARIZER_SYSTEM_PROMPT),
)




# =====================================================================
# 2. Parsing, Validation & Safety Helpers
# =====================================================================

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_CHART_PATH_RE = re.compile(r"^\s*CHART_PATH:\s*(.+?)\s*$", re.MULTILINE)
_CHART_NOTES_RE = re.compile(r"^\s*CHART_NOTES:\s*(.+?)\s*$", re.MULTILINE)


def _parse_json_safely(raw_text: str) -> Dict[str, Any]:
    """
    Parses a JSON object emitted by an LLM, tolerating Markdown code fences
    (```json ... ```) and surrounding prose. Raises ValueError otherwise.
    """
    text = (raw_text or "").strip()
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    elif not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Response JSON must be an object.")
    return parsed


def _message_text(reply: Any) -> str:
    content = getattr(reply, "content", reply)
    return content if isinstance(content, str) else str(content)


def _allowed_agents(state: State) -> set:
    """Worker agents routable in this session; 'synthesizer' is always allowed to terminate."""
    enabled = state.get("enabled_agents") or list(WORKER_AGENTS)
    return (set(enabled) & WORKER_AGENTS) | {"synthesizer"}


def _validate_plan(raw_plan: Dict[str, Any], allowed: set) -> Dict[str, Dict[str, Any]]:
    """
    Validates each step with PlanStep, drops steps assigned to unknown or
    disabled agents, and renumbers the remaining steps from "1".
    """
    if "plan" in raw_plan and isinstance(raw_plan["plan"], dict):
        raw_plan = raw_plan["plan"]

    def order(key: Any) -> Any:
        return (0, int(key)) if str(key).isdigit() else (1, str(key))

    valid_steps: List[Dict[str, Any]] = []
    for key in sorted(raw_plan, key=order):
        try:
            step = PlanStep.model_validate(raw_plan[key])
        except Exception as exc:
            logger.warning(f"Dropping invalid plan step {key!r}: {exc}")
            continue
        if step.agent not in allowed:
            logger.warning(f"Dropping plan step {key!r}: agent {step.agent!r} is not enabled.")
            continue
        valid_steps.append(step.model_dump())

    if not valid_steps:
        logger.warning("Planner produced no valid step. Falling back to the synthesizer.")
        fallback = PlanStep(agent="synthesizer", action="Answer the user query with the available context.")
        valid_steps = [fallback.model_dump()]

    return {str(index): step for index, step in enumerate(valid_steps, start=1)}


def _plan_step(plan: Dict[str, Any], step: int) -> Dict[str, Any]:
    return plan.get(str(step)) or plan.get(step) or {}


def _step_instruction(step_data: Dict[str, Any]) -> str:
    action = step_data.get("action", "")
    goal = step_data.get("goal")
    return f"{action} (goal: {goal})" if goal else action


def _routable_agent(agent: Optional[str], allowed: set) -> str:
    return agent if agent in allowed else "synthesizer"


def safe_chart_path(raw_path: Optional[str], base_dir: Optional[Path] = None) -> Optional[str]:
    """
    Returns `raw_path` normalized when it is a relative image path that resolves
    inside `base_dir` (defaults to the current working directory), else None.
    """
    if not raw_path:
        return None
    candidate = raw_path.strip().strip("`'\"")
    path = Path(candidate)
    if not candidate or path.is_absolute() or path.drive or path.root or ".." in path.parts:
        return None
    if path.suffix.lower() not in ALLOWED_CHART_SUFFIXES:
        return None
    base = (base_dir or Path.cwd()).resolve()
    try:
        (base / path).resolve().relative_to(base)
    except ValueError:
        return None
    return path.as_posix()


def _research_findings(state: State) -> List[str]:
    return [
        _message_text(m) for m in state.get("messages", [])
        if getattr(m, "name", None) in RESEARCH_AGENTS
    ]


def _user_question(state: State) -> str:
    question = state.get("user_query")
    if not question and state.get("messages"):
        question = getattr(state["messages"][0], "content", "")
    return question or ""


def _synthesize_answer(state: State, context_messages: List[str]) -> str:
    context_str = "\n\n---\n\n".join(context_messages) if context_messages else "No research messages recorded."
    summary_prompt = [
        HumanMessage(
            content=(
                f"User question: {_user_question(state)}\n\n"
                f"{SYNTHESIZER_INSTRUCTIONS}\n\n"
                f"Context:\n\n{context_str}"
            )
        )
    ]
    return _message_text(llm.invoke(summary_prompt)).strip()


# =====================================================================
# 3. Orchestrator Nodes: planner_node and executor_node
# =====================================================================

def planner_node(state: State) -> Command[Literal["executor"]]:
    """
    Executes the reasoning LLM with plan_prompt(state) and stores the
    PlanStep-validated multi-step plan (with L6 sub-goals) into state.
    Revised plans are renumbered from "1", so execution restarts at step 1.
    """
    prompt = plan_prompt(state)
    content_str = _message_text(reasoning_llm.invoke([prompt]))

    try:
        raw_plan = _parse_json_safely(content_str)
    except ValueError as exc:
        raise ValueError(f"Planner returned invalid JSON:\n{content_str}") from exc

    replan = bool(state.get("replan_flag", False))
    plan = _validate_plan(raw_plan, _allowed_agents(state))

    update_payload = {
        "plan": plan,
        "messages": [
            AIMessage(
                content=json.dumps(plan),
                name="replan" if replan else "initial_plan"
            )
        ],
        "user_query": _user_question(state),
        "current_step": 1,
        "replan_flag": replan,
        "last_reason": state.get("last_reason", "") if replan else "",
        "enabled_agents": state.get("enabled_agents"),
    }

    return Command(update=update_payload, goto="executor")


MAX_REPLANS = 3

def executor_node(
    state: State,
) -> Command[Literal["cortex_researcher", "web_researcher", "chart_generator", "chart_summarizer", "synthesizer", "planner"]]:
    """
    Dynamic controller of the multi-agent plan lifecycle:
    1. If recently replanned (replan_flag=True), dispatches to the planned agent without reconsidering,
       with a fresh agent_query built from the revised step.
    2. Otherwise, consults the LLM with executor_prompt and validates its ExecutorDecision.
    3. Enforces bounded retry accounting (MAX_REPLANS = 3 per step).
    """
    plan: Dict[str, Any] = state.get("plan") or {}
    step: int = state.get("current_step", 1)
    allowed = _allowed_agents(state)

    # 0) If just replanned, execute the revised step once before re-evaluating
    if state.get("replan_flag"):
        step_data = _plan_step(plan, step) or _plan_step(plan, 1)
        planned_agent = _routable_agent(step_data.get("agent"), allowed)
        instruction = _step_instruction(step_data) or state.get("user_query", "")
        return Command(
            update={
                "replan_flag": False,
                "current_step": step + 1,
                "agent_query": instruction,
                "last_reason": f"Executing revised plan step {step}: {instruction}",
            },
            goto=planned_agent,
        )

    # 1) Build prompt, query reasoning_llm and validate the ExecutorDecision
    content_str = _message_text(reasoning_llm.invoke([executor_prompt(state)]))
    try:
        parsed = _parse_json_safely(content_str)
        decision = ExecutorDecision.model_validate({
            "replan": bool(parsed.get("replan", False)),
            "goto": str(parsed.get("goto", "synthesizer")),
            "reason": str(parsed.get("reason", "")),
            "query": str(parsed.get("query", "")),
        })
    except Exception as exc:
        raise ValueError(f"Invalid executor JSON response:\n{content_str}") from exc

    planned_agent = _plan_step(plan, step).get("agent")
    goto = decision.goto
    if goto not in allowed and goto != "planner":
        fallback = _routable_agent(planned_agent, allowed)
        logger.warning(f"Executor chose unknown node {goto!r}; routing to {fallback!r}.")
        goto = fallback

    updates: Dict[str, Any] = {
        "messages": [AIMessage(content=content_str, name="executor")],
        "last_reason": decision.reason,
        "agent_query": decision.query,
    }

    # Replan accounting per step
    replans: Dict[int, int] = dict(state.get("replan_attempts", {}) or {})
    step_replans = replans.get(step, 0)

    # 2) Replan decision
    if decision.replan or goto == "planner":
        if step_replans < MAX_REPLANS:
            replans[step] = step_replans + 1
            updates.update({
                "replan_attempts": replans,
                "replan_flag": True,   # Next turn executes the revised plan's step once
                "current_step": step,
            })
            return Command(update=updates, goto="planner")
        # Retry cap reached: skip step and delegate to next planned agent or synthesizer
        next_step_data = _plan_step(plan, step + 1)
        updates["current_step"] = step + 1
        if next_step_data:
            updates["agent_query"] = _step_instruction(next_step_data)
        return Command(update=updates, goto=_routable_agent(next_step_data.get("agent"), allowed))

    # 3) Happy path: run selected agent; advance step counter only if following plan
    updates["current_step"] = step + 1 if goto == planned_agent else step
    updates["replan_flag"] = False

    return Command(update=updates, goto=goto)


# =====================================================================
# 4. Specialist Worker Nodes
# =====================================================================

def cortex_agents_research_node(
    state: State,
) -> Command[Literal["executor"]]:
    """
    Sub-agent for internal enterprise retrieval across Snowflake Cortex (Analyst & Search).
    """
    query = state.get("agent_query") or state.get("user_query", "")
    agent_response = cortex_agent.invoke({"messages": [HumanMessage(content=query)]})

    last_content = _message_text(agent_response["messages"][-1])
    new_message = AIMessage(content=last_content, name="cortex_researcher")

    return Command(
        update={"messages": [new_message]},
        goto="executor",
    )


def web_research_node(
    state: State,
) -> Command[Literal["executor"]]:
    """
    Sub-agent for external public web intelligence using TavilySearch.
    """
    agent_query = state.get("agent_query") or state.get("user_query", "")
    result = web_search_agent.invoke({"messages": [HumanMessage(content=agent_query)]})

    # Ensure the terminal message is named web_researcher
    messages = list(result["messages"])
    messages[-1] = AIMessage(content=_message_text(messages[-1]), name="web_researcher")

    return Command(
        update={"messages": messages},
        goto="executor",
    )


def chart_node(state: State) -> Command[Literal["chart_summarizer"]]:
    """
    Sub-agent for generating plotting code and rendering charts to disk via the
    sandboxed Python tool. Receives a focused instruction plus the collected
    research data, and stores the validated CHART_PATH / CHART_NOTES in state.
    """
    instruction = state.get("agent_query") or state.get("user_query", "")
    findings = "\n\n---\n\n".join(_research_findings(state)) or "No research data collected."
    chart_request = HumanMessage(
        content=f"Chart task: {instruction}\n\nData collected by the researchers:\n{findings}"
    )
    result = chart_agent.invoke({"messages": [chart_request]})
    output = _message_text(result["messages"][-1])

    path_match = _CHART_PATH_RE.search(output)
    notes_match = _CHART_NOTES_RE.search(output)
    chart_path = safe_chart_path(path_match.group(1)) if path_match else None
    if path_match and chart_path is None:
        logger.error("Security: rejected unsafe CHART_PATH emitted by chart_generator.")

    return Command(
        update={
            "messages": [AIMessage(content=output, name="chart_generator")],
            "chart_path": chart_path,
            "chart_notes": notes_match.group(1) if notes_match else None,
        },
        goto="chart_summarizer",
    )


def chart_summary_node(
    state: State,
) -> Command[Literal["__end__"]]:
    """
    Generates a concise caption for the saved chart image. When researchers
    collected findings, the caption is combined with an executive synthesis so
    compound questions never lose analytical context. Concludes at END.
    """
    caption_request = HumanMessage(
        content=(
            f"PATH: {state.get('chart_path') or 'unknown'}\n"
            f"CHART_NOTES: {state.get('chart_notes') or 'none'}"
        )
    )
    result = chart_summary_agent.invoke({"messages": [caption_request]})
    caption = _message_text(result["messages"][-1]).strip()

    findings = _research_findings(state)
    if findings:
        synthesis = _synthesize_answer(state, findings + [f"Chart summary: {caption}"])
        final_answer = f"{synthesis}\n\n**Chart summary:** {caption}"
    else:
        final_answer = caption

    return Command(
        update={
            "messages": [AIMessage(content=caption, name="chart_summarizer")],
            "final_answer": final_answer,
        },
        goto=END,
    )


def synthesizer_node(state: State) -> Command[Literal["__end__"]]:
    """
    Executive synthesizer: consolidates findings from researchers and visualization
    into a concise prose report with figures, estimates, and citations, concluding at END.
    """
    relevant_msgs = [
        _message_text(m) for m in state.get("messages", [])
        if getattr(m, "name", None) in (*RESEARCH_AGENTS, "chart_generator", "chart_summarizer")
    ]
    answer = _synthesize_answer(state, relevant_msgs)

    return Command(
        update={
            "final_answer": answer,
            "messages": [AIMessage(content=answer, name="synthesizer")],
        },
        goto=END,
    )
