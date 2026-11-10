"""
prompts.py - Prompt engineering and system directives for the Multi-Agent
Plan & Execute architecture with TruLens observability.

Includes:
1. plan_prompt: Planning prompt generator with sub-goals (L6 GPA Alignment),
   pre/post conditions, dynamic replanning support (replan_flag), and error context (last_reason).
2. executor_prompt: Progress evaluator and sub-agent dispatcher with replanning decisions.
3. System prompts for ReAct sub-agents (Cortex Researcher, Web Researcher, Chart Generator, Chart Summarizer).
4. Executive synthesis prompt for the synthesizer node.
"""

import json
from typing import Any, Dict, Union

from state import DEFAULT_ENABLED_AGENTS, HumanMessage, SystemMessage


# =====================================================================
# System Prompts for Specialized Sub-Agents
# =====================================================================

CORTEX_RESEARCHER_SYSTEM_PROMPT = """You are the Researcher. You can answer questions using customer deal data along with meeting notes.
Do not take any further action."""

WEB_RESEARCHER_SYSTEM_PROMPT = """You are the Researcher. You can ONLY perform research by using the provided search tool (tavily_tool). 
When you have found the necessary information, end your output.  
Do NOT attempt to take further actions."""

CHART_GENERATOR_SYSTEM_PROMPT = """You can only generate charts. You are working with a researcher colleague.
1) Print the chart first.
2) Save the chart as a .png file using a relative path inside the current working directory (for example charts/top_deals.png). Never use absolute paths or '..'.
3) At the very end of your message, output EXACTLY two lines so the summarizer can find them:
   CHART_PATH: <relative_path_to_chart_file>
   CHART_NOTES: <one concise sentence summarizing the main insight in the chart>
Do not include any other trailing text after these two lines."""

CHART_SUMMARIZER_SYSTEM_PROMPT = """You can only generate image captions. You are working with a researcher colleague and a chart generator colleague.
Your task is to generate a standalone, concise summary for the provided chart image saved at a local PATH, where the PATH should be and only be provided by your chart generator colleague.
The summary should be no more than 3 sentences and should not mention the chart itself."""

SYNTHESIZER_INSTRUCTIONS = """You are the Synthesizer. Use the context below to directly answer the user's question. Perform any lightweight calculations, comparisons, or inferences required. Do not invent facts not supported by the context. If data is missing, say what's missing and, if helpful, offer a clearly labeled best-effort estimate with assumptions.

Produce a concise response that fully answers the question, with the following guidance:
- Start with the direct answer (one short paragraph or a tight bullet list).
- Include key figures from any 'Results:' tables (e.g., totals, top items).
- If any message contains citations, include them as a brief 'Citations: [...]' line.
- Keep the output crisp; avoid meta commentary or tool instructions."""


def agent_system_prompt(instructions: str) -> SystemMessage:
    """
    Constructs a standardized SystemMessage for ReAct agents.
    """
    clean_instructions = " ".join(line.strip() for line in instructions.strip().splitlines() if line.strip())
    return SystemMessage(content=clean_instructions)


# =====================================================================
# Planner Prompt (With L6 Enhancements: GPA Alignment, Goals, Pre/Post Conditions)
# =====================================================================

PLANNER_INSTRUCTIONS = """You are an expert multi-agent task planner.
Your goal is to break down the user's query into a sequential, logical, and executable multi-step plan.

Available Sub-Agents and Responsibilities:
- "cortex_researcher": Accesses internal enterprise CRM structured deal metrics (via Text-to-SQL) and unstructured customer meeting notes/transcripts (via Semantic Search). Use whenever the query refers to customer deals, clients, revenue, internal sales metrics, or meeting notes.
- "web_researcher": Searches external public web data, market news, industry trends, competitor movements, stock prices, or public regulatory changes.
- "chart_generator": Generates and executes Python code to produce and save visualizations (charts/plots). MUST be scheduled when the user explicitly asks to "chart", "plot", or "visualize" metrics.
- "chart_summarizer": Generates a concise summary for the chart image saved on disk. MUST be scheduled immediately following "chart_generator".
- "synthesizer": Synthesizes the retrieved information into a clear, executive prose answer. MUST be the terminal agent if no chart was produced.

Output Format:
You MUST respond with a valid, raw JSON object (and nothing else).
The keys must be string step numbers ("1", "2", "3", ...).
Each step MUST adhere strictly to the following schema:
{
  "1": {
    "agent": "<agent_name>",
    "action": "<concise description of the specific action to perform>",
    "goal": "<specific verifiable sub-goal to achieve in this step>",
    "pre_conditions": ["<prerequisite data or condition 1>", ...],
    "post_conditions": ["<expected outcome or artifact 1>", ...]
  }
}
"""

def plan_prompt(state: Union[Dict[str, Any], Any]) -> HumanMessage:
    """
    Generates the prompt for planner_node.

    Incorporates:
    - Modular list of available agents (enabled_agents).
    - Sub-goals, preconditions, and postconditions per step (L6 GPA alignment).
    - Full support for adaptive replanning if 'replan_flag' is True, injecting
      the previous plan, failure rationale ('last_reason'), and recent message history.
    """
    user_query = state.get("user_query")
    if not user_query and state.get("messages"):
        user_query = state["messages"][0].content

    enabled_agents = state.get("enabled_agents") or DEFAULT_ENABLED_AGENTS
    replan_flag = state.get("replan_flag", False)
    current_plan = state.get("plan", {})
    last_reason = state.get("last_reason", "")
    current_step = state.get("current_step", 1)

    prompt_lines = [
        PLANNER_INSTRUCTIONS,
        f"\nUser Query: {user_query}",
        f"Allowed Agents for this session: {json.dumps(enabled_agents)}",
    ]

    if replan_flag:
        prompt_lines.append("\n=== REPLANNING NOTICE ===")
        prompt_lines.append("The previous execution requires revision. You must construct a new, corrected plan.")
        prompt_lines.append(f"Current step index when replanning triggered: {current_step}")
        if last_reason:
            prompt_lines.append(f"Reason for replanning: {last_reason}")
        if current_plan:
            prompt_lines.append(f"Previous plan that failed or was incomplete:\n{json.dumps(current_plan, indent=2)}")

        # Inject recent agent findings and message context if available
        recent_messages = state.get("messages", [])
        if len(recent_messages) > 1:
            history_snippets = []
            for msg in recent_messages[-4:]:
                name = getattr(msg, "name", "agent")
                snippet = msg.content[:300] + ("..." if len(msg.content) > 300 else "")
                history_snippets.append(f"- [{name}]: {snippet}")
            prompt_lines.append("Recent Agent Findings / Errors:\n" + "\n".join(history_snippets))

        prompt_lines.append("\nProduce an updated plan starting from the current situation to successfully fulfill the user query.")
    else:
        prompt_lines.append("\nCreate an optimal initial plan to fulfill the user's query end-to-end.")

    return HumanMessage(content="\n".join(prompt_lines))


# =====================================================================
# Executor Prompt (Dynamic Dispatch, Progress Evaluation & Replan)
# =====================================================================

EXECUTOR_INSTRUCTIONS = """You are the Executor. You manage the execution of the multi-agent plan and decide the next immediate action.

Your Responsibilities:
1. Examine the user's overall query, the active plan, the current step number, and the messages history from previous agents.
2. Evaluate if the last action succeeded or if the plan needs modification:
   - If the previous agent failed, returned empty/invalid data, or discovered that prerequisite information is missing, set "replan": true and "goto": "planner". Explain why in "reason".
   - If the previous step succeeded and there are more steps in the plan, select the agent specified for the current step, set "replan": false, and formulate a crystal-clear, specific instruction in "query".
   - If all necessary research/action steps have completed, route to "synthesizer" (or "chart_summarizer" if a chart was just generated) to conclude.

Output Schema:
You MUST output ONLY a valid JSON object matching this schema:
{
  "replan": <boolean, true if the planner must formulate a revised plan, false otherwise>,
  "goto": "<name of the next node: 'cortex_researcher', 'web_researcher', 'chart_generator', 'chart_summarizer', 'synthesizer', or 'planner'>",
  "reason": "<clear explanation of why this step was chosen or why replanning is required>",
  "query": "<specific, actionable instruction for the target agent at this step>"
}
"""

def executor_prompt(state: Union[Dict[str, Any], Any]) -> HumanMessage:
    """
    Generates the prompt for executor_node.

    Evaluates current state, active plan, current step, and accumulated
    messages to produce an ExecutorDecision in strict JSON format.
    """
    user_query = state.get("user_query", "")
    plan = state.get("plan", {})
    current_step = state.get("current_step", 1)
    enabled_agents = state.get("enabled_agents") or DEFAULT_ENABLED_AGENTS
    replan_attempts = state.get("replan_attempts", {})

    # Format relevant message history
    messages = state.get("messages", [])
    history_lines = []
    for msg in messages:
        sender = getattr(msg, "name", getattr(msg, "type", "user"))
        content_preview = msg.content if isinstance(msg.content, str) else str(msg.content)
        # Truncate overly long content to maintain token efficiency while preserving key context
        if len(content_preview) > 600:
            content_preview = content_preview[:600] + "... [truncated]"
        history_lines.append(f"[{sender}]: {content_preview}")

    history_str = "\n".join(history_lines) if history_lines else "No messages yet."

    prompt_content = f"""{EXECUTOR_INSTRUCTIONS}

Current Execution State:
- Original User Query: {user_query}
- Active Plan:
{json.dumps(plan, indent=2) if plan else "No plan registered."}
- Current Step Number: {current_step}
- Enabled Agents: {json.dumps(enabled_agents)}
- Replan Attempts for current step: {replan_attempts.get(current_step, 0)}

Message History & Evidence Collected:
------------------------------------
{history_str}
------------------------------------

Analyze the latest evidence against step {current_step}'s goal and post-conditions.
Determine the next action and return the JSON response.
"""
    return HumanMessage(content=prompt_content)
