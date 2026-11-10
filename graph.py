"""
graph.py - Multi-Agent StateGraph assembly and orchestration in LangGraph:
1. Constructs the StateGraph(State) workflow adhering to the Plan & Execute pattern.
2. Registers orchestrator and worker nodes (planner, executor, cortex, web, chart, synthesizer).
3. Configures dynamic transitions governed by Command(goto=...).
4. Exposes compilation (build_graph), execution (run_agent), and topology inspection functions.
"""

import os
import logging
from typing import Any, Dict, List, Optional

# Shared State and Message Schemas
from state import DEFAULT_ENABLED_AGENTS, HumanMessage, State

# Orchestrator and Worker Nodes
from nodes import (
    planner_node,
    executor_node,
    cortex_agents_research_node,
    web_research_node,
    chart_node,
    chart_summary_node,
    synthesizer_node,
)

logger = logging.getLogger("sales_data_agent.graph")

# =====================================================================
# Safe LangGraph Import with High-Fidelity Mock Fallback
# =====================================================================

try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    START = "__start__"
    END = "__end__"

    class MockStateGraph:
        """
        Functional emulator of StateGraph for verification in environments
        where langgraph is not yet installed in the global Python path.
        """
        def __init__(self, state_schema: Any):
            self.state_schema = state_schema
            self.nodes: Dict[str, Any] = {}
            self.edges: List[tuple] = []
            self.entry_point: Optional[str] = None

        def add_node(self, name: str, func: Any):
            self.nodes[name] = func

        def add_edge(self, start_node: str, end_node: str):
            if start_node == START or start_node == "__start__":
                self.entry_point = end_node
            self.edges.append((start_node, end_node))

        def compile(self) -> "MockCompiledGraph":
            return MockCompiledGraph(self.nodes, self.entry_point or "planner")


    class MockCompiledGraph:
        """
        Execution emulator for compiled graphs governed by Command(goto=...).
        """
        def __init__(self, nodes: Dict[str, Any], entry_point: str):
            self.nodes = nodes
            self.entry_point = entry_point

        def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
            curr_state = dict(state)
            curr_node = self.entry_point
            max_steps = 30
            step_count = 0

            logger.info(f"[Graph] Starting execution flow at: {curr_node}")

            while curr_node and curr_node not in (END, "__end__") and step_count < max_steps:
                step_count += 1
                node_func = self.nodes.get(curr_node)
                if not node_func:
                    logger.warning(f"[Graph] Node not found: {curr_node}. Halting execution.")
                    break

                logger.debug(f"[Graph] Executing step {step_count}: {curr_node}")
                cmd = node_func(curr_state)

                # Apply state mutations
                if hasattr(cmd, "update") and isinstance(cmd.update, dict):
                    for k, v in cmd.update.items():
                        if k == "messages":
                            curr_msgs = list(curr_state.get("messages", []))
                            if isinstance(v, list):
                                curr_msgs.extend(v)
                            else:
                                curr_msgs.append(v)
                            curr_state["messages"] = curr_msgs
                        else:
                            curr_state[k] = v

                # Route to the target node
                next_node = getattr(cmd, "goto", END)
                logger.debug(f"[Graph] Transition: {curr_node} -> {next_node}")
                curr_node = next_node

            logger.info(f"[Graph] Execution completed in {step_count} steps.")
            return curr_state

        def get_graph(self):
            class DummyGraphRepresentation:
                def draw_ascii(self):
                    return display_ascii_graph()
                def draw_mermaid(self):
                    return get_graph_mermaid()
                def draw_png(self):
                    raise NotImplementedError("PNG drawing requires official langgraph and pygraphviz/graphviz")
            return DummyGraphRepresentation()


# =====================================================================
# 1. StateGraph Construction Function
# =====================================================================

def build_graph() -> Any:
    """
    Constructs and compiles the Multi-Agent workflow in LangGraph.

    Graph Topology:
    - Entry: START -> planner
    - Orchestration: planner <-> executor
    - Specialists:
        * cortex_researcher (Cortex Analyst / Search) -> executor
        * web_researcher (Tavily Search)              -> executor
        * chart_generator (Python REPL Tool)          -> chart_summarizer
        * chart_summarizer                            -> END
        * synthesizer                                 -> END
    """
    GraphClass = StateGraph if LANGGRAPH_AVAILABLE else MockStateGraph
    workflow = GraphClass(State)

    # 1. Register Orchestration Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)

    # 2. Register Specialist Worker Nodes
    workflow.add_node("cortex_researcher", cortex_agents_research_node)
    workflow.add_node("web_researcher", web_research_node)
    workflow.add_node("chart_generator", chart_node)
    workflow.add_node("chart_summarizer", chart_summary_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # 3. Initial Entry Edge
    workflow.add_edge(START, "planner")

    # Remaining transitions are dynamically routed by Command(goto=..., update=...) inside each node.
    compiled_graph = workflow.compile()
    return compiled_graph


# Globally accessible compiled graph instance
graph = build_graph()


# =====================================================================
# 2. Execution and Interface Functions
# =====================================================================

# Backward-compatible alias of the shared agent registry in state.py
DEFAULT_AGENTS = DEFAULT_ENABLED_AGENTS

def run_agent(
    query: str,
    enabled_agents: Optional[List[str]] = None,
    compiled_graph: Optional[Any] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes the multi-agent graph to answer the user query.

    Parameters:
    -----------
    query: str
        The question or objective requested by the user.
    enabled_agents: list, optional
        List of allowed agent names for this run.
    compiled_graph: executable graph, optional
        Compiled graph instance (defaults to global compiled graph).
    verbose: bool
        If True, prints execution banners and final response to console.

    Returns:
    --------
    dict:
        The full final graph state including 'final_answer'.
    """
    active_graph = compiled_graph or graph
    active_agents = enabled_agents or DEFAULT_AGENTS

    if verbose:
        print("\n" + "=" * 60)
        print(">>> STARTING DATA AGENT WORKFLOW")
        print(f"Query: {query}")
        print(f"Enabled Agents: {active_agents}")
        print("=" * 60)

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "enabled_agents": active_agents,
        "current_step": 1,
        "replan_flag": False,
        "replan_attempts": {},
        "chart_path": None,
        "chart_notes": None,
    }

    final_state = active_graph.invoke(initial_state)

    if verbose:
        final_answer = final_state.get("final_answer", "No final answer recorded.")
        total_messages = len(final_state.get("messages", []))
        print("\n" + "-" * 60)
        print(f"[OK] FINAL ANSWER PRODUCED ({total_messages} messages generated):")
        print("-" * 60)
        print(final_answer)
        print("=" * 60 + "\n")

    return final_state


# =====================================================================
# 3. Visualization and Topology Helpers
# =====================================================================

def get_graph_mermaid() -> str:
    """
    Returns the Mermaid syntax diagram of the multi-agent graph.
    """
    return """
flowchart TD
    START([START]) --> planner[planner_node]
    planner -->|goto='executor'| executor[executor_node]

    executor -.->|replan=True| planner
    executor -.->|goto='cortex_researcher'| cortex[cortex_researcher]
    executor -.->|goto='web_researcher'| web[web_researcher]
    executor -.->|goto='chart_generator'| chart[chart_generator]
    executor -.->|goto='synthesizer'| synth[synthesizer]

    cortex -->|goto='executor'| executor
    web -->|goto='executor'| executor
    chart -->|goto='chart_summarizer'| summarizer[chart_summarizer]

    summarizer -->|goto=END| END([END])
    synth -->|goto=END| END([END])
"""

def display_ascii_graph() -> str:
    """
    Returns the formatted ASCII diagram of the graph architecture.
    """
    return """
+-------------------------------------------------------------------------+
|                    SALES DATA AGENT: STATE GRAPH                        |
+-------------------------------------------------------------------------+
                                 [START]
                                    |
                                    v
                            +---------------+
                            | planner_node  |<------------------+
                            +---------------+                   |
                                    | (Command)                 |
                                    v                           | (replan)
                            +---------------+                   |
                            | executor_node |-------------------+
                            +---------------+
                              |    |    |   \
           +------------------+    |    |    +------------------+
           |                       |    |                       |
           v                       v    |                       v
  +------------------+   +----------------+   |          +-------------+
  |cortex_researcher |   | web_researcher |   |          | synthesizer |
  +------------------+   +----------------+   |          +-------------+
           |                       |          |                 |
           +-----------> <---------+          v                 v
                         |             +---------------+      [END]
                         +------------>|chart_generator|
                                       +---------------+
                                              |
                                              v
                                     +------------------+
                                     | chart_summarizer |
                                     +------------------+
                                              |
                                              v
                                            [END]
+-------------------------------------------------------------------------+
"""


# =====================================================================
# 4. Main Execution Block (__main__)
# =====================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print(display_ascii_graph())

    # Test Query 1: Numerical analysis and chart plotting flow
    test_query_1 = "What are our top 3 customer deals? Chart the deal value for each."
    run_agent(test_query_1)

    # Test Query 2: Analytical multi-source synthesis flow
    test_query_2 = "What are our top 3 customer deals and what are the main themes in meeting notes?"
    run_agent(test_query_2)
