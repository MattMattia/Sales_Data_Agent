"""Tests for nodes.py orchestration, graph.py execution flows, and evals.py."""

import json
import typing

import pytest

import evals
import nodes
from graph import run_agent
from nodes import (
    MAX_REPLANS,
    _parse_json_safely,
    chart_node,
    chart_summary_node,
    executor_node,
    planner_node,
    safe_chart_path,
    synthesizer_node,
)
from state import AIMessage, HumanMessage


class FakeLLM:
    """Returns a fixed content string and records every prompt it receives."""

    def __init__(self, content):
        self.content = content
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages)
        return type("Reply", (), {"content": self.content})()


ALL_AGENTS = ["cortex_researcher", "web_researcher", "chart_generator", "chart_summarizer", "synthesizer"]


# ---------------------------------------------------------------------
# _parse_json_safely (CODE-02)
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Here is the plan:\n```json\n{"a": 1}\n```\nThanks',
    ],
)
def test_parse_json_safely_strips_markdown_fences(raw):
    assert _parse_json_safely(raw) == {"a": 1}


def test_parse_json_safely_rejects_non_object():
    with pytest.raises(ValueError):
        _parse_json_safely("not json at all")


# ---------------------------------------------------------------------
# Type hints (CODE-01)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("node", [chart_summary_node, synthesizer_node])
def test_terminal_nodes_use_literal_end_string(node):
    hint = typing.get_type_hints(node)["return"]

    assert typing.get_args(hint) == (typing.Literal["__end__"],)


# ---------------------------------------------------------------------
# planner_node (CODE-02, ARQ-03)
# ---------------------------------------------------------------------

def test_planner_validates_steps_and_drops_unknown_agents(monkeypatch):
    raw_plan = {
        "1": {"agent": "cortex_researcher", "action": "Fetch deals"},
        "2": {"agent": "hacker_agent", "action": "Do evil"},
        "3": {"action": "Missing agent"},
        "4": {"agent": "synthesizer", "action": "Answer"},
    }
    monkeypatch.setattr(nodes, "reasoning_llm", FakeLLM("```json\n" + json.dumps(raw_plan) + "\n```"))

    command = planner_node({"user_query": "Top deals?", "enabled_agents": ALL_AGENTS, "messages": []})

    plan = command.update["plan"]
    assert list(plan) == ["1", "2"]
    assert plan["1"]["agent"] == "cortex_researcher"
    assert plan["2"]["agent"] == "synthesizer"
    assert plan["1"]["pre_conditions"] == []
    assert command.goto == "executor"


def test_planner_rejects_agents_disabled_for_session(monkeypatch):
    raw_plan = {"1": {"agent": "web_researcher", "action": "Search"}, "2": {"agent": "synthesizer", "action": "A"}}
    monkeypatch.setattr(nodes, "reasoning_llm", FakeLLM(json.dumps(raw_plan)))

    command = planner_node({"user_query": "q", "enabled_agents": ["cortex_researcher", "synthesizer"], "messages": []})

    assert [step["agent"] for step in command.update["plan"].values()] == ["synthesizer"]


def test_planner_falls_back_to_synthesizer_when_no_valid_step(monkeypatch):
    monkeypatch.setattr(nodes, "reasoning_llm", FakeLLM(json.dumps({"1": {"agent": "nobody", "action": "x"}})))

    command = planner_node({"user_query": "q", "enabled_agents": ALL_AGENTS, "messages": []})

    assert command.update["plan"]["1"]["agent"] == "synthesizer"


def test_planner_raises_on_invalid_json(monkeypatch):
    monkeypatch.setattr(nodes, "reasoning_llm", FakeLLM("I cannot plan this"))

    with pytest.raises(ValueError, match="Planner returned invalid JSON"):
        planner_node({"user_query": "q", "enabled_agents": ALL_AGENTS, "messages": []})


def test_planner_resets_step_to_one_after_replan(monkeypatch):
    new_plan = {"1": {"agent": "web_researcher", "action": "Search news"}, "2": {"agent": "synthesizer", "action": "A"}}
    monkeypatch.setattr(nodes, "reasoning_llm", FakeLLM(json.dumps(new_plan)))

    command = planner_node({
        "user_query": "q",
        "enabled_agents": ALL_AGENTS,
        "replan_flag": True,
        "current_step": 3,
        "messages": [],
    })

    assert command.update["current_step"] == 1
    assert command.update["replan_flag"] is True
    assert command.update["messages"][0].name == "replan"


# ---------------------------------------------------------------------
# executor_node (ARQ-03, CODE-02, MAX_REPLANS)
# ---------------------------------------------------------------------

REPLANNED_STATE = {
    "user_query": "q",
    "enabled_agents": ALL_AGENTS,
    "replan_flag": True,
    "current_step": 1,
    "agent_query": "stale query from failed attempt",
    "plan": {
        "1": {"agent": "web_researcher", "action": "Search regulatory news", "goal": "Find 2026 rules"},
        "2": {"agent": "synthesizer", "action": "Answer"},
    },
}


def test_executor_after_replan_dispatches_with_fresh_agent_query(monkeypatch):
    monkeypatch.setattr(nodes, "reasoning_llm", FakeLLM("must not be called"))

    command = executor_node(dict(REPLANNED_STATE))

    assert command.goto == "web_researcher"
    assert command.update["agent_query"] == "Search regulatory news (goal: Find 2026 rules)"
    assert command.update["current_step"] == 2
    assert command.update["replan_flag"] is False
    assert nodes.reasoning_llm.prompts == []


def test_executor_after_replan_falls_back_when_step_agent_invalid():
    state = dict(REPLANNED_STATE, plan={"1": {"agent": "rogue", "action": "x"}})

    command = executor_node(state)

    assert command.goto == "synthesizer"


def test_executor_happy_path_advances_step(monkeypatch):
    decision = {"replan": False, "goto": "web_researcher", "reason": "next", "query": "Search news"}
    monkeypatch.setattr(nodes, "reasoning_llm", FakeLLM("```json\n" + json.dumps(decision) + "\n```"))

    command = executor_node(dict(REPLANNED_STATE, replan_flag=False))

    assert command.goto == "web_researcher"
    assert command.update["agent_query"] == "Search news"
    assert command.update["current_step"] == 2


def test_executor_replaces_unknown_goto_with_planned_agent(monkeypatch):
    decision = {"replan": False, "goto": "delete_everything", "reason": "?", "query": "q"}
    monkeypatch.setattr(nodes, "reasoning_llm", FakeLLM(json.dumps(decision)))

    command = executor_node(dict(REPLANNED_STATE, replan_flag=False))

    assert command.goto == "web_researcher"


def test_executor_routes_to_planner_and_counts_replans(monkeypatch):
    decision = {"replan": True, "goto": "planner", "reason": "no data", "query": ""}
    monkeypatch.setattr(nodes, "reasoning_llm", FakeLLM(json.dumps(decision)))

    command = executor_node(dict(REPLANNED_STATE, replan_flag=False, replan_attempts={1: 1}))

    assert command.goto == "planner"
    assert command.update["replan_attempts"] == {1: 2}
    assert command.update["replan_flag"] is True
    assert command.update["last_reason"] == "no data"


def test_executor_stops_replanning_after_max_replans(monkeypatch):
    decision = {"replan": True, "goto": "planner", "reason": "still no data", "query": ""}
    monkeypatch.setattr(nodes, "reasoning_llm", FakeLLM(json.dumps(decision)))

    command = executor_node(dict(REPLANNED_STATE, replan_flag=False, replan_attempts={1: MAX_REPLANS}))

    assert command.goto == "synthesizer"
    assert command.update["current_step"] == 2


def test_executor_raises_on_invalid_json(monkeypatch):
    monkeypatch.setattr(nodes, "reasoning_llm", FakeLLM("no json here"))

    with pytest.raises(ValueError, match="Invalid executor JSON response"):
        executor_node(dict(REPLANNED_STATE, replan_flag=False))


# ---------------------------------------------------------------------
# Research nodes receive message lists (CODE-03)
# ---------------------------------------------------------------------

class RecordingAgent:
    def __init__(self, reply):
        self.reply = reply
        self.inputs = []

    def invoke(self, payload):
        self.inputs.append(payload)
        return {"messages": [AIMessage(content=self.reply)]}


@pytest.mark.parametrize(
    "node_name, agent_attr, author",
    [
        ("cortex_agents_research_node", "cortex_agent", "cortex_researcher"),
        ("web_research_node", "web_search_agent", "web_researcher"),
    ],
)
def test_research_nodes_send_human_message_list(monkeypatch, node_name, agent_attr, author):
    agent = RecordingAgent("findings")
    monkeypatch.setattr(nodes, agent_attr, agent)

    command = getattr(nodes, node_name)({"agent_query": "Find deals", "user_query": "q"})

    sent = agent.inputs[0]["messages"]
    assert isinstance(sent, list)
    assert isinstance(sent[0], HumanMessage)
    assert sent[0].content == "Find deals"
    assert command.update["messages"][-1].name == author
    assert command.update["messages"][-1].content == "findings"
    assert command.goto == "executor"


# ---------------------------------------------------------------------
# Chart flow (ARQ-05, SEC-03)
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw_path, accepted",
    [
        ("top_deals_chart.png", True),
        ("charts/deals.svg", True),
        ("../../.env", False),
        ("charts/../../secret.png", False),
        ("/etc/passwd", False),
        ("chart.exe", False),
        ("", False),
    ],
)
def test_safe_chart_path(isolated_cwd, raw_path, accepted):
    result = safe_chart_path(raw_path)

    assert (result is not None) is accepted


def test_chart_node_extracts_path_and_notes(isolated_cwd, monkeypatch):
    reply = "Done.\nCHART_PATH: charts/deals.png\nCHART_NOTES: HealthTech leads."
    agent = RecordingAgent(reply)
    monkeypatch.setattr(nodes, "chart_agent", agent)
    state = {
        "agent_query": "Plot deal values",
        "messages": [AIMessage(content="HealthTech $1.25M", name="cortex_researcher")],
    }

    command = chart_node(state)

    sent = agent.inputs[0]["messages"][0].content
    assert "Plot deal values" in sent
    assert "HealthTech $1.25M" in sent
    assert command.update["chart_path"] == "charts/deals.png"
    assert command.update["chart_notes"] == "HealthTech leads."
    assert command.goto == "chart_summarizer"


def test_chart_node_drops_unsafe_path(isolated_cwd, monkeypatch):
    monkeypatch.setattr(nodes, "chart_agent", RecordingAgent("CHART_PATH: ../../.env\nCHART_NOTES: n"))

    command = chart_node({"agent_query": "Plot", "messages": []})

    assert command.update["chart_path"] is None


def test_chart_summary_without_research_returns_caption_only(monkeypatch):
    monkeypatch.setattr(nodes, "chart_summary_agent", RecordingAgent("Caption text."))

    command = chart_summary_node({"chart_path": "c.png", "chart_notes": "n", "messages": []})

    assert command.update["final_answer"] == "Caption text."
    assert command.goto == "__end__"


def test_chart_summary_keeps_research_findings(monkeypatch):
    monkeypatch.setattr(nodes, "chart_summary_agent", RecordingAgent("Caption text."))
    monkeypatch.setattr(nodes, "llm", FakeLLM("Synthesized deals answer."))
    state = {
        "user_query": "Top deals and themes? Chart it.",
        "chart_path": "c.png",
        "messages": [AIMessage(content="Themes: compliance", name="cortex_researcher")],
    }

    command = chart_summary_node(state)

    answer = command.update["final_answer"]
    assert "Synthesized deals answer." in answer
    assert "Caption text." in answer
    assert "Themes: compliance" in nodes.llm.prompts[0][0].content


# ---------------------------------------------------------------------
# End-to-end flows through graph.run_agent (mock mode)
# ---------------------------------------------------------------------

def test_run_agent_synthesis_flow(isolated_cwd):
    final_state = run_agent("What are our top 3 customer deals and what are the main themes in meeting notes?", verbose=False)

    assert "HealthTech Corp" in final_state["final_answer"]
    assert final_state["chart_path"] is None
    names = [getattr(m, "name", None) for m in final_state["messages"]]
    assert "cortex_researcher" in names
    assert "web_researcher" in names
    assert names[-1] == "synthesizer"


def test_run_agent_chart_flow(isolated_cwd):
    final_state = run_agent("What are our top 3 customer deals? Chart the deal value for each.", verbose=False)

    assert final_state["chart_path"] == "top_deals_chart.png"
    assert (isolated_cwd / "top_deals_chart.png").exists()
    assert final_state["chart_notes"].startswith("HealthTech Corp holds the largest deal")
    assert "HealthTech Corp" in final_state["final_answer"]
    assert "Chart summary" in final_state["final_answer"]


# ---------------------------------------------------------------------
# evals.py (CODE-04)
# ---------------------------------------------------------------------

EXPECTED_METRICS = {
    "Answer Relevance",
    "Context Relevance",
    "Groundedness",
    "Plan Quality",
    "Plan Adherence",
    "Execution Efficiency",
    "Logical Consistency",
}


def test_all_feedbacks_cover_rag_triad_and_gpa():
    assert {fb.name for fb in evals.ALL_FEEDBACKS} == EXPECTED_METRICS


def test_setup_trulens_recorder_returns_context_manager():
    recorder = evals.setup_trulens_recorder(graph=object(), reset_db=True)

    with recorder as recording:
        assert recording is recorder
    assert len(recorder.feedbacks) == 7


def test_evaluate_query_results_in_mock_mode():
    results = evals.evaluate_query_results({"final_answer": "HealthTech leads."})

    assert {row["metric"] for row in results} == EXPECTED_METRICS
    assert all(0.0 <= row["score"] <= 1.0 for row in results)
    assert all(row["reason"] for row in results)


def test_evaluate_query_results_never_raises_with_real_feedback_objects(monkeypatch):
    class TwoArgFeedback:
        name = "Groundedness"

        def __call__(self, source, statement):
            return 1.0, {}

    monkeypatch.setattr(evals, "TRULENS_INSTALLED", True)
    monkeypatch.setattr(evals, "ALL_FEEDBACKS", [TwoArgFeedback()])
    monkeypatch.setattr(evals, "_fetch_recorded_feedback", lambda recording: {})

    results = evals.evaluate_query_results({"final_answer": "x"}, recording=None)

    assert results == [{"metric": "Groundedness", "score": None, "reason": "Pending: feedback not yet computed by TruLens."}]


# ---------------------------------------------------------------------
# Reliability follow-ups: replan budget, disabled chart_summarizer
# ---------------------------------------------------------------------

def test_executor_stops_replanning_when_total_budget_is_spent(monkeypatch):
    decision = {"replan": True, "goto": "planner", "reason": "still failing", "query": ""}
    monkeypatch.setattr(nodes, "reasoning_llm", FakeLLM(json.dumps(decision)))
    spent = {1: 2, 2: 2, 3: 2}  # renumbered plans spread replans over many indices
    assert sum(spent.values()) >= nodes.MAX_TOTAL_REPLANS

    command = executor_node(dict(REPLANNED_STATE, replan_flag=False, replan_attempts=spent))

    assert command.goto != "planner"
    assert command.update["current_step"] == 2


def test_executor_still_replans_within_total_budget(monkeypatch):
    decision = {"replan": True, "goto": "planner", "reason": "no data", "query": ""}
    monkeypatch.setattr(nodes, "reasoning_llm", FakeLLM(json.dumps(decision)))

    command = executor_node(dict(REPLANNED_STATE, replan_flag=False, replan_attempts={2: 1}))

    assert command.goto == "planner"


def test_chart_node_routes_to_synthesizer_when_summarizer_disabled(isolated_cwd, monkeypatch):
    monkeypatch.setattr(nodes, "chart_agent", RecordingAgent("CHART_PATH: deals.png\nCHART_NOTES: n"))
    enabled = ["cortex_researcher", "chart_generator"]

    command = chart_node({"agent_query": "Plot", "messages": [], "enabled_agents": enabled})

    assert command.goto == "synthesizer"


def test_run_agent_never_runs_disabled_chart_summarizer(isolated_cwd):
    final_state = run_agent(
        "What are our top 3 customer deals? Chart the deal value for each.",
        enabled_agents=["cortex_researcher", "chart_generator"],
        verbose=False,
    )

    names = [getattr(m, "name", None) for m in final_state["messages"]]
    assert "chart_summarizer" not in names
    assert final_state["final_answer"]
