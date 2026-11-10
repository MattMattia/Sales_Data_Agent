"""Tests for state.py (schemas, configuration, shared constants) and prompts.py."""

import pytest
from pydantic import ValidationError

import graph
import prompts
import state
from prompts import executor_prompt, plan_prompt
from state import (
    DEFAULT_ENABLED_AGENTS,
    AIMessage,
    ExecutorDecision,
    HumanMessage,
    PlanStep,
    State,
    load_environment,
)


# ---------------------------------------------------------------------
# PlanStep / ExecutorDecision
# ---------------------------------------------------------------------

def test_plan_step_accepts_full_step_and_defaults_conditions():
    step = PlanStep.model_validate({"agent": "cortex_researcher", "action": "Fetch deals"})

    assert step.agent == "cortex_researcher"
    assert step.goal is None
    assert step.pre_conditions == []
    assert step.post_conditions == []


def test_plan_step_rejects_missing_agent():
    with pytest.raises(ValidationError):
        PlanStep.model_validate({"action": "Fetch deals"})


def test_executor_decision_requires_all_fields():
    with pytest.raises(ValidationError):
        ExecutorDecision.model_validate({"replan": False, "goto": "synthesizer"})


def test_executor_decision_goto_description_lists_chart_summarizer():
    description = ExecutorDecision.model_fields["goto"].description

    assert "'chart_summarizer'" in description


# ---------------------------------------------------------------------
# State schema and shared constants
# ---------------------------------------------------------------------

def test_state_declares_chart_fields():
    annotations = State.__annotations__

    assert "chart_path" in annotations
    assert "chart_notes" in annotations


def test_default_enabled_agents_is_single_shared_constant():
    assert prompts.DEFAULT_ENABLED_AGENTS is DEFAULT_ENABLED_AGENTS
    assert graph.DEFAULT_AGENTS is DEFAULT_ENABLED_AGENTS


def test_message_classes_are_shared_from_state():
    assert prompts.HumanMessage is HumanMessage
    assert prompts.SystemMessage is state.SystemMessage
    assert AIMessage(content="x", name="agent").content == "x"


# ---------------------------------------------------------------------
# load_environment (ARQ-01)
# ---------------------------------------------------------------------

def test_load_environment_reads_dotenv_without_overriding(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("SALES_AGENT_TEST_NEW=from_file\nSALES_AGENT_TEST_SET=from_file\n")
    monkeypatch.delenv("SALES_AGENT_SKIP_DOTENV", raising=False)
    monkeypatch.delenv("SALES_AGENT_TEST_NEW", raising=False)
    monkeypatch.setenv("SALES_AGENT_TEST_SET", "from_shell")

    loaded = load_environment(dotenv_path=str(env_file))

    import os
    assert loaded is True
    assert os.environ["SALES_AGENT_TEST_NEW"] == "from_file"
    assert os.environ["SALES_AGENT_TEST_SET"] == "from_shell"
    monkeypatch.delenv("SALES_AGENT_TEST_NEW", raising=False)


def test_load_environment_respects_skip_flag(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("SALES_AGENT_TEST_SKIPPED=1\n")
    monkeypatch.setenv("SALES_AGENT_SKIP_DOTENV", "1")
    monkeypatch.delenv("SALES_AGENT_TEST_SKIPPED", raising=False)

    loaded = load_environment(dotenv_path=str(env_file))

    import os
    assert loaded is False
    assert "SALES_AGENT_TEST_SKIPPED" not in os.environ


# ---------------------------------------------------------------------
# plan_prompt / executor_prompt
# ---------------------------------------------------------------------

def test_plan_prompt_initial_mode():
    message = plan_prompt({"user_query": "Top deals?", "enabled_agents": ["cortex_researcher", "synthesizer"]})

    assert "User Query: Top deals?" in message.content
    assert '["cortex_researcher", "synthesizer"]' in message.content
    assert "Create an optimal initial plan" in message.content
    assert "REPLANNING NOTICE" not in message.content


def test_plan_prompt_replan_mode_includes_reason_and_previous_plan():
    current_state = {
        "user_query": "Top deals?",
        "replan_flag": True,
        "current_step": 2,
        "last_reason": "Cortex returned no rows",
        "plan": {"1": {"agent": "cortex_researcher", "action": "Fetch"}},
        "messages": [HumanMessage(content="Top deals?"), AIMessage(content="empty", name="cortex_researcher")],
    }

    content = plan_prompt(current_state).content

    assert "=== REPLANNING NOTICE ===" in content
    assert "Reason for replanning: Cortex returned no rows" in content
    assert '"agent": "cortex_researcher"' in content
    assert "[cortex_researcher]: empty" in content


def test_executor_prompt_reports_step_and_attempts():
    content = executor_prompt({
        "user_query": "Top deals?",
        "plan": {"1": {"agent": "cortex_researcher", "action": "Fetch"}},
        "current_step": 1,
        "replan_attempts": {1: 2},
        "messages": [HumanMessage(content="Top deals?")],
    }).content

    assert "You are the Executor" in content
    assert "- Current Step Number: 1" in content
    assert "- Replan Attempts for current step: 2" in content
    assert "'chart_summarizer'" in content
