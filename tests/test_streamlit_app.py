"""
Headless UI tests for app.py using Streamlit's first-party AppTest.

Every test runs the real script in-process in Mock Fallback mode (see
conftest.py), from a temporary working directory so generated charts never
land in the repository.
"""

import sys
from pathlib import Path

import pytest
from streamlit import deprecation_util
from streamlit.testing.v1 import AppTest

from state import DEFAULT_ENABLED_AGENTS

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
APP_TIMEOUT_SECONDS = 60
CHART_QUERY = "What are our top 3 customer deals? Chart the deal value for each."
SCORECARD_METRIC_COUNT = 7


class _DeprecationSpy:
    """Stands in for Streamlit's deprecation logger and records every message."""

    def __init__(self) -> None:
        self.messages = []

    def warning(self, message, *args, **kwargs) -> None:
        self.messages.append(str(message))


@pytest.fixture
def deprecations(monkeypatch):
    spy = _DeprecationSpy()
    monkeypatch.setattr(deprecation_util, "_LOGGER", spy)
    return spy


@pytest.fixture
def app(isolated_cwd):
    return AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT_SECONDS)


def _button_by_label(at: AppTest, label: str):
    return next(button for button in at.button if button.label == label)


def _assistant_messages(at: AppTest):
    return [msg for msg in at.session_state["messages"] if msg["role"] == "assistant"]


def test_initial_load_shows_welcome_agents_and_trulens_toggle(app):
    at = app.run()

    assert not at.exception
    assert "Welcome! I am the **Sales Data Agent**" in at.chat_message[0].markdown[0].value
    assert at.multiselect[0].value == list(DEFAULT_ENABLED_AGENTS)
    assert len(at.multiselect[0].value) == 5
    assert at.checkbox[0].label == "Run TruLens Evaluation Scorecard"
    assert at.checkbox[0].value is True


def test_app_uses_width_stretch_without_use_container_width_deprecation(app, deprecations):
    at = app.run()
    at.button(key="sample_0").click().run()

    assert not at.exception
    assert [msg for msg in deprecations.messages if "use_container_width" in msg] == []


def test_chat_query_renders_answer_plan_and_full_scorecard(app):
    at = app.run()
    at.chat_input[0].set_value("What are the main themes in our meeting notes?").run()

    assert not at.exception
    reply = _assistant_messages(at)[-1]
    assert reply["content"].strip()
    assert reply["plan"]
    assert len(reply["scores"]) == SCORECARD_METRIC_COUNT
    assert len(at.metric) == SCORECARD_METRIC_COUNT
    assert any("Execution Plan" in expander.label for expander in at.expander)


def test_chart_query_stores_immutable_chart_bytes_in_history(app):
    at = app.run()
    at.chat_input[0].set_value(CHART_QUERY).run()

    assert not at.exception
    chart_bytes = _assistant_messages(at)[-1]["chart_bytes"]
    assert isinstance(chart_bytes, bytes)
    assert chart_bytes.startswith(b"\x89PNG")


def test_clear_conversation_button_empties_history(app):
    at = app.run()
    at.chat_input[0].set_value("What are our top 3 customer deals?").run()
    assert len(at.session_state["messages"]) > 1

    _button_by_label(at, "🗑️ Clear Conversation").click().run()

    assert not at.exception
    assert at.session_state["messages"] == []


def test_graph_import_failure_is_reported_in_chat_instead_of_crashing(app, monkeypatch):
    monkeypatch.setitem(sys.modules, "graph", None)  # forces ImportError on `from graph import ...`

    at = app.run()
    at.chat_input[0].set_value("What are our top 3 customer deals?").run()

    assert not at.exception
    assert "Graph engine could not be loaded" in _assistant_messages(at)[-1]["content"]
