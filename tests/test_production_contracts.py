"""
Contract tests for the production (credentialed) code paths.

They exercise the exact code that runs against OpenAI, Snowflake Cortex and
the TruLens database, replacing only the network boundary with stubs, so the
suite stays offline and deterministic.
"""

import json
import math
import sys
import types

import pandas as pd
import pytest

import evals
import nodes
import tools
from tools import CortexAgentTool


# ---------------------------------------------------------------------
# CortexAgentTool._consume_stream (Snowflake Cortex SSE streaming)
# ---------------------------------------------------------------------

class _Event:
    def __init__(self, data):
        self.data = data


class _Stream:
    def __init__(self, events):
        self._events = events

    def events(self):
        return iter(self._events)


def _tool_results_block(payload):
    return {"type": "tool_results", "tool_results": {"content": [{"type": "json", "json": payload}]}}


def test_consume_stream_parses_dict_and_json_deltas_and_skips_malformed_events():
    stream = _Stream([
        _Event({"delta": {"content": [{"type": "text", "text": "Top deals: "}]}}),
        _Event("not-json{"),
        _Event({"event": "ping"}),
        _Event({"delta": {"content": [{"type": "tool_results"}]}}),
        _Event({"delta": {"content": "unexpected-string"}}),
        _Event(json.dumps({"delta": {"content": [_tool_results_block({
            "text": "HealthTech leads.",
            "sql": "SELECT company_name FROM sales_metrics",
            "searchResults": [{"source_id": "conv_101", "doc_id": "meeting_healthtech"}],
        })]}})),
        _Event(json.dumps({"data": {"delta": {"content": [
            {"type": "tool_results", "tool_results": {"content": [{"type": "text", "text": "ignored"}]}},
            {"type": "text", "text": " Done."},
        ]}}})),
    ])

    text, sql, citations = CortexAgentTool()._consume_stream(stream)

    assert text == "Top deals: HealthTech leads. Done."
    assert sql == "SELECT company_name FROM sales_metrics"
    assert citations == [{"source_id": "conv_101", "doc_id": "meeting_healthtech"}]


def test_production_run_falls_back_to_mock_on_service_error(monkeypatch):
    tool = CortexAgentTool()
    query = "What are our top 3 customer deals?"

    class _FailingService:
        def run(self, request):
            raise RuntimeError("network error")

    tool._is_mock = False
    tool._session = object()
    tool._agent_service = _FailingService()
    monkeypatch.setattr(tool, "_build_request", lambda q: {"query": q})

    assert tool.run(query) == tool._run_mock(query)


# ---------------------------------------------------------------------
# nodes.initialize_chat_model (OpenAI production path)
# ---------------------------------------------------------------------

@pytest.fixture
def fake_chat_openai(monkeypatch):
    created = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            created.append(kwargs)

    module = types.ModuleType("langchain_openai")
    module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-test")
    return FakeChatOpenAI, created


@pytest.mark.parametrize(
    "json_mode, expected_kwargs",
    [(True, {"response_format": {"type": "json_object"}}), (False, {})],
)
def test_initialize_chat_model_builds_chat_openai_with_live_key(fake_chat_openai, json_mode, expected_kwargs):
    fake_class, created = fake_chat_openai

    model = nodes.initialize_chat_model("gpt-4o", json_mode=json_mode)

    assert isinstance(model, fake_class)
    assert created == [{"model": "gpt-4o", "model_kwargs": expected_kwargs}]


# ---------------------------------------------------------------------
# evals: reading persisted TruLens feedback (production mode)
# ---------------------------------------------------------------------

def _recording(records_df=None, names=None, error=None):
    class _Session:
        def get_records_and_feedback(self):
            if error is not None:
                raise error
            return records_df, names

    return types.SimpleNamespace(session=_Session())


@pytest.fixture
def feedback_names():
    return [evals._feedback_name(fb) for fb in evals.ALL_FEEDBACKS]


@pytest.fixture
def trulens_mode(monkeypatch):
    monkeypatch.setattr(evals, "TRULENS_INSTALLED", True)


def test_fetch_recorded_feedback_reads_latest_row_and_drops_nan(feedback_names):
    first, second = feedback_names[0], feedback_names[1]
    records = pd.DataFrame([
        {first: 0.10, second: 0.20},
        {first: 0.95, second: float("nan")},
    ])

    scores = evals._fetch_recorded_feedback(_recording(records, [first, second]))

    assert scores == {first: 0.95}


def test_evaluate_query_results_maps_scores_and_marks_missing_as_pending(trulens_mode, feedback_names):
    first, second = feedback_names[0], feedback_names[1]
    records = pd.DataFrame([{first: 0.95, second: float("nan")}])

    rows = evals.evaluate_query_results({"final_answer": "ok"}, recording=_recording(records, [first, second]))

    by_metric = {row["metric"]: row for row in rows}
    assert len(rows) == len(feedback_names)
    assert by_metric[first] == {"metric": first, "score": 0.95, "reason": "Computed by TruLens."}
    assert by_metric[second]["score"] is None
    assert by_metric[second]["reason"] == evals.PENDING_REASON
    assert all(not (isinstance(row["score"], float) and math.isnan(row["score"])) for row in rows)


def test_evaluate_query_results_reports_database_errors_without_raising(trulens_mode, feedback_names):
    rows = evals.evaluate_query_results(
        {"final_answer": "ok"}, recording=_recording(error=RuntimeError("database is locked")),
    )

    assert [row["metric"] for row in rows] == feedback_names
    assert all(row["score"] is None for row in rows)
    assert all(row["reason"] == "Unavailable: RuntimeError: database is locked" for row in rows)


# ---------------------------------------------------------------------
# tools: OpenTelemetry RETRIEVAL span attributes
# ---------------------------------------------------------------------

def test_retrieval_attributes_format_query_and_contexts():
    semconv = pytest.importorskip("trulens.otel.semconv.trace")
    retrieval = semconv.SpanAttributes.RETRIEVAL

    positional = tools._retrieval_attributes("Deal table", None, "top deals?")
    keyword = tools._retrieval_attributes("", None, query="empty result")

    assert positional == {retrieval.QUERY_TEXT: "top deals?", retrieval.RETRIEVED_CONTEXTS: ["Deal table"]}
    assert keyword == {retrieval.QUERY_TEXT: "empty result", retrieval.RETRIEVED_CONTEXTS: []}


def test_instrument_retrieval_preserves_the_wrapped_function_result():
    def lookup(query: str) -> str:
        return f"context for {query}"

    instrumented = tools.instrument_retrieval(lookup)

    assert instrumented("deals") == "context for deals"


# ---------------------------------------------------------------------
# evals: TruLens database location (shared Docker volume)
# ---------------------------------------------------------------------

def test_recorder_uses_trulens_database_url_from_environment(monkeypatch):
    monkeypatch.setenv("TRULENS_DATABASE_URL", "sqlite:////app/data/default.sqlite")

    recorder = evals.setup_trulens_recorder(graph=None)

    assert recorder.session.database_url == "sqlite:////app/data/default.sqlite"


def test_recorder_prefers_explicit_database_url_and_defaults_locally(monkeypatch):
    monkeypatch.setenv("TRULENS_DATABASE_URL", "sqlite:////app/data/default.sqlite")
    explicit = evals.setup_trulens_recorder(graph=None, database_url="sqlite:///custom.sqlite")
    monkeypatch.delenv("TRULENS_DATABASE_URL")
    default = evals.setup_trulens_recorder(graph=None)

    assert explicit.session.database_url == "sqlite:///custom.sqlite"
    assert default.session.database_url == "sqlite:///default.sqlite"
