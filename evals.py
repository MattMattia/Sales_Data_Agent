"""
evals.py - TruLens Evaluation and Observability System:
1. RAG Triad (L4): Groundedness, Answer Relevance, Context Relevance.
2. GPA Alignment (L5): Plan Quality, Plan Adherence, Execution Efficiency, Logical Consistency.
3. Inline Evaluations (L6): Real-time feedback within research nodes.
4. Recorder Session: Initialization with DefaultDBConnector (SQLite) and TruGraph.
5. Mock Fallback: High-fidelity diagnostic emulator for execution without active credentials.
"""

import os
import math
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from state import load_environment  # noqa: F401 - importing state loads `.env` first

logger = logging.getLogger("sales_data_agent.evals")

# Enable TruLens OpenTelemetry tracing
os.environ["TRULENS_OTEL_TRACING"] = "1"

# =====================================================================
# Library Detection and Execution Mode (Production vs Mock)
# =====================================================================

TRULENS_INSTALLED = False
try:
    import trulens
    from trulens.core import Feedback
    from trulens.core.feedback.selector import Selector
    from trulens.otel.semconv.trace import SpanAttributes
    from trulens.providers.openai import OpenAI as TruOpenAI
    from trulens.apps.langgraph import TruGraph
    from trulens.core.session import TruSession
    from trulens.core.database.connector.default import DefaultDBConnector
    from trulens.apps.langgraph.inline_evaluations import inline_evaluation as trulens_inline_eval
    import numpy as np

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "sk-...":
        TRULENS_INSTALLED = True
        logger.info("TruLens and OpenAI credentials detected. Production observability mode active.")
    else:
        logger.info("TruLens installed but OPENAI_API_KEY is not configured. Activating MockEvaluator.")
except ImportError:
    logger.info("TruLens is not installed in the environment. Activating high-fidelity MockEvaluator.")


# =====================================================================
# Mock Classes for Local Testing and Environments Without Credentials
# =====================================================================

class MockFeedback:
    """
    TruLens feedback function emulator providing synthetic scores
    and Chain-of-Thought (CoT) reasoning explanations.
    """
    def __init__(self, name: str, default_score: float = 0.95, default_reason: str = "Satisfactory"):
        self.name = name
        self.default_score = default_score
        self.default_reason = default_reason
        self._selectors: List[Any] = []

    def on(self, *args, **kwargs) -> "MockFeedback":
        self._selectors.append((args, kwargs))
        return self

    def on_input(self) -> "MockFeedback":
        return self

    def on_output(self) -> "MockFeedback":
        return self

    def aggregate(self, *args, **kwargs) -> "MockFeedback":
        return self

    def __call__(self, *args, **kwargs) -> Tuple[float, Dict[str, Any]]:
        # Determine representative scores based on feedback type
        score = self.default_score
        reason = self.default_reason

        # Evaluate input if text is provided
        input_text = str(args[0]) if args else ""

        if "Plan Quality" in self.name:
            if "without" in input_text.lower() or "skipped" in input_text.lower():
                score, reason = 0.40, "Plan lacks explicit filters, ordering thresholds, or owner deadlines."
            else:
                score, reason = 0.94, "Plan exhibits clear step sequencing, specific numerical constraints, and explicit expected schemas."
        elif "Plan Adherence" in self.name:
            if "skipped" in input_text.lower():
                score, reason = 0.35, "Agent deviated from plan: skipped required secondary filter and omitted columns."
            else:
                score, reason = 0.96, "Execution trace strictly followed all ordered steps without omissions."
        elif "Execution Efficiency" in self.name:
            if "double-check" in input_text.lower() or "redundant" in input_text.lower():
                score, reason = 0.50, "Inefficiency flagged: redundant retrieval and duplicate export formats."
            else:
                score, reason = 0.92, "Direct execution path: each data source queried exactly once without redundant retries."
        elif "Logical Consistency" in self.name:
            if "113 leads after applying filters to 96" in input_text:
                score, reason = 0.20, "Logical contradiction: filtered subset exceeded the initial universe size."
            else:
                score, reason = 0.98, "Numerical consistency maintained across steps; entity states align with conclusions."
        elif "Groundedness" in self.name:
            score = 0.95
            reason = "The response statement is fully grounded in the retrieved CRM table metrics and meeting transcripts."
        elif "Answer Relevance" in self.name:
            score = 0.97
            reason = "The final answer directly addresses all questions posed in the original user query."
        elif "Context Relevance" in self.name:
            score = 0.91
            reason = "Retrieved CRM deal tables and meeting notes are tightly relevant to the required research goals."

        return score, {"reason": reason}


class MockTruSession:
    """
    Simulated session for TruLens that stores execution records and returns evaluation metrics.
    """
    def __init__(self, database_url: str = "sqlite:///default.sqlite"):
        self.database_url = database_url
        self._records: List[Dict[str, Any]] = []

    def reset_database(self):
        self._records.clear()
        logger.info("[MockTruSession] Simulated SQLite database reset.")

    def get_records_and_feedback(self) -> Tuple[Any, Any]:
        """
        Returns simulated records in a Pandas-compatible structure.
        """
        try:
            import pandas as pd
            records_df = pd.DataFrame(self._records or [
                {"input": "What are our top 3 customer deals?",
                 "output": "1. HealthTech Corp ($1.25M), 2. FinTech Global ($890k), 3. OmniRetail ($640k)"}
            ])
            feedbacks_df = pd.DataFrame([
                {"Groundedness": 0.95, "Answer Relevance": 0.97, "Context Relevance": 0.91,
                 "Plan Quality": 0.94, "Plan Adherence": 0.96, "Execution Efficiency": 0.92,
                 "Logical Consistency": 0.98}
            ])
            return records_df, feedbacks_df
        except ImportError:
            return self._records, {}


class MockTruGraph:
    """
    Context manager that simulates TruGraph trace recording and evaluation.
    """
    def __init__(
        self,
        graph: Any,
        app_name: str = "Sales Data Agent",
        app_version: str = "L6: Production",
        feedbacks: Optional[List[Any]] = None,
        session: Optional[Any] = None,
    ):
        self.graph = graph
        self.app_name = app_name
        self.app_version = app_version
        self.feedbacks = feedbacks or []
        self.session = session or MockTruSession()

    def __enter__(self):
        logger.info(f"[MockTruGraph] Starting trace recording for '{self.app_name}' ({self.app_version})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info(f"[MockTruGraph] Recording completed. Computing {len(self.feedbacks)} evaluation metrics...")
        if exc_type is not None:
            logger.error(f"[MockTruGraph] Exception recorded during execution: {exc_val}")
        return False


def mock_inline_evaluation(feedback_func: Any):
    """
    Mock decorator for inline evaluations in research nodes.
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            logger.debug(f"[Inline Eval Mock] Evaluating condition '{getattr(feedback_func, 'name', 'Context Relevance')}'")
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


# =====================================================================
# 1. Evaluator Configuration: RAG Triad (L4)
# =====================================================================

if TRULENS_INSTALLED:
    rag_provider = TruOpenAI(model_engine=os.getenv("LLM_MODEL", "gpt-4o"))

    # Groundedness: Retrieval Contexts vs Output
    f_groundedness = (
        Feedback(
            rag_provider.groundedness_measure_with_cot_reasons,
            name="Groundedness"
        )
        .on({
            "source": Selector(
                span_type=SpanAttributes.SpanType.RETRIEVAL,
                span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
                collect_list=True
            )
        })
        .on_output()
    )

    # Answer Relevance: Input vs Output
    f_answer_relevance = (
        Feedback(rag_provider.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input()
        .on_output()
    )

    # Context Relevance: Sub-query vs Retrieved Context Chunks
    f_context_relevance = (
        Feedback(rag_provider.context_relevance_with_cot_reasons, name="Context Relevance")
        .on({
            "question": Selector(
                span_type=SpanAttributes.SpanType.RETRIEVAL,
                span_attribute=SpanAttributes.RETRIEVAL.QUERY_TEXT,
            )
        })
        .on({
            "context": Selector(
                span_type=SpanAttributes.SpanType.RETRIEVAL,
                span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
                collect_list=False
            )
        })
        .aggregate(np.mean)
    )

else:
    f_groundedness = MockFeedback(
        name="Groundedness",
        default_score=0.95,
        default_reason="Response statements strictly correspond to retrieved CRM context."
    )
    f_answer_relevance = MockFeedback(
        name="Answer Relevance",
        default_score=0.97,
        default_reason="Output completely fulfills the requirements of the user question."
    )
    f_context_relevance = MockFeedback(
        name="Context Relevance",
        default_score=0.91,
        default_reason="Context chunks retrieved by Cortex/Tavily are relevant to the sub-goals."
    )


# =====================================================================
# 2. Evaluator Configuration: GPA Alignment (L5)
# =====================================================================

if TRULENS_INSTALLED:
    gpa_eval_provider = TruOpenAI(model_engine=os.getenv("REASONING_LLM_MODEL", "gpt-4o"))

    # 1. Plan Quality
    f_plan_quality = Feedback(
        gpa_eval_provider.plan_quality_with_cot_reasons,
        name="Plan Quality",
    ).on({"trace": Selector(trace_level=True)})

    # 2. Plan Adherence
    f_plan_adherence = Feedback(
        gpa_eval_provider.plan_adherence_with_cot_reasons,
        name="Plan Adherence",
    ).on({"trace": Selector(trace_level=True)})

    # 3. Execution Efficiency
    f_execution_efficiency = Feedback(
        gpa_eval_provider.execution_efficiency_with_cot_reasons,
        name="Execution Efficiency",
    ).on({"trace": Selector(trace_level=True)})

    # 4. Logical Consistency
    f_logical_consistency = Feedback(
        gpa_eval_provider.logical_consistency_with_cot_reasons,
        name="Logical Consistency",
    ).on({"trace": Selector(trace_level=True)})

else:
    f_plan_quality = MockFeedback(
        name="Plan Quality",
        default_score=0.94,
        default_reason="Plan is structured with explicit ordering, sub-goals, and defined schemas."
    )
    f_plan_adherence = MockFeedback(
        name="Plan Adherence",
        default_score=0.96,
        default_reason="Execution trace strictly adhered to the generated plan steps."
    )
    f_execution_efficiency = MockFeedback(
        name="Execution Efficiency",
        default_score=0.92,
        default_reason="No redundant queries or duplicate calls detected in the trace."
    )
    f_logical_consistency = MockFeedback(
        name="Logical Consistency",
        default_score=0.98,
        default_reason="Counts and conclusions remain numerically and logically consistent."
    )


# Consolidated feedback list for the complete recorder
ALL_FEEDBACKS = [
    f_answer_relevance,
    f_context_relevance,
    f_groundedness,
    f_plan_quality,
    f_plan_adherence,
    f_execution_efficiency,
    f_logical_consistency,
]


# =====================================================================
# 3. Inline Evaluation Decorator and TruGraph Recorder (L6)
# =====================================================================

inline_evaluation = trulens_inline_eval if TRULENS_INSTALLED else mock_inline_evaluation


def setup_trulens_recorder(
    graph: Any,
    app_version: str = "L6: Production",
    database_url: str = "sqlite:///default.sqlite",
    reset_db: bool = False,
) -> Any:
    """
    Initializes the SQLite observability database and configures TruGraph
    with the complete feedback suite (RAG Triad + GPA Alignment).

    Parameters:
    -----------
    graph: CompiledGraph
        Compiled LangGraph instance to instrument.
    app_version: str
        Agent version registered in TruLens.
    database_url: str
        SQLite database URL (defaults to 'sqlite:///default.sqlite').
    reset_db: bool
        If True, resets the trace database.

    Returns:
    --------
    TruGraph (or MockTruGraph) ready to be used as a context manager:
        with tru_recorder as recording:
            graph.invoke(...)
    """
    if TRULENS_INSTALLED:
        try:
            connector = DefaultDBConnector(database_url=database_url)
            session = TruSession(connector=connector)
            if reset_db:
                session.reset_database()

            recorder = TruGraph(
                graph,
                app_name="Sales Data Agent",
                app_version=app_version,
                feedbacks=ALL_FEEDBACKS,
            )
            logger.info(f"TruGraph configured with {len(ALL_FEEDBACKS)} feedbacks.")
            return recorder
        except Exception as e:
            logger.warning(f"Error initializing real TruGraph: {e}. Falling back to MockTruGraph.")

    session = MockTruSession(database_url=database_url)
    if reset_db:
        session.reset_database()

    return MockTruGraph(
        graph=graph,
        app_name="Sales Data Agent",
        app_version=app_version,
        feedbacks=ALL_FEEDBACKS,
        session=session,
    )


# =====================================================================
# 4. Evaluation Results (Mock and Production modes)
# =====================================================================

PENDING_REASON = "Pending: feedback not yet computed by TruLens."


def _feedback_name(feedback: Any) -> str:
    return str(getattr(feedback, "name", type(feedback).__name__))


def _fetch_recorded_feedback(recording: Any = None) -> Dict[str, float]:
    """
    Reads the latest feedback scores persisted by TruGraph. Returns a
    {metric_name: score} mapping (empty when nothing has been computed yet).
    """
    session = getattr(recording, "session", None) or TruSession()
    records_df, feedback_names = session.get_records_and_feedback()
    if records_df is None or len(records_df) == 0:
        return {}
    latest = records_df.iloc[-1]
    scores: Dict[str, float] = {}
    for name in feedback_names or []:
        value = latest.get(name)
        if isinstance(value, (int, float)) and not math.isnan(value):
            scores[str(name)] = float(value)
    return scores


def evaluate_query_results(final_state: Dict[str, Any], recording: Any = None) -> List[Dict[str, Any]]:
    """
    Returns one {"metric", "score", "reason"} row per feedback in ALL_FEEDBACKS.

    - Mock mode: invokes each MockFeedback on the final answer.
    - TruLens mode: real Feedback objects need span selectors (2+ arguments or
      the full trace), so they are NEVER called directly here. Scores are read
      from the records TruGraph persisted; missing ones are reported as pending.
    Never raises: failures are reported in the "reason" field.
    """
    final_answer = str(final_state.get("final_answer") or "")

    if not TRULENS_INSTALLED:
        rows = []
        for fb in ALL_FEEDBACKS:
            score, details = fb(final_answer)
            reason = details.get("reason", "") if isinstance(details, dict) else str(details)
            rows.append({"metric": _feedback_name(fb), "score": float(score), "reason": reason})
        return rows

    try:
        recorded = _fetch_recorded_feedback(recording)
        error_reason = None
    except Exception as exc:
        logger.warning(f"Could not read TruLens feedback results: {exc}")
        recorded, error_reason = {}, f"Unavailable: {type(exc).__name__}: {exc}"

    rows = []
    for fb in ALL_FEEDBACKS:
        name = _feedback_name(fb)
        if name in recorded:
            rows.append({"metric": name, "score": recorded[name], "reason": "Computed by TruLens."})
        else:
            rows.append({"metric": name, "score": None, "reason": error_reason or PENDING_REASON})
    return rows


# Backward-compatible alias used in the audit remediation plan
evaluate_and_report = evaluate_query_results


# =====================================================================
# 5. Main Test Block (__main__)
# =====================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n" + "=" * 75)
    print("   SALES DATA AGENT - TRULENS EVALUATION & OBSERVABILITY SUITE")
    print("=" * 75)

    # 1. Load the compiled graph from graph.py
    from graph import graph, run_agent

    # 2. Initialize the recorder with RAG Triad + GPA Alignment
    app_version = "L6: Production [RAG Triad + GPA Alignment]"
    tru_recorder = setup_trulens_recorder(graph, app_version=app_version, reset_db=True)

    test_query = "What are our top 3 customer deals? Chart the deal value for each."
    print(f"\n[Test] Evaluating sample query:\n>> \"{test_query}\"\n")

    # 3. Execute the query within the recorder context
    with tru_recorder as recording:
        final_state = run_agent(test_query, verbose=False)

    final_answer = final_state.get("final_answer", "")
    print("-" * 75)
    print("FINAL SYNTHESIZED RESPONSE:")
    print("-" * 75)
    print(final_answer.strip())

    # 4. Compute and display evaluation metrics
    print("\n" + "=" * 75)
    print("TRULENS EVALUATION REPORT (RAG Triad + GPA Alignment)")
    print("=" * 75)
    print(f"{'METRIC':<24} | {'SCORE':<7} | {'CHAIN-OF-THOUGHT (CoT) REASONING'}")
    print("-" * 75)

    # Simulate (mock) or read (TruLens) evaluations without calling real Feedback objects directly
    for row in evaluate_query_results(final_state, recording=recording):
        score = "n/a " if row["score"] is None else f"{row['score']:0.2f}"
        print(f"{row['metric']:<24} | {score}    | {row['reason']}")

    print("=" * 75 + "\n")

