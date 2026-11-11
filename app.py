"""
app.py - Streamlit Web Interface for Sales Data Agent

Launch Command:
    streamlit run app.py

Description:
    Interactive chat interface for the autonomous Plan & Execute multi-agent system.
    Integrates LangGraph orchestration (planner, executor, cortex_researcher,
    web_researcher, chart_generator, chart_summarizer, synthesizer), real-time
    chart artifact rendering, and session-state conversation history.
"""

import os
import sys
import time
import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

# =====================================================================
# Safe Streamlit Import with Informative Fallback
# =====================================================================
try:
    import streamlit as st
except ImportError:
    print("=" * 75)
    print(" [ERROR] Streamlit is not installed in the current environment.")
    print(" Please install project dependencies:")
    print("     pip install -r requirements.txt")
    print(" Then run the web application:")
    print("     streamlit run app.py")
    print("=" * 75)
    sys.exit(1)

# Configure module logger
logger = logging.getLogger("sales_data_agent.app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =====================================================================
# Graph Engine Import
# =====================================================================
from state import DEFAULT_ENABLED_AGENTS as DEFAULT_AGENTS
from i18n import SUPPORTED_LANGUAGES, normalize_language, sample_queries, t

try:
    from graph import graph as compiled_graph, run_agent
except ImportError as err:
    # `err` is unbound once the except block ends, so keep the message for the fallback.
    GRAPH_IMPORT_ERROR = str(err)
    logger.error(f"Failed to import graph engine from graph.py: {GRAPH_IMPORT_ERROR}")
    compiled_graph = None
    def run_agent(query: str, **kwargs) -> Dict[str, Any]:
        return {
            "final_answer": t("graph_unavailable", kwargs.get("language"), error=GRAPH_IMPORT_ERROR),
            "plan": {},
        }

try:
    from evals import evaluate_query_results, setup_trulens_recorder
    EVALS_AVAILABLE = True
except ImportError as err:
    logger.warning(f"TruLens evaluation module unavailable: {err}")
    EVALS_AVAILABLE = False

from artifacts import read_chart_bytes  # noqa: E402


def get_trulens_recorder() -> Any:
    """
    Returns this browser session's TruGraph (or MockTruGraph) recorder.
    One recorder per session: a TruGraph refuses a second concurrent `with`
    block ("Already recording with a context manager, cannot nest!"), so a
    server-wide recorder would fail when two users query at the same time.
    Wrapping the shared compiled graph again is safe: TruLens skips methods
    that already carry its instrumentation flag.
    """
    if "trulens_recorder" not in st.session_state:
        st.session_state.trulens_recorder = setup_trulens_recorder(compiled_graph, app_version="Streamlit UI")
    return st.session_state.trulens_recorder


def browser_language() -> str:
    """Spanish when the browser locale is Spanish (any region), English otherwise."""
    try:
        return normalize_language(st.context.locale)
    except Exception:  # older Streamlit or no browser context (e.g. AppTest)
        return normalize_language(None)


def render_scorecard(scores: List[Dict[str, Any]], lang: str) -> None:
    with st.expander(t("scorecard_title", lang)):
        columns = st.columns(4)
        for index, row in enumerate(scores):
            value = "n/a" if row["score"] is None else f"{row['score']:.2f}"
            columns[index % 4].metric(row["metric"], value, help=row["reason"])


# =====================================================================
# 1. Page Configuration and Styling
# =====================================================================

st.set_page_config(
    page_title="Sales Data Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished interface styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .agent-badge {
        display: inline-block;
        background-color: #e9ecef;
        color: #495057;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
        margin-right: 4px;
        margin-bottom: 4px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# 2. Sidebar: Multi-Agent Architecture and Controls
# =====================================================================

with st.sidebar:
    language_codes = list(SUPPORTED_LANGUAGES)
    default_language = browser_language()
    lang = st.selectbox(
        t("language_label"),
        options=language_codes,
        index=language_codes.index(default_language),
        format_func=SUPPORTED_LANGUAGES.get,
        key="language",
    )

    st.header(t("system_overview", lang))
    st.markdown(t("system_description", lang))

    st.markdown("---")
    st.subheader(t("agents_header", lang))
    st.markdown(t("agents_description", lang))

    st.markdown("---")
    st.subheader(t("run_config", lang))
    selected_agents = st.multiselect(
        t("enabled_agents", lang),
        options=DEFAULT_AGENTS,
        default=DEFAULT_AGENTS,
        help=t("enabled_agents_help", lang),
    )
    run_evaluation = st.checkbox(
        t("run_evaluation", lang),
        value=True,
        disabled=not EVALS_AVAILABLE,
    )

    st.markdown("---")
    st.subheader(t("sample_questions", lang))

    selected_sample = None
    for idx, sq in enumerate(sample_queries(lang)):
        if st.button(f"📌 {sq[:38]}...", key=f"sample_{idx}", width="stretch"):
            selected_sample = sq

    st.markdown("---")
    # Clear conversation history button
    if st.button(t("clear_conversation", lang), width="stretch"):
        st.session_state.messages = []
        st.rerun()

    # Environment Status Indicators
    st.markdown("---")
    st.subheader(t("connection_status", lang))
    openai_set = bool(os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY", "").startswith("sk-..."))
    tavily_set = bool(os.getenv("TAVILY_API_KEY") and not os.getenv("TAVILY_API_KEY", "").startswith("tvly-..."))
    snowflake_set = bool(os.getenv("SNOWFLAKE_ACCOUNT") and os.getenv("SNOWFLAKE_USER"))

    def _status(is_set: bool) -> str:
        return t("connected", lang) if is_set else t("fallback", lang)

    st.caption(f"• OpenAI API: {_status(openai_set)}")
    st.caption(f"• Tavily Search: {_status(tavily_set)}")
    st.caption(f"• Snowflake Cortex: {_status(snowflake_set)}")


# =====================================================================
# 3. Main Chat Interface
# =====================================================================

st.markdown('<div class="main-title">Sales Data Agent 📊</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-title">{t("page_subtitle", lang)}</div>',
    unsafe_allow_html=True,
)

# Initialize chat history in session_state. The welcome entry stores readable
# text, and its `welcome` flag makes the render loop show it in the current
# language, so switching languages translates it.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": t("welcome", lang),
            "welcome": True,
            "chart_bytes": None,
            "plan": None,
            "scores": None,
        }
    ]

# Render persistent conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(t("welcome", lang) if msg.get("welcome") else msg["content"])
        if msg.get("chart_bytes"):
            st.image(msg["chart_bytes"], caption=t("chart_caption", lang), width="stretch")
        if msg.get("plan"):
            with st.expander(t("view_plan", lang)):
                st.json(msg["plan"])
        if msg.get("scores"):
            render_scorecard(msg["scores"], lang)


# =====================================================================
# 4. User Input Handling and Multi-Agent Execution
# =====================================================================

# Get user input from either chat input widget or sidebar sample click
user_query = st.chat_input(t("chat_placeholder", lang))
if selected_sample:
    user_query = selected_sample

if user_query:
    # 1. Display user query in chat
    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
        "chart_bytes": None,
        "plan": None,
        "scores": None,
    })
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Execute graph workflow with progress spinner
    with st.chat_message("assistant"):
        with st.spinner(t("spinner", lang)):
            start_timestamp = time.time()
            final_state: Dict[str, Any] = {}
            chart_bytes: Optional[bytes] = None
            execution_plan: Optional[Dict[str, Any]] = None
            scores: Optional[List[Dict[str, Any]]] = None
            evaluate = run_evaluation and EVALS_AVAILABLE and compiled_graph is not None

            try:
                # Invoke the LangGraph multi-agent pipeline (inside the TruLens recorder when enabled)
                recorder = get_trulens_recorder() if evaluate else nullcontext()
                with recorder as recording:
                    final_state = run_agent(
                        user_query, enabled_agents=selected_agents or None, verbose=False, language=lang,
                    )
                final_answer = final_state.get("final_answer", t("no_final_answer", lang))
                execution_plan = final_state.get("plan")
                if evaluate:
                    scores = evaluate_query_results(final_state, recording=recording)

            except Exception as exc:
                logger.exception("Exception occurred during multi-agent graph execution")
                final_answer = f"{t('execution_error', lang)}\n\n```\n{str(exc)}\n```"

            # 3. Chart detection: graph state chart_path first, legacy file written during this run second
            chart_bytes = read_chart_bytes(final_state, start_timestamp)

            # 4. Render assistant response and artifacts
            st.markdown(final_answer)

            if chart_bytes:
                st.image(chart_bytes, caption=t("chart_caption", lang), width="stretch")

            if execution_plan:
                with st.expander(t("view_multi_agent_plan", lang)):
                    st.json(execution_plan)

            if scores:
                render_scorecard(scores, lang)

            # 5. Persist assistant turn (chart stored as bytes so history never changes)
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "chart_bytes": chart_bytes,
                "plan": execution_plan,
                "scores": scores,
            })


# =====================================================================
# 5. Direct Execution Guardrail
# =====================================================================

if __name__ == "__main__":
    # Check if executed outside of streamlit runtime
    try:
        from streamlit.runtime import exists as streamlit_runtime_exists
        if not streamlit_runtime_exists():
            print("\n" + "=" * 75)
            print("   SALES DATA AGENT - STREAMLIT WEB INTERFACE")
            print("=" * 75)
            print("This application is designed to be served using Streamlit.")
            print("To start the web server, run the following command in your terminal:")
            print("\n    streamlit run app.py\n")
            print("=" * 75 + "\n")
    except ImportError:
        pass

