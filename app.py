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
from typing import Optional, Dict, Any

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
try:
    from graph import run_agent, DEFAULT_AGENTS
except ImportError as err:
    logger.error(f"Failed to import graph engine from graph.py: {err}")
    def run_agent(query: str, **kwargs) -> Dict[str, Any]:
        return {
            "final_answer": f"Error: Graph engine could not be loaded: {err}",
            "plan": {},
        }
    DEFAULT_AGENTS = ["cortex_researcher", "web_researcher", "chart_generator", "chart_summarizer", "synthesizer"]


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
    st.header("🤖 System Overview")
    st.markdown(
        """
        **Sales Data Agent** is an autonomous multi-agent analytics pipeline built with:
        - **LangGraph** (Dynamic Plan & Execute pattern)
        - **Snowflake Cortex** (Analyst & Search)
        - **Tavily Search** (Public web benchmarks)
        - **Python REPL** (Data visualization)
        - **TruLens** (RAG Triad + GPA Alignment)
        """
    )

    st.markdown("---")
    st.subheader("👥 Active Specialist Agents")
    st.markdown(
        """
        - 🧠 **planner**: Decomposes queries into structured sub-goals.
        - 🧭 **executor**: Routes and evaluates sub-agent progression.
        - ❄️ **cortex_researcher**: Queries CRM metrics & meeting notes.
        - 🌐 **web_researcher**: Conducts real-time external research.
        - 📈 **chart_generator**: Synthesizes and plots metrics.
        - 📝 **chart_summarizer**: Generates standalone visual captions.
        - 🎯 **synthesizer**: Compiles final executive briefings.
        """
    )

    st.markdown("---")
    st.subheader("💡 Sample Questions")
    sample_queries = [
        "What are our top 3 customer deals? Chart the deal value for each.",
        "What are our top 3 customer deals and what are the main themes in meeting notes?",
        "Research recent regulatory updates affecting financial services and healthcare deals.",
    ]

    selected_sample = None
    for idx, sq in enumerate(sample_queries):
        if st.button(f"📌 {sq[:38]}...", key=f"sample_{idx}", use_container_width=True):
            selected_sample = sq

    st.markdown("---")
    # Clear conversation history button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Environment Status Indicators
    st.markdown("---")
    st.subheader("⚙️ Connection Status")
    openai_set = bool(os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY", "").startswith("sk-..."))
    tavily_set = bool(os.getenv("TAVILY_API_KEY") and not os.getenv("TAVILY_API_KEY", "").startswith("tvly-..."))
    snowflake_set = bool(os.getenv("SNOWFLAKE_ACCOUNT") and os.getenv("SNOWFLAKE_USER"))

    st.caption(f"• OpenAI API: {'🟢 Connected' if openai_set else '🟡 Fallback / Mock'}")
    st.caption(f"• Tavily Search: {'🟢 Connected' if tavily_set else '🟡 Fallback / Mock'}")
    st.caption(f"• Snowflake Cortex: {'🟢 Connected' if snowflake_set else '🟡 Fallback / Mock'}")


# =====================================================================
# 3. Main Chat Interface
# =====================================================================

st.markdown('<div class="main-title">Sales Data Agent 📊</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Autonomous Multi-Agent Intelligence System for Enterprise Sales Analytics</div>',
    unsafe_allow_html=True,
)

# Initialize chat history in session_state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome! I am the **Sales Data Agent**. You can ask me to analyze CRM enterprise deals, "
                "retrieve customer meeting notes, run real-time market research, or create visual charts."
            ),
            "chart_path": None,
            "plan": None,
        }
    ]

# Render persistent conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chart_path") and os.path.exists(msg["chart_path"]):
            st.image(msg["chart_path"], caption="Generated Visualization", use_container_width=True)
        if msg.get("plan"):
            with st.expander("🔍 View Execution Plan"):
                st.json(msg["plan"])


# =====================================================================
# 4. User Input Handling and Multi-Agent Execution
# =====================================================================

# Get user input from either chat input widget or sidebar sample click
user_query = st.chat_input("Ask a question about sales deals, meeting transcripts, or market research...")
if selected_sample:
    user_query = selected_sample

if user_query:
    # 1. Display user query in chat
    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
        "chart_path": None,
        "plan": None,
    })
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Execute graph workflow with progress spinner
    with st.chat_message("assistant"):
        with st.spinner("Agentes investigando y analizando..."):
            start_timestamp = time.time()
            chart_file = "top_deals_chart.png"
            chart_to_display: Optional[str] = None
            execution_plan: Optional[Dict[str, Any]] = None

            try:
                # Invoke the LangGraph multi-agent pipeline
                final_state = run_agent(user_query, verbose=False)
                final_answer = final_state.get(
                    "final_answer",
                    "The multi-agent workflow completed, but no final synthesis was returned."
                )
                execution_plan = final_state.get("plan")

            except Exception as exc:
                logger.exception("Exception occurred during multi-agent graph execution")
                final_answer = (
                    f"⚠️ **Execution Error**: An error occurred while running the multi-agent graph:\n\n"
                    f"```\n{str(exc)}\n```"
                )

            # 3. Chart verification logic:
            # Check if query requested a chart and whether 'top_deals_chart.png' exists and was modified recently
            needs_chart = any(
                keyword in user_query.lower()
                for keyword in ["chart", "plot", "grafic", "gráfico", "visualize", "visualizar", "graph"]
            )

            if needs_chart and os.path.exists(chart_file):
                mtime = os.path.getmtime(chart_file)
                # Verify modification occurred within a reasonable window around execution
                if mtime >= start_timestamp - 10:
                    chart_to_display = chart_file

            # 4. Render assistant response and artifacts
            st.markdown(final_answer)

            if chart_to_display and os.path.exists(chart_to_display):
                st.image(chart_to_display, caption="Generated Visualization", use_container_width=True)

            if execution_plan:
                with st.expander("🔍 View Multi-Agent Execution Plan"):
                    st.json(execution_plan)

            # 5. Persist assistant turn to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "chart_path": chart_to_display,
                "plan": execution_plan,
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

