# Sales Data Agent: Autonomous Multi-Agent System with LangGraph & TruLens

An enterprise-grade, autonomous Multi-Agent Sales Intelligence system built with **LangGraph**, integrating **Snowflake Cortex** (Analyst Text-to-SQL & Cortex Search), **Tavily Web Search**, dynamic Python REPL visualization, and full-stack observability with **TruLens** (RAG Triad + GPA Alignment).

---

## Architecture Overview

The system implements an adaptive **Dynamic Plan & Execute** pattern. Rather than relying on a static linear pipeline or an unconstrained single ReAct loop, the workflow separates strategic planning, dynamic step dispatching, and specialized task execution.

```mermaid
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
```

### Key Architectural Pillars

1. **Planner Node (`planner_node`)**:
   - Uses an advanced reasoning model (`o3` / `gpt-4o` with strict JSON output format).
   - Generates sequential plans with **GPA Alignment (Goal-Plan-Act)**: each step includes a designated agent, concrete action, explicit verifiable sub-goal, preconditions, and postconditions.
   - Supports **Dynamic Replanning**: if an upstream agent encounters blockers or missing data, the executor triggers a replan; the planner receives the previous plan, failure reasons (`last_reason`), and recent message history to synthesize an updated course of action.

2. **Executor Node (`executor_node`)**:
   - Evaluates step-by-step progress and evidence collected across messages.
   - Enforces execution safeguards with bounded replan accounting (`MAX_REPLANS = 3` per step and `MAX_TOTAL_REPLANS = 5` per run) to eliminate infinite loops.
   - Dispatches dynamically to specialized worker agents using LangGraph's `Command(goto=..., update=...)`.

3. **Heterogeneous Data Retrieval**:
   - **Internal Enterprise Context (`cortex_researcher`)**:
     - *Cortex Analyst*: Converts natural language to SQL queries against structured CRM deal metrics using semantic data models (`@sales_intelligence.data.models/sales_metrics_model.yaml`) on the `SALES_INTELLIGENCE_WH` warehouse.
     - *Cortex Search*: Performs vector/semantic similarity search across unstructured customer meeting notes and call transcripts.
   - **External Public Intelligence (`web_researcher`)**:
     - Real-time web retrieval via Tavily Search for regulatory shifts, public market valuations, competitor actions, and industry news.

4. **Visualization & Synthesis**:
   - **`chart_generator`**: Autonomous ReAct agent that writes Python plotting scripts, executes them in a local REPL environment, and outputs `CHART_PATH` and `CHART_NOTES`.
   - **`chart_summarizer`**: Concise captioning agent that synthesizes the visual findings into 1-3 crisp analytical sentences.
   - **`synthesizer`**: Terminal node that consolidates multi-source evidence into an executive prose report complete with numerical metrics and citations.

5. **Full-Stack Observability & Evaluation with TruLens**:
   - **RAG Triad**: Real-time evaluation of *Groundedness*, *Answer Relevance*, and *Context Relevance*.
   - **GPA Alignment**: Comprehensive LLM-as-a-judge evaluations auditing *Plan Quality*, *Plan Adherence*, *Execution Efficiency*, and *Logical Consistency*.
   - **Inline Evaluations**: Runtime feedback loops (`@inline_evaluation(f_context_relevance)`) on retrieval nodes to trigger corrective replanning when retrieved context quality drops.

---

## Modular Project Structure

```text
Sales_Data_Agent/
├── .env.template          # Environment variables template with configuration guidance
├── requirements.txt       # Production dependencies with minimum versions
├── README.md              # Project documentation, architecture, and setup instructions
├── pytest.ini             # Test runner configuration
├── Dockerfile             # Hardened multi-stage image (non-root, health check, no secrets)
├── docker-compose.yml     # Streamlit app + TruLens dashboard on a shared data volume
├── .dockerignore          # Keeps secrets, caches and local state out of the build context
├── .github/workflows/
│   └── ci.yml             # CI: quality & secret scan, pytest, mock end-to-end smoke test
├── app.py                 # Interactive Streamlit web interface with chat and chart rendering
├── artifacts.py           # Chart artifacts for the UI (only images written during the current run)
├── state.py               # State schema, Pydantic models, `.env` loading, shared messages & agent registry
├── prompts.py             # System prompts, GPA plan prompts, and executor decision prompts
├── tools.py               # CortexAgentTool (read-only SQL guardrail), Tavily, sandboxed Python tool
├── sandbox.py             # Three-layer sandbox for chart code (AST policy, restricted builtins, isolated subprocess)
├── nodes.py               # Orchestrator nodes (planner, executor) & specialized ReAct workers
├── graph.py               # StateGraph construction, build_graph(), run_agent() & topology
├── evals.py               # TruLens RAG Triad + GPA Alignment evaluation suite & recorder
└── tests/                 # pytest suite (see "Testing & Continuous Integration")
```

### Module Breakdown

- **[`app.py`](app.py)**: Streamlit web application providing a conversational UI (`st.chat_message`, `st.chat_input`), real-time execution status (`st.spinner`), an enabled-agents selector, an optional TruLens scorecard per answer, expandable execution plans, and chart rendering stored as bytes per message.
- **[`state.py`](state.py)**: Extends LangGraph's `MessagesState` with typed attributes: `user_query`, `enabled_agents`, `plan`, `current_step`, `agent_query`, `last_reason`, `replan_flag`, `replan_attempts`, `final_answer`, `chart_path`, and `chart_notes`. Includes `PlanStep` and `ExecutorDecision` Pydantic models (enforced by the planner and executor), `load_environment()`, the shared message classes, and `DEFAULT_ENABLED_AGENTS`.
- **[`prompts.py`](prompts.py)**: Centralizes all prompt engineering. Features `plan_prompt` (incorporating L6 GPA sub-goals and replanning notices), `executor_prompt` (JSON router schema), and system prompts for all ReAct agents and the synthesizer.
- **[`tools.py`](tools.py)**: Encapsulates external integrations:
  - `CortexAgentTool`: Snowflake Snowpark and Cortex Lite Agent Service runner. Creates the Snowpark session from `SNOWFLAKE_*` variables, uses `SNOWFLAKE_WAREHOUSE`, and only executes single read-only `SELECT`/`WITH` statements. Includes an automatic high-fidelity mock fallback when credentials are not configured.
  - `tavily_tool` / `get_tavily_tool`: Web search with `MockTavilySearch` fallback.
  - `python_repl_tool` / `PythonREPLTool`: Sandboxed Python shell for plotting charts. Code is validated, then executed in a separate, secret-free child process with a timeout (see [Security Architecture](#security-architecture--defense-in-depth)).
  - `query_cortex_agent`, `search_web`, `run_python_code`: `(str) -> str` wrappers given to the ReAct agents; retrieval wrappers emit TruLens `RETRIEVAL` spans.
- **[`nodes.py`](nodes.py)**: Implements all graph nodes. Encapsulates `planner_node`, `executor_node`, `cortex_agents_research_node`, `web_research_node`, `chart_node`, `chart_summary_node`, and `synthesizer_node`.
- **[`graph.py`](graph.py)**: Assembles and compiles the `StateGraph(State)`. Exposes `build_graph()`, `run_agent()`, and graph topology visualization functions (`get_graph_mermaid()`, `display_ascii_graph()`).
- **[`evals.py`](evals.py)**: Configures the 7 evaluation metrics across the RAG Triad and GPA dimensions. Implements `setup_trulens_recorder()` with SQLite persistence (location from `TRULENS_DATABASE_URL`) and the `@inline_evaluation` decorator.

---

## Security Architecture & Defense-in-Depth

The agent executes LLM-generated code and SQL, so every path that touches code, data, or credentials has more than one guard.

### Chart code sandbox (`sandbox.py`)

| Layer | What it does | What it stops |
|-------|--------------|---------------|
| 1. AST policy | Parses the code before running it: import allowlist (`matplotlib`, `seaborn`, `pandas`, `numpy`, ...), forbidden builtins (`eval`, `exec`, `open`, `getattr`, ...), and forbidden attributes (dunders, private members, `os`, `subprocess`, `environ`, library file readers such as `pd.read_*`, `np.loadtxt`, `plt.imread`, and file writers such as `np.save`, `to_excel`, or matplotlib's `canvas.print_*`). `savefig()` only accepts a literal, relative image path (`.png`, `.jpg`, `.jpeg`, `.svg`) with no `..`, and string formatters (`to_csv`, `to_json`, `to_html`, `to_string`, ...) are only allowed without a file destination. | Imports of OS/process modules, reflection escapes, reading `.env` or overwriting files through allowed libraries. |
| 2. Restricted builtins | Runs the code with a minimal `__builtins__` and a whitelist-checking `__import__`, on the headless `Agg` backend. | Anything layer 1 could miss at runtime. |
| 3. Process isolation | Runs validated code in a short-lived child interpreter (`python -I`) with an allowlisted environment (no variable containing `KEY`, `SECRET`, `TOKEN`, `PASSWORD`), a hard timeout (15 s by default), and a throwaway working directory. Only the chart images declared through `savefig()` are copied back. | Credential theft from process memory or env, infinite loops (`while True: pass`), crashes of the Streamlit process, and files dropped by writers that layer 1 does not know about. |

If process creation is unavailable (for example in a restricted embedded runtime), execution falls back to layers 1 and 2 in-process and logs a warning. The child can still reach absolute paths and the network if a library writer slips past layer 1, so internet-facing deployments should run inside the provided Docker image.

### Data and credentials

- **Read-only SQL guardrail**: `CortexAgentTool` only executes a single `SELECT`/`WITH` statement; DML, DDL, multiple statements, and dangerous functions are rejected before reaching Snowflake. The warehouse name is validated as an identifier.
- **Least-privilege role**: `.env.template` uses `SNOWFLAKE_ROLE=SALES_ANALYST_RO`; a test fails if `ACCOUNTADMIN` is ever reintroduced.
- **Chart path sanitization**: the `CHART_PATH` reported by the chart agent is resolved and only accepted inside the working directory with an image extension, so the UI never reads arbitrary files.
- **Secrets stay out of artifacts**: `.env` files are git- and docker-ignored, the image runs as the unprivileged `appuser`, and CI scans tracked files for leaked keys.

---

## Environment Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 (recommended)
- Git

### 2. Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration (`.env`)

Copy the provided `.env.template` to `.env`:

```bash
# Windows
copy .env.template .env

# macOS / Linux
cp .env.template .env
```

Edit `.env` to supply your credentials. Every module loads `.env` automatically at import time (variables already exported in your shell take precedence). Set `SALES_AGENT_SKIP_DOTENV=1` to disable loading.

> **Security:** connect Snowflake with a least-privilege, read-only role (for example `SALES_ANALYST_RO`). Never use `ACCOUNTADMIN` for an AI agent.


```ini
# ==============================================================================
# SALES DATA AGENT - ENVIRONMENT CONFIGURATION
# ==============================================================================

# --- OpenAI API (Planning, Execution, Synthesis, and TruLens Judge) ---
OPENAI_API_KEY=sk-your-openai-api-key-here

# Model Selection
LLM_MODEL=gpt-4o
REASONING_LLM_MODEL=gpt-4o

# --- Tavily Search API (Public Web Research) ---
TAVILY_API_KEY=tvly-your-tavily-key-here

# --- Snowflake Credentials (Cortex Analyst & Search) ---
SNOWFLAKE_ACCOUNT=your-account-identifier
SNOWFLAKE_USER=your-username
SNOWFLAKE_PASSWORD=your-password
SNOWFLAKE_ROLE=SALES_ANALYST_RO
SNOWFLAKE_WAREHOUSE=SALES_INTELLIGENCE_WH
SNOWFLAKE_DATABASE=SALES_INTELLIGENCE
SNOWFLAKE_SCHEMA=DATA

# --- TruLens Telemetry & Tracing ---
TRULENS_OTEL_TRACING=1
TRULENS_DATABASE_URL=sqlite:///default.sqlite
TRULENS_DASHBOARD_PORT=8001
```

---

## How to Run

### 1. Run the Multi-Agent Pipeline End-to-End
Execute `graph.py` directly from your terminal:

```bash
python graph.py
```

This runs two representative end-to-end flows:
1. **Visualization Flow**: *"What are our top 3 customer deals? Chart the deal value for each."*
   - Plan: `planner` $\rightarrow$ `executor` $\rightarrow$ `cortex_researcher` $\rightarrow$ `executor` $\rightarrow$ `chart_generator` $\rightarrow$ `chart_summarizer` $\rightarrow$ `END`.
   - Generates and saves `top_deals_chart.png` in the current directory; the final answer combines the deal synthesis with the chart summary.
2. **Synthesis Flow**: *"What are our top 3 customer deals and what are the main themes in meeting notes?"*
   - Plan: `planner` $\rightarrow$ `executor` $\rightarrow$ `cortex_researcher` $\rightarrow$ `executor` $\rightarrow$ `web_researcher` $\rightarrow$ `executor` $\rightarrow$ `synthesizer` $\rightarrow$ `END`.
   - Generates an executive summary with figures and citations.

### 2. Run the TruLens Evaluation Suite
Execute `evals.py` to run an instrumented session and inspect the evaluation scorecard:

```bash
python evals.py
```

**Expected Output:**
```text
===========================================================================
   SALES DATA AGENT - TRULENS EVALUATION & OBSERVABILITY SUITE
===========================================================================

[Test] Evaluating sample query:
>> "What are our top 3 customer deals? Chart the deal value for each."

---------------------------------------------------------------------------
FINAL SYNTHESIZED RESPONSE:
---------------------------------------------------------------------------
Top 3 Customer Deals Summary:
1. HealthTech Corp: $1,250,000 (Stage: Negotiation, Product: Enterprise Compliance)
...
**Chart summary:** HealthTech Corp leads the customer pipeline with a deal valuation of $1.25M, ...

===========================================================================
TRULENS EVALUATION REPORT (RAG Triad + GPA Alignment)
===========================================================================
METRIC                   | SCORE   | CHAIN-OF-THOUGHT (CoT) REASONING
---------------------------------------------------------------------------
Answer Relevance         | 0.97    | The final answer directly addresses all questions posed in the original user query.
Context Relevance        | 0.91    | Retrieved CRM deal tables and meeting notes are tightly relevant to the required research goals.
Groundedness             | 0.95    | The response statement is fully grounded in the retrieved CRM table metrics and meeting transcripts.
Plan Quality             | 0.94    | Plan exhibits clear step sequencing, specific numerical constraints, and explicit expected schemas.
Plan Adherence           | 0.96    | Execution trace strictly followed all ordered steps without omissions.
Execution Efficiency     | 0.92    | Direct execution path: each data source queried exactly once without redundant retries.
Logical Consistency      | 0.98    | Numerical consistency maintained across steps; entity states align with conclusions.
===========================================================================
```

### 3. Launch the Streamlit Web Interface
Launch the interactive web application to interact with the multi-agent system via chat and view generated charts:

```bash
streamlit run app.py
```

Features included in the web interface:
- **Conversational Chat**: Preserves multi-turn conversation history using `st.session_state` and `st.chat_message`.
- **Live Status & Feedback**: Displays `st.spinner("Agents researching and analyzing...")` while orchestrating nodes.
- **Agent Selection**: Sidebar `Enabled agents` multiselect passed to `run_agent(enabled_agents=...)`.
- **TruLens Scorecard**: `Run TruLens Evaluation Scorecard` checkbox shows the 7 metrics (RAG Triad + GPA) per answer, recorded by a TruLens recorder dedicated to each browser session.
- **Dynamic Chart Rendering**: Displays the chart reported in the graph state (`chart_path`) only if it was written during the current run, and stores its bytes in the chat history, so earlier charts are never overwritten.
- **Plan Inspection**: Collapsible JSON tree viewer for inspecting the generated multi-agent execution plan.
- **Sidebar Diagnostics**: Real-time connection indicators for OpenAI, Tavily, and Snowflake Cortex.

### 4. Launch the TruLens Observability Dashboard
Once traces and evaluation records have been logged to the SQLite database, launch the TruLens dashboard:

```bash
python -c "from trulens.dashboard import run_dashboard; run_dashboard(port=8001)"
```
Navigate to `http://localhost:8001` in your browser to explore execution traces, latency, token consumption, and feedback distributions.

---

## Running with Docker & Docker Compose

Start the Streamlit app and the TruLens dashboard together:

```bash
docker compose up --build
```

| Service | URL | Purpose |
|---------|-----|---------|
| `sales-data-agent` | http://localhost:8501 | Streamlit chat interface |
| `trulens-dashboard` | http://localhost:8001 | TruLens traces and feedback (starts once the app is healthy; bound to `127.0.0.1` only) |

How it is wired:

- **Credentials**: an optional `.env` in the project root is injected at runtime (`env_file`, `required: false`). Without it, everything runs in Mock Fallback mode. Secrets are never copied into the image.
- **Persistence**: both services mount the `trulens-data` volume at `/app/data` and use `TRULENS_DATABASE_URL=sqlite:////app/data/default.sqlite`, so records survive restarts and the dashboard reads what the app writes. The scorecard reads only the records of the current run. Real TruLens records require `OPENAI_API_KEY`; in mock mode the database stays empty.
- **Dashboard exposure**: the dashboard has no authentication and shows every recorded query and answer, so it listens on the host loopback only and receives no credentials. Put it behind an authenticating proxy before exposing it.
- **Hardening**: the image runs as `appuser` (UID 10001), checks `/_stcore/health`, and the containers drop all Linux capabilities with `no-new-privileges`.

Stop the stack with `docker compose down` (add `-v` only if you want to delete the TruLens data volume).

---

## Testing & Continuous Integration

Run the full suite locally:

```bash
python -m pytest -v
```

The suite forces Mock Fallback mode (no network, no `.env`), so it is deterministic and free to run.

| Test module | What it validates |
|-------------|-------------------|
| `test_state_and_prompts.py` | Pydantic schemas, `.env` loading, prompts, and the `.env.template` least-privilege contract. |
| `test_tools_and_security.py` | Mock integrations, the read-only SQL guardrail, and all three sandbox layers (policy, timeout, secret-free child). |
| `test_nodes_and_graph.py` | Planner/executor routing, per-step and per-run replanning limits, chart flows (including a disabled `chart_summarizer`), and end-to-end graph runs. |
| `test_artifacts.py` | Chart freshness: images left by earlier runs are never attached to a new answer. |
| `test_production_contracts.py` | Production code paths without network: Cortex SSE parsing and fallback, `ChatOpenAI` construction, TruLens feedback reading, `RETRIEVAL` span attributes, database URL resolution. |
| `test_streamlit_app.py` | The real UI via Streamlit `AppTest`: initial load, chat answers with plan and 7-metric scorecard, chart persistence, clearing history, no deprecation warnings. |
| `test_container_config.py` | Docker invariants: non-root user, health check, `Agg` backend, no `.env` in the image, shared TruLens volume. |

[GitHub Actions](.github/workflows/ci.yml) runs on pushes to `main`, `fix/**`, `feat/**`, and on pull requests to `main`, with read-only permissions and cancellation of superseded runs:

1. **quality-and-security**: installs dependencies on Python 3.12, compiles every module, and scans tracked files for leaked secrets or committed `.env` files.
2. **test-suite**: `pytest -v --tb=short`.
3. **e2e-smoke-test**: `python graph.py` and `python evals.py` in Mock Fallback mode.

---

## Offline Fallback & Mock Testing Mode

This project is built with **zero-friction testability**:
- **Automatic Fallback Detection**: If `OPENAI_API_KEY`, `TAVILY_API_KEY`, or `SNOWFLAKE_*` credentials are not present, or if external packages are not yet installed in your environment, the codebase gracefully defaults to internal high-fidelity mock engines (`MockChatModel`, `MockTavilySearch`, `MockCortexAgentTool`, `MockTruGraph`).
- **Reproducible CI/CD**: Unit and integration tests can execute in air-gapped or CI/CD pipelines without making outbound network requests or incurring LLM API costs.
- **Production Readiness**: As soon as valid environment credentials and dependencies are supplied, the exact same interfaces automatically transition to production LLMs and live Snowflake Cortex services with zero code modifications.

---

## License

MIT License. Designed for enterprise multi-agent workflows with LangGraph and TruLens.

