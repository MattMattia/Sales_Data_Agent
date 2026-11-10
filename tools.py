"""
tools.py - Integration tools for the Multi-Agent architecture:
1. TavilySearch: Real-time web search with fallback for local/offline execution.
2. CortexAgentTool: Hybrid integration with Snowflake Cortex (Analyst Text-to-SQL
   and semantic Cortex Search) with synthetic fallback/mock when active credentials are absent.
   Generated SQL passes a read-only guardrail before execution.
3. python_repl_tool: AST-sandboxed Python execution environment for saving charts to disk.
4. LangChain-friendly wrappers with clean `(str) -> str` signatures
   (query_cortex_agent, search_web, run_python_code), instrumented as TruLens
   RETRIEVAL spans when TruLens is installed.
"""

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from state import load_environment  # noqa: F401 - importing state loads `.env` first
from sandbox import execute_sandboxed

try:
    from pydantic import BaseModel, Field, PrivateAttr
except ImportError:
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    def Field(default=None, **kwargs):
        return default
    def PrivateAttr(default=None, **kwargs):
        return default

logger = logging.getLogger("sales_data_agent.tools")


# =====================================================================
# 0. TruLens RETRIEVAL span instrumentation (optional)
# =====================================================================

def _retrieval_attributes(ret: Any, exception: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    from trulens.otel.semconv.trace import SpanAttributes
    query = kwargs.get("query", args[0] if args else "")
    return {
        SpanAttributes.RETRIEVAL.QUERY_TEXT: str(query),
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: [str(ret)] if ret else [],
    }


def instrument_retrieval(func: Callable) -> Callable:
    """
    Marks `func` as a TruLens RETRIEVAL span (QUERY_TEXT + RETRIEVED_CONTEXTS),
    which Groundedness and Context Relevance selectors require. No-op when
    TruLens is not installed, so Mock Fallback keeps working.
    """
    try:
        from trulens.core.otel.instrument import instrument
        from trulens.otel.semconv.trace import SpanAttributes
    except ImportError:
        return func
    try:
        return instrument(span_type=SpanAttributes.SpanType.RETRIEVAL, attributes=_retrieval_attributes)(func)
    except Exception as exc:  # pragma: no cover - defensive against TruLens API drift
        logger.warning(f"TruLens retrieval instrumentation unavailable: {exc}")
        return func


# =====================================================================
# 1. Tavily Search Tool Configuration (Web Research)
# =====================================================================

class TavilyArgs(BaseModel):
    query: str = Field(..., description="The search query to look up on the web")


class MockTavilySearch:
    """
    Simulated web search tool for local testing or environments
    where TAVILY_API_KEY is not configured.
    """
    name: str = "tavily_search"
    description: str = "Search the public web for news, regulatory changes, and company information."
    args_schema: Type[TavilyArgs] = TavilyArgs

    def invoke(self, input_data: Any) -> Any:
        query = input_data.get("query", "") if isinstance(input_data, dict) else str(input_data)
        logger.info(f"[MockTavilySearch] Simulating search for: {query}")

        lower_q = query.lower()
        if "bank" in lower_q or "market cap" in lower_q:
            return [
                {"title": "Largest US Banks by Market Capitalization",
                 "content": "JPMorgan Chase: $580B, Bank of America: $310B, Wells Fargo: $210B, Morgan Stanley: $170B, Goldman Sachs: $160B.",
                 "url": "https://finance.example.com/us-banks-market-cap"}
            ]
        elif "regulat" in lower_q or "compliance" in lower_q:
            return [
                {"title": "Financial Services & Healthcare Regulatory Updates 2026",
                 "content": "New regulatory guidelines emphasize strict compliance on algorithmic transparency, heightened risk controls, and automated compliance auditing.",
                 "url": "https://regulatorynews.example.com/2026-updates"}
            ]
        else:
            return [
                {"title": f"Web Results for: {query[:40]}",
                 "content": f"Relevant public information regarding {query}. Market analysis indicates steady momentum and strategic realignment.",
                 "url": "https://news.example.com/article"}
            ]

    def __call__(self, *args, **kwargs):
        if args:
            return self.invoke(args[0])
        return self.invoke(kwargs)


def get_tavily_tool(max_results: int = 5):
    """
    Returns the official Tavily search tool if TAVILY_API_KEY is present
    and the library is installed, or MockTavilySearch as a reliable fallback.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key and api_key != "tvly-...":
        try:
            from langchain_tavily import TavilySearch
            return TavilySearch(max_results=max_results)
        except ImportError:
            try:
                from langchain_community.tools.tavily_search import TavilySearchResults
                return TavilySearchResults(max_results=max_results)
            except ImportError:
                logger.warning("langchain-tavily is not installed. Using MockTavilySearch fallback.")
    else:
        logger.info("TAVILY_API_KEY not detected. Initializing MockTavilySearch.")

    return MockTavilySearch()


# Default instance for direct compatibility
tavily_tool = get_tavily_tool(max_results=5)


def _format_search_results(raw: Any) -> str:
    items = raw.get("results", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return str(items)
    lines = []
    for item in items:
        if isinstance(item, dict):
            lines.append(f"- {item.get('title', '')}: {item.get('content', '')} (source: {item.get('url', '')})")
        else:
            lines.append(f"- {item}")
    return "\n".join(lines) if lines else "No web results found."


@instrument_retrieval
def search_web(query: str) -> str:
    """Search the public web for news, market data, and regulatory updates. Returns results with source URLs."""
    return _format_search_results(tavily_tool.invoke({"query": query}))


# =====================================================================
# 2. Snowflake Cortex Agent Tool (Cortex Analyst + Cortex Search)
# =====================================================================

SEMANTIC_MODEL_FILE = "@sales_intelligence.data.models/sales_metrics_model.yaml"
CORTEX_SEARCH_SERVICE = "sales_intelligence.data.sales_conversation_search"
DEFAULT_WAREHOUSE = "SALES_INTELLIGENCE_WH"

_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]{0,254}$")
_READONLY_FIRST_TOKENS = frozenset({"SELECT", "WITH"})
_FORBIDDEN_SQL_KEYWORDS = frozenset({
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE",
    "REPLACE", "GRANT", "REVOKE", "COPY", "PUT", "REMOVE", "CALL",
    "EXECUTE", "MERGE", "UNDROP", "USE",
})
_SCALAR_FUNCTION_CALL_RE = re.compile(r"\b(?:REPLACE|INSERT)\s*\(", re.IGNORECASE)


def _normalize_sql(sql: str) -> Optional[str]:
    """
    Removes comments and replaces string literals / quoted identifiers with
    placeholders using a quote-aware scanner, so neither `'--'` nor `';'`
    inside literals can hide statements. Returns None on unterminated quotes
    or block comments.
    """
    out: List[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch, nxt = sql[i], sql[i + 1] if i + 1 < n else ""
        if ch == "-" and nxt == "-":
            end = sql.find("\n", i)
            i = n if end == -1 else end
        elif ch == "/" and nxt == "*":
            end = sql.find("*/", i + 2)
            if end == -1:
                return None
            out.append(" ")
            i = end + 2
        elif ch == "$" and nxt == "$":
            end = sql.find("$$", i + 2)
            if end == -1:
                return None
            out.append(" '' ")
            i = end + 2
        elif ch in ("'", '"'):
            j = i + 1
            while j < n:
                if sql[j] == ch and j + 1 < n and sql[j + 1] == ch:
                    j += 2  # escaped quote ('' or "")
                elif sql[j] == "\\" and ch == "'":
                    j += 2
                elif sql[j] == ch:
                    break
                else:
                    j += 1
            if j >= n:
                return None
            out.append(" '' " if ch == "'" else " ident ")
            i = j + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _is_safe_readonly_sql(sql: str) -> bool:
    """
    Read-only guardrail for model-generated SQL. Accepts a single SELECT/WITH
    statement (one optional trailing ';') with no mutating/administrative keyword.
    """
    if not sql or not sql.strip():
        return False
    normalized = _normalize_sql(sql)
    if normalized is None:
        return False
    body = normalized.strip().rstrip(";").strip()
    if not body or ";" in body:
        return False
    # REPLACE(...) and INSERT(...) are also Snowflake string functions: allow them only as calls.
    body = _SCALAR_FUNCTION_CALL_RE.sub(" fn(", body)
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_$]*", body.upper())
    if not tokens or tokens[0] not in _READONLY_FIRST_TOKENS:
        return False
    return not any(token in _FORBIDDEN_SQL_KEYWORDS for token in tokens)


def configured_warehouse() -> str:
    """Returns SNOWFLAKE_WAREHOUSE when it is a valid SQL identifier, else the default."""
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", DEFAULT_WAREHOUSE).strip()
    if _SQL_IDENTIFIER_RE.match(warehouse):
        return warehouse
    logger.error("Security: SNOWFLAKE_WAREHOUSE is not a valid SQL identifier. Using default warehouse.")
    return DEFAULT_WAREHOUSE


def _has_snowflake_credentials() -> bool:
    return bool(os.getenv("SNOWFLAKE_ACCOUNT") and os.getenv("SNOWFLAKE_USER") and os.getenv("SNOWFLAKE_PASSWORD"))


def snowflake_connection_params() -> Dict[str, str]:
    """Builds Snowpark connection parameters from SNOWFLAKE_* environment variables."""
    mapping = {
        "account": "SNOWFLAKE_ACCOUNT",
        "user": "SNOWFLAKE_USER",
        "password": "SNOWFLAKE_PASSWORD",
        "role": "SNOWFLAKE_ROLE",
        "warehouse": "SNOWFLAKE_WAREHOUSE",
        "database": "SNOWFLAKE_DATABASE",
        "schema": "SNOWFLAKE_SCHEMA",
    }
    return {key: os.environ[env] for key, env in mapping.items() if os.getenv(env)}


def create_snowpark_session() -> Optional[Any]:
    """Creates a Snowpark session from environment credentials, or None on failure."""
    try:
        from snowflake.snowpark import Session
        return Session.builder.configs(snowflake_connection_params()).create()
    except Exception as exc:
        # Never log the parameters: they include the password.
        logger.warning(f"Could not create Snowpark session ({type(exc).__name__}). Using Mock mode.")
        return None


class CortexAgentArgs(BaseModel):
    query: str = Field(..., description="Natural language question regarding CRM deals, metrics, or meeting transcripts")


class CortexAgentTool:
    """
    Tool integrating Snowflake Cortex Analyst (Text-to-SQL over CRM metrics)
    and Cortex Search (semantic search over unstructured customer meeting notes).

    If no active credentials or Snowflake session are detected, runs in
    Fallback/Mock mode returning high-fidelity synthetic enterprise data.
    """
    name: str = "CortexAgent"
    description: str = "answers questions using sales conversations and metrics"
    args_schema: Type[CortexAgentArgs] = CortexAgentArgs

    def __init__(self, session: Optional[Any] = None):
        self._session: Optional[Any] = None
        self._root: Optional[Any] = None
        self._agent_service: Optional[Any] = None
        self._is_mock: bool = True

        if not _has_snowflake_credentials():
            logger.info("[CortexAgentTool] Fallback/Mock mode enabled (no Snowflake credentials).")
            return

        if session is None:
            session = create_snowpark_session()

        if session is not None:
            try:
                from snowflake.core import Root
                self._session = session
                self._root = Root(session)
                self._agent_service = self._root.cortex_agent_service
                self._is_mock = False
                logger.info("CortexAgentTool successfully connected to Snowflake.")
                return
            except Exception as e:
                logger.warning(f"Error initializing Snowflake Cortex service: {e}. Enabling Mock mode.")

        self._session = None
        logger.info("[CortexAgentTool] Fallback/Mock mode enabled (no active Snowflake connection).")

    def _build_request(self, query: str):
        from snowflake.core.cortex.lite_agent_service import AgentRunRequest
        return AgentRunRequest.from_dict({
            "model": "claude-3-5-sonnet",
            "tools": [
                {"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst1"}},
                {"tool_spec": {"type": "cortex_search", "name": "search1"}},
            ],
            "tool_resources": {
                "analyst1": {"semantic_model_file": SEMANTIC_MODEL_FILE},
                "search1": {
                    "name": CORTEX_SEARCH_SERVICE,
                    "max_results": 10,
                    "id_column": "conversation_id"
                }
            },
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": query}]}
            ]
        })

    @staticmethod
    def _event_delta(data: Any) -> Optional[Dict[str, Any]]:
        """Extracts the `delta` payload from a dict or JSON-string SSE event."""
        try:
            payload = data if isinstance(data, dict) else json.loads(data)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        delta = payload.get("delta")
        if delta is None and isinstance(payload.get("data"), dict):
            delta = payload["data"].get("delta")
        return delta if isinstance(delta, dict) else None

    @staticmethod
    def _tool_result_payloads(item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Returns the JSON payloads of a tool_results block, skipping malformed parts."""
        block = item.get("tool_results")
        contents = block.get("content") if isinstance(block, dict) else None
        if not isinstance(contents, list):
            return []
        return [
            result["json"] for result in contents
            if isinstance(result, dict) and result.get("type") == "json" and isinstance(result.get("json"), dict)
        ]

    def _consume_stream(self, stream) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Folds Cortex Agent SSE events into (text, sql, citations). Malformed
        events or content items are skipped so one bad chunk never discards
        the rest of a production answer.
        """
        text, sql, citations = "", "", []
        for evt in stream.events():
            delta = self._event_delta(getattr(evt, "data", None))
            contents = delta.get("content") if delta else None
            if not isinstance(contents, list):
                continue

            for item in contents:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text += str(item.get("text") or "")
                elif item.get("type") == "tool_results":
                    for payload in self._tool_result_payloads(item):
                        text += str(payload.get("text") or "")
                        sql = payload.get("sql", sql)
                        citations.extend({
                            "source_id": s.get("source_id"),
                            "doc_id": s.get("doc_id"),
                        } for s in payload.get("searchResults", []) if isinstance(s, dict))
        return text, sql, citations

    def _run_mock(self, query: str) -> Tuple[str, str, str, str]:
        """
        Synthetic generator for local development and automated testing.
        Simulates the combined responses of Cortex Analyst and Cortex Search.
        """
        lower_q = query.lower()

        # Case A: Questions on deals / CRM metrics (Cortex Analyst)
        if any(w in lower_q for w in ["deal", "top", "metrics", "client", "value", "largest", "revenue"]):
            sql = (
                "SELECT company_name, deal_value, deal_stage, product_line, expected_close_date\n"
                "FROM sales_intelligence.data.sales_metrics\n"
                "ORDER BY deal_value DESC LIMIT 3;"
            )
            results_str = (
                "COMPANY_NAME        DEAL_VALUE    DEAL_STAGE      PRODUCT_LINE           EXPECTED_CLOSE_DATE\n"
                "HealthTech Corp     $1,250,000    Negotiation     Enterprise Compliance  2026-10-15\n"
                "FinTech Global      $890,000      Proposal        Risk Analytics API     2026-11-01\n"
                "OmniRetail Systems  $640,000      Decision Maker  Inventory Forecasting  2026-12-05"
            )
            text = (
                "Cortex Analyst: Successfully retrieved the top 3 client deals ordered by deal value. "
                "The largest deal is HealthTech Corp ($1.25M), followed by FinTech Global ($890k) and OmniRetail Systems ($640k)."
            )
            citations = [
                {"source_id": "cortex_analyst", "doc_id": "sales_intelligence.data.sales_metrics"}
            ]

        # Case B: Questions on meeting notes / transcripts (Cortex Search)
        elif any(w in lower_q for w in ["note", "meeting", "theme", "transcript", "conversation", "regulatory", "value prop"]):
            sql = (
                "SELECT transcript_text\n"
                "FROM sales_intelligence.data.sales_conversations\n"
                "WHERE conversation_id IN ('conv_101', 'conv_102');"
            )
            results_str = (
                "=== Sales Conversation Transcripts ===\n"
                "[HealthTech Corp]: Key stakeholders highlighted growing compliance hurdles with state AI governance. "
                "They urgently need solutions that provide full auditing traceability.\n"
                "[FinTech Global]: Discussed open banking regulatory deadlines for Q1. Expressed hesitation due to legacy tech stack."
            )
            text = (
                "Cortex Search: Analyzed recent meeting notes. The primary recurring theme across customers is urgency around "
                "upcoming regulatory compliance and risk mitigation."
            )
            citations = [
                {"source_id": "conv_101", "doc_id": "meeting_transcript_healthtech"},
                {"source_id": "conv_102", "doc_id": "meeting_transcript_fintech"}
            ]

        # Case C: General query
        else:
            sql = "SELECT * FROM sales_intelligence.data.sales_metrics LIMIT 5;"
            results_str = "COMPANY_NAME: Generic Corp, VALUE: $500,000, STATUS: Active"
            text = f"Cortex Agent processed the query: '{query}'. Retrieved relevant CRM records and related conversation notes."
            citations = [{"source_id": "cortex_default", "doc_id": "doc_general"}]

        return text, str(citations), sql, results_str

    def _execute_sql(self, sql: str) -> str:
        if not _is_safe_readonly_sql(sql):
            logger.error("Security: rejected non read-only SQL generated by Cortex Analyst.")
            return "SQL execution blocked: only single read-only SELECT/WITH statements are allowed."
        try:
            self._session.sql(f"USE WAREHOUSE {configured_warehouse()}").collect()
            df = self._session.sql(sql.strip().rstrip(";")).to_pandas()
            return df.to_string(index=False)
        except Exception as e:
            return f"SQL execution error: {e}"

    def run(self, query: str, **kwargs) -> Tuple[str, str, str, str]:
        """
        Executes query against Cortex Agent Service or the Mock fallback engine.
        Returns the 4-tuple expected by downstream LangGraph nodes:
        (text, citations_str, sql, results_str)
        """
        if self._is_mock or self._session is None or self._agent_service is None:
            return self._run_mock(query)

        try:
            req = self._build_request(query)
            stream = self._agent_service.run(req)
            text, sql, citations = self._consume_stream(stream)
            results_str = self._execute_sql(sql) if sql else ""
            return text, str(citations), sql, results_str

        except Exception as exc:
            logger.error(f"Error in production CortexAgentTool: {exc}. Falling back to mock response.")
            return self._run_mock(query)

    def format_output(self, query: str) -> str:
        """Runs the query and renders the 4-tuple as labelled text for LLM agents."""
        text, citations, _sql, results_str = self.run(query)
        return f"{text}\n\nResults:\n{results_str}\n\nCitations: {citations}"

    def __call__(self, query: str, **kwargs) -> Tuple[str, str, str, str]:
        return self.run(query, **kwargs)


# Default initialization of CortexAgentTool (creates a Snowpark session when credentials exist)
cortex_agent_tool = CortexAgentTool(session=None)


@instrument_retrieval
def query_cortex_agent(query: str) -> str:
    """Answer questions about CRM customer deals, sales metrics, and customer meeting notes using Snowflake Cortex."""
    return cortex_agent_tool.format_output(query)


# =====================================================================
# 3. Python REPL Tool (Chart Generation & Disk Persistence)
# =====================================================================

class PythonREPLTool:
    """
    Sandboxed Python execution tool for generating and persisting charts to disk.
    Code is AST-validated (see sandbox.py) before running with restricted builtins.
    """
    name: str = "python_repl"
    description: str = "A sandboxed Python shell (matplotlib, seaborn, pandas, numpy) to plot and save charts."

    def run(self, code: str) -> str:
        """Validates and executes the provided Python code, returning captured stdout."""
        result = execute_sandboxed(code)
        if result.startswith("Python execution blocked"):
            logger.warning(f"[PythonREPLTool] {result}")
        return result

    def __call__(self, code: str) -> str:
        return self.run(code)


def get_python_repl_tool() -> PythonREPLTool:
    """
    Returns the sandboxed REPL tool. The unsandboxed langchain_experimental
    PythonREPLTool is intentionally never used (remote code execution risk).
    """
    return PythonREPLTool()


python_repl_tool = get_python_repl_tool()


def run_python_code(code: str) -> str:
    """Execute sandboxed Python (matplotlib/seaborn/pandas/numpy only) to build and save charts. Use print() to show output."""
    return python_repl_tool.run(code)
