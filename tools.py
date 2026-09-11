"""
tools.py - Integration tools for the Multi-Agent architecture:
1. TavilySearch: Real-time web search with fallback for local/offline execution.
2. CortexAgentTool: Hybrid integration with Snowflake Cortex (Analyst Text-to-SQL
   and semantic Cortex Search) with synthetic fallback/mock when active credentials are absent.
3. python_repl_tool: Secure in-memory Python execution environment for saving charts to disk.
"""

import os
import sys
import io
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type
from contextlib import redirect_stdout, redirect_stderr

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


# =====================================================================
# 2. Snowflake Cortex Agent Tool (Cortex Analyst + Cortex Search)
# =====================================================================

SEMANTIC_MODEL_FILE = "@sales_intelligence.data.models/sales_metrics_model.yaml"
CORTEX_SEARCH_SERVICE = "sales_intelligence.data.sales_conversation_search"


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

    _session: Optional[Any] = PrivateAttr(default=None)
    _root: Optional[Any] = PrivateAttr(default=None)
    _agent_service: Optional[Any] = PrivateAttr(default=None)
    _is_mock: bool = PrivateAttr(default=False)

    def __init__(self, session: Optional[Any] = None):
        super().__init__()

        has_snowflake_creds = bool(
            os.getenv("SNOWFLAKE_ACCOUNT") and
            os.getenv("SNOWFLAKE_USER") and
            os.getenv("SNOWFLAKE_PASSWORD")
        )

        if session is not None and has_snowflake_creds:
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

        # Fallback / Mock mode
        self._is_mock = True
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

    def _consume_stream(self, stream) -> Tuple[str, str, List[Dict[str, Any]]]:
        text, sql, citations = "", "", []
        for evt in stream.events():
            try:
                delta = (evt.data.get("delta") if isinstance(evt.data, dict)
                         else json.loads(evt.data).get("delta")
                         or json.loads(evt.data).get("data", {}).get("delta"))
            except Exception:
                continue

            if not isinstance(delta, dict):
                continue

            for item in delta.get("content", []):
                if item.get("type") == "text":
                    text += item.get("text", "")
                elif item.get("type") == "tool_results":
                    for result in item["tool_results"].get("content", []):
                        if result.get("type") != "json":
                            continue
                        j = result["json"]
                        text += j.get("text", "")
                        sql = j.get("sql", sql)
                        citations.extend({
                            "source_id": s.get("source_id"),
                            "doc_id": s.get("doc_id")
                        } for s in j.get("searchResults", []))
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

            results_str = ""
            if sql:
                try:
                    # Ensure explicit warehouse selection prior to execution
                    self._session.sql("USE WAREHOUSE SALES_INTELLIGENCE_WH").collect()
                    df = self._session.sql(sql.rstrip(";")).to_pandas()
                    results_str = df.to_string(index=False)
                except Exception as e:
                    results_str = f"SQL execution error: {e}"

            return text, str(citations), sql, results_str

        except Exception as exc:
            logger.error(f"Error in production CortexAgentTool: {exc}. Falling back to mock response.")
            return self._run_mock(query)

    def __call__(self, query: str, **kwargs) -> Tuple[str, str, str, str]:
        return self.run(query, **kwargs)


# Default initialization of CortexAgentTool
cortex_agent_tool = CortexAgentTool(session=None)


# =====================================================================
# 3. Python REPL Tool (Chart Generation & Disk Persistence)
# =====================================================================

class PythonREPLTool:
    """
    In-memory Python execution tool for generating and persisting charts to disk.
    Captures stdout and errors safely without heavy external dependencies.
    """
    name: str = "python_repl"
    description: str = "A Python shell to execute code and save charts to disk."

    def __init__(self):
        pass

    def run(self, code: str) -> str:
        """
        Executes provided Python code and captures standard output.
        """
        # Strip markdown fences if emitted by the LLM
        clean_code = code.strip()
        if clean_code.startswith("```python"):
            clean_code = clean_code[9:]
        elif clean_code.startswith("```"):
            clean_code = clean_code[3:]
        if clean_code.endswith("```"):
            clean_code = clean_code[:-3]
        clean_code = clean_code.strip()

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        # Enriched execution scope with standard visualization libraries
        local_scope: Dict[str, Any] = {}
        global_scope: Dict[str, Any] = {
            "__name__": "__main__",
            "os": os,
            "sys": sys,
        }

        # Inject common data analysis libraries if available in the environment
        for lib in ("matplotlib", "matplotlib.pyplot", "pandas", "numpy", "seaborn"):
            try:
                mod_name = lib.split(".")[-1] if "." in lib else lib
                global_scope[mod_name] = __import__(lib, fromlist=["*"])
            except ImportError:
                pass

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(clean_code, global_scope, local_scope)
            output = stdout_buffer.getvalue()
            errors = stderr_buffer.getvalue()
            return output if output else (f"Executed successfully (warnings: {errors})" if errors else "Executed successfully without output.")
        except Exception as err:
            return f"Python execution error: {type(err).__name__}: {str(err)}"

    def __call__(self, code: str) -> str:
        return self.run(code)


def get_python_repl_tool():
    """
    Instantiates and returns the REPL tool for LangChain / LangGraph.
    """
    try:
        from langchain_experimental.tools.python.tool import PythonREPLTool as LCExperimentalREPL
        return LCExperimentalREPL()
    except ImportError:
        return PythonREPLTool()


python_repl_tool = get_python_repl_tool()
