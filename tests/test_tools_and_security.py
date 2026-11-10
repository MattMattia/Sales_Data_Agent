"""Tests for tools.py: mock integrations, SQL read-only guardrail, and Python sandbox."""

import os

import pytest

import tools
from tools import (
    CortexAgentTool,
    MockTavilySearch,
    PythonREPLTool,
    _is_safe_readonly_sql,
    get_python_repl_tool,
    query_cortex_agent,
    run_python_code,
    search_web,
)


# ---------------------------------------------------------------------
# MockTavilySearch
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "query, expected_fragment",
    [
        ("largest bank market cap", "JPMorgan Chase"),
        ("new regulatory compliance rules", "Regulatory Updates"),
        ("acme corp news", "Web Results for: acme corp news"),
    ],
)
def test_mock_tavily_search_branches(query, expected_fragment):
    results = MockTavilySearch().invoke({"query": query})

    assert len(results) == 1
    assert expected_fragment in results[0]["title"] + results[0]["content"]


def test_search_web_returns_plain_text_with_sources():
    output = search_web("regulatory updates")

    assert isinstance(output, str)
    assert "https://regulatorynews.example.com/2026-updates" in output


# ---------------------------------------------------------------------
# CortexAgentTool (mock mode, tool wrapper, session bootstrap)
# ---------------------------------------------------------------------

def test_cortex_tool_mock_run_keeps_four_tuple_contract():
    text, citations, sql, results = CortexAgentTool().run("top deals by value")

    assert "HealthTech Corp" in text
    assert "cortex_analyst" in citations
    assert sql.startswith("SELECT")
    assert "DEAL_VALUE" in results


def test_query_cortex_agent_returns_labelled_string():
    output = query_cortex_agent("meeting notes themes")

    assert isinstance(output, str)
    assert "Results:" in output
    assert "Citations:" in output


def test_cortex_tool_creates_snowpark_session_when_credentials_exist(monkeypatch):
    for key, value in {
        "SNOWFLAKE_ACCOUNT": "acct",
        "SNOWFLAKE_USER": "user",
        "SNOWFLAKE_PASSWORD": "secret",
    }.items():
        monkeypatch.setenv(key, value)
    created = []

    def fake_factory():
        created.append(True)
        return None  # connection failed -> must stay in mock mode

    monkeypatch.setattr(tools, "create_snowpark_session", fake_factory)

    tool = CortexAgentTool(session=None)

    assert created == [True]
    assert tool._is_mock is True


def test_cortex_tool_skips_session_factory_without_credentials(monkeypatch):
    monkeypatch.setattr(tools, "create_snowpark_session", lambda: pytest.fail("must not connect"))

    assert CortexAgentTool(session=None)._is_mock is True


def test_snowflake_connection_params_use_env_values(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct")
    monkeypatch.setenv("SNOWFLAKE_USER", "user")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "secret")
    monkeypatch.setenv("SNOWFLAKE_ROLE", "SALES_ANALYST_RO")
    monkeypatch.delenv("SNOWFLAKE_SCHEMA", raising=False)

    params = tools.snowflake_connection_params()

    assert params["account"] == "acct"
    assert params["role"] == "SALES_ANALYST_RO"
    assert "schema" not in params


@pytest.mark.parametrize(
    "env_value, expected",
    [
        (None, "SALES_INTELLIGENCE_WH"),
        ("ANALYTICS_WH", "ANALYTICS_WH"),
        ("WH; DROP TABLE x", "SALES_INTELLIGENCE_WH"),
    ],
)
def test_configured_warehouse_is_validated(monkeypatch, env_value, expected):
    if env_value is None:
        monkeypatch.delenv("SNOWFLAKE_WAREHOUSE", raising=False)
    else:
        monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", env_value)

    assert tools.configured_warehouse() == expected


class _FakeFrame:
    def to_string(self, index=False):
        return "ROWS"


class _FakeResult:
    def collect(self):
        return []

    def to_pandas(self):
        return _FakeFrame()


class _FakeSession:
    def __init__(self):
        self.executed = []

    def sql(self, statement):
        self.executed.append(statement)
        return _FakeResult()


def _production_tool(monkeypatch, generated_sql):
    tool = CortexAgentTool()
    session = _FakeSession()
    tool._session = session
    tool._agent_service = object()
    tool._is_mock = False
    monkeypatch.setattr(tool, "_build_request", lambda query: query)
    tool._agent_service = type("Svc", (), {"run": lambda self, req: req})()
    monkeypatch.setattr(tool, "_consume_stream", lambda stream: ("answer", generated_sql, []))
    return tool, session


def test_production_run_executes_readonly_sql_with_configured_warehouse(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "ANALYTICS_WH")
    tool, session = _production_tool(monkeypatch, "SELECT 1;")

    _, _, sql, results = tool.run("q")

    assert session.executed == ["USE WAREHOUSE ANALYTICS_WH", "SELECT 1"]
    assert results == "ROWS"
    assert sql == "SELECT 1;"


def test_production_run_blocks_destructive_sql(monkeypatch):
    tool, session = _production_tool(monkeypatch, "DROP TABLE sales_metrics")

    _, _, _, results = tool.run("q")

    assert session.executed == []
    assert results.startswith("SQL execution blocked")


# ---------------------------------------------------------------------
# _is_safe_readonly_sql (SEC-02)
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM sales_metrics",
        "select company_name from t order by deal_value desc limit 3;",
        "WITH top AS (SELECT * FROM t) SELECT * FROM top",
        "-- leading comment\nSELECT 1",
        "SELECT 'please delete me' AS note FROM t",
        "SELECT note FROM t WHERE note = 'a;b'",
        "SELECT REPLACE(company_name, 'Corp', '') FROM t",
    ],
)
def test_readonly_sql_is_allowed(sql):
    assert _is_safe_readonly_sql(sql) is True


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "DROP TABLE sales_metrics",
        "DELETE FROM sales_metrics",
        "UPDATE t SET x = 1",
        "INSERT INTO t VALUES (1)",
        "SELECT 1; DROP TABLE t",
        "SELECT 1; SELECT 2",
        "WITH x AS (SELECT 1) DELETE FROM t",
        "SELECT * FROM t /* comment */ ; TRUNCATE TABLE t",
        "SELECT '--'; DROP TABLE t",
        "/* SELECT */ GRANT ROLE ACCOUNTADMIN TO USER bob",
        "CALL system$cancel_all_queries()",
        "MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN DELETE",
        "SELECT 'unterminated; DROP TABLE t",
    ],
)
def test_mutating_or_multi_statement_sql_is_rejected(sql):
    assert _is_safe_readonly_sql(sql) is False


# ---------------------------------------------------------------------
# PythonREPLTool sandbox (SEC-01)
# ---------------------------------------------------------------------

def test_get_python_repl_tool_always_returns_sandboxed_tool():
    assert isinstance(get_python_repl_tool(), PythonREPLTool)
    assert isinstance(tools.python_repl_tool, PythonREPLTool)


def test_sandbox_runs_calculations_and_markdown_fences():
    output = PythonREPLTool().run("```python\nimport math\nvalues = [1, 2, 3]\nprint(sum(values) * math.pi)\n```")

    assert output.strip().startswith("18.849")


def test_sandbox_functions_see_module_level_names():
    output = run_python_code("factor = 3\ndef scale(x):\n    return x * factor\nprint(scale(2))")

    assert output.strip() == "6"


def test_sandbox_renders_matplotlib_chart_with_agg_backend(isolated_cwd):
    code = (
        "import matplotlib.pyplot as plt\n"
        "plt.bar(['A', 'B'], [1, 2])\n"
        "plt.savefig('chart.png')\n"
        "plt.close()\n"
        "print('saved')"
    )

    output = PythonREPLTool().run(code)

    import matplotlib
    assert output.strip() == "saved"
    assert (isolated_cwd / "chart.png").exists()
    assert matplotlib.get_backend().lower() == "agg"


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "import subprocess",
        "import sys",
        "from os import system",
        "import socket",
        "import importlib",
        "open('secrets.txt').read()",
        "__import__('os')",
        "eval('1 + 1')",
        "exec('x = 1')",
        "getattr(1, 'real')",
        "print(__builtins__)",
        "().__class__.__bases__[0].__subclasses__()",
        "import pandas as pd\npd.io.common.os.system('echo pwned')",
        "import matplotlib\nmatplotlib.cbook.os.getcwd()",
        "globals()",
    ],
)
def test_sandbox_blocks_dangerous_code(code):
    output = PythonREPLTool().run(code)

    assert output.startswith("Python execution blocked by security policy")


def test_sandbox_does_not_expose_os_or_sys_names():
    output = PythonREPLTool().run("print(os.getcwd())")

    assert "NameError" in output


def test_sandbox_reports_syntax_errors():
    output = PythonREPLTool().run("def broken(:\n    pass")

    assert output.startswith("Python execution blocked by security policy")
    assert "SyntaxError" in output
