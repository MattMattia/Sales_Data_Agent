# Audit remediation (Sales_Data_Agent)
Objective: fix every finding in CLAUDE_AUDIT_AND_REMEDIATION_PROMPT.md without breaking Mock Fallback or TruLens contracts.
Branch: fix/audit-remediation · Delivery: single branch, one commit per phase · Runner: `.venv/Scripts/python -m pytest -v`

## Specs
- S0 Paso 0: verify ECC agents/skills installed (no clone if present).
- S1 Fase 1: `.gitignore` (+ `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `*.py[cod]`, `.coverage`, `charts/`), `.env.template` `SNOWFLAKE_ROLE=SALES_ANALYST_RO`, `pytest>=8.0.0` in requirements.
- S2 Fase 2 (ARQ-01, CODE-05, ARQ-05.1): `load_dotenv()` safely; centralize fallback messages + `DEFAULT_ENABLED_AGENTS` in `state.py`; `chart_path`/`chart_notes` in `State`; `ExecutorDecision.goto` mentions `'chart_summarizer'`.
- S3 Fase 3 (ARQ-02, SEC-01, SEC-02, CODE-03.1, ARQ-04.1): Snowpark session auto-creation, configurable warehouse, `_is_safe_readonly_sql`, AST-sandboxed `PythonREPLTool` with restricted builtins and `matplotlib.use("Agg")`, clean `(query: str) -> str` tool wrappers, RETRIEVAL span instrumentation.
- S4 Fase 4 (CODE-01, CODE-02, ARQ-03, CODE-03.2, ARQ-05, SEC-03.2): `Command[Literal["__end__"]]`, `_parse_json_safely`, `PlanStep`/`ExecutorDecision` validation, replan `agent_query` fix, `[HumanMessage]` inputs, safe `chart_path` parsing, chart summary keeps research findings; `graph.py` initial state.
- S5 Fase 5 (CODE-04, UI-01): `evaluate_query_results` safe in mock and real TruLens; Streamlit agent multiselect, TruLens scorecard, per-message chart bytes, English spinner.
- S6 Fase 6 (TEST-01): `tests/test_state_and_prompts.py`, `tests/test_tools_and_security.py`, `tests/test_nodes_and_graph.py`; py_compile + pytest + `graph.py` + `evals.py`.

## Tasks
- T0 S0 inline — done (ECC agents/skills present in ~/.claude)
- T1 S1 inline — see Log
- T2 S2 inline
- T3 S3 inline
- T4 S4 inline
- T5 S5 inline
- T6 S6 inline (tests written first as RED)

## Log
- L1 (2026-11-10) "Lee el archivo CLAUDE_AUDIT_AND_REMEDIATION_PROMPT.md y ejecuta todas las fases desde el Paso 0 hasta la Fase 6"
- L2 Verification: `.gitignore` already existed (TEST-01.2 partially false). `.env.template` read/edit denied by the user's permission settings — SEC-03.1 left for the user.
- L3 Baseline (mock mode): `python graph.py` runs both flows; chart flow final answer = caption only.
