"""
Shared pytest configuration.

Forces every external integration into Mock Fallback mode BEFORE any project
module is imported, so the suite never reaches OpenAI, Tavily, or Snowflake
and never reads the developer's local `.env` file.
"""

import os

os.environ["SALES_AGENT_SKIP_DOTENV"] = "1"
for _key in (
    "OPENAI_API_KEY",
    "TAVILY_API_KEY",
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
):
    os.environ.pop(_key, None)

import pytest  # noqa: E402


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch):
    """Run the test from an empty temporary directory so charts never land in the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
