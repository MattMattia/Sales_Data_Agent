"""Security invariants for the container build (Dockerfile, .dockerignore, docker-compose.yml)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    path = ROOT / name
    assert path.is_file(), f"{name} is missing"
    return path.read_text(encoding="utf-8")


def _instructions(dockerfile: str) -> list:
    """Dockerfile instructions with comments and blank lines removed, continuations joined."""
    joined = dockerfile.replace("\\\n", " ")
    return [line.strip() for line in joined.splitlines() if line.strip() and not line.strip().startswith("#")]


@pytest.fixture(scope="module")
def dockerfile_lines():
    return _instructions(_read("Dockerfile"))


def test_dockerfile_runs_as_unprivileged_user_with_healthcheck(dockerfile_lines):
    assert "USER appuser" in dockerfile_lines
    assert any(line.startswith("HEALTHCHECK") and "/_stcore/health" in line for line in dockerfile_lines)
    assert any(line.startswith("ENV") and "MPLBACKEND=Agg" in line for line in dockerfile_lines)
    assert any(line.startswith("EXPOSE") and "8501" in line for line in dockerfile_lines)
    # The runtime stage must switch away from root before the app starts.
    last_user = [line for line in dockerfile_lines if line.startswith("USER")][-1]
    assert last_user == "USER appuser"


def test_dockerfile_never_copies_env_files(dockerfile_lines):
    copies = [line for line in dockerfile_lines if line.startswith(("COPY", "ADD"))]
    assert copies
    assert all(".env" not in line for line in copies)


def test_dockerignore_excludes_secrets_and_local_state():
    entries = set(_read(".dockerignore").split())

    for required in (".git", ".venv", ".env", ".env.*", "*.sqlite", ".atl", ".claude", "charts/"):
        assert required in entries, f".dockerignore must exclude {required}"
    assert "!.env.template" in entries


def test_compose_persists_trulens_data_and_gates_dashboard_on_health():
    compose = _read("docker-compose.yml")

    assert "sales-data-agent:" in compose and "trulens-dashboard:" in compose
    assert "TRULENS_DATABASE_URL=sqlite:////app/data/default.sqlite" in compose
    assert compose.count("trulens-data:/app/data") == 2
    assert "condition: service_healthy" in compose
    assert "required: false" in compose
    assert "restart: unless-stopped" in compose
