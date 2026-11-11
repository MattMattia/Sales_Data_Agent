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


def test_trulens_dashboard_is_loopback_only_and_gets_no_secrets():
    yaml = pytest.importorskip("yaml")
    services = yaml.safe_load(_read("docker-compose.yml"))["services"]
    dashboard = services["trulens-dashboard"]

    assert dashboard["ports"] == ["127.0.0.1:${DASHBOARD_PORT:-8001}:8001"]
    assert "env_file" not in dashboard
    assert "env_file" in services["sales-data-agent"]


def test_ci_builds_the_image_and_waits_for_a_healthy_container():
    yaml = pytest.importorskip("yaml")
    job = yaml.safe_load(_read(".github/workflows/ci.yml"))["jobs"]["docker-image"]
    script = "\n".join(step.get("run", "") for step in job["steps"])

    assert "docker build" in script
    assert "{{.State.Health.Status}}" in script and "healthy" in script
    assert "id -u" in script and "10001" in script


def _published_ports(env_overrides):
    """Host ports Docker Compose would publish, resolved by the real compose CLI."""
    import json
    import os
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        pytest.skip("docker CLI not available")
    env = {**os.environ, **env_overrides}
    for name in ("APP_PORT", "DASHBOARD_PORT"):
        if name not in env_overrides:
            env.pop(name, None)
    result = subprocess.run(
        ["docker", "compose", "config", "--format", "json"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"docker compose config unavailable: {result.stderr.strip()[:200]}")
    services = json.loads(result.stdout)["services"]
    return {
        name: [(port.get("host_ip", ""), str(port["published"])) for port in service.get("ports", [])]
        for name, service in services.items()
    }


def test_compose_ports_default_to_8501_and_loopback_8001():
    ports = _published_ports({})

    assert ports["sales-data-agent"] == [("", "8501")]
    assert ports["trulens-dashboard"] == [("127.0.0.1", "8001")]


def test_compose_ports_can_be_overridden_to_avoid_conflicts():
    ports = _published_ports({"APP_PORT": "8502", "DASHBOARD_PORT": "8002"})

    assert ports["sales-data-agent"] == [("", "8502")]
    assert ports["trulens-dashboard"] == [("127.0.0.1", "8002")]
