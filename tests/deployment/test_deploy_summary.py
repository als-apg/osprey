"""Tests for the post-deploy endpoint summary.

Every ``osprey deploy up`` ends with a summary of what is reachable where,
derived from the rendered compose files' published host ports — plus an
unconditional web-terminal line, so a project *without* a web tier says so
explicitly instead of silently binding nothing.
"""

from __future__ import annotations

import pytest

from osprey.deployment import deploy_summary


@pytest.fixture
def compose_file(tmp_path):
    path = tmp_path / "docker-compose.yml"
    path.write_text(
        """
services:
  event-dispatcher:
    ports:
      - "127.0.0.1:8020:8020"
  openobserve:
    ports:
      - "127.0.0.1:5080:5080"
  postgresql:
    ports:
      - "127.0.0.1:5432:5432"
""",
        encoding="utf-8",
    )
    return str(path)


def test_summary_lists_published_ports(compose_file):
    text = deploy_summary.format_endpoint_summary({"project_name": "demo"}, [compose_file])
    assert "demo" in text
    assert "event-dispatcher" in text
    assert "http://127.0.0.1:8020" in text
    assert "http://127.0.0.1:5080" in text
    # postgres is not an HTTP service — plain address, no scheme
    assert "127.0.0.1:5432" in text
    assert "http://127.0.0.1:5432" not in text


def test_summary_says_web_terminal_not_configured(compose_file):
    """The load-bearing line: absence must be an explicit negative signal."""
    text = deploy_summary.format_endpoint_summary({"project_name": "demo"}, [compose_file])
    assert "web terminal" in text
    assert "not configured" in text


def test_summary_shows_landing_url_when_web_enabled(compose_file):
    config = {
        "project_name": "demo",
        "modules": {"web_terminals": {"enabled": True, "nginx_port": 9080}},
    }
    text = deploy_summary.format_endpoint_summary(config, [compose_file])
    assert "http://127.0.0.1:9080" in text
    assert "not configured" not in text


def test_summary_handles_no_published_ports(tmp_path):
    empty = tmp_path / "docker-compose.yml"
    empty.write_text("services: {}\n", encoding="utf-8")
    text = deploy_summary.format_endpoint_summary({"project_name": "demo"}, [str(empty)])
    assert "web terminal" in text  # the unconditional line survives an empty stack


def test_log_endpoint_summary_never_raises(tmp_path):
    """Advisory output must not be able to fail a deploy that succeeded."""
    deploy_summary.log_endpoint_summary({"project_name": "demo"}, [str(tmp_path / "missing.yml")])
