"""Render tests for the graphdb service's container healthcheck.

The healthcheck runs INSIDE the ``neo4j:<ver>-community`` container, so the
probe binary has to be one that image actually carries. It ships ``wget`` (the
entrypoint itself fetches the n10s jar with it) and no ``curl`` — a probe that
shells curl fails on every attempt with "command not found", and the container
settles into ``unhealthy`` permanently while the store serves normally. That
false negative poisons everything that treats container health as ground
truth: ``podman ps``, smoke rosters, and any ``depends_on: service_healthy``
gate (issue #717).

These tests render the packaged compose template exactly as ``osprey up``
does and pin the probe to the binary the default image has, on both network
modes — the bridge default (container-internal 7474) and ``host`` (the
published port, since there is no port map left to translate).
"""

from __future__ import annotations

from importlib import resources

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

GRAPHDB_TEMPLATE = "services/graphdb/docker-compose.yml.j2"


def _render(**graphdb_block: object) -> dict:
    """Render the packaged graphdb compose template at its defaults.

    An Environment rooted at the packaged ``templates/`` directory, not a bare
    ``jinja2.Template``: the template imports the shared network-axis macros by
    a project-root-relative path, which only a loader can resolve.
    """
    templates_root = resources.files("osprey").joinpath("templates")
    env = Environment(loader=FileSystemLoader(str(templates_root)), autoescape=False)
    rendered = env.get_template(GRAPHDB_TEMPLATE).render(
        services={"graphdb": graphdb_block},
        deployment={},
        system={"timezone": "UTC"},
        osprey_labels={"project_name": "probe-test", "project_root": "/r/probe-test"},
    )
    return yaml.safe_load(rendered)


def _probe_command(doc: dict) -> str:
    test = doc["services"]["graphdb"]["healthcheck"]["test"]
    assert test[0] == "CMD-SHELL"
    return test[1]


def test_healthcheck_probes_with_wget():
    """The default render's probe shells a binary the default image carries."""
    command = _probe_command(_render())
    assert command.startswith("wget ")
    assert "http://127.0.0.1:7474/" in command


def test_healthcheck_never_shells_curl():
    """No curl anywhere: the neo4j community image does not ship it."""
    command = _probe_command(_render())
    assert "curl" not in command


@pytest.mark.parametrize("http_port_host", [7474, 17474])
def test_host_mode_probe_uses_wget_on_the_published_port(http_port_host: int):
    """Under ``network: host`` the probe dials the published HTTP port, still
    with the binary the image has."""
    command = _probe_command(_render(network="host", http_port_host=http_port_host))
    assert command.startswith("wget ")
    assert f"http://127.0.0.1:{http_port_host}/" in command
    assert "curl" not in command
