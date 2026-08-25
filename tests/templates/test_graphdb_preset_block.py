"""Render test for the `services.graphdb` block in the shipped app presets.

The graph store is the one service whose declaration is *mandatory* when it is
deployed: a `graphdb` entry in `deployed_services` with no `services.graphdb`
block is refused before compose runs, because nothing else says which ports to
publish. A preset that emitted one half of that pair would therefore ship a
project that cannot come up — a failure every adopter hits on their first
`osprey up`, not a rare edge case.

So the pairing is pinned here for both presets that carry it, in both arms of
the `deploy_services` switch: on, and both halves are present; off (the attached
project, which connects to another project's stack), and neither is. The
last test closes the loop by running the preflight that does the refusing
against the rendered preset, so the guard fails if the block drifts into a shape
the deploy path would reject rather than merely into a shape that parses.

Values are pinned only where the schema module does *not* default them —
`path` and `ttl_path`, which no code fills in — plus the key set itself, which
is what pins the absent `password` key (the credential is minted into the
project `.env` as GRAPHDB_PASSWORD; one written here would be read by nobody).
Ports, image and the memory knobs are `graphdb_service.py` constants, already
covered by that module's own tests, and re-asserting them here would only
duplicate the pin.
"""

import pytest
import yaml

from osprey.cli.templates.manager import TemplateManager
from osprey.deployment.graphdb_service import (
    GRAPHDB_SERVICE_NAME,
    preflight_graphdb_config,
    resolve_graphdb_service_config,
)

#: Every app preset that ships the graph store. Both carry the identical block.
TEMPLATE_PATHS = [
    "apps/control_assistant/config.yml.j2",
    "apps/ariel_standalone/config.yml.j2",
]

#: The block's exact key set. Equality rather than membership, so a stray
#: `password:` — the one key that must never appear — fails the test.
EXPECTED_KEYS = {
    "path",
    "image",
    "port_host",
    "http_port_host",
    "ttl_path",
    "heap_initial_size",
    "heap_max_size",
    "pagecache_size",
    "query_timeout_s",
    "query_max_rows",
}


def _render(template_path: str, **context) -> dict:
    template = TemplateManager().jinja_env.get_template(template_path)
    return yaml.safe_load(template.render(**context))


@pytest.mark.parametrize("template_path", TEMPLATE_PATHS)
def test_block_and_deployed_entry_are_emitted_together(template_path: str):
    config = _render(template_path)
    assert GRAPHDB_SERVICE_NAME in config["services"]
    assert GRAPHDB_SERVICE_NAME in config["deployed_services"]


@pytest.mark.parametrize("template_path", TEMPLATE_PATHS)
def test_attached_project_emits_neither(template_path: str):
    """`deploy_services: false` scaffolds no services, graph store included."""
    config = _render(template_path, deploy_services=False)
    assert config["services"] == {}
    assert config["deployed_services"] == []


@pytest.mark.parametrize("template_path", TEMPLATE_PATHS)
def test_block_carries_exactly_the_expected_keys(template_path: str):
    block = _render(template_path)["services"][GRAPHDB_SERVICE_NAME]
    assert set(block) == EXPECTED_KEYS


#: The corpus each preset seeds, keyed by template. Both seed the demo machine
#: generated from the control-assistant channel database — the same devices
#: the agent's channel search returns, so graph answers and channel answers
#: describe one machine — and each ships its own copy in `data/`, so a build
#: profile that replaces the data tree takes the corpus with it rather than
#: quietly keeping a graph of a machine its channel database no longer knows.
EXPECTED_TTL_PATHS = {
    "apps/control_assistant/config.yml.j2": "./data/demo_machine.ttl",
    "apps/ariel_standalone/config.yml.j2": "./data/demo_machine.ttl",
}


@pytest.mark.parametrize("template_path", TEMPLATE_PATHS)
def test_paths_are_the_ones_nothing_defaults(template_path: str):
    """`path` and `ttl_path` have no fallback in code, so the template is the
    only thing that supplies them — `path` points the compose generator at the
    service fragment, `ttl_path` at the corpus that preset seeds from."""
    block = _render(template_path)["services"][GRAPHDB_SERVICE_NAME]
    assert block["path"] == "./services/graphdb"
    assert block["ttl_path"] == EXPECTED_TTL_PATHS[template_path]


@pytest.mark.parametrize("template_path", TEMPLATE_PATHS)
def test_rendered_preset_survives_the_deploy_preflight(template_path: str):
    """The refusal this pairing exists to avoid, run against the real render."""
    config = _render(template_path)
    preflight_graphdb_config(config)
    assert resolve_graphdb_service_config(config) is not None
