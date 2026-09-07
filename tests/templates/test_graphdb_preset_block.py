"""The `services.graphdb` block in the shipped presets that deploy the graph store.

The graph store is the one service whose declaration is *mandatory* when it is
deployed: a `graphdb` entry in `deployed_services` with no `services.graphdb`
block is refused before compose runs, because nothing else says which ports to
publish. A preset that emitted one half of that pair would therefore ship a
project that cannot come up — a failure every adopter hits on their first
`osprey up`, not a rare edge case.

The preset's ``config:`` block is where both halves come from — the framework
template renders no service at all — so the pairing is pinned on the presets'
resolved config, for both presets that carry it. The ports are the one thing a
preset does not spell: `layout_port_fill` writes them at build time from the
deployment's port layout, so the rendered block is the preset's block plus the
filled ports, and that is what the last test runs the deploy preflight against,
so the guard fails if the block drifts into a shape the deploy path would reject
rather than merely into a shape that parses.

An attached project (`deploy_services: false`) deploys neither half. That is a
build-side rule now — `_attached_service_overrides` drops the claimed
`services.*` blocks and empties `deployed_services` on the override path — and
is pinned by tests/cli/test_deployed_services_injection.py, not here.

Values are pinned only where the schema module does *not* default them —
`path` and `ttl_path`, which no code fills in — plus the key set itself, which
is what pins the absent `password` key (the credential is minted into the
project `.env` as GRAPHDB_PASSWORD; one written here would be read by nobody).
Image and the memory knobs are `graphdb_service.py` constants, already covered
by that module's own tests, and re-asserting them here would only duplicate the
pin.
"""

import pytest

from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_ports import layout_port_fill
from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey.deployment.graphdb_service import (
    GRAPHDB_SERVICE_NAME,
    preflight_graphdb_config,
    resolve_graphdb_service_config,
)
from osprey.port_layout import DEFAULT_PORT_BASE

#: Every preset that ships the graph store. Both carry the identical block.
PRESETS = ["control-assistant", "ariel-standalone"]

#: The block's exact key set as rendered. Equality rather than membership, so a
#: stray `password:` — the one key that must never appear — fails the test.
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


def _rendered(preset: str) -> dict:
    """The preset's resolved ``config:`` as a deployment renders it: dotted keys
    folded in, the layout's ports filled under the service blocks it names."""
    profile, _profile_dir = resolve_build_profile(None, preset)
    config = {**layout_port_fill(profile.config, DEFAULT_PORT_BASE), **profile.config}
    return _expand_dotted(config)


@pytest.mark.parametrize("preset", PRESETS)
def test_block_and_deployed_entry_are_emitted_together(preset: str):
    config = _rendered(preset)
    assert GRAPHDB_SERVICE_NAME in config["services"]
    assert GRAPHDB_SERVICE_NAME in config["deployed_services"]


@pytest.mark.parametrize("preset", PRESETS)
def test_block_carries_exactly_the_expected_keys(preset: str):
    block = _rendered(preset)["services"][GRAPHDB_SERVICE_NAME]
    assert set(block) == EXPECTED_KEYS


#: The corpus each preset seeds. Both seed the demo machine generated from the
#: control-assistant channel database — the same devices the agent's channel
#: search returns, so graph answers and channel answers describe one machine —
#: and each ships its own copy in its bundle's `data/`, so a build profile that
#: replaces the data tree takes the corpus with it rather than quietly keeping
#: a graph of a machine its channel database no longer knows.
EXPECTED_TTL_PATH = "./data/demo_machine.ttl"


@pytest.mark.parametrize("preset", PRESETS)
def test_paths_are_the_ones_nothing_defaults(preset: str):
    """`path` and `ttl_path` have no fallback in code, so the preset is the
    only thing that supplies them — `path` points the compose generator at the
    service fragment, `ttl_path` at the corpus that preset seeds from."""
    block = _rendered(preset)["services"][GRAPHDB_SERVICE_NAME]
    assert block["path"] == "./services/graphdb"
    assert block["ttl_path"] == EXPECTED_TTL_PATH


@pytest.mark.parametrize("preset", PRESETS)
def test_rendered_preset_survives_the_deploy_preflight(preset: str):
    """The refusal this pairing exists to avoid, run against the rendered shape."""
    config = _rendered(preset)
    preflight_graphdb_config(config)
    assert resolve_graphdb_service_config(config) is not None
