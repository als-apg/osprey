"""The `services.graphdb` block in the shipped presets that deploy the graph store.

The graph store is the one service whose declaration is *mandatory* when it is
deployed: a `graphdb` entry in `deployed_services` with no `services.graphdb`
block is refused before compose runs, because nothing else says which ports to
publish. A preset that emitted one half of that pair would therefore ship a
project that cannot come up — a failure every adopter hits on their first
`osprey up`, not a rare edge case.

The preset's ``config:`` block is where both halves come from — the framework
template renders no service at all — so the pairing is pinned on the presets'
resolved config, for both presets that carry it. The ports and the corpus are
the things a preset does not spell: `layout_port_fill` writes the ports at build
time from the deployment's port layout, and `graphdb_corpus_fill` writes the
corpus, the graph view the build renders from the facility file. So the
rendered block is the preset's block plus those fills, and that is what the
last test runs the deploy preflight against,
so the guard fails if the block drifts into a shape the deploy path would reject
rather than merely into a shape that parses.

An attached project (`deploy_services: false`) deploys neither half. That is a
build-side rule now — `_attached_service_overrides` drops the claimed
`services.*` blocks and empties `deployed_services` on the override path — and
is pinned by tests/cli/test_deployed_services_injection.py, not here.

Values are pinned only where the schema module does *not* default them —
`path`, which no code fills in, and `ttl_path`, which the build fills — plus the
key set itself, which
is what pins the absent `password` key (the credential is minted into the
project `.env` as GRAPHDB_PASSWORD; one written here would be read by nobody).
Image and the memory knobs are `graphdb_service.py` constants, already covered
by that module's own tests, and re-asserting them here would only duplicate the
pin.
"""

import pytest

from osprey.cli.build_injectors import graphdb_corpus_fill
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
    folded in, the layout's ports and the corpus filled under the service blocks
    it names."""
    profile, _profile_dir = resolve_build_profile(None, preset)
    config = {
        **layout_port_fill(profile.config, DEFAULT_PORT_BASE),
        **graphdb_corpus_fill(profile.config),
        **profile.config,
    }
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


#: The corpus each preset seeds: the graph view the build writes from the
#: facility file, so graph answers and channel answers describe one machine.
EXPECTED_TTL_PATH = "./data/graph/facility.ttl"


@pytest.mark.parametrize("preset", PRESETS)
def test_paths_are_the_ones_nothing_defaults(preset: str):
    """`path` has no fallback in code, so the preset supplies it and points the
    compose generator at the service fragment; `ttl_path` is the build's fill,
    the corpus that preset seeds from."""
    block = _rendered(preset)["services"][GRAPHDB_SERVICE_NAME]
    assert block["path"] == "./services/graphdb"
    assert block["ttl_path"] == EXPECTED_TTL_PATH


@pytest.mark.parametrize("preset", PRESETS)
def test_rendered_preset_survives_the_deploy_preflight(preset: str):
    """The refusal this pairing exists to avoid, run against the rendered shape."""
    config = _rendered(preset)
    preflight_graphdb_config(config)
    assert resolve_graphdb_service_config(config) is not None
