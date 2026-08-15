"""The network-axis macros are the single renderer the compose templates share.

Every assertion here is about a *fragment*: the macros are imported directly and
called on their own, so a break shows up as a wrong block rather than as a
compose file that fails to come up three templates later. Three properties carry
the design:

* **The unset axis reproduces today.** With no ``network:`` declared each macro
  emits exactly the literal block the service templates carry today — the no-op
  guarantee that lets the axis land without moving a single existing deployment.
* **Host mode is honoured mechanically.** ``network_mode: host`` replaces the
  network membership, the file-level ``networks:`` stanza disappears with it,
  and published ports are suppressed — none of which a template can opt out of,
  because none of it is a template's to reimplement.
* **One call idiom covers both uses.** Replacing a block a template already
  carries reproduces it byte-for-byte; adding a block that is usually absent
  costs not even a blank line when the axis is unset.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

#: The packaged service templates — the macro file's own directory, and the
#: directory the service compose templates sit beside.
SERVICES_DIR = Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates" / "services"

MACROS = "_network_axis.j2"

#: The service-level membership block the compose templates carry today.
TODAYS_NETWORKS_BLOCK = "\n    networks:\n      - osprey-network"

#: The file-level stanza those same templates end with today.
TODAYS_DECLARATION = "\nnetworks:\n  osprey-network:"


@pytest.fixture
def net():
    """The macro module, imported the way a template imports it.

    Rendering environments differ in where their loader is rooted (the project
    root for service templates, this directory for a standalone render), so the
    fixture roots the loader at the macro file's directory and imports by bare
    name — the axis behaviour under test is the same either way.
    """
    env = Environment(loader=FileSystemLoader(str(SERVICES_DIR)), autoescape=False)
    return env.get_template(MACROS).module


def render(source: str, **ctx) -> str:
    """Render a one-off template that imports the macros, for the placement tests.

    ``keep_trailing_newline`` is on so a fixture's final newline survives into
    the comparison. It is off in the real rendering environment, which drops
    that one newline from every rendered file — uniformly, on both sides of any
    comparison, so it can hide nothing the macros do.
    """
    env = Environment(
        loader=FileSystemLoader(str(SERVICES_DIR)),
        autoescape=False,
        keep_trailing_newline=True,
    )
    return env.from_string(source).render(**ctx)


#: Every spelling of "this service declares no network" a render context can
#: produce, all of which must render the default behaviour identically.
#: Undefined is handled separately — it is the only one that cannot be passed as
#: a Python value.
UNSET = [
    pytest.param({}, id="empty-mapping"),
    pytest.param({"port": 8020}, id="other-keys-only"),
    pytest.param(None, id="none"),
]


# ---------------------------------------------------------------------------
# network() — service-level attachment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("svc", UNSET)
def test_unset_axis_renders_todays_membership_block(net, svc):
    """A service that declares nothing joins the project network, as today."""
    assert net.network(svc) == TODAYS_NETWORKS_BLOCK


def test_undefined_service_renders_todays_membership_block(net):
    """A template may pass a `services.<name>` key that does not exist."""
    rendered = render(
        '{% import "_network_axis.j2" as net %}'
        "    restart: unless-stopped\n"
        "{{- net.network(services.absent) }}\n",
        services={},
    )
    assert rendered == "    restart: unless-stopped" + TODAYS_NETWORKS_BLOCK + "\n"


def test_explicit_bridge_is_the_default_shape(net):
    """`bridge` is the name of the behaviour the unset axis already had."""
    assert net.network({"network": "bridge"}) == TODAYS_NETWORKS_BLOCK


def test_host_mode_replaces_membership_with_the_host_namespace(net):
    assert net.network({"network": "host"}) == "\n    network_mode: host"


@pytest.mark.parametrize("indent", [0, 2, 4, 6])
def test_indent_places_both_shapes(net, indent):
    pad = " " * indent
    assert net.network({}, indent) == f"\n{pad}networks:\n{pad}  - osprey-network"
    assert net.network({"network": "host"}, indent) == f"\n{pad}network_mode: host"


def test_network_name_is_overridable(net):
    assert net.network({}, 4, "other-network") == "\n    networks:\n      - other-network"


def test_unrecognized_value_renders_the_default_shape(net):
    """Values are validated where the profile is parsed.

    Should one slip past that gate, the macro renders the network-joined shape
    rather than inventing a topology from a word it does not know.
    """
    assert net.network({"network": "somehow-not-a-mode"}) == TODAYS_NETWORKS_BLOCK


# ---------------------------------------------------------------------------
# network_declaration() — the file-level stanza
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("svc", UNSET)
def test_unset_axis_declares_the_network_as_today(net, svc):
    assert net.network_declaration(svc) == TODAYS_DECLARATION


def test_host_mode_declares_no_network(net):
    """No service in the file joins a network, so declaring one is noise."""
    assert net.network_declaration({"network": "host"}) == ""


def test_declaration_honours_indent_and_name(net):
    assert net.network_declaration({}, 2, "other-network") == ("\n  networks:\n    other-network:")


# ---------------------------------------------------------------------------
# ports() — published ports
# ---------------------------------------------------------------------------


def test_ports_are_published_double_quoted(net):
    """An unquoted `hh:mm`-shaped mapping is a YAML sexagesimal trap."""
    assert net.ports({}, ["127.0.0.1:8020:8020"]) == ('\n    ports:\n      - "127.0.0.1:8020:8020"')


def test_ports_emits_every_entry_in_order(net):
    assert net.ports({}, ["127.0.0.1:9190:9190", "127.0.0.1:9191:9191"]) == (
        '\n    ports:\n      - "127.0.0.1:9190:9190"\n      - "127.0.0.1:9191:9191"'
    )


def test_host_mode_suppresses_published_ports_entirely(net):
    """Under host networking the service's ports are simply the host's."""
    assert net.ports({"network": "host"}, ["127.0.0.1:8020:8020"]) == ""


@pytest.mark.parametrize("entries", [[], None], ids=["empty", "none"])
def test_no_entries_emits_no_key(net, entries):
    """An empty `ports:` key is invalid compose, so nothing is emitted."""
    assert net.ports({}, entries) == ""


def test_ports_honour_indent(net):
    assert net.ports({}, ["8020:8020"], 6) == '\n      ports:\n        - "8020:8020"'


# ---------------------------------------------------------------------------
# Placement — the whitespace contract in a template
# ---------------------------------------------------------------------------


def test_replacing_a_literal_block_is_byte_identical(net):
    """The call idiom reproduces a block a template already carries.

    This is the guarantee the whole axis rests on: adopting the macro must not
    move a byte of an existing render.
    """
    literal = (
        "    volumes:\n"
        "      - ./build/config.yml:/app/config.yml:ro\n"
        "    networks:\n"
        "      - osprey-network\n"
        "    restart: unless-stopped\n"
    )
    through_macro = render(
        '{% import "_network_axis.j2" as net %}'
        "    volumes:\n"
        "      - ./build/config.yml:/app/config.yml:ro\n"
        "{{- net.network(services.event_dispatcher) }}\n"
        "    restart: unless-stopped\n",
        services={"event_dispatcher": {}},
    )
    assert through_macro == literal


def test_a_suppressed_block_costs_not_even_a_blank_line(net):
    """Host mode drops `ports:` without leaving the line an `{% if %}` would."""
    rendered = render(
        '{% import "_network_axis.j2" as net %}'
        "    restart: unless-stopped\n"
        "{{- net.ports(services.event_dispatcher, published) }}\n"
        "    environment:\n",
        services={"event_dispatcher": {"network": "host"}},
        published=["127.0.0.1:8020:8020"],
    )
    assert rendered == "    restart: unless-stopped\n    environment:\n"


def test_file_level_stanza_closes_the_file_as_today(net):
    """The declaration is the last thing a compose template emits."""
    rendered = render(
        '{% import "_network_axis.j2" as net %}'
        "volumes:\n"
        "  dispatch_workspace_1:\n"
        "{{- net.network_declaration(services.dispatch_worker) }}\n",
        services={"dispatch_worker": {}},
    )
    assert rendered == "volumes:\n  dispatch_workspace_1:\nnetworks:\n  osprey-network:\n"


def test_host_mode_leaves_no_trailing_stanza(net):
    rendered = render(
        '{% import "_network_axis.j2" as net %}'
        "volumes:\n"
        "  dispatch_workspace_1:\n"
        "{{- net.network_declaration(services.dispatch_worker) }}\n",
        services={"dispatch_worker": {"network": "host"}},
    )
    assert rendered == "volumes:\n  dispatch_workspace_1:\n"
