"""Tests for the post-deploy endpoint summary.

Every ``osprey up`` ends with a summary of what is reachable where,
derived from the rendered compose files' published host ports — plus an
unconditional web-terminal line, so a project *without* a web tier says so
explicitly instead of silently binding nothing.

The block is what orders it: the heading names the thousand ports the
deployment reserved, and the rows are grouped into the layout's own tiers in
ascending offset order rather than sorted by service name.

The summary is the deploy's own output, so it is *printed* rather than logged:
an INFO record is not rendered on a normal run, and a fact only a
``--verbose`` run shows is a fact the deploy did not report. The needle tests
below hold both halves of that claim side by side — the block is in the default
view, its logged form no longer paints there, and the record is still emitted
for file and aggregation sinks.
"""

from __future__ import annotations

import io
import logging
import re

import pytest
import yaml
from rich.console import Console
from rich.logging import RichHandler

from osprey.cli.altitude import install_gate, lift_gate
from osprey.cli.phase_reporter import PhaseReporter, install_reporter
from osprey.cli.styles import osprey_theme
from osprey.deployment import deploy_summary
from osprey.port_layout import (
    CA_DEFAULT_PORT,
    DEFAULT_PORT_BASE,
    PORT_BASE_CONFIG_KEY,
    SLOTS_BY_NAME,
    block_range,
    layout_ports,
)

#: Anything Rich writes that is not text: styles, cursor moves, erases.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

#: The other escape family, and the one a URL arrives wrapped in: an OSC string
#: (``\x1b]8;;<url>\x1b\\``) is how Rich emits a terminal hyperlink. Stripped
#: alongside the CSI codes, since neither is text.
_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

#: The heading the ``demo`` fixtures print under, taken from the one function
#: that composes it: the tests below are about what reaches the terminal, not
#: about the wording, which ``test_the_heading_names_the_whole_block…`` pins.
_TITLE = deploy_summary.summary_title({"project_name": "demo"})


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
    """A web tier with no port keys at all still names its front door.

    The gateway slot of this deployment's own block, resolved rather than read:
    the stanza names an override, never a requirement.
    """
    config = {"project_name": "demo", "modules": {"web_terminals": {"enabled": True}}}
    text = deploy_summary.format_endpoint_summary(config, [compose_file])
    assert f"http://127.0.0.1:{layout_ports(DEFAULT_PORT_BASE)['nginx']}" in text
    assert "not configured" not in text


def test_the_heading_names_the_whole_block_this_deployment_reserved():
    """A port an operator does not recognise came from one knob, and it is named.

    The rows below the heading are a handful of numbers out of a thousand
    consecutive ones; without the range and the key, nothing on the surface says
    where the next number would come from.

    The key is spelled in FULL, matching the port preflight's own block line
    ("This deployment's block is ports <first>-<last> (deployment.port_base
    <base>)."). One key, one spelling, across every surface that names it.
    """
    first, last = block_range(DEFAULT_PORT_BASE)

    title = deploy_summary.summary_title({"project_name": "demo"})

    assert title == (
        f"Service endpoints (demo) — ports {first}-{last} ({PORT_BASE_CONFIG_KEY} {first}):"
    )
    assert PORT_BASE_CONFIG_KEY == "deployment.port_base"


def test_a_moved_block_is_described_in_its_own_numbers():
    """Never the framework default: the base this config resolved, and no other."""
    first, last = block_range(20000)

    title = deploy_summary.summary_title(
        {"project_name": "demo", "deployment": {"port_base": 20000}}
    )

    assert f"ports {first}-{last}" in title
    assert f"{PORT_BASE_CONFIG_KEY} {first}" in title
    assert str(DEFAULT_PORT_BASE) not in title


def test_a_base_no_block_can_start_at_costs_the_range_not_the_summary():
    """Refusing a base is the preflight's job. A summary that cannot name the
    block still names the deployment, rather than failing the deploy it closes."""
    title = deploy_summary.summary_title({"project_name": "demo", "deployment": {"port_base": 80}})

    assert title == "Service endpoints (demo):"


def test_summary_handles_no_published_ports(tmp_path):
    empty = tmp_path / "docker-compose.yml"
    empty.write_text("services: {}\n", encoding="utf-8")
    text = deploy_summary.format_endpoint_summary({"project_name": "demo"}, [str(empty)])
    assert "web terminal" in text  # the unconditional line survives an empty stack


def test_log_endpoint_summary_never_raises(tmp_path):
    """Advisory output must not be able to fail a deploy that succeeded."""
    deploy_summary.log_endpoint_summary({"project_name": "demo"}, [str(tmp_path / "missing.yml")])


# ---------------------------------------------------------------------------
# The promoted summary: printed, not logged
# ---------------------------------------------------------------------------


class _RecordingReporter(PhaseReporter):
    """A reporter whose console is a buffer, so printed output is readable.

    Subclasses the real reporter rather than standing in for it: what is being
    pinned is that the summary goes through the renderer's own ``out()`` seam,
    which a look-alike would not prove.
    """

    def __init__(self, console: Console) -> None:
        super().__init__(color=False)
        self._console = console

    def out(self) -> Console:
        return self._console


class _RecordSink(logging.Handler):
    """Every record the loggers emitted, captured before any handler filter."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _Probe:
    """What the terminal showed, and what the loggers emitted, side by side.

    ``printed`` is the echo path — where a promoted fact now lives.
    ``rendered`` is what the root ``RichHandler`` painted, after the altitude
    gate has had its say. ``messages`` is the raw record stream, which the gate
    never touches: it is a handler filter, so a record it drops is still
    emitted and still reaches every sink.
    """

    def __init__(
        self,
        printed: io.StringIO,
        painted: io.StringIO,
        sink: _RecordSink,
        handler: RichHandler,
    ) -> None:
        self._printed = printed
        self._painted = painted
        self._sink = sink
        self.handler = handler

    @property
    def printed(self) -> str:
        """The verb's own output, with escape sequences stripped."""
        return _OSC.sub("", _ANSI.sub("", self._printed.getvalue()))

    @property
    def rendered(self) -> str:
        """What the logging handler painted, with escape sequences stripped."""
        return _OSC.sub("", _ANSI.sub("", self._painted.getvalue()))

    @property
    def messages(self) -> list[str]:
        """The formatted message of every captured record."""
        return [record.getMessage() for record in self._sink.records]


@pytest.fixture
def probe(restore_root_logging):
    """Capture the printed stream and the painted log stream of one call.

    A local instrument rather than ``tests/cli``'s ``terminal_probe`` fixture,
    which that package's ``conftest`` does not share with this one. Both halves
    are built the same way: a themed terminal console writing into a buffer, the
    altitude gate a normal run installs, and a record sink underneath the gate.
    """
    printed = io.StringIO()
    painted = io.StringIO()
    console_kwargs = {
        "theme": osprey_theme,
        "force_terminal": True,
        "width": 100,
        "color_system": "truecolor",
    }

    previous = install_reporter(_RecordingReporter(Console(file=printed, **console_kwargs)))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = RichHandler(
        console=Console(file=painted, **console_kwargs),
        markup=True,
        show_path=False,
        show_time=False,
        show_level=True,
    )
    install_gate(handler)
    sink = _RecordSink()
    root.addHandler(handler)
    root.addHandler(sink)
    try:
        yield _Probe(printed, painted, sink, handler)
    finally:
        root.removeHandler(handler)
        root.removeHandler(sink)
        install_reporter(previous)


def test_the_endpoint_summary_is_in_the_default_view(probe, compose_file):
    """The needle: printed on a normal run, no longer painted by the logger."""
    deploy_summary.log_endpoint_summary({"project_name": "demo"}, [compose_file])
    # ERROR, not WARNING: the gate keeps WARNING off the terminal while a
    # reporter is installed, and this fixture installs one.
    logging.getLogger("deployment.summary").error("armed witness")

    # Echoed: what the operator ran the verb to find out.
    assert _TITLE in probe.printed
    assert "event-dispatcher" in probe.printed
    assert "http://127.0.0.1:8020" in probe.printed

    # The armed witness proves the painting console is live and the gate is the
    # only reason the summary is missing from it — an absence assertion against
    # a console nobody ever wrote to passes for the wrong reason.
    assert "armed witness" in probe.rendered
    assert "Service endpoints" not in probe.rendered
    assert "http://127.0.0.1:8020" not in probe.rendered

    # Nothing was lost: the record still carries the whole block for sinks.
    assert any(_TITLE in message for message in probe.messages)
    assert any("http://127.0.0.1:8020" in message for message in probe.messages)


def test_the_logged_block_still_reaches_a_verbose_transcript(probe, compose_file):
    """Lifting the gate is what ``-v`` does, and the record paints again there.

    The record half is kept at its old level deliberately: a file or
    aggregation sink gets the whole summary in one record, and a transcript run
    is the one place a reader asked for it twice.
    """
    lift_gate(probe.handler)

    deploy_summary.log_endpoint_summary({"project_name": "demo"}, [compose_file])

    assert _TITLE in probe.printed  # echo class is unconditional
    assert _TITLE in probe.rendered


def test_printed_summary_keeps_the_entries_and_their_order(probe, compose_file):
    """Every entry reaches the terminal, in the order the derivation put it in."""
    config = {"project_name": "demo", "modules": {"web_terminals": {"enabled": True}}}
    entries = deploy_summary.endpoint_entries(config, [compose_file])

    deploy_summary.log_endpoint_summary(config, [compose_file])

    rows = [line.rstrip() for line in probe.printed.splitlines() if line.startswith("    ")]
    assert len(rows) == len(entries)
    for row, (_tier, service, address) in zip(rows, entries, strict=True):
        assert row.startswith(f"    {service}")
        assert row.endswith(address)


def test_the_printed_block_is_sectioned_by_tier_in_layout_order(probe, compose_file):
    """The shape of the block, not an alphabet.

    The fixture publishes one service in each of three tiers, plus the landing
    page and the panel bands, and they print gateway-first in ascending offset
    order — which is nothing like the alphabetical order the same service names
    would have taken.
    """
    config = {"project_name": "demo", "modules": {"web_terminals": {"enabled": True}}}

    deploy_summary.log_endpoint_summary(config, [compose_file])

    printed = probe.printed
    # A heading is indented once, a service under it twice, so the two are told
    # apart by their indent rather than by matching the tier names again.
    headings = [
        line.strip()
        for line in printed.splitlines()
        if line.startswith("  ") and not line.startswith("   ")
    ]
    assert headings == ["gateway", "dispatch", "services", "panels", "stores"]
    for tier, service in (
        ("gateway", "web terminal"),
        ("dispatch", "event-dispatcher"),
        ("services", "openobserve"),
        ("panels", "artifact"),
        ("stores", "postgresql"),
    ):
        assert printed.index(tier) < printed.index(service)


def test_printed_summary_uses_the_section_shape(probe, compose_file):
    """``output.section`` vocabulary: a title, then one indented aligned row."""
    deploy_summary.log_endpoint_summary({"project_name": "demo"}, [compose_file])

    lines = [line.rstrip() for line in probe.printed.splitlines() if line.strip()]
    assert lines[0] == _TITLE
    rows = lines[1:]
    assert all(row.startswith("  ") for row in rows)
    # Labels are padded to one width, so the addresses line up as a column.
    starts = {row.index("http") for row in rows if "http" in row}
    assert len(starts) == 1


def test_a_tier_with_nothing_in_it_is_not_printed(probe, compose_file):
    """A deployment that runs no panels has no panel section.

    An empty heading would read as a section whose contents failed to render,
    which is the opposite of what it would mean.
    """
    deploy_summary.log_endpoint_summary({"project_name": "demo"}, [compose_file])

    assert "panels" not in probe.printed
    assert "facility" not in probe.printed


def test_a_failed_summary_prints_nothing_and_is_recorded(probe, monkeypatch):
    """A summary that cannot be derived stays silent on the operator's terminal."""

    def boom(config, compose_files):
        raise RuntimeError("no compose files")

    monkeypatch.setattr(deploy_summary, "endpoint_entries", boom)
    # The reason lives at debug grade, so the run has to be one that emits it.
    logging.getLogger().setLevel(logging.DEBUG)
    deploy_summary.log_endpoint_summary({"project_name": "demo"}, [])

    assert "Service endpoints" not in probe.printed
    assert any("Endpoint summary skipped" in message for message in probe.messages)


# ---------------------------------------------------------------------------
# Port-aware HTTP predicate: one service, two ports, two vocabularies
# ---------------------------------------------------------------------------


def _graph_row(entries: list[tuple[str, str, str]]) -> str:
    """The one ``graphdb`` address, both of its ports included.

    A service is one row however many ports it answers at, so this asserts there
    is exactly one — two rows carrying one name would invite a reader to treat
    the store as two services with a spare port between them.
    """
    matches = [address for _tier, service, address in entries if service == "graphdb"]
    assert len(matches) == 1, f"expected one graphdb entry, got {matches}"
    return matches[0]


def _address_for(entries: list[tuple[str, str, str]], host_port: int) -> str:
    """The part of the merged ``graphdb`` row that describes ``host_port``.

    Asserts on that one address rather than on a substring of the whole block,
    so a URL leaking onto the bolt address cannot hide behind the Browser one
    matching.
    """
    parts = _graph_row(entries).removesuffix("  (host network)").split(" · ")
    matches = [part for part in parts if f":{host_port}" in part]
    assert len(matches) == 1, f"expected one graphdb address on {host_port}, got {matches}"
    return matches[0]


@pytest.fixture
def graphdb_compose_file(tmp_path):
    """A graph store publishing both of its ports the way the template does."""
    path = tmp_path / "docker-compose.graphdb.yml"
    path.write_text(
        """
services:
  graphdb:
    ports:
      - "127.0.0.1:7687:7687"
      - "127.0.0.1:7474:7474"
""",
        encoding="utf-8",
    )
    return str(path)


def test_only_the_browser_port_of_the_graph_store_is_a_url(graphdb_compose_file):
    """The needle: one service, two ports, and only 7474 speaks HTTP.

    Bolt is a binary protocol — a ``http://`` prefix on 7687 would hand the
    operator a link that cannot open, which is worse than no link at all.
    """
    entries = deploy_summary.endpoint_entries({"project_name": "demo"}, [graphdb_compose_file])

    assert _address_for(entries, 7474) == "browser http://127.0.0.1:7474"
    assert _address_for(entries, 7687) == "bolt 127.0.0.1:7687"


def test_the_graph_store_is_one_row_that_says_which_port_is_which(graphdb_compose_file):
    """One service, one row — and each address prefixed with what it is.

    "127.0.0.1:7687 · 127.0.0.1:7474" says nothing about which of the two a
    browser can open, and bolt first is the layout's own order: the bolt slot
    sits one offset below the HTTP one.
    """
    entries = deploy_summary.endpoint_entries({"project_name": "demo"}, [graphdb_compose_file])

    assert _graph_row(entries) == "bolt 127.0.0.1:7687 · browser http://127.0.0.1:7474"


def test_moved_graph_store_ports_stay_port_aware(tmp_path):
    """The decision follows the CONTAINER port, not the host port it was moved to.

    A project that published the graph store somewhere else still gets its
    Browser link and its bare bolt address — keying the predicate on the host
    port would have silently swapped the two the moment either moved.
    """
    path = tmp_path / "docker-compose.moved.yml"
    path.write_text(
        """
services:
  graphdb:
    ports:
      - "127.0.0.1:17687:7687"
      - "127.0.0.1:17474:7474"
""",
        encoding="utf-8",
    )
    entries = deploy_summary.endpoint_entries({"project_name": "demo"}, [str(path)])

    assert _address_for(entries, 17474) == "browser http://127.0.0.1:17474"
    assert _address_for(entries, 17687) == "bolt 127.0.0.1:17687"


def test_host_network_graph_store_renders_the_same_way(tmp_path):
    """A derived binding and a published one describe the same endpoint.

    Both carry the container port the image fixes, so the summary cannot say
    "Browser" about one deployment's 7474 and "bolt" about another's.
    """
    empty = tmp_path / "docker-compose.yml"
    empty.write_text("services: {}\n", encoding="utf-8")
    config = {
        "project_name": "demo",
        "services": {"graphdb": {"network": "host", "port_host": 7687, "http_port_host": 7474}},
    }

    entries = deploy_summary.endpoint_entries(config, [str(empty)])

    assert _address_for(entries, 7474) == "browser http://127.0.0.1:7474"
    assert not _address_for(entries, 7687).startswith("http://")
    # The host-network annotation describes the service, not one of its ports,
    # so the merged row carries it once at the end rather than on each address.
    assert _graph_row(entries).endswith("  (host network)")
    assert _graph_row(entries).count("(host network)") == 1


def test_host_network_and_published_graph_stores_agree_on_moved_ports(tmp_path):
    """The two sources render identically for the same moved ports."""
    published = tmp_path / "docker-compose.published.yml"
    published.write_text(
        """
services:
  graphdb:
    ports:
      - "127.0.0.1:27687:7687"
      - "127.0.0.1:27474:7474"
""",
        encoding="utf-8",
    )
    empty = tmp_path / "docker-compose.empty.yml"
    empty.write_text("services: {}\n", encoding="utf-8")
    derived_config = {
        "project_name": "demo",
        "services": {"graphdb": {"network": "host", "port_host": 27687, "http_port_host": 27474}},
    }

    parsed = deploy_summary.endpoint_entries({"project_name": "demo"}, [str(published)])
    derived = deploy_summary.endpoint_entries(derived_config, [str(empty)])

    # Only the "(host network)" annotation may differ between the two.
    assert _graph_row(derived).removesuffix("  (host network)") == _graph_row(parsed)


def test_single_port_http_services_are_unchanged_beside_the_graph_store(tmp_path):
    """Every existing member still renders as a URL on whatever port it published."""
    path = tmp_path / "docker-compose.mixed.yml"
    path.write_text(
        """
services:
  graphdb:
    ports:
      - "127.0.0.1:7687:7687"
      - "127.0.0.1:7474:7474"
  openobserve:
    ports:
      - "127.0.0.1:5080:5080"
  tiled:
    ports:
      - "127.0.0.1:8000:8000"
  postgresql:
    ports:
      - "127.0.0.1:5432:5432"
""",
        encoding="utf-8",
    )
    entries = {
        service: address
        for _tier, service, address in deploy_summary.endpoint_entries(
            {"project_name": "demo"}, [str(path)]
        )
    }

    assert entries["openobserve"] == "http://127.0.0.1:5080"
    assert entries["tiled"] == "http://127.0.0.1:8000"
    assert entries["postgresql"] == "127.0.0.1:5432"


def test_an_unrecognised_graph_store_port_stays_a_bare_address(tmp_path):
    """A port the image does not fix gets no link it cannot honour."""
    path = tmp_path / "docker-compose.extra.yml"
    path.write_text(
        """
services:
  graphdb:
    ports:
      - "127.0.0.1:2004:2004"
""",
        encoding="utf-8",
    )
    entries = deploy_summary.endpoint_entries({"project_name": "demo"}, [str(path)])

    assert _address_for(entries, 2004) == "127.0.0.1:2004"


# ---------------------------------------------------------------------------
# The block: tiers, order, and the numbers a whole deployment lands on
# ---------------------------------------------------------------------------


def test_the_display_tiers_agree_with_the_preflights_slot_table():
    """One correspondence between compose services and layout slots, not two.

    ``host_ports`` maps the same services to the slots a binding may be sitting
    on, so it can tell "moves with the block" from "was hand-placed". This
    module maps them to the slot a row is *shown* under. The two answer
    different questions off the same fact, and this is what stops them drifting:
    every service the preflight places is placed the same way here.
    """
    from osprey.deployment.host_ports import _SERVICE_LAYOUT_SLOTS

    for service, slots in _SERVICE_LAYOUT_SLOTS.items():
        assert deploy_summary._SERVICE_SLOTS[service] == slots[0], service

    # The display-only entries, and only those: a service the preflight does not
    # place still has to be shown under some tier.
    extra = set(deploy_summary._SERVICE_SLOTS) - set(_SERVICE_LAYOUT_SLOTS)
    assert extra == {"nginx", "auth", "virtual-accelerator"}


def test_every_slot_a_service_names_is_a_real_layout_slot():
    """A typo here is otherwise a service silently filed under the facility band."""
    for service, slot in deploy_summary._SERVICE_SLOTS.items():
        assert slot in SLOTS_BY_NAME, service


def test_a_service_the_layout_does_not_name_is_shown_under_the_facility_band(tmp_path):
    """A deployment's own services are what the facility band is for.

    Dropping the row would have made the summary silently short of the very
    service the facility added; inventing a section for it would have said the
    framework placed it.
    """
    path = tmp_path / "docker-compose.facility.yml"
    path.write_text(
        'services:\n  beamline-gateway:\n    ports:\n      - "127.0.0.1:10900:80"\n',
        encoding="utf-8",
    )

    entries = deploy_summary.endpoint_entries({"project_name": "demo"}, [str(path)])

    assert ("facility", "beamline-gateway", "127.0.0.1:10900") in entries


def _control_assistant_compose(path, base):
    """A compose file publishing a control-assistant-shaped stack at ``base``.

    Every port comes from :func:`~osprey.port_layout.layout_ports`, so moving a
    slot moves this fixture with it rather than leaving a stale literal that
    still passes. The virtual accelerator is the one exception the layout
    itself carves out: instance 1 serves Channel Access on 5064, the one host
    port ``port_base`` cannot move.
    """
    ports = layout_ports(base)
    path.write_text(
        f"""
services:
  nginx:
    ports:
      - "127.0.0.1:{ports["nginx"]}:80"
  event-dispatcher:
    ports:
      - "127.0.0.1:{ports["dispatcher"]}:8020"
  openobserve:
    ports:
      - "127.0.0.1:{ports["openobserve"]}:5080"
  qmd:
    ports:
      - "127.0.0.1:{ports["qmd"]}:8000"
  virtual-accelerator:
    ports:
      - "127.0.0.1:{CA_DEFAULT_PORT}:{CA_DEFAULT_PORT}"
  postgresql:
    ports:
      - "127.0.0.1:{ports["postgres"]}:5432"
  graphdb:
    ports:
      - "127.0.0.1:{ports["graphdb_bolt"]}:7687"
      - "127.0.0.1:{ports["graphdb_http"]}:7474"
""",
        encoding="utf-8",
    )
    return str(path)


@pytest.mark.parametrize("base", [DEFAULT_PORT_BASE, 20000])
def test_a_whole_deployment_prints_as_its_own_block(tmp_path, base):
    """The acceptance shape of ``osprey init … --up``, at two different bases.

    Tier-grouped in layout order, headed by the block this deployment reserved,
    and every framework port inside that block — except the one the layout says
    is outside it.
    """
    compose = _control_assistant_compose(tmp_path / f"docker-compose.{base}.yml", base)
    config = {
        "project_name": "demo",
        "deployment": {"port_base": base},
        "modules": {"web_terminals": {"enabled": True}},
    }

    text = deploy_summary.format_endpoint_summary(config, [compose])

    first, last = block_range(base)
    assert text.splitlines()[0] == (
        f"Service endpoints (demo) — ports {first}-{last} ({PORT_BASE_CONFIG_KEY} {first}):"
    )
    headings = [
        line.strip() for line in text.splitlines() if line.startswith("  ") and line[2] != " "
    ]
    assert headings == ["gateway", "dispatch", "services", "panels", "stores"]

    entries = deploy_summary.endpoint_entries(config, [compose])
    # Every number that follows a colon or a dash: the port of an address, and
    # BOTH ends of a panel band's range. Keying the sweep on a literal
    # 127.0.0.1 would silently skip an address on any other interface, and a
    # port that escaped the block is exactly what this is looking for.
    published = [
        int(port)
        for _tier, _service, address in entries
        for port in re.findall(r"(?<=[:-])\d+", address)
    ]
    outside = [port for port in published if not first <= port <= last]
    assert outside == [CA_DEFAULT_PORT]


@pytest.mark.parametrize("base", [DEFAULT_PORT_BASE, 20000])
def test_the_whole_block_moves_with_one_key(tmp_path, base):
    """Nothing is described in the default block's numbers when it is elsewhere."""
    compose = _control_assistant_compose(tmp_path / f"docker-compose.{base}.yml", base)
    config = {"project_name": "demo", "deployment": {"port_base": base}}

    entries = deploy_summary.endpoint_entries(config, [compose])
    addresses = {service: address for _tier, service, address in entries}

    ports = layout_ports(base)
    assert addresses["openobserve"] == f"http://127.0.0.1:{ports['openobserve']}"
    assert addresses["postgresql"] == f"127.0.0.1:{ports['postgres']}"
    assert addresses["graphdb"] == (
        f"bolt 127.0.0.1:{ports['graphdb_bolt']} · browser http://127.0.0.1:{ports['graphdb_http']}"
    )
    assert addresses["virtual-accelerator"] == f"127.0.0.1:{CA_DEFAULT_PORT}"


# ---------------------------------------------------------------------------
# The panels tier: the largest stretch of the block, and the one neither
# binding source can see
# ---------------------------------------------------------------------------


def _panel_rows(entries):
    """The ``{family: address}`` of every panel row in ``entries``."""
    return {service: address for tier, service, address in entries if tier == "panels"}


@pytest.mark.parametrize("base", [DEFAULT_PORT_BASE, 20000])
def test_a_single_user_deployment_shows_every_panel_band(tmp_path, base):
    """The per-user panels reach the endpoint list, at the base this config resolved.

    They publish nothing a compose file records — the containers run on the host
    namespace — and their ports live under ``modules.web_terminals`` rather than
    ``services.<name>``, so neither binding source can see them. Without a third
    derivation the seven largest bands of the block are simply missing from
    ``osprey up`` and ``osprey status``.

    EVERY band, deliberately: this roster names no persona, so nothing on disk
    says which panels its user actually serves, and the whole-roster reading is
    the honest one. The bands are narrowed to who serves them only where a
    persona project answers the question — see
    ``test_a_band_lists_only_the_users_whose_persona_serves_it``.
    """
    empty = tmp_path / "docker-compose.yml"
    empty.write_text("services: {}\n", encoding="utf-8")
    config = {
        "project_name": "demo",
        "deployment": {"port_base": base},
        "modules": {"web_terminals": {"enabled": True}},
    }

    panels = _panel_rows(deploy_summary.endpoint_entries(config, [str(empty)]))

    ports = layout_ports(base)
    assert panels == {
        family: f"http://127.0.0.1:{ports[family]}"
        for family in (
            "web",
            "artifact",
            "ariel",
            "lattice",
            "channel_finder",
            "okf",
            "system_health",
        )
    }


@pytest.mark.parametrize("base", [DEFAULT_PORT_BASE, 20000])
def test_a_roster_shows_each_family_as_one_band_not_one_row_per_user(tmp_path, base):
    """One row per FAMILY, covering the roster — never one per family and user.

    A family IS a band, and the band is the fact. Four users across seven
    families would otherwise contribute twenty-eight rows to a list whose whole
    point is to show the shape of the block at a glance.

    No ``http://`` on a range: the renderer linkifies a whitespace-delimited
    address, so a scheme here would hand the operator a link that 404s on the
    dash — the same rule that keeps a URL off the graph store's bolt address.

    All three users on every band, deliberately: a bare-string roster names no
    persona, so no rendered project says which panels any of them serves and
    the whole roster is what this deployment can honestly claim.
    """
    empty = tmp_path / "docker-compose.yml"
    empty.write_text("services: {}\n", encoding="utf-8")
    config = {
        "project_name": "demo",
        "deployment": {"port_base": base},
        "modules": {"web_terminals": {"enabled": True, "users": ["alice", "bob", "carol"]}},
    }

    entries = deploy_summary.endpoint_entries(config, [str(empty)])
    panels = _panel_rows(entries)

    ports = layout_ports(base)
    assert len(panels) == 7
    assert panels["web"] == f"127.0.0.1:{ports['web']}-{ports['web'] + 2}  (alice, bob, carol)"
    assert panels["okf"] == f"127.0.0.1:{ports['okf']}-{ports['okf'] + 2}  (alice, bob, carol)"
    assert "http://" not in panels["web"]


def test_a_roster_too_long_to_name_is_counted_instead(tmp_path):
    """A fifty-user facility would push the row past any width, and say no more."""
    empty = tmp_path / "docker-compose.yml"
    empty.write_text("services: {}\n", encoding="utf-8")
    users = ["alice", "bob", "carol", "dave", "erin", "frank"]
    config = {
        "project_name": "demo",
        "modules": {"web_terminals": {"enabled": True, "users": users}},
    }

    panels = _panel_rows(deploy_summary.endpoint_entries(config, [str(empty)]))

    assert panels["web"].endswith("  (6 users)")
    assert "alice" not in panels["web"]


def test_a_deployment_with_no_web_tier_has_no_panels_section(tmp_path):
    """Absence of a module is a fact; an empty section would read as a failure."""
    empty = tmp_path / "docker-compose.yml"
    empty.write_text("services: {}\n", encoding="utf-8")

    entries = deploy_summary.endpoint_entries({"project_name": "demo"}, [str(empty)])

    assert _panel_rows(entries) == {}


def test_a_roster_index_past_its_band_costs_the_panels_not_the_summary(tmp_path):
    """Advisory to the end: an unallocatable roster is lint's refusal to raise."""
    empty = tmp_path / "docker-compose.yml"
    empty.write_text("services: {}\n", encoding="utf-8")
    config = {
        "project_name": "demo",
        "modules": {
            "web_terminals": {"enabled": True, "users": [{"name": "alice", "index": 5000}]}
        },
    }

    entries = deploy_summary.endpoint_entries(config, [str(empty)])

    assert _panel_rows(entries) == {}
    assert any(service == "web terminal" for _tier, service, _address in entries)


# ---------------------------------------------------------------------------
# Whose band it is: the panels a persona actually serves
# ---------------------------------------------------------------------------


def _persona_deployment(tmp_path, personas, users, base=DEFAULT_PORT_BASE):
    """A deploy repo whose personas' rendered projects are on disk.

    The shape ``osprey build`` leaves behind: a catalog under
    ``modules.web_terminals.personas`` whose ``project_path`` names a rendered
    project inside ``build/``, and each of those projects carrying the
    ``web.panels`` block its own profile asked for.

    Args:
        tmp_path: The deploy repo root.
        personas: ``{persona: [panel_id, ...]}`` — the panels that persona's
            rendered project declares. An empty list is a project that declares
            none, which still serves the universal WORKSPACE panel.
        users: The roster, raw, exactly as ``modules.web_terminals.users``.
        base: The deployment's port base.

    Returns:
        ``(config, compose_file_path)`` ready for :func:`endpoint_entries`.
    """
    for persona, panel_ids in personas.items():
        project_dir = tmp_path / "build" / f"demo-{persona}"
        project_dir.mkdir(parents=True)
        (project_dir / "config.yml").write_text(
            yaml.safe_dump({"web": {"panels": dict.fromkeys(panel_ids, True)}}),
            encoding="utf-8",
        )
    empty = tmp_path / "docker-compose.yml"
    empty.write_text("services: {}\n", encoding="utf-8")
    config = {
        "project_name": "demo",
        # What `osprey build` writes into the rendered config, and the only
        # anchor `osprey status` has for the persona projects: it hands
        # `endpoint_entries` a config and compose files, never a root.
        "project_root": str(tmp_path),
        "deployment": {"port_base": base},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "users": users,
                "personas": {
                    persona: {
                        "project": f"demo-{persona}",
                        "project_path": f"build/demo-{persona}",
                    }
                    for persona in personas
                },
            }
        },
    }
    return config, str(empty)


#: The two-persona roster every test below shares: alice runs a persona whose
#: project declares only the ARIEL panel, bob one that declares the channel
#: finder and the health dashboard. Neither declares LATTICE or KNOWLEDGE.
_SPLIT_PERSONAS = {"ariel": ["ariel"], "readonly": ["channel-finder", "system-health"]}
_SPLIT_ROSTER = [
    {"name": "alice", "index": 0, "persona": "ariel"},
    {"name": "bob", "index": 1, "persona": "readonly"},
]


@pytest.mark.parametrize("base", [DEFAULT_PORT_BASE, 20000])
def test_a_band_lists_only_the_users_whose_persona_serves_it(tmp_path, base):
    """A band names the users who answer on it, not everyone who has a port there.

    Every user is allocated a port in every family — the allocator reserves the
    whole block whatever the roster runs — but only the users whose persona
    declares that panel ever listen on theirs. Naming the whole roster beside
    the ARIEL band told an operator that bob answers there, and he does not:
    his container never starts the server.
    """
    config, compose = _persona_deployment(tmp_path, _SPLIT_PERSONAS, _SPLIT_ROSTER, base)

    panels = _panel_rows(deploy_summary.endpoint_entries(config, [compose]))

    ports = layout_ports(base)
    # One user on the band, so one port to open — and the scheme goes back on,
    # by the same rule that keeps it off a range.
    assert panels["ariel"] == f"http://127.0.0.1:{ports['ariel']}  (alice)"
    assert panels["channel_finder"] == f"http://127.0.0.1:{ports['channel_finder'] + 1}  (bob)"
    assert panels["system_health"] == f"http://127.0.0.1:{ports['system_health'] + 1}  (bob)"
    # The terminal itself and the universal WORKSPACE panel are served by
    # everyone: no persona can switch off the tab it cannot switch off.
    assert panels["web"] == f"127.0.0.1:{ports['web']}-{ports['web'] + 1}  (alice, bob)"
    assert panels["artifact"] == (
        f"127.0.0.1:{ports['artifact']}-{ports['artifact'] + 1}  (alice, bob)"
    )


def test_a_family_no_persona_serves_prints_no_band_at_all(tmp_path):
    """A band nothing listens on is not a quieter row — it is not a row.

    LATTICE and KNOWLEDGE are reserved by the layout for every deployment, and
    neither persona here declares them. Printing their bands with a hedge
    beside them would be the same claim in smaller type: the address would
    still be there to copy, and nothing answers at it.
    """
    config, compose = _persona_deployment(tmp_path, _SPLIT_PERSONAS, _SPLIT_ROSTER)

    panels = _panel_rows(deploy_summary.endpoint_entries(config, [compose]))

    assert "lattice" not in panels
    assert "okf" not in panels
    assert str(layout_ports(DEFAULT_PORT_BASE)["lattice"]) not in "".join(panels.values())


def test_the_reserved_but_unserved_bands_are_one_note_not_a_row_each(tmp_path):
    """Said once, at the end of the tier: the block still reserves them.

    An operator who reads six families where the layout has seven needs to know
    the seventh was not lost — but a per-family row for each would re-introduce
    exactly the addresses the rows above deliberately dropped.
    """
    config, compose = _persona_deployment(tmp_path, _SPLIT_PERSONAS, _SPLIT_ROSTER)

    entries = deploy_summary.endpoint_entries(config, [compose])
    notes = [
        address
        for tier, service, address in entries
        if tier == "panels" and service == deploy_summary.RESERVED_BANDS_LABEL
    ]

    assert len(notes) == 1
    assert notes[0].startswith("lattice, okf ")
    # Last in its tier, so it reads as a footnote to the bands above it rather
    # than as a band between them.
    panel_services = [service for tier, service, _address in entries if tier == "panels"]
    assert panel_services[-1] == deploy_summary.RESERVED_BANDS_LABEL


def test_a_deployment_every_persona_serves_carries_no_note(tmp_path):
    """Nothing reserved-and-unserved is nothing to say; a note would be noise."""
    every_panel = ["ariel", "channel-finder", "lattice", "okf", "system-health"]
    config, compose = _persona_deployment(
        tmp_path,
        {"admin": every_panel},
        [{"name": "alice", "index": 0, "persona": "admin"}],
    )

    entries = deploy_summary.endpoint_entries(config, [compose])
    panels = _panel_rows(entries)

    assert deploy_summary.RESERVED_BANDS_LABEL not in panels
    assert len(panels) == 7


def test_an_unbuilt_persona_project_degrades_to_the_whole_roster(tmp_path):
    """Advisory to the end: an unreadable persona costs the narrowing, not the summary.

    A catalog naming a project that has not been rendered — a repo cloned but
    not built, a persona added since the last build — leaves nothing on disk to
    read. Guessing which panels its users serve would be worse than the wide
    answer, and failing the summary would be worse than both.
    """
    config, compose = _persona_deployment(tmp_path, _SPLIT_PERSONAS, _SPLIT_ROSTER)
    (tmp_path / "build" / "demo-readonly" / "config.yml").unlink()

    panels = _panel_rows(deploy_summary.endpoint_entries(config, [compose]))

    ports = layout_ports(DEFAULT_PORT_BASE)
    assert len(panels) == 7
    assert panels["lattice"] == f"127.0.0.1:{ports['lattice']}-{ports['lattice'] + 1}  (alice, bob)"
    assert deploy_summary.RESERVED_BANDS_LABEL not in panels


def test_a_persona_whose_project_declares_no_panels_still_serves_the_universal_one(tmp_path):
    """WORKSPACE is not a panel a project can decline, so its band is always served."""
    config, compose = _persona_deployment(
        tmp_path,
        {"minimal": []},
        [{"name": "alice", "index": 0, "persona": "minimal"}],
    )

    panels = _panel_rows(deploy_summary.endpoint_entries(config, [compose]))

    ports = layout_ports(DEFAULT_PORT_BASE)
    assert panels["artifact"] == f"http://127.0.0.1:{ports['artifact']}  (alice)"
    assert panels["web"] == f"http://127.0.0.1:{ports['web']}  (alice)"
    assert set(panels) == {"web", "artifact", deploy_summary.RESERVED_BANDS_LABEL}


def test_a_panel_switched_off_by_a_persona_is_not_served(tmp_path):
    """``enabled: false`` is a declaration too, and it is a declaration of absence."""
    tmp_path.joinpath("build", "demo-readonly").mkdir(parents=True)
    tmp_path.joinpath("build", "demo-readonly", "config.yml").write_text(
        yaml.safe_dump({"web": {"panels": {"ariel": True, "okf": {"enabled": False}}}}),
        encoding="utf-8",
    )
    empty = tmp_path / "docker-compose.yml"
    empty.write_text("services: {}\n", encoding="utf-8")
    config = {
        "project_name": "demo",
        "project_root": str(tmp_path),
        "modules": {
            "web_terminals": {
                "enabled": True,
                "users": [{"name": "alice", "index": 0, "persona": "readonly"}],
                "personas": {"readonly": {"project_path": "build/demo-readonly"}},
            }
        },
    }

    panels = _panel_rows(deploy_summary.endpoint_entries(config, [str(empty)]))

    assert "ariel" in panels
    assert "okf" not in panels


def test_the_narrowing_reaches_status_through_the_entries_it_shares(tmp_path):
    """``osprey status`` and the deploy summary move together, or they contradict.

    Status hands :func:`endpoint_entries` a config and compose files and no
    root, so the persona projects are found through the rendered config's own
    ``project_root``. If that anchor were dropped, the deploy that printed the
    narrow bands would be followed by a status printing the wide ones, and the
    operator would have no way to tell which surface was lying.
    """
    config, compose = _persona_deployment(tmp_path, _SPLIT_PERSONAS, _SPLIT_ROSTER)

    entries = deploy_summary.endpoint_entries(config, [compose])
    rows = dict(deploy_summary.summary_rows(entries))

    ports = layout_ports(DEFAULT_PORT_BASE)
    assert rows["  ariel"] == f"http://127.0.0.1:{ports['ariel']}  (alice)"
    assert "  lattice" not in rows


def test_a_root_in_hand_beats_the_one_the_config_declares(tmp_path):
    """A moved repo is read where it IS, for a caller that knows where that is.

    ``as_built_endpoint_entries`` holds the repo it was pointed at; the
    ``project_root`` inside a rendered config was written wherever the build
    happened to run. When the two disagree the caller's is the live one — a
    deployment copied to another host would otherwise resolve its personas
    against a path that belongs to a different machine.
    """
    config, _compose = _persona_deployment(tmp_path, _SPLIT_PERSONAS, _SPLIT_ROSTER)
    config["project_root"] = str(tmp_path / "somewhere-else")
    (tmp_path / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")

    panels = _panel_rows(deploy_summary.as_built_endpoint_entries(tmp_path))

    ports = layout_ports(DEFAULT_PORT_BASE)
    assert panels["ariel"] == f"http://127.0.0.1:{ports['ariel']}  (alice)"
    assert "lattice" not in panels


def test_a_roster_user_the_catalog_cannot_place_degrades_the_whole_tier(tmp_path):
    """One unplaceable user is not narrowed around — the tier goes wide.

    Reporting the bands the OTHER users serve would silently drop this one from
    every band he might be on, which reads as "carol serves nothing" rather
    than as "nothing here knows what carol serves".
    """
    config, compose = _persona_deployment(
        tmp_path,
        _SPLIT_PERSONAS,
        [*_SPLIT_ROSTER, {"name": "carol", "index": 2}],
    )

    panels = _panel_rows(deploy_summary.endpoint_entries(config, [compose]))

    ports = layout_ports(DEFAULT_PORT_BASE)
    assert len(panels) == 7
    assert panels["lattice"] == (
        f"127.0.0.1:{ports['lattice']}-{ports['lattice'] + 2}  (alice, bob, carol)"
    )


def _unparseable_persona_config(tmp_path, config):
    """A rendered ``config.yml`` that is not YAML — a build interrupted mid-write."""
    path = tmp_path / "build" / "demo-readonly" / "config.yml"
    path.write_text("web: [unclosed\n", encoding="utf-8")


def _catalog_entry_without_a_project_path(tmp_path, config):
    """A persona declared but never pointed at a project."""
    config["modules"]["web_terminals"]["personas"]["readonly"].pop("project_path")


def _roster_entry_that_is_not_a_user(tmp_path, config):
    """A bare scalar where a roster entry belongs — a hand-edited config, or ``--no-lint``."""
    config["modules"]["web_terminals"]["users"] = [*_SPLIT_ROSTER, 42]


def _authorization_stanza_that_does_not_parse(tmp_path, config):
    """A declared role that names no persona — one of the parser's refusals.

    Not a scalar ``authorization: 42``: that one parses to the inert defaults
    (``as_dict`` reads a non-mapping as empty), so it degrades nothing and is
    not a case. What the parser genuinely refuses is a binding it cannot
    resolve, and a personaless role is the shortest of the three.
    """
    config["modules"]["web_terminals"]["authorization"] = {"roles": {"operator": {}}}


@pytest.mark.parametrize(
    "break_it",
    [
        _unparseable_persona_config,
        _catalog_entry_without_a_project_path,
        _roster_entry_that_is_not_a_user,
        _authorization_stanza_that_does_not_parse,
    ],
    ids=["unparseable-project", "no-project-path", "roster-entry-is-a-scalar", "bad-authorization"],
)
def test_anything_that_cannot_be_read_degrades_the_tier_rather_than_narrowing(tmp_path, break_it):
    """Every way the persona walk can fail ends in the wide bands, not in a traceback.

    The narrowing reads config the deploy has already accepted and a directory
    tree it does not own, so each step has a way to come back with nothing:
    an unparseable rendered project, a catalog entry pointing nowhere, a roster
    entry that is not a user, a role table that is not a table. All four are
    reachable past ``--no-lint`` or by hand-editing ``build/config.yml``, and
    all four must cost the *claim about who answers where* — never the summary
    that a successful deploy ends with.
    """
    config, compose = _persona_deployment(tmp_path, _SPLIT_PERSONAS, _SPLIT_ROSTER)
    break_it(tmp_path, config)

    panels = _panel_rows(deploy_summary.endpoint_entries(config, [compose]))

    ports = layout_ports(DEFAULT_PORT_BASE)
    assert len(panels) == 7
    assert deploy_summary.RESERVED_BANDS_LABEL not in panels
    assert panels["lattice"] == f"127.0.0.1:{ports['lattice']}-{ports['lattice'] + 1}  (alice, bob)"


def test_a_roster_entry_that_is_not_a_user_reaches_no_caller_as_an_exception(tmp_path):
    """The two surfaces that do not wrap this call are the ones that must not raise.

    ``format_endpoint_summary`` and the closing card's
    ``as_built_endpoint_entries`` both call the derivation bare, so an
    exception out of it replaces a finished deploy's report with a traceback
    and tells the operator the deploy failed when it did not. Asserted through
    both, because the crash was in a helper neither of them can see.
    """
    config, compose = _persona_deployment(
        tmp_path, _SPLIT_PERSONAS, [*_SPLIT_ROSTER, 42], base=DEFAULT_PORT_BASE
    )
    (tmp_path / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")

    text = deploy_summary.format_endpoint_summary(config, [compose])
    card_rows = deploy_summary.as_built_endpoint_entries(tmp_path)

    ports = layout_ports(DEFAULT_PORT_BASE)
    # Degraded, and degraded to something that still says where the block is.
    assert f"127.0.0.1:{ports['lattice']}-{ports['lattice'] + 1}  (alice, bob)" in text
    assert len(_panel_rows(card_rows)) == 7


def test_a_persona_project_with_no_web_block_serves_the_universal_bands(tmp_path):
    """A project that declares nothing is an ANSWER, not a failure to read one.

    ``web:`` missing entirely is what a rendered project without any panel
    configuration looks like, and the terminal's own reading of that is
    unambiguous: it starts the WORKSPACE panel and nothing else. So this narrows
    to two bands rather than degrading to seven — degrading here would report
    five addresses that this deployment genuinely does not answer on.
    """
    config, compose = _persona_deployment(
        tmp_path, {"minimal": []}, [{"name": "alice", "index": 0, "persona": "minimal"}]
    )
    (tmp_path / "build" / "demo-minimal" / "config.yml").write_text(
        yaml.safe_dump({"project_name": "demo-minimal"}), encoding="utf-8"
    )

    panels = _panel_rows(deploy_summary.endpoint_entries(config, [compose]))

    ports = layout_ports(DEFAULT_PORT_BASE)
    assert set(panels) == {"web", "artifact", deploy_summary.RESERVED_BANDS_LABEL}
    assert panels["artifact"] == f"http://127.0.0.1:{ports['artifact']}  (alice)"


# ---------------------------------------------------------------------------
# Placement: a framework-placed port is never filed under the facility's band
# ---------------------------------------------------------------------------


def test_a_per_user_container_that_published_a_port_lands_in_panels(tmp_path):
    """``web-alice`` is the web family, whatever suffix the roster gave it.

    The per-user containers are host-mode today and publish nothing, so this is
    the topology guard rather than a live path: if they ever publish, the row
    must not appear beside the facility's own services.
    """
    path = tmp_path / "docker-compose.web.yml"
    ports = layout_ports(DEFAULT_PORT_BASE)
    path.write_text(
        f'services:\n  web-alice:\n    ports:\n      - "127.0.0.1:{ports["web"]}:8080"\n',
        encoding="utf-8",
    )

    entries = deploy_summary.endpoint_entries({"project_name": "demo"}, [str(path)])

    assert ("panels", "web-alice", f"127.0.0.1:{ports['web']}") in entries


def test_an_unknown_service_is_placed_by_the_band_it_actually_binds(tmp_path):
    """A port still inside the block was placed by the layout, whatever it is called.

    Filing it under ``facility`` would say the facility placed a port the
    framework placed — and would silently absorb any framework service someone
    forgets to add to the service table.
    """
    path = tmp_path / "docker-compose.unknown.yml"
    ports = layout_ports(DEFAULT_PORT_BASE)
    path.write_text(
        f'services:\n  mystery:\n    ports:\n      - "127.0.0.1:{ports["openobserve"]}:80"\n',
        encoding="utf-8",
    )

    entries = deploy_summary.endpoint_entries({"project_name": "demo"}, [str(path)])

    assert ("services", "mystery", f"127.0.0.1:{ports['openobserve']}") in entries


def test_a_port_outside_the_block_still_falls_to_the_facility_band(tmp_path):
    """The catch-all survives: nothing in the layout claims a port outside it."""
    path = tmp_path / "docker-compose.outside.yml"
    path.write_text(
        'services:\n  mystery:\n    ports:\n      - "127.0.0.1:4000:80"\n',
        encoding="utf-8",
    )

    entries = deploy_summary.endpoint_entries({"project_name": "demo"}, [str(path)])

    assert ("facility", "mystery", "127.0.0.1:4000") in entries


def test_the_name_wins_over_the_band_a_service_was_moved_to(tmp_path):
    """A service's identity outlives whichever port it was moved to.

    ``postgresql`` published on the openobserve band is still a store, and
    placing it by where it landed would file it under `services`.
    """
    path = tmp_path / "docker-compose.moved.yml"
    ports = layout_ports(DEFAULT_PORT_BASE)
    path.write_text(
        f'services:\n  postgresql:\n    ports:\n      - "127.0.0.1:{ports["openobserve"]}:5432"\n',
        encoding="utf-8",
    )

    entries = deploy_summary.endpoint_entries({"project_name": "demo"}, [str(path)])

    assert ("stores", "postgresql", f"127.0.0.1:{ports['openobserve']}") in entries


# ---------------------------------------------------------------------------
# One sectioning, two paddings
# ---------------------------------------------------------------------------


def test_the_logged_block_and_the_printed_section_carry_the_same_sections(compose_file):
    """The record a log aggregator holds is sectioned like the one on screen.

    Only the padding differs — the record keeps a fixed column so one long
    service name cannot re-flow every other line of it — and both take their
    grouping from the same function, so neither can grow a section the other
    lacks.
    """
    config = {"project_name": "demo", "modules": {"web_terminals": {"enabled": True}}}
    entries = deploy_summary.endpoint_entries(config, [compose_file])

    printed = [label.strip() for label, value in deploy_summary.summary_rows(entries) if not value]
    logged = [
        line.strip()
        for line in deploy_summary._summary_text("t", entries).splitlines()[1:]
        if not line.startswith("    ")
    ]

    assert printed == logged
    assert printed == ["gateway", "dispatch", "services", "panels", "stores"]


def test_every_deploy_exit_path_shares_this_one_seam():
    """The exit paths inherit the printed form because they all land here.

    ``deploy_up`` reports its endpoints from several exits — the empty-stack
    early return, the web-terminals-only return, the detached run and the
    foreground ``execvpe`` — and each calls this one function, so the promotion
    lives in one place rather than in four.
    """
    from osprey.deployment import container_lifecycle

    assert container_lifecycle.log_endpoint_summary is deploy_summary.log_endpoint_summary
