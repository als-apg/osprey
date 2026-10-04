"""TTL-based OKF stub document seeder.

Reads a canonical NARAD RDF/Turtle file and emits one
:class:`DeviceStub` per device node (``?d a narad_sem:<Class>``).

**rdflib is imported lazily inside each public function** — never at module
top — so that importing this module keeps rdflib out of the import graph.
rdflib is a core dependency, so an absent one means a broken environment:
callers that invoke :func:`seed_from_ttl` without it receive an
:class:`ImportError` saying how to repair the installation.

Stub document format (OKF §9)::

    ---
    type: device_stub
    title: GTL:BC1 (BuckingCoil)
    description: Auto-generated stub for BuckingCoil GTL/BC1.
    resource: https://narad.example.org/device/ex_device_GTL_x2F_BC1
    device_class: BuckingCoil
    device_id: GTL/BC1
    ---

    # Schema

    | Channel | PV | Signal | Direction |
    | ------- | -- | ------ | --------- |
    | GTL:BC1:CurrentRBV | GTL:BC1:CurrentRBV | current_readback | reads |
    ...

Idempotency: callers are responsible for comparing :attr:`DeviceStub.body`
against any existing file content and skipping the write when they match.
This module never touches the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported at type-checking time so the runtime import stays lazy.
    from rdflib import Graph, URIRef  # noqa: F401

# ---------------------------------------------------------------------------
# NARAD namespace constants (strings — avoid importing rdflib at module level)
# ---------------------------------------------------------------------------

_NARAD_P = "https://narad.example.org/property/"
_NARAD_SEM = "https://narad.example.org/schema/shared_semantics/"
_DEVICE_IRI_PREFIX = "https://narad.example.org/device/"
_RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

_P_DEVICE_ID = _NARAD_P + "deviceId"
_P_FACILITY = _NARAD_P + "facility"
_P_SOURCE_NAME = _NARAD_P + "sourceName"
_P_SECTION_CODE = _NARAD_P + "sectionCode"
_P_HAS_BINDING = _NARAD_P + "hasBinding"
_P_BINDING_ID = _NARAD_P + "bindingId"
_P_FULL_PV = _NARAD_P + "fullPv"
_P_READS_SIGNAL = _NARAD_P + "readsSignal"
_P_WRITES_SIGNAL = _NARAD_P + "writesSignal"


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class ChannelRow:
    """One row in the channel-bindings schema table.

    Args:
        channel: The channel's address, decoded from the binding IRI.
        pv: Full EPICS PV string (``narad_p:fullPv``).
        signal: Semantic signal name (local name of the ``readsSignal``/``writesSignal`` object).
        direction: ``"reads"`` or ``"writes"``.
    """

    channel: str
    pv: str
    signal: str
    direction: str


@dataclass
class DeviceStub:
    """An auto-generated OKF stub for a single device node.

    Args:
        resource: Canonical device IRI (verbatim from the TTL subject).
        device_class: Leaf class local name (e.g. ``Quadrupole``).
        device_id: Value of ``narad_p:deviceId``.
        title: Human-readable title string.
        channels: Channel binding rows sorted by channel name.
        body: Fully rendered OKF document text (frontmatter + Markdown body).
    """

    resource: str
    device_class: str
    device_id: str
    title: str
    channels: list[ChannelRow] = field(default_factory=list)
    body: str = ""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def local_name(iri: str) -> str:
    """Return the local name fragment of an IRI (after the last ``/`` or ``#``).

    This is the single source of truth for IRI → local-name derivation.  It is
    exported from the ``seeder`` subpackage and reused by the knowledge CLI so
    that stub *placement* (the bundle filename) and stub *identity* (the OKF §2
    concept ID) are computed by exactly one rule, including ``#``-fragment IRIs.
    """
    for sep in ("#", "/"):
        idx = iri.rfind(sep)
        if idx != -1:
            return iri[idx + 1 :]
    return iri


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _binding_channel_name(binding_iri: str, code: str, binding_id: str) -> str:
    """Name a channel by the address its binding IRI was minted from.

    The graph view mints ``.../channel/<code>_channel_<escaped address>``, which
    :func:`osprey.facility.views.graph_iri.decode` reverses with the facility's
    identity ``code``. An IRI it did not mint names the channel by its
    ``bindingId`` literal, and by the IRI's local name when that is absent.

    Args:
        binding_iri: The binding node's IRI.
        code: The owning device's ``narad_p:facility`` literal; empty when it
            carries none.
        binding_id: The binding's ``narad_p:bindingId`` literal; empty when it
            carries none.

    Returns:
        The channel name.
    """
    from osprey.facility.views.graph_iri import decode

    try:
        return decode(binding_iri, code)
    except ValueError:
        return binding_id or local_name(binding_iri)


def _build_channel_rows(g: Graph, binding_iris: list[str], code: str) -> list[ChannelRow]:
    """Build sorted channel rows from a list of binding IRIs."""
    # Import lazily — callers guarantee rdflib is available by this point.
    from rdflib import URIRef

    rows: list[ChannelRow] = []
    for b_iri in binding_iris:
        b_node = URIRef(b_iri)
        pv = str(g.value(b_node, URIRef(_P_FULL_PV)) or "")
        binding_id = str(g.value(b_node, URIRef(_P_BINDING_ID)) or "")

        reads_obj = g.value(b_node, URIRef(_P_READS_SIGNAL))
        writes_obj = g.value(b_node, URIRef(_P_WRITES_SIGNAL))

        if reads_obj is not None:
            direction = "reads"
            signal = local_name(str(reads_obj))
        elif writes_obj is not None:
            direction = "writes"
            signal = local_name(str(writes_obj))
        else:
            direction = "unknown"
            signal = ""

        channel = _binding_channel_name(b_iri, code, binding_id)
        rows.append(ChannelRow(channel, pv, signal, direction))

    rows.sort(key=lambda r: r.channel)
    return rows


def _render_schema_table(channels: list[ChannelRow]) -> str:
    """Render the channel list as a Markdown table."""
    if not channels:
        return "_No channel bindings declared._\n"

    lines = [
        "| Channel | PV | Signal | Direction |",
        "| ------- | -- | ------ | --------- |",
    ]
    for row in channels:
        lines.append(f"| {row.channel} | {row.pv} | {row.signal} | {row.direction} |")
    return "\n".join(lines) + "\n"


def _render_stub(stub: DeviceStub) -> str:
    """Render the full OKF document text for *stub*.

    Delegates frontmatter + body serialization to
    :meth:`~osprey.services.facility_knowledge.okf.document.OKFDocument.serialize`
    so the on-disk OKF format (YAML dump options, ``---`` fences, trailing-newline
    policy) has a single owner and stub output can never drift from documents
    written by any other path.
    """
    from osprey.services.facility_knowledge.okf.document import OKFDocument

    frontmatter = {
        "type": "device_stub",
        "title": stub.title,
        "description": f"Auto-generated stub for {stub.device_class} {stub.device_id}.",
        "resource": stub.resource,
        "device_class": stub.device_class,
        "device_id": stub.device_id,
    }
    schema_table = _render_schema_table(stub.channels)
    body = f"# Schema\n\n{schema_table}"
    return OKFDocument(frontmatter=frontmatter, body=body).serialize()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def seed_from_ttl(ttl_path: Path | str | None) -> list[DeviceStub]:
    """Seed OKF stub documents from a NARAD TTL file.

    Parses *ttl_path* with rdflib, then for every subject whose type IRI
    starts with the ``narad_sem:`` namespace prefix and is a device node
    (IRI starts with the device IRI prefix), emits a :class:`DeviceStub`
    with:

    * ``resource`` — the verbatim device IRI.
    * ``device_class`` — the local name of the ``rdf:type`` class.
    * ``device_id`` — value of ``narad_p:deviceId`` (fallback: local name of the
      device IRI when ``narad_p:deviceId`` is absent).
    * ``title`` — ``"<sectionCode>:<sourceName> (<device_class>)"`` when both
      section and source are present, otherwise ``"<device_class> <device_id>"``.
    * ``channels`` — sorted :class:`ChannelRow` list from all ``narad_p:hasBinding``
      objects.
    * ``body`` — fully rendered OKF document text.

    The result list is sorted by ``resource`` IRI for deterministic output.

    Args:
        ttl_path: Path to the ``.ttl`` file, or ``None`` / falsy value.  When
            *ttl_path* is ``None`` or empty, returns an empty list (no-op).

    Returns:
        List of :class:`DeviceStub` instances.  Empty when *ttl_path* is
        ``None`` or the file contains no device nodes.

    Raises:
        ImportError: If ``rdflib`` is not importable, which means the
            installation is incomplete.
        FileNotFoundError: If *ttl_path* does not exist.
    """
    if not ttl_path:
        return []

    ttl_path = Path(ttl_path)

    # Lazy import — keeps rdflib out of this module's import graph.
    try:
        from rdflib import Graph, URIRef
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"rdflib is not importable: {exc}\n"
            "It is a core dependency, so this environment is incomplete. "
            "Reinstall it with: pip install --upgrade osprey-framework"
        ) from exc

    g = Graph()
    g.parse(str(ttl_path), format="turtle")

    rdf_type = URIRef(_RDF_TYPE)
    stubs: list[DeviceStub] = []

    for subject in set(g.subjects(rdf_type, None)):
        subject_str = str(subject)
        if not subject_str.startswith(_DEVICE_IRI_PREFIX):
            # Skip groups, bindings, and schema nodes.
            continue

        # Collect all narad_sem rdf:type classes and pick the lexicographically
        # first local name.  Sorting makes device_class deterministic even if a
        # future node carries >1 narad_sem type (graph iteration order is not
        # guaranteed by rdflib).
        narad_sem_classes = sorted(
            local_name(str(type_obj))
            for type_obj in g.objects(subject, rdf_type)
            if str(type_obj).startswith(_NARAD_SEM)
        )
        if not narad_sem_classes:
            continue
        device_class = narad_sem_classes[0]

        device_id = str(g.value(subject, URIRef(_P_DEVICE_ID)) or "")
        source_name = str(g.value(subject, URIRef(_P_SOURCE_NAME)) or "")
        section_code = str(g.value(subject, URIRef(_P_SECTION_CODE)) or "")
        code = str(g.value(subject, URIRef(_P_FACILITY)) or "")

        # Build a human-readable title.
        if section_code and source_name:
            title = f"{section_code}:{source_name} ({device_class})"
        else:
            title = f"{device_class} {device_id or local_name(subject_str)}"

        # Collect binding IRIs.
        binding_iris = [str(b) for b in g.objects(subject, URIRef(_P_HAS_BINDING))]
        channels = _build_channel_rows(g, binding_iris, code)

        stub = DeviceStub(
            resource=subject_str,
            device_class=device_class,
            device_id=device_id or local_name(subject_str),
            title=title,
            channels=channels,
        )
        stub.body = _render_stub(stub)
        stubs.append(stub)

    stubs.sort(key=lambda s: s.resource)
    return stubs
