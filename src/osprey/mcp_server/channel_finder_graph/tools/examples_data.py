"""Curated Cypher examples for the graph channel-finder paradigm.

Pure data: no imports beyond the standard library and the
:class:`~osprey.mcp_server.graph.tools.examples_data.ExampleQuery` dataclass the
main-agent graph server already defines, so this module can be read by the
``example_queries`` tool, by tests and by documentation without touching the
driver or the store.

This catalogue is **not** the main-agent graph server's catalogue. That one
teaches an operator the shape of the store — how many devices there are, what
the ontology looks like, whether the import was complete. This one does the
channel finder's job and only that job: **a phrase an operator said turns into
the addresses a connector can talk to**. Every query is therefore about
``fullPv`` — reaching it, or starting from it.

Two ways in
-----------
The queries fall into two halves, and the split is the point:

* *Search by meaning* — the operator says "vacuum gauge" or "bending magnet"
  and the graph matches that against the prose the corpus carries: a per-address
  description, the meaning of the address' field and subfield, what the device
  family does, which system it belongs to.
* *Search by structure* — the operator already knows a word, a place, a device
  or an address, and the graph walks the machine: a class and everything under
  it, a whole section, one device's addresses split into reads and writes, one
  address back to the device that owns it.

Query shape
-----------
Every query is written against the shape neosemantics produces from a
NARAD-convention Turtle corpus with ``applyNeo4jNaming`` on, which is what
``osprey knowledge seed-graph`` loads:

* node labels are the RDF class local names — ``:Resource`` on every node, plus
  ``:ChannelBinding``, ``:Class``, and one device-class label per device. There
  is **no ``:Device`` label**: a device is reached as ``:Resource`` plus the
  properties only a device carries, or through its ``narad_sem`` class;
* relationship types are UPPERCASED — ``:HASBINDING``, ``:READSSIGNAL``,
  ``:WRITESSIGNAL``, ``:SUBCLASSOF``, ``:TYPE``;
* property names keep their ``narad_p:`` local spelling — ``.fullPv``,
  ``.sourceName``, ``.sectionCode``, ``.description``, ``.system``;
* ``skos:altLabel`` arrives as a **list**, because the store is configured with
  ``handleMultival: ARRAY`` for that predicate. Synonym matching therefore uses
  list semantics (``ANY(l IN cls.altLabel WHERE ...)``); a scalar ``CONTAINS``
  against ``altLabel`` would be a bug, not a shortcut.

Corpora
-------
The Cypher is corpus-agnostic — nothing about a particular facility is baked
into a query, only into its parameter values. Each example carries one parameter
set per corpus it can run against: ``"demo"`` for the generated demo-machine
corpus, ``"als"`` for the ALS GTB corpus that ships as ``als_gtb.ttl``.

The two corpora are not equally rich. The demo corpus is generated from a
channel database that carries prose, so it has the description predicates and
the ``system`` token; ``als_gtb.ttl`` is a fixed artifact with no in-repo source
database and carries none of them. An example that needs prose is therefore
**demo-only**: it omits the ``"als"`` key and names the exclusion in
``not_applicable``, which ``example_queries`` renders as a stated exclusion
rather than leaving the caller to guess. No parameter set is ever empty — every
example takes at least one parameter, and a declared corpus always supplies
exactly the parameters its query references.

Every query is bounded: it ends in a ``LIMIT``, and that ``LIMIT`` is at most
``services.graphdb.query_max_rows``' default of 200. Both halves matter. Ending
in a bound is what keeps an example from being the reason a result comes back
truncated; keeping the bound under the row cap is what makes the number in the
query the number a caller actually receives. An example limited to 500 does not
return 500 rows — it returns the cap's 200 and a truncation flag, which reads to
the caller as an answer that was cut short rather than as the whole answer the
query asked for.
"""

from __future__ import annotations

from dataclasses import dataclass

from osprey.mcp_server.graph.tools.examples_data import ExampleQuery

__all__ = ["CORPORA", "EXAMPLE_QUERIES", "ChannelFinderExample"]

#: The corpora this catalogue is written against, in the order they are reported.
CORPORA: tuple[str, ...] = ("als", "demo")


@dataclass(frozen=True)
class ChannelFinderExample(ExampleQuery):
    """An :class:`ExampleQuery` that can state a corpus it does not apply to.

    The main-agent catalogue carries a parameter set for every shipped corpus
    because every one of its queries runs everywhere. Half of this catalogue
    searches prose that only the demo corpus has, so an example needs a way to
    say "not this one" that is louder than a missing dictionary key.

    Attributes:
        not_applicable: Corpora this example cannot run against, each with no
            parameter set. Together with ``parameters`` this covers
            :data:`CORPORA` exactly: a corpus is either supplied or excluded,
            never silently absent.
    """

    not_applicable: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Search by meaning — the operator's words against the corpus' prose.
# These read the description predicates and the SYSTEM token, which only the
# generated demo corpus carries; als_gtb.ttl has no source database to enrich
# from, so each of them is demo-only by construction.
# ---------------------------------------------------------------------------

_BY_DESCRIPTION = ChannelFinderExample(
    key="by_description",
    title="Addresses whose description mentions a phrase",
    description=(
        "The first query to try on a phrase an operator used. Every address in "
        "the corpus carries a one-line description of what it is, and this "
        "matches the phrase against all of them at once, case-insensitively. "
        "Rows come back as address plus description, so the caller can see why "
        "each one matched instead of trusting a bare list.\n"
        "\n"
        "$phrase — the operator's words, matched as a substring. Keep it to the "
        "part that names the thing ('vacuum gauge', 'beam position') rather "
        "than a whole sentence; a longer phrase matches fewer descriptions, not "
        "more."
    ),
    cypher="""
MATCH (b:ChannelBinding)
WHERE toLower(b.description) CONTAINS toLower($phrase)
RETURN b.fullPv AS pv,
       b.description AS description
ORDER BY pv
LIMIT 200
""".strip(),
    parameters={"demo": {"phrase": "vacuum gauge"}},
    not_applicable=("als",),
)

_BY_FIELD_MEANING = ChannelFinderExample(
    key="by_field_meaning",
    title="Addresses whose field and subfield mean what was asked for",
    description=(
        "Narrower than a plain description search, and better when the operator "
        "named a quantity and a way of getting it rather than a device. The "
        "last two tokens of an address — its field and its subfield — each "
        "carry their own explanation, and this matches a phrase against both, "
        "so the two conditions have to hold on the same address.\n"
        "\n"
        "That is what separates readings that look alike. Asking for a field "
        "about pressure and a subfield about a gauge returns the directly "
        "measured pressures and leaves out the ones a pump infers from its own "
        "current, even though both are pressure readbacks.\n"
        "\n"
        "$field_meaning — a phrase from the quantity the field names.\n"
        "$subfield_meaning — a phrase from how that quantity is obtained "
        "(measured, commanded, calculated, archived as a reference)."
    ),
    cypher="""
MATCH (b:ChannelBinding)
WHERE toLower(b.fieldDescription) CONTAINS toLower($field_meaning)
  AND toLower(b.subfieldDescription) CONTAINS toLower($subfield_meaning)
RETURN b.fullPv AS pv,
       b.fieldDescription AS field_meaning,
       b.subfieldDescription AS subfield_meaning
ORDER BY pv
LIMIT 200
""".strip(),
    parameters={"demo": {"field_meaning": "pressure", "subfield_meaning": "gauge"}},
    not_applicable=("als",),
)

_BY_FAMILY_ROLE = ChannelFinderExample(
    key="by_family_role",
    title="Addresses of every device whose family does a described job",
    description=(
        "Goes through the *device* rather than the address. Each device carries "
        "an explanation of what its family is for, and this matches a phrase "
        "against that and then returns every address of every device that "
        "matched. Use it when the operator described a job — bending the beam, "
        "correcting the orbit, holding vacuum — instead of naming a device or a "
        "quantity.\n"
        "\n"
        "The result is wider than a description search: one matching family is "
        "tens of devices and hundreds of addresses, so read the device column "
        "before acting on a row.\n"
        "\n"
        "$role — a phrase from what the family does."
    ),
    cypher="""
MATCH (d:Resource)-[:HASBINDING]->(b:ChannelBinding)
WHERE toLower(d.familyDescription) CONTAINS toLower($role)
RETURN b.fullPv AS pv,
       d.sourceName AS device,
       d.sectionCode AS section,
       d.familyDescription AS family_role
ORDER BY pv
LIMIT 200
""".strip(),
    parameters={"demo": {"role": "bending magnet"}},
    not_applicable=("als",),
)

_BY_SYSTEM = ChannelFinderExample(
    key="by_system",
    title="Every address in one system of the machine",
    description=(
        "Scopes a search to a whole engineering system — magnets, vacuum, "
        "diagnostics, RF — by the system token the device carries, matched "
        "exactly rather than as a substring. This is the query to run before a "
        "phrase search when the operator has already said which system they "
        "mean: it turns 'the vacuum ones' into a bounded set instead of leaving "
        "the phrase to do that work.\n"
        "\n"
        "The system's own description comes back on every row, which is how a "
        "caller confirms it scoped to the system it meant.\n"
        "\n"
        "$system — the system token, exactly as the corpus spells it (upper "
        "case, no punctuation). Distinct values are the second token of any "
        "address the other examples return."
    ),
    cypher="""
MATCH (d:Resource)-[:HASBINDING]->(b:ChannelBinding)
WHERE d.system = $system
RETURN b.fullPv AS pv,
       d.sourceName AS device,
       d.systemDescription AS system_meaning
ORDER BY pv
LIMIT 200
""".strip(),
    parameters={"demo": {"system": "VAC"}},
    not_applicable=("als",),
)


# ---------------------------------------------------------------------------
# Search by structure — the machine's own shape, available on every corpus.
# ---------------------------------------------------------------------------

_BY_SYNONYM = ChannelFinderExample(
    key="by_synonym",
    title="Addresses of every device an operator's word can mean",
    description=(
        "Turns a piece of control-room vocabulary into addresses. The ontology "
        "records the words operators actually use for each class — 'quad' for a "
        "quadrupole, 'steerer' for a corrector, 'cold cathode' for a gauge — "
        "and this looks the word up there, then walks *down* the class "
        "hierarchy so a word naming a broad class also finds every kind "
        "beneath it: ask for 'magnet' and horizontal correctors come back "
        "without the query naming them.\n"
        "\n"
        "Synonyms are a list on each class, so the match is over the list's "
        "elements. It is an exact match on the whole synonym, not a substring, "
        "which is what keeps 'pm' from matching 'pump'.\n"
        "\n"
        "$synonym — one word or phrase as an operator would say it. Case does "
        "not matter."
    ),
    cypher="""
MATCH (cls:Class)
WHERE ANY(l IN cls.altLabel WHERE toLower(l) = toLower($synonym))
MATCH (sub:Class)-[:SUBCLASSOF*0..]->(cls)
MATCH (d:Resource)-[:TYPE]->(sub)
MATCH (d)-[:HASBINDING]->(b:ChannelBinding)
RETURN DISTINCT b.fullPv AS pv,
       d.sourceName AS device,
       d.sectionCode AS section
ORDER BY pv
LIMIT 200
""".strip(),
    parameters={"als": {"synonym": "quad"}, "demo": {"synonym": "quad"}},
)

_IN_SECTION = ChannelFinderExample(
    key="in_section",
    title="Every address in one section of the machine",
    description=(
        "Lists the addresses of every device sitting in a single section — a "
        "transfer line, a ring, a sector — with the device and its concrete "
        "class on each row. Sections are how operators partition the machine "
        "when they talk about a region of it, so this is the bounded set to "
        "hand back when the phrase named a place rather than a thing.\n"
        "\n"
        "$section — the section code, exactly as the corpus spells it. Values "
        "come from the ``section`` column of the other examples."
    ),
    cypher="""
MATCH (d:Resource)-[:HASBINDING]->(b:ChannelBinding)
WHERE d.sectionCode = $section
RETURN b.fullPv AS pv,
       d.sourceName AS device,
       [l IN labels(d) WHERE l <> "Resource"][0] AS device_class
ORDER BY pv
LIMIT 200
""".strip(),
    parameters={"als": {"section": "GTL"}, "demo": {"section": "SR"}},
)

_DEVICE_ADDRESSES = ChannelFinderExample(
    key="device_addresses",
    title="One device's addresses, split into reads and writes",
    description=(
        "The last step of most channel-finding: the operator named a device and "
        "the caller needs its addresses, with the ones that can be written "
        "marked as such. Each row is an address, the semantic signal it "
        "carries, whether it reads or writes, and how confident the corpus is "
        "in the binding.\n"
        "\n"
        "Direction here is what the corpus records, which is derived from the "
        "facility's own limits file rather than guessed from the address text. "
        "It says which addresses are *writable in principle*; whether a write "
        "is permitted right now is the control-system connector's decision, not "
        "the graph's.\n"
        "\n"
        "$name — the device's source name as the corpus records it.\n"
        "$section — the section it sits in. Both are needed because a source "
        "name can repeat across sections."
    ),
    cypher="""
MATCH (d:Resource {sourceName: $name, sectionCode: $section})-[:HASBINDING]->(b:ChannelBinding)
OPTIONAL MATCH (b)-[:READSSIGNAL]->(rs)
OPTIONAL MATCH (b)-[:WRITESSIGNAL]->(ws)
RETURN b.fullPv AS pv,
       last(split(coalesce(rs.uri, ws.uri), "/")) AS signal,
       CASE WHEN ws IS NULL THEN "read" ELSE "write" END AS direction,
       b.confidence AS confidence
ORDER BY pv
LIMIT 200
""".strip(),
    parameters={
        "als": {"name": "BC1", "section": "GTL"},
        "demo": {"name": "DIPOLE01", "section": "SR"},
    },
)

_BY_ADDRESS = ChannelFinderExample(
    key="by_address",
    title="The device behind an address",
    description=(
        "The reverse lookup: an address came from somewhere — a log entry, an "
        "alarm, a screen — and the caller needs to know what it belongs to. "
        "Returns the owning device with its section and class.\n"
        "\n"
        "More than one device on the row is the interesting case: those devices "
        "share a single control endpoint, so writing that address moves all of "
        "them at once. The generated demo corpus has no shared addresses by "
        "construction, so ``device_count`` is always 1 there; a corpus built "
        "from real wiring can have them.\n"
        "\n"
        "$pv — the full address to look up, exactly as the corpus records it in "
        "``fullPv``."
    ),
    cypher="""
MATCH (d:Resource)-[:HASBINDING]->(b:ChannelBinding {fullPv: $pv})
RETURN b.fullPv AS pv,
       collect(DISTINCT d.sourceName) AS devices,
       collect(DISTINCT d.sectionCode) AS sections,
       count(DISTINCT d) AS device_count
LIMIT 5
""".strip(),
    parameters={
        "als": {"pv": "GTL:BC1:Setpoint"},
        "demo": {"pv": "SR:MAG:DIPOLE:01:CURRENT:SP"},
    },
)


EXAMPLE_QUERIES: tuple[ChannelFinderExample, ...] = (
    _BY_DESCRIPTION,
    _BY_FIELD_MEANING,
    _BY_FAMILY_ROLE,
    _BY_SYSTEM,
    _BY_SYNONYM,
    _IN_SECTION,
    _DEVICE_ADDRESSES,
    _BY_ADDRESS,
)
"""The curated set, ordered from the loosest way in to the tightest.

The order is the reading order, and it follows how a channel-finding request
narrows: the first four take an operator's words and search the corpus' prose,
the next two scope a search to a class or a place, and the last two work from a
device or an address that is already known.
"""
