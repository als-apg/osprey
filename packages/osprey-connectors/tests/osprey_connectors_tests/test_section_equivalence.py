"""The section functions read a ControlSystemSection exactly as they read the raw mapping.

The 23 section functions (and the four write-authorization verdicts built on
them) take a :class:`~osprey_connectors.config_sections.ControlSystemSection`
read once, where they used to take the raw ``control_system`` mapping and
re-walk it. This is the safety gate for that change: the functions as they
were at a4d8f67c62 are copied verbatim into
:mod:`osprey_connectors_tests._sections_at_a4d8f67c62`, and both generations
answer every section of one corpus. They must agree on every answer, and where
one raises, on the exception's class, ``key`` and message.

The corpus is every ``control_system`` block written in the tree at a4d8f67c62
(:mod:`osprey_connectors_tests.fixtures.section_corpus`: the rendered
sections of the goldens and fixtures, and every section literal a test built)
and, generated from it, the malformed shapes a hand-edited render can carry:
each leaf of each section replaced in turn by a ``null``, an unexpanded
``${VAR}``, a quoted ``'true'``, a number, a list, an empty string, an empty
mapping, a word padded with spaces and words of the vocabulary in the wrong
place, a non-string key beside each mapping's own, and sections that are not
mappings at all.
"""

from __future__ import annotations

import copy
import logging
import random
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from osprey_connectors import _control_state
from osprey_connectors.config_sections import ControlSystemSection
from osprey_connectors.connector_types import (
    connector_transport,
    resolve_control_system_type,
    resolve_serving,
    speaks_channel_access,
    talks_to_network,
)
from osprey_connectors.identity import acting_identity
from osprey_connectors.limits_posture import (
    incomplete_limits_blocks,
    most_restrictive_limits_posture,
    target_limits_posture,
    tier_limits_database_targets,
    type_limits_posture,
)
from osprey_connectors.targets import (
    baseline_target,
    configured_targets,
    machine_kind,
    resolve_target,
    switch_capable,
    target_configured,
)
from osprey_connectors.write_switch import (
    NO_OWNER,
    backstop_verdict,
    execution_verdict,
    owner_tier_term,
    write_ceiling,
    write_verdict,
)
from osprey_connectors.write_tier import tier_allows, tier_writes
from osprey_connectors_tests import _sections_at_a4d8f67c62 as old
from osprey_connectors_tests._harness._control_context_fixtures import write_control_context
from osprey_connectors_tests._harness._write_verdict_cases import case_environment, host_root
from osprey_connectors_tests.fixtures.section_corpus import CORPUS

__all__ = ["case_environment"]

#: Every target word, and the shapes a caller reading one from somewhere untyped hands over.
TARGETS: tuple[object, ...] = ("live", "va", "standin", "labatory", "", None, 3)

#: Connector types a caller names, beside every type a section's table keys.
TYPES: tuple[str | None, ...] = (
    None,
    "",
    "epics",
    "virtual_accelerator",
    "live_standin",
    "doocs",
    "tango",
    "mypackage.SiteConnector",
)

#: The values a leaf is replaced by.
MALFORMED: tuple[object, ...] = (
    None,
    "${UNSET_VAR}",
    "true",
    1,
    0,
    ["x"],
    [],
    "",
    {},
    True,
    " epics ",
    "allowed",
    "in_process",
)

#: The sections that are not mappings at all.
NOT_MAPPINGS: tuple[object, ...] = (None, [], ["type"], "control_system", 5, True, ())

#: How many generated shapes the verdicts, which read a record, are asked about.
VERDICT_SAMPLE = 600


def _mutations(section: dict[Any, Any]) -> Iterator[dict[Any, Any]]:
    """Every leaf of *section* replaced by each malformed value, and a non-string key beside each mapping."""

    def walk(node: dict[Any, Any], path: tuple[Any, ...]) -> Iterator[tuple[Any, ...]]:
        yield path
        for key, value in node.items():
            if isinstance(value, dict):
                yield from walk(value, (*path, key))
            else:
                yield (*path, key)

    def at(root: dict[Any, Any], path: tuple[Any, ...]) -> Any:
        node = root
        for key in path:
            node = node[key]
        return node

    for path in walk(section, ()):
        node = at(section, path)
        if isinstance(node, dict):
            shaped = copy.deepcopy(section)
            at(shaped, path)[7] = "seven"
            yield shaped
            continue
        for bad in MALFORMED:
            shaped = copy.deepcopy(section)
            at(shaped, path[:-1])[path[-1]] = copy.deepcopy(bad)
            yield shaped


def _sections() -> list[object]:
    sections: list[object] = list(NOT_MAPPINGS)
    seen: set[str] = set()
    for base in CORPUS:
        for shape in (base, *_mutations(base)):
            key = repr(shape)
            if key not in seen:
                seen.add(key)
                sections.append(shape)
    return sections


SECTIONS = _sections()


def _outcome(call: Callable[[], object]) -> tuple[str, object]:
    """What a call answered: its value, or its exception's class, ``key`` and message."""
    try:
        return ("answer", call())
    except Exception as error:  # the comparison is of what each generation raises
        return ("raised", (type(error), getattr(error, "key", None), str(error)))


def _owners(section: object) -> list[object]:
    owners: list[object] = ["alice", "bob", " alice", "", None, NO_OWNER, acting_identity()]
    tiers = section.get("login_tiers") if isinstance(section, dict) else None
    if isinstance(tiers, dict):
        owners.extend(name for name in tiers if isinstance(name, str))
    return owners


def _types(section: object) -> list[str | None]:
    types = list(TYPES)
    table = section.get("connector") if isinstance(section, dict) else None
    if isinstance(table, dict):
        types.extend(name for name in table if isinstance(name, str))
    return types


def _pairs(section: object) -> Iterator[tuple[str, Callable[[], object], Callable[[], object]]]:
    """``(name, the old call, the new call)`` for every reading of *section* compared."""
    read = ControlSystemSection.read(section)
    raw: Any = section
    yield (
        "resolve_control_system_type",
        lambda: old._old_resolve_control_system_type(raw),
        lambda: resolve_control_system_type(read),
    )
    yield "resolve_serving", lambda: old._old_resolve_serving(raw), lambda: resolve_serving(read)
    yield "baseline_target", lambda: old._old_baseline_target(raw), lambda: baseline_target(read)
    yield "switch_capable", lambda: old._old_switch_capable(raw), lambda: switch_capable(read)
    yield (
        "configured_targets",
        lambda: old._old_configured_targets(raw),
        lambda: configured_targets(read),
    )
    yield "tier_writes", lambda: old._old_tier_writes(raw), lambda: tier_writes(read)
    yield "tier_allows", lambda: old._old_tier_allows(raw), lambda: tier_allows(read)
    yield (
        "most_restrictive_limits_posture",
        lambda: old._old_most_restrictive_limits_posture(raw),
        lambda: most_restrictive_limits_posture(read),
    )
    yield (
        "incomplete_limits_blocks",
        lambda: old._old_incomplete_limits_blocks(raw),
        lambda: incomplete_limits_blocks(read),
    )
    for configured in (True, False):
        yield (
            f"tier_limits_database_targets[{configured}]",
            lambda c=configured: old._old_tier_limits_database_targets(raw, path_configured=c),
            lambda c=configured: tier_limits_database_targets(read, path_configured=c),
        )
    for target in TARGETS:
        t: Any = target
        yield (
            f"resolve_target[{t!r}]",
            lambda t=t: old._old_resolve_target(raw, t),
            lambda t=t: resolve_target(read, t),
        )
        yield (
            f"target_configured[{t!r}]",
            lambda t=t: old._old_target_configured(raw, t),
            lambda t=t: target_configured(read, t),
        )
        yield (
            f"machine_kind[{t!r}]",
            lambda t=t: old._old_machine_kind(raw, t),
            lambda t=t: machine_kind(read, t),
        )
        yield (
            f"target_limits_posture[{t!r}]",
            lambda t=t: old._old_target_limits_posture(raw, t),
            lambda t=t: target_limits_posture(read, t),
        )
    for kind in _types(section):
        yield (
            f"type_limits_posture[{kind!r}]",
            lambda k=kind: old._old_type_limits_posture(raw, k),
            lambda k=kind: type_limits_posture(read, k),
        )
        yield (
            f"connector_transport[{kind!r}]",
            lambda k=kind: old._old_connector_transport(raw, k),
            lambda k=kind: connector_transport(read, k),
        )
        yield (
            f"speaks_channel_access[{kind!r}]",
            lambda k=kind: old._old_speaks_channel_access(raw, k),
            lambda k=kind: speaks_channel_access(read, k),
        )
        yield (
            f"talks_to_network[{kind!r}]",
            lambda k=kind: old._old_talks_to_network(raw, k),
            lambda k=kind: talks_to_network(read, k),
        )
    for owner in _owners(section):
        o: Any = owner
        yield (
            f"owner_tier_term[{o!r}]",
            lambda o=o: old._old_owner_tier_term(raw, o),
            lambda o=o: owner_tier_term(read, o),
        )
        for bound in (True, False, None):
            yield (
                f"write_ceiling[{o!r}, {bound}]",
                lambda o=o, b=bound: old._old_write_ceiling(raw, o, tree_bound=b),
                lambda o=o, b=bound: write_ceiling(read, o, tree_bound=b),
            )


def _verdict_pairs(
    section: object,
) -> Iterator[tuple[str, Callable[[], object], Callable[[], object]]]:
    """The verdicts, which also read the record and the run mode."""
    read = ControlSystemSection.read(section)
    raw: Any = section
    me = acting_identity()
    yield "backstop_verdict", lambda: old._old_backstop_verdict(raw), lambda: backstop_verdict(read)
    for target in ("live", "va", "standin", "labatory", None):
        t: Any = target
        yield (
            f"write_verdict[{t!r}]",
            lambda t=t: old._old_write_verdict(raw, t, owner=me),
            lambda t=t: write_verdict(read, t, owner=me),
        )
        yield (
            f"execution_verdict[{t!r}]",
            lambda t=t: old._old_execution_verdict(raw, t, owner=me),
            lambda t=t: execution_verdict(read, t, owner=me),
        )


def _differences(
    sections: list[object],
    pairs: Callable[[object], Iterator[tuple[str, Callable[[], object], Callable[[], object]]]],
) -> list[str]:
    found: list[str] = []
    for section in sections:
        for name, before, after in pairs(section):
            was, now = _outcome(before), _outcome(after)
            if was != now:
                found.append(f"{name} over {section!r}: was {was!r}, now {now!r}")
    return found


@pytest.fixture(autouse=True)
def _quiet(caplog):
    """The second-real-block warning is said once per shape per generation; it is not compared."""
    caplog.set_level(logging.ERROR)


def test_the_corpus_is_large_enough_to_mean_something():
    assert len(CORPUS) >= 300
    assert len(SECTIONS) >= 10_000


#: The corpus is compared in this many slices, so a parallel run spreads it.
SLICES = 8


@pytest.mark.parametrize("piece", range(SLICES))
@pytest.mark.parametrize("tree_bound", [False, True])
def test_every_section_reads_the_same_answers(monkeypatch, tree_bound, piece):
    monkeypatch.setattr(_control_state, "tree_is_bound", lambda: tree_bound)
    differences = _differences(SECTIONS[piece::SLICES], _pairs)
    assert not differences, f"{len(differences)} differences, first: " + "\n".join(differences[:5])


@pytest.mark.usefixtures("case_environment")
@pytest.mark.parametrize("record", ["absent", "every switch allowed", "every switch asked"])
def test_every_section_reaches_the_same_verdict(tmp_path, record):
    if record != "absent":
        position = record.split()[-1]
        write_control_context(
            host_root(tmp_path),
            owned_by=None,
            switch=dict.fromkeys(("live", "va", "standin"), position),
        )
    sample = list(CORPUS) + random.Random(2026).sample(SECTIONS, VERDICT_SAMPLE)
    differences = _differences(sample, _verdict_pairs)
    assert not differences, f"{len(differences)} differences, first: " + "\n".join(differences[:5])
