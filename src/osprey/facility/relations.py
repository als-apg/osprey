"""Relation claim rules: shape, keys, coarseness and standing.

Every function here is pure and takes the relation type rows as an argument
(the shape of :func:`osprey.facility.validate.relation_types`): the end kinds
``subject`` and ``object``, ``symmetric`` and the ``single_per`` key fields are
read from those rows only, never from a second table.

A claim's *key* is the fact it states a value for: the type with the values of
its ``single_per`` fields. ``subject`` is the keying end as ``(kind, id)``,
``port`` the keying end's port (``None`` when the end states none: two
port-less claims on one subject share a key), ``attributes.<x>`` that attribute
of the claim. A claim that leaves out a key attribute has no key there. A
symmetric type keys a claim once from each end, the other end being the object;
a type keyed by attributes only (``cabled_to``) gives one key from both views.

A *coarse* claim states less than the facility can know: an end names a place
where the type also allows a device, it leaves out a key attribute, or its
object leaves out a ``port`` or ``unit`` that another claim on the same key and
the same object states.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from osprey.facility.errors import FacilityBuildError, FacilityBuildWarning
from osprey.facility.provenance import add_defaults
from osprey.facility.sources import AUTHORED

if TYPE_CHECKING:
    from osprey.facility.validate import Validated

#: The kinds one relation end may name; an end names exactly one.
END_KINDS: tuple[str, ...] = ("device", "place", "group")

#: Attributes a type cannot be read without: a claim leaving one out stops.
REQUIRED_ATTRIBUTES: dict[str, tuple[str, ...]] = {"supplied_by": ("medium",)}

#: The reasons a claim is coarse, in the order :func:`coarse_reasons` lists them.
COARSE_PLACE = "place end"
COARSE_KEY_ATTRIBUTE = "missing key attribute"
COARSE_PORT = "missing port"
COARSE_UNIT = "missing unit"
COARSE_REASONS: tuple[str, ...] = (COARSE_PLACE, COARSE_KEY_ATTRIBUTE, COARSE_PORT, COARSE_UNIT)

#: A key: ``(type, (field, value), ...)`` in the type's ``single_per`` order.
Key = tuple[Any, ...]

#: One view of a claim: its key, the keying end and the other end.
View = tuple[Key, Mapping[str, Any], Mapping[str, Any]]


def _end(claim: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    end = claim.get(side)
    return end if isinstance(end, Mapping) else {}


def _kinds(end: Mapping[str, Any]) -> list[str]:
    return [kind for kind in END_KINDS if end.get(kind) is not None]


def end_identity(end: Mapping[str, Any]) -> tuple[str, str] | None:
    """The ``(kind, id)`` an end names, or ``None`` when it names no kind.

    An end naming more than one kind (a shape problem) reads as its first in
    :data:`END_KINDS` order.
    """
    kinds = _kinds(end)
    return (kinds[0], str(end[kinds[0]])) if kinds else None


def _end_value(end: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(end.get(name) for name in (*END_KINDS, "port", "unit"))


def _hashable(value: Any) -> Hashable:
    if isinstance(value, str | int | float | bool):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _attribute(claim: Mapping[str, Any], name: str) -> Any:
    attributes = claim.get("attributes")
    return attributes.get(name) if isinstance(attributes, Mapping) else None


def _key(claim: Mapping[str, Any], row: Mapping[str, Any], keying: Mapping[str, Any]) -> Key | None:
    values: list[tuple[str, Any]] = []
    for field in row.get("single_per") or []:
        if field == "subject":
            values.append((field, end_identity(keying)))
        elif field == "port":
            port = keying.get("port")
            values.append((field, None if port is None else str(port)))
        elif field.startswith("attributes."):
            value = _attribute(claim, field.removeprefix("attributes."))
            if value is None:
                return None
            values.append((field, _hashable(value)))
        else:
            values.append((field, _hashable(claim.get(field))))
    return (claim.get("type"), *values) if values else None


def views(claim: Mapping[str, Any], types: Mapping[str, Mapping[str, Any]]) -> list[View]:
    """Each key of *claim* with the end it is read from and the other end.

    Args:
        claim: One relation claim.
        types: The relation type rows, keyed by name.

    Returns:
        One view per end the claim is keyed from: one for a directed type, two
        for a symmetric one (``subject`` first). Empty for a type with no
        ``single_per``, for an unknown type, and when a key attribute is missing.
    """
    row = types.get(claim.get("type"))  # type: ignore[arg-type]
    if row is None:
        return []
    subject, object_ = _end(claim, "subject"), _end(claim, "object")
    sides = [(subject, object_)]
    if row.get("symmetric"):
        sides.append((object_, subject))
    found: list[View] = []
    for keying, other in sides:
        key = _key(claim, row, keying)
        if key is not None:
            found.append((key, keying, other))
    return found


def claim_keys(claim: Mapping[str, Any], types: Mapping[str, Mapping[str, Any]]) -> list[Key]:
    """The distinct keys of *claim*, in view order.

    A symmetric type keyed by ``subject`` gives one key per end; ``cabled_to``,
    keyed by ``attributes.cable`` only, gives one key from both.
    """
    keys: list[Key] = []
    for key, _keying, _other in views(claim, types):
        if key not in keys:
            keys.append(key)
    return keys


def spell_key(key: Key) -> str:
    """*key* as the subject of a warning line.

    ``("powered_by", ("subject", ("device", "Q1")), ("port", None))`` reads
    ``device Q1, no port``; an attribute field reads by its attribute name, as in
    ``cable C-12``.
    """
    parts: list[str] = []
    for field, value in key[1:]:
        name = field.removeprefix("attributes.")
        if field == "subject":
            parts.append("no end" if value is None else f"{value[0]} {value[1]}")
        elif value is None:
            parts.append(f"no {name}")
        else:
            parts.append(f"{name} {value}")
    return ", ".join(parts)


def coarse_reasons(
    claim: Mapping[str, Any],
    claims_on_key: Sequence[Mapping[str, Any]],
    types: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Why *claim* is coarse, in :data:`COARSE_REASONS` order; empty when it is not.

    Args:
        claim: One relation claim.
        claims_on_key: The claims sharing a key with *claim* (*claim* itself may
            be among them; it is skipped).
        types: The relation type rows, keyed by name.

    Returns:
        ``place end`` when an end names a place where the type allows a device;
        ``missing key attribute`` when a ``single_per`` attribute is left out;
        ``missing port`` / ``missing unit`` when, on a shared key, the object
        names the same record as another claim's but leaves out the ``port`` /
        ``unit`` that claim states.
    """
    row = types.get(claim.get("type"))  # type: ignore[arg-type]
    if row is None:
        return []
    reasons: set[str] = set()
    for side in ("subject", "object"):
        if _end(claim, side).get("place") is not None and "device" in (row.get(side) or []):
            reasons.add(COARSE_PLACE)
    for field in row.get("single_per") or []:
        if (
            field.startswith("attributes.")
            and _attribute(claim, field.removeprefix("attributes.")) is None
        ):
            reasons.add(COARSE_KEY_ATTRIBUTE)
    mine = views(claim, types)
    for other in claims_on_key:
        if other is claim or (other.get("id") is not None and other.get("id") == claim.get("id")):
            continue
        for key, keying, object_ in mine:
            for other_key, other_keying, other_object in views(other, types):
                if other_key != key or end_identity(keying) != end_identity(other_keying):
                    continue
                if end_identity(object_) is None or end_identity(object_) != end_identity(
                    other_object
                ):
                    continue
                for slot, reason in (("port", COARSE_PORT), ("unit", COARSE_UNIT)):
                    if object_.get(slot) is None and other_object.get(slot) is not None:
                        reasons.add(reason)
    return [reason for reason in COARSE_REASONS if reason in reasons]


def _allowed(kinds: Sequence[str]) -> str:
    return " or ".join(kinds)


def shape_problems(
    claim: Mapping[str, Any], types: Mapping[str, Mapping[str, Any]]
) -> list[tuple[str, str]]:
    """Every ``relation-invalid`` condition *claim* breaks, as ``(detail, remedy)``.

    The order is fixed: per end (``subject``, then ``object``) no kind or more
    than one, ``unit`` on an end naming no place, a kind the type does not allow
    there; then ``basis: inferred`` without ``rule``; a required attribute left
    out; subject equal to object, port included. A type missing from *types*
    skips the type's checks (the schema stage reports it).

    Args:
        claim: One relation claim.
        types: The relation type rows, keyed by name.

    Returns:
        The problems; empty when the claim is sound.
    """
    rtype = claim.get("type")
    row = types.get(rtype)  # type: ignore[arg-type]
    problems: list[tuple[str, str]] = []
    for side in ("subject", "object"):
        end = _end(claim, side)
        kinds = _kinds(end)
        if not kinds:
            problems.append(
                (
                    f"`{side}` names no device, place or group",
                    f"name one of `device`, `place`, `group` in `{side}`",
                )
            )
        elif len(kinds) > 1:
            problems.append(
                (
                    f"`{side}` names more than one of device, place, group",
                    f"keep one of `device`, `place`, `group` in `{side}`",
                )
            )
        if end.get("unit") is not None and "place" not in kinds:
            problems.append(
                (
                    f"`{side}` gives `unit` on an end that names no place",
                    f"drop `unit` from `{side}`, or name the place it counts in",
                )
            )
        if row is not None:
            allowed = list(row.get(side) or [])
            problems.extend(
                (
                    f"`{side}` names a {kind}; {rtype} allows {_allowed(allowed)} there",
                    f"name a {_allowed(allowed)} as the {side} of {rtype}",
                )
                for kind in kinds
                if kind not in allowed
            )
    if claim.get("basis") == "inferred" and not claim.get("rule"):
        problems.append(
            (
                "`basis` is inferred and `rule` is missing",
                "name the rule that derived the claim in `rule`, or drop `basis: inferred`",
            )
        )
    if row is not None:
        problems.extend(
            (f"{rtype} gives no `attributes.{name}`", f"state the {name} in `attributes.{name}`")
            for name in REQUIRED_ATTRIBUTES.get(str(rtype), ())
            if _attribute(claim, name) is None
        )
    subject, object_ = _end(claim, "subject"), _end(claim, "object")
    if _kinds(subject) and _end_value(subject) == _end_value(object_):
        problems.append(
            (
                "`subject` and `object` name the same end",
                "name two different ends, or tell them apart by `port`",
            )
        )
    return problems


#: The standings a claim can take, best first.
ACCEPTED = "accepted"
CONTESTED = "contested"
SUPERSEDED = "superseded"


def _object_identity(key: Key, keying: Mapping[str, Any], other: Mapping[str, Any]) -> Hashable:
    """What a claim states for *key*: the other end, or both ends when no end keys it."""
    if any(field == "subject" for field, _value in key[1:]):
        return _end_value(other)
    return frozenset((_end_value(keying), _end_value(other)))


#: A claim's rank under a policy: (precedence place, inferred); lower is better.
Rank = tuple[int, int]


def _rank_by(
    policy: Mapping[str, Any] | None, documents: Sequence[Mapping[str, Any]]
) -> Callable[[Mapping[str, Any]], Rank] | None:
    """A function ranking one claim under *policy*, or ``None`` when nothing ranks."""
    if policy is None:
        return None
    relations = policy.get("relations")
    relations = relations if isinstance(relations, Mapping) else {}
    precedence = relations.get("precedence")
    precedence = precedence if isinstance(precedence, Mapping) else {}
    basis_ranks = relations.get("stated_over_inferred", True) is not False
    kinds = {str(doc.get("id")): doc.get("kind") for doc in documents if doc.get("id") is not None}

    def rank(claim: Mapping[str, Any]) -> Rank:
        order = precedence.get(claim.get("type"), precedence.get("default"))
        listed = list(order) if isinstance(order, list) else []
        source = claim.get("source")
        document = source.get("document") if isinstance(source, Mapping) else None
        kind = kinds.get(str(document)) if document is not None else None
        place = listed.index(kind) if kind in listed else len(listed)
        inferred = 1 if basis_ranks and claim.get("basis") == "inferred" else 0
        return (place, inferred)

    return rank


def _claims_by_key(
    claims: Sequence[Mapping[str, Any]], types: Mapping[str, Mapping[str, Any]]
) -> dict[Key, list[tuple[Mapping[str, Any], Hashable]]]:
    """Each key with the claims on it and the object each states there."""
    by_key: dict[Key, list[tuple[Mapping[str, Any], Hashable]]] = {}
    for claim in claims:
        seen: set[Key] = set()
        for key, keying, other in views(claim, types):
            if key in seen:
                continue
            seen.add(key)
            by_key.setdefault(key, []).append((claim, _object_identity(key, keying, other)))
    return by_key


def _disputes(
    claims: Sequence[Mapping[str, Any]],
    types: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any] | None,
    documents: Sequence[Mapping[str, Any]],
) -> dict[Key, dict[str, tuple[str, set[str]]]]:
    """Every disputed key with each non-coarse claim's verdict there and the ids it contests.

    A key is disputed when its non-coarse claims state two or more objects; the
    verdicts follow :func:`compute_standing`.
    """
    rank = _rank_by(policy, documents)
    disputes: dict[Key, dict[str, tuple[str, set[str]]]] = {}
    for key, entries in _claims_by_key(claims, types).items():
        on_key = [claim for claim, _identity in entries]
        groups: dict[Hashable, list[str]] = {}
        ranks: dict[Hashable, Rank] = {}
        for claim, identity in entries:
            if coarse_reasons(claim, on_key, types):
                continue
            groups.setdefault(identity, []).append(str(claim.get("id")))
            if rank is not None:
                mine = rank(claim)
                ranks[identity] = min(ranks.get(identity, mine), mine)
        if len(groups) < 2:
            continue
        best = min(ranks.values()) if rank is not None else None
        tied = rank is None or sum(1 for r in ranks.values() if r == best) > 1
        verdicts: dict[str, tuple[str, set[str]]] = {}
        for identity, members in groups.items():
            if rank is None or ranks[identity] == best:
                verdict = CONTESTED if tied else ACCEPTED
            else:
                verdict = SUPERSEDED
            others = {rid for other, ids in groups.items() if other != identity for rid in ids}
            for rid in members:
                verdicts[rid] = (verdict, others)
        disputes[key] = verdicts
    return disputes


def compute_standing(
    claims: Sequence[Mapping[str, Any]],
    types: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any] | None,
    documents: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, list[str]]]:
    """Each claim's standing and the claims it disagrees with.

    Per key, the claims that are not coarse there group by the object they
    state (port and unit included; both ends, unordered, for a key no end
    names). One group: all ``accepted``. More: without *policy* every group is
    ``contested``; with it each claim ranks by its cited document's kind in
    ``precedence.<type>`` (else ``precedence.default``; an uncited claim, an
    undeclared document or an unlisted kind ranks last, ``undecided`` ranks no
    kind), then stated before inferred unless ``stated_over_inferred`` is false.
    A group ranks by its best claim; the strictly best group is ``accepted``,
    groups tied at the best are ``contested``, the rest ``superseded``. Coarse
    claims are ``accepted`` on the key and take no part. Across keys the worst
    standing holds.

    Args:
        claims: The relation claims, each with an ``id``.
        types: The relation type rows, keyed by name.
        policy: ``policy.yaml`` without its header, or ``None`` when absent.
        documents: The declared documents (``id``, ``kind``).

    Returns:
        Claim id to ``(standing, contests)``, contests the ids of every claim
        naming a different object on one of its keys, sorted. The result does
        not depend on the order of *claims* or *documents*.
    """
    severity = {ACCEPTED: 0, CONTESTED: 1, SUPERSEDED: 2}
    standing: dict[str, str] = {str(claim.get("id")): ACCEPTED for claim in claims}
    contests: dict[str, set[str]] = {rid: set() for rid in standing}
    for verdicts in _disputes(claims, types, policy, documents).values():
        for rid, (verdict, others) in verdicts.items():
            if severity[verdict] > severity[standing[rid]]:
                standing[rid] = verdict
            contests[rid] |= others
    return {rid: (standing[rid], sorted(contests[rid])) for rid in sorted(standing)}


def _contested_warnings(
    disputes: Mapping[Key, Mapping[str, tuple[str, set[str]]]],
) -> list[FacilityBuildWarning]:
    """One ``relation-contested`` line per key a dispute leaves unsettled."""
    found: list[FacilityBuildWarning] = []
    for key, verdicts in disputes.items():
        tied = sorted(rid for rid, (verdict, _others) in verdicts.items() if verdict == CONTESTED)
        if not tied:
            continue
        found.append(
            FacilityBuildWarning(
                kind="relation-contested",
                record_kind=str(key[0]),
                record_id=spell_key(key),
                detail=f"claims {', '.join(tied)} state different ends and none ranks first",
                remedy="rank the kinds of their documents in policy.yaml, or correct the wrong claim",
            )
        )
    return found


def _coarse_warnings(
    claims: Sequence[Mapping[str, Any]], types: Mapping[str, Mapping[str, Any]]
) -> list[FacilityBuildWarning]:
    """One ``relation-coarse`` line per type with coarse claims, counted by reason."""
    sharing: dict[str, list[Mapping[str, Any]]] = {}
    for entries in _claims_by_key(claims, types).values():
        on_key = [claim for claim, _identity in entries]
        for claim in on_key:
            sharing.setdefault(str(claim.get("id")), []).extend(on_key)
    claims_per_type: Counter[str] = Counter()
    reasons_per_type: dict[str, Counter[str]] = {}
    for claim in claims:
        reasons = coarse_reasons(claim, sharing.get(str(claim.get("id")), []), types)
        if not reasons:
            continue
        rtype = str(claim.get("type"))
        claims_per_type[rtype] += 1
        reasons_per_type.setdefault(rtype, Counter()).update(reasons)
    found: list[FacilityBuildWarning] = []
    for rtype, count in claims_per_type.items():
        by_reason = ", ".join(
            f"{reasons_per_type[rtype][reason]} {reason}"
            for reason in COARSE_REASONS
            if reasons_per_type[rtype][reason]
        )
        noun = "claim states" if count == 1 else "claims state"
        found.append(
            FacilityBuildWarning(
                kind="relation-coarse",
                record_kind="relation type",
                record_id=rtype,
                detail=f"{count} {noun} less than the facility knows ({by_reason})",
                remedy="name the device, port, unit or key attribute the claims leave out",
            )
        )
    return found


#: The types whose claims chain records into a whole: a loop over them is a ``part-cycle``.
CYCLE_TYPES: tuple[str, ...] = ("part_of", "runs_on")

#: The class a layer gives a device it knows only as a claim end.
STUB_CLASS = "Unidentified"


def _records(document: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Every place and device as ``(kind, record)``."""
    return [
        (kind, record)
        for kind, section in (("device", "devices"), ("place", "places"))
        for record in document.get(section) or []
        if isinstance(record, Mapping)
    ]


def _imported_layers(record: Mapping[str, Any]) -> set[str]:
    """The imported layers a record's provenance names (``authored`` left out)."""
    provenance = record.get("provenance")
    sources = provenance.get("sources") if isinstance(provenance, Mapping) else None
    return {
        str(source.get("layer"))
        for source in sources or []
        if isinstance(source, Mapping) and source.get("layer") not in (None, AUTHORED)
    }


def _spell_records(records: Sequence[tuple[str, str]]) -> str:
    """``devices A, B`` for one kind; ``device A, place P`` for several."""
    kinds = {kind for kind, _rid in records}
    if len(kinds) == 1:
        (kind,) = kinds
        noun = kind if len(records) == 1 else f"{kind}s"
        return f"{noun} {', '.join(rid for _kind, rid in records)}"
    return ", ".join(f"{kind} {rid}" for kind, rid in records)


def _alias_warnings(document: Mapping[str, Any]) -> list[FacilityBuildWarning]:
    """One ``alias-shared`` line per scheme name two or more records carry."""
    carriers: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for kind, record in _records(document):
        for alias in record.get("scheme_names") or []:
            if isinstance(alias, Mapping) and alias.get("name") is not None:
                name = (str(alias.get("scheme")), str(alias["name"]))
                carriers.setdefault(name, set()).add((kind, str(record.get("id"))))
    return [
        FacilityBuildWarning(
            kind="alias-shared",
            record_kind="scheme",
            record_id=f"{scheme} {name}",
            detail=f"{_spell_records(sorted(records))} carry the name",
            remedy="give each record its own name in the scheme if the scheme is one-to-one",
        )
        for (scheme, name), records in sorted(carriers.items())
        if len(records) > 1
    ]


def _device_edges(
    claims: Sequence[Mapping[str, Any]], rtypes: Sequence[str]
) -> list[tuple[str, str]]:
    """``(subject, object)`` device ids of the non-superseded claims of *rtypes*."""
    edges: set[tuple[str, str]] = set()
    for claim in claims:
        if claim.get("type") not in rtypes or claim.get("standing") == SUPERSEDED:
            continue
        subject = end_identity(_end(claim, "subject"))
        obj = end_identity(_end(claim, "object"))
        if subject and obj and subject[0] == obj[0] == "device":
            edges.add((subject[1], obj[1]))
    return sorted(edges)


def _identity_warnings(
    document: Mapping[str, Any], claims: Sequence[Mapping[str, Any]]
) -> list[FacilityBuildWarning]:
    """One ``identity-ambiguous`` line per same_as component and imported layer it holds twice."""
    parent: dict[str, str] = {}

    def root(rid: str) -> str:
        parent.setdefault(rid, rid)
        while parent[rid] != rid:
            parent[rid] = parent[parent[rid]]
            rid = parent[rid]
        return rid

    for subject, obj in _device_edges(claims, ("same_as",)):
        a, b = root(subject), root(obj)
        if a != b:
            parent[max(a, b)] = min(a, b)
    components: dict[str, list[str]] = {}
    for rid in sorted(parent):
        components.setdefault(root(rid), []).append(rid)
    devices = {
        str(record.get("id")): record for kind, record in _records(document) if kind == "device"
    }
    found: list[FacilityBuildWarning] = []
    for members in sorted(components.values()):
        held: dict[str, list[str]] = {}
        for rid in members:
            for layer in _imported_layers(devices.get(rid, {})):
                held.setdefault(layer, []).append(rid)
        for layer, twice in sorted(held.items()):
            if len(twice) < 2:
                continue
            found.append(
                FacilityBuildWarning(
                    kind="identity-ambiguous",
                    record_kind="layer",
                    record_id=layer,
                    detail=(
                        f"same_as joins devices {', '.join(members)} into one, "
                        f"and this layer states {', '.join(twice)} apart"
                    ),
                    remedy="correct the same_as claims or the import that describes one device twice",
                )
            )
    return found


def _cycles(edges: Sequence[tuple[str, str]]) -> list[list[str]]:
    """Every elementary cycle once, starting at its smallest id, sorted."""
    successors: dict[str, list[str]] = {}
    for subject, obj in edges:
        successors.setdefault(subject, []).append(obj)
    found: list[list[str]] = []

    def walk(start: str, path: list[str]) -> None:
        for nxt in sorted(successors.get(path[-1], [])):
            if nxt == start:
                found.append([*path, start])
            elif nxt > start and nxt not in path:
                walk(start, [*path, nxt])

    for start in sorted(successors):
        walk(start, [start])
    return sorted(found)


def _cycle_warnings(claims: Sequence[Mapping[str, Any]]) -> list[FacilityBuildWarning]:
    """One ``part-cycle`` line per loop of non-superseded part_of and runs_on claims."""
    return [
        FacilityBuildWarning(
            kind="part-cycle",
            record_kind="device",
            record_id=cycle[0],
            detail=f"part_of and runs_on claims loop {' -> '.join(cycle)}",
            remedy="correct the claim that closes the loop",
        )
        for cycle in _cycles(_device_edges(claims, CYCLE_TYPES))
    ]


def _stub_warnings(
    document: Mapping[str, Any], claims: Sequence[Mapping[str, Any]]
) -> list[FacilityBuildWarning]:
    """One ``relation-stub`` line per imported layer with Unidentified devices claims name."""
    named = {
        identity[1]
        for claim in claims
        for side in ("subject", "object")
        if (identity := end_identity(_end(claim, side))) and identity[0] == "device"
    }
    per_layer: Counter[str] = Counter()
    for kind, record in _records(document):
        if (
            kind == "device"
            and record.get("class") == STUB_CLASS
            and str(record.get("id")) in named
        ):
            per_layer.update(_imported_layers(record))
    found: list[FacilityBuildWarning] = []
    for layer, count in sorted(per_layer.items()):
        noun = "device" if count == 1 else "devices"
        verb = "stands" if count == 1 else "stand"
        found.append(
            FacilityBuildWarning(
                kind="relation-stub",
                record_kind="layer",
                record_id=layer,
                detail=f"{count} {noun} of class {STUB_CLASS} {verb} only for claim ends",
                remedy="import the devices the claims name, or name known devices",
            )
        )
    return found


def fill_relation_slots(validated: Validated) -> list[FacilityBuildError]:
    """Write each relation claim's ``standing`` and ``contests``; warn of what the claims leave doubtful.

    ``standing`` is always written, ``contests`` only when it names a claim;
    each filled slot is recorded in the claim's ``provenance.defaults``. The
    stage appends one ``relation-contested`` warning per key a dispute leaves
    unsettled, one ``relation-coarse`` warning per type with coarse claims, one
    ``alias-shared`` per scheme name two records carry, one
    ``identity-ambiguous`` per same_as component holding two records of one
    imported layer, one ``part-cycle`` per loop of part_of and runs_on claims
    and one ``relation-stub`` per imported layer with Unidentified devices that
    claims name, sorted by (kind, record kind, record id). Superseded claims
    join no component and close no loop. A file without claims is left as it
    is and can only warn ``alias-shared``.

    Args:
        validated: What the earlier stages produced.

    Returns:
        No stops, ever.

    Raises:
        RuntimeError: The stage ran before S2 produced the document.
    """
    from osprey.facility.validate import need, relation_types

    document: dict[str, Any] = need(validated.document)
    claims: list[dict[str, Any]] = document.get("relations") or []
    if not claims:
        _emit(validated, _alias_warnings(document))
        return []
    types = relation_types()
    policy = validated.sources.policy if validated.sources is not None else None
    documents = document.get("documents") or []
    standings = compute_standing(claims, types, policy, documents)
    for claim in claims:
        standing, contests = standings[str(claim.get("id"))]
        slots: dict[str, Any] = {"standing": standing}
        if contests:
            slots["contests"] = contests
        claim.update(slots)
        claim["provenance"] = add_defaults(claim.get("provenance", {}), slots)
    warnings = [
        *_contested_warnings(_disputes(claims, types, policy, documents)),
        *_coarse_warnings(claims, types),
        *_alias_warnings(document),
        *_identity_warnings(document, claims),
        *_cycle_warnings(claims),
        *_stub_warnings(document, claims),
    ]
    _emit(validated, warnings)
    return []


def _emit(validated: Validated, warnings: Sequence[FacilityBuildWarning]) -> None:
    validated.warnings.extend(sorted(warnings, key=lambda w: (w.kind, w.record_kind, w.record_id)))
