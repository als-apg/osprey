"""The example facility's prose names only devices and addresses the built facility holds.

The prose is the scenario logbook entries (``title`` and ``text``) and the
knowledge pages. Two grammars find the names in it:

- A family-plus-index token (``BPM17``, ``BPM 17``, ``bpm-17``) resolves when
  some device of that family carries the index. The families are the ones the
  built facility's devices carry, never a list kept here.
- An address (three or more ``:``-separated segments) resolves when it is a
  channel id; a wildcard address (``*`` or ``{...}``) resolves when it matches
  at least one channel id.

The pages in ``KNOWN_UNRESOLVED`` name addresses from a scheme the facility does
not serve. That list only shrinks: a new unresolved token fails, and so does a
pinned token that resolves or has left its page.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FACILITY = REPO_ROOT / "src/osprey/templates/facilities/example"

#: The scenarios whose YAML holds a non-empty ``logbook``, pinned by name so the
#: scan cannot pass on an empty set.
LOGBOOK_SCENARIOS: frozenset[str] = frozenset({"bpm-polarity", "nominal", "rf-thermal"})

#: Three or more ``:``-separated segments starting with an uppercase letter.
#: Times (``07:15:00``) and URLs (``http://``) are not addresses.
ADDRESS = re.compile(r"\b[A-Z][A-Z0-9]*(?::[A-Za-z0-9_*{}]+){2,}")

#: A device name split into its family and its index (``BPM 01`` -> ``BPM``, ``1``).
_LEAF = re.compile(r"([A-Za-z]+)[-_ ]?0*(\d+)")

#: Per knowledge page, the addresses it names that the facility does not serve.
KNOWN_UNRESOLVED: dict[str, frozenset[str]] = {
    "devices/bpm.md": frozenset(
        {
            "BPM:{ID}:{param}",
            "TD:BPM:H01:Quality",
            "TD:BPM:H01:Sum",
            "TD:BPM:H01:X",
            "TD:BPM:H01:Y",
        }
    ),
    "devices/ion-pump.md": frozenset(
        {
            "TD:VAC:IP01:Current",
            "TD:VAC:IP01:Enable",
            "TD:VAC:IP01:Fault",
            "TD:VAC:IP01:Voltage",
            "VAC:IP{N}:{param}",
            "VAC:{Section}:Pressure",
        }
    ),
    "physics/beam-current-calibration.md": frozenset(
        {
            "PS:DCCT:Cal:Date",
            "PS:DCCT:Cal:Gain",
            "PS:DCCT:Cal:Offset",
            "PS:DCCT:Current",
        }
    ),
    "physics/quadrupole-scan.md": frozenset(
        {
            "PS:DCCT:Current",
            "TD:BPM:H02:Sum",
            "TD:BPM:H02:X",
            "TD:BPM:H02:Y",
            "TD:QF:QF01:CURRENT:SP",
        }
    ),
    "procedures/orbit-correction.md": frozenset(
        {
            "PS:DCCT:Current",
            "TD:BPM:H01:X",
            "TD:BPM:H01:Y",
            "TD:BPM:H02:X",
            "TD:BPM:H02:Y",
            "TD:HCM:*:CURRENT",
            "TD:VCM:*:CURRENT",
        }
    ),
    "procedures/ps-startup.md": frozenset(
        {
            "PS:DCCT:Current",
            "PS:GATE:INJ:Cmd",
            "PS:MAG:SOL01:CURRENT:SP",
            "PS:PWR:Main:Status",
            "PS:RF:CAV01:Phase:SP",
            "PS:RF:CAV01:Power:SP",
            "PS:RF:LLRF:Enable",
            "PS:VAC:GV01:State",
            "PS:VAC:INJ:Pressure",
            "TIM:EVT:Injection:Ena",
            "TIM:MODE:Current",
        }
    ),
    "procedures/pss-reset.md": frozenset(
        {
            "PSS:Global:BeamInhibit",
            "PSS:{Area}:BeamPermit",
            "PSS:{Area}:Door:State",
            "PSS:{Area}:Interlock",
            "PSS:{Area}:Search:Status",
        }
    ),
    "procedures/sample-scan.md": frozenset(
        {
            "MOT:X:Position",
            "MOT:X:Position:SP",
            "MOT:X:Status",
            "PS:DCCT:Current",
            "PSS:ES{N}:BeamPermit",
        }
    ),
    "procedures/vacuum-recovery.md": frozenset(
        {
            "PS:VAC:RP01:Enable",
            "PS:VAC:TMP01:Enable",
            "PSS:Global:BeamInhibit",
            "VAC:GV01:Cmd",
            "VAC:GV01:State",
            "VAC:IP01:Enable",
            "VAC:IP01:Voltage",
        }
    ),
    "references/epics-channel-access.md": frozenset(
        {
            "ES3:SHUTTER:Cmd",
            "ES3:SHUTTER:State",
            "MOT:{Axis}:Position",
            "PS:DCCT:Current",
            "PS:RF:CAV01:Power:SP",
            "TD:BPM:H01:Quality",
        }
    ),
    "references/safety-rules.md": frozenset(
        {
            "MOT:*:Position:SP",
            "PS:DCCT:Current",
            "PS:RF:*:Power:SP",
            "PSS:*:BeamPermit",
            "PSS:*:Interlock",
            "PSS:Global:BeamInhibit",
            "TD:BPM:*:X",
            "VAC:*:Pressure",
        }
    ),
    "subsystems/experimental-stations.md": frozenset(
        {
            "ES3:SHUTTER:Cmd",
            "ES3:SHUTTER:State",
            "MOT:{Axis}:Position",
            "MOT:{Axis}:Position:SP",
            "MOT:{Axis}:Status",
        }
    ),
    "subsystems/primary-source.md": frozenset(
        {
            "PS:BPM:*:X",
            "PS:DCCT:Current",
            "PS:MAG:*:CURRENT",
            "PS:RF:*:Phase",
            "PS:RF:*:Power",
            "PS:RF:CAV01:Phase:SP",
            "PS:RF:CAV01:Power:SP",
            "PS:VAC:*:Pressure",
        }
    ),
    "subsystems/pss.md": frozenset(
        {
            "PSS:Global:BeamInhibit",
            "PSS:{Area}:BeamPermit",
            "PSS:{Area}:Door:State",
            "PSS:{Area}:Interlock",
            "PSS:{Area}:Search:Status",
        }
    ),
    "subsystems/timing.md": frozenset(
        {
            "TIM:CLOCK:Freq",
            "TIM:EVT:DataAcq",
            "TIM:EVT:DataAcq:Delay",
            "TIM:EVT:Injection",
            "TIM:EVT:Injection:Ena",
            "TIM:EVT:Sync",
            "TIM:MODE:Current",
        }
    ),
    "subsystems/transport-delivery.md": frozenset(
        {
            "TD:BM:*:CURRENT",
            "TD:BPM:*:X",
            "TD:BPM:*:Y",
            "TD:BPM:H01:X",
            "TD:BPM:H01:Y",
            "TD:HCM:*:CURRENT",
            "TD:QD:*:CURRENT",
            "TD:QF:*:CURRENT",
            "TD:SW:*:Position",
            "TD:SW:01:Position",
            "TD:SW:01:Position:SP",
            "TD:VCM:*:CURRENT",
        }
    ),
    "subsystems/vacuum.md": frozenset(
        {
            "ES3:VAC:End:Pressure",
            "PS:VAC:INJ:Pressure",
            "PS:VAC:RP01:*",
            "PS:VAC:TMP01:*",
            "TD:VAC:GV01:Cmd",
            "TD:VAC:GV01:State",
            "TD:VAC:T01:Pressure",
            "VAC:IP{N}:*",
            "VAC:{Section}:Pressure",
        }
    ),
}


def families(doc: Mapping[str, Any]) -> dict[str, frozenset[int]]:
    """Map each device family to the indices its devices carry.

    Args:
        doc: A facility document with a ``devices`` list.

    Returns:
        The upper-cased family mapped to every index some device of it carries,
        read from the id's last ``/`` segment, the ``names`` and the ``label``.
    """
    held: dict[str, set[int]] = {}
    for device in doc.get("devices", []):
        spellings = [device["id"].rsplit("/", 1)[-1], *device.get("names", [])]
        if device.get("label"):
            spellings.append(device["label"])
        for spelling in spellings:
            match = _LEAF.fullmatch(spelling)
            if match:
                held.setdefault(match.group(1).upper(), set()).add(int(match.group(2)))
    return {family: frozenset(indices) for family, indices in sorted(held.items())}


def _family_pattern(held: Mapping[str, frozenset[int]]) -> re.Pattern[str] | None:
    """The family-plus-index grammar over ``held``'s families, longest first."""
    if not held:
        return None
    alternation = "|".join(re.escape(f) for f in sorted(held, key=lambda f: (-len(f), f)))
    return re.compile(rf"\b({alternation})[-_ ]?(\d+)\b", re.IGNORECASE)


def family_tokens(text: str, held: Mapping[str, frozenset[int]]) -> list[str]:
    """Every family-plus-index token in ``text``.

    Args:
        text: The prose to scan.
        held: The families to look for, as :func:`families` returns them.

    Returns:
        Each matched token, verbatim, in the order it appears.
    """
    pattern = _family_pattern(held)
    if pattern is None:
        return []
    return [match.group(0) for match in pattern.finditer(text)]


def resolves_device(token: str, held: Mapping[str, frozenset[int]]) -> bool:
    """Whether some device of the token's family carries the token's index.

    Args:
        token: A family-plus-index token.
        held: The families and their indices, as :func:`families` returns them.

    Returns:
        True when the index is one the family's devices carry.
    """
    pattern = _family_pattern(held)
    match = pattern.fullmatch(token) if pattern else None
    if match is None:
        return False
    return int(match.group(2)) in held[match.group(1).upper()]


def resolves_address(token: str, served: frozenset[str]) -> bool:
    """Whether an address names a served channel.

    Args:
        token: An address, possibly holding ``*`` or ``{...}`` wildcards.
        served: Every channel id the facility holds.

    Returns:
        True when the plain address is served, or the wildcard address matches
        at least one served id.
    """
    if "*" not in token and "{" not in token:
        return token in served
    glob = re.sub(r"\{[^}]*\}", "*", token)
    return any(fnmatch.fnmatchcase(address, glob) for address in served)


def unresolved(text: str, doc: Mapping[str, Any]) -> list[str]:
    """The tokens in ``text`` that name nothing the facility holds.

    Args:
        text: The prose to scan.
        doc: The facility document, with ``devices`` and ``channels``.

    Returns:
        The sorted, de-duplicated device and address tokens that do not resolve.
    """
    held = families(doc)
    served = frozenset(channel["id"] for channel in doc.get("channels", []))
    missing = {t for t in family_tokens(text, held) if not resolves_device(t, held)}
    missing |= {t for t in ADDRESS.findall(text) if not resolves_address(t, served)}
    return sorted(missing)


def logbook_texts() -> dict[str, str]:
    """Every scenario logbook entry's prose, keyed ``scenarios/<file>#<entry_id>``.

    Returns:
        The entry's title and text, joined by a newline.
    """
    texts: dict[str, str] = {}
    for path in sorted((FACILITY / "scenarios").glob("*.yaml")):
        scenario = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in scenario.get("logbook") or []:
            texts[f"scenarios/{path.name}#{entry['entry_id']}"] = (
                f"{entry['title']}\n{entry['text']}"
            )
    return texts


def page_texts() -> dict[str, str]:
    """Every knowledge page's text, keyed by its posix path under ``knowledge/``.

    Returns:
        The whole file per page.
    """
    root = FACILITY / "knowledge"
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.md"))
    }


# --- the grammars, on a synthetic document -----------------------------------------

DOC: dict[str, Any] = {
    "devices": [
        {"id": "SR/BPM01", "names": ["BPM01"], "label": "BPM 01"},
        {"id": "SR/QF1"},
        {"id": "SR/QFA2"},
    ],
    "channels": [
        {"id": "SR:DIAG:BPM:01:POSITION:X"},
        {"id": "SR:DIAG:BPM:01:POSITION:Y"},
    ],
}


def test_an_index_no_device_of_the_family_carries_is_named() -> None:
    assert unresolved("checked BPM99", DOC) == ["BPM99"]


def test_a_held_device_resolves_in_any_spelling() -> None:
    for text in ("BPM01", "BPM 1", "bpm-01", "QF01", "QFA2"):
        assert unresolved(text, DOC) == [], text


def test_an_address_the_facility_does_not_serve_is_named() -> None:
    assert unresolved("read TD:BPM:H01:X", DOC) == ["TD:BPM:H01:X"]


def test_a_wildcard_address_resolves_when_it_matches_a_served_address() -> None:
    assert unresolved("SR:DIAG:BPM:*:POSITION:X", DOC) == []
    assert unresolved("SR:DIAG:BPM:{N}:POSITION:Y", DOC) == []
    assert unresolved("check PSS:{Area}:BeamPermit", DOC) == ["PSS:{Area}:BeamPermit"]


def test_words_without_an_index_and_times_are_not_tokens() -> None:
    assert unresolved("the quadrupoles, BPM readings at 07:15:00 (Sector 7)", DOC) == []


# --- the scenario logbooks ----------------------------------------------------------


def test_the_scan_reads_the_logbook_of_every_narrating_scenario() -> None:
    scanned = {key.split("#")[0] for key in logbook_texts()}
    assert scanned == {f"scenarios/{name}.yaml" for name in LOGBOOK_SCENARIOS}


def test_every_logbook_entry_names_only_devices_and_addresses_the_facility_holds(
    built_control_assistant: Any,
) -> None:
    facility = built_control_assistant.facility
    found = {key: unresolved(text, facility) for key, text in logbook_texts().items()}
    found = {key: tokens for key, tokens in found.items() if tokens}
    assert found == {}, f"logbook entries name what the facility does not hold: {found}"


# --- the knowledge pages ------------------------------------------------------------


def test_the_scan_reads_every_knowledge_page() -> None:
    procedures = sorted(FACILITY.glob("knowledge/procedures/*.md"))
    assert procedures
    pages = page_texts()
    assert {f"procedures/{path.name}" for path in procedures} <= set(pages)


def test_no_knowledge_page_names_an_unresolved_token_beyond_its_pinned_set(
    built_control_assistant: Any,
) -> None:
    facility = built_control_assistant.facility
    beyond = {
        page: sorted(set(unresolved(text, facility)) - KNOWN_UNRESOLVED.get(page, frozenset()))
        for page, text in page_texts().items()
    }
    beyond = {page: tokens for page, tokens in beyond.items() if tokens}
    assert beyond == {}, f"knowledge pages name what the facility does not hold: {beyond}"


def test_every_pinned_unresolved_token_is_still_unresolved_on_its_page(
    built_control_assistant: Any,
) -> None:
    facility = built_control_assistant.facility
    pages = page_texts()
    stale: dict[str, list[str]] = {}
    for page, pinned in KNOWN_UNRESOLVED.items():
        found = set(unresolved(pages[page], facility)) if page in pages else set()
        if pinned - found:
            stale[page] = sorted(pinned - found)
    assert stale == {}, f"delete these KNOWN_UNRESOLVED entries, they no longer apply: {stale}"
