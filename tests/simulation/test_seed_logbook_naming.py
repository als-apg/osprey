"""Standing drift guard for the harmonized ARIEL seed-logbook prose.

The five shipped seed logbooks (four control_assistant scenario bundles plus the
ariel_standalone demo seed) narrate device activity in prose. Phase 3 harmonized
that prose onto the flat ``^{FAM}{NN}$`` naming of the one ring
(:data:`osprey.simulation.facility_spec.ALS_U_AR`): ``DIPOLE-07`` became
``DIPOLE07``, ``cavity C1`` became ``CAVITY01``, ``PS-QF-08`` became a ``QF08``
prose reference, and channel-less legacy designators (``ID-07``, ``BLM-09C``,
``HCM-TL04`` …) were rewritten as generic prose.

This module is the *standing guard* that keeps that harmonization from silently
regressing. It is hermetic: it reads only committed repo files (the seed globs
and the committed tier-3 channel DB), never a database or the network. Every
assertion is a deterministic, case-insensitive regex scan:

* the glob resolves to exactly the five known seed files (an empty or shrunken
  glob fails loudly, so a moved/renamed seed cannot slip the guard);
* every family-token + designator reference (over the spec families *and* the
  non-spec tier-3 families) is the canonical ``FAM`` + two-digit id, with the id
  in range;
* no bare ``C\\d+`` cavity device token survives;
* no legacy ``XXX-\\d+`` designator survives.

The family/id ranges are derived, not pinned: spec families come from
``ALS_U_AR`` counts, non-spec families from the committed tier-3 device list, so
the guard tracks the spec as it evolves. Bare family words without a designator
(``the quadrupoles``, ``BPM readings``) are legal prose and are not flagged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from osprey.simulation.facility_spec import ALS_U_AR

# Repo root: tests/simulation/<this file> -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The two seed locations the guard sweeps (glob, relative to the repo root).
_SEED_GLOBS = (
    "src/osprey/templates/apps/control_assistant/data/simulation/scenarios/*/logbook.json",
    "src/osprey/templates/apps/ariel_standalone/data/logbook_seed/*.json",
)

# The exact seed set the sweep MUST resolve to. Pinned by name so an empty or
# shrunken glob (a moved/renamed/deleted seed) fails loudly instead of vacuously
# passing. Note: scenarios/vacuum-burst is telemetry-only (no logbook.json) and
# is intentionally absent.
_EXPECTED_SEEDS = frozenset(
    {
        "src/osprey/templates/apps/control_assistant/data/simulation/scenarios/bpm-polarity/logbook.json",
        "src/osprey/templates/apps/control_assistant/data/simulation/scenarios/errant-quad/logbook.json",
        "src/osprey/templates/apps/control_assistant/data/simulation/scenarios/nominal/logbook.json",
        "src/osprey/templates/apps/control_assistant/data/simulation/scenarios/rf-thermal/logbook.json",
        "src/osprey/templates/apps/ariel_standalone/data/logbook_seed/demo_logbook.json",
    }
)

# Canonical tier-3 channel DB — source of the non-spec family device lists.
_TIER3_IN_CONTEXT = (
    _REPO_ROOT
    / "src/osprey/templates/apps/control_assistant/data/channel_databases/tiers/tier3/in_context.json"
)

# Designator that follows a family token: a plain number, or a legacy ``C``-id
# (``C1``/``C2`` cavity token) that harmonization was meant to retire.
_DESIGNATOR = r"(?:\d+|C\d+)"

# Bare cavity device token (``C1``, ``C2``) — must not survive harmonization.
_BARE_C_RE = re.compile(r"\bC\d+\b", re.IGNORECASE)

# Legacy channel-less designators that harmonization rewrote to generic prose.
# First clause: ``K-01``, ``BL-03``, ``BLM-09C`` (trailing letter), ``COMP-02``,
# ``SCU-01``, ``TDR-01``, ``DAC-014``, ``INJ-01``, ``ID-07``. Second clause:
# transfer-line / injection steerers (``HCM-TL04``, ``BPM-INJ-02``).
_LEGACY_RE = re.compile(
    r"\b(?:K|BL|BLM|COMP|SCU|TDR|DAC|INJ|ID)-\d+[A-Z]?\b"
    r"|\b(?:HCM|VCM|BPM)-(?:TL|INJ)-?\d+\b",
    re.IGNORECASE,
)


def _load_family_valid_ids() -> dict[str, set[int]]:
    """Return ``{family_token: {valid device ids}}`` for the guard.

    Spec families use ``ALS_U_AR`` counts (ids ``1..count``); non-spec families
    use the committed tier-3 device list (the actual set of device ids present
    in the tier-3 channel DB).
    """
    spec_counts = ALS_U_AR.counts()
    valid: dict[str, set[int]] = {
        fam: set(range(1, count + 1)) for fam, count in spec_counts.items()
    }

    nonspec: dict[str, set[int]] = {}
    channels = json.loads(_TIER3_IN_CONTEXT.read_text())["channels"]
    for chan in channels:
        parts = chan["address"].split(":")
        if len(parts) < 4:
            continue
        fam = parts[2]
        if fam in spec_counts:
            continue
        # Device index is the trailing digit run of the id field (``01`` in
        # ``BPM:01``, ``SR01`` -> ``1`` for ring-prefixed gauge/valve ids).
        m = re.search(r"\d+$", parts[3])
        if m:
            nonspec.setdefault(fam, set()).add(int(m.group()))
    valid.update(nonspec)
    return valid


_FAMILY_VALID_IDS = _load_family_valid_ids()


def _family_pattern(family: str) -> re.Pattern[str]:
    """Regex matching a (possibly non-canonical) reference to ``family``.

    Catches an optional ``PS-`` power-supply prefix, an optional ``-``/``_``/space
    separator, and either a numeric or ``C``-prefixed designator — i.e. exactly
    the non-canonical forms (``PS-QF-08``, ``QF 08``, ``cavity C1``) the guard
    must reject alongside the canonical ``QF08``.
    """
    return re.compile(
        r"\b(PS-)?(" + re.escape(family) + r")[-_ ]?(" + _DESIGNATOR + r")\b",
        re.IGNORECASE,
    )


def _family_designator_violations(text: str, family_valid_ids: dict[str, set[int]]) -> list[str]:
    """Return the non-canonical / out-of-range family references in ``text``.

    A reference is clean iff the whole matched token equals ``f"{FAM}{id:02d}"``
    (canonical family casing, no prefix, no separator, two-digit zero-padded id)
    and the id is in that family's valid range. Everything else — a ``PS-``
    prefix, a ``-``/space separator, a ``C``-designator, lowercase family casing,
    an out-of-range id — is a violation.
    """
    violations: list[str] = []
    for family, valid_ids in family_valid_ids.items():
        canonical_family = family
        for match in _family_pattern(family).finditer(text):
            whole = match.group(0)
            number = int(re.search(r"\d+", match.group(3)).group())
            canonical = f"{canonical_family}{number:02d}"
            if whole != canonical or number not in valid_ids:
                violations.append(whole)
    return violations


def _seed_files() -> list[Path]:
    """Return the seed logbook files the two globs resolve to, sorted."""
    found: list[Path] = []
    for pattern in _SEED_GLOBS:
        found.extend(_REPO_ROOT.glob(pattern))
    return sorted(found)


def _relative(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def seed_files() -> list[Path]:
    return _seed_files()


# ── The glob-integrity gate ─────────────────────────────────────────────────


def test_glob_resolves_to_exact_seed_set(seed_files: list[Path]) -> None:
    """The sweep must cover exactly the five known seed files, by name."""
    resolved = {_relative(p) for p in seed_files}
    assert resolved == set(_EXPECTED_SEEDS), (
        "seed glob drifted from the pinned five-file set: "
        f"missing={set(_EXPECTED_SEEDS) - resolved}, unexpected={resolved - set(_EXPECTED_SEEDS)}"
    )
    # Redundant with the equality above, but pins the intent: never vacuous.
    assert len(seed_files) == len(_EXPECTED_SEEDS)


# ── The prose drift gates ────────────────────────────────────────────────────


def test_family_designators_are_canonical(seed_files: list[Path]) -> None:
    """Every family-token+designator reference is canonical and in range."""
    offenders: dict[str, list[str]] = {}
    for path in seed_files:
        violations = _family_designator_violations(path.read_text(), _FAMILY_VALID_IDS)
        if violations:
            offenders[_relative(path)] = sorted(set(violations))
    assert not offenders, (
        "non-canonical family device references in the harmonized seeds "
        f"(want ^{{FAM}}{{NN}} in range): {offenders}"
    )


def test_no_bare_cavity_device_token(seed_files: list[Path]) -> None:
    """No bare ``C\\d+`` cavity device token survives harmonization."""
    offenders: dict[str, list[str]] = {}
    for path in seed_files:
        hits = sorted({m.group(0) for m in _BARE_C_RE.finditer(path.read_text())})
        if hits:
            offenders[_relative(path)] = hits
    assert not offenders, f"bare C-number cavity tokens remain: {offenders}"


def test_no_legacy_designator(seed_files: list[Path]) -> None:
    """No legacy ``XXX-\\d+`` channel-less designator survives harmonization."""
    offenders: dict[str, list[str]] = {}
    for path in seed_files:
        hits = sorted({m.group(0) for m in _LEGACY_RE.finditer(path.read_text())})
        if hits:
            offenders[_relative(path)] = hits
    assert not offenders, f"legacy device designators remain: {offenders}"


# ── Self-tests: prove the guard logic actually bites ─────────────────────────


def test_guard_flags_qf08_regression() -> None:
    """A seeded ``QF-08`` is flagged; the canonical ``QF08`` is not."""
    assert _family_designator_violations("replaced quadrupole QF-08 supply", _FAMILY_VALID_IDS)
    assert not _family_designator_violations("replaced quadrupole QF08 supply", _FAMILY_VALID_IDS)


def test_guard_flags_ps_prefixed_and_out_of_range() -> None:
    """The ``PS-`` prefix and an out-of-range id are both violations."""
    assert _family_designator_violations("locked out PS-QF08", _FAMILY_VALID_IDS)
    # QF has 24 devices; QF99 is out of range even though it is otherwise canonical.
    assert _family_designator_violations("checked QF99", _FAMILY_VALID_IDS)


def test_guard_flags_bare_c_and_legacy_tokens() -> None:
    """The bare-C and legacy-designator scans bite on seeded regressions."""
    assert _BARE_C_RE.search("rf cavity C1 tripped")
    assert _LEGACY_RE.search("beam loss monitor BLM-09C")
    assert _LEGACY_RE.search("transfer-line steerer HCM-TL04")
    # Bare family words without a designator are legal prose, not flagged.
    assert not _BARE_C_RE.search("the storage-ring cavities were retuned")
