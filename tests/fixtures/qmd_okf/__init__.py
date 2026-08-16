"""Curated OKF query fixtures for ranked-search acceptance testing.

This package pairs a small checked-in OKF bundle (``bundle/``, an ALS-style
facility knowledge base of 18 concepts) with a curated set of
``(query, expected_concept_id)`` pairs in :data:`OKF_QUERY_FIXTURES`.

Every fixture is chosen because the **substring** implementation of
``OKFBundle.search()`` does not return the expected concept at rank 1.  Each
fixture records *how* it defeats substring search in its
:attr:`OKFQueryFixture.defeat_mode` field:

* ``NO_SUBSTRING_MATCH`` — a paraphrase query whose literal text appears in no
  document, so the substring scan returns nothing at all.
* ``WRONG_CONCEPT_FIRST`` — a jargon-mismatch query whose literal text appears
  in some *other* concept that sorts earlier in filesystem traversal order, so
  the substring scan confidently returns the wrong document first.

``tests/services/facility_knowledge/test_okf_search_fixtures.py`` asserts that
construction rule against the live bundle, which is what makes the downstream
acceptance criterion — "qmd ranks the expected concept in its top 3" — a
mechanical check rather than a judgement call.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from osprey.services.facility_knowledge.okf.bundle import OKFBundle

#: Root of the checked-in OKF bundle these fixtures are written against.
BUNDLE_ROOT: Path = Path(__file__).parent / "bundle"


class DefeatMode(StrEnum):
    """Why a fixture query defeats substring search at rank 1.

    Attributes:
        NO_SUBSTRING_MATCH: The query text appears verbatim in no document, so
            the substring scan returns an empty result list.
        WRONG_CONCEPT_FIRST: The query text appears verbatim in a decoy concept
            that sorts before the expected one, so the substring scan returns
            the wrong concept at rank 1.
    """

    NO_SUBSTRING_MATCH = "no_substring_match"
    WRONG_CONCEPT_FIRST = "wrong_concept_first"


@dataclass(frozen=True)
class OKFQueryFixture:
    """One curated query and the concept a good ranker must surface for it.

    Args:
        query: The natural-language or jargon query to issue.
        expected_concept_id: OKF §2 concept ID that correctly answers *query*.
        defeat_mode: How this query defeats the substring scan at rank 1.
        decoy_concept_id: For :attr:`DefeatMode.WRONG_CONCEPT_FIRST`, the
            concept the substring scan returns instead.  Empty for
            :attr:`DefeatMode.NO_SUBSTRING_MATCH`.
        rationale: Why the expected concept is the right answer — the human
            judgement the mechanical assertions cannot make.
    """

    query: str
    expected_concept_id: str
    defeat_mode: DefeatMode
    decoy_concept_id: str
    rationale: str


OKF_QUERY_FIXTURES: tuple[OKFQueryFixture, ...] = (
    OKFQueryFixture(
        query="steering magnets",
        expected_concept_id="magnets/corrector_magnets",
        defeat_mode=DefeatMode.WRONG_CONCEPT_FIRST,
        decoy_concept_id="diagnostics/beam_position_monitors",
        rationale=(
            "'Steering magnet' is the operator word for a corrector dipole, but the "
            "corrector document only ever calls them correctors and trim dipoles. "
            "The phrase appears verbatim in the position-monitor document, which "
            "merely mentions what the orbit loop drives."
        ),
    ),
    OKFQueryFixture(
        query="chromaticity correction",
        expected_concept_id="magnets/sextupoles",
        defeat_mode=DefeatMode.WRONG_CONCEPT_FIRST,
        decoy_concept_id="magnets/quadrupoles",
        rationale=(
            "Sextupoles are the chromaticity magnets, but the document describes "
            "the job physically ('cancel the energy dependence of quadrupole "
            "focusing') and never names it. The quadrupole document mentions the "
            "phrase only as something that happens before a tune re-optimization."
        ),
    ),
    OKFQueryFixture(
        query="beam abort",
        expected_concept_id="safety/machine_protection",
        defeat_mode=DefeatMode.WRONG_CONCEPT_FIRST,
        decoy_concept_id="diagnostics/beam_loss_monitors",
        rationale=(
            "The protection chain is what actually aborts the beam, but it "
            "describes the action as opening the chain and firing the dump kicker. "
            "The loss-monitor document uses the words 'beam abort' for the request "
            "it sends to that chain."
        ),
    ),
    OKFQueryFixture(
        query="PV name",
        expected_concept_id="controls/channel_naming",
        defeat_mode=DefeatMode.WRONG_CONCEPT_FIRST,
        decoy_concept_id="beamlines/insertion_devices",
        rationale=(
            "The naming convention is the answer, but it consistently says "
            "'channel' where a user would say 'PV'. The insertion-device document "
            "uses the literal phrase in passing while giving a gap-motor example."
        ),
    ),
    OKFQueryFixture(
        query="cavity tuner loop",
        expected_concept_id="rf/low_level_rf_control",
        defeat_mode=DefeatMode.WRONG_CONCEPT_FIRST,
        decoy_concept_id="rf/accelerating_cavities",
        rationale=(
            "The plunger regulation loop lives in the low-level chain, which calls "
            "it the third, mechanical loop. The cavity document names the loop but "
            "only to say the plunger exists."
        ),
    ),
    OKFQueryFixture(
        query="pressure readback",
        expected_concept_id="vacuum/ion_pumps",
        defeat_mode=DefeatMode.WRONG_CONCEPT_FIRST,
        decoy_concept_id="safety/machine_protection",
        rationale=(
            "Ion-pump current is the facility's per-sector pressure gauge, but the "
            "document explains that through discharge current and residual-gas "
            "density. The protection document lists 'pressure readback' as one of "
            "its interlock inputs."
        ),
    ),
    OKFQueryFixture(
        query="lifetime measurement",
        expected_concept_id="diagnostics/current_monitor",
        defeat_mode=DefeatMode.WRONG_CONCEPT_FIRST,
        decoy_concept_id="diagnostics/beam_loss_monitors",
        rationale=(
            "Lifetime is measured by fitting the stored-current decay, which is "
            "exactly what the current-monitor document describes without using the "
            "word. The loss-monitor document mentions the phrase only as a "
            "neighbouring column in the shift report."
        ),
    ),
    OKFQueryFixture(
        query="focusing strength",
        expected_concept_id="magnets/quadrupoles",
        defeat_mode=DefeatMode.WRONG_CONCEPT_FIRST,
        decoy_concept_id="magnets/corrector_magnets",
        rationale=(
            "Quadrupoles are the focusing elements, described via gradients, lenses "
            "and families. The corrector document contains the literal phrase in a "
            "sentence stating that correctors contribute none."
        ),
    ),
    OKFQueryFixture(
        query="how do we keep the stored current constant during user shifts",
        expected_concept_id="operations/top_off_injection",
        defeat_mode=DefeatMode.NO_SUBSTRING_MATCH,
        decoy_concept_id="",
        rationale=(
            "A full-sentence paraphrase of what top-off is for. No document "
            "contains the sentence, so substring search returns nothing."
        ),
    ),
    OKFQueryFixture(
        query="why does the vertical orbit drift after a fill",
        expected_concept_id="operations/orbit_feedback",
        defeat_mode=DefeatMode.NO_SUBSTRING_MATCH,
        decoy_concept_id="",
        rationale=(
            "The orbit-feedback document is the one that explains thermal wander "
            "over a fill and the loops that remove it, but it never phrases it as a "
            "question."
        ),
    ),
    OKFQueryFixture(
        query="what happens to the beam when an interlock opens the shutter",
        expected_concept_id="safety/personnel_safety_system",
        defeat_mode=DefeatMode.NO_SUBSTRING_MATCH,
        decoy_concept_id="",
        rationale=(
            "Only the credited personnel chain closes photon shutters and inhibits "
            "every source at once; the machine-protection document is the plausible "
            "near-miss a weak ranker would return."
        ),
    ),
    OKFQueryFixture(
        query="who do I call at 3am when the beam is down",
        expected_concept_id="operations/shift_handover",
        defeat_mode=DefeatMode.NO_SUBSTRING_MATCH,
        decoy_concept_id="",
        rationale=(
            "The escalation list and its out-of-hours rules live in the handover "
            "procedure, which is not where a keyword search would look."
        ),
    ),
    OKFQueryFixture(
        query="how do beamlines change photon energy",
        expected_concept_id="beamlines/insertion_devices",
        defeat_mode=DefeatMode.NO_SUBSTRING_MATCH,
        decoy_concept_id="",
        rationale=(
            "Photon energy is tuned by moving the insertion-device jaws; the "
            "document explains gap-versus-harmonic behaviour without ever using "
            "the querent's phrasing."
        ),
    ),
    OKFQueryFixture(
        query="why is the pressure high in sector 4 after maintenance",
        expected_concept_id="vacuum/bakeout_procedure",
        defeat_mode=DefeatMode.NO_SUBSTRING_MATCH,
        decoy_concept_id="",
        rationale=(
            "A vented sector sits two decades high until it is baked; the bakeout "
            "document is the answer, while the ion-pump document is the keyword-"
            "plausible distractor."
        ),
    ),
)


def load_fixture_bundle() -> OKFBundle:
    """Open the checked-in fixture bundle as an :class:`OKFBundle`.

    Returns:
        A bundle rooted at :data:`BUNDLE_ROOT`.
    """
    from osprey.services.facility_knowledge.okf.bundle import OKFBundle

    return OKFBundle(BUNDLE_ROOT)
