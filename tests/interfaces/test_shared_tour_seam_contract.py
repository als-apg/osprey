"""One seam dismisses the onboarding tour for every real-browser suite.

Every Playwright suite under ``tests/interfaces/`` takes its pages from the
``chromium_browser`` fixture, whose ``new_context``/``new_page`` wrappers add
``conftest._DISMISS_TOUR`` to each one. A suite that also seeds the key in an
init script of its own is not merely repeating the seam — it is repeating the
narrower HALF of it, because the seam answers the key as well as writing it
and a bare write misses the ``--``-scoped variants a multi-user mount reads.
Eleven suites carried such a copy. This is the failure that was missing.

Two rules, because there are two ways back:

- **The seam must ANSWER the key, not only write it.** A seam narrowed to a
  lone ``localStorage.setItem`` passes a count-the-copies check and leaves
  every scoped mount's invite armed.
- **No other module may seed it.** One home, and a new suite that grows a
  private copy is red here rather than silently arming the invite for the
  scoped case.

Python modules only. ``tour.test.mjs`` and ``js/storage-scope-keys.test.js``
name the key because the key IS their subject, and they open no browser.
"""

from __future__ import annotations

from pathlib import Path

from tests.interfaces.conftest import _DISMISS_TOUR

REPO = Path(__file__).resolve().parents[2]
INTERFACE_TESTS = REPO / "tests" / "interfaces"
CONFTEST = INTERFACE_TESTS / "conftest.py"

#: The key ``tour.js`` writes when the invite is dismissed.
DISMISS_KEY = "osprey-tour-dismissed-v1"

#: The two files allowed to name it: the seam, and this guard.
SEAM_OWNERS = frozenset({"conftest.py", Path(__file__).name})

#: The narrower half the eleven private copies carried, verbatim.
BARE_WRITE = "try { localStorage.setItem('osprey-tour-dismissed-v1', '1') } catch (e) {}"


def _seam_shortfalls(seam: str) -> list[str]:
    """Every part of the contract *seam* does not meet."""
    missing = []
    if DISMISS_KEY not in seam:
        missing.append("does not name the dismissal key")
    if "localStorage.setItem" not in seam:
        missing.append("does not write the key")
    if "Storage.prototype.getItem =" not in seam:
        missing.append("does not answer the key — a page that READS it still sees the invite")
    if "'--'" not in seam:
        missing.append("does not answer the scoped variants a multi-user mount reads")
    return missing


def _private_seeds(root: Path) -> list[str]:
    """Python modules under *root* naming the key that are not the seam."""
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path.name not in SEAM_OWNERS and DISMISS_KEY in path.read_text(encoding="utf-8")
    )


def test_the_seam_writes_and_answers_the_key() -> None:
    shortfalls = _seam_shortfalls(_DISMISS_TOUR)
    assert shortfalls == [], (
        "conftest._DISMISS_TOUR is the one tour seed every browser page carries, and it "
        f"{'; '.join(shortfalls)}. tour.js reads the key through scopedStorageKey, so a "
        "multi-user mount asks for the '--'-scoped variant: the seam has to answer reads, "
        "not only write the bare key."
    )


def test_the_seam_writes_and_answers_the_key__mutation_narrows_to_a_bare_write() -> None:
    """A seam narrowed back to a lone write must be caught."""
    assert _seam_shortfalls(BARE_WRITE) != [], (
        "a bare localStorage.setItem must not satisfy the seam contract — it is exactly "
        "the narrower half the private copies carried"
    )


def test_every_page_the_fixture_hands_out_carries_the_seam() -> None:
    source = CONFTEST.read_text(encoding="utf-8")
    assert source.count("add_init_script(_DISMISS_TOUR)") == 2, (
        "both factories _install_auth_seam wraps must seed the tour: new_context and "
        "new_page. new_page builds its own context internally rather than routing through "
        "new_context, so wrapping one is not wrapping both, and a suite calling the "
        "unseeded one gets the invite back."
    )


def test_no_suite_seeds_the_tour_itself() -> None:
    offenders = _private_seeds(INTERFACE_TESTS)
    assert offenders == [], (
        f"these modules seed the onboarding tour themselves: {', '.join(offenders)}. "
        "Every page under tests/interfaces/ comes from the chromium_browser fixture, "
        "which already seeds it — and answers it for the scoped key a private copy misses. "
        "Delete the private seed."
    )


def test_no_suite_seeds_the_tour_itself__mutation_reintroduces_a_private_seed(
    tmp_path: Path,
) -> None:
    """A suite that grows a private copy must be named by the scan."""
    suite = tmp_path / "web_terminal"
    suite.mkdir()
    (suite / "test_private_seed_browser.py").write_text(
        f'_DISMISS_TOUR = "{BARE_WRITE}"\n', encoding="utf-8"
    )
    assert _private_seeds(tmp_path) == ["web_terminal/test_private_seed_browser.py"]
