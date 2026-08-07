"""The live Channel Access venue must exist on the platform that must run it.

``tests/va/test_record_factory.py`` and ``tests/va/test_apply_fault.py`` are the
only places the write contract is asserted from the far side of a real Channel
Access wire. Both guard their server import with ``pytest.importorskip``, which
is the right behaviour on a host that cannot load it -- an honest, loud skip
beats a hollow pass -- but it means a run in which nothing was exercised exits
0 exactly like a run in which everything was.

The container venue closes that gap for itself (``scripts/va/live_ca/gate.py``
rejects any run with a skip in it). CI does not go through that gate: it
installs the ``virtual-accelerator`` extra and runs the suites as part of the
ordinary ``pytest tests/`` job, where a skip is silent. So if the extra, the
environment marker, or wheel availability ever changes, CI reverts to a green
that never touched Channel Access -- the same hole, merely relocated.

**This module is that assertion.** On any platform where the declared
requirement says the server is installable, it must actually be installed and
importable, and the ordinary unit lane fails loudly if it is not.

Two things keep it honest.

*The platform condition is read from the packaging metadata, never hardcoded.*
The guard evaluates the very marker ``pyproject.toml`` declares, so the guard
and the packaging cannot drift: widen the marker to a new platform and the
guard starts demanding the venue there, with no edit here.

*The guard's failure path is exercised on every host.* On a developer's Mac the
real check is inert by construction -- the marker excludes macOS arm64, and it
excludes it for a documented reason. A guard that is inert everywhere it is
developed and load-bearing only where nobody watches is worth very little, so
:class:`TestTheGuardCanFail` drives the same guard against simulated
environments and asserts it rejects the ones it must.
"""

from __future__ import annotations

import importlib
import re
import tomllib
from collections.abc import Callable
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement

# Floor for this module's own test count -- a guard against a refactor that
# leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 15

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: The extra that carries the Channel Access serving stack.
EXTRA = "virtual-accelerator"

#: The distribution the live suites stand their server up from, and the name
#: they pass to ``importorskip``. The two must agree: a guard checking a
#: module the suites do not gate on would pass while they skipped.
LIVE_CA_DISTRIBUTION = "pcaspy"

#: The suites whose assertions are only observable over a real CA wire.
LIVE_SUITES = ("test_record_factory.py", "test_apply_fault.py")

#: The version that must not be admitted, and why. 0.8.0's
#: ``SimplePV.writeNotify`` aborts the server process on any asynchronous
#: write and the client is told the put succeeded -- the worst possible
#: failure for a venue whose entire job is to prove what a client sees.
REJECTED_VERSION = "0.8.0"
REQUIRED_VERSION = "0.8.1"

#: Marker environments the decision is asserted against. Only the keys that
#: matter are given: ``Marker.evaluate`` merges these over the real
#: environment, so a marker later grown a ``python_version`` clause still
#: evaluates against a coherent environment rather than an empty one.
LINUX_X86_64 = {"sys_platform": "linux", "platform_machine": "x86_64"}
MACOS_ARM64 = {"sys_platform": "darwin", "platform_machine": "arm64"}
LINUX_AARCH64 = {"sys_platform": "linux", "platform_machine": "aarch64"}
WINDOWS_X86_64 = {"sys_platform": "win32", "platform_machine": "AMD64"}

MISSING_VENUE_HINT = (
    f"the live Channel Access venue is required on this platform but {LIVE_CA_DISTRIBUTION} "
    f"is not importable. The live suites ({', '.join(LIVE_SUITES)}) will SKIP, and a skipped "
    f"live suite proves nothing -- this lane would report green without ever touching Channel "
    f"Access. Install the extra: `uv sync --extra dev --extra {EXTRA}`."
)


def live_ca_requirement() -> Requirement:
    """The Channel Access server requirement the extra declares.

    Read from ``pyproject.toml`` rather than from installed metadata, because
    the case this exists to catch is precisely the one where nothing is
    installed.
    """
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    extras = project["optional-dependencies"]
    assert EXTRA in extras, f"pyproject declares no {EXTRA!r} extra"

    for entry in extras[EXTRA]:
        requirement = Requirement(entry)
        if requirement.name == LIVE_CA_DISTRIBUTION:
            return requirement
    raise AssertionError(
        f"the {EXTRA!r} extra no longer carries {LIVE_CA_DISTRIBUTION!r}; "
        f"the live Channel Access suites have no venue to run in"
    )


def venue_required_in(environment: dict[str, str]) -> bool:
    """Does the declared marker demand the venue in ``environment``?

    An unmarked requirement demands it everywhere, which is the honest reading
    and also a configuration this repo deliberately does not use -- see the
    marker's rationale in ``pyproject.toml``.
    """
    marker = live_ca_requirement().marker
    return True if marker is None else bool(marker.evaluate(environment))


def check_venue(
    environment: dict[str, str],
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
    installed_version: Callable[[str], str] | None = None,
) -> None:
    """Raise ``AssertionError`` if ``environment`` demands the venue and lacks it.

    The guard body, with its two environment lookups injectable so that the
    failure path can be driven on a host where the real one is inert.

    Args:
        environment: the marker environment to decide against.
        import_module: how to import the server module. The default is a real
            import, because "installed" is not the claim -- the broken macOS
            wheels install perfectly and fail to load.
        installed_version: how to read the installed version, or ``None`` to
            read it from the imported module's own distribution metadata.
    """
    requirement = live_ca_requirement()
    if not venue_required_in(environment):
        # The venue is not expected here, so the suites' skip is the honest
        # outcome and there is nothing to enforce.
        return

    try:
        import_module(requirement.name)
    except Exception as exc:  # noqa: BLE001 - any import failure is the failure
        raise AssertionError(f"{MISSING_VENUE_HINT} Import failed with: {exc!r}") from exc

    read_version = installed_version if installed_version is not None else distribution_version
    found = read_version(requirement.name)
    assert requirement.specifier.contains(found), (
        f"{requirement.name} {found} does not satisfy the declared floor "
        f"{requirement.specifier}. The floor is a correctness requirement, not a "
        f"preference: below it an asynchronous write aborts the server and the client "
        f"is still told the put succeeded."
    )


class TestDeclaredRequirement:
    """What ``pyproject.toml`` says the venue is."""

    def test_the_extra_still_carries_the_channel_access_server(self) -> None:
        assert live_ca_requirement().name == LIVE_CA_DISTRIBUTION

    def test_the_requirement_is_marked(self) -> None:
        """Unmarked, it would send every macOS and linux-aarch64 host into a
        source build that needs EPICS Base plus PCAS, and fail. Marked, those
        hosts skip honestly and run the suites in the container venue."""
        assert live_ca_requirement().marker is not None

    def test_the_floor_rejects_the_version_that_loses_writes(self) -> None:
        """Asserted behaviourally rather than as a string, so `>=0.8.1` and
        `>0.8.0` are both accepted spellings of the same floor."""
        specifier = live_ca_requirement().specifier
        assert not specifier.contains(REJECTED_VERSION)
        assert specifier.contains(REQUIRED_VERSION)


class TestPlatformDecision:
    """Where the declared marker says the venue must exist.

    This is the platform condition simulated -- the same evaluation the real
    check makes, driven against environments this host is not.
    """

    def test_the_venue_is_required_on_linux_x86_64(self) -> None:
        """CI's ubuntu lanes, and the container venue. The one platform with a
        loadable wheel, and therefore the one where a skip is a defect."""
        assert venue_required_in(LINUX_X86_64) is True

    def test_the_venue_is_not_required_on_macos_arm64(self) -> None:
        """The published arm64 wheels link a libc++ path with no surviving
        rpath entry, so they install and then fail to load."""
        assert venue_required_in(MACOS_ARM64) is False

    def test_the_venue_is_not_required_on_linux_aarch64(self) -> None:
        """No wheel is published there at any interpreter."""
        assert venue_required_in(LINUX_AARCH64) is False

    def test_the_venue_is_not_required_on_windows(self) -> None:
        assert venue_required_in(WINDOWS_X86_64) is False


class TestTheVenueIsPresentWhereItIsRequired:
    """The guard itself, against the platform this run is actually on."""

    def test_the_live_channel_access_venue_is_installed(self) -> None:
        """On linux/x86_64 -- CI's ubuntu cells -- this fails the unit lane
        outright when the server is missing, instead of letting the live
        suites skip their way to a green that proves nothing. On a platform
        the marker excludes it is inert by construction, and
        :class:`TestTheGuardCanFail` is what proves it still bites."""
        check_venue(default_environment())


class TestTheGuardCanFail:
    """The failure path, driven on every host including the ones it spares.

    Each case is the real guard body against a simulated environment and a
    simulated lookup -- nothing here is a restatement of the assertion under
    test.
    """

    @staticmethod
    def _absent(name: str) -> Any:
        raise ModuleNotFoundError(f"No module named {name!r}")

    @staticmethod
    def _present(name: str) -> Any:
        return object()

    def test_a_required_venue_that_is_missing_is_rejected(self) -> None:
        with pytest.raises(AssertionError, match="skipped"):
            check_venue(LINUX_X86_64, import_module=self._absent)

    def test_the_rejection_names_the_command_that_fixes_it(self) -> None:
        with pytest.raises(AssertionError, match=re.escape(f"--extra {EXTRA}")):
            check_venue(LINUX_X86_64, import_module=self._absent)

    def test_a_required_venue_below_the_floor_is_rejected(self) -> None:
        """Importable is not enough: the version that silently loses writes
        imports perfectly."""
        with pytest.raises(AssertionError, match=REJECTED_VERSION):
            check_venue(
                LINUX_X86_64,
                import_module=self._present,
                installed_version=lambda name: REJECTED_VERSION,
            )

    def test_a_required_venue_that_is_present_is_accepted(self) -> None:
        check_venue(
            LINUX_X86_64,
            import_module=self._present,
            installed_version=lambda name: REQUIRED_VERSION,
        )

    @pytest.mark.parametrize(
        ("platform", "environment"),
        [("macos-arm64", MACOS_ARM64), ("linux-aarch64", LINUX_AARCH64)],
    )
    def test_a_missing_venue_is_tolerated_where_it_is_not_required(
        self, platform: str, environment: dict[str, str]
    ) -> None:
        """The suites' skip there is the honest outcome, and this guard must
        not turn a documented absence into a red lane."""
        check_venue(environment, import_module=self._absent)


class TestTheGuardWatchesTheRightModule:
    """A guard on a module the suites do not gate on would pass while they skipped."""

    @pytest.mark.parametrize("suite", LIVE_SUITES)
    def test_the_live_suite_exists(self, suite: str) -> None:
        assert (Path(__file__).parent / suite).is_file()

    @pytest.mark.parametrize("suite", LIVE_SUITES)
    def test_the_live_suite_gates_on_the_module_this_guard_checks(self, suite: str) -> None:
        source = (Path(__file__).parent / suite).read_text()
        gated = set(re.findall(r"""importorskip\(\s*["']([\w.]+)["']""", source))
        assert gated == {LIVE_CA_DISTRIBUTION}, (
            f"{suite} gates its live venue on {sorted(gated)}, but this guard enforces "
            f"{LIVE_CA_DISTRIBUTION!r}; one of the two has moved"
        )


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_live_ca_venue_guard.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
