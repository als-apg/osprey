"""The live Channel Access venue must exist on the platform that must run it.

``tests/va/test_record_factory.py`` and ``tests/va/test_apply_fault.py`` are the
only places the write contract is asserted from the far side of a real Channel
Access wire. Both guard their server import with ``pytest.importorskip``, which
is the right behaviour on a host that cannot load it -- an honest, loud skip
beats a hollow pass -- but it means a run in which nothing was exercised exits
0 exactly like a run in which everything was. Measured: with the server made
unimportable, the two suites report 9 passed, 44 skipped, and exit 0.

The container venue closes that gap for itself (``scripts/va/live_ca/gate.py``
rejects any run containing a skip). CI does not go through that gate: it
installs the serving stack and runs the suites as part of the ordinary
``pytest tests/`` job, where a skip is silent. So if the packaging, the
environment marker, or wheel availability ever changes, CI reverts to a green
that never touched Channel Access -- the same hole, merely relocated.

**This module is that assertion.** On linux/x86_64 the server must import, and
the ordinary unit lane fails loudly when it does not.

Two things about its shape are deliberate.

*The platform is named here, not read from the packaging.* Deriving the
condition from the marker in ``pyproject.toml`` would look tidier and be
strictly weaker: narrowing that marker would silence this guard at the same
moment it stopped the dependency from arriving, which is precisely the silent
revert being guarded against. Naming linux/x86_64 outright means no edit
anywhere else can make this inert. Nothing here reads ``pyproject.toml`` or
asserts that someone typed a marker -- what is asserted is that the dependency
*arrived*.

*The failure path is exercised on every host.* On a developer's Mac the real
check is inert by construction, and a guard whose only real exercise happens
where nobody is watching is worth very little. :class:`TestTheGuardCanFail`
drives the same guard body against simulated platforms and asserts it rejects
the ones it must.
"""

from __future__ import annotations

import importlib
import platform
import re
import sys
from collections.abc import Callable
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import pytest

# Floor for this module's own test count -- a guard against a refactor that
# leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 15

#: The module the live suites stand their server up from, and the name they
#: pass to ``importorskip``. The two must agree, or this guard would pass
#: while they skipped; :class:`TestTheGuardWatchesTheRightModule` pins that.
LIVE_CA_MODULE = "pcaspy"

#: The suites whose assertions are only observable over a real CA wire.
LIVE_SUITES = ("test_record_factory.py", "test_apply_fault.py")

#: The one platform with a loadable Channel Access server wheel, and therefore
#: the one where a skipped live suite is a defect rather than an honest
#: absence. CI's ubuntu cells and the container venue are both this.
#:
#: Named here rather than derived from packaging metadata: see the module
#: docstring. ``platform.machine()`` reports ``x86_64`` on linux; ``amd64`` is
#: accepted as the same architecture under its other spelling.
LIVE_CA_SYS_PLATFORM = "linux"
LIVE_CA_MACHINES = frozenset({"x86_64", "amd64"})

#: Below this, ``SimplePV.writeNotify`` aborts the server process on any
#: asynchronous write and the client is still told the put succeeded -- the
#: worst possible failure for a venue whose whole job is to prove what a
#: client sees. Importable is therefore not the whole claim.
#:
#: ``pyproject.toml`` remains the authority that installs it; this is the
#: independent check that what arrived is usable. A floor that falls behind
#: pyproject's makes this test weaker, never falsely red.
MINIMUM_VERSION = (0, 8, 1)
REJECTED_VERSION = "0.8.0"
REQUIRED_VERSION = "0.8.1"

MISSING_VENUE_HINT = (
    f"the live Channel Access venue is required on this platform but {LIVE_CA_MODULE} is not "
    f"importable. The live suites ({', '.join(LIVE_SUITES)}) will SKIP, and a skipped live "
    f"suite proves nothing -- this lane would report green without ever having touched Channel "
    f"Access. Install the serving stack: `uv sync --extra dev --extra virtual-accelerator`."
)


def on_live_ca_platform(sys_platform: str, machine: str) -> bool:
    """Is this the platform where a skipped live suite is a defect?

    Args:
        sys_platform: ``sys.platform``-shaped value.
        machine: ``platform.machine()``-shaped value. Compared
            case-insensitively, because it is the kernel's own spelling and
            differs in case across platforms.
    """
    return sys_platform == LIVE_CA_SYS_PLATFORM and machine.lower() in LIVE_CA_MACHINES


def _version_tuple(version: str) -> tuple[int, ...]:
    """The leading numeric components of ``version``, for ordering.

    A trailing suffix (``rc1``, ``.dev0``) stops the parse rather than failing
    it: what matters is which release the numbers name.
    """
    components: list[int] = []
    for part in version.split("."):
        digits = re.match(r"\d+", part)
        if digits is None:
            break
        components.append(int(digits.group()))
    return tuple(components)


def check_venue(
    *,
    sys_platform: str,
    machine: str,
    import_module: Callable[[str], Any] = importlib.import_module,
    installed_version: Callable[[str], str] = distribution_version,
) -> None:
    """Raise ``AssertionError`` if this platform must serve Channel Access and cannot.

    The guard body, with the platform and both environment lookups passed in,
    so that the failure path can be driven on a host where the real one is
    inert.

    Args:
        sys_platform: the platform to decide against.
        machine: the machine architecture to decide against.
        import_module: how to import the server. A real import, not a metadata
            lookup: the wheels this guard exists to notice the absence of
            install perfectly and then fail to load.
        installed_version: how to read the installed version.
    """
    if not on_live_ca_platform(sys_platform, machine):
        # Not a platform that can host the server, so the suites' skip is the
        # honest outcome and there is nothing to enforce.
        return

    try:
        import_module(LIVE_CA_MODULE)
    except Exception as exc:  # noqa: BLE001 - any import failure is the failure
        raise AssertionError(f"{MISSING_VENUE_HINT} Import failed with: {exc!r}") from exc

    found = installed_version(LIVE_CA_MODULE)
    assert _version_tuple(found) >= MINIMUM_VERSION, (
        f"{LIVE_CA_MODULE} {found} is below "
        f"{'.'.join(str(part) for part in MINIMUM_VERSION)}. The floor is a correctness "
        f"requirement, not a preference: below it an asynchronous write aborts the server "
        f"and the client is still told the put succeeded, so the venue would certify a "
        f"contract it never actually upheld."
    )


class TestThePlatformThatMustRunIt:
    """Which platform evaluates the assertion, simulated rather than assumed.

    This host is not that platform, so the decision is driven against values
    it does not have. The linux/x86_64 case is the one that matters: it is
    what CI's ubuntu cells report, and asserting it True here is what makes
    the real check below live rather than universally inert.
    """

    def test_the_venue_is_required_on_linux_x86_64(self) -> None:
        assert on_live_ca_platform("linux", "x86_64") is True

    def test_the_architecture_spelling_does_not_change_the_answer(self) -> None:
        """``amd64`` and ``x86_64`` name one architecture; a runner reporting
        the other spelling must not fall out of the guard's scope."""
        assert on_live_ca_platform("linux", "amd64") is True
        assert on_live_ca_platform("linux", "X86_64") is True

    @pytest.mark.parametrize(
        ("name", "sys_platform", "machine"),
        [
            ("macos-arm64", "darwin", "arm64"),
            ("macos-x86_64", "darwin", "x86_64"),
            ("linux-aarch64", "linux", "aarch64"),
            ("windows-x86_64", "win32", "AMD64"),
        ],
    )
    def test_the_venue_is_not_required_elsewhere(
        self, name: str, sys_platform: str, machine: str
    ) -> None:
        """No loadable wheel exists on any of these, so an honest skip is the
        right outcome and this guard must not turn it into a red lane."""
        assert on_live_ca_platform(sys_platform, machine) is False


class TestTheVenueIsPresentWhereItIsRequired:
    """The guard itself, against the platform this run is actually on."""

    def test_the_live_channel_access_venue_is_installed(self) -> None:
        """On linux/x86_64 -- CI's ubuntu cells -- this fails the unit lane
        outright when the server is missing, instead of letting the live
        suites skip their way to a green that proves nothing. Elsewhere it is
        inert by construction, and :class:`TestTheGuardCanFail` is what proves
        it still bites."""
        check_venue(sys_platform=sys.platform, machine=platform.machine())


class TestTheGuardCanFail:
    """The failure path, driven on every host including the ones it spares.

    Each case runs the real guard body against a simulated platform and a
    simulated lookup; nothing here restates the assertion under test.
    """

    @staticmethod
    def _absent(name: str) -> Any:
        raise ModuleNotFoundError(f"No module named {name!r}")

    @staticmethod
    def _present(name: str) -> Any:
        return object()

    def test_a_required_venue_that_is_missing_is_rejected(self) -> None:
        with pytest.raises(AssertionError, match="SKIP"):
            check_venue(sys_platform="linux", machine="x86_64", import_module=self._absent)

    def test_the_rejection_names_the_command_that_fixes_it(self) -> None:
        with pytest.raises(AssertionError, match=re.escape("--extra virtual-accelerator")):
            check_venue(sys_platform="linux", machine="x86_64", import_module=self._absent)

    def test_a_venue_that_installed_but_cannot_load_is_rejected(self) -> None:
        """The failure mode of the wheels excluded on other platforms: the
        distribution is present and the extension will not load."""

        def unloadable(name: str) -> Any:
            raise ImportError("dlopen failed: libc++.1.dylib not found")

        with pytest.raises(AssertionError, match="dlopen"):
            check_venue(sys_platform="linux", machine="x86_64", import_module=unloadable)

    def test_a_required_venue_below_the_floor_is_rejected(self) -> None:
        """Importable is not enough: the release that silently loses writes
        imports perfectly."""
        with pytest.raises(AssertionError, match=re.escape(REJECTED_VERSION)):
            check_venue(
                sys_platform="linux",
                machine="x86_64",
                import_module=self._present,
                installed_version=lambda name: REJECTED_VERSION,
            )

    def test_a_required_venue_that_is_present_is_accepted(self) -> None:
        check_venue(
            sys_platform="linux",
            machine="x86_64",
            import_module=self._present,
            installed_version=lambda name: REQUIRED_VERSION,
        )

    def test_a_prerelease_suffix_does_not_defeat_the_floor(self) -> None:
        check_venue(
            sys_platform="linux",
            machine="x86_64",
            import_module=self._present,
            installed_version=lambda name: "0.8.1.dev0",
        )

    @pytest.mark.parametrize(
        ("name", "sys_platform", "machine"),
        [("macos-arm64", "darwin", "arm64"), ("linux-aarch64", "linux", "aarch64")],
    )
    def test_a_missing_venue_is_tolerated_where_it_is_not_required(
        self, name: str, sys_platform: str, machine: str
    ) -> None:
        """The developer skip on this very host, which must stay honest."""
        check_venue(sys_platform=sys_platform, machine=machine, import_module=self._absent)


class TestTheGuardWatchesTheRightModule:
    """A guard on a module the suites do not gate on would pass while they skipped."""

    @pytest.mark.parametrize("suite", LIVE_SUITES)
    def test_the_live_suite_exists(self, suite: str) -> None:
        assert (Path(__file__).parent / suite).is_file()

    @pytest.mark.parametrize("suite", LIVE_SUITES)
    def test_the_live_suite_gates_on_the_module_this_guard_checks(self, suite: str) -> None:
        source = (Path(__file__).parent / suite).read_text()
        gated = set(re.findall(r"""importorskip\(\s*["']([\w.]+)["']""", source))
        assert gated == {LIVE_CA_MODULE}, (
            f"{suite} gates its live venue on {sorted(gated)}, but this guard enforces "
            f"{LIVE_CA_MODULE!r}; one of the two has moved"
        )


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_live_ca_venue_guard.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
