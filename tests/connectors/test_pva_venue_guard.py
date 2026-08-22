"""The live PVAccess venue must exist on the platforms that must run it.

``tests/connectors/test_pva_live_fixture.py`` is the only place the PVA read
path is asserted from the far side of a real PVAccess wire: a real ``p4p``
server, real normative-type structures, the real connector, and the real
``channel_read`` tool body. It gates itself with a module-level
``pytest.importorskip("p4p", ...)``, which is the right behaviour on a host
that genuinely cannot load p4p -- an honest, loud skip beats a hollow pass --
but it means a run in which nothing was exercised exits 0 exactly like a run
in which everything was.

CI installs ``osprey-connectors``, and ``p4p>=4.2`` is a core dependency of
that package with no environment marker, so on the unit-lane cells the wheel
must arrive and must load. If packaging, a marker, or wheel availability ever
changes, the unit lane would silently revert to a green that never touched
PVAccess -- the live suite would skip its sixteen tests and nothing would say
so.

**This module is that assertion.** On linux/x86_64 and on macOS the p4p import
must succeed and the live fixture suite must be collectible and unskipped; the
ordinary unit lane fails loudly when either is untrue.

Three things about its shape are deliberate.

*The platforms are named here, not read from the packaging.* Deriving the
condition from ``packages/osprey-connectors/pyproject.toml`` would look tidier
and be strictly weaker: adding a marker there would silence this guard at the
same moment it stopped the dependency from arriving, which is precisely the
silent revert being guarded against. Naming linux/x86_64 and macOS outright
means no edit anywhere else can make this inert. Nothing here reads
``pyproject.toml`` or asserts that someone typed a dependency -- what is
asserted is that the dependency *arrived* and *works*.

*The version floor is read from installed metadata, not from p4p itself.* p4p
exposes no ``__version__`` attribute, so ``importlib.metadata`` is the only
honest source. The floor mirrors the declared ``p4p>=4.2``: below it the
normative-type helpers this feature reads through are not the ones the mapping
was written against.

*The failure path is exercised on every host.* On bare-metal arm64 linux --
where no p4p wheel exists and building from source needs a C toolchain and the
EPICS base sources -- the real check is inert by construction, and a guard
whose only real exercise happens where nobody is watching is worth very
little. :class:`TestTheGuardCanFail` drives the same guard bodies against
simulated platforms and simulated import outcomes, and asserts they reject the
combinations they must.
"""

from __future__ import annotations

import ast
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
MIN_COLLECTED_TESTS = 20

#: The module the live suite stands its server up from, and the name it passes
#: to ``importorskip``. The two must agree, or this guard would pass while the
#: suite skipped; :class:`TestTheGuardWatchesTheRightModule` pins that.
PVA_MODULE = "p4p"

#: The suite whose assertions are only observable over a real PVAccess wire,
#: as a file name and as the dotted path this guard imports it by.
LIVE_SUITE = "test_pva_live_fixture.py"
LIVE_SUITE_MODULE = "tests.connectors.test_pva_live_fixture"

#: Floor for the live suite's own size, so that emptying it out cannot pass
#: this guard. Well below the sixteen tests it ships with: this is a
#: tripwire for deletion, not a headcount to keep in sync.
MIN_LIVE_SUITE_TESTS = 12

#: The platforms with a loadable p4p wheel, and therefore the ones where a
#: skipped live suite is a defect rather than an honest absence. Both CI
#: unit-lane cells -- ubuntu and macOS -- are among them.
#:
#: Named here rather than derived from packaging metadata: see the module
#: docstring. ``platform.machine()`` reports ``x86_64`` on linux; ``amd64`` is
#: accepted as the same architecture under its other spelling. macOS is
#: required on either architecture, because p4p ships both.
PVA_LINUX_MACHINES = frozenset({"x86_64", "amd64"})
PVA_MACOS_PLATFORM = "darwin"

#: The declared floor in ``packages/osprey-connectors/pyproject.toml``. p4p has
#: no ``__version__``, so the installed distribution metadata is the source.
#: A floor that falls behind pyproject's makes this test weaker, never
#: falsely red.
MINIMUM_VERSION = (4, 2)
REJECTED_VERSION = "4.1.12"
REQUIRED_VERSION = "4.2.2"

MISSING_VENUE_HINT = (
    f"the live PVAccess venue is required on this platform but {PVA_MODULE} is not importable. "
    f"The live suite ({LIVE_SUITE}) will SKIP, and a skipped live suite proves nothing -- this "
    f"lane would report green without ever having touched PVAccess. Install the connectors "
    f"package: `uv sync --extra dev`."
)

SKIPPED_SUITE_HINT = (
    f"the live PVAccess venue is required on this platform but {LIVE_SUITE} does not collect "
    f"as a live suite. A module-level SKIP here is a silent hole: sixteen wire-level "
    f"assertions would vanish and the lane would still exit 0."
)


def on_pva_platform(sys_platform: str, machine: str) -> bool:
    """Is this a platform where a skipped live PVA suite is a defect?

    Args:
        sys_platform: ``sys.platform``-shaped value.
        machine: ``platform.machine()``-shaped value. Compared
            case-insensitively, because it is the kernel's own spelling and
            differs in case across platforms.
    """
    if sys_platform == PVA_MACOS_PLATFORM:
        return True
    return sys_platform == "linux" and machine.lower() in PVA_LINUX_MACHINES


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
    """Raise ``AssertionError`` if this platform must serve PVAccess and cannot.

    The guard body, with the platform and both environment lookups passed in,
    so that the failure path can be driven on a host where the real one is
    inert.

    Args:
        sys_platform: the platform to decide against.
        machine: the machine architecture to decide against.
        import_module: how to import p4p. A real import, not a metadata
            lookup: the extension modules this guard exists to notice the
            absence of can install perfectly and then fail to load.
        installed_version: how to read the installed version.
    """
    if not on_pva_platform(sys_platform, machine):
        # No wheel exists here, so the suite's skip is the honest outcome and
        # there is nothing to enforce.
        return

    try:
        import_module(PVA_MODULE)
    except Exception as exc:  # noqa: BLE001 - any import failure is the failure
        raise AssertionError(f"{MISSING_VENUE_HINT} Import failed with: {exc!r}") from exc

    found = installed_version(PVA_MODULE)
    assert _version_tuple(found) >= MINIMUM_VERSION, (
        f"{PVA_MODULE} {found} is below "
        f"{'.'.join(str(part) for part in MINIMUM_VERSION)}, the floor declared in "
        f"packages/osprey-connectors/pyproject.toml. The normative-type helpers the PVA read "
        f"mapping is written against are the ones from that release onward, so an older p4p "
        f"would serve a venue that certifies a mapping nobody wrote."
    )


def check_live_suite_runs(
    *,
    sys_platform: str,
    machine: str,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> None:
    """Raise ``AssertionError`` if the live suite would skip where it must run.

    Importing the suite module is the collection step: its module-level
    ``importorskip`` raises :class:`Skipped` before any test is gathered, so a
    module that imports cleanly is exactly a module that collects unskipped.

    Args:
        sys_platform: the platform to decide against.
        machine: the machine architecture to decide against.
        import_module: how to import the live suite module.
    """
    if not on_pva_platform(sys_platform, machine):
        return

    try:
        import_module(LIVE_SUITE_MODULE)
    except pytest.skip.Exception as exc:
        raise AssertionError(f"{SKIPPED_SUITE_HINT} It skipped with: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - a suite that cannot import cannot run
        raise AssertionError(f"{SKIPPED_SUITE_HINT} It failed to import with: {exc!r}") from exc


class TestThePlatformsThatMustRunIt:
    """Which platforms evaluate the assertion, simulated rather than assumed.

    Only one of these is the host running them, so the decision is driven
    against values it does not have. The linux/x86_64 and darwin cases are the
    ones that matter: they are what CI's two unit-lane cells report, and
    asserting them True here is what makes the real checks below live rather
    than universally inert.
    """

    def test_the_venue_is_required_on_linux_x86_64(self) -> None:
        assert on_pva_platform("linux", "x86_64") is True

    def test_the_venue_is_required_on_macos_both_architectures(self) -> None:
        """p4p ships arm64 and x86_64 macOS wheels, so neither Mac is exempt."""
        assert on_pva_platform("darwin", "arm64") is True
        assert on_pva_platform("darwin", "x86_64") is True

    def test_the_architecture_spelling_does_not_change_the_answer(self) -> None:
        """``amd64`` and ``x86_64`` name one architecture; a runner reporting
        the other spelling must not fall out of the guard's scope."""
        assert on_pva_platform("linux", "amd64") is True
        assert on_pva_platform("linux", "X86_64") is True

    @pytest.mark.parametrize(
        ("name", "sys_platform", "machine"),
        [
            ("linux-aarch64", "linux", "aarch64"),
            ("linux-armv7l", "linux", "armv7l"),
            ("windows-x86_64", "win32", "AMD64"),
        ],
    )
    def test_the_venue_is_not_required_elsewhere(
        self, name: str, sys_platform: str, machine: str
    ) -> None:
        """No p4p wheel exists on any of these -- bare-metal arm64 linux would
        have to build p4p, pvxslibs and epicscorelibs from source -- so an
        honest skip is the right outcome and this guard must not redden it."""
        assert on_pva_platform(sys_platform, machine) is False


class TestTheVenueIsPresentWhereItIsRequired:
    """The guards themselves, against the platform this run is actually on."""

    def test_the_pvaccess_client_library_is_installed(self) -> None:
        """On linux/x86_64 and macOS -- CI's two unit-lane cells -- this fails
        the unit lane outright when p4p is missing, instead of letting the
        live suite skip its way to a green that proves nothing. Elsewhere it
        is inert by construction, and :class:`TestTheGuardCanFail` is what
        proves it still bites."""
        check_venue(sys_platform=sys.platform, machine=platform.machine())

    def test_the_live_pva_suite_collects_unskipped(self) -> None:
        """Importable p4p is not the whole claim: the suite could gate itself
        on something else, or stop importing for its own reasons. What has to
        be true is that the wire-level suite actually runs here."""
        check_live_suite_runs(sys_platform=sys.platform, machine=platform.machine())


class TestTheGuardCanFail:
    """The failure paths, driven on every host including the ones they spare.

    Each case runs a real guard body against a simulated platform and a
    simulated import outcome; nothing here restates the assertion under test.
    """

    @staticmethod
    def _absent(name: str) -> Any:
        raise ModuleNotFoundError(f"No module named {name!r}")

    @staticmethod
    def _present(name: str) -> Any:
        return object()

    @pytest.mark.parametrize(
        ("name", "sys_platform", "machine"),
        [
            ("linux-x86_64", "linux", "x86_64"),
            ("macos-arm64", "darwin", "arm64"),
        ],
    )
    def test_a_required_venue_that_is_missing_is_rejected(
        self, name: str, sys_platform: str, machine: str
    ) -> None:
        with pytest.raises(AssertionError, match="SKIP"):
            check_venue(sys_platform=sys_platform, machine=machine, import_module=self._absent)

    def test_the_rejection_names_the_command_that_fixes_it(self) -> None:
        with pytest.raises(AssertionError, match=re.escape("uv sync --extra dev")):
            check_venue(sys_platform="linux", machine="x86_64", import_module=self._absent)

    def test_a_venue_that_installed_but_cannot_load_is_rejected(self) -> None:
        """The failure mode a metadata-only check would miss: the distribution
        is present and its extension module will not load."""

        def unloadable(name: str) -> Any:
            raise ImportError("dlopen failed: libpvxs.so.1.3 not found")

        with pytest.raises(AssertionError, match="dlopen"):
            check_venue(sys_platform="linux", machine="x86_64", import_module=unloadable)

    def test_a_required_venue_below_the_floor_is_rejected(self) -> None:
        """Importable is not enough. p4p exposes no ``__version__``, so this is
        the only place the floor can be read at all."""
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
            sys_platform="darwin",
            machine="arm64",
            import_module=self._present,
            installed_version=lambda name: "4.2.0rc1",
        )

    @pytest.mark.parametrize(
        ("name", "sys_platform", "machine"),
        [("linux-aarch64", "linux", "aarch64"), ("windows-x86_64", "win32", "AMD64")],
    )
    def test_a_missing_venue_is_tolerated_where_it_is_not_required(
        self, name: str, sys_platform: str, machine: str
    ) -> None:
        """Bare-metal arm64 linux has no wheel; its skip must stay honest."""
        check_venue(sys_platform=sys_platform, machine=machine, import_module=self._absent)

    def test_a_skipped_live_suite_is_rejected_where_it_is_required(self) -> None:
        """The exact hole this module exists for: the suite's own
        ``importorskip`` firing on a platform that must run it."""

        def skipping(name: str) -> Any:
            raise pytest.skip.Exception("could not import 'p4p'")

        with pytest.raises(AssertionError, match="silent hole"):
            check_live_suite_runs(sys_platform="linux", machine="x86_64", import_module=skipping)

    def test_a_live_suite_that_cannot_import_is_rejected(self) -> None:
        """A collection error is as blind as a skip -- neither runs the wire."""

        def broken(name: str) -> Any:
            raise ImportError("cannot import name 'NTNDArray' from 'p4p.nt'")

        with pytest.raises(AssertionError, match="NTNDArray"):
            check_live_suite_runs(sys_platform="linux", machine="x86_64", import_module=broken)

    def test_a_skipped_live_suite_is_tolerated_where_it_is_not_required(self) -> None:
        def skipping(name: str) -> Any:
            raise pytest.skip.Exception("could not import 'p4p'")

        check_live_suite_runs(sys_platform="linux", machine="aarch64", import_module=skipping)


class TestTheGuardWatchesTheRightModule:
    """A guard on a suite that gates on something else would pass while it skipped."""

    def test_the_live_suite_exists(self) -> None:
        assert (Path(__file__).parent / LIVE_SUITE).is_file()

    def test_the_live_suite_gates_on_the_module_this_guard_checks(self) -> None:
        source = (Path(__file__).parent / LIVE_SUITE).read_text()
        gated = set(re.findall(r"""importorskip\(\s*["']([\w.]+)["']""", source))
        assert gated == {PVA_MODULE}, (
            f"{LIVE_SUITE} gates its live venue on {sorted(gated)}, but this guard enforces "
            f"{PVA_MODULE!r}; one of the two has moved"
        )

    def test_the_live_suite_still_has_tests_to_lose(self) -> None:
        """Enforcing that a suite runs is worth nothing if the suite is empty;
        this is the tripwire for that, read from source so it holds on hosts
        where the suite cannot be imported at all."""
        tree = ast.parse((Path(__file__).parent / LIVE_SUITE).read_text())
        tests = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name.startswith("test_")
        ]
        assert len(tests) >= MIN_LIVE_SUITE_TESTS


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_pva_venue_guard.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
