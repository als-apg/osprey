"""The live PVAccess venue must exist on the platforms that must run it.

``tests/connectors/test_pva_live_fixture.py`` is the only place the PVA read
path is asserted from the far side of a real PVAccess wire: a real pvapy
``PvaServer``, real normative-type structures, the real connector, and the real
``channel_read`` tool body. It gates itself with a module-level
``pytest.importorskip("pvaccess", ...)``, which is the right behaviour on a
host that genuinely cannot load pvapy -- an honest, loud skip beats a hollow
pass -- but it means a run in which nothing was exercised exits 0 exactly like
a run in which everything was.

CI installs ``osprey-connectors``, and ``pvapy>=5.6,<6`` is a core dependency
of that package with no environment marker, so on the unit-lane cells the
wheel must arrive and must load. If packaging, a marker, or wheel availability
ever changes, the unit lane would silently revert to a green that never touched
PVAccess -- the live suite would skip its sixteen tests and nothing would say
so.

**This module is that assertion.** On linux (x86_64 and aarch64) and on macOS
the ``pvaccess`` import must succeed and the live fixture suite must be
collectible and unskipped; the ordinary unit lane fails loudly when either is
untrue.

Three things about its shape are deliberate.

*The platforms are named here, not read from the packaging.* Deriving the
condition from ``packages/osprey-connectors/pyproject.toml`` would look tidier
and be strictly weaker: adding a marker there would silence this guard at the
same moment it stopped the dependency from arriving, which is precisely the
silent revert being guarded against. Naming the platforms outright means no
edit anywhere else can make this inert. Nothing here reads ``pyproject.toml``
or asserts that someone typed a dependency -- what is asserted is that the
dependency *arrived* and *works*.

*The version floor is read from installed metadata, not from the module.* The
import name (``pvaccess``) is not the distribution name (``pvapy``), and
``importlib.metadata`` on the distribution is the honest source. The floor
mirrors the declared ``pvapy>=5.6``: the connector classifies pvapy's single
exception type by its message text, pinned against that release.

*The failure path is exercised on every host.* Where the real check is inert
by construction, a guard whose only real exercise happens where nobody is
watching is worth very little. :class:`TestTheGuardCanFail` drives the same
guard bodies against simulated platforms and simulated import outcomes, and
asserts they reject the combinations they must.
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

#: The module the live suite stands its server up from, and the name it passes
#: to ``importorskip``. The two must agree, or this guard would pass while the
#: suite skipped; :class:`TestTheGuardWatchesTheRightModule` pins that.
PVA_MODULE = "pvaccess"

#: The distribution that installs :data:`PVA_MODULE`, whose metadata carries
#: the version.
PVA_DISTRIBUTION = "pvapy"

#: The suite whose assertions are only observable over a real PVAccess wire,
#: as a file name and as the dotted path this guard imports it by.
LIVE_SUITE = "test_pva_live_fixture.py"
LIVE_SUITE_MODULE = "tests.connectors.test_pva_live_fixture"

#: Floor for the live suite's own size, so that emptying it out cannot pass
#: this guard. Well below the sixteen tests it ships with: this is a
#: tripwire for deletion, not a headcount to keep in sync.
MIN_LIVE_SUITE_TESTS = 12

#: The platforms with a loadable pvapy wheel that a lane runs on, and
#: therefore the ones where a skipped live suite is a defect rather than an
#: honest absence. Both CI unit-lane cells -- ubuntu and macOS -- are among
#: them, as is the arm64 linux image.
#:
#: Named here rather than derived from packaging metadata: see the module
#: docstring. ``platform.machine()`` reports ``x86_64``/``aarch64`` on linux;
#: ``amd64``/``arm64`` are accepted as the same architectures under their other
#: spellings. macOS is required on either architecture, because pvapy ships
#: both. Windows is exempt: no lane runs there, and the live suite's server
#: process stops itself by noticing it was reparented, which is POSIX
#: behaviour.
PVA_LINUX_MACHINES = frozenset({"x86_64", "amd64", "aarch64", "arm64"})
PVA_MACOS_PLATFORM = "darwin"

#: The declared floor in ``packages/osprey-connectors/pyproject.toml``, read
#: from the installed ``pvapy`` distribution's metadata. A floor that falls
#: behind pyproject's makes this test weaker, never falsely red.
MINIMUM_VERSION = (5, 6)
REJECTED_VERSION = "5.5.2"
REQUIRED_VERSION = "5.6.0"

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
        import_module: how to import pvaccess. A real import, not a metadata
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
    except Exception as exc:  # any import failure is the failure
        raise AssertionError(f"{MISSING_VENUE_HINT} Import failed with: {exc!r}") from exc

    found = installed_version(PVA_DISTRIBUTION)
    assert _version_tuple(found) >= MINIMUM_VERSION, (
        f"{PVA_DISTRIBUTION} {found} is below "
        f"{'.'.join(str(part) for part in MINIMUM_VERSION)}, the floor declared in "
        f"packages/osprey-connectors/pyproject.toml. The connector classifies pvapy's one "
        f"exception type by the message texts of that release onward, so an older pvapy "
        f"would serve a venue that certifies a classification nobody pinned."
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
    except Exception as exc:  # a suite that cannot import cannot run
        raise AssertionError(f"{SKIPPED_SUITE_HINT} It failed to import with: {exc!r}") from exc


class TestThePlatformsThatMustRunIt:
    """Which platforms evaluate the assertion, simulated rather than assumed.

    Only one of these is the host running them, so the decision is driven
    against values it does not have. The linux and darwin cases are the ones
    that matter: they are what CI's unit-lane cells and the arm64 image
    report, and asserting them True here is what makes the real checks below
    live rather than universally inert.
    """

    def test_the_venue_is_required_on_linux_both_architectures(self) -> None:
        """pvapy ships x86_64 and aarch64 linux wheels, so neither is exempt."""
        assert on_pva_platform("linux", "x86_64") is True
        assert on_pva_platform("linux", "aarch64") is True

    def test_the_venue_is_required_on_macos_both_architectures(self) -> None:
        """pvapy ships arm64 and x86_64 macOS wheels, so neither Mac is exempt."""
        assert on_pva_platform("darwin", "arm64") is True
        assert on_pva_platform("darwin", "x86_64") is True

    def test_the_architecture_spelling_does_not_change_the_answer(self) -> None:
        """``amd64``/``x86_64`` and ``arm64``/``aarch64`` name one architecture
        each; a runner reporting the other spelling must not fall out of the
        guard's scope."""
        assert on_pva_platform("linux", "amd64") is True
        assert on_pva_platform("linux", "X86_64") is True
        assert on_pva_platform("linux", "arm64") is True

    @pytest.mark.parametrize(
        ("_name", "sys_platform", "machine"),
        [
            ("linux-armv7l", "linux", "armv7l"),
            ("linux-ppc64le", "linux", "ppc64le"),
            ("windows-x86_64", "win32", "AMD64"),
        ],
    )
    def test_the_venue_is_not_required_elsewhere(
        self, _name: str, sys_platform: str, machine: str
    ) -> None:
        """No pvapy wheel exists for 32-bit ARM or POWER linux, and no lane runs
        on Windows -- so an honest skip is the right outcome there and this
        guard must not redden it."""
        assert on_pva_platform(sys_platform, machine) is False


class TestTheVenueIsPresentWhereItIsRequired:
    """The guards themselves, against the platform this run is actually on."""

    def test_the_pvaccess_client_library_is_installed(self) -> None:
        """On linux and macOS -- CI's unit-lane cells -- this fails the unit
        lane outright when pvapy is missing, instead of letting the
        live suite skip its way to a green that proves nothing. Elsewhere it
        is inert by construction, and :class:`TestTheGuardCanFail` is what
        proves it still bites."""
        check_venue(sys_platform=sys.platform, machine=platform.machine())

    def test_the_live_pva_suite_collects_unskipped(self) -> None:
        """Importable pvapy is not the whole claim: the suite could gate itself
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
        ("_name", "sys_platform", "machine"),
        [
            ("linux-x86_64", "linux", "x86_64"),
            ("macos-arm64", "darwin", "arm64"),
            ("linux-aarch64", "linux", "aarch64"),
        ],
    )
    def test_a_required_venue_that_is_missing_is_rejected(
        self, _name: str, sys_platform: str, machine: str
    ) -> None:
        with pytest.raises(AssertionError, match="SKIP"):
            check_venue(sys_platform=sys_platform, machine=machine, import_module=self._absent)

    def test_the_rejection_names_the_command_that_fixes_it(self) -> None:
        with pytest.raises(AssertionError, match=re.escape("uv sync --extra dev")):
            check_venue(sys_platform="linux", machine="x86_64", import_module=self._absent)

    def test_a_venue_that_installed_but_cannot_load_is_rejected(self) -> None:
        """The failure mode a metadata-only check would miss: the distribution
        is present and its extension module will not load."""

        def unloadable(_name: str) -> Any:
            raise ImportError("dlopen failed: libpvAccess.so.7.1 not found")

        with pytest.raises(AssertionError, match="dlopen"):
            check_venue(sys_platform="linux", machine="x86_64", import_module=unloadable)

    def test_a_required_venue_below_the_floor_is_rejected(self) -> None:
        """Importable is not enough: a pvapy below the floor classifies its
        failures by texts the connector was never pinned against."""
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

    def test_the_version_is_read_from_the_distribution_not_the_module(self) -> None:
        """``pvaccess`` is installed by ``pvapy``; metadata is keyed on the latter."""
        asked: list[str] = []

        def version_of(name: str) -> str:
            asked.append(name)
            return REQUIRED_VERSION

        check_venue(
            sys_platform="linux",
            machine="x86_64",
            import_module=self._present,
            installed_version=version_of,
        )

        assert asked == [PVA_DISTRIBUTION]

    def test_a_prerelease_suffix_does_not_defeat_the_floor(self) -> None:
        check_venue(
            sys_platform="darwin",
            machine="arm64",
            import_module=self._present,
            installed_version=lambda name: "5.6.0rc1",
        )

    @pytest.mark.parametrize(
        ("_name", "sys_platform", "machine"),
        [("linux-armv7l", "linux", "armv7l"), ("windows-x86_64", "win32", "AMD64")],
    )
    def test_a_missing_venue_is_tolerated_where_it_is_not_required(
        self, _name: str, sys_platform: str, machine: str
    ) -> None:
        """32-bit ARM linux has no wheel; its skip must stay honest."""
        check_venue(sys_platform=sys_platform, machine=machine, import_module=self._absent)

    def test_a_skipped_live_suite_is_rejected_where_it_is_required(self) -> None:
        """The exact hole this module exists for: the suite's own
        ``importorskip`` firing on a platform that must run it."""

        def skipping(_name: str) -> Any:
            raise pytest.skip.Exception("could not import 'pvaccess'")

        with pytest.raises(AssertionError, match="silent hole"):
            check_live_suite_runs(sys_platform="linux", machine="x86_64", import_module=skipping)

    def test_a_live_suite_that_cannot_import_is_rejected(self) -> None:
        """A collection error is as blind as a skip -- neither runs the wire."""

        def broken(_name: str) -> Any:
            raise ImportError("cannot import name 'NtNdArray' from 'pvaccess'")

        with pytest.raises(AssertionError, match="NtNdArray"):
            check_live_suite_runs(sys_platform="linux", machine="x86_64", import_module=broken)

    def test_a_skipped_live_suite_is_tolerated_where_it_is_not_required(self) -> None:
        def skipping(_name: str) -> Any:
            raise pytest.skip.Exception("could not import 'pvaccess'")

        check_live_suite_runs(sys_platform="linux", machine="armv7l", import_module=skipping)


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
