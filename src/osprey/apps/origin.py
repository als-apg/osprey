"""Where an application comes from: a deployment's ``apps/`` folder, or an installed package.

A distribution ships an application by declaring an entry point in the
:data:`ENTRY_POINT_GROUP` group, named for the application and pointing at a
package::

    [project.entry-points."osprey.apps"]
    my-app = "my_app"

The package's directory is the application's directory: it holds ``app.yml``,
``panel/``, ``skills/`` and the application's Python modules, exactly as
``apps/<name>/`` does in a deployment. Discovery reads entry points and locates
each package without importing it, so listing what is installed never runs an
application. Which applications a deployment runs is still the profile's
``apps:`` block alone.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from osprey.apps.errors import AppManifestError, Violation

if TYPE_CHECKING:
    from osprey.apps.manifest import AppManifest

#: The entry-point group an installed application is declared in.
ENTRY_POINT_GROUP = "osprey.apps"

#: The file that makes a directory an application.
MANIFEST_FILE = "app.yml"


@dataclass(frozen=True)
class AppOrigin:
    """Where one application's directory is, and who put it there.

    Attributes:
        kind: ``deployment`` for ``apps/<name>/`` of the deployment,
            ``installed`` for a package an installed distribution ships.
        directory: The application's directory.
        distribution: The installing distribution's name; ``None`` for a
            deployment application.
        version: That distribution's version; ``None`` for a deployment application.
        module: The application's package, as the entry point names it; ``None``
            for a deployment application.
    """

    kind: Literal["deployment", "installed"]
    directory: Path
    distribution: str | None = None
    version: str | None = None
    module: str | None = None

    def __post_init__(self) -> None:
        if self.installed != (self.distribution is not None and self.module is not None):
            raise ValueError(
                "an installed application's origin names its distribution and package; "
                "a deployment application's names neither"
            )

    @property
    def installed(self) -> bool:
        """Whether an installed distribution ships the application."""
        return self.kind == "installed"

    @property
    def installer(self) -> str:
        """``<distribution> <version>``: who installs the application."""
        return f"{self.distribution} {self.version}"

    def display(self, rel: str = "") -> str:
        """How a report names *rel*, a path relative to the application's directory.

        ``apps/<name>/<rel>`` for a deployment application,
        ``<distribution>:<module path>/<rel>`` for an installed one; the directory
        itself when *rel* is empty.
        """
        if self.module is not None:
            base = f"{self.distribution}:{self.module.replace('.', '/')}"
        else:
            from osprey.utils.workspace import APPS_DIRNAME

            base = f"{APPS_DIRNAME}/{self.directory.name}"
        return f"{base}/{rel}" if rel else base

    @property
    def manifest_source(self) -> str | None:
        """The file a refusal of the manifest names, ``None`` for ``apps/<name>/app.yml``."""
        if self.module is None:
            return None
        return f"{self.module.replace('.', '/')}/{MANIFEST_FILE} (installed by {self.installer})"


def deployment_origin(root: Path, name: str) -> AppOrigin:
    """The origin of ``apps/<name>/`` under the deployment root *root*."""
    from osprey.utils.workspace import APPS_DIRNAME

    return AppOrigin(kind="deployment", directory=root / APPS_DIRNAME / name)


def origin_of(manifest: AppManifest, root: Path) -> AppOrigin:
    """Where *manifest* came from; ``apps/<name>/`` of *root* for one no loader read."""
    return manifest.origin or deployment_origin(root, manifest.name)


def installed_apps(entry_points: Iterable[Any] | None = None) -> dict[str, AppOrigin]:
    """Every application the environment's installed distributions declare, by name.

    Args:
        entry_points: The ``osprey.apps`` entry points to read; ``None`` lists
            the environment's.

    Returns:
        Each application's origin, keyed by its entry-point name.

    Raises:
        AppManifestError: Two entry points share a name (``installed_duplicate``),
            or an entry point's module is not a package directory holding
            ``app.yml`` (``installed_not_a_package``); every one found.
    """
    if entry_points is None:
        entry_points = metadata.entry_points(group=ENTRY_POINT_GROUP)
    found: dict[str, AppOrigin] = {}
    violations: list[Violation] = []
    for ep in entry_points:
        dist = getattr(ep, "dist", None)
        distribution = str(getattr(dist, "name", None) or "an unknown distribution")
        version = str(getattr(dist, "version", None) or "?")
        module = str(getattr(ep, "module", None) or str(ep.value).split(":")[0])
        directory = _package_directory(module)
        if directory is None or not (directory / MANIFEST_FILE).is_file():
            violations.append(
                Violation(
                    ep.name,
                    "",
                    "installed_not_a_package",
                    f"the entry point {ep.name} = {module!r} of {distribution} {version} "
                    f"names no package directory holding {MANIFEST_FILE}",
                    source=f"{ENTRY_POINT_GROUP} entry point",
                )
            )
            continue
        origin = AppOrigin(
            kind="installed",
            directory=directory,
            distribution=distribution,
            version=version,
            module=module,
        )
        other = found.get(ep.name)
        if other is not None:
            violations.append(
                Violation(
                    ep.name,
                    "",
                    "installed_duplicate",
                    f"both {other.installer} and {origin.installer} install an application "
                    f"named {ep.name}; uninstall one",
                    source=f"{ENTRY_POINT_GROUP} entry point",
                )
            )
            continue
        found[ep.name] = origin
    if violations:
        raise AppManifestError.from_violations(violations)
    return found


def listed_installed(
    names: Iterable[str], installed: Mapping[str, AppOrigin] | None = None
) -> dict[str, AppOrigin]:
    """The installed applications among *names*, by name.

    Args:
        names: The applications a profile's ``apps:`` block lists.
        installed: From :func:`installed_apps`; ``None`` reads the environment.
    """
    if installed is None:
        installed = installed_apps()
    return {name: installed[name] for name in names if name in installed}


def _package_directory(module: str) -> Path | None:
    """The directory of the package *module*, found without importing it.

    ``None`` when *module* does not resolve or is a plain module. Locating a
    dotted name imports its parent packages, never the package itself.
    """
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    locations = list(spec.submodule_search_locations or ()) if spec is not None else []
    if not locations:
        return None
    return Path(locations[0]).resolve()
