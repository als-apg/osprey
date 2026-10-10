"""The packaged data files a preset materializes into a profile's ``data/``.

A preset names two packaged sources: an app template (``app_template:``,
``templates/apps/<name>/``) and a bundled facility (``facility:``,
``templates/facilities/<name>/``). A facility is owned by no app template and
no preset; it lands in the profile at ``data/facility/``, beside the app
template's own ``data/`` tree, so the operator's layout is the same whichever
preset produced it.

:class:`PresetData` is that one composition. Everything that reads a preset's
packaged data — ``osprey init`` materializing a profile, ``osprey scaffold
pull`` listing and copying it — reads it through here, so a facility file is
indistinguishable from one the app template ships itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from osprey.errors import BuildProfileError

#: The directory under the template root holding the bundled facilities.
FACILITIES_DIRNAME = "facilities"

#: Where a facility lands inside a profile's ``data/`` tree.
FACILITY_DATA_DIRNAME = "facility"


@dataclass(frozen=True)
class PresetData:
    """One preset's packaged data: an app template's ``data/`` plus its facility.

    Attributes:
        app_root: The app template directory (``templates/apps/<name>``). Its
            ``data/`` tree, when it ships one, is copied as it stands.
        facility_root: The bundled facility directory
            (``templates/facilities/<name>``), landed at ``data/facility/``;
            ``None`` when the preset names no facility.
    """

    app_root: Path
    facility_root: Path | None = None

    @property
    def app_data(self) -> Path:
        """The app template's own ``data/`` tree; it may not exist."""
        return self.app_root / "data"

    def placed_files(self) -> dict[str, Path]:
        """The files that land in ``data/`` from outside the app template's ``data/``.

        The facility's files, at ``facility/<path>``.

        Returns:
            Data-relative POSIX path -> the packaged file it is copied from,
            sorted by path.
        """
        files: dict[str, Path] = {}
        if self.facility_root is not None:
            for path in sorted(self.facility_root.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(self.facility_root).as_posix()
                    files[f"{FACILITY_DATA_DIRNAME}/{relative}"] = path
        return dict(sorted(files.items()))

    def copy_into(
        self,
        destination: Path,
        ignore: Callable[[str, list[str]], set[str]] | None = None,
    ) -> None:
        """Write the composed ``data/`` tree at *destination*, which must not exist.

        The app template's ``data/`` is copied as it stands (filtered by
        *ignore*, a :func:`shutil.copytree` ignore callable), then every
        :meth:`placed_files` entry lands beside it, byte-identical.
        """
        import shutil

        placed = self.placed_files()
        if self.app_data.is_dir():
            shutil.copytree(self.app_data, destination, ignore=ignore)
        else:
            destination.mkdir(parents=True)
        for relative, source in placed.items():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def app_template_root(template_root: Path, app_template: str) -> Path:
    """The packaged app-template directory *app_template* names.

    Checked up front, before anything is written, so a packaging regression
    surfaces as an actionable error rather than as a missing file mid-copy.

    Raises:
        BuildProfileError: If the template is absent from the installation.
    """
    root = Path(template_root) / "apps" / app_template
    if not root.is_dir():
        raise BuildProfileError(
            f"App template {app_template!r} is absent from this installation at "
            f"{root}. This is a packaging bug — reinstall osprey-framework."
        )
    return root


def facility_root(template_root: Path, facility: str) -> Path:
    """The bundled facility directory *facility* names.

    Raises:
        BuildProfileError: If the facility is absent from the installation.
    """
    root = Path(template_root) / FACILITIES_DIRNAME / facility
    if not root.is_dir():
        raise BuildProfileError(
            f"Facility {facility!r} is absent from this installation at "
            f"{root}. This is a packaging bug — reinstall osprey-framework."
        )
    return root


def compose_preset_data(template_root: Path, app_template: str, facility: str | None) -> PresetData:
    """Compose the packaged data a preset naming *app_template* and *facility* copies.

    Args:
        template_root: The installed template root.
        app_template: The app template the preset names.
        facility: The bundled facility the preset names, or ``None``.

    Returns:
        The composition, with both directories checked to exist.

    Raises:
        BuildProfileError: If either directory is absent, or the preset names a
            facility while its app template still ships a ``data/facility/`` of
            its own — two facilities for one ``data/facility/``.
    """
    app_root = app_template_root(template_root, app_template)
    if facility is None:
        return PresetData(app_root)
    if (app_root / "data" / FACILITY_DATA_DIRNAME).exists():
        raise BuildProfileError(
            f"App template {app_template!r} ships data/{FACILITY_DATA_DIRNAME}/ and its "
            f"preset names facility {facility!r}: remove the app template's copy."
        )
    return PresetData(app_root, facility_root(template_root, facility))
