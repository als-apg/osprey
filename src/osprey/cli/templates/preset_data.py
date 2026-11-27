"""The packaged files a preset materializes into a deployment's source trees.

A preset names two packaged sources: an app template (``app_template:``,
``templates/apps/<name>/``) and a bundled facility (``facility:``,
``templates/facilities/<name>/``). A facility is owned by no app template and
no preset; it lands in the deployment at ``facility/``, beside the trees the
app template ships itself (its ``web/`` content), so the operator's layout is
the same whichever preset produced it. The starter reviewer prompt (``templates/code_review/``) is
owned by no preset either: every composition lands it in the code reviewer's
directory (``agent/code_review/reviewer.md``), and no preset names it.

:class:`PresetData` is that one composition. Everything that reads a preset's
packaged data — ``osprey init`` materializing a profile, ``osprey scaffold
pull`` listing and copying it — reads it through here, so a facility file is
indistinguishable from one the app template ships itself. Every text file of a
bundled facility opens with the bundled-origin line (``osprey.facility.bundled``),
except the knowledge indexes ``osprey knowledge regen-index`` writes.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from osprey import repo_layout
from osprey_connectors.errors import BuildProfileError

#: The directory under the template root holding the bundled facilities.
FACILITIES_DIRNAME = "facilities"

#: Where a facility lands, relative to the deployment root.
FACILITY_DATA_DIRNAME = repo_layout.entry("facility").path

#: The starter reviewer prompt under the template root.
_STARTER_PROMPT_TEMPLATE = "code_review/reviewer.md"

#: Where the starter reviewer prompt lands, relative to the repo root: the code
#: reviewer's directory. ``osprey.services.code_review.settings.DEFAULT_PROMPT``
#: names the same file.
STARTER_PROMPT_PATH = repo_layout.display_path(
    "code_review", PurePosixPath(_STARTER_PROMPT_TEMPLATE).name
)


def _web_dir(root: Path) -> Path:
    """The web tree of *root*, a tree laid out as a deployment root (an app template)."""
    return root.joinpath(*repo_layout.entry("web").path.split("/"))


@dataclass(frozen=True)
class PresetData:
    """One preset's packaged data: an app template's own trees, its facility, the starter prompt.

    Attributes:
        app_root: The app template directory (``templates/apps/<name>``). Its
            ``web/`` tree, when it ships one, is copied as it stands.
        facility_root: The bundled facility directory
            (``templates/facilities/<name>``), landed at ``facility/``;
            ``None`` when the preset names no facility.
        starter_prompt: The packaged starter reviewer prompt, landed at
            :data:`STARTER_PROMPT_PATH`; ``None`` for a bare tree that was not
            composed from a preset.
    """

    app_root: Path
    facility_root: Path | None = None
    starter_prompt: Path | None = None

    @property
    def app_web(self) -> Path:
        """The app template's own ``web/`` tree; it may not exist."""
        return _web_dir(self.app_root)

    def placed_files(self) -> dict[str, Path]:
        """The files that land in the repo from outside the app template's own trees.

        The facility's files, at ``facility/<path>``, and the starter
        reviewer prompt at :data:`STARTER_PROMPT_PATH`.

        Returns:
            Repo-relative POSIX path -> the packaged file it is copied from,
            sorted by path.
        """
        files: dict[str, Path] = {}
        if self.facility_root is not None:
            for path in sorted(self.facility_root.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(self.facility_root).as_posix()
                    files[f"{FACILITY_DATA_DIRNAME}/{relative}"] = path
        if self.starter_prompt is not None:
            files[STARTER_PROMPT_PATH] = self.starter_prompt
        return dict(sorted(files.items()))

    def copy_into(self, repo_root: Path) -> None:
        """Write the composition into *repo_root*, which must hold none of its trees.

        The app template's ``web/``, when it ships one, is copied as it stands,
        the facility directory is created, then every :meth:`placed_files` entry
        lands at its repo-relative path, byte-identical. No ``web/`` is
        created for a template that ships none.
        """
        placed = self.placed_files()
        if self.app_web.is_dir():
            shutil.copytree(self.app_web, _web_dir(repo_root))
        (repo_root / FACILITY_DATA_DIRNAME).mkdir(parents=True)
        for relative, source in placed.items():
            target = repo_root.joinpath(*relative.split("/"))
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


def starter_prompt_path(template_root: Path) -> Path:
    """The packaged starter reviewer prompt.

    Raises:
        BuildProfileError: If the prompt is absent from the installation.
    """
    path = Path(template_root) / _STARTER_PROMPT_TEMPLATE
    if not path.is_file():
        raise BuildProfileError(
            f"The starter reviewer prompt is absent from this installation at "
            f"{path}. This is a packaging bug — reinstall osprey-framework."
        )
    return path


def compose_preset_data(template_root: Path, app_template: str, facility: str | None) -> PresetData:
    """Compose the packaged data a preset naming *app_template* and *facility* copies.

    Args:
        template_root: The installed template root.
        app_template: The app template the preset names.
        facility: The bundled facility the preset names, or ``None``.

    Returns:
        The composition, with everything it names checked to exist.

    Raises:
        BuildProfileError: If a packaged source is absent, the preset names a
            facility while its app template still ships a ``facility/`` of
            its own — two facilities for one ``facility/`` — or the app
            template ships a file at the starter prompt's path, which landing
            the starter would silently overwrite.
    """
    app_root = app_template_root(template_root, app_template)
    starter = starter_prompt_path(template_root)
    if app_root.joinpath(*STARTER_PROMPT_PATH.split("/")).exists():
        raise BuildProfileError(
            f"App template {app_template!r} ships {STARTER_PROMPT_PATH}, where every "
            f"preset lands the starter reviewer prompt: remove the app template's copy."
        )
    if facility is None:
        return PresetData(app_root, starter_prompt=starter)
    if (app_root / FACILITY_DATA_DIRNAME).exists():
        raise BuildProfileError(
            f"App template {app_template!r} ships "
            f"{repo_layout.display_path('facility')}/ and its "
            f"preset names facility {facility!r}: remove the app template's copy."
        )
    return PresetData(app_root, facility_root(template_root, facility), starter)
