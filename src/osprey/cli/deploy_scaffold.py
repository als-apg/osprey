"""Emitting a deployment repo's CI files, and re-emitting them safely.

A deployment repo holds its ``profile.yml`` at the root and everything the
profile implies around it. Two of those surrounding files are generated from
the profile rather than written by hand: the CI pipeline at the repo root, and
the post-deploy health check at ``scripts/verify.sh``. This module turns a
profile into both files and puts them where they belong — both paths
repo-relative, because the repo is the deployment: there is no project sibling
to key them off.

``osprey scaffold ci`` re-emits them; ``osprey init`` emits them the first
time. One engine, called twice, so a repo created today and a repo
re-scaffolded a year later carry the same two files.

``osprey scaffold systemd`` emits a third file through the same engine: the
boot unit at :data:`SYSTEMD_OUTPUT_NAME`. It is not part of what ``init``
writes, because it is rendered for a HOST rather than for the repo — the two
absolute paths in it are properties of the machine that will run the
deployment, so it is emitted when an operator is standing on that machine and
asks for it.

Re-emission is the whole difficulty. A generated file that lands in a git
repository is read, diffed and eventually edited by people, so the engine has
to answer two questions before it writes anything:

*Is this file still ours?* Every emitted file carries a marker line naming what
generated it. A file without the marker was hand-written, renamed into place,
or predates the scaffolder — the engine refuses to touch it, and ``--force`` is
how an operator says they meant it. The marker is checked in the file's header
only, so a marker quoted further down in prose is not mistaken for provenance.

*Did anything actually change?* The version stamp beside the marker moves with
every OSPREY release, so a byte comparison would rewrite both files on every
upgrade and fill the repo's history with diffs that change nothing. The stamp
is normalized away before comparing: a file whose content is otherwise
identical is left exactly as it is, stamp included. The stamp then says which
release produced the content that is actually there, which is what it is for.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from osprey.errors import ConfigurationError

from .build_profile_deploy import SUPPORTED_CI_PLATFORMS, DeployConfig, parse_deploy_block
from .build_profile_document import _read_profile_document
from .build_profile_merge import resolve_profile_document
from .deploy_scaffold_templates import (
    CI_MARKER,
    CI_TEMPLATES,
    SYSTEMD_MARKER,
    SYSTEMD_TEMPLATE,
    SYSTEMD_UNIT_NAME,
    VERIFY_MARKER,
    VERIFY_PATH,
    VERIFY_TEMPLATE,
    build_ci_context,
    build_systemd_context,
    build_verify_context,
    render,
)
from .repo_resolver import PROFILE_FILENAME

#: CI platform -> the file name its pipeline is emitted under, at the repo
#: root. Separate from the template lookup because the name is the platform's
#: convention rather than ours: GitLab reads ``.gitlab-ci.yml`` and nothing
#: else. Adding a platform therefore means three deliberate edits — a template,
#: an entry here, and a marker of its own (:data:`CI_MARKER` is a single
#: constant only because there is a single platform).
CI_OUTPUT_NAMES: dict[str, str] = {"gitlab": ".gitlab-ci.yml"}

#: Where the health check is emitted, as path segments relative to the repo
#: root. It lands in the source zone, beside the profile that renders it, and
#: nothing copies it anywhere else: the repo is the deployment, so the path the
#: pipeline invokes and the path the file sits at are the same path.
#:
#: Split from the emitted files' own spelling of it rather than written out
#: again — :data:`~.deploy_scaffold_templates.VERIFY_PATH` is what the rendered
#: pipeline tells an operator to run, and the same path the post-up hook looks
#: for after a deploy. A second literal here could move the file out from under
#: both without a single test noticing.
VERIFY_OUTPUT_PATH: tuple[str, ...] = tuple(VERIFY_PATH.split("/"))

#: Where the boot unit is emitted, relative to the repo root. Not the directory
#: it is installed in: a unit takes effect from ``~/.config/systemd/user/``, and
#: writing there directly would put a generated file outside the repo and
#: outside review, in a directory the operator's other units live in. It lands
#: beside the profile instead, under the name ``systemctl`` will know it by, and
#: the emitted header plus the verb's own output say to copy it across.
SYSTEMD_OUTPUT_NAME: str = SYSTEMD_UNIT_NAME

#: The health check is meant to be runnable by hand (``./scripts/verify.sh``),
#: as its own header advertises. The post-up hook runs it through ``bash`` and
#: would not care, but an operator following the header would.
_EXECUTABLE_MODE = 0o755
_REGULAR_MODE = 0o644

#: How far into a file the provenance marker is looked for. The header is the
#: first handful of lines by construction; bounding the search keeps a marker
#: mentioned in prose — a README pasted into a comment, this docstring quoted
#: in a template — from being read as provenance.
_MARKER_SCAN_LINES = 20

_MARKER_RE = re.compile(r"^#\s*osprey-scaffold:\s*(?P<marker>\S+)\s*$")
_VERSION_RE = re.compile(r"^#\s*osprey-version:.*$", re.MULTILINE)

#: What the version stamp is replaced with before two files are compared.
_VERSION_PLACEHOLDER = "# osprey-version:"

#: What an emission did to one file.
#:
#: ``created`` — nothing was there. ``updated`` — ours, and the render differs.
#: ``unchanged`` — ours, and the render matches (nothing was written).
#: ``refused`` — something is there that we did not write, and no ``--force``.
ScaffoldAction = Literal["created", "updated", "unchanged", "refused"]


@dataclass(frozen=True)
class ScaffoldedFile:
    """One file the scaffolder emitted, or declined to.

    Attributes:
        path: Absolute path to the file.
        marker: Provenance marker the file carries, or would have carried.
        action: What happened — see :data:`ScaffoldAction`.
        reason: Why a ``refused`` file was left alone, phrased for an operator.
            Empty for every other action.
    """

    path: Path
    marker: str
    action: ScaffoldAction
    reason: str = ""

    @property
    def written(self) -> bool:
        """Whether this emission put bytes on disk."""
        return self.action in ("created", "updated")

    @property
    def refused(self) -> bool:
        """Whether an existing file was left alone."""
        return self.action == "refused"


def scaffold_deploy_files(
    repo_root: Path,
    *,
    force: bool = False,
    osprey_version: str | None = None,
) -> list[ScaffoldedFile]:
    """Emit a deployment repo's CI pipeline and post-deploy health check.

    Both destinations are fixed properties of the layout rather than parameters:
    the repo root holds the profile, the pipeline sits beside it, and the health
    check goes to :data:`VERIFY_OUTPUT_PATH`. A caller that could move either
    one could put a file somewhere the emitted pipeline does not look, which is
    a failure nothing downstream can detect.

    Args:
        repo_root: The deployment repo — the directory holding ``profile.yml``.
            Both files are written relative to it.
        force: Overwrite files that carry no marker of ours. Without it such a
            file is reported and left alone.
        osprey_version: Version for the provenance stamp. Defaults to the
            installed framework's; tests pass a frozen value.

    Returns:
        One :class:`ScaffoldedFile` per emitted file, in emission order. A
        refusal is reported here rather than raised: the two files have
        independent histories, and a hand-edited pipeline is no reason to leave
        the health check stale.

    Raises:
        ConfigurationError: If the profile is missing, declares no ``deploy:``
            block, or names a CI platform with no pipeline template.
    """
    repo_root = repo_root.resolve()
    profile_file = repo_root / PROFILE_FILENAME
    profile, profile_dir, deploy = _load_deploy_profile(profile_file)

    ci_template = CI_TEMPLATES.get(deploy.ci)
    ci_name = CI_OUTPUT_NAMES.get(deploy.ci)
    if ci_template is None or ci_name is None:
        supported = ", ".join(repr(name) for name in SUPPORTED_CI_PLATFORMS)
        raise ConfigurationError(
            f"{profile_file}: deploy.ci is {deploy.ci!r}, which has no pipeline "
            f"template to render. Supported platforms: {supported}."
        )

    ci_text = render(
        ci_template,
        build_ci_context(profile, deploy, profile_dir, repo_root.name, osprey_version),
    )
    verify_text = render(VERIFY_TEMPLATE, build_verify_context(profile, osprey_version))

    return [
        _emit(
            repo_root / ci_name,
            ci_text,
            CI_MARKER,
            mode=_REGULAR_MODE,
            force=force,
            command="osprey scaffold ci",
        ),
        _emit(
            repo_root.joinpath(*VERIFY_OUTPUT_PATH),
            verify_text,
            VERIFY_MARKER,
            mode=_EXECUTABLE_MODE,
            force=force,
            command="osprey scaffold ci",
        ),
    ]


def scaffold_systemd_unit(
    repo_root: Path,
    *,
    force: bool = False,
    osprey_bin: str | None = None,
    osprey_version: str | None = None,
) -> ScaffoldedFile:
    """Emit the boot unit that brings this deployment up after a reboot.

    Rendered from the profile's ``name:`` and from two paths on the machine the
    unit will run on: the repo, and the ``osprey`` executable. No ``deploy:``
    block is read, because none of it applies — a unit runs the deployment where
    it already is, rather than shipping it anywhere.

    Args:
        repo_root: The deployment repo. Also the unit's ``WorkingDirectory``,
            resolved absolute.
        force: Overwrite a file that carries no marker of ours.
        osprey_bin: Absolute path to the ``osprey`` executable the unit invokes.
            Defaults to the one resolvable here.
        osprey_version: Version for the provenance stamp. Defaults to the
            installed framework's; tests pass a frozen value.

    Returns:
        What the emission did to the one file it writes.

    Raises:
        ConfigurationError: If the repo holds no profile, or holds one that is
            not a YAML mapping.
        FileNotFoundError: If no ``osprey`` executable can be found and none was
            given. A unit naming a command that is not there fails at boot,
            where nobody is watching, so it is not written.
    """
    repo_root = repo_root.resolve()
    profile, _ = _load_profile(repo_root / PROFILE_FILENAME, command="osprey scaffold systemd")

    text = render(
        SYSTEMD_TEMPLATE,
        build_systemd_context(profile, repo_root, osprey_bin, osprey_version),
    )
    return _emit(
        repo_root / SYSTEMD_OUTPUT_NAME,
        text,
        SYSTEMD_MARKER,
        mode=_REGULAR_MODE,
        force=force,
        command="osprey scaffold systemd",
    )


def _load_profile(profile_file: Path, *, command: str) -> tuple[dict[str, Any], Path]:
    """Read the profile a scaffolding verb renders from.

    Resolved the same way a build resolves it — aliases normalized, ``extends:``
    followed, a persona delta anchored at its root — so a file is rendered from
    what the profile *means* rather than from what one file happens to say.

    The profile is not otherwise validated: ``osprey validate`` is that check,
    and the emitted pipeline runs it as its first job. Scaffolding a repo whose
    data tree is not yet populated is a normal thing to do.

    Args:
        profile_file: The repo's ``profile.yml``.
        command: The verb asking, named in the message an operator sees when
            there is no profile there. The two verbs render different files from
            the same profile, and "run it from a repo holding profile.yml" is
            only actionable if it says which command to re-run.
    """
    if not profile_file.is_file():
        raise ConfigurationError(
            f"No profile at {profile_file}. '{command}' renders from a "
            f"deployment repo's profile — run it from a repo holding "
            f"{PROFILE_FILENAME}, or create one with 'osprey init'."
        )

    raw = _read_profile_document(profile_file)
    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"{profile_file} must be a YAML mapping, got {type(raw).__name__}."
        )
    document = resolve_profile_document(raw, profile_file.resolve())
    return document.raw, document.root_dir


def _load_deploy_profile(profile_file: Path) -> tuple[dict[str, Any], Path, DeployConfig]:
    """Read the profile, and the deployment coordinates the CI files need."""
    raw, root_dir = _load_profile(profile_file, command="osprey scaffold ci")

    deploy = parse_deploy_block(raw)
    if deploy is None:
        raise ConfigurationError(
            f"{profile_file} declares no 'deploy:' block, so there are no "
            f"deployment coordinates to render a pipeline from. An emitted "
            f"profile carries a commented 'deploy:' example among the commented "
            f"blocks at the end of the file — uncomment it, fill in the CI "
            f"platform, the registry and the deploy host, then re-run."
        )
    return raw, root_dir, deploy


def _emit(
    path: Path, text: str, marker: str, *, mode: int, force: bool, command: str
) -> ScaffoldedFile:
    """Write one emitted file, unless what is already there says not to."""
    existing = _read_existing(path)

    if existing is not None:
        if _marker_of(existing) != marker and not force:
            reason = (
                f"carries no '{marker}' marker, so it was not written by the "
                f"scaffolder. Re-run with '{command} --force' to replace it."
            )
            return ScaffoldedFile(path, marker, "refused", reason)
        if _normalized(existing) == _normalized(text):
            return ScaffoldedFile(path, marker, "unchanged")

    _write_atomically(path, text, mode)
    return ScaffoldedFile(path, marker, "created" if existing is None else "updated")


def _read_existing(path: Path) -> str | None:
    """The file already at *path*, or ``None`` when there is nothing readable.

    A file that cannot be decoded as text is returned as an empty string rather
    than as "absent": it exists, it is not ours, and overwriting it silently is
    exactly what the marker check is there to prevent.
    """
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError):
        return ""


def _marker_of(text: str) -> str | None:
    """The provenance marker in a file's header, if it carries one."""
    for line in text.splitlines()[:_MARKER_SCAN_LINES]:
        match = _MARKER_RE.match(line)
        if match:
            return match.group("marker")
    return None


def _normalized(text: str) -> str:
    """A file's content with the release-dependent version stamp masked."""
    return _VERSION_RE.sub(_VERSION_PLACEHOLDER, text)


def _write_atomically(path: Path, text: str, mode: int) -> None:
    """Replace *path* in one step, so no reader ever sees a half-written file.

    A pipeline is read by CI and a health check by the post-up hook or by an
    operator, any of which can be running while a scaffold is. Writing to a
    temporary file in the destination directory and renaming it over the target
    keeps those readers seeing one version or the other, never a truncated one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(temp_path, mode)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
