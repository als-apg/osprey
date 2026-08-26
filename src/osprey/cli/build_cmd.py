"""Build command — render a deployment repo's ``build/`` zone from its profile.

``osprey build``, run anywhere inside a deployment repo, renders that repo's
OUTPUT zone: ``build/`` is the rendered project — ``config.yml``, ``.mcp.json``,
``.claude/``, the service tree, the compose files, the project venv — derived in
full from the SOURCE zone (``profile.yml`` at the repo root and the trees it
names). Nothing durable lives there, so it is wiped and re-rendered whole every
time; ``rm -rf build/`` loses nothing.

A repo with a ``personas/`` directory renders more than one project: the
deployment's own, plus ``build/<repo>-<persona>/`` for every delta in there
(:func:`_render_persona_projects`). That is the whole of when a persona project
is written — no start verb renders one — so ``build/`` is a complete account of
what a deploy will run, personas included.

The render is atomic. It lands in ``build/.tmp/`` and is swapped in by rename
only once every step has succeeded, so a build that fails — or is killed
mid-flight — leaves the previous ``build/`` exactly as it was, still able to
``osprey down`` the stack it started. :func:`_swap_in_render` documents the
rename sequence and what each failure point leaves behind.

Usage:
    osprey build                 # render this repo's build/
    osprey build --repo PATH     # …or another repo's, without cd-ing to it

The build pipeline's helper concerns live in sibling modules that this command
orchestrates: venv + ``.env`` templating in :mod:`osprey.cli.build_environment`,
lifecycle-phase execution in :mod:`osprey.cli.build_lifecycle`, service
injectors in :mod:`osprey.cli.build_injectors`, and project-directory
persistence (config overrides, convention artifacts, MCP servers, git init) in
:mod:`osprey.cli.build_persistence`. They are re-exported below so
``from osprey.cli.build_cmd import <helper>`` keeps working.
"""

from __future__ import annotations

import errno
import json
import os
import shlex
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

import click

from osprey.deployment.compose_merge import MERGED_COMPOSE_FILENAME
from osprey.errors import BuildProfileError
from osprey.utils.logger import get_logger
from osprey.utils.workspace import (
    BUILD_DIR_NAME,
    IMAGE_DIR_NAME,
    STATE_DIR_NAME,
    STATE_ZONE_DIRS,
)

from .build_environment import (
    _create_project_venv,
    _resolve_osprey_spec,
    report_provider_credentials,
)
from .build_injectors import (
    _copy_service_templates,
    _inject_bluesky,
    _inject_bluesky_web,
    _inject_dispatch,
    _inject_gchat_bridge,
    _inject_nextcloud_bridge,
    _inject_profile_services,
    _inject_va,
    _inject_va_archiver,
    _locate_pkg_services,
)
from .build_lifecycle import (
    _SHELL_METACHARACTERS,
    _format_junit_summary,
    _run_lifecycle_phase,
)
from .build_persistence import (
    _apply_config_overrides,
    _apply_conventions,
    _persist_artifact_server,
    _persist_mcp_servers,
    _profile_known_root_entries,
    _register_convention_artifacts,
    _resolve_context_roster,
)
from .repo_resolver import PROFILE_FILENAME, find_repo_root, repo_option
from .templates.manager import TemplateManager

logger = get_logger("build")


def _report_fact(message: str) -> None:
    """Report ``message`` under this module's logger.

    The promotion contract lives in :func:`osprey.cli.output.report_fact`; this
    binds it to the build logger so call sites pass the line alone.

    Args:
        message: The finished line, built by the caller.
    """
    from . import output

    output.report_fact(logger, message)


__all__ = [
    "_SHELL_METACHARACTERS",
    "_apply_config_overrides",
    "_apply_conventions",
    "_copy_service_templates",
    "_create_project_venv",
    "_format_junit_summary",
    "_inject_bluesky",
    "_inject_bluesky_web",
    "_inject_dispatch",
    "_inject_gchat_bridge",
    "_inject_nextcloud_bridge",
    "_inject_profile_services",
    "_inject_va",
    "_inject_va_archiver",
    "_locate_pkg_services",
    "_persist_artifact_server",
    "_persist_mcp_servers",
    "_profile_known_root_entries",
    "_register_convention_artifacts",
    "_resolve_context_roster",
    "_resolve_osprey_spec",
    "_run_lifecycle_phase",
    "build",
    "report_provider_credentials",
]


# ---------------------------------------------------------------------------
# Four-zone repo build: the atomic render
# ---------------------------------------------------------------------------

#: Staging root, inside the output zone it replaces. Inside rather than beside
#: it so a half-written render is obviously part of the zone that owns it, and
#: so ``build/`` remains the only directory a repo's ``.gitignore`` has to name.
_STAGE_DIRNAME = ".tmp"

#: Repo-root entries that are NOT source, and so never enter a container image:
#: the two derived zones, git's own directory, the merged compose document a
#: deploy writes at the root, and every ``.env`` variant. This is the
#: ``.gitignore`` the emitted repo ships, said in Python — a container gets what
#: a fresh clone gets. Secrets are excluded here rather than left to the image's
#: ``.dockerignore``, because this is the copy that decides what the build
#: context contains at all.
_NON_SOURCE_ROOT_ENTRIES: frozenset[str] = frozenset(
    {BUILD_DIR_NAME, STATE_DIR_NAME, ".git", MERGED_COMPOSE_FILENAME}
)

#: The interpreter every OSPREY process inside a container image is launched
#: with — MCP servers, framework hooks, and the registry's
#: ``{current_python_env}`` substitution alike.
#:
#: Pinned rather than derived because the render happens HERE and the
#: interpreter exists THERE: the derivation
#: (:func:`~osprey.cli.templates.claude_code._derive_runtime_interpreter`) can
#: only answer from the filesystem it is standing on, and for a ``--runtime-root``
#: render its honest answer is this machine's — a path no container has, which
#: leaves every server and every hook in the image unable to start. Its value is
#: the interpreter of ``templates/project/Dockerfile.j2``'s
#: ``FROM python:3.12-slim`` base, which the official python images install at
#: ``/usr/local/bin/python``. The coupling to that base image is pinned by a
#: test, so a base change cannot silently invalidate this constant.
_CONTAINER_INTERPRETER = "/usr/local/bin/python"

#: The two directories the swap renames through, in the STATE zone. They live
#: under ``var/`` — already git-ignored and on the same filesystem as
#: ``build/`` — so a swap interrupted between two renames leaves nothing in
#: ``git status`` and nothing that a later rename could cross a device boundary
#: to reach.
_INCOMING_DIRNAME = ".osprey-build-incoming"
_OUTGOING_DIRNAME = ".osprey-build-outgoing"

#: The errnos that mean "the filesystem will not let go of this yet", as opposed
#: to "this code is wrong about the path". ENOTEMPTY is what a directory reports
#: once a sweep has taken everything it could and an NFS ``.nfs*`` sillyrename
#: is all that is left; EBUSY is the same story for a mount or an open file;
#: EACCES/EPERM is a subtree written by another user, which for a deployment
#: repo is usually a container that ran as root. Every one of them can clear on
#: its own — the client drops its reference, the mount goes — so they are worth
#: one retry. Anything else is re-raised.
_LEFTOVER_ERRNOS: frozenset[int] = frozenset(
    {errno.ENOTEMPTY, errno.EBUSY, errno.EACCES, errno.EPERM}
)

#: How long to wait before the single retry. Long enough for an NFS client to
#: release a sillyrenamed file, short enough that a build nobody can fix anyway
#: still fails promptly.
_LEFTOVER_RETRY_SECONDS = 1.0

#: Marks a leftover that could not be deleted and was renamed out of the way
#: instead. Carries a unique suffix, because a second stuck build must not
#: collide with the first one's rescue.
_HELD_ASIDE_SUFFIX = ".undeletable"

#: The STATE zone a build guarantees exists — created empty, and otherwise the
#: agent's to write, not the build's. Imported from the conventions module that
#: owns the zone layout rather than spelled again: ``osprey init`` emits the
#: same pair, and a build that recreated a DIFFERENT pair would make a reset
#: repo and a fresh clone stop looking alike.
_STATE_DIRS: tuple[str, ...] = STATE_ZONE_DIRS

#: Where the outgoing render's Claude Code artifacts are snapshotted before the
#: swap replaces them — the durable zone, so the snapshot survives the wipe it
#: exists to protect against.
_BACKUP_RELDIR = f"{STATE_DIR_NAME}/agent_data/backup"

#: The rendered files a Claude Code regeneration owns, relative to the render
#: root. Snapshotted before an overwrite that would change them.
_CLAUDE_ARTIFACT_ROOTS: tuple[str, ...] = (".claude", ".mcp.json", "CLAUDE.md")


class _RenderZones(NamedTuple):
    """The four paths one atomic render moves between.

    Named as a group because the swap's correctness is a property of the
    *sequence* of renames between them, and a helper holding three of the four
    could not state that property.
    """

    repo_root: Path
    """The deployment repo — the directory holding ``profile.yml``."""

    build_dir: Path
    """The OUTPUT zone, ``<repo>/build``. What the swap replaces."""

    stage: Path
    """Where this render is written: ``<repo>/build/.tmp/<repo name>``.

    One level below the staging root because
    :meth:`~osprey.cli.templates.manager.TemplateManager.create_project`
    renders into ``<output_dir>/<project_name>`` — so the staging root is the
    output directory and the deployment name is the leaf, exactly as it is for
    every other render.
    """

    incoming: Path
    """``<repo>/var/.osprey-build-incoming`` — the completed render, mid-swap."""

    outgoing: Path
    """``<repo>/var/.osprey-build-outgoing`` — the replaced render, mid-swap."""

    @property
    def stage_root(self) -> Path:
        """The staging root, ``<repo>/build/.tmp`` — removed with the old build."""
        return self.stage.parent


def _render_zones(repo_root: Path) -> _RenderZones:
    """The paths one ``osprey build`` of *repo_root* moves between."""
    build_dir = repo_root / BUILD_DIR_NAME
    return _RenderZones(
        repo_root=repo_root,
        build_dir=build_dir,
        stage=build_dir / _STAGE_DIRNAME / repo_root.name,
        incoming=repo_root / STATE_DIR_NAME / _INCOMING_DIRNAME,
        outgoing=repo_root / STATE_DIR_NAME / _OUTGOING_DIRNAME,
    )


def _ensure_state_zone(repo_root: Path) -> None:
    """Create the ``var/`` skeleton when it is absent (idempotent).

    A build is the first command a fresh clone runs, and a clone carries no
    git-ignored directory: ``var/agent_data`` and ``var/audit`` have to exist
    before anything writes into them. Creating them here rather than at ``up``
    keeps a cloned repo and an ``osprey init``-ed one identical from the first
    command either one runs.

    Existing content is left exactly as it is — this is the one directory a
    build must never touch, and ``mkdir -p`` is the whole operation.
    """
    for relative in _STATE_DIRS:
        (repo_root / relative).mkdir(parents=True, exist_ok=True)


def _is_leftover_stuck(error: OSError) -> bool:
    """Whether *error* is the filesystem holding on, rather than a wrong path.

    NFS reports its sillyrenamed files by name rather than by errno on some
    clients, so the message is checked too.
    """
    return error.errno in _LEFTOVER_ERRNOS or "nfs" in str(error).lower()


def _clear_leftover(path: Path, *, aside_root: Path) -> None:
    """Free the name *path* occupies — by deleting it, or by moving it aside.

    Every caller is about to ``rename`` something onto this path, and
    ``rename(2)`` onto a non-empty directory fails. So the name has to end up
    free, and "we tried to delete it" is not the same thing: the failure mode
    this replaces was a swallowed ``rmtree`` whose leftover surfaced as a bare
    ENOTEMPTY from a rename several hundred lines away, on this build and on
    every build after it, with nothing in the message about which path or why.

    Three attempts at freeing the name, in order:

    1. Delete it. The overwhelmingly common case, and the only one with no
       trace left behind.
    2. Delete it again, after :data:`_LEFTOVER_RETRY_SECONDS`, if it is still
       standing — for one of the reasons in :data:`_LEFTOVER_ERRNOS`, which are
       the ones that clear on their own, or for no stated reason at all. Any
       other ``OSError`` is this code being wrong about the path, and is
       re-raised untouched.
    3. Rename it aside, under *aside_root*, with a unique suffix. The tree is
       still on disk and still undeletable, but it is no longer in the way, so
       the build proceeds and the operator is told where it went.

    Only if the rename aside fails too does the caller fail — and then with the
    path, the reason it could not be deleted, and the reason it could not be
    moved, all named.

    :param path: The file or directory to clear. Absent is success.
    :param aside_root: Where an undeletable tree is parked. Must be on the same
        filesystem as *path*, since step 3 is a rename; both callers pass a
        directory inside the same repo.
    :raises click.ClickException: When the name could not be freed at all.
    """
    reason: OSError | None = None
    for attempt in (1, 2):
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink()
        except FileNotFoundError:
            pass
        except OSError as error:
            if not _is_leftover_stuck(error):
                raise
            reason = error
        # The gone-ness is checked rather than inferred from a call that
        # returned: what the name looks like now is the whole question, and a
        # removal that reports success over a tree that is still standing is
        # exactly the silence this function replaces.
        if not path.exists() and not path.is_symlink():
            return
        if attempt == 1:
            time.sleep(_LEFTOVER_RETRY_SECONDS)

    detail = reason if reason is not None else "it is still there afterwards"
    aside = aside_root / f"{path.name}{_HELD_ASIDE_SUFFIX}.{uuid4().hex[:8]}"
    try:
        aside_root.mkdir(parents=True, exist_ok=True)
        os.replace(path, aside)
    except OSError as error:
        raise click.ClickException(
            f"Could not clear {path}: {detail}\n"
            f"Nor move it aside to {aside}: {error}\n"
            "Remove that path yourself — as its owner, or once whatever holds "
            "it open has stopped — and run the command again."
        ) from error

    logger.warning("  Could not delete %s (%s)", path, detail)
    logger.warning("  Moved it aside to %s — delete it when you can", aside)


def _repair_interrupted_swap(zones: _RenderZones) -> None:
    """Restore a ``build/`` left missing by an interrupted swap, then clean up.

    :func:`_swap_in_render` has one window — between two consecutive renames,
    with no I/O in between — where ``build/`` does not exist while both the old
    and the new render sit under ``var/``. A process killed exactly there would
    otherwise leave a repo with no output zone at all and two unexplained
    directories beside it.

    The rule is one sentence: **``build/`` wins when it exists; otherwise the
    incoming render, then the outgoing one.** A build that did not report
    success never happened, so an incoming render found *beside* a present
    ``build/`` is discarded rather than adopted — the last render an operator
    was told about is the one they keep.

    The sweep afterwards is what makes the repair self-healing rather than
    self-defeating. Each of the three names it clears is a name the NEXT
    :func:`_swap_in_render` renames onto, so a leftover left standing does not
    stay quiet — it fails that rename with a bare ENOTEMPTY, and keeps failing
    it forever. :func:`_clear_leftover` therefore frees each name or says why
    it could not.
    """
    if not zones.build_dir.exists():
        for candidate in (zones.incoming, zones.outgoing):
            if candidate.is_dir():
                logger.warning("  Recovering build/ from an interrupted build (%s)", candidate.name)
                os.replace(candidate, zones.build_dir)
                break

    # Parked in the STATE zone rather than beside each leftover: `var/` is
    # git-ignored, is on the same filesystem as all three, and is the one of
    # the four zones a build never rewrites — so a tree parked there survives
    # the swap that is about to replace `build/` instead of riding along inside
    # it and coming back as the next build's leftover.
    aside_root = zones.repo_root / STATE_DIR_NAME
    for leftover in (zones.incoming, zones.outgoing, zones.stage_root):
        _clear_leftover(leftover, aside_root=aside_root)


def _swap_in_render(zones: _RenderZones) -> None:
    """Replace ``build/`` with the completed render, by rename only.

    The sequence, and what a process killed at each point leaves behind:

    ==== ================================== =========================================
    Step Operation                          State if killed immediately after
    ==== ================================== =========================================
    0    (the whole render, into ``.tmp``)  ``build/`` is the previous render, intact
    1    ``build/.venv`` -> ``stage/.venv`` ``build/`` intact, minus its venv
    2    ``stage``       -> ``incoming``    ``build/`` intact (still the old render)
    3    ``build``       -> ``outgoing``    **no ``build/``** — repaired on next build
    4    ``incoming``    -> ``build``       ``build/`` is the new render; junk in var/
    5    remove ``outgoing``                done
    ==== ================================== =========================================

    Step 0 is where every realistic failure happens: it is the whole render,
    seconds to minutes of work, and it cannot touch ``build/`` because it writes
    somewhere else entirely. Steps 1-4 are four ``rename(2)`` calls with nothing
    between them, all within one repo and therefore one filesystem, so each is
    atomic and the whole sequence is over in microseconds. Only step 3 leaves no
    ``build/``, and :func:`_repair_interrupted_swap` restores it from
    ``incoming`` at the start of the next build.

    The venv moves *into* the staged tree (step 1) rather than being rendered
    there, because a virtual environment is the one artifact that records its
    own absolute location — in the shebang of every console script, in
    ``activate``'s ``VIRTUAL_ENV``. It is therefore created at the path it will
    be used from, ``build/.venv``, transits the swap with everything else, and
    lands back at exactly that path. A venv rendered in the staging directory
    would arrive at its destination quietly broken.
    """
    venv = zones.build_dir / ".venv"
    if venv.is_dir():
        os.replace(venv, zones.stage / ".venv")
    os.replace(zones.stage, zones.incoming)
    os.replace(zones.build_dir, zones.outgoing)
    os.replace(zones.incoming, zones.build_dir)
    shutil.rmtree(zones.outgoing, ignore_errors=True)


#: Runtime-state directories a renderer must never leave inside the tree it is
#: rendering — the two spellings of "agent data anchored on the render root",
#: from the layout where a project directory WAS its own runtime root. Under
#: the four-zone layout the durable location is ``<repo>/var/agent_data``,
#: created by :func:`_ensure_state_zone` before the render and never touched by
#: it, so either of these appearing under a render is state the next
#: ``osprey build`` would discard along with the rest of ``build/``.
_STAGE_RUNTIME_STATE: tuple[str, ...] = ("_agent_data", f"{STATE_DIR_NAME}/agent_data")


def _prune_runtime_state_from_stage(zones: _RenderZones, *, renders: Sequence[Path] = ()) -> None:
    """Drop runtime-state directories a render wrote into the staged tree.

    ``build/`` is output: 100% derived, wiped and re-rendered by every build.
    Nothing durable may live there, and ``rm -rf build/`` losing nothing is the
    property the whole zone layout rests on.

    No renderer writes agent data into its own render, so on a correct build
    this finds nothing and removes nothing. It is kept as the invariant's
    enforcement point rather than dropped as dead code, because the failure it
    prevents is silent in both directions: a
    renderer that starts writing state under the tree it is rendering neither
    fails nor logs, and the state it wrote is deleted at the next build with
    nothing to say it existed. Stating the rule once, in the one place that owns
    the staged tree, is cheaper than re-auditing every renderer whenever one
    gains a ``mkdir``.

    Runs before the swap, so ``build/`` is never published with the directories
    in it. Best-effort: a directory that cannot be removed is a cosmetic wart,
    never a reason to fail a render that otherwise succeeded.

    Args:
        zones: The render's paths.
        renders: Every render root this build produced, the staged tree
            included. A persona project is rendered by the same three producers
            and so grows the same directories one level deeper; passing the
            roots rather than walking the tree keeps the rule "each render's own
            top-level runtime state" rather than "anything anywhere that looks
            like it".
    """
    for render in renders or (zones.stage,):
        for relative in _STAGE_RUNTIME_STATE:
            target = render / relative
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
        # `var/` itself only existed to hold agent_data; leave it only if the
        # render put something else there.
        state = render / STATE_DIR_NAME
        if state.is_dir() and not any(state.iterdir()):
            state.rmdir()


def _backup_outgoing_claude_artifacts(zones: _RenderZones) -> Path | None:
    """Snapshot the outgoing render's Claude Code artifacts that are about to change.

    ``build/`` is disposable and not a place to edit anything — ``osprey
    scaffold claim`` is how a file becomes source — but an operator can still
    edit a rendered hook or agent by mistake, and a snapshot that lets them get
    it back costs nothing.

    Only files that actually differ are copied, so a rebuild that changes
    nothing leaves no backup directory. Best-effort throughout: this protects
    against a mistake, and must never itself become one that fails a build.

    Returns:
        The backup directory, or ``None`` when nothing differed.
    """
    from datetime import UTC, datetime

    try:
        changed: list[Path] = []
        for entry in _CLAUDE_ARTIFACT_ROOTS:
            source = zones.build_dir / entry
            if source.is_file():
                candidates = [source]
            elif source.is_dir():
                candidates = [path for path in source.rglob("*") if path.is_file()]
            else:
                continue
            for path in candidates:
                relative = path.relative_to(zones.build_dir)
                incoming = zones.stage / relative
                if not incoming.is_file() or incoming.read_bytes() != path.read_bytes():
                    changed.append(path)
        if not changed:
            return None

        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_dir = zones.repo_root / _BACKUP_RELDIR / f"claude-code-{stamp}"
        for path in changed:
            destination = backup_dir / path.relative_to(zones.build_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        return backup_dir
    except OSError as exc:
        logger.debug("Claude Code backup skipped: %s", exc)
        return None


def _stamp_repo_manifest(
    render_dir: Path, repo_root: Path, profile_path: Path, *, key_digests: bool
) -> None:
    """Add the drift check's per-key commentary to a render's manifest.

    ``reproducible_command`` is deliberately not rewritten here.
    :data:`~osprey.cli.templates.manifest.REPO_REPRODUCIBLE_COMMAND` is written
    directly, so the manifest arrives here already correct — generating a wrong
    answer and patching it afterwards would leave every manifest written by any
    other path carrying the wrong one.

    The drift fingerprint's per-key digests are added to the DEPLOYMENT's
    manifest only. The fingerprint itself — ``creation.preset_hash``, a
    :func:`~osprey.cli.build_profile.compute_profile_hash` of the resolved
    profile, the file material it names, and the host overlay ``.env.variant``
    selects — is stamped by the manifest generator on every render, and on
    ``build/.osprey-manifest.json`` it is the *only* thing the drift verdict
    reads. The variant is read off the profile's own directory by the hash, not
    passed in from here: this build already resolved a selection, and a second
    resolution at the stamp could disagree with the one ``osprey up`` makes,
    which is precisely how a build goes stale without anyone noticing. What is added here is commentary on
    it: the per-top-level-key digests that let
    :func:`osprey.deployment.staleness.check_drift` name which part of
    ``profile.yml`` moved when it refuses to start a stale build. Deliberately
    not a second hash mechanism — nothing decides drift by reading these — which
    is exactly why a persona render does not get them: the drift gate reads one
    manifest per repo, the deployment's, and the persona deltas are already
    folded into ITS fingerprint
    (:func:`~osprey.cli.build_profile_merge._fold_profile_material`). A second
    set of digests under each persona would be a fingerprint nothing consults.

    Args:
        render_dir: The rendered project whose manifest is rewritten. For the
            deployment this is the staged tree that becomes ``build/``.
        repo_root: The deployment repo, consulted only for the manifest's
            filename — read off the one function that knows it rather than
            spelled a second time here.
        profile_path: The profile this render came from.
        key_digests: Whether to add the drift check's per-key commentary.
    """
    from osprey.deployment import staleness

    if not key_digests:
        return

    fingerprint = staleness.profile_fingerprint(profile_path)
    if fingerprint is None or not fingerprint.key_digests:
        return

    manifest_path = render_dir / staleness.build_manifest_path(repo_root).name
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("Manifest fingerprint detail skipped: %s", exc)
        return

    creation = manifest.setdefault("creation", {})
    creation[staleness.KEY_DIGESTS_MANIFEST_KEY] = fingerprint.key_digests
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# The network axis, checked against what was actually rendered
# ---------------------------------------------------------------------------

#: The attachment that takes a service OFF the project network and onto the
#: host's own namespace. Spelled here because the schema exports the vocabulary
#: (:data:`~osprey.cli.build_profile_schema.VALID_NETWORK_MODES`) and its
#: DEFAULT, not the non-default member; the test suite pins this value against
#: that vocabulary so a change there cannot leave these checks reading a mode
#: nothing renders.
_HOST_NETWORK = "host"

#: The variables a co-deployed consumer of the dispatch pair is rendered with to
#: reach it. Both are written on an ASSUMPTION about the pair's network — the
#: compose DNS name when the consumer is on the project network, the host's
#: loopback and the pair's own ports when it is not — and neither template can
#: check that assumption, because the pair's mode is not the consumer's to read
#: at render time. :func:`_dispatch_parity_errors` is where it is checked, which
#: is why the contract is named here rather than inferred: a rewritten address
#: no longer carries the pair's name, so there is nothing left in the VALUE to
#: recognize it by.
_DISPATCH_PAIR_ADDRESS_VARS = ("DISPATCHER_URL", "WORKER_URL")


class _RenderedService(NamedTuple):
    """One container stanza as this build rendered it.

    The checks below reason about what came OUT of the templates rather than
    what went in, which is the whole point of running them after the render: a
    ``network:`` key that no template reads produces a config that says ``host``
    and a compose file that says otherwise, and only the rendered side knows.
    """

    compose_name: str
    """The compose service key — and so the DNS name it answers to on the
    project network. Under ``network_mode: host`` it answers to nothing."""

    config_key: str | None
    """The ``services.<key>`` block this stanza was rendered from, or ``None``
    for a stanza no declared service accounts for (the top-level file's own
    contents, or a facility template that renders more than its own service)."""

    on_host: bool
    """Whether the rendered stanza declares ``network_mode: host``."""

    environment: dict[str, str]
    """The rendered ``environment:`` block, both compose spellings normalized
    to a mapping of strings."""

    compose_file: str
    """Path of the rendered file this stanza came from, for the error text."""


def _mapping(value: Any) -> dict[str, Any]:
    """*value* when it is a mapping, an empty one otherwise.

    The config these checks read is loaded from YAML a person edits, so any
    block they describe may arrive as something else entirely. A check is not
    the place to report that — the consumer of the key will, in its own words —
    but it must not crash the build on the way past.
    """
    return value if isinstance(value, dict) else {}


def _service_network_mode(service_block: Any) -> str:
    """The network attachment one rendered ``services.<name>`` block declares.

    Read through the schema's own accessor rather than by reaching for the key,
    so the default lives in exactly one place and this module never spells the
    default mode.
    """
    from .build_profile_schema import ServiceDef

    return ServiceDef(template="", config=_mapping(service_block)).network_mode()


def _compose_service_environment(service: dict[str, Any]) -> dict[str, str]:
    """The rendered ``environment:`` of one compose service, as a mapping.

    Compose accepts both the mapping form and the ``- KEY=value`` list form;
    a facility template may use either, and a check that understood only one
    would pass a service it never actually read.
    """
    declared = service.get("environment")
    if isinstance(declared, dict):
        return {str(key): "" if value is None else str(value) for key, value in declared.items()}
    if isinstance(declared, list):
        pairs = {}
        for entry in declared:
            key, separator, value = str(entry).partition("=")
            pairs[key] = value if separator else ""
        return pairs
    return {}


def _rendered_compose_path(build_dir: str, service_path: str) -> str:
    """Where the render put one service's compose file, resolved.

    Mirrors :func:`~osprey.deployment.compose_generator.setup_build_dir`'s own
    derivation (the service's template directory, taken relative to the working
    directory, under the output base) rather than assuming the directory is
    named after the service — a facility declares its own ``path``, and the two
    need not agree.
    """
    source_dir = os.path.relpath(service_path, os.getcwd())
    return os.path.realpath(os.path.join(build_dir, source_dir, "docker-compose.yml"))


def _index_rendered_services(
    config: dict[str, Any], compose_files: Sequence[str]
) -> list[_RenderedService]:
    """Read every rendered compose file back into one list of stanzas.

    Parsed rather than pattern-matched: the checks ask which services are on
    the host namespace and what addresses they were handed, and both answers are
    structure, not text.

    A file that will not parse is skipped with a debug line. It was written by
    this same render seconds ago, so an unreadable one is a rendering failure —
    which the compose invocation reports in full, naming the file and the line.
    Translating it here would only put a network-axis heading on it.
    """
    import yaml

    owners = {}
    build_dir = str(config.get("build_dir", "./build"))
    for name, block in _mapping(config.get("services")).items():
        if isinstance(block, dict) and block.get("path"):
            owners[_rendered_compose_path(build_dir, str(block["path"]))] = str(name)

    indexed: list[_RenderedService] = []
    for path in compose_files:
        try:
            document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            logger.debug("Network checks skipped %s: %s", path, exc)
            continue
        if not isinstance(document, dict):
            continue
        owner = owners.get(os.path.realpath(path))
        for compose_name, service in _mapping(document.get("services")).items():
            if not isinstance(service, dict):
                continue
            indexed.append(
                _RenderedService(
                    compose_name=str(compose_name),
                    config_key=owner,
                    on_host=service.get("network_mode") == _HOST_NETWORK,
                    environment=_compose_service_environment(service),
                    compose_file=str(path),
                )
            )
    return indexed


def _service_label(service: _RenderedService) -> str:
    """How an error names one rendered service to the operator.

    Its ``services.<key>`` spelling when the build knows which declaration
    produced it — that is the line the remedy is applied to — and the compose
    service key otherwise.
    """
    if service.config_key:
        return f"services.{service.config_key}"
    return f"the compose service '{service.compose_name}'"


def _network_remedy_key(config_key: str | None, compose_name: str) -> str:
    """The config key that moves one service onto the host network.

    The dispatcher and its workers share ONE knob: a directly-authored
    ``services.<half>.network`` is rejected by profile validation, so naming it
    as the remedy would send the operator into a second error.
    """
    from .build_profile_model import DISPATCH_PAIR_SERVICES

    if config_key in DISPATCH_PAIR_SERVICES:
        return "dispatch.network"
    return f"services.{config_key or compose_name}.network"


def _names_service(value: str, dns_name: str) -> bool:
    """Whether one rendered env value addresses *dns_name*.

    Matched as an address rather than as a substring: the name must start the
    value or follow something that is not part of a name (never a path
    separator, except the ``//`` that opens a URL authority), and must be
    followed by a port, a path, or the end of the value. So
    ``http://event-dispatcher:8020``, ``event-dispatcher:8020`` and a bare
    ``event-dispatcher`` all count, while ``/app/event-dispatcher`` and
    ``event-dispatcher-external`` do not.
    """
    import re

    pattern = rf"(?:(?<=//)|(?<![A-Za-z0-9_.\-/])){re.escape(dns_name)}(?=[:/]|$)"
    return re.search(pattern, value) is not None


def _inert_axis_errors(config: dict[str, Any], indexed: Sequence[_RenderedService]) -> list[str]:
    """Services whose declared ``network: host`` no template acted on.

    The axis is only real where a template renders it. A service template that
    never adopted the shared macro silently keeps its ``networks:`` block, and
    the deployment comes up on the compose bridge while the profile — and every
    address the rest of the build derived from it — says host.
    """
    errors = []
    services = _mapping(config.get("services"))
    for name in [str(entry) for entry in (config.get("deployed_services") or [])]:
        block = services.get(name)
        if _service_network_mode(block) != _HOST_NETWORK:
            continue
        stranded = [
            service for service in indexed if service.config_key == name and not service.on_host
        ]
        if not stranded:
            continue
        template = os.path.join(
            str(_mapping(block).get("path") or f"./services/{name}"), "docker-compose.yml.j2"
        )
        errors.append(
            f"services.{name}.network is '{_HOST_NETWORK}', but the compose file this "
            f"build rendered for it declares no `network_mode: {_HOST_NETWORK}` "
            f"({', '.join(sorted({service.compose_file for service in stranded}))}). "
            f"The service's template does not render the network axis. In {template}, "
            'import the shared macro — {% import "services/_network_axis.j2" as net %} '
            "— and replace the service's `networks:` block with "
            f"{{{{- net.network(services.{name}) }}}}."
        )
    return errors


def _dispatch_parity_errors(
    indexed: Sequence[_RenderedService],
) -> tuple[list[str], set[str]]:
    """Co-deployed dispatch consumers that do not share the pair's network.

    The pair moves by ONE knob, and everything that talks to it has to move with
    it. Both of the addresses a consumer is rendered with
    (:data:`_DISPATCH_PAIR_ADDRESS_VARS`) are written for one side of the
    boundary or the other, and the template writing them cannot know which side
    the pair ended up on — so a consumer that disagrees with the pair is
    rendered with addresses that are wrong in a way no other check can see:

    * consumer on the bridge, pair on the host — the addresses are the pair's
      compose DNS names, and a host-network service has none;
    * consumer on the host, pair on the bridge — the addresses were rewritten
      onto the host's loopback and the pair's own ports, of which the pair
      publishes at most the dispatcher's.

    The second direction is invisible to :func:`_cross_network_errors`,
    precisely because the rewrite already happened: a ``localhost`` address
    carries nothing to recognize the pair by. This check reads the variable
    NAMES instead, which are osprey's own consumer contract.

    Returns:
        The errors, and the compose names of every consumer this check examined
        — the pairing :func:`_cross_network_errors` must then leave alone, so
        one topology mistake is reported once, in the words that name the knob.
    """
    from .build_profile_model import DISPATCH_PAIR_SERVICES

    pair = [service for service in indexed if service.config_key in DISPATCH_PAIR_SERVICES]
    if not pair:
        return [], set()

    pair_on_host = any(service.on_host for service in pair)
    pair_named = ", ".join(sorted({_service_label(service) for service in pair}))
    errors: list[str] = []
    examined: set[str] = set()
    for service in indexed:
        if service.config_key in DISPATCH_PAIR_SERVICES:
            continue
        addresses = {
            key: value
            for key, value in service.environment.items()
            if key in _DISPATCH_PAIR_ADDRESS_VARS and "${" not in value
        }
        if not addresses:
            continue
        examined.add(service.compose_name)
        if service.on_host == pair_on_host:
            continue
        rendered = ", ".join(f"{key} -> {value}" for key, value in sorted(addresses.items()))
        remedy = _network_remedy_key(service.config_key, service.compose_name)
        if pair_on_host:
            errors.append(
                f"{_service_label(service)} is a co-deployed consumer of the dispatch pair "
                f"({rendered}), but it stays on the compose bridge while {pair_named} run on "
                "the host network, where they have no compose DNS name — so those addresses "
                f"resolve to nothing. Put the consumer on the same network: set `{remedy}: "
                f"{_HOST_NETWORK}`."
            )
        else:
            errors.append(
                f"{_service_label(service)} runs on the host network while the co-deployed "
                f"{pair_named} stay on the compose bridge. The pair addresses it is rendered "
                f"with ({rendered}) are written for a host-side pair — from the host "
                "namespace they reach only what the pair publishes there, which is not the "
                "worker. Put the two on one network: set `dispatch.network: "
                f"{_HOST_NETWORK}` to move the pair onto the host as well, or drop "
                f"`{remedy}` so the consumer rejoins the compose network."
            )
    return errors, examined


def _cross_network_errors(
    indexed: Sequence[_RenderedService], dispatch_consumers: set[str]
) -> list[str]:
    """Addresses that cross the boundary between the host namespace and the bridge.

    A compose DNS name exists only for services on the project network, so any
    rendered address that names one from the other side of that boundary
    resolves to nothing — in both directions, and silently, at the first request
    rather than at boot.

    Osprey rewrites the addresses it emits itself onto ``localhost`` when it
    moves a service to the host namespace, which is exactly why matching on the
    DNS NAME is enough to leave those alone: a rewritten address no longer
    carries one. What is left is the two cases the build cannot fix for the
    operator — a consumer left behind on the bridge (whose target has no address
    at all to rewrite to) and a hand-authored address (which osprey does not
    rewrite by design) — so both fail the build with the remedy named.

    Args:
        indexed: Every stanza this build rendered.
        dispatch_consumers: Compose names :func:`_dispatch_parity_errors` has
            already answered for against the dispatch pair. Their addresses ARE
            the pair's compose DNS names when they disagree with it, so without
            this the same mistake would be reported twice — once naming the knob
            that fixes it, once naming the name that broke.
    """
    from .build_profile_model import DISPATCH_PAIR_SERVICES

    errors = []
    for service in indexed:
        for other in indexed:
            if other.compose_name == service.compose_name or other.on_host == service.on_host:
                continue
            if (
                service.compose_name in dispatch_consumers
                and other.config_key in DISPATCH_PAIR_SERVICES
            ):
                continue
            hits = {
                key: value
                for key, value in service.environment.items()
                if _names_service(value, other.compose_name)
            }
            if not hits:
                continue
            rendered = ", ".join(f"{key} -> {value}" for key, value in sorted(hits.items()))
            if service.on_host:
                remedy = _network_remedy_key(other.config_key, other.compose_name)
                errors.append(
                    f"{_service_label(service)} runs on the host network, but its rendered "
                    f"environment reaches {_service_label(other)} by its compose DNS name "
                    f"'{other.compose_name}' ({rendered}). A host-network container is not "
                    "on the compose network, so that name resolves to nothing there. Osprey "
                    "does not rewrite hand-authored addresses: either put both services on "
                    f"one network by setting `{remedy}: {_HOST_NETWORK}`, or point the "
                    "variable at an address that is reachable from the host namespace "
                    f"(the port {_service_label(other)} publishes on localhost)."
                )
            else:
                remedy = _network_remedy_key(service.config_key, service.compose_name)
                errors.append(
                    f"{_service_label(other)} runs on the host network, where it has no "
                    f"compose DNS name, but the co-deployed {_service_label(service)} stays "
                    f"on the compose bridge and is still rendered with its name "
                    f"'{other.compose_name}' ({rendered}) — an address that resolves to "
                    f"nothing. Move it onto the same network: set `{remedy}: "
                    f"{_HOST_NETWORK}`."
                )
    return errors


def _network_check_errors(config: dict[str, Any], compose_files: Sequence[str]) -> list[str]:
    """Everything wrong with the network axis of the render that just happened.

    Accumulated rather than raised one at a time, the way profile validation
    reports, so an operator who moved a stack onto the host network sees every
    consequence in one build instead of one per build.
    """
    indexed = _index_rendered_services(config, compose_files)
    parity_errors, dispatch_consumers = _dispatch_parity_errors(indexed)
    return [
        *_inert_axis_errors(config, indexed),
        *parity_errors,
        *_cross_network_errors(indexed, dispatch_consumers),
    ]


def _render_compose_files(
    zones: _RenderZones, runtime_root: str | None = None, dev_mode: bool = False
) -> dict[str, Any] | None:
    """Render the deployment's compose files into the staged tree.

    Compose files are rendered in the same pass as everything else, because they
    are derived from the same ``config.yml``: emitting them a verb later would
    leave ``build/`` an incomplete description of the deployment.

    The generator resolves its inputs relative to the working directory, which
    is what makes it stageable at all: run from the staged render, every service
    template it reads is the one this build just rendered.

    Its OUTPUT base is passed explicitly as the staging root, because the staged
    tree IS the future ``build/``. The staged ``config.yml`` says
    ``build_dir: ./build`` — correct for the deploy, which reads it from the
    repo root — and taking that value here would append a second ``build/`` to a
    directory that is already the output zone, landing every compose file at
    ``build/build/services/…``. Nothing would then resolve: the rendered files
    spell their own mounts against the repo root (``./build/services/<svc>/…``),
    which is where compose looks for them under the pinned
    ``--project-directory``. Rendered output and the service templates it is
    rendered FROM therefore share ``build/services/<svc>/`` and differ only by
    filename (``docker-compose.yml`` beside ``docker-compose.yml.j2``).

    Skipped under ``--runtime-root``. Compose generation is a *host-side* act:
    it resolves bind-mount sources against the config's ``project_root`` and
    pre-creates them, so that a container runtime does not create them itself
    as root-owned. ``--runtime-root`` deliberately makes that value a path on
    another machine, where pre-creating anything is at best phantom directories
    on this host and at worst — ``/app`` — a hard failure. The compose files for
    such a deployment are rendered by the build that runs on the host it
    deploys to.

    Args:
        zones: The render's paths.
        runtime_root: ``--runtime-root``, or ``None``.
        dev_mode: ``--dev`` — stage a wheel from the local checkout into each
            service build context and emit the ``OSPREY_DEV`` build arg, so the
            images run this checkout instead of the pinned release.

    Returns:
        The loaded config, for callers that need the deployment's identity;
        ``None`` when compose generation did not run.
    """
    if runtime_root:
        _report_fact(
            f"Compose files not rendered: --runtime-root points this build at {runtime_root}, "
            "and compose bind sources are resolved on the host that deploys it."
        )
        return None

    from osprey.deployment.compose_generator import prepare_compose_files

    previous = Path.cwd()
    os.chdir(zones.stage)
    try:
        # ``deployed_config_dir`` is passed explicitly for the same reason
        # ``output_root`` is, and is its mirror image: this render reads its
        # config from the staging ROOT, but the config it produces will be read
        # from ``build/`` once the swap lands. Derived from the config path it
        # would come out empty, and every path spelled against it would resolve
        # one directory too high at deploy time.
        config, compose_files = prepare_compose_files(
            str(zones.stage / "config.yml"),
            dev_mode=dev_mode,
            output_root=".",
            deployed_config_dir=BUILD_DIR_NAME,
        )
        # Read back inside the chdir: every path involved — the rendered files,
        # the service directories the config names — is spelled relative to the
        # staged tree, which is what the render itself was rooted at.
        network_errors = _network_check_errors(config, compose_files)
    finally:
        os.chdir(previous)
    if network_errors:
        raise click.UsageError("Network axis check failed:\n  - " + "\n  - ".join(network_errors))
    # Not reported here: the render phase's `compose files` step fires the
    # moment this returns and says the same thing.
    logger.debug("Rendered %d compose file(s)", len(compose_files))
    return dict(config)


def _warn_if_deployment_running(config: dict[str, Any] | None, project_name: str) -> None:
    """Say that a fresh render does not reach containers that are already up.

    A build renders files; it never touches a running container. An operator
    who rebuilds while the stack is up would otherwise have every reason to
    believe the change is live, and find out at the worst possible moment that
    it is not.

    Best-effort: no container runtime, or a runtime that cannot be asked, means
    no warning rather than a failed build.
    """
    import subprocess

    try:
        from osprey.deployment.compose_generator import resolve_project_name
        from osprey.deployment.runtime_helper import get_runtime_command

        project = resolve_project_name(config or {"project_name": project_name})
        runtime = get_runtime_command(config)[0]
        result = subprocess.run(
            [
                runtime,
                "ps",
                "--filter",
                f"label=com.docker.compose.project={project}",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        running = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception as exc:  # a warning must never be the reason a build fails
        logger.debug("Running-container check skipped: %s", exc)
        return

    if running:
        logger.warning(
            "  %d container(s) of this deployment are running (%s). The new build "
            "takes effect at the next `osprey up` or `osprey restart`.",
            len(running),
            ", ".join(sorted(running)),
        )


class _SharedRenderInputs(NamedTuple):
    """What every project one ``osprey build`` renders has in common.

    A repo with personas renders several projects — the deployment's own and one
    per delta — and the whole point of a persona is that it differs from its host
    in the profile and in NOTHING ELSE. Everything that is a property of the
    build rather than of the profile therefore lives here and is passed to each
    render unchanged: the same repo root, the same interpreter, the same
    dependency list, the same ``--runtime-root``.
    """

    repo_root: Path
    """The deployment repo. Every render anchors its ``project_root`` here."""

    build_dir: Path
    """``<repo>/build`` — where the one project venv is, for every render."""

    runtime_root: str | None
    """``--runtime-root``, or ``None``."""

    project_deps: list[str]
    """The dependency list the venv install resolved, for the rendered Dockerfile."""

    skip_deps: bool
    """``--skip-deps``: no venv was created, so the interpreter is this one."""

    manager: TemplateManager
    """One template manager, so the template root is resolved once."""

    va_manifests: dict[tuple[str, int], Any]
    """Prepared virtual-accelerator manifests, memoized by ``(data root, tier)``.

    Preparing one parses the channel databases under the data tree, which is
    seconds of work on a real facility. Personas overwhelmingly share their
    host's data tree and tier, so the second and third renders would otherwise
    re-derive a manifest byte-for-byte identical to the first. Keyed on the two
    inputs that decide it, so a delta that *does* move either still gets its own.
    """

    profile_overlays: tuple[Path, ...] = ()
    """Profile layers merged over EVERY profile this build resolves.

    The host-variant overlay (:mod:`~osprey.cli.variant_selection`), or empty
    when this host builds the tracked profile. It belongs here for the same
    reason ``--runtime-root`` does: it is a property of the build, not of a
    profile, so the deployment's render and each persona's must see the same
    one. A variant that reached only the deployment would leave every persona
    project — and every persona image — rendered for a different host than the
    stack they ship in.
    """

    runtime_interpreter: str | None = None
    """The interpreter this render's artifacts launch with, when it is KNOWN
    rather than derivable.

    ``None`` for every render that runs on this machine: there the interpreter
    is derived from the filesystem — the project venv, else the generating
    process — which is the only answer that survives the project being moved or
    rebuilt. A render destined for a container image runs on a filesystem that
    is not here, so the derivation cannot see it and instead bakes THIS
    machine's ``.venv`` into every MCP server command and every framework hook.
    See :data:`_CONTAINER_INTERPRETER`.
    """

    host_config: Mapping[str, Any] | None = None
    """The deployment's own rendered ``config.yml``, once it has been rendered.

    What every attached render in this repo is told about the services the
    deployment runs (:func:`osprey.deployment.reach.project_attached_overrides`):
    the ports it publishes, the panel URLs its injectors derived. ``None``
    until the deployment's render is done — it is the first render of every
    build — and ``None`` throughout a build whose profile is itself attached,
    which has no host in this repo and is told the app template's defaults
    instead (:func:`_template_host_config`).
    """


def _rendered_config(render_dir: Path) -> dict[str, Any]:
    """The ``config.yml`` a render wrote, as a plain mapping."""
    import yaml

    with (render_dir / "config.yml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _template_host_config(
    shared: _SharedRenderInputs,
    build_profile: Any,
    *,
    project_name: str,
    render_dir: Path,
    profile_dir: Path,
    context: Mapping[str, Any],
    artifacts: dict[str, list[str]] | None,
) -> dict[str, Any]:
    """What the app template deploys at its defaults — the host an attached
    profile built with no deployment in its repo is told about.

    The template rendered AS a deployment (``deploy_services: true``) with
    this render's own context, the profile's ``config:`` laid over it (the
    same overlay the render itself gets, because a host that differs from
    the template's defaults is named there), and then the service injectors
    run over it exactly as for a deploying profile: an attached profile
    inherits the blocks they read (``dispatch:``, ``bluesky_web:``, …) from
    the deployment profile it extends, and what they derive — the EVENTS and
    BLUESKY tab entries above all — is what a persona beside a real host is
    told from that host's render. Rendered into a scratch directory the build
    zone never sees — it is a reading, not a render.
    """
    import tempfile
    from dataclasses import replace

    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)
        shared.manager.render_config(
            project_name,
            render_dir,
            scratch_dir / "config.yml",
            data_bundle=build_profile.data_bundle,
            context={**context, "deploy_services": True},
            artifacts=artifacts,
        )
        if build_profile.config:
            _apply_config_overrides(scratch_dir, build_profile.config)
        # A view of the profile as the deployment it extends: the injectors
        # skip an attached profile wholesale, and this reading is of the host.
        _inject_services(replace(build_profile, deploy_services=True), profile_dir, scratch_dir)
        return _rendered_config(scratch_dir)


def _render_project(
    shared: _SharedRenderInputs,
    resolved: Any,
    *,
    profile_path: Path,
    project_name: str,
    output_dir: Path,
    deployment: bool,
    progress: Any,
    extra_known: Sequence[str] = (),
    injected_out: list[str] | None = None,
) -> Path:
    """Render one resolved profile into ``<output_dir>/<project_name>``.

    The build's whole render pass, and the one place it is written: the base
    template, the profile's config overrides, its services, its convention
    artifacts, the virtual-accelerator manifest, the MCP servers, the build
    manifest, and the Claude Code artifacts regenerated over the lot. A
    deployment's own project and a persona's go through it identically — same
    steps, same order, same inputs bar the profile — because a persona that
    differed from its host in any of them would be a different deployment
    wearing its name.

    What is NOT here is what a persona does not get, and each is deliberate:

    * **the venv** — one per repo, at ``build/.venv``. A persona project is a
      container image build context, and the image installs its own
      dependencies from ``OSPREY_PIP_SPEC``;
    * **the compose files** — the deployment's compose describes the whole
      stack, personas included. A persona project is never deployed on its own;
    * **the lifecycle phases** — ``pre_build``/``post_build``/``validate`` are
      the profile's own shell commands, and a delta inherits them from the root.
      Running them once per persona would run the same commands three times over
      for one build.

    Args:
        shared: What this render has in common with every other in this build.
        resolved: The ``LoadedProfile`` for this project — for a persona, the
            delta already merged over the root profile, with the root as its
            profile directory.
        profile_path: The profile FILE this render came from, recorded in the
            manifest. The delta itself for a persona, so a rendered project can
            always name the source it is derived from.
        project_name: The rendered project's name and directory leaf.
        output_dir: The directory the render lands under.
        deployment: Whether this is the deployment's own project rather than a
            persona's. Gates the two things only it gets: the drift check's
            per-key manifest digests, and the provider-credential report (an
            account of the repo's one ``.env``, identical for every render).
        progress: Where the per-step progress lines go — ``logger.info`` for
            the deployment, ``logger.debug`` for a persona, whose steps are
            summarized in one line by :func:`_render_persona_projects`.
        extra_known: Repo-root entry names to exempt from the unknown-entry
            warning on top of the profile's own, so a persona render does not
            repeat a warning the deployment's render already made.
        injected_out: A list to record this render's injected service names in,
            for a caller that reports them. Passed rather than returned because
            a build renders six times and every pass injects the same set: the
            caller names the ONE pass whose set it reports (``deployment`` does
            not identify it — the deployment's container copy renders with it
            set too).

    Returns:
        The rendered project directory.
    """
    from osprey.build.build_tiers import tier_mode_conflict
    from osprey.build.claude_code_resolver import load_provider_spec
    from osprey.deployment.reach import reach_errors
    from osprey.services.virtual_accelerator.manifest.build import (
        prepare_project_manifest,
        write_project_manifest,
    )

    from .build_profile_archiver import va_archiver_config_overrides
    from .build_profile_deploy import deploy_config_overrides
    from .build_profile_reach import (
        attached_render_overrides,
        orphan_panel_fragments,
        reach_override_errors,
        selected_panel_errors,
    )
    from .validate_claude_artifacts import validate_agent_tools_against_permissions

    build_profile = resolved.profile
    repo_root = shared.repo_root
    manager = shared.manager
    render_dir = output_dir / project_name

    artifacts = _collect_profile_artifacts(build_profile, progress=progress)
    if build_profile.web_panels:
        artifacts["web_panels"] = list(build_profile.web_panels)

    context = _repo_render_context(
        build_profile,
        repo_root=repo_root,
        build_dir=shared.build_dir,
        runtime_root=shared.runtime_root,
        project_deps=shared.project_deps,
        skip_deps=shared.skip_deps,
        runtime_interpreter=shared.runtime_interpreter,
    )

    # Prepared before the render (which prunes the tiers/ subtree the paradigm
    # databases live in) and written after it, so the decision is settled before
    # anything is written.
    va_data_root = build_profile.resolved_data_root(repo_root) or (
        manager.template_root / "apps" / build_profile.data_bundle / "data"
    )
    va_key = (str(va_data_root), build_profile.resolved_tier())
    if va_key not in shared.va_manifests:
        shared.va_manifests[va_key] = prepare_project_manifest(va_data_root, va_key[1])
    prepared_va_manifest = shared.va_manifests[va_key]

    # ``create_project``'s ``tier`` argument means "the tier the profile PINNED",
    # not "the tier to use": given ``None`` it applies the same paradigm-aware
    # derivation ``resolved_tier()`` does, and given a value it enforces
    # ``tier_mode_conflict`` against it. So a paradigm that refuses an explicit
    # tier — one whose store is a service rather than tiered database files —
    # must be handed ``None`` here, or its own derived default comes back as a
    # pin and the build refuses to render at all. Ask the rule rather than
    # naming the paradigm, so this stays true as paradigms are added.
    derived_tier = build_profile.resolved_tier()
    pinned_tier = (
        None
        if tier_mode_conflict(derived_tier, build_profile.channel_finder_mode)
        else derived_tier
    )

    # The render. No `.env` is carried in from anywhere: the repo's own `.env`
    # is the deployment's whole secret store, it is mounted from the repo root,
    # and a build neither reads nor rewrites it — copying it into the disposable
    # zone would put the facility's keys somewhere a `rm -rf build/` is
    # documented to make safe. A persona render is inside that same zone and is
    # a container build context on top of it, so it gets no `.env` either; its
    # container is handed one at run time through compose.
    manager.create_project(
        project_name=project_name,
        output_dir=output_dir,
        data_bundle=build_profile.data_bundle,
        context=context,
        force=True,
        artifacts=artifacts or None,
        tier=pinned_tier,
        data_root=build_profile.resolved_data_root(repo_root),
    )
    progress("  ✓ Base template rendered")

    # What the deploy and va_archiver blocks contribute to the rendered config,
    # applied with the profile's own `config:` entries in one pass. Derived
    # keys the profile also spells are rejected at validation, so winning here
    # can never silently overwrite a facility's own value.
    derived_by_block = {
        "deploy": deploy_config_overrides(build_profile.deploy, build_profile.config),
        "va_archiver": va_archiver_config_overrides(build_profile.va_archiver),
    }
    derived = {key: value for block in derived_by_block.values() for key, value in block.items()}
    config_overrides = {**build_profile.config, **derived}
    if config_overrides:
        _apply_config_overrides(render_dir, config_overrides)
        progress("  ✓ Applied %d config override(s)", len(config_overrides))
        for block, entries in derived_by_block.items():
            for key, value in entries.items():
                progress("      %s: %s (from the profile's %s block)", key, value, block)

    # What an attached render is told about the services its host deploys —
    # the ports it publishes, the panel URLs its injectors derived — copied
    # from the deployment's own render (the Reach Contract,
    # osprey.deployment.reach). After the profile's overlay, because the
    # projection is gated on what THIS render switches on; before the
    # service injection and the Claude Code re-render, because a projected
    # block is what makes a server render at all (`graphdb_configured`).
    # A deploying project projects nothing: its injectors write the blocks.
    if not build_profile.deploy_services:
        if shared.host_config is not None:
            host_config, told_by = shared.host_config, "the hosting deployment's render"
        else:
            # Built alone, with no deployment in this repo: the profile extends
            # a deployment of the same app template, so that template rendered
            # AS a deployment is what the host says at the shipped defaults —
            # and the profile's own `config:` is where a host that differs is
            # named, so it is laid over those defaults first.
            host_config = _template_host_config(
                shared,
                build_profile,
                project_name=project_name,
                render_dir=render_dir,
                profile_dir=repo_root,
                context=context,
                artifacts=artifacts or None,
            )
            told_by = "the app template's defaults"
        projected = attached_render_overrides(
            host_config,
            _rendered_config(render_dir),
            selected_panels=build_profile.web_panels or (),
        )
        # A `config:` spelling that contradicts the projection is refused;
        # one that agrees (inherited from the hosting profile, or laid over
        # the template's defaults when built alone) is the same fact.
        duplicate_errors = reach_override_errors(build_profile.config, projected)
        if duplicate_errors:
            raise BuildProfileError(
                "Profile validation failed:\n  " + "\n  ".join(duplicate_errors)
            )
        if projected:
            _apply_config_overrides(render_dir, projected)
            progress(
                "  ✓ Told this attached render %d fact(s) about the host's services", len(projected)
            )
            for key, value in projected.items():
                progress("      %s: %s (from %s)", key, value, told_by)
        # The reverse: a tab this profile does NOT select, inherited from the
        # hosting profile's `config:` as a url-less fragment (a pinned route),
        # would render as an empty-url panel. Dropped — the selection is what
        # puts a projected tab in an attached render.
        from osprey.utils.config_writer import config_delete_field

        for key in orphan_panel_fragments(
            build_profile.web_panels or (), _rendered_config(render_dir)
        ):
            config_delete_field(render_dir / "config.yml", key)
            progress(
                "  ✓ Dropped %s: a tab this profile does not select, inherited with no url", key
            )
        # A tab this profile selects (`web_panels:`) that the projection gave
        # no address — the host it was told about runs no such sidecar — would
        # otherwise vanish from the render without a word.
        tabless = selected_panel_errors(
            build_profile.web_panels or (), _rendered_config(render_dir), told_by=told_by
        )
        if tabless:
            raise BuildProfileError("Profile validation failed:\n  " + "\n  ".join(tabless))

    injected = _inject_services(build_profile, repo_root, render_dir)
    if injected_out is not None:
        injected_out.extend(injected)

    # The ground truth every client loads is the rendered config, so this is
    # where a consumer switched on with nothing to dial is refused — for a
    # deployment that dropped a block its modules still use, and for an
    # attached render that was told nothing (a host, or an app template, that
    # deploys no such service) alike. AFTER the injectors: a deploying profile
    # that pins only `web.panels.events.path` has its `url` written by the
    # dispatch injector, and refusing before that would name a key the build
    # was about to write.
    unreachable = reach_errors(_rendered_config(render_dir))
    if unreachable:
        raise BuildProfileError("Profile validation failed:\n  " + "\n  ".join(unreachable))

    applied = _apply_conventions(
        repo_root,
        render_dir,
        _resolve_context_roster(render_dir),
        extra_known=[
            *_profile_known_root_entries(build_profile, profile_path),
            *extra_known,
        ],
        excluded=resolved.excluded_artifacts,
    )
    if applied.copied:
        progress(
            "  ✓ Applied %d profile artifact(s): %s",
            applied.copied,
            ", ".join(f"{count} {key}" for key, count in sorted(applied.by_category.items())),
        )
        reg_count = _register_convention_artifacts(render_dir, applied)
        if reg_count:
            progress("  ✓ Registered %d profile artifact(s) in config.yml", reg_count)

    if prepared_va_manifest is not None:
        write_project_manifest(prepared_va_manifest, render_dir / "data")
        progress(
            "  ✓ Generated virtual-accelerator channel manifest (%d channels)",
            prepared_va_manifest.manifest["_metadata"]["total_channels"],
        )

    if build_profile.mcp_servers:
        _persist_mcp_servers(render_dir, build_profile.mcp_servers)
        progress("  ✓ Persisted %d MCP server(s) to config.yml", len(build_profile.mcp_servers))
    if build_profile.artifact_server:
        _persist_artifact_server(render_dir, build_profile.artifact_server)
        progress("  ✓ Merged artifact_server overrides into config.yml")

    if deployment:
        report_provider_credentials(render_dir, build_profile.provider, profile_dir=repo_root)

    manifest_context: dict[str, Any] = {
        "default_provider": build_profile.provider,
        "default_model": build_profile.model,
        # Absolute, and on every build: for the deployment this is the profile
        # the drift check re-hashes to decide whether `build/` still describes
        # the source; for a persona it is the delta the render came from.
        "profile_path_abs": str(profile_path),
    }
    if build_profile.channel_finder_mode is not None:
        manifest_context["channel_finder_mode"] = build_profile.channel_finder_mode
    if build_profile.claude_md_template:
        manifest_context["claude_md_template"] = build_profile.claude_md_template
    manager.generate_manifest(
        project_dir=render_dir,
        project_name=project_name,
        data_bundle=build_profile.data_bundle,
        context=manifest_context,
        artifacts=artifacts or None,
        profile_path=str(profile_path),
    )
    _stamp_repo_manifest(render_dir, repo_root, profile_path, key_digests=deployment)

    manager.regenerate_claude_code(
        render_dir,
        project_root_override=shared.runtime_root or str(repo_root),
        # The venv is at its final path in the output zone and only joins the
        # staged tree during the swap, so the tree being written has none to
        # find. Not named for a --runtime-root render: the venv that exists at
        # run time is then the other machine's, not this one's.
        runtime_venv_dir=None if shared.runtime_root else shared.build_dir,
        # ...and when the caller KNOWS that other machine's interpreter, the
        # derivation is not consulted at all. Without this, every server command
        # and every hook command in a container render names this host's venv.
        runtime_interpreter=shared.runtime_interpreter,
    )
    progress("  ✓ Re-rendered Claude Code artifacts")

    validation_errors = validate_agent_tools_against_permissions(render_dir)
    if validation_errors:
        raise BuildProfileError(
            "Agent tool/permission drift detected:\n  " + "\n  ".join(validation_errors)
        )
    try:
        # Reachability: with defer_unresolved_telemetry_creds=True, an
        # unresolved "${VAR}" in an openobserve credential (user/password) no
        # longer raises here — _openobserve_auth_header
        # (claude_code_telemetry.py) warns and omits the auth header instead.
        # Only the missing/blank-credential arm of that same check still
        # raises ObservabilityCredentialError into this ValueError catch.
        # Pinned by test_deferred_var_credential_warns_not_raises_through_resolve
        # in tests/cli/test_telemetry_env.py.
        load_provider_spec(render_dir, defer_unresolved_telemetry_creds=True)
    except ValueError as e:
        raise BuildProfileError(str(e)) from e

    return render_dir


def _persona_deltas(repo_root: Path) -> list[Path]:
    """Every persona delta a build of *repo_root* renders a project for.

    The direct children of ``personas/`` that are files and are not dot-prefixed,
    sorted by name — deliberately the same enumeration as the drift
    fingerprint's
    (:func:`~osprey.cli.build_profile_merge._fold_profile_material`), and for the
    same reason it uses: root discovery reads a file as a delta only when its
    parent directory IS ``personas/``
    (:func:`~osprey.cli.profile_root.resolve_profile_root`), so a nested tree is
    not build input.

    The two enumerations have to agree exactly. A file the fingerprint covers but
    the render skips would make an edit to it refuse the next start with nothing
    to re-render; a file the render picks up but the fingerprint misses would let
    an edit to it ship silently. That is also why there is no suffix filter here:
    ``personas/`` holds deltas and nothing else, so a file in there that will not
    parse as a profile is a mistake the build should name rather than walk past.

    Returns an empty list when there is no ``personas/`` directory, which is what
    keeps a repo without personas rendering exactly what it rendered before.

    Raises:
        BuildProfileError: When a delta does not anchor back at *repo_root* —
            see the containment check below.
    """
    from .profile_root import PERSONA_DIRNAME, resolve_profile_root

    persona_dir = repo_root / PERSONA_DIRNAME
    if not persona_dir.is_dir():
        return []
    deltas = sorted(
        (
            entry
            for entry in persona_dir.iterdir()
            if entry.is_file() and not entry.name.startswith(".")
        ),
        key=lambda entry: entry.name,
    )

    # The property the rest of this rests on: a file found under `personas/`
    # must be read AS a delta over THIS repo's profile. The enumeration above is
    # lexical and cannot see a symlink; root discovery resolves one. So a delta
    # symlinked in from elsewhere, or a symlinked `personas/` directory, would
    # otherwise be read as a standalone profile — rendering a hollow project
    # with none of this repo's data tree, conventions or secrets — or as a delta
    # over a different facility's profile, and either way reporting nothing. The
    # drift fingerprint folds the same files without resolving them, so a build
    # that walked past this would also be stamping a hash over material it never
    # rendered from.
    for delta in deltas:
        anchored_root, is_delta = resolve_profile_root(delta)
        if not is_delta or anchored_root != repo_root:
            landed = (
                f"a delta over the profile at {anchored_root}"
                if is_delta
                else f"a standalone profile at {anchored_root}"
            )
            raise BuildProfileError(
                f"{delta} resolves to {landed}, not a delta over this repo's own "
                f"{PROFILE_FILENAME} at {repo_root}. A symlinked delta, or a symlinked "
                f"{PERSONA_DIRNAME}/ directory, does this: the persona would be built "
                "without this deployment's data tree, conventions and secrets, or over "
                f"somebody else's. Keep every delta a real file inside {persona_dir}."
            )
    return deltas


def _render_persona_projects(shared: _SharedRenderInputs, zones: _RenderZones) -> list[Path]:
    """Render ``build/<repo>-<persona>/`` for every delta in ``personas/``.

    Personas are rendered HERE, by the build, and nowhere else. No start verb
    renders one: ``build/`` is the complete account of what a deploy will run,
    and a persona project appearing at ``osprey up`` would have made that false
    exactly when it mattered — an operator whose delta changed would have had a
    fresh persona beside a stale deployment, from a start that re-rendered half
    the stack.

    Into the STAGED tree, so persona projects transit the atomic swap with
    everything else (:func:`_swap_in_render`): a persona whose delta will not
    resolve fails the whole build and leaves the previous ``build/`` — every
    project in it — exactly as it was. Never a half-written set.

    The naming is the catalog's, not this function's invention:
    ``<repo>-<persona>`` is what ``osprey init`` writes into every catalog
    entry's ``project`` and ``project_path``
    (:func:`~osprey.cli.profile_cmd._persona_catalog_layer`), which is how the
    deploy finds the render this produced. Every delta is rendered whether the
    catalog names it or not — the catalog decides which personas are *deployed*,
    ``personas/`` decides which exist — so a delta added before its catalog entry
    is already built when the entry lands.

    Returns:
        The persona render directories, for the runtime-state prune.
    """
    from .build_profile import resolve_build_document
    from .profile_conventions import unknown_root_entries

    deltas = _persona_deltas(shared.repo_root)
    if not deltas:
        return []

    # Already warned about once, by the deployment's own render. The entries are
    # a property of the repo root, so every render after the first would report
    # the same list again.
    already_warned = unknown_root_entries(shared.repo_root)

    rendered: list[Path] = []
    for delta in deltas:
        project_name = f"{shared.repo_root.name}-{delta.stem}"
        logger.debug("  Rendering persona %r → %s/", delta.stem, project_name)
        rendered.append(
            _render_project(
                shared,
                resolve_build_document(delta, None, shared.profile_overlays),
                profile_path=delta,
                project_name=project_name,
                output_dir=zones.stage,
                deployment=False,
                progress=logger.debug,
                extra_known=already_warned,
            )
        )
    return rendered


def _stage_source_zone(repo_root: Path, image_root: Path) -> None:
    """Copy the repo's SOURCE zone into the container repo being assembled.

    A container image carries a deployment repo, so it carries the repo's
    source, not just its render. Two things need it, and both are load-bearing:

    * ``profile.yml`` is the repo marker. Every repo-scoped verb — the image's
      own ``CMD`` among them — finds its deployment by walking up to one
      (:func:`~osprey.cli.repo_resolver.find_repo_root`), with no container
      exception, so an image without one boots into a repo-not-found refusal.
    * the rest is what the drift fingerprint folds. ``check_drift`` recomputes
      it from the source beside the render
      (:func:`~osprey.cli.build_profile_merge._fold_profile_material` covers the
      ``data:`` tree, every convention directory, ``triggers.yml`` and every
      persona delta), and a container carrying the profile alone reports its own
      untouched build as DRIFTED, naming files nobody edited.

    Copied by EXCLUSION rather than by naming what to take: everything at the
    repo root that is not a derived zone, git's own directory, or a secret. A
    list of wanted names would be a second enumeration of the fold's inputs, and
    the day the two disagreed a container would report false drift — the exact
    failure this exists to prevent. Excluding is a superset by construction, so
    it cannot drift from the fold; it is also just the truth, since the source
    zone is what a fresh clone of this repo holds.

    ``.env`` is excluded HERE, not left to the image's ``.dockerignore``: this
    copy decides what the build context contains at all, and a secret that never
    enters the context cannot be baked in by a later pattern that fails to match
    it at the depth it landed.

    :param repo_root: The deployment repo whose source zone this is.
    :param image_root: The container repo root being assembled.
    """

    def _ignore_env_files(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name.startswith(".env")}

    for entry in sorted(repo_root.iterdir()):
        if entry.name in _NON_SOURCE_ROOT_ENTRIES or entry.name.startswith(".env"):
            continue
        target = image_root / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, symlinks=False, ignore=_ignore_env_files)
        else:
            shutil.copy2(entry, target)


def _strip_secrets(image_root: Path) -> None:
    """Remove every ``.env`` from the container repo — at any depth.

    The repo's ``.env`` is the deployment's whole secret store and it is a
    HOST file: compose reads it and mounts what the containers need. An image
    must never hold one — secrets reach a container at run time, through
    ``--env-file`` / ``env_file:``. This sweeps the assembled tree for any, at
    any depth, whatever wrote it.

    Deliberately a sweep rather than a rule about who writes what. Two things
    already keep secrets out — :func:`_stage_source_zone` copies no ``.env`` in,
    and the context-root ``.dockerignore``
    (:func:`_write_image_context_dockerignore`) excludes them at every depth —
    and this is the third, the one that does not depend on either of the others
    having anticipated where a secret came from. Cheap, and the failure it
    guards is a facility's provider keys inside a distributable image.

    Done on the tree rather than left to the ``.dockerignore`` alone because a
    file that never enters the build context cannot be baked in by a pattern
    that failed to match it — and patterns here are easy to get wrong, since
    this context has depth and a root-anchored one matches nothing at it.

    ``.env.example`` stays. It carries no secrets, documents what an operator has
    to supply, and the shipped ``.dockerignore`` already makes that exception.
    """
    for path in image_root.rglob(".env*"):
        if path.is_file() and path.name != ".env.example":
            path.unlink()


def _write_image_context_dockerignore(image_root: Path) -> list[str]:
    """Emit the ``.dockerignore`` that governs the image's build context.

    A ``.dockerignore`` is read at the context ROOT and nowhere else, and this
    context is a repo: its root holds ``profile.yml`` and the source zone, and
    everything the shipped patterns name — the render's own files above all —
    sits one level down under ``build/``. The rendered ``.dockerignore``
    (``templates/project/dockerignore``) was written for a context whose root
    *was* the render, so every one of its patterns is root-anchored, and at this
    depth root-anchored means matching nothing. Measured, not reasoned: in a
    context of exactly this shape, a plain ``.env*`` line let a ``.env`` one
    level down into the image; ``**/.env*`` kept it out. (The measurement is not
    re-stated with a specific path, because which file sat there depended on
    what the render carried at the time — the pattern-anchoring result is the
    durable part, and it is what the rest of this function rests on.)

    So the patterns are re-spelled ``**/``-anchored rather than copied. Docker
    matches a pattern against each path element, so the ``**/`` prefix is what
    turns "this name at the root" into "this name at any depth" — which is what
    every one of them meant in the first place. The file is derived from the
    render's own copy rather than restated here, so the two cannot drift: one
    list of what must never enter an image, spelled for two context shapes.

    Two deliberate differences from the source list:

    * ``Dockerfile`` is NOT excluded, so the image ships its own build recipe at
      ``build/Dockerfile``. Accepted deliberately, not by omission: at this depth
      the pattern would name the very file the build is driven from
      (``-f <context>/build/Dockerfile``), and :func:`_prune_ignored_entries`
      removes what this file excludes from the context tree — so excluding it
      would delete the recipe before docker could read it. It is also the honest
      shape: ``build/`` here IS a rendered deployment, and a rendered deployment
      on a host has its Dockerfile in it.
    * There is no ``build/`` entry (the source has none either, for the same
      reason): here ``build/`` IS the deployment being shipped, and excluding it
      would produce an image with no config, no ``.mcp.json`` and no Claude Code
      artifacts at all.

    This is the SECOND guard on the repo's secrets, not the first:
    :func:`_strip_secrets` has already removed every ``.env`` from the tree. Two
    independent guards is the right number for a facility's provider keys — but
    only if the one that runs at build time actually matches.

    :returns: The patterns written, in file order, for
        :func:`_prune_ignored_entries` to apply to the tree itself.
    """
    source = (image_root / BUILD_DIR_NAME / ".dockerignore").read_text(encoding="utf-8")
    patterns: list[str] = []
    for raw in source.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line == "Dockerfile":
            continue
        negation, pattern = ("!", line[1:]) if line.startswith("!") else ("", line)
        patterns.append(f"{negation}**/{pattern}")
    image_root.joinpath(".dockerignore").write_text(
        "# Generated by `osprey build` for THIS context: a deployment repo, whose\n"
        "# root is one level above the render. Every pattern is the rendered\n"
        "# `.dockerignore`'s, re-spelled `**/`-anchored — a root-anchored pattern\n"
        "# matches nothing at this depth, which would let the render's own .env\n"
        "# through. Do not 'fix' the spelling back.\n"
        "#\n"
        "# `Dockerfile` is deliberately absent: at this depth it names the build's\n"
        "# own recipe (`-f build/Dockerfile`) and the service recipes under it —\n"
        "# the image therefore ships them, which is what a rendered deployment\n"
        "# looks like on a host too. `.dockerignore` itself is not excluded\n"
        "# either — the Dockerfile's `COPY .dockerignore *.wh[l]` uses it as the\n"
        "# guaranteed-present sibling that keeps the glob matching when no wheel\n"
        "# is staged.\n"
        "#\n"
        "# The build also DELETES everything below from the context tree, so this\n"
        "# list is a record of what is already gone rather than the only thing\n"
        "# keeping it out. That is what makes the tree the build fingerprints and\n"
        "# the tree the image receives the same tree.\n\n" + "\n".join(patterns) + "\n",
        encoding="utf-8",
    )
    return patterns


def _matches_ignore_pattern(relative_posix: str, pattern: str) -> bool:
    """Whether a context-relative path is named by one ``**/``-anchored pattern.

    Only the spelling :func:`_write_image_context_dockerignore` emits is
    supported: a ``**/`` prefix over one or more literal-or-glob path segments,
    with an optional trailing ``/``. Matching is done segment by segment against
    the path's TAIL, which is what ``**/`` means — "this name at any depth" —
    and case-sensitively, because docker is.

    A directory match is not extended to its contents here; the caller removes a
    matched directory whole, which does that and is cheaper.
    """
    from fnmatch import fnmatchcase

    segments = pattern.removeprefix("**/").strip("/").split("/")
    parts = relative_posix.split("/")
    if len(parts) < len(segments):
        return False
    return all(
        fnmatchcase(part, segment)
        for part, segment in zip(parts[-len(segments) :], segments, strict=True)
    )


def _prune_ignored_entries(image_root: Path, patterns: Sequence[str]) -> None:
    """Delete from the context tree whatever the context's ``.dockerignore`` excludes.

    A ``.dockerignore`` decides what the IMAGE receives; it does not touch the
    context on disk. That difference is invisible until something fingerprints
    the context — and :func:`_relocate_container_manifest` does, stamping the
    profile hash that ``osprey status`` and the drift gate re-check inside the
    container. The hash folds the profile's file inputs (``data/``, the
    convention directories, ``personas/``), and those patterns are ``**/``-
    anchored, so a stray ``data/ingest.log`` or a ``__pycache__/`` under a
    convention directory would be folded into the stamp here and then be missing
    from the image there. The container would recompute a different hash and
    report its own untouched build as drifted — the exact failure the whole
    relocation exists to prevent, arriving from the other direction.

    So the exclusion is applied to the tree, once, BEFORE the stamp: what the
    build fingerprints is then byte for byte what the image gets, for every fold
    input, by construction rather than by two lists agreeing. It also stops the
    same files being silently absent from a shipped ``data/`` tree the profile
    says is there.

    Negations (``!**/.env.example``) are honored, so what the list keeps, this
    keeps.

    :param image_root: The container repo root being assembled.
    :param patterns: What :func:`_write_image_context_dockerignore` emitted.
    """
    keep = [pattern[1:] for pattern in patterns if pattern.startswith("!")]
    drop = [pattern for pattern in patterns if not pattern.startswith("!")]

    for path in sorted(image_root.rglob("*"), key=lambda entry: len(entry.parts)):
        if not path.exists():
            # A parent directory matched and was already removed whole.
            continue
        relative = path.relative_to(image_root).as_posix()
        if any(_matches_ignore_pattern(relative, pattern) for pattern in keep):
            continue
        if not any(_matches_ignore_pattern(relative, pattern) for pattern in drop):
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _relocate_container_manifest(image_root: Path, runtime_root: str, profile_relpath: str) -> None:
    """Make the container repo's build manifest describe the container's repo.

    Two records in a rendered manifest name the machine that built it, and both
    are wrong once the tree is an image:

    * ``build_args.profile_path`` / ``profile_path_abs`` — the profile this
      render came from, recorded as the building host's absolute path. Rewritten
      to paths inside the container, which is where those files actually are: the
      source zone travels with the render, so both the root profile and the
      persona deltas ARE in the image. These are the last host strings in the
      tree; with them gone, "no path in an image names the build machine" is a
      property that can be asserted in one line.

      The two are rewritten to DIFFERENT files, on purpose.
      ``profile_path`` keeps naming the file this render came from — the
      persona delta for a persona image — because that is its job: provenance.
      ``profile_path_abs`` is not provenance; it is the input
      :func:`~osprey.deployment.staleness.staleness_reasons` hashes and compares
      against ``creation.preset_hash``, so it must name the same file the stamp
      below is computed from — the repo's own ``profile.yml``. Pointed at the
      delta it would hash the delta against a root-profile stamp and every
      persona container would print "profile has changed" on an untouched build:
      a false drift report from the advisory while
      :func:`~osprey.deployment.staleness.check_drift`, which reads the profile
      off the repo root, correctly says CLEAN. Advisory and gate now agree by
      construction, because they hash the same file.
    * ``creation.preset_hash`` — the fingerprint
      :func:`osprey.deployment.staleness.check_drift` holds ``build/`` against.
      For the deployment's own image this is already right (same profile, same
      fold) and the rewrite is a no-op. For a PERSONA's image it is not: that
      render came from ``personas/<name>.yml`` merged over the root, while the
      ``profile.yml`` at the root of the repo the image carries is the root
      profile — so an operator running ``osprey status`` in that container would
      be told its own untouched build had drifted. Re-stamped from the profile
      the container actually holds, which is the profile that produced this
      whole tree; the persona delta is folded into that fingerprint too.

    Skipped quietly if the manifest or the fingerprint cannot be read: a build
    that has rendered a complete tree must not fail on its commentary.
    """
    from osprey.deployment import staleness

    manifest_path = image_root / BUILD_DIR_NAME / staleness.build_manifest_path(image_root).name
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("Container manifest relocation skipped: %s", exc)
        return

    container_profile = f"{runtime_root}/{profile_relpath}"
    build_args = manifest.get("build_args")
    if isinstance(build_args, dict):
        if "profile_path" in build_args:
            build_args["profile_path"] = container_profile
        # Set unconditionally rather than only when already present: this is the
        # file the container's staleness advisory hashes, and leaving it absent
        # would fall the advisory back to `profile_path` (the delta) or to
        # `preset`, neither of which is what the stamp below is computed from.
        build_args["profile_path_abs"] = f"{runtime_root}/{PROFILE_FILENAME}"

    fingerprint = staleness.profile_fingerprint(image_root / PROFILE_FILENAME)
    if fingerprint is not None:
        manifest.setdefault("creation", {})["preset_hash"] = fingerprint.profile_hash

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _render_container_project(
    shared: _SharedRenderInputs,
    resolved: Any,
    zones: _RenderZones,
    *,
    profile_path: Path,
    project_name: str,
    deployment: bool = True,
) -> Path:
    """Render the copy of a project that a container image is built from.

    A rendered project records absolute paths — ``project_root`` in
    ``config.yml``, and the ``OSPREY_CONFIG`` / ``CONFIG_FILE`` every MCP server
    in ``.mcp.json`` is handed — and the host's are not the container's. An image
    built from the host render therefore ships servers pointed at the building
    machine's directories, and an agent whose ``agent_data.base_dir`` resolves
    beside the mounted volume rather than into it. So the build renders the
    project a second time, against the path the container will see it at.

    ``--runtime-root`` is that substitution and already exists; this is its
    caller. The render is otherwise the deployment's own, from the same resolved
    profile, so the image cannot disagree with the host about anything except
    where it is.

    The result is a REPO, not a render: ``profile.yml`` and the rest of the
    source zone at ``<image root>/``, the render below it in ``build/``. That is
    the one container layout — ``{project_root}/build/config.yml`` is then true
    in a container exactly as it is on a host, which is what
    :data:`~osprey.registry.mcp.RENDERED_CONFIG_ENV_VALUE` has always claimed —
    and it is what lets the image's ``Dockerfile`` stay a verbatim ``COPY`` of
    its build context.

    Lands in the staged tree, so it transits the atomic swap with everything
    else: a container copy that fails to render fails the whole build and leaves
    the previous ``build/`` — every project in it — exactly as it was.

    What it deliberately does NOT get, beyond what a persona render skips:

    * **the venv and the dependency install** — an image builds its own
      environment from ``OSPREY_PIP_SPEC``, and a venv recording the host's
      interpreter would be actively wrong inside a container. ``shared`` already
      carries the resolved dependency list, so the rendered ``Dockerfile`` still
      names the right requirements;
    * **the compose files** — skipped by :func:`_render_compose_files` under a
      runtime root anyway, and compose is a host-side concern: it is what starts
      the container, not something the container holds;
    * **the lifecycle phases** — ``pre_build``/``post_build``/``validate`` are
      the profile's own shell commands and ran once for this build already.

    :param deployment: Whether this is the deployment's own image rather than a
        persona's — see :func:`_render_project`'s parameter of the same name.
    :returns: The container repo root, ready to be a build context.
    """
    from .profile_conventions import unknown_root_entries

    image_root = zones.stage / IMAGE_DIR_NAME / project_name
    runtime_root = f"/app/{project_name}"
    # Rendered one level down and then renamed up, because `_render_project`
    # writes to `<output_dir>/<project_name>` and this render's directory has to
    # be `build/` while its project keeps its own name. A rename inside the
    # staged tree, so it stays on one filesystem and costs nothing.
    scratch = image_root / ".render"
    rendered = _render_project(
        shared._replace(runtime_root=runtime_root, runtime_interpreter=_CONTAINER_INTERPRETER),
        resolved,
        profile_path=profile_path,
        project_name=project_name,
        output_dir=scratch,
        deployment=deployment,
        progress=logger.debug,
        extra_known=unknown_root_entries(shared.repo_root),
    )
    rendered.rename(image_root / BUILD_DIR_NAME)
    scratch.rmdir()
    _stage_source_zone(shared.repo_root, image_root)
    _strip_secrets(image_root)
    # Written, then applied to the tree, and only then is the manifest stamped:
    # the fingerprint has to be taken over the tree the image will actually
    # receive, or the container recomputes a different one and reports drift.
    _prune_ignored_entries(image_root, _write_image_context_dockerignore(image_root))
    _relocate_container_manifest(
        image_root,
        runtime_root,
        profile_path.relative_to(shared.repo_root).as_posix(),
    )
    return image_root


def _render_container_projects(
    shared: _SharedRenderInputs,
    resolved: Any,
    zones: _RenderZones,
    *,
    profile_path: Path,
    project_name: str,
) -> list[Path]:
    """Every image context this build produces — the deployment's and each persona's.

    One rule, applied as many times as this repo has images to build: an image's
    build context is ``build/.image/<the image's project name>/``, and it is a
    deployment repo rendered against the ``/app/<project name>`` path that image
    will see itself at. The deployment gets one; every delta under ``personas/``
    gets one, because a persona is deployed as its own image and needs the same
    things the deployment's does.

    A persona's needs are in fact sharper. Its render is the ONLY place its
    ``config:`` deltas exist — ``control_system.writes_enabled`` among them —
    and a persona image built from the deployment's render would come up with
    the deployment's config and none of that persona's, which is the difference
    between a read-only terminal and one that writes to hardware. The persona's
    ``.mcp.json`` names ``/app/<repo>-<persona>/build/config.yml``, so its
    servers read ITS config, and only its own image carries that file.

    The host-side persona renders at ``build/<repo>-<persona>/`` are untouched
    and stay FLAT: the credential sweep
    (:func:`osprey.deployment.web_terminals.env_production._claude_code_auth_secret_vars`),
    the render check
    (:func:`osprey.deployment.web_terminals.persona_images._check_existing_render`)
    and the lint rule all read ``config.yml`` at that render's root, and the
    catalog's ``project_path`` is an externally-pinned contract. Personas get a
    container copy in addition, exactly as the deployment does — not instead.

    :returns: The image context roots, in build order.
    """
    from .build_profile import resolve_build_document

    contexts = [
        _render_container_project(
            shared,
            resolved,
            zones,
            profile_path=profile_path,
            project_name=project_name,
            deployment=True,
        )
    ]
    for delta in _persona_deltas(shared.repo_root):
        contexts.append(
            _render_container_project(
                shared,
                resolve_build_document(delta, None, shared.profile_overlays),
                zones,
                profile_path=delta,
                project_name=f"{shared.repo_root.name}-{delta.stem}",
                deployment=False,
            )
        )
    return contexts


def _wire_build_derived_env(repo_root: Path, build_dir: Path) -> None:
    """Point the deployment's ``.env`` at the manifest this build generated.

    The last link of the virtual accelerator's channel chain, and the only one
    that reaches outside ``build/``. The generator writes its manifest into the
    output zone (:func:`_render_project`); the VA compose service mounts that
    directory; and the address of the manifest *inside* the mount travels as
    ``VA_CHANNELS_FILE``, which compose can only substitute from the repo-root
    ``.env`` it is handed as ``--env-file``. Nothing else reads these keys —
    the container's entrypoint takes them straight from its environment — so
    this is where the pointer is written or it is not written at all.

    Two rules, and neither is negotiable:

    * **Append-only.** That ``.env`` is the deployment's whole secret store:
      hand-edited, and written back to by ``osprey up`` with tokens the running
      volumes are pinned to. The build writes it through the same
      :func:`~osprey.utils.dotenv.append_profile_env` every other writer uses,
      so a value already on file always wins and a disagreement is *reported*
      rather than resolved. Repointing a running IOC's channel set from under
      an operator, on a rebuild they ran for some unrelated reason, is not a
      thing a build gets to do.
    * **After the swap.** Called once ``build/`` is the tree this render
      produced, and gated on the manifest being in it — so the pointer is only
      ever written when the file it names is already there to be found. A build
      that fails leaves ``build/`` as it was and this never runs.

    The reverse case — a repo whose ``.env`` still carries a pointer from a
    build that could generate a manifest, run again on a tree that cannot — is
    the one thing append-only cannot fix by itself, so it is warned about by
    name. The stale pointer is not harmless: the entrypoint *raises* on a
    manifest file it cannot find rather than falling back to the packaged
    channel set, so the next start gets a container that will not boot.

    Args:
        repo_root: The deployment repo — the compose project directory, whose
            ``.env`` is the file compose interpolates from.
        build_dir: The output zone, after the swap.
    """
    from osprey.deployment.compose_generator import COMPOSE_ENV_FILENAME
    from osprey.services.virtual_accelerator.manifest.build import MANIFEST_FILENAME
    from osprey.utils.dotenv import (
        BUILD_DERIVED_BANNER,
        BUILD_DERIVED_KEYS,
        append_profile_env,
        parse_dotenv_file,
    )

    env_path = repo_root / COMPOSE_ENV_FILENAME
    manifest = build_dir / "data" / "simulation" / MANIFEST_FILENAME

    if not manifest.is_file():
        on_file = parse_dotenv_file(env_path) if env_path.is_file() else {}
        for key in sorted(BUILD_DERIVED_KEYS & on_file.keys()):
            logger.warning(
                "  %s is set in %s, but this build generated no virtual-accelerator "
                "channel manifest for it to point at. The value was left alone, since it is "
                "yours and not the build's. The IOC will fail to start against a "
                "manifest that is not there. Remove the line, or restore the channel "
                "databases the manifest is generated from.",
                key,
                env_path,
            )
        return

    # A name, not a path: the entrypoint resolves a relative VA_CHANNELS_FILE
    # against its data mount, which is the directory the manifest was just
    # written into.
    #
    # VA_LATTICE is stated rather than left to default, and stated
    # unconditionally, because the entrypoint's default for a FILE-backed
    # source is `none` — it assumes a facility manifest has no PyAT model
    # behind it. A generated manifest is the other case: it is only ever built
    # from a tree carrying the paradigm channel databases, whose machine is the
    # lattice-backed one, so defaulting here would drop the physics bridge on
    # exactly the projects that have a lattice to run.
    entries = {"VA_CHANNELS_FILE": MANIFEST_FILENAME, "VA_LATTICE": "builtin"}
    result = append_profile_env(env_path, entries, BUILD_DERIVED_BANNER)

    if result.added:
        # The build's one write outside build/, into a file that is the
        # operator's rather than a build artifact. It runs after the render
        # phase has closed, so it is reported rather than stepped.
        _report_fact(
            f"Pointed {COMPOSE_ENV_FILENAME} at the generated channel manifest "
            f"({', '.join(sorted(result.added))})"
        )
    for conflict in result.conflicts:
        # Named, never valued: the store this reads is the one holding the
        # facility's provider keys, and a warning is not a safe place for it.
        logger.warning(
            "  %s in %s disagrees with what this build generated. Your value was kept, "
            "because the build never overwrites this file. The IOC will serve the channel "
            "set you named, not the one in build/. Remove the line to take the build's.",
            conflict.key,
            env_path,
        )


def _build_repo(
    repo: Path | None,
    *,
    stream: bool,
    skip_lifecycle: bool,
    skip_deps: bool,
    runtime_root: str | None,
    dev: bool = False,
) -> None:
    """Render a deployment repo's ``build/`` zone from its ``profile.yml``.

    The zero-argument build. Everything it needs it derives: the repo from where
    the operator is standing, the deployment name from that repo's directory
    name, the profile from its root, and the destination from the zone layout.

    What it renders is the whole OUTPUT zone in one pass — the project (config,
    Claude Code artifacts, data tree, service templates, injected services), the
    compose files that deploy it, and one project per persona delta
    (:func:`_render_persona_projects`). It lands in
    ``build/.tmp`` and replaces ``build/`` only once every step below has
    succeeded; see :func:`_swap_in_render`.

    *Which* profile it renders is the one thing the host gets a say in. A
    ``.env.variant`` at the repo root may name an overlay under ``profiles/``
    (:func:`~osprey.cli.variant_selection.resolve_variant_selection`); when it
    does, that overlay is merged over ``profile.yml`` as a profile layer before
    resolution, for the deployment and for every persona alike. One repo, one
    tracked profile, and as many host renders as the repo has variants.

    Args:
        repo: ``--repo``, or ``None`` to walk up from the working directory.
        stream: Stream lifecycle-phase output as it is produced.
        skip_lifecycle: Skip the profile's ``pre_build``/``post_build``/
            ``validate`` phases.
        skip_deps: Skip the project venv and its dependency install (CI mode).
        runtime_root: Absolute path the *container* will see this repo at, when
            that differs from where it is being built. Substituted for the repo
            root everywhere the render records one.
        dev: ``--dev`` — render a dev build: each service build context gets a
            wheel from the local checkout and the ``OSPREY_DEV`` build arg, so
            the images run this checkout instead of the pinned release.

    Raises:
        click.Abort: On any failure, a persona delta that will not resolve
            included. ``build/`` is left as it was found.
        click.UsageError: When no deployment repo encloses the search path.
    """
    from osprey.deployment.errors import CapturedProcessError
    from osprey.deployment.subprocess_capture import diagnose_captured_failure

    from .build_profile import resolve_build_document
    from .build_profile_deploy import deploy_aware_config_errors, deploy_aware_config_warnings
    from .phase_reporter import current_reporter
    from .variant_selection import VARIANT_DIRNAME, resolve_variant_selection

    # Whatever the verb at the top of this run installed — this build's own
    # reporter under `osprey build`, and the one `init --up` or `up --build`
    # already had open when they chained here.
    reporter = current_reporter()

    repo_root = find_repo_root(repo)
    profile_path = repo_root / PROFILE_FILENAME
    # The repo is the deployment and its directory name is the deployment's
    # name: it is what the compose project, the container labels and the local
    # image tags are derived from. There is no name to pass and none to store.
    name = repo_root.name
    zones = _render_zones(repo_root)

    logger.debug("Building %s", repo_root)

    # The durable zone first: a fresh clone carries no git-ignored directory,
    # and the render below records paths into it.
    _ensure_state_zone(repo_root)
    # Then anything a previous run left behind — including, in the one window
    # where it can happen, a `build/` that a kill removed.
    _repair_interrupted_swap(zones)

    # Both are reported after the swap, and both are only meaningful once it
    # has happened; named here so every exit path below has them.
    backup_dir: Path | None = None
    config: dict[str, Any] | None = None

    try:
        # Which profile this HOST builds, decided before anything is resolved.
        # The overlay is a profile layer like a `-O` file: it merges over
        # `profile.yml` by the same deep merge, ahead of `extends:` resolution,
        # and anchors at the repo root — so the render reads the merged result
        # and every profile-relative path still points at this repo. A future
        # `osprey build --variant` would win over the file; nothing in the
        # command line names a variant today.
        variant = resolve_variant_selection(repo_root)
        profile_overlays: tuple[Path, ...] = (variant.path,) if variant.path is not None else ()

        resolved = resolve_build_document(profile_path, None, profile_overlays)
        build_profile = resolved.profile

        # The profile's own checks first, then the ones that need the config
        # this build is about to render (the `config:` block plus what the
        # `deploy:` block contributes to it).
        # `repo_root`, not the working directory: a catalog entry's
        # `build_profile: personas/<name>.yml` is named relative to the profile.
        # Without it, `osprey build` from a subdirectory or through `--repo`
        # read no persona delta at all and passed a roster the gate exists to
        # refuse.
        web_errors = deploy_aware_config_errors(
            build_profile.deploy, build_profile.config, profile_root=repo_root
        )
        if web_errors:
            raise click.UsageError("Profile validation failed:\n  - " + "\n  - ".join(web_errors))
        web_warnings = deploy_aware_config_warnings(
            build_profile.deploy, build_profile.config, profile_root=repo_root
        )
        if not build_profile.provider:
            raise click.UsageError(
                f"{PROFILE_FILENAME} names no provider. Add `provider: "
                "<als-apg|cborg|anthropic|amsc-i2|argo>` — or any custom "
                "provider you declare under `config:` api.providers — or run "
                "`osprey set provider=<...>`."
            )
        _check_osprey_version_requirement(build_profile)

        # Outside every phase, on purpose. The next thing this build opens is
        # `Preparing the project environment`, the long pole: hung under a
        # phase this identity would arrive minutes after the wait it labels,
        # and as a step it would vanish entirely (no phase is open here). A
        # note rather than a report line: it is context for the phases that
        # follow, not something the operator ran the verb to find out.
        from . import output

        output.note(
            f"profile {build_profile.name} (bundle {build_profile.data_bundle}, "
            f"tier {build_profile.resolved_tier()})"
        )
        # Said out loud whenever a variant is in force: the same repo builds
        # differently on this host than on the next one, and an operator
        # reading a render has to be told which of the two they are looking at.
        if variant.selected:
            output.note(
                f"host variant {variant.name} "
                f"({VARIANT_DIRNAME}/{variant.name}.yml over {PROFILE_FILENAME})"
            )

        # Advisory lint findings — real exposures that are deliberately not
        # build-failing (a privileged terminal in front of `auth.method: none`
        # is the shipped loopback posture, not a mistake). Printed here, beside
        # the profile identity, because a finding nobody prints is a finding
        # nobody has: every surface that gates on the errors discards these.
        for web_warning in web_warnings:
            output.note(f"⚠ {web_warning}")

        # A fresh staging tree, and the output zone it will replace. Both are
        # created now: the venv below is written into `build/` directly, at the
        # path it will be used from.
        zones.build_dir.mkdir(parents=True, exist_ok=True)
        zones.stage.mkdir(parents=True)

        if build_profile.lifecycle.pre_build and not skip_lifecycle:
            _run_lifecycle_phase(
                "pre_build",
                build_profile.lifecycle.pre_build,
                repo_root,
                zones.stage,
                stream=stream,
            )

        # The project venv, at its final path. It is the one artifact that
        # cannot be rendered somewhere and moved (see `_swap_in_render`), so it
        # is written where it will be read from and joins the staged tree at
        # swap time. Its dependency record, which `_create_project_venv` writes
        # beside it, is copied into the staged tree — the record and the
        # environment it describes must ship together.
        if not skip_deps:
            # Reported on its own, not folded into the render below: it is the
            # longest thing a first build does by a wide margin, and a build
            # that looks stuck for two minutes is the moment an operator most
            # needs to be told what it is waiting on.
            with reporter.phase("Preparing the project environment"):
                project_deps = _create_project_venv(zones.build_dir, build_profile)
                recorded = zones.build_dir / "pyproject.toml"
                if recorded.is_file():
                    shutil.copy2(recorded, zones.stage / "pyproject.toml")
        else:
            project_deps = list(build_profile.dependencies or [])

        shared = _SharedRenderInputs(
            repo_root=repo_root,
            build_dir=zones.build_dir,
            runtime_root=runtime_root,
            project_deps=project_deps,
            skip_deps=skip_deps,
            manager=TemplateManager(),
            va_manifests={},
            profile_overlays=profile_overlays,
        )

        # One reported phase over every render pass this build makes — the
        # deployment, the compose files, one per persona delta, one per image
        # copy. They are six passes over the same profile producing one build/,
        # and reported one by one they read as six builds. Each pass gets a step
        # line naming what it produced instead.
        with reporter.phase("Rendering the configuration") as phase:
            # Only this pass records what it injected: all six inject the same
            # set, and the operator wants the components named once.
            injected: list[str] = []
            host_render = _render_project(
                shared,
                resolved,
                profile_path=profile_path,
                project_name=name,
                output_dir=zones.stage.parent,
                deployment=True,
                # The step lines below are the default view of this pass now, so
                # its own line-by-line narration joins the other five passes at
                # DEBUG, where `--verbose` still has all of it.
                progress=logger.debug,
                injected_out=injected,
            )
            # Every render after this one — each persona's, and each image
            # copy's — is told about the deployment's services from the render
            # just written (see _SharedRenderInputs.host_config). A profile
            # that is itself attached hosts nothing and projects nothing.
            if build_profile.deploy_services:
                shared = shared._replace(host_config=_rendered_config(host_render))
            phase.step("project files, agent artifacts and services")
            if injected:
                phase.step(f"services injected: {', '.join(injected)}")

            config = _render_compose_files(zones, runtime_root, dev_mode=dev)
            phase.step("compose files")

            persona_renders = _render_persona_projects(shared, zones)
            if persona_renders:
                phase.step(f"{len(persona_renders)} persona render(s)")

            # The copies the images are built from — the deployment's and one per
            # persona — each rendered against the path its container sees itself at
            # rather than the host's. Skipped under an explicit --runtime-root: that
            # build is ALREADY aimed at a runtime elsewhere, and rendering a second
            # relocation inside it would be guessing which of the two the operator
            # meant.
            image_renders = (
                []
                if runtime_root
                else _render_container_projects(
                    shared, resolved, zones, profile_path=profile_path, project_name=name
                )
            )
            if image_renders:
                phase.step(f"{len(image_renders)} image build context(s)")

        # Both remaining phases run against the staged tree, before the swap:
        # a profile whose own validation fails must not be able to replace a
        # build that worked.
        if build_profile.lifecycle.post_build and not skip_lifecycle:
            _run_lifecycle_phase(
                "post_build",
                build_profile.lifecycle.post_build,
                zones.stage,
                zones.stage,
                stream=stream,
            )
        if build_profile.lifecycle.validate and not skip_lifecycle:
            _run_lifecycle_phase(
                "validate",
                build_profile.lifecycle.validate,
                zones.stage,
                zones.stage,
                abort_on_failure=False,
                stream=stream,
            )

        backup_dir = _backup_outgoing_claude_artifacts(zones)
        # After the backup (which reads the outgoing build/ against the staged
        # tree) and before the swap, so build/ is never published carrying
        # runtime-state directories that belong at the repo root.
        # Each container copy's render is a render like any other, so it gets the
        # same prune — its runtime-state directories would otherwise be baked
        # into the image, where the agent-data volume mounts straight over them.
        _prune_runtime_state_from_stage(
            zones,
            renders=[
                zones.stage,
                *persona_renders,
                *(image_root / BUILD_DIR_NAME for image_root in image_renders),
            ],
        )
        _swap_in_render(zones)
        # The one write outside build/, and last for that reason: it names a
        # file in the tree the line above just published.
        _wire_build_derived_env(repo_root, zones.build_dir)

    except click.Abort:
        raise
    except click.UsageError as e:
        logger.error("✗ %s", e)
        raise
    except BuildProfileError as e:
        logger.error("✗ Build error: %s", e)
        raise click.Abort() from e
    except ValueError as e:
        logger.error("✗ Error: %s", e)
        raise click.Abort() from e
    except CapturedProcessError as e:
        # A child that exited non-zero is a build failure, not an unexpected
        # one. Its output has already been replayed by the phase this ran
        # under, so the message names the spool rather than repeating it — and
        # carries no ✗ of its own, because that phase's failure line has one.
        logger.error("Build failed: %s", e)
        # Some of those failures say nothing an operator can act on — a registry
        # 401 naming a password that was never sent is the standing example. The
        # same seam `osprey up` uses answers them here, and stays silent on
        # every failure it cannot name.
        remedy = diagnose_captured_failure(e)
        if remedy is not None:
            logger.error("→ %s", remedy)
        raise click.Abort() from e
    except Exception as e:
        logger.error("✗ Unexpected error: %s", e)
        import traceback

        logger.debug(traceback.format_exc())
        raise click.Abort() from e
    finally:
        # The staging tree never outlives the build that made it, however that
        # build ended. A successful swap has already moved it; anything left
        # here is the debris of one that did not.
        if zones.stage_root.exists():
            shutil.rmtree(zones.stage_root, ignore_errors=True)

    if backup_dir is not None:
        logger.debug("  ✓ Previous Claude Code artifacts saved to %s", backup_dir)
    _warn_if_deployment_running(config, name)


def _check_osprey_version_requirement(build_profile: Any) -> None:
    """Refuse a profile that declares an OSPREY it does not have.

    Compares the release *lineage* rather than the running version: a
    development checkout carries a post/local segment no release specifier is
    written against. The schema floor this code ships is also a floor on what it
    satisfies — between releases a checkout writes the next release's profile
    schema while its tag still names the previous one, so judging it by tag
    alone would make it refuse profiles it just wrote.
    """
    if not build_profile.requires_osprey_version:
        return

    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    from osprey.version import get_release_version

    from .build_profile_load import _PROFILE_SCHEMA_MIN_OSPREY

    spec = SpecifierSet(build_profile.requires_osprey_version, prereleases=True)
    current = max(Version(get_release_version()), Version(_PROFILE_SCHEMA_MIN_OSPREY))
    if current not in spec:
        logger.error(
            "  ✗ OSPREY %s does not satisfy requires_osprey_version: %s",
            current,
            build_profile.requires_osprey_version,
        )
        raise click.Abort()
    logger.debug("  ✓ OSPREY %s satisfies %s", current, build_profile.requires_osprey_version)


def _collect_profile_artifacts(
    build_profile: Any, *, progress: Any = logger.info
) -> dict[str, list[str]]:
    """The artifact selections this profile makes, validated against the library.

    ``web_panels`` is validated at manifest load time (warn-only) and is not
    file-backed, so it bypasses the library check and is added by the caller.

    :param progress: Where the validation line goes — see
        :func:`_render_project`'s ``progress``.
    """
    artifacts: dict[str, list[str]] = {}
    for artifact_type in ("hooks", "rules", "skills", "agents", "output_styles"):
        names = getattr(build_profile, artifact_type, [])
        if names:
            artifacts[artifact_type] = list(names)

    if artifacts:
        from osprey.cli.templates.artifact_library import validate_artifacts

        validate_artifacts(artifacts)
        total = sum(len(v) for v in artifacts.values())
        progress(
            "  ✓ Validated %d artifact(s): %s",
            total,
            ", ".join(f"{len(v)} {k}" for k, v in artifacts.items()),
        )
    return artifacts


def _profile_setup_patch_capable(build_profile: Any) -> bool:
    """Whether *build_profile*'s render leaves the setup capability in place.

    :func:`~osprey.cli.profile_conventions.is_setup_patch_capable` answers this
    for a rendered persona config; this reaches the same answer one step
    earlier, from the ``config:`` overrides that config is about to be written
    from. The composition — deny minus ``remove_deny``, because a persona tier
    that lifts an inherited deny is capable — is NOT repeated here: the
    overrides are assembled into the ``config.yml``-shaped document the
    predicate reads and it does the subtraction, so there is one composition
    for both callers to be wrong or right together.

    The SPELLING is not owned here either, and deliberately no longer is.
    ``claude_code.permissions.deny`` can be written as one dotted key, as a
    ``claude_code.permissions:`` mapping holding ``deny``, or as a fully nested
    ``claude_code:`` block, and all three reach the same leaf in the rendered
    ``config.yml`` — ``config_update_fields`` takes any of them. This function
    used to read the first and the third; the middle one renders and was read as
    nothing, so a profile that denied the setup tool through
    ``claude_code.permissions: {deny: [...]}`` was reported capable and the
    Dockerfile chowned ``build/config.yml`` to a persona whose ``settings.json``
    denies the tool. Rather than adding the third reader,
    :func:`~osprey.deployment.web_terminals.personas.persona_capability_document`
    — the guard belt's reader, which already tries every split point — assembles
    the document, so exactly one spelling reader exists in the codebase and the
    container's verdict cannot disagree with the guard's about what a profile
    says.

    Reading every spelling and unioning them is exact rather than merely broad
    because a profile cannot carry two of them at once:
    :func:`~osprey.cli.build_profile_load._reject_mixed_claude_code_spellings`
    refuses a ``config:`` block that spells any ``claude_code`` path two ways,
    since ``config_update_fields`` would silently apply one and discard the
    other. Absent that refusal the union would be wrong in the direction that
    matters: ``remove_deny`` GRANTS the capability, so unioning a dotted lift
    with a nested deny would report ``True`` for a render whose
    ``settings.json`` denies the tool.

    Args:
        build_profile: The resolved profile for the render being built.

    Returns:
        ``True`` when nothing the profile denies takes the setup tool away.
    """
    from osprey.deployment.web_terminals.personas import persona_capability_document

    from .profile_conventions import is_setup_patch_capable

    overrides = build_profile.config if isinstance(build_profile.config, dict) else {}
    return is_setup_patch_capable(persona_capability_document(overrides))


def _repo_render_context(
    build_profile: Any,
    *,
    repo_root: Path,
    build_dir: Path,
    runtime_root: str | None,
    project_deps: list[str],
    skip_deps: bool,
    runtime_interpreter: str | None = None,
) -> dict[str, Any]:
    """The template context for a four-zone render.

    Two values in here are what make the render a *repo's* render rather than a
    directory's:

    ``project_root`` is the REPO root, not the directory being written. The
    render lives one level down in ``build/``, and every relative path in the
    generated config — the agent-data root, the audit log, the mounted ``.env``
    — is anchored on this value, so anchoring it on the render would put the
    deployment's durable state inside the zone that is wiped on every build.
    ``--runtime-root`` substitutes the path a *container* sees the repo at, for
    a build whose output runs somewhere other than where it was made.

    ``current_python_env`` is the interpreter every process that imports osprey
    is launched with — MCP servers, framework hooks, ``{current_python_env}``
    substitution in the registry — so every branch must produce one that has
    osprey importable. It names the venv's FINAL path, which is where the venv
    is created and where the swap leaves it. *runtime_interpreter* replaces the
    whole derivation for a render whose processes start on another machine
    (:data:`_CONTAINER_INTERPRETER`), where none of this filesystem's answers
    exist.
    """
    context: dict[str, Any] = {
        # Gates the rendered config's `services:`/`deployed_services:` blocks.
        "deploy_services": build_profile.deploy_services,
        "project_root": str(runtime_root or repo_root),
        "dependencies": project_deps,
        "pip_dependency_args": " ".join(shlex.quote(d) for d in project_deps),
        # Gates the ONE line of Dockerfile.j2 that hands `build/config.yml` to
        # the agent's user: a persona that can still reach the setup tool needs
        # the file that tool edits to be writable, and every other tier must
        # leave it root-owned. Composed here the way settings.json renders it —
        # the profile's own deny minus its `remove_deny`, which is how a tier
        # lifts the base floor — because the rendered config those keys land in
        # does not exist yet when this context is built.
        "is_setup_patch_capable": _profile_setup_patch_capable(build_profile),
    }
    if build_profile.provider:
        context["default_provider"] = build_profile.provider
    if build_profile.model:
        context["default_model"] = build_profile.model
    if build_profile.channel_finder_mode is not None:
        context["channel_finder_mode"] = build_profile.channel_finder_mode
    if build_profile.default_panel:
        context["default_panel"] = build_profile.default_panel
    if build_profile.panel_presets:
        context["panel_presets"] = build_profile.panel_presets
    if build_profile.claude_md_template:
        context["claude_md_template"] = build_profile.claude_md_template

    python_env = build_profile.python_env or "project"
    if runtime_interpreter:
        # Known, not derived: the processes this render describes start inside
        # an image, so no path on this filesystem — venv or generating
        # interpreter — is one of them. Ahead of the profile's own
        # `python_env:`, which names a path on the machine that BUILDS.
        context["current_python_env"] = runtime_interpreter
    elif skip_deps:
        # No venv was created. The interpreter running osprey is the one
        # interpreter guaranteed to have osprey importable — a bare "python"
        # gambles on a PATH that subprocess contexts do not inherit.
        context["current_python_env"] = sys.executable
    elif python_env == "project":
        context["current_python_env"] = str(build_dir / ".venv" / "bin" / "python")
    elif python_env == "build":
        context["current_python_env"] = sys.executable
    else:
        context["current_python_env"] = python_env

    # Provenance, not configuration: the profile's `environment:` block
    # verbatim, so a rendered config states which environment it was built from.
    environment = build_profile.environment
    context["environment_python"] = environment.python
    context["environment_packages"] = list(environment.packages or [])
    context["environment_inherit_exclude"] = list(environment.inherit_exclude or [])
    return context


def _inject_services(build_profile: Any, profile_dir: Path, project_path: Path) -> list[str]:
    """Scaffold the service tree and inject every service the profile declares.

    Skipped wholesale for an attached project (``deploy_services: false``): its
    service sections were parsed and validated, but it deploys nothing of its
    own and connects to a services stack another OSPREY deployment runs on the
    same host. The rendered config already carries an empty
    ``deployed_services: []``, so nothing here needs to run.

    Order is load-bearing. The two chat bridges gate their ``depends_on`` and
    their in-network dispatcher URLs on ``event_dispatcher``/``dispatch_worker``
    already being in ``deployed_services``, which is what the dispatch injector
    writes there; the bluesky-web sidecar read-proxies the bluesky bridge and
    follows it for the same reason.

    Returns:
        The name of each component injected, in injection order — what the
        build reports as one step line. Empty when nothing was injected.
    """
    if not build_profile.deploy_services:
        logger.debug(
            "deploy_services: false. This is an attached project, so no services were "
            "scaffolded (it connects to a shared OSPREY services stack)."
        )
        return []

    injected: list[str] = []

    svc_count = _copy_service_templates(project_path)
    if svc_count:
        logger.debug("  ✓ Copied %d service template(s)", svc_count)

    if build_profile.services:
        psvc_count = _inject_profile_services(profile_dir, project_path, build_profile.services)
        logger.debug("  ✓ Injected %d profile service(s)", psvc_count)
        if psvc_count:
            # Counted rather than named: an unresolvable template is warned
            # about and skipped, so the profile's own keys can name more
            # services than were injected.
            injected.append(f"{psvc_count} profile service(s)")
    if build_profile.dispatch is not None:
        _inject_dispatch(build_profile.dispatch, profile_dir, project_path)
        injected.append("event dispatch")
    if build_profile.nextcloud_bridge is not None:
        _inject_nextcloud_bridge(build_profile.nextcloud_bridge, project_path)
        injected.append("Nextcloud Talk bridge")
    if build_profile.gchat_bridge is not None:
        _inject_gchat_bridge(build_profile.gchat_bridge, project_path)
        injected.append("Google Chat bridge")
    if build_profile.bluesky is not None:
        # The VA block is handed over because a two-lane deploy on a live
        # baseline puts its second lane on the virtual accelerator, and
        # _inject_va has not written that service to config.yml yet.
        _inject_bluesky(build_profile.bluesky, project_path, build_profile.virtual_accelerator)
        injected.append("bluesky bridge")
    if build_profile.bluesky_web is not None:
        _inject_bluesky_web(build_profile.bluesky_web, project_path)
        injected.append("bluesky web")
    if build_profile.virtual_accelerator is not None:
        _inject_va(build_profile.virtual_accelerator, project_path)
        injected.append("virtual accelerator")
    # Must follow the VA injector: the recorder's compose template gates its
    # image source, startup ordering and Channel Access addressing on
    # `virtual_accelerator` being in `deployed_services`, which is exactly what
    # _inject_va writes there. The connection block and the archive's knobs are
    # not written here — they reach the rendered config through the derived
    # overrides, which an attached project also gets (see
    # va_archiver_config_overrides).
    if build_profile.va_archiver is not None:
        _inject_va_archiver(build_profile.va_archiver, project_path)
        injected.append("archiver store")

    return injected


@click.command()
@click.option("--stream", "-s", is_flag=True, help="Stream lifecycle step output in real-time")
@click.option(
    "--skip-lifecycle", is_flag=True, help="Skip pre_build, post_build, and validate phases"
)
@click.option(
    "--skip-deps", is_flag=True, help="Skip venv creation and dependency installation (CI mode)"
)
@click.option(
    "--runtime-root",
    type=click.Path(),
    default=None,
    help="Override project_root in the rendered config, for a build whose output "
    "runs somewhere other than where it was made (e.g. --runtime-root /app/als-assistant)",
)
@click.option(
    "--dev",
    is_flag=True,
    help="Render a dev build: bake the local osprey checkout into the service "
    "images instead of the published release.",
)
@repo_option
def build(
    stream: bool,
    skip_lifecycle: bool,
    skip_deps: bool,
    runtime_root: str | None,
    dev: bool,
    repo: Path | None,
) -> None:
    """Render this deployment repo's build/ from its profile.

    Run with no arguments, anywhere inside a deployment repo. It walks up to the
    repo's profile.yml and renders the whole OUTPUT zone from it: config.yml,
    the Claude Code artifacts, the data tree, the service templates and the
    compose files that deploy them.

    build/ is derived in full and holds nothing durable — your keys are in .env,
    the agent's memory is in var/ — so every build wipes and re-renders it. The
    render lands in build/.tmp and replaces build/ only once it has succeeded: a
    build that fails, or one you interrupt, leaves the previous build exactly as
    it was, still able to stop the stack it started.

    Renders files, never containers: rebuild while the stack is up and the
    change takes effect at the next `osprey up` or `osprey restart`.

    Examples:

    \b
      # Render this repo's build/
      $ osprey build

      # Render another repo's, without cd-ing to it
      $ osprey build --repo ~/deployments/als-assistant

      # CI: no venv, no lifecycle hooks
      $ osprey build --skip-lifecycle --skip-deps

    \b
      # Dev build: images run this checkout, not the published release
      $ osprey build --dev
    """
    from .main import lifecycle_reporter
    from .summary_card import owns_summary_card, print_summary_card

    # Both decided here rather than in `_build_repo`, because a chained build —
    # `init --up`, `up --build` — reaches that function with the chaining verb's
    # reporter already installed, and this is the entry that has to leave it
    # alone. `lifecycle_reporter` makes that decision for the reporter and
    # `owns_summary_card` the same one for the card, which the chaining verb
    # prints at the end of the whole run instead.
    owns_card = owns_summary_card()
    with lifecycle_reporter():
        _build_repo(
            repo,
            stream=stream,
            skip_lifecycle=skip_lifecycle,
            skip_deps=skip_deps,
            runtime_root=runtime_root,
            dev=dev,
        )
        if owns_card:
            # Resolved again rather than returned: `_build_repo` derives the
            # same root from the same argument, and a build that got here
            # resolved it successfully.
            print_summary_card(find_repo_root(repo), "built")
