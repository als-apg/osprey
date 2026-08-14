"""Factory reset for a deployment repo — the destructive end of the lifecycle.

``osprey reset`` returns a deployment to the state a fresh ``osprey init``
leaves behind, without touching the source zone: the stack is stopped, the
containers and volumes carrying this checkout's identity are removed along with
the images this deployment built, the agent's memory is destroyed, the tokens
``osprey up`` minted are stripped out of ``.env``, and ``build/`` is deleted.
(Images are scoped differently, and the boundary below says how.)

**What survives is kept deliberately, and the printed plan says so out loud.**
``var/audit/`` — the safety audit log, the record of what the agent was asked to
do and what it did — survives a reset, because the reason to keep an audit trail
is strongest exactly when someone is wiping the thing it describes.
``--purge-audit`` destroys it under the same typed gate. And ``.env`` keeps
every value that was not minted by a deploy: the provider keys an operator
supplied are theirs, not this deployment's, and a reset that silently took them
would make a re-deploy impossible for reasons the operator could not see. The
web tier's credential files (:data:`WEB_CREDENTIAL_FILES`) survive on that same
principle — they are write-once, so what is in them is what an operator or one
earlier deploy put there — and they are disclosed for the opposite reason: a
reset that quietly left every account able to log in is as much a surprise as
one that quietly locked them all out.

SCOPING BOUNDARY — the safety core of this module
=================================================

``COMPOSE_PROJECT_NAME`` is derived from the repo's *directory name*, so two
checkouts of one deployment on a single host — a clone beside the original, a
git worktree, a colleague's copy under another home directory — share a project
name, a volume namespace, and an image tag. A destructive verb scoped by project
name alone would reach into the other checkout's data.

So nothing here is removed on a project-name match. A container or volume is
removed only when it carries
:data:`~osprey.deployment.compose_generator.REPO_ID_LABEL` set to *this*
checkout's identity (:func:`~osprey.deployment.compose_generator.repo_identity`,
the single spelling of that rule, shared with the render that baked the label
in). Both conditions, never one:

* **Project name AND matching identity** — removed.
* **Project name, different identity** — this resource was created from a
  different repo PATH, which is all the hash proves and all the refusal claims.
  Usually that means another checkout; it equally means a directory the operator
  renamed or moved, and the message says both. Reset REFUSES outright
  (:class:`ForeignCheckoutError`), before the plan is confirmed and before
  anything at all is removed, naming that path *when a label records one*.
  Not skipped-with-a-warning: a same-name foreign resource means the operator
  may well be standing in the wrong directory, and continuing would remove
  *this* repo's data on that assumption.
* **Project name, no identity label at all** — NOT removed, and listed as such.
  Labels are applied at CREATE time, so a stack started before OSPREY labelled
  its resources carries none, and nothing can prove whose it is. The remedy
  differs by kind and both are stated where they are printed: a container gets
  its label back the moment something recreates it, but *a named volume takes
  labels only when it is first created and is never relabelled by a later
  deploy*, so an unlabelled volume can only ever be removed by hand.

Images are the one class this scoping cannot cover, and the plan says why.
``<project>:local`` and the persona tags are host-global and derived from the
project name alone: two same-named checkouts do not have one image each, they
share one image. There is no identity label to check because there is no
per-checkout image to put it on. Each candidate tag is instead individually
``image inspect``-verified against its own ``com.osprey.project`` label — the
same check ``nuke`` uses — so a same-named tag belonging to something that is
not an OSPREY deployment is never touched. The residual exposure is bounded and
one-directional: reset reaches image removal only when the identity scan found
no foreign resource under this project name, and what a shared tag's removal
costs another checkout is a rebuild, never data.

EXECUTION SHAPE
===============

Discovery is read-only and happens first, so the plan an operator confirms is
built from what the runtime actually has rather than from what the config
predicts. The plan object that is printed is the same object
:func:`execute_reset` iterates — every removal argv names an entry of it, and it
issues no argv for anything else.

Nothing is re-discovered after the confirmation. Stated precisely, because one
step necessarily re-reads its input: the ``.env`` rewrite has to open the file
again in order to write it, and it is handed the planned key set so that the
re-read decides only WHERE those keys sit, never WHICH keys go. A value minted
between the plan and the confirmation therefore survives — it was never shown,
so it was never agreed to. The rule the whole verb holds to is that the removal
set is fixed at plan time and can only ever shrink afterwards (a resource that
turns out to be already gone), never widen.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from osprey.deployment.compose_generator import (
    COMPOSE_ENV_FILENAME,
    REPO_ID_LABEL,
    repo_identity,
    resolve_project_name,
)
from osprey.deployment.container_lifecycle import as_built_config_path, down_deployment
from osprey.deployment.runtime_helper import get_runtime_command, runtime_env
from osprey.deployment.staleness import BUILD_DIRNAME
from osprey.deployment.web_terminals.auth_credentials import AUTH_ENV_FILENAME
from osprey.deployment.web_terminals.lifecycle import confirm_destroy
from osprey.utils.dotenv import parse_dotenv_text
from osprey.utils.logger import get_logger
from osprey.utils.workspace import STATE_DIR_NAME

logger = get_logger("deployment.reset")

#: Compose's own owning-project label. The *first* half of the scoping rule and
#: the one that makes discovery a filtered listing rather than a name scan:
#: compose stamps it on every container and volume it creates, pinned by
#: :func:`osprey.deployment.runtime_helper.runtime_env`'s
#: ``COMPOSE_PROJECT_NAME``.
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"

#: The label a locally-built OSPREY image carries. Images have no per-checkout
#: identity to label (see the module docstring), so this is what a candidate tag
#: is verified against before removal — exactly as ``nuke`` verifies it.
OSPREY_PROJECT_LABEL = "com.osprey.project"

#: The web tier's two credential files, as ``(filename, what it holds, how it is
#: refreshed)``. Both are write-once — a deploy creates each when it is absent
#: and never rewrites an existing one — and reset removes neither, which is why
#: they are disclosed (:meth:`ResetPlan._kept_lines`) rather than left for an
#: operator to discover after wiping the deployment they belong to.
#:
#: The refresh clauses are per-file because the two do NOT behave alike: the
#: same removal that gets one re-derived costs every user their password in the
#: other, and stops a registry-mode deploy outright. The ``.env.production``
#: spelling is a literal because the module that writes it
#: (:mod:`osprey.deployment.web_terminals.env_production`) has none to import;
#: the disclosure is gated on the file being there, so a spelling that ever
#: drifted apart would drop the line rather than promise something untrue.
WEB_CREDENTIAL_FILES: tuple[tuple[str, str, str], ...] = (
    (
        ".env.production",
        "the runtime secrets every web-terminal container reads",
        "Remove it and a local-mode deploy re-derives one from .env; a registry-mode "
        "deploy expects it to be there already.",
    ),
    (
        AUTH_ENV_FILENAME,
        "the web terminals' password hashes and cookie-signing secrets",
        "Remove it and the next deploy mints a NEW password for every user; "
        "`osprey users decommission` is what drops one departed user's entries.",
    ),
)

#: Where a foreign checkout's PATH is read from, most authoritative first.
#: ``repo_id`` is a one-way hash, so the identity that proves a resource is
#: someone else's cannot itself produce the directory to go look in — these can.
#:
#: ``com.docker.compose.project.working_dir`` is compose's own record of the
#: ``--project-directory`` it was pinned to, which
#: :func:`~osprey.deployment.compose_generator.compose_base_cmd` sets to the repo
#: root on every invocation. ``osprey.project.root`` is OSPREY's render-time
#: label and is the fallback rather than the primary because a ``--runtime-root``
#: build records the path the project will run at *inside a container*, which is
#: not a host path at all.
#:
#: When neither is present the refusal says the path is not recoverable. It
#: never guesses one.
PATH_EVIDENCE_LABELS: tuple[str, ...] = (
    "com.docker.compose.project.working_dir",
    "osprey.project.root",
)

#: Header comments of the ``.env`` blocks a deploy writes, carried at EVERY
#: spelling that can be sitting in a ``.env`` on disk. The text is what
#: identifies the block, so it is a data format, not a message: renaming it in
#: the writer without adding the new spelling here orphans every block written
#: under the other one, leaving stale tokens in place while reset reports them
#: stripped. Entries are only ever added, never rewritten.
#:
#: ``tests/deployment/test_reset_scoping.py`` reads the writer's call sites out
#: of :mod:`osprey.deployment.container_lifecycle` and fails when a block is
#: written under a banner this tuple does not carry.
MINTED_ENV_BANNERS: tuple[str, ...] = (
    "Auto-generated service auth tokens (osprey deploy up)",
    "Auto-configured bluesky bridge scan devices (osprey deploy up)",
    "Auto-generated bluesky RE manager control-socket keypair (osprey deploy up)",
    "Credentials adopted from pre-existing data volumes (osprey --reuse-stores)",
)


def confirmation_token(repo_root: Path) -> str:
    """The exact string an operator must type to confirm a reset.

    The repo's own directory name — specific to what is being destroyed in the
    way a generic "yes" is not, and specific to the hazard this verb actually
    has, which is being run in the wrong checkout. Typing the name is the
    operator reading it off the plan and agreeing that it is the deployment they
    meant.
    """
    return Path(repo_root).resolve().name


class ResetOutcome(Enum):
    """How a reset ended — five states a boolean could not tell apart.

    The exit status is the machine-readable form of this verb's core contract,
    "what is printed is what happens", and :data:`EXIT_CODES` is where the two
    are tied together. ``osprey reset && osprey build && osprey up`` is the
    documented start-over recipe, so an ending that exits 0 is an assertion to
    the next command in the chain that the deployment is now clean.

    Where an ending is ambiguous, it does NOT default toward success. A caller
    that wants to proceed on a partial reset can test for its code explicitly;
    a caller that assumes success cannot un-proceed.

    * ``0`` — the reset did what it said, or was never asked to do anything.
    * ``1`` — the operator declined. Nothing was touched.
    * ``2`` — **never emitted here.** Click owns it: ``UsageError.exit_code``
      is 2, so a mistyped flag already exits 2. Reusing it for a partly-reset
      deployment would make "your command line was wrong" and "your data is
      half-gone" the same signal to a script — the precise ambiguity this
      mapping exists to remove. The gap is deliberate; do not close it.
    * ``3`` — the reset ran, but something on the plan survived.
    """

    #: Everything on the plan was removed.
    COMPLETED = "completed"
    #: The reset ran, but at least one planned removal did not take. Exits 3:
    #: letting a chained rebuild or handoff proceed here would be the
    #: false-completion defect moved out of the prose and into ``$?``. stdout
    #: already names each survivor, so a caller that wants to continue anyway
    #: can test for 3 deliberately. Not 2 — see the class docstring; click
    #: already spends 2 on usage errors.
    COMPLETED_WITH_FAILURES = "completed_with_failures"
    #: ``--dry-run``. Nothing was stopped, removed, or written.
    DRY_RUN = "dry_run"
    #: There was nothing to remove. Not an error, and no gate was shown.
    NOTHING_TO_DO = "nothing_to_do"
    #: The operator declined, typed a non-matching token, or interrupted the
    #: prompt. Nothing was touched.
    DECLINED = "declined"


#: The one mapping from outcome to process exit status. A table rather than a
#: chain of ``if`` branches in the CLI, so that adding an outcome without
#: deciding its exit code is a ``KeyError`` at the call site instead of a silent
#: fall-through to 0 — which for this verb is the direction that lies.
#:
#: 2 is absent on purpose (see :class:`ResetOutcome`): click spends it on usage
#: errors, so this verb never emits it.
EXIT_CODES: Mapping[ResetOutcome, int] = {
    ResetOutcome.COMPLETED: 0,
    ResetOutcome.DRY_RUN: 0,
    ResetOutcome.NOTHING_TO_DO: 0,
    ResetOutcome.DECLINED: 1,
    ResetOutcome.COMPLETED_WITH_FAILURES: 3,
}

#: The status a partly-reset deployment reports, reachable from two paths: a
#: removal that would not take (:data:`ResetOutcome.COMPLETED_WITH_FAILURES`)
#: and an interrupt landing mid-teardown. Both leave the same state on disk, so
#: both have to leave the same signal in ``$?``.
PARTIAL_RESET_EXIT_CODE = EXIT_CODES[ResetOutcome.COMPLETED_WITH_FAILURES]


class ForeignCheckoutError(RuntimeError):
    """Same-named resources were created from a different repo path — nothing was touched.

    Carries the offending resources so a caller can render them its own way. The
    string form is already operator-facing and names each resource, the identity
    it carries, and the path that identity was recorded against *when one of the
    path-evidence labels supplies it* — a volume with no such label leaves the
    path genuinely unknown, and the message says that rather than filling it in.
    """

    def __init__(self, message: str, *, resources: Sequence[Resource]) -> None:
        super().__init__(message)
        #: The foreign resources, in discovery order.
        self.resources = list(resources)


# ---------------------------------------------------------------------------
# What the runtime has
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Resource:
    """One container, volume or image, with the labels that decide its fate.

    ``name`` is the exact removal target and is always a name or an id the
    runtime resolves on its own — never a glob, a label selector, or ``-a``.
    """

    kind: str
    name: str
    labels: Mapping[str, str] = field(default_factory=dict)

    @property
    def repo_id(self) -> str | None:
        """This resource's checkout identity, or ``None`` when it carries none."""
        value = self.labels.get(REPO_ID_LABEL)
        return value if isinstance(value, str) and value else None

    @property
    def recorded_path(self) -> str | None:
        """The repo path recorded on this resource, or ``None`` if it has none.

        See :data:`PATH_EVIDENCE_LABELS`. ``None`` is a real answer that callers
        must render as such: the path of a foreign checkout is not derivable
        from its identity hash, and inventing one would be worse than admitting
        it is unknown.
        """
        for label in PATH_EVIDENCE_LABELS:
            value = self.labels.get(label)
            if isinstance(value, str) and value:
                return value
        return None

    def describe_origin(self) -> str:
        """Operator-facing "whose is this?", claiming only what the labels say.

        Two branches, and each says exactly what its own evidence supports. With
        a path label the identity is quoted alongside the path it was *recorded*
        at — "recorded at", not "lives at", because a label is a record of where
        a deploy ran, which is not a promise about what is on disk now (the
        refusal adds that separately, from the filesystem). Without one, the
        identity is all there is, and the line says so rather than leaving a
        reader to assume the path was simply omitted.
        """
        identity = self.repo_id or "no identity label"
        path = self.recorded_path
        if path is None:
            return (
                f"{identity} — no path is recorded on this resource, and a repo path "
                "cannot be derived back out of the identity hash"
            )
        return f"{identity}, recorded at {path}"


@dataclass(frozen=True)
class Partition:
    """A discovered resource set, split by the two-condition scoping rule."""

    ours: list[Resource] = field(default_factory=list)
    foreign: list[Resource] = field(default_factory=list)
    unidentified: list[Resource] = field(default_factory=list)


def partition_by_identity(resources: Iterable[Resource], identity: str) -> Partition:
    """Split *resources* into ours / another checkout's / unprovable.

    The single implementation of the AND rule, so containers, volumes and every
    later caller cannot come to different conclusions from the same labels.

    Args:
        resources: Already project-name-filtered resources.
        identity: This checkout's :func:`~osprey.deployment.compose_generator.repo_identity`.

    Returns:
        A :class:`Partition`. Membership is exclusive and total: every input
        lands in exactly one list.
    """
    partition = Partition()
    for resource in resources:
        repo_id = resource.repo_id
        if repo_id is None:
            partition.unidentified.append(resource)
        elif repo_id == identity:
            partition.ours.append(resource)
        else:
            partition.foreign.append(resource)
    return partition


class RuntimeProbe:
    """Every conversation this module has with the container runtime.

    One seam, for one reason: it is the only place a destructive argv is built,
    so the whole destructive surface of ``osprey reset`` can be read in one
    screen and exercised in tests without a runtime. ``run`` is injected for
    exactly that — the tests pass a recorder, and no test in this feature ever
    reaches a real daemon.

    Every listing is label-filtered and read-only; every removal names exactly
    one resource. There is no ``prune``, no ``-a``/``--all``, no wildcard, and
    no ``down -v`` anywhere in this class.
    """

    def __init__(
        self,
        runtime: str,
        *,
        env: Mapping[str, str] | None = None,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        #: Runtime binary — ``docker`` or ``podman``, never the compose argv.
        self.runtime = runtime
        #: Resources a removal was attempted on and did not remove, as
        #: ``(subject, reason)``. Read back by the caller so a reset that left
        #: something behind cannot report itself complete.
        self.failures: list[tuple[str, str]] = []
        self._env = dict(env) if env is not None else None
        self._run = run

    # -- reads ------------------------------------------------------------

    def _capture(self, args: Sequence[str]) -> subprocess.CompletedProcess:
        logger.debug("Running command:\n    %s", " ".join([self.runtime, *args]))
        return self._run(
            [self.runtime, *args], capture_output=True, text=True, env=self._env, check=False
        )

    def _labels_of(self, kind: str, ref: str) -> Mapping[str, str]:
        """The label map of one container/volume/image, ``{}`` when unreadable.

        The label map is parsed from JSON in Python rather than indexed inside
        the Go template: ``index`` on a missing or nil label map is fragile
        across docker and podman versions, and this map decides what gets
        destroyed.
        """
        field_by_kind = {
            "container": "{{json .Config.Labels}}",
            "volume": "{{json .Labels}}",
            "image": "{{json .Config.Labels}}",
        }
        result = self._capture([kind, "inspect", ref, "--format", field_by_kind[kind]])
        if result.returncode != 0:
            return {}
        try:
            labels = json.loads((result.stdout or "").strip() or "null")
        except json.JSONDecodeError:
            return {}
        if not isinstance(labels, dict):
            return {}
        return {k: v for k, v in labels.items() if isinstance(v, str)}

    def containers_for_project(self, project: str) -> list[Resource]:
        """Read-only: this compose project's containers, with their labels.

        ``ps -a --filter label=com.docker.compose.project=<project>``. The
        ``-a`` includes stopped containers on purpose — a stopped container
        still holds its name and its published-port reservation, so a reset
        that skipped it would leave the next deploy colliding with it.
        """
        listing = self._capture(
            [
                "ps",
                "-a",
                "--filter",
                f"label={COMPOSE_PROJECT_LABEL}={project}",
                "--format",
                "{{.Names}}",
            ]
        )
        if listing.returncode != 0:
            raise RuntimeError(
                f"Could not list this project's containers: `{self.runtime} ps` exited "
                f"{listing.returncode}. {(listing.stderr or '').strip()}"
            )
        names = [line.strip() for line in (listing.stdout or "").splitlines() if line.strip()]
        return [Resource("container", name, self._labels_of("container", name)) for name in names]

    def volumes_for_project(self, project: str) -> list[Resource]:
        """Read-only: this compose project's named volumes, with their labels."""
        listing = self._capture(
            [
                "volume",
                "ls",
                "--filter",
                f"label={COMPOSE_PROJECT_LABEL}={project}",
                "--format",
                "{{.Name}}",
            ]
        )
        if listing.returncode != 0:
            raise RuntimeError(
                f"Could not list this project's volumes: `{self.runtime} volume ls` exited "
                f"{listing.returncode}. {(listing.stderr or '').strip()}"
            )
        names = [line.strip() for line in (listing.stdout or "").splitlines() if line.strip()]
        return [Resource("volume", name, self._labels_of("volume", name)) for name in names]

    def env_of_container(self, name: str) -> Mapping[str, str] | None:
        """Read-only: one container's environment, or ``None`` if unreadable.

        The deploy path's stale-volume preflight uses this and nothing else
        does: a store container's environment is the only host-side record of
        the credential its data volume was initialized with, so while the
        container survives the volume can still be reopened, and once it has
        been recreated the volume is orphaned.

        ``None`` means "no such container" — which for that preflight is not an
        error but the answer itself, and the reason it must run before anything
        recreates one.

        Scoped by name rather than by label filter, unlike every listing above,
        because the caller wants one exact container: the compose template
        derives ``container_name`` from the project name, so the name is already
        project-qualified. Nothing here removes anything.
        """
        result = self._capture(
            ["container", "inspect", name, "--format", "{{range .Config.Env}}{{println .}}{{end}}"]
        )
        if result.returncode != 0:
            return None
        env: dict[str, str] = {}
        for line in (result.stdout or "").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip():
                env[key.strip()] = value
        return env

    def image_if_ours(self, tag: str, project: str) -> Resource | None:
        """Read-only: *tag* as a removal candidate, or ``None`` if it is not one.

        ``None`` covers both "no such tag on this host" (nothing to remove,
        never an error) and "the tag exists but its ``com.osprey.project`` label
        does not say this deployment built it" — a same-named image belonging to
        something else, which must survive.
        """
        labels = self._labels_of("image", tag)
        if not labels:
            return None
        if labels.get(OSPREY_PROJECT_LABEL) != project:
            return None
        return Resource("image", tag, labels)

    # -- writes -----------------------------------------------------------

    def remove_container(self, name: str) -> None:
        """Remove one exact-named container; absence is success, not an error.

        Absence is the normal case rather than the exception: the ``down`` in
        step 1 has usually already removed it, and this sweep exists for the
        ones a ``down`` could not reach.
        """
        self._remove(["rm", "-f", name], f"container {name!r}")

    def remove_volume(self, name: str) -> None:
        """Remove one exact-named volume. Never a glob, a filter, or ``prune``."""
        self._remove(["volume", "rm", name], f"volume {name!r}")

    def remove_image(self, tag: str) -> None:
        """Remove one exact-named image tag, already label-verified as ours."""
        self._remove(["image", "rm", tag], f"image {tag!r}")

    def _remove(self, args: Sequence[str], subject: str) -> None:
        result = self._capture(args)
        if result.returncode == 0:
            logger.debug("Removed %s", subject)
            return
        stderr = (result.stderr or "").strip()
        if _is_absent(stderr):
            logger.debug("%s was already gone", subject)
            return
        # Recorded, not raised: one volume that will not go is a reason to tell
        # the operator exactly which one, not a reason to abandon the rest of
        # the reset half-done and leave them with two problems. Recording it is
        # what stops the run reporting itself complete — see
        # :func:`reset_deployment`'s closing report.
        reason = stderr or f"exit {result.returncode}"
        self.failures.append((subject, reason))
        logger.warning("Could not remove %s: %s", subject, reason)


def _is_absent(stderr: str) -> bool:
    """Whether a failed removal failed only because there was nothing there.

    Matched on the runtime's own wording because neither docker nor podman
    distinguishes "already gone" from "would not go" by exit code. Kept
    deliberately narrow: anything else is surfaced to the operator.
    """
    lowered = stderr.lower()
    return "no such" in lowered or "not found" in lowered


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvBlock:
    """One minted ``.env`` block reset will strip, named but never quoted.

    ``keys`` are variable names only. Their values are secrets and are never
    printed, logged, or carried anywhere — the same "log which keys, never what
    they are" rule the minting side follows.
    """

    banner: str
    keys: tuple[str, ...]


@dataclass
class ResetPlan:
    """Exactly what a reset will remove, and exactly what it will keep.

    This object is both the printed plan and the execution program: rendering it
    and running it read the same lists, so the two cannot describe different
    removal sets. Anything reset does that is not on it is a defect, and
    ``tests/deployment/test_reset_scoping.py`` asserts the equality in both
    directions.
    """

    repo_root: Path
    project: str
    identity: str
    project_name_source: str
    containers: list[Resource] = field(default_factory=list)
    volumes: list[Resource] = field(default_factory=list)
    images: list[Resource] = field(default_factory=list)
    unidentified: list[Resource] = field(default_factory=list)
    paths: list[Path] = field(default_factory=list)
    #: Directories this deployment owns that reset will NOT remove because they
    #: resolve outside the repo root. Disclosed, never deleted.
    external_paths: list[Path] = field(default_factory=list)
    #: The resolved agent-data root, wherever ``agent_data.base_dir`` puts it.
    #: Held even when it is not removable, so the plan can name the real
    #: directory rather than the default it is not using.
    agent_data_dir: Path | None = None
    env_blocks: list[EnvBlock] = field(default_factory=list)
    env_kept_keys: tuple[str, ...] = ()
    purge_audit: bool = False

    @property
    def audit_dir(self) -> Path:
        """The safety audit log — kept unless ``--purge-audit`` was passed."""
        return self.repo_root / STATE_DIR_NAME / "audit"

    @property
    def removes_anything(self) -> bool:
        """Whether this plan would change anything at all."""
        return bool(self.containers or self.volumes or self.images or self.paths or self.env_blocks)

    def confirmation_summary(self) -> str:
        """What the prompt says is about to be destroyed, counted from the plan.

        Built from the plan's own lists rather than written as a sentence,
        because the prompt is the last thing an operator reads before typing:
        naming the agent's memory on a deployment that has none, or a volume
        count that does not match the list printed above it, is how a gate stops
        being believed.
        """
        parts = []
        for count, noun in (
            (len(self.containers), "container"),
            (len(self.volumes), "volume"),
            (len(self.images), "image"),
        ):
            if count:
                parts.append(f"{count} {noun}{'' if count == 1 else 's'}")
        parts.extend(_relative_to(path, self.repo_root) for path in self.paths)
        if self.env_blocks:
            minted = sum(len(block.keys) for block in self.env_blocks)
            plural = "" if minted == 1 else "s"
            parts.append(f"{minted} minted value{plural} in {COMPOSE_ENV_FILENAME}")
        return ", ".join(parts) if parts else "nothing"

    def render(self) -> list[str]:
        """The plan an operator reads before typing the confirmation.

        Every removal this reset performs appears here, and nothing appears here
        that it does not perform. The kept section is not a courtesy: ``reset``
        is the verb whose whole risk is destroying something the operator
        assumed was safe, so what survives is stated as explicitly as what does
        not.
        """
        lines = [
            f"osprey reset — {self.repo_root}",
            f"  project name   {self.project}  ({self.project_name_source})",
            f"  checkout id    {REPO_ID_LABEL}={self.identity}",
            "",
            "WILL BE REMOVED",
        ]

        lines.append(f"  containers ({len(self.containers)})")
        for resource in self.containers:
            lines.append(f"    {resource.name}")
        lines.append(f"  volumes ({len(self.volumes)}) — the data in them is not recoverable")
        for resource in self.volumes:
            lines.append(f"    {resource.name}")
        lines.append(f"  images ({len(self.images)}) — rebuilt by the next `osprey up`")
        for resource in self.images:
            lines.append(f"    {resource.name}  ({OSPREY_PROJECT_LABEL}={self.project} verified)")

        lines.append("  files")
        for path in self.paths:
            lines.append(f"    {_relative_to(path, self.repo_root)}{self._path_note(path)}")
        for block in self.env_blocks:
            keys = ", ".join(block.keys)
            lines.append(f"    {COMPOSE_ENV_FILENAME}  minted block '{block.banner}' — {keys}")

        lines.extend(
            [
                "",
                "WILL BE KEPT",
                *self._kept_lines(),
            ]
        )

        if self.unidentified or self.external_paths:
            lines.extend(["", "NOT REMOVED"])
            lines.extend(self._external_path_lines())
            lines.extend(self._unidentified_lines())

        lines.extend(
            [
                "",
                "Networks are removed by the `down` in step 1, which is compose's own "
                "teardown of the project's network.",
                "  Nothing labels a network with a checkout identity, so reset never "
                "removes one on its own. If the",
                f"  `down` had no compose files to work from (no {BUILD_DIRNAME}/), the "
                "network survives and the next `osprey up` reuses it.",
            ]
        )
        return lines

    def _kept_lines(self) -> list[str]:
        kept = []
        if self.purge_audit:
            kept.append(
                f"    (nothing under {STATE_DIR_NAME}/ — --purge-audit was passed, so the "
                "audit log is in the removal list above)"
            )
        else:
            kept.append(
                f"    {STATE_DIR_NAME}/audit/  the safety audit log — what the agent was "
                "asked to do and what it did."
            )
            kept.append("      `osprey reset --purge-audit` destroys this too, under this gate.")
        env_note = f"{len(self.env_kept_keys)} entr{'y' if len(self.env_kept_keys) == 1 else 'ies'}"
        kept.append(
            f"    {COMPOSE_ENV_FILENAME}  everything that was not minted by a deploy, "
            f"including your provider keys ({env_note})."
        )
        kept.append(
            "    profile.yml, data/, personas/, triggers.yml, scripts/ — the tracked source "
            "zone, and .git/ with it."
        )
        for filename, holds, refresh in WEB_CREDENTIAL_FILES:
            if (self.repo_root / filename).is_file():
                kept.append(
                    f"    {filename}  {holds} — operator-owned and written once, so a "
                    "reset leaves it exactly as it is."
                )
                kept.append(f"      {refresh}")
        kept.append(
            f"    {STATE_DIR_NAME}/  everything outside the entries listed above, including "
            "any --archive tarballs."
        )
        curve_dir = self.repo_root / "data" / "bluesky_curve"
        if curve_dir.is_dir():
            kept.append(
                "    data/bluesky_curve/  the document-plane CURVE certificates, which live "
                "in the source zone. They are"
            )
            kept.append(
                "      regenerated only when absent, so a reset leaves this deployment's "
                "existing key material in place."
            )
        return kept

    def _path_note(self, path: Path) -> str:
        """The one-line "what is this" for a directory in the removal list.

        Matched by IDENTITY against the directories this plan actually resolved,
        not by suffix against the default spellings. A deployment whose
        ``agent_data.base_dir`` is ``var/memory`` still gets the note that says
        this is the agent's memory — which is the line that tells an operator
        what they are about to lose.
        """
        if self.agent_data_dir is not None and path == self.agent_data_dir:
            return "  the agent's memory, sessions and artifacts — not recoverable"
        if path == self.repo_root / BUILD_DIRNAME:
            return "  the render zone — regenerated in full by `osprey build`"
        if path == self.audit_dir:
            return "  the safety audit log — not recoverable"
        return ""

    def _external_path_lines(self) -> list[str]:
        """Directories reset leaves alone because removing them is not its to do.

        Printed rather than passed over in silence, which is the whole point: an
        operator resetting a deployment whose ``agent_data.base_dir`` points
        somewhere reset will not go needs to know the agent's memory is still
        there, and needs to know it from the plan rather than from discovering
        it later.

        Two reasons, kept apart because they are not the same reason. A root
        that sits somewhere else on the host is refused at the repo boundary. A
        root that IS the repo root is refused for the opposite kind of reason —
        it is not outside anything, it is the whole deployment, and deleting it
        would take the tracked source zone and ``.git`` along with the memory.
        Printing the boundary rationale for that case would state something
        false about the very path it is printing.
        """
        lines: list[str] = []
        outside = [path for path in self.external_paths if path != self.repo_root]

        if self.repo_root in self.external_paths:
            lines.append(
                f"    {self.repo_root}  the agent-data root, which resolves to the repo root "
                "itself — removing it"
            )
            lines.append(
                "      would take the source zone and .git with it. agent_data.base_dir points "
                "at '.' (or resolves"
            )
            lines.append(
                "      there). Reset destroys directories, and this one is the deployment, so "
                "it is left entirely alone."
            )

        for path in outside:
            lines.append(f"    {path}  the agent-data root, which is OUTSIDE this repo")
        if outside:
            lines.append(
                "      agent_data.base_dir puts it there. Reset removes directories outright, "
                "and its authority to do"
            )
            lines.append(
                "      that comes from their being inside the repo you named — so it stops at "
                "the repo boundary, whatever"
            )
            lines.append(
                "      the config says. The agent's memory is still there; remove it by hand "
                "if you meant to lose it."
            )
        return lines

    def _unidentified_lines(self) -> list[str]:
        lines = []
        for resource in self.unidentified:
            lines.append(
                f"    {resource.kind} {resource.name}  carries no {REPO_ID_LABEL} label, so it "
                "cannot be proved to be this repo's"
            )
        if any(r.kind == "container" for r in self.unidentified):
            lines.append(
                "      Containers are labelled when they are CREATED, so one started before "
                "this OSPREY version has no"
            )
            lines.append(
                "      identity. `osprey up` recreates it with the label, after which reset "
                "can see whose it is."
            )
        if any(r.kind == "volume" for r in self.unidentified):
            lines.append(
                "      A named volume takes labels only when it is FIRST created and is "
                "never relabelled by a later"
            )
            lines.append(
                "      deploy, so no OSPREY command can ever make this one provable. Inspect "
                "it, then remove it by hand"
            )
            lines.append(
                "      if it is yours — reset will not remove it, on this run or any later one."
            )
        return lines


def _relative_to(path: Path, repo_root: Path) -> str:
    """*path* spelled against the repo root when it is inside it, else absolute."""
    try:
        return f"{path.relative_to(repo_root)}/"
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def load_as_built_config(repo_root: Path) -> dict | None:
    """The rendered config, or ``None`` when there is no readable build.

    Loaded ONCE per plan and threaded to everything that needs it, rather than
    re-read per question. Two things are derived from it — the project name and
    the agent-data root — and they have to come from the same document: a plan
    that named one build's project while wiping another build's directories
    would be describing a deployment that never existed.

    An unreadable or unparseable config degrades to ``None`` rather than
    raising. A reset is the verb an operator reaches for when a deployment is
    already broken, so a corrupt build must not be the thing that blocks
    cleaning it up — it costs the config-derived detail, which the plan then
    reports as derived rather than read.
    """
    config_path = as_built_config_path(repo_root)
    if not config_path.is_file():
        return None
    try:
        from osprey.utils.config import load_project_config

        return load_project_config(str(config_path), wrap_errors=True) or None
    except Exception:  # noqa: BLE001 - an unreadable build must not block a reset
        return None


def resolve_reset_project_name(repo_root: Path, config: dict | None) -> tuple[str, str]:
    """This repo's compose project name, and where it was read from.

    Read from the rendered config when there is one, because that is the value
    the running containers were actually labelled with. With no readable config
    — a repo that was never built, or one whose build was already wiped — it
    falls back to the repo directory name through the same
    :func:`~osprey.deployment.compose_generator.resolve_project_name` rule,
    which is exactly what compose itself would derive.

    The source is returned alongside the name, not kept internal: a reset
    planned against a *guessed* project name is a different thing to confirm
    than one planned against the build's own, and the operator is the one who
    can tell whether the guess is right.

    Returns:
        ``(project_name, human-readable source)``.
    """
    if config:
        return resolve_project_name(config), f"from {BUILD_DIRNAME}/config.yml"
    return (
        resolve_project_name({"project_root": str(repo_root)}),
        "derived from this directory's name — no build/config.yml to read it from",
    )


def resolve_agent_data_target(repo_root: Path, config: dict | None) -> tuple[Path, bool]:
    """Where this deployment's agent data actually lives, and whether reset may touch it.

    ``agent_data.base_dir`` is a real config key that every shipped template
    emits, and a deployment is free to move its agent data off the default
    ``var/agent_data``. Reset reads it through
    :func:`~osprey.utils.workspace.agent_data_base_dir` and anchors it with
    :func:`~osprey.utils.workspace.anchored_path` — the same pair every other
    consumer uses — because the alternative is the failure this verb can least
    afford: hardcoding the default wipes nothing on a relocated deployment while
    the plan says the agent's memory is gone, and the operator has no way to see
    the difference.

    The second return value is a CONTAINMENT verdict, and it is a hard boundary
    rather than a warning. ``base_dir`` may be absolute or ``~``-relative, so it
    can resolve anywhere on the host — a shared scratch filesystem, a home
    directory, ``/``. Reset deletes directories outright, and its whole authority
    to do that comes from those directories being inside the repo the operator
    named. Outside it, that authority does not exist, whatever the config says,
    so an external root is reported and never removed.

    Returns:
        ``(resolved_path, is_inside_repo)``.
    """
    from osprey.utils.workspace import agent_data_base_dir, anchored_path

    repo_root = Path(repo_root).resolve()
    target = anchored_path(agent_data_base_dir(config), repo_root).resolve()
    return target, target.is_relative_to(repo_root) and target != repo_root


def _candidate_image_tags(repo_root: Path, project: str) -> list[str]:
    """Every locally-built tag this deployment could own, deduplicated.

    ``<project>:local`` is the project image ``osprey up`` builds for the
    dispatch worker. The rest are the web-terminal persona images and the auth
    sidecar, resolved from the rendered config exactly as ``nuke`` resolves them
    — leniently, so a stale persona reference degrades to "not a candidate"
    rather than blocking the reset. A deployment with no build, or no web
    terminals, contributes only the project tag.
    """
    tags = [f"{project}:local"]

    config_path = as_built_config_path(repo_root)
    if not config_path.is_file():
        return tags

    try:
        from osprey.deployment.web_terminals.personas import (
            as_dict,
            effective_image_source,
            resolve_personas,
        )
        from osprey.utils.config import ConfigBuilder

        config = ConfigBuilder(str(config_path)).raw_config
        web_terminals = as_dict(as_dict(config.get("modules")).get("web_terminals"))
        if not web_terminals:
            return tags
        personas = resolve_personas(
            web_terminals,
            as_dict(config.get("registry")),
            as_dict(config.get("facility")).get("prefix") or "",
            strict=False,
        )
        for entry in personas:
            image = entry.get("image")
            if isinstance(image, str) and image.endswith(":local") and image not in tags:
                tags.append(image)

        from osprey.deployment.web_terminals.provision import auth_sidecar_local_tag
        from osprey.deployment.web_terminals.render import _auth_tls_context

        auth_ctx = _auth_tls_context(web_terminals)
        if (
            auth_ctx["auth_method"] != "none"
            and effective_image_source(web_terminals) == "local"
            and not auth_ctx["auth_image"]
        ):
            auth_tag = auth_sidecar_local_tag(config)
            if auth_tag not in tags:
                tags.append(auth_tag)
    except Exception as exc:  # noqa: BLE001 - a bad roster must not block a reset
        logger.warning(
            "Could not resolve this deployment's web-terminal image tags from %s (%s). "
            "Reset will remove the project image only; any persona image is left in "
            "place and can be removed by hand.",
            config_path,
            exc,
        )
    return tags


def _minted_env_blocks(env_path: Path) -> tuple[list[EnvBlock], tuple[str, ...]]:
    """The minted blocks in *env_path*, and the names of everything that survives.

    The kept names are read off the text that WOULD remain rather than by
    subtracting the minted names from a parse of the whole file. The two differ
    whenever an operator's own line happens to share a name with a minted one —
    subtraction drops their entry from the count even though the strip leaves it
    exactly where it is, so the plan would under-report what it is keeping.
    Deriving both halves from :func:`strip_minted_env_blocks` means the count and
    the edit cannot disagree, whatever the file looks like.
    """
    if not env_path.is_file():
        return [], ()
    text = env_path.read_text(encoding="utf-8")
    blocks = [
        EnvBlock(banner, tuple(keys)) for banner, keys in _scan_minted_blocks(text.splitlines())
    ]
    minted = {key for block in blocks for key in block.keys}
    surviving, _ = strip_minted_env_blocks(text, minted)
    return blocks, tuple(parse_dotenv_text(surviving))


def _scan_minted_blocks(lines: Sequence[str]) -> list[tuple[str, list[str]]]:
    """Find each minted block and the variable names inside it.

    A block is the header comment (``# `` plus one of
    :data:`MINTED_ENV_BANNERS`) and the ``KEY=value`` lines that follow it, up to
    the next comment, the next blank line, or the end of the file — which is
    exactly the shape ``container_lifecycle._append_env_block`` writes.

    Values are read past, never captured.
    """
    found: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        banner = stripped[1:].strip() if stripped.startswith("#") else None
        if banner not in MINTED_ENV_BANNERS:
            index += 1
            continue
        keys: list[str] = []
        index += 1
        while index < len(lines):
            entry = lines[index].strip()
            if not entry or entry.startswith("#") or "=" not in entry:
                break
            keys.append(entry.split("=", 1)[0].strip())
            index += 1
        found.append((str(banner), keys))
    return found


def strip_minted_env_blocks(text: str, planned_keys: set[str]) -> tuple[str, list[str]]:
    """Remove the *planned* minted values from ``.env`` *text*, and nothing else.

    Pure, and the only thing that decides what leaves the file. Two rules, and
    the second is the one that makes this safe to run after a confirmation:

    * Lines outside a minted block — the operator's provider keys, their
      comments, their blank lines — come out byte-identical. ``.env`` is a file
      an operator reads and edits by hand, and a reset has no business
      reformatting it.
    * Inside a minted block, a key is dropped only if it is in *planned_keys*.
      A value minted after the plan was printed survives, because the operator
      never saw it and so never agreed to it. Its banner survives with it: a
      block is only removed entirely when every key under it was planned, so a
      surviving key is never orphaned from the header that explains where it
      came from.

    Returns:
        ``(new_text, removed_keys)``.
    """
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    removed: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        banner = stripped[1:].strip() if stripped.startswith("#") else None
        if banner not in MINTED_ENV_BANNERS:
            kept.append(lines[index])
            index += 1
            continue

        header = lines[index]
        index += 1
        block_kept: list[str] = []
        block_removed: list[str] = []
        while index < len(lines):
            entry = lines[index].strip()
            if not entry or entry.startswith("#") or "=" not in entry:
                break
            key = entry.split("=", 1)[0].strip()
            if key in planned_keys:
                block_removed.append(key)
            else:
                block_kept.append(lines[index])
            index += 1

        removed.extend(block_removed)
        # The header goes only when nothing under it is left to explain.
        if block_kept or not block_removed:
            kept.append(header)
            kept.extend(block_kept)
    return "".join(kept), removed


def plan_reset(repo_root: Path, *, probe: RuntimeProbe, purge_audit: bool = False) -> ResetPlan:
    """Read the runtime and the repo, and decide what a reset would remove.

    Read-only from end to end. It is also where the refusal lives: a
    same-name resource belonging to another checkout raises
    :class:`ForeignCheckoutError` here, before a plan exists to confirm, so a
    ``--dry-run`` surfaces the refusal exactly as a real run does.

    Args:
        repo_root: The deployment repo.
        probe: The runtime seam. Its listings are label-filtered reads.
        purge_audit: Extend the wipe to ``var/audit`` (same typed gate).

    Returns:
        The plan, ready to render and to execute.

    Raises:
        ForeignCheckoutError: When a resource carrying this project's name
            carries a different checkout's identity.
    """
    # Resolved once, here, so every path this plan holds is comparable to every
    # other. `resolve_agent_data_target` resolves its answer in order to judge
    # containment, and a plan mixing a resolved agent-data root with an
    # unresolved repo root cannot spell the first relative to the second — on a
    # symlinked path (macOS /var -> /private/var) it would print an absolute
    # path where it means "inside your repo".
    repo_root = Path(repo_root).resolve()
    config = load_as_built_config(repo_root)
    project, source = resolve_reset_project_name(repo_root, config)
    identity = repo_identity(repo_root)

    containers = partition_by_identity(probe.containers_for_project(project), identity)
    volumes = partition_by_identity(probe.volumes_for_project(project), identity)

    foreign = [*containers.foreign, *volumes.foreign]
    if foreign:
        raise ForeignCheckoutError(
            _foreign_refusal(repo_root, project, identity, foreign), resources=foreign
        )

    images = [
        image
        for tag in _candidate_image_tags(repo_root, project)
        if (image := probe.image_if_ours(tag, project)) is not None
    ]

    agent_data, contained = resolve_agent_data_target(repo_root, config)

    paths = [repo_root / BUILD_DIRNAME]
    if contained:
        paths.insert(0, agent_data)
    if purge_audit:
        paths.append(repo_root / STATE_DIR_NAME / "audit")
    paths = [path for path in paths if path.exists()]

    # An agent-data root outside the repo is reported and never removed. It is
    # only worth a line when it is actually there — naming a configured path
    # that does not exist would add a scary entry for nothing.
    external = [] if contained or not agent_data.exists() else [agent_data]

    env_blocks, env_kept = _minted_env_blocks(repo_root / COMPOSE_ENV_FILENAME)

    return ResetPlan(
        repo_root=repo_root,
        project=project,
        identity=identity,
        project_name_source=source,
        containers=containers.ours,
        volumes=volumes.ours,
        images=images,
        unidentified=[*containers.unidentified, *volumes.unidentified],
        paths=paths,
        external_paths=external,
        agent_data_dir=agent_data,
        env_blocks=env_blocks,
        env_kept_keys=env_kept,
        purge_audit=purge_audit,
    )


def _foreign_refusal(
    repo_root: Path, project: str, identity: str, foreign: Sequence[Resource]
) -> str:
    """The refusal text, claiming exactly as much as the labels support.

    An operator who sees this is in one of three situations, and the message has
    to serve all of them without deciding between them: they are standing in the
    wrong directory, two checkouts of one deployment are genuinely sharing a
    host, or they renamed or moved *this* directory and are meeting their own
    resources under the old path. The recorded path is what tells them which —
    so it is quoted from the resource rather than reconstructed, and when no
    resource carries one, the message says so instead of guessing.

    What is a *label* fact and what is a *filesystem* fact are kept apart on
    purpose: the path is reported as recorded, and whether it still exists is
    added separately by :func:`_path_liveness_note`. The remedy then branches on
    that (:func:`_foreign_remedies`), because "go and reset it over there" is
    wrong advice for a directory that is no longer on the host.
    """
    lines = [
        f"Refusing to reset {repo_root}: {len(foreign)} container(s)/volume(s) carrying this "
        f"deployment's project name ({project!r})",
        "were created from a DIFFERENT repo path.",
        "",
        f"This repo is {REPO_ID_LABEL}={identity}. These are not. Every line below is read off "
        "the resource's own",
        "labels — none of it is inferred:",
    ]
    for resource in foreign:
        note = _path_liveness_note(resource.recorded_path)
        lines.append(f"  {resource.kind} {resource.name}  — {resource.describe_origin()}{note}")

    lines.extend(
        [
            "",
            "A compose project name is the repo's DIRECTORY name, so two clones or worktrees "
            "of one deployment share",
            "it — and would share a volume namespace. The identity is a hash of that path, so "
            "a directory you RENAMED",
            "or MOVED also reads as a different repo from here: if that is what happened, "
            "these are your own resources",
            "under their old path, and reset still will not remove them, because from here it "
            "cannot tell them apart",
            "from a colleague's.",
            "",
            "Nothing has been stopped, removed, or written.",
            "",
        ]
    )
    lines.extend(_foreign_remedies(foreign))
    return "\n".join(lines)


def _path_liveness_note(path: str | None) -> str:
    """What the filesystem adds to a recorded path — nothing, when there is none.

    Separate from :meth:`Resource.describe_origin` on purpose: that method
    reports what the labels say, and this reports what is on disk *now*. Keeping
    them apart is what lets the refusal state a recorded path and its current
    absence as two different kinds of fact, rather than one blurred claim.
    """
    if path is None:
        return ""
    return "" if Path(path).is_dir() else "  (no such directory on this host now)"


def _foreign_remedies(foreign: Sequence[Resource]) -> list[str]:
    """The way out, branched on what the refusal actually knows.

    Three states, because the honest advice differs and a single paragraph
    covering all of them would be wrong in two: a recorded path that still
    exists can be reset from; one that no longer exists cannot, and naming it as
    a remedy would send the operator to a directory that is not there; and with
    no path recorded at all there is nowhere to send them, which has to be said
    rather than papered over with generic advice.
    """
    recorded = [path for resource in foreign if (path := resource.recorded_path)]
    live = [path for path in recorded if Path(path).is_dir()]

    if live:
        return [
            f"That repo is still here. Run `osprey reset` from {live[0]} (or "
            "`osprey reset --repo <path>`) if that",
            "is the deployment you meant to wipe.",
        ]
    if recorded:
        return [
            "None of the paths above exist on this host any more, so these are leftovers from "
            "a deployment whose repo",
            "is gone. Reset will not remove them from here: the identity they carry is tied to "
            "that path, not to this",
            "one. Check them yourself and remove them by hand.",
        ]
    return [
        "No path is recorded on any of them, so this refusal cannot point you at the repo they "
        "came from. If you",
        "know which one it is, run `osprey reset` from there. Otherwise inspect them and remove "
        "them by hand.",
    ]


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def execute_reset(plan: ResetPlan, *, probe: RuntimeProbe) -> None:
    """Carry out *plan*, in the order the plan was printed in.

    The caller is responsible for the confirmation: by the time this is called,
    the operator has agreed to this exact plan, and nothing here re-discovers
    what to remove. Every runtime removal names a resource already on the plan,
    and the ``.env`` rewrite is handed the planned key set rather than re-scanning
    for minted blocks — so the removal set can only shrink between the plan and
    the act (a resource that is already gone), never widen. That is what makes
    "what was printed is what was removed" true even when the runtime's state, or
    the ``.env``, changed underneath in between.

    Order, and why:

    1. ``down`` — stop the stack first, so nothing is holding a volume open and
       nothing writes to ``var/agent_data`` while it is being deleted. A ``down``
       that fails aborts the whole reset before anything is destroyed: a volume
       removed out from under containers that would not stop just fails again
       downstream while masking the real error.
    2. containers, then volumes, then images — dependents before dependencies,
       which is the only order the runtime will accept.
    3. the filesystem: ``var/agent_data`` and ``build/`` (and ``var/audit`` under
       ``--purge-audit``), each removed and — for the state directories — left
       behind empty, so the repo has the shape a fresh ``osprey init`` gives it.
    4. ``.env`` last. It is the one file here that is edited rather than deleted,
       and doing it last means a failure anywhere above leaves the deployment's
       secrets exactly as they were.

    Raises:
        RuntimeError: When the ``down`` fails. Nothing has been removed.
    """
    down_deployment(plan.repo_root)

    for resource in plan.containers:
        probe.remove_container(resource.name)
    for resource in plan.volumes:
        probe.remove_volume(resource.name)
    for resource in plan.images:
        probe.remove_image(resource.name)

    for path in plan.paths:
        _wipe_directory(path, recreate=path != plan.repo_root / BUILD_DIRNAME)

    if plan.env_blocks:
        planned_keys = {key for block in plan.env_blocks for key in block.keys}
        _strip_env_file(plan.repo_root / COMPOSE_ENV_FILENAME, planned_keys)


def _wipe_directory(path: Path, *, recreate: bool) -> None:
    """Delete *path*, optionally leaving an empty directory in its place.

    ``build/`` is left absent — it is output, and its absence is what
    ``osprey up`` reads as "no build found". The state directories are recreated
    empty so the repo matches what ``osprey init`` produces, and so a bind-mount
    source a container expects exists before the next start.
    """
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    if recreate:
        path.mkdir(parents=True, exist_ok=True)


def _strip_env_file(env_path: Path, planned_keys: set[str]) -> None:
    """Rewrite ``.env`` without the minted values the PLAN named, and no others.

    *planned_keys* is what makes "nothing is re-discovered after the
    confirmation" true of this step rather than merely intended. The file is
    necessarily re-read here — it has to be, to rewrite it — but the re-read
    decides only WHERE the planned keys sit, never WHICH keys go. A minted value
    written between the plan and the confirmation (a concurrent ``osprey up``,
    or an operator in another terminal) is left in place: it was never shown, so
    it was never agreed to, and this verb removes only what an operator saw.

    Written in place rather than deleted and recreated so the file keeps its
    ``0o600`` mode across the rewrite: a secrets file that briefly widens its
    permissions during a reset is a real exposure, and it would be invisible.
    """
    if not env_path.is_file():
        return
    original_mode = env_path.stat().st_mode
    text = env_path.read_text(encoding="utf-8")
    stripped, removed = strip_minted_env_blocks(text, planned_keys)
    env_path.write_text(stripped, encoding="utf-8")
    os.chmod(env_path, original_mode)
    if removed:
        # Names only, never values — the same rule the mint side follows.
        logger.key_info(
            "Stripped %d deploy-minted value(s) from %s: %s. Every other entry, including "
            "your provider keys, is untouched.",
            len(removed),
            env_path,
            ", ".join(removed),
        )


# ---------------------------------------------------------------------------
# The whole verb
# ---------------------------------------------------------------------------


def reset_deployment(
    repo_root: Path | str,
    *,
    dry_run: bool = False,
    assume_yes: bool = False,
    purge_audit: bool = False,
    emit: Callable[[str], None] = print,
    probe: RuntimeProbe | None = None,
) -> bool:
    """Plan, show, confirm, and carry out a factory reset.

    Args:
        repo_root: The deployment repo.
        dry_run: Print the plan and stop. Nothing is confirmed and nothing runs.
        assume_yes: Skip the typed confirmation — the scripted path.
        purge_audit: Destroy ``var/audit`` too, under the same gate.
        emit: Where the plan is written. The CLI passes its own printer.
        probe: The runtime seam. Resolved from the deployment's configured
            runtime when not supplied; tests inject one.

    Returns:
        A :class:`ResetOutcome`. The caller maps it to an exit code through
        :data:`EXIT_CODES`; ``DECLINED`` and ``COMPLETED_WITH_FAILURES`` are the
        non-zero endings.

    Raises:
        ForeignCheckoutError: Same-name resources belong to another checkout.
        RuntimeError: The container runtime is unreachable, or the ``down``
            failed. Nothing is removed on either path.
    """
    repo_root = Path(repo_root)
    if probe is None:
        probe = _default_probe(repo_root)

    plan = plan_reset(repo_root, probe=probe, purge_audit=purge_audit)
    for line in plan.render():
        emit(line)

    if dry_run:
        emit("")
        emit("--dry-run: nothing was stopped, removed, or written.")
        return ResetOutcome.DRY_RUN

    if not plan.removes_anything:
        emit("")
        emit("Nothing to reset: this deployment has no containers, volumes, images, agent")
        emit("data, build output, or minted tokens. Nothing was touched.")
        return ResetOutcome.NOTHING_TO_DO

    emit("")
    token = confirmation_token(repo_root)
    try:
        confirmed = confirm_destroy(
            f"This PERMANENTLY destroys {plan.confirmation_summary()}. Type '{token}' to confirm: ",
            assume_yes,
            expected=token,
        )
    except KeyboardInterrupt:
        # Provably before the first destructive call — the gate is what stands
        # between here and `execute_reset`. An interrupt anywhere LATER reaches
        # the caller instead, which must not repeat this claim.
        emit("")
        emit("Reset aborted at the confirmation prompt. Nothing was touched.")
        return ResetOutcome.DECLINED
    if not confirmed:
        emit("Reset aborted: confirmation did not match. Nothing was touched.")
        return ResetOutcome.DECLINED

    execute_reset(plan, probe=probe)
    emit("")
    if probe.failures:
        # The plan promised these would go and they did not. Saying "complete"
        # here would be the one lie this verb cannot afford: an operator who
        # believes a volume is gone will not go looking for the data in it.
        emit(
            f"Reset finished, but {len(probe.failures)} resource(s) on the plan could not "
            "be removed:"
        )
        for subject, reason in probe.failures:
            emit(f"  {subject}: {reason}")
        emit("Everything else on the plan was removed. Re-run `osprey reset` after clearing")
        emit("whatever is holding them, or remove them by hand.")
        return ResetOutcome.COMPLETED_WITH_FAILURES
    emit(f"Reset complete. `osprey build && osprey up -d` starts {plan.project} again.")
    return ResetOutcome.COMPLETED


def _default_probe(repo_root: Path) -> RuntimeProbe:
    """The real runtime seam, refusing early when the daemon is unreachable.

    The refusal is the point. Every listing this module makes is a filtered
    read, and a filtered read against a runtime that is not running succeeds
    with empty output — which reset would render as "no containers, no volumes"
    and an operator would read as "already clean". Checking first is what keeps
    a plan from stating an absence it never verified.
    """
    from osprey.deployment.runtime_helper import verify_runtime_is_running

    config_path = as_built_config_path(repo_root)
    config: dict | None = None
    if config_path.is_file():
        try:
            from osprey.utils.config import load_project_config

            config = load_project_config(str(config_path), wrap_errors=True)
        except Exception:  # noqa: BLE001 - runtime selection falls back to detection
            config = None

    is_running, error = verify_runtime_is_running(config)
    if not is_running:
        raise RuntimeError(
            f"{error}\n\nReset needs the container runtime to see what this deployment owns. "
            "Against a runtime that is not running every listing comes back empty, which "
            "would look like a deployment that is already clean. Nothing was touched."
        )

    return RuntimeProbe(get_runtime_command(config)[0], env=runtime_env(config, os.environ.copy()))
