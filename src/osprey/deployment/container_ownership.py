"""Which containers on a host belong to one deployment.

One rule, read by every surface that grades or lists a deployment's containers
(the ``container`` health probe and ``osprey status``):

* A row labelled for this project is ours.
* A row labelled for another project is never ours, whatever it is called.
* A row with no project label at all is ours only when one of its names matches
  a service by whole name segments.

The project a row is labelled for is its :data:`PROJECT_LABEL`, or, when that
is absent, the :data:`COMPOSE_PROJECT_LABEL` compose stamps on every container
it creates. Every OSPREY compose call pins the compose project to
:func:`~osprey.deployment.compose_generator.resolve_project_name`, so the
compose label names this deployment on every container compose created for it,
including one whose template writes no OSPREY label.

Rows are decoded runtime ``ps --format json`` records in either shape: podman
emits ``Labels`` as an object and ``Names`` as a list, docker emits a
comma-joined ``k=v`` string and a single name.

A second question is answered here too: which *copy* of a deployment holds its
project name on this host (:func:`host_claim`). The compose project name says
which deployment a container belongs to, and the
:data:`~osprey.deployment.compose_generator.REPO_ID_LABEL` says which checkout
created it. :func:`claims_row` is that rule, read by every verb that acts on the
host: ``up`` and ``restart`` refuse to start over another copy's containers,
``reset`` refuses to remove them, and the host-port preflight does not count a
port another copy holds as this deployment's own. The label is on containers
only. Volumes belong to the project by name, so they are listed but never
partitioned: a second checkout that declares the same project name is the same
instance and shares its data by declaration.

This module imports nothing from :mod:`osprey.deployment.reset` or
:mod:`osprey.deployment.container_lifecycle`: both import it.
"""

from __future__ import annotations

import enum
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from osprey.deployment.compose_generator import PROJECT_LABEL, PROJECT_ROOT_LABEL, REPO_ID_LABEL

#: Label naming the compose project of a container. Compose itself stamps it on
#: every container it creates.
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"

#: Label naming the compose service a container runs. Compose itself stamps it
#: on every container it creates.
COMPOSE_SERVICE_LABEL = "com.docker.compose.service"


def container_label(container: Mapping[str, Any], key: str) -> str | None:
    """Read one label off a runtime ``ps --format json`` record.

    Two shapes, because two runtimes: podman emits ``Labels`` as an object,
    docker as a comma-joined ``k=v`` string. ``None`` when the label is absent,
    and that absence is a real answer here rather than a parse failure — a
    container created by an OSPREY that did not stamp :data:`REPO_ID_LABEL`
    carries none.

    Args:
        container: One decoded ``ps`` record.
        key: Label key to read.

    Returns:
        The label's value, or ``None``.
    """
    labels = container.get("Labels", {})
    if isinstance(labels, dict):
        value = labels.get(key)
        return value if isinstance(value, str) else None
    if isinstance(labels, str):
        for label in labels.split(","):
            if "=" in label:
                name, value = label.split("=", 1)
                if name.strip() == key:
                    return value.strip()
    return None


def container_names(container: Mapping[str, Any]) -> list[str]:
    """Every name a ``ps`` record carries, without docker's leading ``/``."""
    names = container.get("Names", [])
    candidates = names if isinstance(names, list) else [names]
    return [str(name).lstrip("/") for name in candidates if name]


def first_container_name(container: Mapping[str, Any]) -> str:
    """The first name a ``ps`` record carries, or ``""`` when it has none."""
    names = container_names(container)
    return names[0] if names else ""


def row_project(row: Mapping[str, Any]) -> str | None:
    """The project a row is labelled for, or ``None`` when it carries no project label.

    The OSPREY :data:`PROJECT_LABEL` wins over :data:`COMPOSE_PROJECT_LABEL`
    when both are present.

    Args:
        row: One decoded ``ps`` record.

    Returns:
        The labelled project name, or ``None``.
    """
    project = container_label(row, PROJECT_LABEL)
    if project is not None:
        return project
    return container_label(row, COMPOSE_PROJECT_LABEL)


def _tokens(name: str) -> list[str]:
    """Lower-case *name*, read ``_`` as ``-``, and split it into its segments."""
    return [token for token in name.lower().replace("_", "-").split("-") if token]


def names_match_service(row: Mapping[str, Any], service: str, project_name: str) -> bool:
    """Whether one of a row's names matches *service* by whole name segments.

    The service is reduced to its last dotted segment. The row's candidates are
    its container names plus its :data:`COMPOSE_SERVICE_LABEL`. A leading
    ``<project_name>-`` is removed from each candidate before it is split, so a
    project name never reads as a service name. Case and ``_``/``-`` do not
    matter. The row matches when the service's segments appear as a contiguous
    run of some candidate's segments: ``dispatch_worker`` matches
    ``p-dispatch-worker-2``, and ``archive`` does not match
    ``p-archiver-recorder``.

    Args:
        row: One decoded ``ps`` record.
        service: A service name, dotted or short.
        project_name: This deployment's compose project name.

    Returns:
        ``True`` when a candidate name contains the service's segments in order.
    """
    wanted = _tokens(service.split(".")[-1])
    if not wanted:
        return False
    prefix = "-".join(_tokens(project_name))
    candidates = list(container_names(row))
    compose_service = container_label(row, COMPOSE_SERVICE_LABEL)
    if compose_service:
        candidates.append(compose_service)
    width = len(wanted)
    for candidate in candidates:
        normalized = candidate.lower().replace("_", "-")
        if prefix and normalized.startswith(prefix + "-"):
            normalized = normalized[len(prefix) + 1 :]
        tokens = _tokens(normalized)
        if any(tokens[i : i + width] == wanted for i in range(len(tokens) - width + 1)):
            return True
    return False


class ClaimBasis(enum.Enum):
    """Why a row was claimed for a deployment."""

    #: Its :data:`REPO_ID_LABEL` equals the given checkout identity.
    CHECKOUT = "checkout"
    #: The project it is labelled for equals this deployment's project name.
    PROJECT = "project"
    #: It carries no project label, and one of its names matches a service.
    NAME = "name"


@dataclass(frozen=True)
class ClaimedContainer:
    """One row claimed for a deployment, and the basis it was claimed on."""

    row: Mapping[str, Any]
    basis: ClaimBasis


@dataclass(frozen=True)
class DeploymentContainers:
    """The rows one deployment claims, and the rows other OSPREY projects hold.

    Attributes:
        project_name: The compose project name the rows were claimed for.
        ours: Every claimed row, in host order.
        other_projects: Rows whose :data:`PROJECT_LABEL` names a different
            project. A row labelled only with another compose project is not an
            OSPREY container and is not listed.
    """

    project_name: str
    ours: tuple[ClaimedContainer, ...]
    other_projects: tuple[Mapping[str, Any], ...]

    def for_service(self, service: str) -> list[Mapping[str, Any]]:
        """The claimed rows whose names match *service*, sorted by first name.

        Args:
            service: A service name, dotted or short.

        Returns:
            The matching rows of :attr:`ours`.
        """
        rows = [
            claimed.row
            for claimed in self.ours
            if names_match_service(claimed.row, service, self.project_name)
        ]
        return sorted(rows, key=first_container_name)


def deployment_containers(
    rows: Iterable[Mapping[str, Any]],
    *,
    project_name: str,
    services: Iterable[str],
    identity: str | None = None,
) -> DeploymentContainers:
    """Claim the rows on a host that belong to one deployment.

    Per row, in this order, the first hit wins:

    1. *identity* is given and the row's :data:`REPO_ID_LABEL` equals it →
       :attr:`ClaimBasis.CHECKOUT`, whatever project the row is labelled for.
    2. The project the row is labelled for (:func:`row_project`) equals
       *project_name* → :attr:`ClaimBasis.PROJECT`.
    3. The row is labelled for another project → not ours; listed in
       ``other_projects`` when it carries :data:`PROJECT_LABEL`.
    4. The row carries no project label → :attr:`ClaimBasis.NAME` when a name
       matches any of *services* (:func:`names_match_service`), else ignored.

    ``osprey status`` passes the checkout identity and the build's
    ``deployed_services``, because it lists a whole deployment and must agree
    with the label ``osprey down`` selects on. The ``container`` health probe
    passes no identity and only the service it grades. For any deployed service
    the two therefore claim the same rows.

    Args:
        rows: Decoded ``ps`` records.
        project_name: This deployment's compose project name.
        services: Service names an unlabelled row may match by name.
        identity: This checkout's repo identity, or ``None`` to claim by
            project and name only.

    Returns:
        The claimed rows and the rows of other OSPREY projects.
    """
    service_names = tuple(services)
    ours: list[ClaimedContainer] = []
    other_projects: list[Mapping[str, Any]] = []
    for row in rows:
        if identity is not None and container_label(row, REPO_ID_LABEL) == identity:
            ours.append(ClaimedContainer(row, ClaimBasis.CHECKOUT))
            continue
        project = row_project(row)
        if project == project_name:
            ours.append(ClaimedContainer(row, ClaimBasis.PROJECT))
        elif project is not None:
            if container_label(row, PROJECT_LABEL) is not None:
                other_projects.append(row)
        elif any(names_match_service(row, s, project_name) for s in service_names):
            ours.append(ClaimedContainer(row, ClaimBasis.NAME))
    return DeploymentContainers(
        project_name=project_name, ours=tuple(ours), other_projects=tuple(other_projects)
    )


# ---------------------------------------------------------------------------
# Which checkout holds this project's name on the host
# ---------------------------------------------------------------------------

#: Where a foreign checkout's PATH is read from, most authoritative first.
#: ``repo_id`` is a one-way hash, so the identity that proves a container is
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
    PROJECT_ROOT_LABEL,
)


def claims_row(
    *, row_repo_id: str | None, row_project: str | None, project: str, repo_id: str | None
) -> bool:
    """Whether a container on the host is this checkout's own.

    The one ownership rule, in two steps:

    * the row and this checkout both carry a repo id: the ids decide, outright.
      Another checkout of the same repo shares this deployment's compose
      project name, so the project alone would call a real collision ours;
    * either side carries none (an OSPREY that stamped no label created the
      row, or this checkout cannot be identified): the compose project decides.
      An unlabelled container of this project is therefore ours.

    Args:
        row_repo_id: The row's :data:`REPO_ID_LABEL`, empty or ``None`` when absent.
        row_project: The compose project the row is labelled for, if any.
        project: This deployment's compose project name.
        repo_id: This checkout's repo identity, empty or ``None`` when unknown.

    Returns:
        ``True`` when the row is this checkout's own.
    """
    if row_repo_id and repo_id:
        return row_repo_id == repo_id
    return bool(row_project) and row_project == project


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
    """One project's containers, split by which checkout created them."""

    ours: list[Resource] = field(default_factory=list)
    foreign: list[Resource] = field(default_factory=list)
    unidentified: list[Resource] = field(default_factory=list)


def partition_by_identity(resources: Iterable[Resource], identity: str) -> Partition:
    """Split one project's *resources* into ours / another checkout's / unprovable.

    :func:`claims_row` decides between ours and foreign. A resource with no
    repo id is held apart as unidentified, because the callers treat it
    differently: ``reset`` will not remove what it cannot prove is this
    checkout's, and a start counts it as ours.

    Args:
        resources: Resources already filtered to one compose project.
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
        elif claims_row(row_repo_id=repo_id, row_project=None, project="", repo_id=identity):
            partition.ours.append(resource)
        else:
            partition.foreign.append(resource)
    return partition


@dataclass(frozen=True)
class RefusalWording:
    """What one verb says when another copy holds its project name.

    The facts — which containers, which path, whether it is still on disk — are
    the same for every verb and are spelled once, by :func:`foreign_refusal`.
    What differs is what the verb was about to do, and so why it stopped and
    where it sends the operator.

    Attributes:
        summary_tail: The opening line, minus the verb the operator typed.
        why_stopped: The sentence that says what acting would have done.
        renamed: The paragraph for an operator who moved this directory.
        live_remedy: The way out when the other copy is on disk, with ``{path}``.
        leftovers: The way out when every recorded path is gone from this host,
            with ``{rm}`` (the removal command) and ``{by_hand}`` (that command
            as a ``": `...`"`` suffix, empty when there is none to spell).
        elsewhere: What to run in a copy the refusal cannot name.
        closing: The last line of the cause, or ``None`` when the verb prints
            its own.
    """

    summary_tail: str
    why_stopped: str
    renamed: str
    live_remedy: str
    leftovers: str
    elsewhere: str
    closing: str | None


#: ``osprey reset`` and ``osprey init --reset``: a removal refused.
RESET_REFUSAL = RefusalWording(
    summary_tail="will not remove containers and volumes from another copy of this repo",
    why_stopped=(
        "Removing them from here would mean taking resources this repo cannot prove are its "
        "own, which is what keeps one checkout from destroying another's, so it stopped."
    ),
    renamed=(
        "If you renamed or moved this directory, these are your own resources under its old "
        "path. Reset still will not take them: from here they cannot be told apart from a "
        "colleague's."
    ),
    live_remedy="reset that deployment where it lives: `osprey reset --repo {path}`",
    leftovers=(
        "these are leftovers from a deployment whose repo is gone, and the identity they "
        "carry is tied to that path rather than to this one, so check them and remove "
        "them by hand{by_hand}"
    ),
    elsewhere="run `osprey reset` there",
    closing="Nothing has been stopped, removed, or written.",
)

#: ``osprey up`` and ``osprey restart``: a start refused.
START_REFUSAL = RefusalWording(
    summary_tail="will not start over containers from another copy of this repo",
    why_stopped=(
        "Starting from here would replace that copy's containers with this one's while both "
        "point at the same volumes, so it stopped."
    ),
    renamed=(
        "If you renamed or moved this directory, these are your own containers under its old "
        "path. They are refused all the same: from here they cannot be told apart from a "
        "colleague's."
    ),
    live_remedy="stop that deployment where it lives: `osprey down --repo {path}`",
    leftovers=(
        "the copy they came from is no longer on this host, so remove its containers: "
        "`{rm}`. The project's volumes are untouched and keep their data."
    ),
    elsewhere="run `osprey down` there",
    closing=None,
)


class ForeignCheckoutError(RuntimeError):
    """Same-named containers were created from a different repo path — nothing was touched.

    Carried in parts rather than as one block of text, because the verbs that
    meet this refusal have different room for it and different advice to give.
    A verb renders :meth:`summary`, :attr:`cause` and :attr:`remedy` through
    :func:`osprey.cli.foreign_refusal.render_foreign_refusal` and shows
    :attr:`inventory` only under ``--verbose``: the inventory is the evidence,
    and on a real deployment it runs to dozens of lines that repeat one path
    between them, which is how an operator ends up reading past the sentence
    that would have explained it.

    The parts claim exactly as much as the labels support, which is the property
    the split has to preserve. :attr:`cause` names a path only when a
    path-evidence label supplies one, and says the path is unknown when none
    does, because a foreign checkout's path is not derivable from its identity
    hash and inventing one would be worse than admitting it is unknown.

    ``str(e)`` remains the whole refusal, evidence included. That is what a log
    record and a ``--verbose`` run keep, and what a caller that has no renderer
    still gets by printing the exception.
    """

    #: The removal verbs' opening line, minus the verb that provoked it.
    SUMMARY_TAIL = RESET_REFUSAL.summary_tail

    def __init__(
        self,
        *,
        project: str,
        identity: str,
        resources: Sequence[Resource],
        cause: str,
        remedy: str | None,
        inventory: Sequence[str],
        summary_tail: str | None = None,
        extra_remedy: str | None = None,
    ) -> None:
        #: The compose project name both copies claim.
        self.project = project
        #: This repo's identity, the one the foreign resources do not carry.
        self.identity = identity
        #: The foreign resources, in discovery order.
        self.resources = list(resources)
        #: Why, in full sentences: what was found, where it came from, and why
        #: refusing is the right answer. Multi-line.
        self.cause = cause
        #: The one thing to do about it, or ``None`` where nothing honest fits.
        self.remedy = remedy
        #: One line per foreign resource, read off its own labels.
        self.inventory = list(inventory)
        #: The opening line, minus the verb.
        self.summary_tail = summary_tail or self.SUMMARY_TAIL
        #: A second way out the refusing verb offers, or ``None``.
        self.extra_remedy = extra_remedy
        super().__init__(self.full_text())

    def summary(self, actor: str = "reset") -> str:
        """The opening line, named for the verb the operator actually typed."""
        return f"{actor} {self.summary_tail}"

    def full_text(self, actor: str = "reset") -> str:
        """Every part, evidence included: the record, and the ``--verbose`` view."""
        parts = [self.summary(actor), "", self.cause]
        if self.inventory:
            parts += [
                "",
                "Read off the resources' own labels, none of it inferred:",
                *self.inventory,
            ]
        if self.remedy:
            parts += ["", f"-> {self.remedy}"]
        if self.extra_remedy:
            parts += [f"   or {self.extra_remedy}"]
        return "\n".join(parts)


def foreign_refusal(
    project: str,
    identity: str,
    foreign: Sequence[Resource],
    wording: RefusalWording = RESET_REFUSAL,
    *,
    runtime: str = "docker",
    volumes: Sequence[Resource] = (),
    extra_remedy: str | None = None,
) -> ForeignCheckoutError:
    """Build the refusal, claiming exactly as much as the labels support.

    An operator who sees this is in one of three situations, and the parts have
    to serve all of them without deciding between them: they are standing in the
    wrong directory, two checkouts of one deployment are genuinely sharing a
    host, or they renamed or moved *this* directory and are meeting their own
    containers under the old path. The recorded path is what tells them which,
    so it is quoted from the resource rather than reconstructed, and when no
    resource carries one, the message says so instead of guessing.

    Neutral about WHICH of the three it is; not neutral about ORDER. The
    conclusion, the path and the way out come first, and the per-resource
    evidence goes to :attr:`ForeignCheckoutError.inventory` for a verb to show
    under ``--verbose``.

    What is a *label* fact and what is a *filesystem* fact are kept apart on
    purpose: the path is reported as recorded, and whether it still exists is
    added separately by :func:`_path_liveness_note`. The remedy then branches on
    that (:func:`_foreign_remedy`), because "go and do it over there" is wrong
    advice for a directory that is no longer on the host.

    Args:
        project: The compose project name both copies claim.
        identity: This checkout's repo identity.
        foreign: The resources another checkout created.
        wording: The refusing verb's sentences.
        runtime: The runtime binary a hand-removal command is spelled with.
        volumes: The project's volumes, named in the cause as shared by name.
        extra_remedy: A second way out the refusing verb offers.

    Returns:
        The refusal, ready to raise.
    """
    inventory = [
        f"  {resource.kind} {resource.name}  — {resource.describe_origin()}"
        f"{_path_liveness_note(resource.recorded_path)}"
        for resource in foreign
    ]
    return ForeignCheckoutError(
        project=project,
        identity=identity,
        resources=foreign,
        cause=_foreign_cause(project, identity, foreign, wording, volumes),
        remedy=_foreign_remedy(foreign, wording, runtime),
        inventory=inventory,
        summary_tail=wording.summary_tail,
        extra_remedy=extra_remedy,
    )


def _distinct(values: Iterable[str]) -> list[str]:
    """``values`` without repeats, in first-seen order."""
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


def _counted(resources: Sequence[Resource]) -> str:
    """``"2 containers are"`` or ``"1 container and 1 volume are"``, from the list."""
    counts = [
        f"{n} {kind}{'' if n == 1 else 's'}"
        for kind in _distinct(resource.kind for resource in resources)
        if (n := sum(1 for resource in resources if resource.kind == kind))
    ]
    return f"{' and '.join(counts)} {'is' if len(resources) == 1 else 'are'}"


def _foreign_cause(
    project: str,
    identity: str,
    foreign: Sequence[Resource],
    wording: RefusalWording,
    volumes: Sequence[Resource],
) -> str:
    """Why this refused, in the order the reader needs it.

    Every path is listed once rather than once per resource. Fifteen containers
    of one deployment record one directory between them, and printing it fifteen
    times says nothing the first line did not while burying what follows.

    Wrapped here rather than left to the renderer:
    :func:`osprey.cli.output.fail` prints each cause line as given, on the rule
    that a caller passes lines already the shape it wants them. So this is where
    the shape is decided.
    """
    recorded = _distinct(path for resource in foreign if (path := resource.recorded_path))
    theirs = _distinct(repo_id for resource in foreign if (repo_id := resource.repo_id))
    ids = f"{identity} here" + (f", {', '.join(theirs)} on those" if theirs else "")

    lines = _wrap(
        f"{_counted(foreign)} named {project!r}, but they were created from a different "
        "copy of this repo:"
    )
    lines.append("")
    if recorded:
        lines += [f"    {path}  ({_path_state(path)})" for path in recorded]
    else:
        lines += _wrap(
            "the path is unknown: none of them carries a label recording it, and a repo path "
            "cannot be derived back out of the identity hash",
            indent="    ",
        )
    lines.append("")
    shared = (
        f" The project's {len(volumes)} volume{'' if len(volumes) == 1 else 's'} belong "
        "to that name, not to either copy."
        if volumes
        else ""
    )
    lines += _wrap(
        f"Both copies take the compose project name {project!r}, so they claim the same "
        f"containers and the same volumes.{shared} The repo ids say they are not the same "
        f"copy ({ids}). {wording.why_stopped}"
    )
    lines.append("")
    lines += _wrap(wording.renamed)
    if wording.closing:
        lines += ["", wording.closing]
    return "\n".join(lines)


#: Where the refusal's prose wraps. Narrow enough that the renderer's indent
#: still leaves it inside a default terminal, since nothing downstream will
#: re-wrap it.
_WRAP_WIDTH = 88


def _wrap(paragraph: str, *, indent: str = "") -> list[str]:
    """``paragraph`` as terminal-width lines, each carrying ``indent``."""
    return textwrap.wrap(
        paragraph,
        width=_WRAP_WIDTH,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _path_state(path: str) -> str:
    """Whether ``path`` is on this host now, as the filesystem answers it."""
    return "still on disk" if Path(path).is_dir() else "no such directory on this host now"


def _foreign_remedy(foreign: Sequence[Resource], wording: RefusalWording, runtime: str) -> str:
    """The way out, branched on what the refusal actually knows.

    Three states, because the honest advice differs and one line covering all of
    them would be wrong in two: a recorded path that still exists can be acted
    on from there; one that no longer exists cannot, and naming it as a remedy
    would send the operator to a directory that is not there; and with no path
    recorded at all there is nowhere to send them, which has to be said rather
    than papered over with generic advice. Where every foreign resource is a
    container, the hand removal is spelled out as the one command it is.
    """
    recorded = [path for resource in foreign if (path := resource.recorded_path)]
    live = [path for path in recorded if Path(path).is_dir()]

    if live:
        return wording.live_remedy.format(path=live[0])
    names = [resource.name for resource in foreign if resource.kind == "container"]
    rm = f"{runtime} rm -f {' '.join(names)}" if names and len(names) == len(foreign) else ""
    command = f": `{rm}`" if rm else ""
    if recorded:
        return wording.leftovers.format(rm=rm, by_hand=command)
    return (
        f"this refusal cannot point you at the repo they came from; {wording.elsewhere} if "
        f"you know which one it is, and otherwise inspect them and remove them by hand{command}"
    )


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


class ProjectListings(Protocol):
    """The two read-only listings :func:`host_claim` asks of a runtime."""

    runtime: str

    def containers_for_project(
        self, project: str, *, include_stopped: bool = True
    ) -> list[Resource]:
        """This compose project's containers, with their labels."""
        ...

    def volumes_for_project(self, project: str) -> list[Resource]:
        """This compose project's named volumes, with their labels."""
        ...


@dataclass(frozen=True)
class HostClaim:
    """Which copy of a deployment holds its project name on this host.

    Attributes:
        project: The compose project name that was asked about.
        identity: This checkout's repo identity.
        runtime: The runtime binary that answered.
        containers: The project's containers, partitioned by
            :func:`partition_by_identity`.
        volumes: The project's volumes. Listed, never partitioned: a volume
            belongs to the project by name.
    """

    project: str
    identity: str
    runtime: str
    containers: Partition
    volumes: tuple[Resource, ...] = ()

    @property
    def foreign(self) -> list[Resource]:
        """Containers another checkout created under this project name."""
        return self.containers.foreign

    @property
    def held_elsewhere(self) -> bool:
        """Whether another checkout's containers hold this project name."""
        return bool(self.containers.foreign)

    @property
    def other_copy_gone(self) -> bool:
        """Whether every foreign container records a path, and none is on this host now.

        That is the moved or deleted checkout: removing its containers frees the
        name, and the volumes, which belong to the name, carry over. A path
        still on disk, or a container that records none, is not that case.
        """
        paths = [resource.recorded_path for resource in self.foreign]
        return bool(paths) and all(path is not None and not Path(path).is_dir() for path in paths)

    def refusal(
        self, wording: RefusalWording, *, extra_remedy: str | None = None
    ) -> ForeignCheckoutError:
        """The refusal for a verb that will not act over another copy's containers.

        Args:
            wording: The refusing verb's sentences.
            extra_remedy: A second way out that verb offers.

        Returns:
            The refusal, ready to raise.
        """
        return foreign_refusal(
            self.project,
            self.identity,
            self.foreign,
            wording,
            runtime=self.runtime,
            volumes=self.volumes,
            extra_remedy=extra_remedy,
        )


def host_claim(
    project: str,
    repo_id: str,
    *,
    probe: ProjectListings,
    include_stopped: bool = True,
) -> HostClaim:
    """Ask the runtime which checkout holds *project* on this host.

    Read-only. Containers are partitioned by :data:`REPO_ID_LABEL`; an
    unlabelled container counts as ours, as :func:`claims_row` rules. Volumes
    are listed for the refusal's message and are never partitioned.

    Args:
        project: This deployment's compose project name.
        repo_id: This checkout's repo identity.
        probe: The runtime seam (``osprey.deployment.reset.RuntimeProbe``).
        include_stopped: ``False`` lists running containers only.

    Returns:
        The claim.

    Raises:
        RuntimeError: When the runtime cannot list the project's resources.
    """
    containers = probe.containers_for_project(project, include_stopped=include_stopped)
    return HostClaim(
        project=project,
        identity=repo_id,
        runtime=probe.runtime,
        containers=partition_by_identity(containers, repo_id),
        volumes=tuple(probe.volumes_for_project(project)),
    )
