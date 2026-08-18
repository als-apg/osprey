"""Deployment-subsystem errors.

Recurring domain errors for container deployment. Framework-level, cross-cutting
errors live in :mod:`osprey.errors`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


class DeploymentError(Exception):
    """Base class for deployment failures."""


class CapturedProcessError(DeploymentError):
    """A captured child process exited non-zero.

    Carries the ``spool_path`` holding that process's full output so the verb's
    handler can replay it beneath the failed phase line. In verbose mode there
    is no spool — the output already went to the terminal — and ``spool_path``
    is ``None``.

    This replaces :class:`subprocess.CalledProcessError` for runs made through
    :func:`osprey.deployment.subprocess_capture.run_captured`, so callers have
    one exception type to catch whichever mode they ran in.
    """

    def __init__(self, cmd: Sequence[str], returncode: int, spool_path: Path | None = None) -> None:
        self.cmd = list(cmd)
        self.returncode = returncode
        self.spool_path = spool_path
        where = f" (output: {spool_path})" if spool_path is not None else ""
        super().__init__(f"Command '{' '.join(self.cmd)}' exited {returncode}{where}")


class ComposeInterpolationError(DeploymentError):
    """A secret bound for a compose ``env_file:`` contains ``$``.

    Docker Compose interpolates env_file *values*, so such a secret reaches the
    container truncated (or, when the text after ``$`` names a variable set on
    the deploy host, with the host's value spliced in) while the file on disk
    reads correctly. podman-compose mangles a smaller but different set (the
    braced ``${...}`` form only), so the two deliver different wrong values from
    the same file. Nothing downstream can tell the difference between a
    truncated secret and a wrong one, so the symptom is an authentication
    failure pointing nowhere near the cause.

    That invisibility is why this refuses the deploy rather than warning. A
    warning would scroll past and leave a stack running with a credential no
    service accepts; the honest outcome is to stop before compose is invoked,
    while the operator still has the context to fix it.

    Carries the offending variable *names* only. The values are secrets and are
    never rendered — the same discipline as
    ``service_tokens._raise_invalid_var``.
    """

    def __init__(self, variables: Sequence[str], path: str | Path) -> None:
        self.variables = list(variables)
        self.path = str(path)
        named = ", ".join(self.variables)
        super().__init__(
            f"{named} in {self.path} contain(s) '$'. Docker Compose interpolates "
            f"env_file values, so the container would receive a truncated secret "
            f"while {self.path} still reads correctly. Refusing to deploy. "
            f"(Values not shown.) Re-issue or rotate the listed secret(s) to a "
            f"'$'-free value — '$' cannot be escaped portably here, because '$$' "
            f"means a literal '$' to Docker Compose but two characters to a "
            f"runtime that does not interpolate."
        )


class DeploymentPreconditionError(DeploymentError):
    """A deploy verb refused to act because a precondition is not met.

    Every one of these means the same three things to an operator: the verb did
    nothing, the deployment is exactly as it was, and there is one fixed thing
    to do about it. So the two halves the operator needs are *fields* rather
    than prose baked into a message — ``reason`` says what is not true, and
    ``remedy`` says what makes it true (a command, possibly multi-line). A CLI
    handler renders the pair and never has to know which precondition failed.

    ``summary`` is the subclass's one-line lead-in, so ``str(exc)`` reads as
    ``"{summary}: {reason}"``.

    Every unmet-precondition refusal on a deploy path belongs here, so that
    ``except DeploymentPreconditionError`` means exactly "the deploy refused,
    nothing happened" — and, crucially, so that no such refusal escapes as a
    bare :class:`RuntimeError`, indistinguishable from a genuine bug.
    """

    summary = "This deployment cannot proceed"

    def __init__(self, reason: str, remedy: str) -> None:
        self.reason = reason
        self.remedy = remedy
        super().__init__(f"{self.summary}: {reason}")


class DevModeUnavailableError(DeploymentPreconditionError):
    """``--dev`` was requested but the local osprey wheel cannot be produced.

    ``--dev`` is a statement about *which code runs in the containers*: build a
    wheel from the local checkout so the containers run it instead of the pinned
    PyPI release. When that cannot be done, continuing would deploy a different
    codebase than the one asked for — the containers would come up healthy,
    running released osprey, with no signal that the local checkout was never
    involved. That is worse than not deploying: it silently invalidates whatever
    the deployment was meant to test.

    So this is an error rather than a warning. It carries a ``remedy`` describing
    how to satisfy the precondition, which the CLI renders beneath the reason.
    """

    summary = "--dev cannot be honored"


class UnreleasedVersionPinError(DeploymentPreconditionError):
    """An ``osprey-framework==`` pin was needed, but this build is not a release.

    The same reasoning as :class:`DevModeUnavailableError`, from the other
    direction. A development checkout sits some number of commits past the last
    tag, and no distribution on PyPI corresponds to it. Pinning to the nearest
    release would produce containers running code the operator never wrote, with
    nothing in the logs saying the two differ; emitting no pin at all would leave
    the image tracking whatever PyPI resolves to that day.

    So this refuses, and carries a ``remedy`` the CLI renders beneath the reason.
    """

    summary = "Cannot pin osprey-framework"


class NoRenderedBuildError(DeploymentPreconditionError):
    """A deploy verb was pointed at a repo whose ``build/`` holds no render.

    Not a build *failure* — nothing was attempted. ``build/`` either does not
    hold a rendered ``config.yml`` at all, or holds one declaring services that
    no compose file was rendered for. Either way the verb has nothing to act on
    and the remedy is singular: run ``osprey build``. It is also the one refusal
    on this path that ``--as-built`` does not answer, since ``--as-built`` means
    "start what is already rendered".
    """

    summary = "No rendered build to start from"


class UnmetPreconditionsError(DeploymentPreconditionError):
    """Every cheaply checkable precondition a start failed, reported at once.

    The other subclasses each describe ONE thing that is not true, because each
    is raised the moment its own check fails. That is the right shape for a
    check that can only be made once the step before it has run — but most of a
    start's preconditions are answerable from this deployment's own files,
    before anything is built or started, and reporting those one at a time costs
    the operator a full deploy attempt per finding: fix, re-run, wait, discover
    the next one.

    So the collectable ones are probed together and land here. ``summary`` is
    the count, and ``reason`` is the findings enumerated in the order the deploy
    would have hit them, each followed by its own remedy line. There is no
    top-level ``remedy``: with more than one thing to fix there is no single
    ``→`` line that is honest, and naming one of several would read as *the* way
    through. This is the convention ``gate_start_from_build`` already follows
    for its two-exits refusal — the alternatives go in the cause as an aligned
    block, and the operator picks. It holds at a count of one too, where the
    finding's own remedy line is the whole answer and a duplicate of it at the
    bottom would be noise.

    Raised for **one** finding as well as for many, deliberately. A start that
    refuses over a single precondition then reads the same as one that refuses
    over four: the same frame, the same enumeration, the same place to look for
    the remedy. The alternative — a bare message below some threshold and a
    structured one above it — would make the common case the unfamiliar one.

    Args:
        findings: ``(problem, remedy)`` pairs, in the order the deploy would
            have reached them. ``remedy`` may be empty for a problem whose own
            text already says what to do about it, which is the shape of the
            refusals that carry their fix in prose.
    """

    def __init__(self, findings: Sequence[tuple[str, str]]) -> None:
        self.findings = [(str(problem), str(remedy)) for problem, remedy in findings]
        count = len(self.findings)
        noun = "precondition" if count == 1 else "preconditions"
        # Instance attribute shadowing the class one: the lead-in is a fact
        # about THIS refusal (how many things are wrong), not about the type.
        self.summary = f"{count} unmet {noun} on this deployment"
        blocks = []
        for index, (problem, remedy) in enumerate(self.findings, start=1):
            text = f"{problem}\n{remedy}" if remedy else problem
            head, *rest = text.splitlines() or [""]
            # Continuation lines are indented under the number so a multi-line
            # finding reads as one item rather than as several.
            blocks.append("\n".join([f"{index}. {head}", *(f"   {line}" for line in rest)]))
        super().__init__("\n\n".join(blocks), "")


class ArchiverClientMissingError(DeploymentPreconditionError):
    """A deploy would seed an archiver store, but pymongo is not importable here.

    "Here" is the load-bearing word, and it is why this carries its own type
    rather than a bare :class:`RuntimeError`. OSPREY seeds the store *in the
    process running the CLI*, not in the project's ``build/.venv`` and not in
    the project image — so the one lever an operator reaches for first, a
    ``dependencies:`` entry in the build profile, cannot satisfy it. That entry
    is real and does install pymongo, into all three of the environments the
    project owns; none of them is this one. An operator who has declared it and
    still sees a refusal is owed that sentence explicitly, so ``reason`` names
    the interpreter instead of only naming the package.

    pymongo is a core dependency, so reaching this at all means an incomplete
    environment — a partial install, a stale venv, a wheel built from a narrower
    dependency set — rather than an unselected option.
    """

    summary = "The archiver store cannot be seeded"


class NoComposeFilesError(DeploymentPreconditionError):
    """A read verb needs the compose files a build rendered, and there are none.

    The counterpart of :class:`NoRenderedBuildError` for the verbs that only
    *report* on a deployment (``osprey logs``): they never render, so an absent
    or empty ``build/`` leaves them nothing to read.
    """

    summary = "No rendered compose files to read"
