"""Named-volume teardown hygiene for the deploy-backed e2e suites.

``osprey down`` keeps named volumes by design: stopping a deployment must never
destroy its data, and the destructive verb is ``osprey reset``. For a TEST
deployment that durability inverts into a liability: a volume left behind by
one attempt survives into the next -- ``pytest.mark.flaky`` reruns included --
carrying whatever state the earlier attempt wrote into it. Stale queue
contents, store credentials from a previous init, an agent-data tree owned by
the wrong user: each of these has produced a failure that pointed at the code
under test when the actual cause was inherited disk state.

So every deploy fixture removes its own volumes in the same ``finally`` that
runs ``osprey down`` -- the teardown of the attempt that created them, which is
the only place cleanup is guaranteed to know the volumes are no longer wanted.
A pre-flight in the NEXT attempt is not equivalent: it cannot tell a leftover
from a deployment someone wants to keep, which is exactly the judgment
``osprey reset`` refuses to automate.

The removal is scoped the way the shipped destructive verb scopes it (see
``osprey.deployment.reset``): volumes are LISTED via compose's own project
stamp -- the ``com.docker.compose.project`` label, valued with the
``COMPOSE_PROJECT_NAME`` the deploy path pins -- then removed one exact name at
a time. Never a prune, never ``-a``/``--all``, never a name glob: any volume
of any other compose project on the host is invisible to the listing. External
(host-global) volumes a deployment merely mounts carry no project stamp and
are equally invisible, so a suite that manages one (e.g. the OpenObserve data
volume) still handles it explicitly.

Do NOT call this around a mid-test ``down`` whose point is that data survives
it -- ``test_deploy_writeback_e2e`` proves precisely that guarantee.
"""

from __future__ import annotations

import subprocess

#: The label compose stamps on every volume (and container) it creates, valued
#: with the compose project name. Spelled the same way as
#: ``osprey.deployment.reset.COMPOSE_PROJECT_LABEL``; duplicated here so the
#: e2e helper stays importable without the deployment package on the path.
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"


def remove_project_volumes(project: str, *, runtime: str = "docker") -> None:
    """Remove every named volume compose created for ``project``, if any.

    Best-effort by construction: this runs inside teardown ``finally`` blocks,
    after the test verdict exists, and must never replace that verdict with a
    cleanup error. Problems are printed (landing in the CI log next to the
    ``osprey down`` output the same blocks already surface) rather than raised.
    A volume still held by a container is left in place and reported; a volume
    already gone is a success.

    :param project: The RESOLVED compose project name -- the value
        ``resolve_project_name`` returns and the deploy pins as
        ``COMPOSE_PROJECT_NAME`` (``_orm_stack.project_prefix`` for the suites
        that already derive it there).
    :param runtime: Container CLI to drive, for the suites that run under a
        detected ``docker``/``podman`` runtime.
    """
    if not project:
        # An empty value would still be a syntactically valid label filter --
        # one that no longer says "this deployment's volumes". Refuse loudly:
        # only a caller bug can produce it.
        raise ValueError("remove_project_volumes needs a non-empty project name")

    try:
        listed = subprocess.run(
            [
                runtime,
                "volume",
                "ls",
                "--filter",
                f"label={COMPOSE_PROJECT_LABEL}={project}",
                "--format",
                "{{.Name}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"could not list {project!r} volumes for teardown: {exc}")  # noqa: T201
        return
    if listed.returncode != 0:
        print(  # noqa: T201
            f"could not list {project!r} volumes for teardown: "
            f"`{runtime} volume ls` rc={listed.returncode} {(listed.stderr or '').strip()}"
        )
        return

    for name in (line.strip() for line in listed.stdout.splitlines()):
        if not name:
            continue
        try:
            removed = subprocess.run(
                [runtime, "volume", "rm", "-f", name],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"volume {name!r} not removed at teardown: {exc}")  # noqa: T201
            continue
        if removed.returncode != 0:
            print(  # noqa: T201
                f"volume {name!r} not removed at teardown: "
                f"rc={removed.returncode} {(removed.stderr or '').strip()}"
            )
