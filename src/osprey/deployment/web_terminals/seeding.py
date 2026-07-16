"""Seed per-user CLAUDE.md and skills into web-terminal containers.

Runs both as the ``deploy up`` post-up hook and standalone via ``osprey deploy
seed``, sharing one implementation driven directly off the parsed facility
config. Reads the on-disk ``docker/web-terminal-context/`` overlay tree and
reconciles each live user container.

Contract:

* ``CLAUDE.md`` is REPLACED every run: ``docker/web-terminal-context/base.md``
  concatenated with the user's ``extra.md`` (or the legacy flat ``<user>.md``),
  piped into the container's ``/data/claude-config/CLAUDE.md`` (user scope —
  not gated by ``--setting-sources``, unlike skills).
* ``skills/`` is idempotent and non-destructive via a ``.deploy-managed``
  sentinel dropped inside the container: only sentinel-bearing skill dirs are
  ever touched. A user's live-installed skill (e.g. ``osprey skills install``,
  no sentinel) always survives a reseed. Every overlay-shipped skill dir gets
  re-stamped; a previously-managed skill the overlay no longer ships is
  removed.
* A user whose container isn't up yet is skipped (logged), not fatal — the
  rest of the roster is still seeded.
* ``docker/web-terminal-context/base.md`` is required; its absence aborts the
  whole seed up front (a misconfiguration, not a per-user issue) — mirroring
  the bash source's ``exit 1``.
"""

from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from osprey.deployment.runtime_helper import get_runtime_command, runtime_env
from osprey.deployment.web_terminals.ports import normalize_users
from osprey.utils.config import ConfigBuilder
from osprey.utils.logger import get_logger

logger = get_logger("deployment.web_terminals.seeding")

# Overlay tree root, relative to the project cwd (osprey deploy has already
# chdir'd into the project root by the time this module runs).
_CONTEXT_DIR = Path("docker/web-terminal-context")

_CLAUDE_MD_TARGET = "/data/claude-config/CLAUDE.md"

# Container-side script the concatenated CLAUDE.md content is piped into.
# Runs as root (-u 0): the claude-config volume is root-owned until its first
# chown, and only root can chown it to dispatch.
_CLAUDE_MD_SH = (
    "set -e\n"
    "chown dispatch:dispatch /data/claude-config\n"
    f"cat > {_CLAUDE_MD_TARGET}\n"
    f"chown dispatch:dispatch {_CLAUDE_MD_TARGET}\n"
)

# Container-side script the skills tar stream is piped into. Implements the
# three-phase skill reconcile (see module docstring):
#   1. drop deploy-managed dirs this overlay no longer ships
#   2. drop + re-extract every currently-shipped skill (so edits/removed files
#      inside an already-managed skill land too)
#   3. re-stamp .deploy-managed on each
# $1 = space-separated skill names this overlay currently ships (possibly
# empty); $2 = the target project_skills_dir.
_SKILLS_RECONCILE_SH = (
    "set -e\n"
    'target="$2"\n'
    'mkdir -p "$target"\n'
    'cd "$target"\n'
    'names="$1"\n'
    "for d in */; do\n"
    '  d="${d%/}"\n'
    '  [ -f "$d/.deploy-managed" ] || continue\n'
    "  keep=0\n"
    "  for name in $names; do\n"
    '    [ "$name" = "$d" ] && keep=1 && break\n'
    "  done\n"
    '  [ "$keep" -eq 0 ] && rm -rf -- "$d"\n'
    "done\n"
    "for name in $names; do\n"
    '  rm -rf -- "$name"\n'
    "done\n"
    "tar -xf -\n"
    "for name in $names; do\n"
    '  [ -d "$name" ] && touch "$name/.deploy-managed"\n'
    "done\n"
    'chown -R dispatch:dispatch "$target"\n'
)


def seed_web_terminals(config_path: str | Path, user: str | None = None) -> None:
    """Load ``config_path`` and (re)seed one or all live web-terminal users' containers.

    Entry point for ``osprey deploy seed`` and any other standalone caller.

    Args:
        config_path: Path to the facility ``config.yml``.
        user: If given, seed only this user's container. If ``None`` (default),
            seed every user currently on the roster.

    Raises:
        RuntimeError: If ``docker/web-terminal-context/base.md`` is missing, or
            if every ready container's seed failed (see
            :func:`seed_user_containers`).
        ValueError: If ``user`` is given but not present in
            ``modules.web_terminals.users``.
    """
    config = ConfigBuilder(str(config_path)).raw_config
    seed_user_containers(config, user=user)


def seed_user_containers(
    config: dict[str, Any], *, user: str | None = None, env: dict[str, str] | None = None
) -> None:
    """(Re)seed CLAUDE.md and skills into one or all live web-terminal containers.

    Callable standalone (env resolved from ``config`` via
    :func:`runtime_helper.runtime_env`) or from the ``deploy up`` post-up hook
    with an already-pinned env, so both paths share one implementation of the
    container-side reconcile contract.

    A user whose container isn't up yet is logged and skipped (not fatal —
    seeding continues for the rest of the roster); the same is true of a
    single ready container whose seed fails, so long as at least one *other*
    ready container in this run succeeds. But when every container this run
    actually attempted (i.e. every container that existed and was execed into)
    fails, that is treated as a systemic misconfiguration — e.g. the container
    image is missing the ``dispatch`` user every seed step chowns to — rather
    than an isolated per-user issue, and raised so ``deploy up``/``deploy seed``
    surfaces it instead of silently reporting success with nothing seeded.

    A missing ``base.md`` is a misconfiguration too, and not a per-user
    problem, so it aborts before any user is touched. No-op if web terminals
    are disabled or the roster is empty (and ``user`` was not given).

    Args:
        config: Parsed facility config (a ``ConfigBuilder.raw_config`` dict).
        user: If given, seed only this user's container; a user not present in
            ``modules.web_terminals.users`` raises rather than silently
            no-op'ing. If ``None`` (default), seed every user on the roster.
        env: Environment for runtime subprocess calls. Defaults to
            ``runtime_env(config)`` (``os.environ`` pinned with
            ``COMPOSE_PROJECT_NAME``).

    Raises:
        RuntimeError: If ``docker/web-terminal-context/base.md`` is missing, or
            if at least one container was ready and every ready container's
            seed failed.
        ValueError: If ``user`` is given but not present in
            ``modules.web_terminals.users``.
    """
    modules = config.get("modules") or {}
    web_terminals = modules.get("web_terminals") or {}
    if not web_terminals.get("enabled"):
        return

    roster = normalize_users(web_terminals.get("users"))
    if user is not None:
        targets = [entry for entry in roster if entry["name"] == user]
        if not targets:
            raise ValueError(
                f"User {user!r} is not present in modules.web_terminals.users; nothing was seeded."
            )
    else:
        if not roster:
            return
        targets = roster

    base_md_path = _CONTEXT_DIR / "base.md"
    if not base_md_path.is_file():
        raise RuntimeError(
            f"{base_md_path} not found — cannot seed CLAUDE.md. Every web-terminal "
            "deploy requires a base.md context file."
        )
    base_content = base_md_path.read_text(encoding="utf-8")

    runtime = get_runtime_command(config)[0]
    run_env = env if env is not None else runtime_env(config)
    facility_prefix = (config.get("facility") or {}).get("prefix") or ""
    # Project scope, not $CLAUDE_CONFIG_DIR — the launcher runs the CLI with
    # --setting-sources project, which makes $CLAUDE_CONFIG_DIR/skills/ inert.
    project_skills_dir = f"/app/{facility_prefix}-assistant/.claude/skills"

    logger.info("Seeding per-user CLAUDE.md and skills into claude-config volumes...")
    attempted = 0
    failed = 0
    for entry in targets:
        outcome = _seed_one_user(
            runtime, entry["name"], facility_prefix, base_content, project_skills_dir, env=run_env
        )
        if outcome is None:
            continue  # container not ready — never counts toward the systemic check
        attempted += 1
        if not outcome:
            failed += 1

    if attempted and failed == attempted:
        raise RuntimeError(
            f"Seeding failed for all {attempted} ready web-terminal container(s) — "
            "see the warnings above for each container's error. This looks like a "
            "systemic misconfiguration (e.g. the image is missing something every "
            "seed step depends on), not an isolated per-container issue."
        )


def _seed_one_user(
    runtime: str,
    user: str,
    facility_prefix: str,
    base_content: str,
    project_skills_dir: str,
    *,
    env: dict[str, str] | None,
) -> bool | None:
    """Seed one user's container; never raise.

    Returns:
        ``None`` if the container isn't ready (skipped, doesn't count toward
        the caller's systemic-failure check); ``True`` if the seed succeeded;
        ``False`` if the container was ready but the seed failed.
    """
    container = f"{facility_prefix}-web-{user}"
    if not _container_exists(runtime, container, env=env):
        logger.info(f"  (skipped {user}: container not ready)")
        return None
    try:
        extra_content = _resolve_extra_md(user)
        _seed_claude_md(runtime, container, base_content + extra_content, env=env)
        skills_src = _CONTEXT_DIR / user / "skills"
        _seed_skills(runtime, container, skills_src, project_skills_dir, env=env)
        logger.info(f"  seeded {user}")
        return True
    except Exception as exc:
        logger.warning(f"  (skipped {user}: seeding failed: {_describe_seed_error(exc)})")
        return False


def _describe_seed_error(exc: Exception) -> str:
    """Render ``exc`` for the per-user warning, including subprocess stderr when present.

    A bare ``CalledProcessError`` stringifies to just "returned non-zero exit
    status N", which drops the one piece of information (the container's
    stderr) that would tell an operator *why* — critical for diagnosing the
    systemic-failure case in :func:`seed_user_containers`.
    """
    if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
        stderr = exc.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stderr = stderr.strip()
        if stderr:
            return f"{exc} — stderr: {stderr}"
    return str(exc)


def _resolve_extra_md(user: str) -> str:
    """The per-user ``extra.md`` content, or ``""`` if neither path exists.

    Per-user overlay lives at ``docker/web-terminal-context/<user>/extra.md``;
    falls back to the legacy flat ``docker/web-terminal-context/<user>.md`` for
    facilities that haven't migrated to the directory layout yet. Matches the
    bash source's ``cat base.md "$extra_md" 2>/dev/null``: a missing extra file
    is not an error, it just contributes no content.
    """
    extra_md = _CONTEXT_DIR / user / "extra.md"
    legacy_md = _CONTEXT_DIR / f"{user}.md"
    if not extra_md.is_file() and legacy_md.is_file():
        extra_md = legacy_md
    if extra_md.is_file():
        return extra_md.read_text(encoding="utf-8")
    return ""


def _container_exists(runtime: str, name: str, *, env: dict[str, str] | None) -> bool:
    """True if a container named ``name`` exists (any state) in the runtime.

    Uses ``<runtime> inspect --type container <name>`` rather than podman-only
    ``container exists``, so the check works identically for both docker and
    podman — either of which :func:`runtime_helper.get_runtime_command` may
    select.
    """
    result = subprocess.run(
        [runtime, "inspect", "--type", "container", name],
        capture_output=True,
        text=True,
        env=env,
    )
    return result.returncode == 0


def _seed_claude_md(
    runtime: str, container: str, payload: str, *, env: dict[str, str] | None
) -> None:
    """Pipe ``payload`` into ``container``'s ``/data/claude-config/CLAUDE.md``."""
    subprocess.run(
        [runtime, "exec", "-u", "0", "-i", container, "sh", "-c", _CLAUDE_MD_SH],
        input=payload.encode("utf-8"),
        check=True,
        env=env,
        capture_output=True,
    )


def _seed_skills(
    runtime: str,
    container: str,
    skills_src: Path,
    project_skills_dir: str,
    *,
    env: dict[str, str] | None,
) -> None:
    """Tar ``skills_src`` and reconcile it into ``project_skills_dir`` inside ``container``.

    Tars an empty stream when ``skills_src`` doesn't exist, so the
    container-side reconcile still runs and cleans up any previously-managed
    skill dirs even after the overlay's ``skills/`` disappears entirely.
    """
    names = (
        sorted(p.name for p in skills_src.iterdir() if p.is_dir()) if skills_src.is_dir() else []
    )
    tar_bytes = _build_skills_tar(skills_src)
    subprocess.run(
        [
            runtime,
            "exec",
            "-u",
            "0",
            "-i",
            container,
            "sh",
            "-c",
            _SKILLS_RECONCILE_SH,
            "sh",
            " ".join(names),
            project_skills_dir,
        ],
        input=tar_bytes,
        check=True,
        env=env,
        capture_output=True,
    )


def _build_skills_tar(skills_src: Path) -> bytes:
    """Tar the contents of ``skills_src`` (entries relative to it), excluding ``.DS_Store``.

    Mirrors ``tar -C "$skills_src" -cf - --exclude=.DS_Store .``: entries land
    at ``<skill_name>/...`` inside the archive, not ``<skills_src>/<skill_name>/...``.
    An empty/missing ``skills_src`` produces a valid, empty tar stream.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        if skills_src.is_dir():
            # Sort by path parts (not the path string) so a directory always
            # sorts before its own children regardless of naming, which tar
            # extraction order requires.
            for path in sorted(
                skills_src.rglob("*"), key=lambda p: p.relative_to(skills_src).parts
            ):
                if path.name == ".DS_Store":
                    continue
                arcname = path.relative_to(skills_src).as_posix()
                tf.add(path, arcname=arcname, recursive=False)
    return buf.getvalue()
