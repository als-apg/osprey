"""Minimal ``.env`` file parsing shared across the CLI and deployment layers.

A tiny dependency-free parser (no ``python-dotenv`` import) used where we only
need to read the current ``KEY=VALUE`` pairs from a project ``.env`` — e.g. the
build lifecycle env injection and the deploy-time dispatch-token bootstrap.
"""

from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def parse_dotenv_file(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file into a dict of environment variables.

    Handles ``KEY=VALUE`` lines, ``#`` comments, blank lines, an optional
    ``export`` prefix, and quoted values (single or double quotes stripped from
    the value boundaries).
    """
    return parse_dotenv_text(path.read_text(encoding="utf-8"))


def parse_dotenv_text(text: str) -> dict[str, str]:
    """Parse ``.env`` *text* into a dict of environment variables.

    The in-memory half of :func:`parse_dotenv_file`, for callers holding the
    file's contents already (a locked read-modify-write, a rendered template).
    """
    env: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Skip lines without =
        if "=" not in line:
            continue
        # Skip `export` prefix (common in .env files)
        if line.startswith("export "):
            line = line[7:]
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        # Strip matching surrounding quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        env[key] = value
    return env


def _dotenv_raw_lines(text: str) -> dict[str, str]:
    """Map ``KEY`` -> its raw ``KEY=VALUE`` line (quoting intact) from ``text``."""
    raw: dict[str, str] = {}
    for line in text.splitlines():
        key = _line_key(line)
        if key:
            raw[key] = line.strip()
    return raw


# Keys the build derives from the project's own content rather than from the
# user's environment, and therefore owns outright: preserving them across a
# re-render would latch them on. A build that can no longer generate a
# virtual-accelerator manifest must be able to un-write the keys pointing at
# one -- keeping a stale VA_CHANNELS_FILE would leave the IOC serving a
# manifest with no drive limits beside it.
BUILD_DERIVED_KEYS = frozenset({"VA_CHANNELS_FILE", "VA_LATTICE"})


# Keys written into the *project* ``.env`` by something other than the build --
# a running scenario, a deploy, an operator arming a service -- and whose value
# is therefore pinned to live state the profile cannot reproduce: a docker
# volume initialized with a minted password, a container already trusting a
# token, the physics fault a scenario is currently applying. Re-deriving them
# from the profile would silently break the running stack, so a value already
# in the project ``.env`` always wins, and a key present ONLY there (a
# degraded-topology mint, when deploy could not write back to the profile)
# survives verbatim.
#
# Enumerated rather than imported: this module is the bottom of the import
# graph (``deployment.container_lifecycle`` imports *from* here). The names are
# cross-checked against their owning definitions in
# ``tests/utils/test_dotenv_derivation.py`` so the two cannot drift.
RUNTIME_WRITER_KEYS = frozenset(
    {
        # osprey.simulation.apply._PHYSICS_ENV_VARS -- the active scenario's
        # physics fault, reconciled per scenario switch.
        "VA_BPM_ERRORS",
        "VA_CORR_GAIN",
        # bluesky bridge substrate wiring, derived at deploy time
        # (osprey.services.bluesky_bridge.substrate_devices).
        "BLUESKY_EPICS_SUBSTRATE",
        "BLUESKY_EPICS_MOTORS",
        "BLUESKY_EPICS_DETECTORS",
        # osprey.deployment.container_lifecycle._SERVICE_TOKEN_VARS -- minted
        # once, then pinned by the volumes/containers that adopted them.
        "EVENT_DISPATCHER_TOKEN",
        "DISPATCH_WORKER_TOKEN",
        "BLUESKY_LAUNCH_TOKEN",
        "BLUESKY_TILED_API_KEY",
        "ZO_ROOT_USER_PASSWORD",
        "ARIEL_DB_PASSWORD",
    }
)

#: Banner introducing profile-carried keys the render itself did not emit.
PROFILE_CARRIED_BANNER = "# ── Carried from the profile .env ──"

#: Banner introducing runtime-written keys preserved from the project ``.env``.
RUNTIME_PRESERVED_BANNER = "# ── Preserved from the project .env (runtime-written) ──"


def derive_project_env(
    profile_text: str,
    rendered_text: str,
    existing_text: str,
    mode: str,
    *,
    build_derived_keys: frozenset[str] = BUILD_DERIVED_KEYS,
    runtime_writer_keys: frozenset[str] = RUNTIME_WRITER_KEYS,
) -> str:
    """Derive a project ``.env`` from the profile, the render, and what is there.

    The profile ``.env`` is the source of truth for facility secrets; the render
    supplies structure, comments, and build-computed values; the existing
    project ``.env`` contributes only what a runtime writer put there. Each key
    resolves through one of three classes, in this precedence:

    1. ``build_derived_keys`` -- the build owns them outright: the rendered
       value wins, and a key the render no longer emits is *un-written* rather
       than preserved (a stale ``VA_CHANNELS_FILE`` would point the IOC at a
       manifest the build can no longer regenerate).
    2. ``runtime_writer_keys`` -- an existing project value wins over both the
       profile and the render, whether or not the render emits the key, and a
       key present only in the project survives verbatim (see
       :data:`RUNTIME_WRITER_KEYS`). A profile value is therefore never carried
       in over one the project owns; when the two disagree, the divergence is
       warned about by name. An *empty* project value still owns the key: it
       means "no token", which the deploy layer treats as a deliberate
       fail-closed setting rather than a blank.
    3. profile-carried -- any key the profile ``.env`` sets wins over the
       render; profile keys the render does not emit are appended.

    Anything else falls back to the rendered value.

    ``mode`` selects how far the derivation goes:

    ``"build"``
        Full semantics. The render provides the file's shape and a key that
        survives in none of the three classes is *dropped* -- a project ``.env``
        rebuilt from a profile carries no facility state the profile cannot
        regenerate.
    ``"deploy"``
        Overlay only. The existing file provides the shape, profile-carried
        values are updated in place, missing keys are appended, and nothing is
        ever deleted -- a deploy refreshing secrets must not prune a project it
        did not build.
    """
    if mode not in ("build", "deploy"):
        raise ValueError(f"unknown derivation mode {mode!r} (expected 'build' or 'deploy')")

    profile = _dotenv_raw_lines(profile_text)
    existing = _dotenv_raw_lines(existing_text)
    rendered = _dotenv_raw_lines(rendered_text)

    if mode == "deploy":
        return _overlay_project_env(
            profile, rendered, existing_text, build_derived_keys, runtime_writer_keys
        )

    profile_values = parse_dotenv_text(profile_text)
    existing_values = parse_dotenv_text(existing_text)

    # The class-2 rule, decided ONCE. :func:`resolve` and the appended sections
    # below both read this set, so they cannot answer "does the project own
    # this key?" differently -- a divergence between those two answers is
    # exactly the seam the profile-carried append opened before.
    #
    # PRESENCE, not a non-empty value, is deliberately the test. An empty
    # class-2 value is a meaningful setting rather than a blank: an empty
    # service token means "no token", which
    # `container_lifecycle._ensure_service_tokens` explicitly declines to mint
    # over ("generating would silently override a deliberate value") and which
    # makes the service fail closed -- for an exposed deploy it refuses to bind
    # rather than fail open. Treating an empty line as absent would let a
    # rebuild re-arm a service the operator deliberately disarmed. Pinned by
    # test_an_empty_project_value_still_outranks_the_profile.
    project_owned = frozenset(key for key in runtime_writer_keys if key in existing)

    def resolve(key: str) -> str | None:
        """The line ``key`` should carry, or ``None`` if it is un-written."""
        if key in build_derived_keys:
            return rendered.get(key)
        if key in project_owned:
            return existing[key]
        if key in profile:
            return profile[key]
        return rendered.get(key)

    _warn_class2_divergence(project_owned, profile_values, existing_values, build_derived_keys)

    consumed: set[str] = set()
    out_lines: list[str] = []
    for line in rendered_text.splitlines():
        key = _line_key(line)
        if key is None:
            out_lines.append(line)
            continue
        consumed.add(key)
        resolved = resolve(key)
        if resolved is not None:
            out_lines.append(resolved)

    carried_keys = [
        key
        for key in profile
        if key not in consumed and key not in build_derived_keys and key not in project_owned
    ]
    if carried_keys:
        out_lines.extend(["", PROFILE_CARRIED_BANNER, *(profile[key] for key in carried_keys)])
    # Only what was actually carried. Marking every profile key consumed would
    # swallow the class-2 keys deliberately left out just above, and the
    # preserved section below -- the one place a project-pinned secret survives
    # a render that does not emit it -- would then drop them.
    consumed.update(carried_keys)

    preserved = [
        existing[key]
        for key in existing
        if key in project_owned and key not in consumed and key not in build_derived_keys
    ]
    if preserved:
        out_lines.extend(["", RUNTIME_PRESERVED_BANNER, *preserved])

    return "\n".join(out_lines) + "\n"


def _warn_class2_divergence(
    project_owned: frozenset[str],
    profile_values: Mapping[str, str],
    existing_values: Mapping[str, str],
    build_derived_keys: frozenset[str],
) -> None:
    """Warn for each owned key whose profile copy disagrees with the project's.

    Scoped to ``project_owned`` — the same set the derivation resolves with —
    so the warning fires exactly when the project's value is the one KEPT. A
    class-2 key the profile wins (because the project's line is empty, so the
    project does not own it) is an ordinary profile-carried value, not a
    conflict, and saying otherwise would train an operator to ignore the log.

    The disagreement itself is what they need told: it means the profile can no
    longer reproduce this project's secrets, so a project built from that
    profile elsewhere would come up on a value the running containers do not
    trust.

    Values are compared parsed, not as raw lines, so a difference in quoting
    alone is not a disagreement. Neither value is ever logged — which variable
    diverged is the information; what either side holds is a secret. This is the
    build-side twin of the write-back's own conflict warning
    (``osprey.deployment.container_lifecycle``); deploy mode does not come
    through here, because that path has already reported the same fact.
    """
    diverged = sorted(
        key
        for key in project_owned
        if key not in build_derived_keys
        and key in profile_values
        and profile_values[key] != existing_values[key]
    )
    if not diverged:
        return

    # Imported here rather than at module scope, and below the guard rather
    # than above it: this module is deliberately dependency-light so the layers
    # that import it (deployment, the build CLI) can do so from anywhere, and
    # the logger pulls in rich and the config loader. Every build runs this
    # function; almost none of them have anything to say, and those pay nothing.
    from osprey.utils.logger import get_logger

    logger = get_logger("build")
    for key in diverged:
        logger.warning(
            "  %s differs between this project's .env and the profile's. The project's "
            "value was kept — a runtime-written secret is pinned by the volumes and "
            "containers that adopted it, and this build must not replace it. The profile "
            "therefore cannot reproduce this project's secrets; reconcile the two by hand "
            "if they are meant to match.",
            key,
        )


def _overlay_project_env(
    profile: dict[str, str],
    rendered: dict[str, str],
    existing_text: str,
    build_derived_keys: frozenset[str],
    runtime_writer_keys: frozenset[str],
) -> str:
    """Deploy-mode derivation: update in place, append missing, never delete."""
    seen: set[str] = set()
    out_lines: list[str] = []
    for line in existing_text.splitlines():
        key = _line_key(line)
        if key is None:
            out_lines.append(line)
            continue
        seen.add(key)
        if key in build_derived_keys:
            out_lines.append(rendered.get(key, line.strip()))
        elif key in runtime_writer_keys:
            out_lines.append(line)
        elif key in profile:
            out_lines.append(profile[key])
        else:
            out_lines.append(line)

    appended = [profile[key] for key in profile if key not in seen]
    appended += [rendered[key] for key in rendered if key not in seen and key not in profile]
    if appended:
        # Unlike build mode, this derivation runs on its own previous output --
        # every deploy re-reads the file it last wrote -- so an unconditional
        # header stacks another banner each time the profile gains a key. Emit
        # it only when the file does not already carry one, the same guard
        # ``append_profile_env`` applies to its minted section.
        if PROFILE_CARRIED_BANNER not in out_lines:
            out_lines.extend(["", PROFILE_CARRIED_BANNER])
        out_lines.extend(appended)

    return "\n".join(out_lines) + "\n"


#: Section header the deploy write-back groups its minted secrets under.
DEPLOY_MINTED_BANNER = "# ── Minted by deploy ──"

#: Permission bits a ``.env`` this module creates is born with (and tightened
#: to on every rewrite): the profile ``.env`` holds facility secrets.
ENV_FILE_MODE = 0o600


@dataclass(frozen=True)
class EnvConflict:
    """A supplied entry that disagrees with the value already on file."""

    key: str
    existing: str
    supplied: str


@dataclass(frozen=True)
class ProfileEnvAppendResult:
    """What :func:`append_profile_env` did, for the caller to report on."""

    added: tuple[str, ...]
    unchanged: tuple[str, ...]
    conflicts: tuple[EnvConflict, ...]


def append_profile_env(
    profile_env_path: Path,
    entries: Mapping[str, str],
    section_banner: str = DEPLOY_MINTED_BANNER,
) -> ProfileEnvAppendResult:
    """Append ``entries`` to a profile ``.env`` -- atomically, and append-only.

    Used by ``deploy up`` to persist the secrets it minted back into the
    profile that owns them. **A key already in the file is never rewritten**: a
    minted password is pinned by the docker volume that was initialized with it
    and by every container already trusting it, so the value on file always
    wins. An entry whose value disagrees with the one on file is reported in
    :attr:`ProfileEnvAppendResult.conflicts` for the caller to warn about --
    the mismatch is real information (a shell export diverging from the volume's
    password), but resolving it is an operator's decision, not this function's.

    Concurrency: a sibling ``<name>.lock`` file serializes the whole
    read-modify-write across processes (``flock``), and the new contents are
    written to a temp file in the same directory and ``os.replace``d over the
    target, so a concurrent reader sees either the complete old file or the
    complete new one, never a partial write. The lock lives beside the ``.env``
    rather than on it precisely because ``os.replace`` swaps the inode out from
    under any lock held on the file itself.

    New keys go at the end of the file, under ``section_banner`` -- emitted
    only when that banner is not already present, so repeated appends grow one
    section instead of stacking headers. The file is created with mode
    ``0600`` when absent, and rewrites tighten an existing file to the same.
    The parent directory is *not* created: a missing profile directory raises,
    which is the signal the caller's degraded path is looking for.
    """
    lock_path = profile_env_path.with_name(profile_env_path.name + ".lock")
    with open(lock_path, "a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            return _append_profile_env_locked(profile_env_path, entries, section_banner)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _append_profile_env_locked(
    profile_env_path: Path,
    entries: Mapping[str, str],
    section_banner: str,
) -> ProfileEnvAppendResult:
    """The read-modify-write half of :func:`append_profile_env`, under lock."""
    existing_text = (
        profile_env_path.read_text(encoding="utf-8") if profile_env_path.is_file() else ""
    )
    on_file = parse_dotenv_text(existing_text)

    added: list[str] = []
    unchanged: list[str] = []
    conflicts: list[EnvConflict] = []
    new_lines: list[str] = []
    for key, value in entries.items():
        if key in on_file:
            if on_file[key] == value:
                unchanged.append(key)
            else:
                conflicts.append(EnvConflict(key=key, existing=on_file[key], supplied=value))
            continue
        added.append(key)
        new_lines.append(_format_env_line(key, value))

    if new_lines:
        lines = existing_text.splitlines()
        if section_banner and section_banner not in lines:
            if lines:
                lines.append("")
            lines.append(section_banner)
        lines.extend(new_lines)
        atomic_write(profile_env_path, "\n".join(lines) + "\n")

    return ProfileEnvAppendResult(
        added=tuple(added), unchanged=tuple(unchanged), conflicts=tuple(conflicts)
    )


def atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a same-directory temp file + ``os.replace``.

    Every writer of a ``.env`` this package owns goes through here. These files
    are rewritten whole, so a crash mid-write would truncate an operator's
    secrets; the replace is atomic and the temp file is a sibling, keeping the
    rename inside one filesystem. The result is always
    :data:`ENV_FILE_MODE` — a ``.env`` holds facility secrets whether it is
    being created or refreshed.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(tmp_name, ENV_FILE_MODE)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _format_env_line(key: str, value: str) -> str:
    """Render ``KEY=VALUE`` so :func:`parse_dotenv_text` reads back ``value``."""
    if "\n" in value or "\r" in value:
        raise ValueError(f"cannot write {key} to a .env file: the value contains a newline")
    needs_quoting = value != value.strip() or any(char in value for char in (" ", "\t", "#"))
    if not needs_quoting:
        return f"{key}={value}"
    if '"' not in value:
        return f'{key}="{value}"'
    if "'" not in value:
        return f"{key}='{value}'"
    raise ValueError(
        f"cannot write {key} to a .env file: the value needs quoting but contains "
        "both single and double quotes"
    )


def _line_key(line: str) -> str | None:
    """The ``KEY`` a raw ``.env`` line assigns, or ``None`` for comments/blanks."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    candidate = stripped[7:] if stripped.startswith("export ") else stripped
    key = candidate.partition("=")[0].strip()
    return key or None


def merge_env_preserving_existing(
    rendered_text: str,
    existing_text: str,
    *,
    build_derived_keys: frozenset[str] = BUILD_DERIVED_KEYS,
) -> str:
    """Merge a freshly rendered ``.env`` with an existing one; existing wins.

    Used when a build re-renders a project in place (``osprey build --force``)
    or a profile ships a template ``.env``: the rendered text provides the
    structure, comments, and any newly introduced variables, while every value
    the user already has keeps its existing setting (their secrets, and the
    service tokens/passwords that live containers and docker volumes were
    initialized with). Keys present only in the existing file are appended at
    the end so nothing the user set is ever dropped.

    ``build_derived_keys`` are the exception in both directions: the rendered
    value wins, and a key the rendered text no longer carries is dropped
    instead of preserved. Pass an empty set for a merge whose rendered side is
    a fragment rather than the build's own full render.
    """
    existing = _dotenv_raw_lines(existing_text)
    for key in build_derived_keys:
        existing.pop(key, None)
    consumed: set[str] = set()
    out_lines: list[str] = []
    for line in rendered_text.splitlines():
        key = _line_key(line)
        if key is not None and key in existing:
            out_lines.append(existing[key])
            consumed.add(key)
            continue
        out_lines.append(line)
    leftovers = [existing[key] for key in existing if key not in consumed]
    if leftovers:
        out_lines.extend(["", "# Preserved from existing .env"])
        out_lines.extend(leftovers)
    return "\n".join(out_lines) + "\n"
