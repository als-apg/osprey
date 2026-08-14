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


def compose_unsafe_vars(values: Mapping[str, str]) -> list[str]:
    """Names of vars whose value a compose ``env_file:`` would not deliver intact.

    Compose implementations disagree about what they do to an ``env_file:``
    value, and the deploy does not get to choose which one reads the file:

    * Docker Compose (measured on v2.34) interpolates the *values*, not just
      the compose document. A ``$`` followed by an identifier character is
      replaced with the named variable's value — the empty string when that
      name is unset — so the container receives a truncated secret while the
      file on disk stays byte-perfect. Two forms carry no warning whatsoever:
      ``$$`` collapses to a single ``$``, and a ``$`` followed by a name that
      *is* set on the deploy host splices the host's value into the secret.
    * podman-compose hands the file to python-dotenv, which resolves only the
      braced ``${...}`` form — and resolves it against an earlier entry in the
      SAME file ahead of the host environment, so the identical file can yield
      a different secret than Docker Compose does.
    * Older podman-compose passes ``--env-file`` to podman, whose parser
      substitutes nothing at all.

    The test is therefore "contains ``$``", not "contains an interpolating
    ``$``". Escaping cannot bridge the set: ``$$`` yields ``$`` where values are
    interpolated and a literal ``$$`` where they are not, so no single file is
    correct everywhere. Which forms count as a reference is compose's business
    and may widen. Refusing ``$`` outright is the only rule that stays true —
    and the one the codebase already designs to
    (``auth_sidecar.passwords.FIELD_SEP``, the minted alphabets in
    :mod:`osprey.deployment.service_tokens`).

    Returns variable *names*, sorted — deliberately never the values, so that no
    caller can accidentally render a secret into an error message or a log line.

    :param values: Parsed ``KEY=VALUE`` pairs bound for a compose ``env_file:``.
    :return: Sorted names of the offending variables; empty when all are safe.
    """
    return sorted(name for name, value in values.items() if "$" in value)


def dotenv_line_var(line: str) -> str | None:
    """The variable a raw ``.env`` line assigns, or ``None`` for a non-assignment.

    Comments, blank lines, and anything without an ``=`` yield ``None``; an
    optional ``export`` prefix is stripped, matching :func:`parse_dotenv_file`.
    Callers that need to rewrite a ``.env`` line-by-line (rather than parse it
    into a dict) use this so they classify lines exactly the way the parser
    does — a divergence would let a rewrite keep a line the parser still reads.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    candidate = stripped[7:] if stripped.startswith("export ") else stripped
    return candidate.partition("=")[0].strip() or None


def _dotenv_raw_lines(text: str) -> dict[str, str]:
    """Map ``KEY`` -> its raw ``KEY=VALUE`` line (quoting intact) from ``text``."""
    raw: dict[str, str] = {}
    for line in text.splitlines():
        key = dotenv_line_var(line)
        if key:
            raw[key] = line.strip()
    return raw


# Keys whose value the build derives from the project's own content rather than
# from the user's environment: the pointers at the virtual-accelerator channel
# manifest `osprey build` generates into the output zone. They are named here,
# once, because two places need the same answer to "which keys does the build
# speak for?" -- the build appends them when it generates a manifest, and it
# scans for them to report a pointer left over from a build that no longer can
# (`osprey.cli.build_cmd`). Their VALUES are not here: they come from the
# virtual-accelerator manifest module, and this module is deliberately the
# bottom of the import graph.
#
# Owning a key does not mean overwriting it. The repo `.env` is the
# deployment's one secret store -- hand-edited, and written back to by
# `osprey up` -- so the build appends through `append_profile_env` like every
# other writer of that file, and a value already on file always wins.
BUILD_DERIVED_KEYS = frozenset({"VA_CHANNELS_FILE", "VA_LATTICE"})


#: Section header the deploy write-back groups its minted secrets under.
DEPLOY_MINTED_BANNER = "# ── Minted by deploy ──"

#: Section header the build groups :data:`BUILD_DERIVED_KEYS` under. A separate
#: banner from the deploy's, because the two sections answer different questions
#: for whoever opens the file: one holds secrets no rebuild can reproduce, this
#: one holds pointers at artifacts in ``build/`` that every build regenerates.
BUILD_DERIVED_BANNER = "# ── Derived by build ──"

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

    Used by ``osprey up`` to persist the secrets it minted back into the
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


def merge_env_preserving_existing(
    rendered_text: str,
    existing_text: str,
    *,
    build_derived_keys: frozenset[str] = BUILD_DERIVED_KEYS,
) -> str:
    """Merge a freshly rendered ``.env`` with an existing one; existing wins.

    Used when a build re-renders a project in place
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
        key = dotenv_line_var(line)
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
