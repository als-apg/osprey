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

#: The committed, non-secret defaults half of the env chain, at the repo root.
ENV_SHARED_FILENAME = ".env.shared"

#: The host-local half of the env chain, at the repo root: the deployment's one
#: secret store, hand-edited and written back to by ``osprey up``.
ENV_LOCAL_FILENAME = ".env"

#: The env chain in ASCENDING precedence order: every mechanism that delivers
#: these files must end up with ``.env`` winning over ``.env.shared`` on a key
#: both set. Spelled here, once, because the mechanics of "later wins" differ
#: per delivery path — a compose ``env_file:`` list is read in this order, an
#: ``override=True`` loader loads in this order, and an ``override=False``
#: (first-loaded-wins) loader must load it REVERSED to reach the same answer.
#: Membership is fixed: a chain member is a file the deployment knows by name,
#: not a directory scan.
ENV_CHAIN_FILENAMES: tuple[str, ...] = (ENV_SHARED_FILENAME, ENV_LOCAL_FILENAME)

#: Header written at the top of a merged env file (:func:`write_env_merged`).
#: The file is a machine artifact — regenerated from the chain on every write —
#: so whoever opens it is told where the editable originals are.
ENV_MERGED_BANNER = (
    "# Generated by osprey — do not edit.\n"
    f"# Merged from {' + '.join(ENV_CHAIN_FILENAMES)} (later file wins); edit those instead.\n"
)

# ---------------------------------------------------------------------------
# Readme headers, one per generated env file. THE registry: every env file
# OSPREY creates is stamped with its header at creation (and only then — the
# writers are append-only, so an existing file's opening lines are the
# operator's). Defined together so whoever opens any of these files gets the
# same story about how they relate, and so a new env file cannot ship without
# saying who reads it.
# ---------------------------------------------------------------------------

#: Header stamped when :func:`append_profile_env` CREATES the repo ``.env``.
#: Existing files are never rewritten (the write is deliberately append-only),
#: so a hand-authored ``.env`` keeps its own opening lines.
ENV_LOCAL_BANNER = (
    "# OSPREY deployment secrets (.env) — host-local, gitignored, mode 0600.\n"
    "# Read by compose on every lifecycle verb (--env-file, after .env.shared,\n"
    "# so on a key both files set this file wins) and appended to by `osprey up`\n"
    "# when it mints a secret. Belongs here: provider API keys, service\n"
    "# passwords/tokens, plaintext OSPREY_AUTH_PW_<USER> convenience passwords\n"
    "# (hashed into .env.auth, never shipped), and per-host overrides of the\n"
    "# committed defaults. Related files: .env.shared holds the committed,\n"
    "# non-secret defaults; build/.env.merged is the generated single-file\n"
    "# collapse handed to podman-compose; .env.example documents the variables.\n"
)

#: Header stamped at the top of every rendered ``.env.users`` — the deploy
#: generator and ``osprey users env`` share one render helper carrying it
#: (``osprey.deployment.web_terminals.env_production.render_env_users``).
ENV_USERS_BANNER = (
    "# OSPREY web-terminal runtime env (.env.users) — gitignored, mode 0600.\n"
    "# Handed whole to every per-user web-terminal container via the\n"
    "# `env_file: .env.users` entries in docker-compose.web.yml, baked in when\n"
    "# a container is created. Belongs here: the runtime subset the terminals\n"
    "# need — LLM provider credentials and module runtime vars — never\n"
    "# build/CI-only variables, and never auth secrets (.env.auth is the auth\n"
    "# sidecar's file). Derived from .env.shared + .env (later file wins) by\n"
    "# `osprey up` or `osprey users env`; a deploy never overwrites an existing\n"
    "# file. .env.example documents the variables themselves.\n"
)

#: Header stamped when the auth-credential writer CREATES ``.env.auth``
#: (``osprey.deployment.web_terminals.auth_credentials._append_entries``).
ENV_AUTH_BANNER = (
    "# OSPREY web-terminal auth secrets (.env.auth) — gitignored, mode 0600.\n"
    "# Read ONLY by the auth sidecar, via its `env_file: .env.auth` entry in\n"
    "# docker-compose.web.yml — baked in at container creation, so a change\n"
    "# takes effect at the next sidecar recreate; no per-user terminal ever\n"
    "# receives this file. Belongs here: the OSPREY_AUTH_* variables — per-user\n"
    "# password hashes (OSPREY_AUTH_PW_HASH_<USER>), the cookie-signing\n"
    "# secrets, and a hand-added OIDC client secret. A plaintext\n"
    "# OSPREY_AUTH_PW_<USER> convenience password goes in .env instead and is\n"
    "# hashed in from there; .env.example documents the variables.\n"
)


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
    section instead of stacking headers. A file created here (or one found
    empty) is stamped with :data:`ENV_LOCAL_BANNER` first — creation is the
    only moment the readme header can be added, since an existing file's
    opening lines are the operator's. The file is created with mode
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
        new_lines.append(format_env_line(key, value))

    if new_lines:
        lines = existing_text.splitlines()
        if not lines:
            # Born carrying the readme header: this write is creating the file
            # (or filling an empty one), and the append-only contract means no
            # later write may touch the top of it.
            lines = ENV_LOCAL_BANNER.splitlines()
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


def format_env_line(key: str, value: str) -> str:
    """Render ``KEY=VALUE`` so :func:`parse_dotenv_text` reads back ``value``.

    THE quoting helper: every writer of a ``.env``-shaped file in this package
    renders its lines through here, so a value with leading whitespace, an
    inline ``#``, or an embedded space survives the round-trip through
    :func:`parse_dotenv_text` (and through the compose implementations that
    read the same file) instead of being silently truncated.

    :raises ValueError: when the value cannot be represented on one ``.env``
        line — it contains a newline, or it needs quoting and already contains
        both quote characters.
    """
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


def chain_files(repo_root: Path) -> list[Path]:
    """The env-chain files that exist under ``repo_root``, ascending precedence.

    Returns the existing members of :data:`ENV_CHAIN_FILENAMES` in that order —
    ``.env.shared`` before ``.env`` — so a caller can hand the list straight to
    whatever consumes it (a compose ``env_file:`` list, an ``override=True``
    loader, a ``$``-scan) and get local-over-shared precedence from the order
    alone. Missing members are skipped rather than reported: a project with
    neither file, only defaults, or only local settings are all valid shapes,
    and the list is empty/short accordingly.

    :param repo_root: Directory the chain lives in (the deployment repo root).
    :return: Existing chain files, lowest precedence first.
    """
    root = Path(repo_root)
    return [path for path in (root / name for name in ENV_CHAIN_FILENAMES) if path.is_file()]


def merge_chain(repo_root: Path) -> dict[str, str]:
    """Parse the env chain under ``repo_root`` into one dict; later file wins.

    The chain is read in :func:`chain_files` order, each file's entries laid
    over the previous file's, so a key set in both ``.env.shared`` and ``.env``
    ends up with the ``.env`` value — the single in-process answer to "what
    does this deployment's env chain say?".

    Key order in the result is first-seen: the shared defaults in file order,
    then keys only the local file introduces. A key both files set keeps its
    shared-file position and takes the local value, so the merged view reads
    like the defaults file with the local overrides applied in place.

    Nothing is written to :data:`os.environ` here — the caller decides whether
    these values overlay a subprocess environment, a compose file, or nothing
    at all.

    :param repo_root: Directory the chain lives in (the deployment repo root).
    :return: Merged ``KEY=VALUE`` pairs; empty when no chain file exists.
    """
    merged: dict[str, str] = {}
    for path in chain_files(repo_root):
        merged.update(parse_dotenv_file(path))
    return merged


def write_env_merged(repo_root: Path, dest: Path) -> Path:
    """Write the merged env chain under ``repo_root`` to ``dest`` as one file.

    For the delivery mechanisms that can only be handed ONE env file: the
    merge happens here, ahead of them, so they still see local-over-shared
    precedence. Values are rendered through :func:`format_env_line` (so a
    quoted or space-carrying value survives whatever parser reads the result)
    and the file is written through :func:`atomic_write`, which lands it whole
    and at :data:`ENV_FILE_MODE` (``0600``) — the merged file inherits the
    local half's secrets, so it inherits its permissions too.

    The file is a machine artifact carrying :data:`ENV_MERGED_BANNER`, is
    rewritten from the chain on every call, and is never a place to edit a
    value. ``dest``'s parent directory is created when missing.

    :param repo_root: Directory the chain lives in (the deployment repo root).
    :param dest: Path to write the merged file to.
    :return: ``dest``, for the caller to pass on.
    :raises ValueError: when a value cannot be written to a ``.env`` line (see
        :func:`format_env_line`).
    """
    merged = merge_chain(repo_root)
    lines = [format_env_line(key, value) for key, value in merged.items()]
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(dest, ENV_MERGED_BANNER + "".join(f"{line}\n" for line in lines))
    return dest


def compose_unsafe_vars_in_chain(repo_root: Path) -> list[str]:
    """:func:`compose_unsafe_vars` across every existing env-chain file.

    Each chain member is scanned on its own rather than the merged result: a
    compose ``env_file:`` list reads every file it is given, so a ``$`` in the
    shared defaults is delivered through the same interpolating parsers as a
    ``$`` in the local file, whether or not the local file happens to override
    that key.

    Returns variable *names*, sorted and de-duplicated across the chain —
    deliberately never the values, so no caller can render a secret into an
    error message or a log line.

    :param repo_root: Directory the chain lives in (the deployment repo root).
    :return: Sorted names of the offending variables; empty when all are safe.
    """
    offenders: set[str] = set()
    for path in chain_files(repo_root):
        offenders.update(compose_unsafe_vars(parse_dotenv_file(path)))
    return sorted(offenders)
