"""The deployment's env chain: which files it is, how they are read, and how it is applied.

The chain is ``.env.shared`` then ``.env`` at the repo root, later file wins.
Every OSPREY process applies it on start-up (:func:`load_project_dotenv`), and
the files of the chain are read for that with python-dotenv's ``dotenv_values``
alone: what the start-up writes into the environment and what it records of the
shell values it replaced come from one parse, so the two can never disagree on
an inline comment, an escape or a multi-line value.

:func:`parse_dotenv_file` and :func:`parse_dotenv_text` are a separate, tiny
dependency-free parser that the framework's ``.env`` file editors use, and
:func:`merge_chain` reads the chain with it for the framework's deploy-time
readers; writing the files is the framework's deployment layer's
(``osprey.deployment.dotenv``).
"""

from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping, Sequence
from pathlib import Path

__all__ = [
    "ENV_CHAIN_APPLIED_ENV",
    "ENV_CHAIN_FILENAMES",
    "ENV_LOCAL_FILENAME",
    "ENV_SHARED_FILENAME",
    "apply_env_chain",
    "chain_files",
    "dotenv_shell_overrides",
    "load_project_dotenv",
    "merge_chain",
    "parse_dotenv_file",
    "parse_dotenv_text",
]

# The configuration loader's logger, so ``quiet_logger(["CONFIG"])`` quiets the
# chain's start-up lines with the load's own.
logger = logging.getLogger("CONFIG")


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
        line = line.removeprefix("export ")
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

#: Set by a parent that already applied the env chain and hands its child a
#: resolved environment it chose. A process that carries it does not load the
#: chain again: reloading would put back what the parent withheld.
ENV_CHAIN_APPLIED_ENV = "OSPREY_ENV_CHAIN_APPLIED"


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


#: Variables the env chain never sets in a process environment. Each one is a
#: name that chooses which program runs, what code a runtime loads at start, or
#: what code an installer fetches, so it comes from the operator's shell, never
#: from a repo file. Names compare case-insensitively; the entries here and in
#: :data:`_SHELL_ONLY_ENV_PREFIXES` are upper-case.
_SHELL_ONLY_ENV_VARS: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "SHELL",
        "ENV",
        "BASH_ENV",
        "ZDOTDIR",
        "NODE_OPTIONS",
        "NODE_PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "PYTHONWARNINGS",
        "PYTHONINSPECT",
        "PYTHONBREAKPOINT",
        "PYTHONEXECUTABLE",
        "PYTHONPLATLIBDIR",
        "PYTHONPYCACHEPREFIX",
        "NODE_REPL_EXTERNAL_MODULE",
        "PERL5LIB",
        "PERLLIB",
        "PERL5OPT",
        "PERL5DB",
        "RUBYOPT",
        "RUBYLIB",
        "JAVA_TOOL_OPTIONS",
        "_JAVA_OPTIONS",
        "JDK_JAVA_OPTIONS",
        "CLASSPATH",
        "GCONV_PATH",
        "GLIBC_TUNABLES",
        "LOCPATH",
        "NLSPATH",
        "HOSTALIASES",
        "OPENSSL_CONF",
        "OPENSSL_MODULES",
        "OPENSSL_ENGINES",
        "XDG_CONFIG_HOME",
        "XDG_CONFIG_DIRS",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "PROMPT_COMMAND",
        "PS4",
        "IFS",
        "SHELLOPTS",
        "BASHOPTS",
        "CC",
        "CXX",
        "LDSHARED",
        "CFLAGS",
        "CPPFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "ARCHFLAGS",
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "LIBRARY_PATH",
        "COMPILER_PATH",
        "GCC_EXEC_PREFIX",
        "DOCKER_HOST",
        "DOCKER_CONFIG",
        "DOCKER_CONTEXT",
        "CLAUDE_CONFIG_DIR",
        "CLAUDE_CODE_SHELL",
        "CLAUDE_CODE_SHELL_PREFIX",
        "CLAUDE_CODE_GIT_BASH_PATH",
        "CLAUDE_ENV_FILE",
        "SSH_ASKPASS",
        "SUDO_ASKPASS",
        "LESSOPEN",
        "LESSCLOSE",
        "PAGER",
        "MANPAGER",
        "EDITOR",
        "VISUAL",
        "BROWSER",
    }
)

#: Name prefixes of the dynamic loader's variables, bash's exported functions
#: and the variables git, npm, uv, pip and compose read, shell-only for the same reason
#: as :data:`_SHELL_ONLY_ENV_VARS`.
_SHELL_ONLY_ENV_PREFIXES: tuple[str, ...] = (
    "LD_",
    "DYLD_",
    "BASH_FUNC_",
    "GIT_",
    "NPM_CONFIG_",
    "UV_",
    "PIP_",
    "COMPOSE_",
)

#: ``(resolved file, name)`` pairs whose ignored shell-only name was already
#: reported, so a chain applied many times in one process warns once.
_shell_only_reported: set[tuple[str, str]] = set()


def _is_shell_only_env_var(name: str) -> bool:
    """Whether ``name`` is a launch variable the env chain never applies, in any case."""
    upper = name.upper()
    return upper in _SHELL_ONLY_ENV_VARS or upper.startswith(_SHELL_ONLY_ENV_PREFIXES)


#: Shell values :func:`load_project_dotenv`'s ``override=True`` replaced, keyed
#: by variable name. The entry-time passthrough makes ``.env`` win inside the
#: osprey *process*; compose's own precedence is the opposite (a shell export
#: beats ``--env-file``), so the deploy path needs the shell as it really was —
#: both to warn about a divergent export and to hand compose an environment
#: that honors it. Empty when nothing differing was overridden.
_shell_overrides: dict[str, str] = {}


def apply_env_chain(
    environ: MutableMapping[str, str],
    paths: Sequence[Path],
    *,
    override: bool = True,
) -> list[str]:
    """Apply env-chain files to ``environ``, except the shell-only launch variables.

    ``paths`` is in ascending precedence (``.env.shared`` before ``.env``); a
    path that is not a file is skipped. Each file is read with python-dotenv's
    ``dotenv_values`` and written into ``environ`` in turn, so a later file
    wins on a key both set. Every key is applied except those
    :func:`_is_shell_only_env_var` names: for those ``environ`` keeps exactly
    what it held (set, or absent), and one warning per file names the ignored
    keys — never their values.

    With ``override=False`` a name ``environ`` held when the call began is
    never written, while a later file may still replace a value this call
    wrote, so the local file outranks the shared one and the shell outranks
    both.

    :return: The sorted names written into ``environ``.
    """
    from dotenv import dotenv_values

    preset = set() if override else set(environ)
    written: set[str] = set()
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        ignored: list[str] = []
        for name, value in dotenv_values(path).items():
            if value is None:
                continue
            if _is_shell_only_env_var(name):
                ignored.append(name)
                continue
            if name in preset:
                continue
            environ[name] = value
            written.add(name)
        _report_shell_only(path, ignored)
    return sorted(written)


def _report_shell_only(path: Path, names: list[str]) -> None:
    """Warn once per process about the shell-only names ``path`` sets."""
    resolved = str(path.resolve())
    fresh = sorted({name for name in names if (resolved, name) not in _shell_only_reported})
    if not fresh:
        return
    _shell_only_reported.update((resolved, name) for name in fresh)
    logger.warning(
        "Ignored %s in %s: launch variables come from the shell only. "
        "Export them in your shell instead.",
        ", ".join(fresh),
        path,
    )


def dotenv_shell_overrides() -> dict[str, str]:
    """The shell's own values for variables the ``.env`` entry-load overrode.

    Only variables whose exported value *differed* from the one the chain wrote
    are recorded: a key the shell never set, or set to the same value, was not
    shadowed. Overlaying this onto ``os.environ`` reconstructs the environment
    the operator's shell actually provided, which is what compose interpolation
    is documented against.
    """
    return dict(_shell_overrides)


def load_project_dotenv() -> None:
    """Load the cwd's env chain into ``os.environ``, overriding existing values.

    This is the env → environ passthrough the framework depends on: it
    feeds the ``${VAR}`` references Claude Code expands in ``.mcp.json`` at MCP
    server launch time (``EPICS_CA_ADDR_LIST``, ``PHOEBUS_BRIDGE_URL``,
    ``BLUESKY_*``), and it makes the chain the source of truth for API keys over
    a stale shell export. Every key in the files is passed through, not a
    declared subset — narrowing it would drop Channel Access addressing with no
    error. The one exception is the shell-only launch variables
    (:func:`_is_shell_only_env_var`): :func:`apply_env_chain` leaves them as the
    shell set them and warns.

    The chain is ``.env.shared`` then ``.env`` (:data:`ENV_CHAIN_FILENAMES`),
    each applied with override, so the host-local ``.env`` wins over the
    committed defaults on a key both set — the same local-wins precedence every
    other delivery path applies. A project with no ``.env.shared`` behaves
    exactly as it did when ``.env`` was the whole chain.

    What the override replaces is not discarded: the shell's own differing
    values are recorded (:func:`dotenv_shell_overrides`) so the deploy path can
    still see — and warn about — an export that disagrees with the store, and
    hand compose an environment with the shell's precedence intact. The record
    is the write's own: a name is recorded when :func:`apply_env_chain` wrote
    it over a shell value that differs from the value it ended with, so a key
    both files set is judged once, against the local file's value, and the
    record and the environment come from the same parse.

    ``override=True`` and the breadth of the copy are exactly why this must be
    called deliberately. Call it from process entry points only; never at
    import time, and never from library code that a host application merely
    imports. Missing files and an unreadable file are non-fatal.

    A process stamped with :data:`ENV_CHAIN_APPLIED_ENV` was handed its
    environment by a parent that chose it, and this returns without loading
    anything: reloading the chain would put back what the parent withheld.
    """
    if os.environ.get(ENV_CHAIN_APPLIED_ENV):
        return

    repo_root = Path.cwd()
    try:
        paths = chain_files(repo_root)
        if not paths:
            logger.debug(f"No .env file found at {repo_root}")
            return

        # Accumulate, never clear: this runs more than once per process, and
        # after the first load os.environ already matches the chain — a later
        # call sees no difference and must not erase the genuine shell values
        # the first one recorded. First-seen wins; only the pre-load value is
        # the shell's own. A shell-only name is never written, so it is never
        # recorded.
        before = dict(os.environ)
        for name in apply_env_chain(os.environ, paths):
            shell = before.get(name)
            if shell is not None and shell != os.environ[name]:
                _shell_overrides.setdefault(name, shell)
        logger.debug("Loaded the env chain from %s", ", ".join(str(path) for path in paths))
    except OSError as e:
        # e.g. a 0600 .env owned by another uid mounted into a non-root
        # container (dispatch worker on a uid-mismatched host). Provider
        # env should already be in os.environ by the time config is built,
        # so degrade gracefully instead of crash-looping the process.
        logger.warning(f"Could not read the .env chain at {repo_root}: {e}")


def _clear_process_state() -> None:
    """Forget the shell-override record and the shell-only warnings (see ``testing.clear_config_state``)."""
    _shell_only_reported.clear()
    _shell_overrides.clear()
