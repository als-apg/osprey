"""The deployment's env chain: which files it is, and how they are read.

A tiny dependency-free parser (no ``python-dotenv`` import) and the chain it
reads — ``.env.shared`` then ``.env`` at the repo root, later file wins. Every
OSPREY process loads that chain on start-up
(:func:`osprey_connectors.config.load_project_dotenv`), so the reading half
lives here. Writing the files is the framework's deployment layer's
(``osprey.deployment.dotenv``).
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "ENV_CHAIN_APPLIED_ENV",
    "ENV_CHAIN_FILENAMES",
    "ENV_LOCAL_FILENAME",
    "ENV_SHARED_FILENAME",
    "chain_files",
    "merge_chain",
    "parse_dotenv_file",
    "parse_dotenv_text",
]


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
