"""What the deployment-reading commands share: the build they act on and the operator login.

``osprey feedback``, ``osprey agent-record`` and ``osprey users`` all act on one
deployment repo through its rendered ``build/config.yml``. ``feedback`` reads the
agent record with the operator login the build minted, and ``agent-record`` does
too whenever ``OSPREY_AGENT_RECORD_URL`` names the record. The lookups here only
report a miss with its fix and return ``None``; how the command then stops, and
whether a verb needs the login at all, stays with the command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

from osprey.cli.output import fail
from osprey.cli.repo_resolver import find_repo_root


class Deployment(NamedTuple):
    """The deployment one verb reads: its repo root and its rendered config."""

    root: Path
    config: dict[str, Any]


def find_build(repo: Path | None, purpose: str) -> tuple[Path, Path] | None:
    """The deployment repo root and its rendered config path, when it has a build.

    :param repo: Value of ``--repo``, or ``None`` to walk up from the working
        directory.
    :param purpose: The sentence the refusal adds after naming the missing
        config, saying what the command reads through it.
    :returns: ``(repo_root, config_path)``, or ``None`` once the missing build
        has been reported.
    :raises click.ClickException: If no deployment repo encloses the search start.
    """
    from osprey_connectors.workspace import rendered_config_path

    repo_root = find_repo_root(repo)
    config_path = rendered_config_path(repo_root)
    if not config_path.is_file():
        fail(
            f"No build found in {repo_root}",
            f"{config_path} does not exist. {purpose}",
            "run `osprey build` first",
        )
        return None
    return repo_root, config_path


def operator_login(repo_root: Path) -> tuple[str, str] | None:
    """The operator login and its token, from the repo's build-minted login file.

    :returns: ``(login, token)``, or ``None`` once the missing login has been
        reported.
    """
    from osprey.agent_record import credentials as creds
    from osprey.agent_record import identities as ids

    login = ids.login_for(ids.OPERATOR_IDENTITY)
    path = repo_root / creds.login_file(login)
    token = creds.read_login(path, login)
    if token is None:
        fail(
            "No operator login for the agent record",
            f"{path} is missing or holds no {login} token.",
            "run `osprey build` first",
        )
        return None
    return login, token
