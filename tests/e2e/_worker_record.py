"""A real agent-record service, in a thread, for a dispatch worker run outside a stack.

The deployed worker records every run in the agent record: its ``run`` lines go
to its own spool directory, the run's facts come from its transcript, and every
read of a run that is not in flight (its terminal status, the dashboard's run
list) is a read of the record under the worker's own login. A lane that runs
the worker as a host subprocess therefore hands it the same three things the
compose template does: the record's URL with the worker's login, its spool
directory, and a Claude config directory whose ``projects/`` is the transcript
root the record tails. A bridge beside it reads its conversation history from
the same record under its own login.

The worker runs its agent with ``CLAUDE_CONFIG_DIR`` on the repo's agent-data
zone (:func:`~osprey.agent_runner.artifact_resolve.dispatch_claude_config_dir`),
where compose mounts the worker's transcripts volume over ``projects/``; here
that ``projects/`` is a link to the transcript root the record tails.

The service is the shipped one (:func:`osprey.agent_record.service.create_app`)
over an embedded backend, served by :func:`run_app_server` on a free port.
"""

from __future__ import annotations

import base64
import contextlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from osprey.agent_record import credentials as record_creds
from osprey.agent_record import identities as record_ids
from osprey.agent_record import service as record_service
from osprey.agent_record.backends.embedded import EmbeddedBackend
from osprey.agent_record.client import URL_ENV
from osprey.agent_record.contract import SPOOL_DIR_ENV
from osprey.interfaces._serving import run_app_server
from tests.agent_record.test_identities import _config as record_config

#: The identity the one worker these lanes run records under.
WORKER_IDENTITY = "dispatch-worker-1"


@dataclass(frozen=True)
class WorkerRecord:
    """A running record, and the environment each of its sources is started with."""

    url: str
    spool_root: Path
    tokens: Mapping[str, str]

    def login(self, identity: str) -> str:
        """``<login>:<token>`` for *identity*, the value of the login variable."""
        login = record_ids.login_for(identity)
        return f"{login}:{self.tokens[login]}"

    def env_for(self, identity: str) -> dict[str, str]:
        """The record URL, *identity*'s login and its own spool directory."""
        return {
            URL_ENV: self.url,
            record_creds.LOGIN_ENV: self.login(identity),
            SPOOL_DIR_ENV: str(self.spool_root / identity),
        }

    def worker_env(self) -> dict[str, str]:
        """:meth:`env_for` the worker."""
        return self.env_for(WORKER_IDENTITY)


@contextlib.contextmanager
def worker_record(
    base: Path, repo: Path, *, bridges: tuple[str, ...] = ()
) -> Iterator[WorkerRecord]:
    """Serve a record for one dispatch worker of *repo* and *bridges* under *base*.

    Args:
        base: The directory the record, its spools and transcripts live in;
            created when absent, and empty if it exists.
        repo: The built deployment repo the worker serves; its dispatch
            agent's ``projects/`` is pointed at this record's transcript root,
            replacing a link an earlier record left.
        bridges: Deployed bridge service names (keys of
            :data:`~osprey.agent_record.identities.BRIDGE_IDENTITIES`) that
            read from this record.

    Yields:
        The running record.
    """
    base.mkdir(parents=True, exist_ok=True)
    config = {**record_config(workers=1, bridges=bridges), "project_root": str(base)}
    transcripts = base / "transcripts"
    root = next(
        name
        for name, identity in record_ids.transcript_roots(config).items()
        if identity == WORKER_IDENTITY
    )
    (transcripts / root).mkdir(parents=True)
    projects = _dispatch_claude_config_dir(repo) / "projects"
    projects.parent.mkdir(parents=True, exist_ok=True)
    if projects.is_symlink():
        projects.unlink()
    projects.symlink_to(transcripts / root, target_is_directory=True)
    spool_root = base / "spool"
    spool_root.mkdir()

    tokens = {
        login: f"e2e-token-of-{login}-0123456789" for login in record_ids.credential_table(config)
    }
    verifiers = "".join(
        record_creds.verifier_line(login, token) + "\n" for login, token in tokens.items()
    )
    env = {
        record_creds.VERIFIER_ENV: base64.b64encode(verifiers.encode()).decode(),
        record_creds.SECRETS_ENV: base64.b64encode(b"[]").decode(),
        record_service.SPOOL_ROOT_ENV: str(spool_root),
        record_service.RECORD_DIR_ENV: str(base / "record"),
        record_service.TRANSCRIPTS_DIR_ENV: str(transcripts),
    }
    with EmbeddedBackend(base / "record" / "record.duckdb") as backend:
        app = record_service.create_app(config=config, backend=backend, env=env)
        with run_app_server(app) as url:
            yield WorkerRecord(url=url, spool_root=spool_root, tokens=tokens)


def _dispatch_claude_config_dir(repo: Path) -> Path:
    """Where *repo*'s worker runs its agent's Claude state, as the worker derives it."""
    import yaml

    from osprey.agent_runner.artifact_resolve import DISPATCH_CLAUDE_CONFIG_DIRNAME
    from osprey_connectors.workspace import agent_data_base_dir, anchored_path

    config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8")) or {}
    return anchored_path(agent_data_base_dir(config), repo) / DISPATCH_CLAUDE_CONFIG_DIRNAME
