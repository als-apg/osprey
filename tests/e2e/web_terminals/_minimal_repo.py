"""The smallest deployment repo a host ``osprey web`` serves: a profile, a render, a record.

``osprey web`` walks up from its working directory to the nearest
``profile.yml`` and serves that repo's ``build/config.yml``, and every host
terminal holds the deployment's agent record for its lifetime: the host-local
record, which it starts when none answers. A host record needs what a build
stamps and mints for it — ``deployment.host_identity`` in the render and the
host login beside the profile — so a lane that writes its repo by hand states
the identity and runs the build's credential step itself.

The record listens on a port of its own, reserved for as long as the lane
needs it, so lanes running side by side never find each other's record
answering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from osprey.agent_record import credentials as record_creds
from osprey.agent_record import host as host_record

#: The identity the terminal's host sessions record under.
HOST_IDENTITY = "e2e-host"


def write_minimal_repo(repo: Path, config: dict[str, Any], record_port: int) -> Path:
    """Write *repo* serving *config*, with the host record's identity, port and login.

    Args:
        repo: The repo root to create.
        config: The render's ``config.yml`` content; ``project_root``,
            ``deployment.host_identity`` and ``agent_record.port`` are added.
        record_port: The host record's port, which the caller holds reserved
            (``reserved_port()``) while the terminal runs: the record binds it
            only once the terminal starts it.

    Returns:
        *repo*.
    """
    rendered = {
        **config,
        "project_root": str(repo),
        "deployment": {**(config.get("deployment") or {}), "host_identity": HOST_IDENTITY},
        "agent_record": {**(config.get("agent_record") or {}), "port": record_port},
    }
    (repo / "build").mkdir(parents=True)
    name = (config.get("system") or {}).get("name") or repo.name
    (repo / "profile.yml").write_text(f"name: {name}\n", encoding="utf-8")
    (repo / "build" / "config.yml").write_text(yaml.safe_dump(rendered), encoding="utf-8")
    record_creds.ensure_agent_record_credentials(repo, rendered, reader_keys=False)
    return repo


def stop_host_record(repo: Path) -> None:
    """Stop *repo*'s host-local record if a lane's teardown left it running."""
    config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    state_dir = host_record.host_state_dir(repo, config)
    if state_dir.exists():
        host_record.stop_host_process(state_dir, wait_s=10)
