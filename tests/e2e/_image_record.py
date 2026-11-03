"""The record credentials a bare ``docker run`` of the project image is handed.

A terminal the image serves holds the deployment's agent record for its
lifetime, as every host terminal does, and reads the build's host login and the
record's service file to do it. Both are secrets the build writes beside the
profile, and like the provider key they are kept out of the image and supplied
when the container is started: bound read-only where the project inside the
image reads them, ``/app/<project>/<file>``.

The build writes them ``0600`` for the deploying account, and the image's
terminal runs as its own ``osprey`` user, which a runner's account need not
match; so each is bound from a copy the container's user can read, made in a
directory of the test's own.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from osprey.agent_record import credentials as record_creds
from osprey.agent_record import identities as record_ids


def record_credential_mounts(repo: Path, project_name: str, workdir: Path) -> list[str]:
    """``docker run`` arguments binding *repo*'s host login and service file into the image.

    Args:
        repo: The deployment repo the image was built from; its build wrote both files.
        project_name: The project directory under ``/app`` inside the image.
        workdir: A directory of the test's own the readable copies are made in.

    Returns:
        ``-v`` arguments, one read-only bind per file.
    """
    config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8")) or {}
    host = record_ids.host_identity(config)
    assert host, "the build recorded no deployment.host_identity"
    mounts: list[str] = []
    for name in (record_creds.login_file(record_ids.login_for(host)), record_creds.SERVICE_FILE):
        source = repo / name
        assert source.is_file(), f"the build wrote no {name} at {source}"
        copy = workdir / name
        shutil.copyfile(source, copy)
        copy.chmod(0o644)
        mounts += ["-v", f"{copy}:/app/{project_name}/{name}:ro"]
    return mounts
