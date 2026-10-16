"""The ``docker run`` arguments that hand an image its telemetry sender login.

A container is handed its sender's login the way a deployment hands it: the
``.env.telemetry-sender.<identity>`` file ``osprey build`` mints, passed as an
env file. An image run without ``OSPREY_OTEL_GATEWAY_HOST`` is a host launch,
so it sends under ``deployment.host_identity``. The file name, the identity and
the enabled-gateway predicate all come from the production modules; nothing
here spells one of them.
"""

from __future__ import annotations

import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.build.claude_code_telemetry import build_gateway_url
from osprey.deployment import telemetry_config as tc
from osprey.deployment import telemetry_credentials as creds


def sender_env_file_args(repo: Path, config: Mapping[str, Any]) -> list[str]:
    """``["--env-file", <sender file>]`` under an enabled gateway block, else ``[]``.

    Asserts that the file exists and is mode 0600; the assertion names the
    path, never the file's content.
    """
    if build_gateway_url(config) is None:
        return []
    identity = tc.host_identity(config)
    assert identity, "the rendered config records no deployment.host_identity"
    path = Path(repo) / creds.sender_file(identity)
    assert path.is_file(), f"`osprey build` minted no sender login file at {path}"
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"{path} is mode {mode:o}, not 600"
    return ["--env-file", str(path)]
