"""The telemetry gateway under one container runtime: file access, forced recreate, the ingest host rule.

One case, run by every lane that boots a stack under a runtime of its own
(docker, podman with the compose plugin, podman-compose 1.0.6), so the three
facts it checks are decided once across all of them:

* **File access.** The gateway reads the senders' htpasswd file (mode 0600)
  and a ledger file (mode 0640) the operator owns, from a container whose only
  capability is ``DAC_READ_SEARCH``. A record posted with a sender's login
  lands, and so does the ledger line.
* **Forced recreate.** Deleting ``.env.telemetry-ingest`` and running
  ``osprey up`` recreates the gateway; the new container, at a new address,
  still inserts with the rotated ``ingest`` password.
* **The ``ingest`` host rule.** The store accepts ``ingest`` by password from
  any address: no ``HOST NAME`` restriction is rendered, because a host name
  resolved from a client address is not the same string under every runtime.
  The case pins that rendering.

The caller names the runtime, its own compose project and port block, and the
environment ``osprey`` runs with. Every resource the case creates is named off
that project, and it removes exactly those in its ``finally``.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

from osprey.deployment import telemetry_config as tc
from osprey.deployment import telemetry_credentials as creds

VERB_TIMEOUT_S = 600
ROW_TIMEOUT_S = 120.0


def _osprey_bin() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


class _Case:
    def __init__(
        self,
        *,
        runtime: str,
        project_name: str,
        port_base: int,
        workdir: Path,
        osprey_env: Mapping[str, str],
    ) -> None:
        self.runtime = runtime
        self.project = project_name
        self.port_base = port_base
        self.repo = workdir / project_name
        self.osprey = _osprey_bin()
        self.env = {
            k: v
            for k, v in {**os.environ, **osprey_env}.items()
            if k not in (tc.GATEWAY_HOST_ENV, tc.GATEWAY_PORT_ENV, creds.LOGIN_ENV)
        }
        self.store = f"{project_name}-telemetry-store"
        self.gateway = f"{project_name}-telemetry-gateway"
        self.volumes = (f"{project_name}_telemetry_store_data", f"{project_name}_gateway-state")
        self.config: dict = {}

    def cli(self, *args: str, timeout: int = 120) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.runtime, *args], capture_output=True, text=True, timeout=timeout, check=False
        )

    def verb(self, *args: str, cwd: Path | None = None) -> None:
        result = subprocess.run(
            [str(self.osprey), *args],
            cwd=str(cwd or self.repo),
            capture_output=True,
            text=True,
            timeout=VERB_TIMEOUT_S,
            env=self.env,
            check=False,
        )
        assert result.returncode == 0, (
            f"osprey {' '.join(args)} failed (rc={result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )

    def sweep(self) -> None:
        """Remove this case's containers and volumes by exact name."""
        for name in (self.gateway, self.store):
            self.cli("rm", "-f", name)
        for volume in self.volumes:
            self.cli("volume", "rm", volume)

    def sql(self, query: str) -> str:
        result = self.cli(
            "exec", self.store, "clickhouse-client", "--format", "TSV", "-q", query, timeout=60
        )
        if result.returncode != 0:
            raise AssertionError(f"store query failed: {result.stderr.strip()}\n{query}")
        return result.stdout

    def wait_count(self, query: str, expected: int) -> int:
        deadline = time.monotonic() + ROW_TIMEOUT_S
        value = -1
        while time.monotonic() < deadline:
            try:
                value = int(self.sql(query).strip() or 0)
            except (AssertionError, ValueError):
                value = -1
            if value >= expected:
                return value
            time.sleep(1.0)
        return value

    def wait_ready(self) -> None:
        deadline = time.monotonic() + 180
        last = ""
        while time.monotonic() < deadline:
            try:
                if self.sql("SELECT 1").strip() == "1":
                    httpx.get(
                        f"http://127.0.0.1:{tc.metrics_port(self.config)}/metrics", timeout=3
                    ).raise_for_status()
                    return
            except (AssertionError, httpx.HTTPError) as exc:
                last = str(exc)
            time.sleep(1.0)
        raise AssertionError(f"store and gateway not ready: {last}")

    def container_id(self, name: str) -> str:
        result = self.cli("inspect", "--type", "container", "-f", "{{.Id}}", name)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    def post_log(self, case: str) -> int:
        identity = tc.host_identity(self.config)
        login = creds.read_sender_login(self.repo, identity) if identity else None
        assert login, f"no sender login for {identity!r}"
        payload = {
            "resourceLogs": [
                {
                    "resource": {
                        "attributes": [{"key": "service.name", "value": {"stringValue": "e2e"}}]
                    },
                    "scopeLogs": [
                        {
                            "logRecords": [
                                {
                                    "timeUnixNano": str(time.time_ns()),
                                    "body": {"stringValue": case},
                                    "attributes": [
                                        {"key": "e2e.case", "value": {"stringValue": case}}
                                    ],
                                }
                            ]
                        }
                    ],
                }
            ]
        }
        response = httpx.post(
            f"http://127.0.0.1:{tc.gateway_port(self.config)}/v1/logs",
            content=json.dumps(payload),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Basic " + base64.b64encode(login.encode()).decode(),
            },
            timeout=15,
        )
        return response.status_code

    def logs_count(self, case: str) -> int:
        return self.wait_count(
            f"SELECT count() FROM telemetry.otel_logs WHERE LogAttributes['e2e.case'] = '{case}'",
            1,
        )


def run_gateway_runtime_case(
    *,
    runtime: str,
    project_name: str,
    port_base: int,
    workdir: Path,
    osprey_env: Mapping[str, str],
) -> None:
    """Boot hello-world under *runtime* and check file access, forced recreate and the host rule."""
    case = _Case(
        runtime=runtime,
        project_name=project_name,
        port_base=port_base,
        workdir=workdir,
        osprey_env=osprey_env,
    )
    case.sweep()
    try:
        case.verb(
            "init",
            str(case.repo),
            "--preset",
            "hello-world",
            "--no-git",
            "--set",
            f"port_base={port_base}",
            cwd=workdir,
        )
        case.verb("build", "--skip-deps", "--skip-lifecycle")
        case.config = yaml.safe_load((case.repo / "build" / "config.yml").read_text())
        identity = tc.host_identity(case.config)
        assert identity, "the render records no deployment.host_identity"
        (case.repo / ".env").touch()

        # A ledger line the operator owns, readable by group only.
        marker = f"e2e-ledger-{uuid.uuid4().hex[:12]}"
        ledger_dir = case.repo / "var" / "audit" / identity
        ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger = ledger_dir / "e2e.jsonl"
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        ledger.write_text(
            json.dumps(
                {
                    "ts": stamp,
                    "surface": "maintenance",
                    "actor": identity,
                    "posture": "e2e",
                    "posture_source": "e2e",
                    "session": None,
                    "subject": marker,
                    "decision": "allow",
                    "reason": "e2e",
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        ledger.chmod(0o640)
        htpasswd = case.repo / creds.HTPASSWD_FILE
        assert htpasswd.stat().st_mode & 0o777 == 0o600, oct(htpasswd.stat().st_mode)

        case.verb("up", "-d")
        case.wait_ready()

        first = f"e2e-{uuid.uuid4().hex[:12]}"
        assert case.post_log(first) == 200, "the gateway refused a sender login it holds"
        assert case.logs_count(first) == 1, "a record posted with a sender login did not land"
        assert (
            case.wait_count(
                f"SELECT count() FROM telemetry.audit_ledger WHERE Body LIKE '%{marker}%'", 1
            )
            == 1
        ), "the ledger line the gateway should read did not land"

        before = case.container_id(case.gateway)
        old_ingest = (case.repo / creds.INGEST_FILE).read_bytes()
        (case.repo / creds.INGEST_FILE).unlink()
        case.verb("up", "-d")
        case.wait_ready()
        assert (case.repo / creds.INGEST_FILE).read_bytes() != old_ingest
        assert case.container_id(case.gateway) != before, "the gateway was not recreated"

        second = f"e2e-{uuid.uuid4().hex[:12]}"
        assert case.post_log(second) == 200
        assert case.logs_count(second) == 1, "the recreated gateway could not insert as ingest"

        rule = case.sql(
            "SELECT host_names, host_names_regexp, host_names_like, host_ip "
            "FROM system.users WHERE name = 'ingest'"
        ).rstrip("\n")
        assert rule.split("\t") == ["[]", "[]", "[]", "['::/0']"], (
            f"the ingest account carries a host restriction: {rule!r}"
        )
    finally:
        if case.repo.is_dir():
            subprocess.run(
                [str(case.osprey), "down"],
                cwd=str(case.repo),
                capture_output=True,
                text=True,
                timeout=300,
                env=case.env,
                check=False,
            )
        case.sweep()
