"""No container is handed a telemetry credential that is not its own.

The containment rule the credential files exist for: a secret reaches a
container only as that container's own ``env_file`` or read-only bind. Two
ways to break it, both checked over **every** bundled service template and the
web overlay, so a template added later is covered without being listed here:

* a bind whose source is the repo root, or any directory above it, hands the
  container every credential file at once;
* a credential file named on a service that does not own it.

Rendered with the gateway deployed, because that is when the files are named.
"""

from __future__ import annotations

import copy
from pathlib import Path, PurePosixPath

import pytest
import yaml
from tests.deployment.web_terminals.test_golden_render import EXAMPLE_CONFIG
from tests.templates.test_render_defaults_golden import (
    _PINNED_PROJECT_ROOT,
    TEMPLATES,
    _pinned_context,
    _probe_repo,
    _raw_default_config,
    _render_templates,
)

from osprey.deployment import telemetry_credentials as creds
from osprey.deployment.web_terminals.render import render_web_terminals

_TELEMETRY_SERVICES = ["telemetry_store", "telemetry_gateway", "dispatch_worker"]


def _is_credential(name: str) -> bool:
    return creds.is_credential_file(name)


def _telemetry_config(repo_root: Path) -> dict:
    config = _raw_default_config(repo_root)
    config["deployed_services"] = list(_TELEMETRY_SERVICES)
    config["services"]["dispatch_worker"] = {"worker_count": 2}
    config["deployment"] = {"host_identity": "builder"}
    return config


@pytest.fixture(scope="module")
def service_documents() -> dict[str, dict]:
    """Every bundled service template, rendered with the gateway deployed."""
    with _probe_repo() as repo_root:
        context = _pinned_context(_telemetry_config(repo_root))
        rendered = _render_templates(context, TEMPLATES)
    return {name: yaml.safe_load(text) or {} for name, text in rendered.items()}


def _web_config() -> dict:
    config = copy.deepcopy(EXAMPLE_CONFIG)
    config["deployed_services"] = list(_TELEMETRY_SERVICES)
    config["modules"]["web_terminals"]["auth"] = {
        "method": "password",
        "allow_insecure_http": True,
    }
    return config


def _web_document() -> dict:
    return yaml.safe_load(render_web_terminals(_web_config())["docker-compose.web.yml"])


def _services(document: dict) -> dict[str, dict]:
    return {
        name: service
        for name, service in (document.get("services") or {}).items()
        if isinstance(service, dict)
    }


def _bind_sources(service: dict) -> list[str]:
    """The host side of every bind this service mounts."""
    sources = []
    for mount in service.get("volumes") or []:
        if isinstance(mount, dict):
            if mount.get("type") == "bind":
                sources.append(str(mount.get("source")))
            continue
        source = str(mount).split(":", 1)[0]
        # A named volume is a bare name; a bind is a path.
        if source.startswith((".", "/", "~", "$")):
            sources.append(source)
    return sources


def _env_files(service: dict) -> list[str]:
    stated = service.get("env_file") or []
    entries = stated if isinstance(stated, list) else [stated]
    return [str(entry["path"] if isinstance(entry, dict) else entry) for entry in entries]


def _is_root_or_above(source: str) -> bool:
    """Whether a bind source is the repo root or a directory containing it."""
    root = PurePosixPath(_PINNED_PROJECT_ROOT)
    if source.startswith("/"):
        path = PurePosixPath(source)
        return path == root or path in root.parents
    parts = [part for part in PurePosixPath(source).parts if part != "."]
    return not parts or parts[0] == ".."


def _all_services(service_documents: dict[str, dict]) -> list[tuple[str, str, dict]]:
    found = [
        (golden, name, service)
        for golden, document in sorted(service_documents.items())
        for name, service in _services(document).items()
    ]
    found.extend(("web overlay", name, svc) for name, svc in _services(_web_document()).items())
    return found


def test_the_render_covers_every_bundled_service(service_documents: dict[str, dict]) -> None:
    assert len(service_documents) == len(TEMPLATES)
    names = {name for _, name, _ in _all_services(service_documents)}
    assert {"dispatch-worker-1", "dispatch-worker-2", "web-alice", "nginx"} <= names
    assert "archiver-recorder" in names or "archiver_recorder" in names


def test_no_bind_source_is_the_repo_root_or_above_it(service_documents: dict[str, dict]) -> None:
    offenders = [
        f"{golden}: {name} binds {source}"
        for golden, name, service in _all_services(service_documents)
        for source in _bind_sources(service)
        if _is_root_or_above(source)
    ]
    assert not offenders, (
        "a bind of the repo root hands the container every credential file "
        "at the root:\n  " + "\n  ".join(offenders)
    )


def test_each_credential_file_is_named_only_on_its_owner(
    service_documents: dict[str, dict],
) -> None:
    owners = creds.owner_files(
        {
            "deployed_services": _TELEMETRY_SERVICES,
            "services": {"dispatch_worker": {"worker_count": 2}},
        }
    )
    owned = {service: set(files) for service, files in owners.items()}
    for user in ("alice", "bob"):
        owned[f"web-{user}"] = {creds.sender_file(user)}

    named_somewhere = set()
    offenders = []
    for golden, name, service in _all_services(service_documents):
        handed = [PurePosixPath(entry).name for entry in _env_files(service)]
        handed += [PurePosixPath(source).name for source in _bind_sources(service)]
        for filename in filter(_is_credential, handed):
            named_somewhere.add(filename)
            if filename not in owned.get(name, set()):
                offenders.append(f"{golden}: {name} is handed {filename}")
    assert not offenders, "\n".join(offenders)
    # The check is live: the senders rendered today are handed their files.
    assert {
        creds.sender_file("dispatch-worker-1"),
        creds.sender_file("dispatch-worker-2"),
        creds.sender_file("alice"),
        creds.sender_file("bob"),
    } <= named_somewhere


def test_each_owner_carries_its_own_digest_label(service_documents: dict[str, dict]) -> None:
    """A service handed a credential file is recreated when that file changes."""
    for _golden, name, service in _all_services(service_documents):
        handed = [PurePosixPath(entry).name for entry in _env_files(service)]
        if not any(map(_is_credential, handed)) or name.startswith("web-"):
            continue
        labels = service.get("labels") or {}
        assert labels.get(creds.DIGEST_LABEL) == "${" + creds.digest_env_name(name) + ":-}", name


@pytest.mark.parametrize(
    ("source", "refused"),
    [
        (".", True),
        ("./", True),
        ("..", True),
        ("../other", True),
        ("/", True),
        ("/deploy", True),
        (_PINNED_PROJECT_ROOT, True),
        ("./build", False),
        ("./build/data/simulator", False),
        ("./var/audit/alice", False),
        (f"{_PINNED_PROJECT_ROOT}/build", False),
        ("/etc/ssl/certs", False),
    ],
)
def test_the_root_rule_reads_a_source_the_way_compose_does(source: str, refused: bool) -> None:
    assert _is_root_or_above(source) is refused


def test_every_credential_file_a_service_names_is_one_the_build_mints(
    service_documents: dict[str, dict], tmp_path: Path
) -> None:
    """A compose document naming a missing ``env_file`` fails every command.

    So whatever a rendered service names, ``osprey build`` has created: that is
    what lets ``status``, ``down`` and ``logs`` run on any built repo.
    """
    with _probe_repo() as repo_root:
        config = _telemetry_config(repo_root)
    web = _web_config()
    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=False)
    (tmp_path / "web").mkdir()
    creds.ensure_telemetry_credentials(tmp_path / "web", web, reader_keys=False)
    minted = {path.name for path in tmp_path.iterdir() if path.is_file()} | {
        path.name for path in (tmp_path / "web").iterdir()
    }

    named = {
        PurePosixPath(entry).name
        for _golden, _name, service in _all_services(service_documents)
        for entry in _env_files(service)
        if _is_credential(PurePosixPath(entry).name)
    }
    assert named and named <= minted, sorted(named - minted)
