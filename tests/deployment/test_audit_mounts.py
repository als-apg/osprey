"""The audit mount: one subdirectory per identity, bound into its own container.

The unified audit ledger lives at ``var/audit/<identity>/`` on the deployment
host, and every containerized service that hosts an interface app or launches
framework MCP servers binds **only its own subdirectory** read-write. That
sentence carries three separable properties, and each one has a section here:

* **The mount follows the middleware, not the login wall.** A web terminal
  records under its own identity whether or not the deployment authenticates,
  and a dispatch worker — which hosts no interface app but launches the
  framework MCP servers through its CLI subprocess — gets the same mount as a
  terminal does. The emitters are therefore unconditional, which is also what
  keeps an ``auth.method: none`` render byte-identical to today's except for
  the audit lines themselves.

* **The isolation is the MOUNT, not a permission bit.** alice's container is
  handed ``var/audit/alice`` and nothing else under ``var/audit``, so bob's
  records are not merely unreadable to it — they are not in its filesystem to
  read. This is what makes "root and osprey never share a file" true by
  construction rather than by careful chmod.

* **The identity and the path are one derivation.** ``OSPREY_AUDIT_IDENTITY``
  is rendered from the same name that names the mount, so the identity a
  record is stamped with and the directory it lands in cannot come apart. A
  writer whose stamped identity and whose path disagree is an audit trail that
  attributes one user's actions to another, which is worse than no trail.

The host side has its own section at the bottom: the subdirectories are
provisioned setgid + group-writable *before* any container starts, on the same
inherit-never-invent precedent as the shared corpus, because a bind source left
to a rootful daemon comes up root-owned and the dropped ``osprey`` process
inside then writes nothing while the deployment looks healthy.
"""

from __future__ import annotations

import copy
import logging
import os
import stat
from importlib import resources
from pathlib import Path

import pytest
import yaml

import osprey
from osprey.deployment.compose_generator import (
    FIXED_SERVICE_AUDIT_IDENTITIES,
    _ensure_agent_data_structure,
    _inject_project_metadata,
    audit_identity_dir,
    dispatch_worker_audit_identities,
    ensure_audit_dir,
    service_audit_identities,
)
from osprey.deployment.web_terminals.render import (
    AUTH_SIDECAR_AUDIT_IDENTITY,
    render_web_terminals,
)
from osprey.utils.workspace import AUDIT_DIR_RELPATH

from .web_terminals.test_golden_render import EXAMPLE_CONFIG

#: The bundle the reference config mounts when a test needs the bundle half.
BUNDLE_PATH = "data/facility_knowledge"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _web_config(*, auth: bool = False, bundle: bool = False, personas: bool = False) -> dict:
    """The reference roster, with the optional halves this file exercises."""
    config = copy.deepcopy(EXAMPLE_CONFIG)
    web_terminals = config["modules"]["web_terminals"]
    if auth:
        # allow_insecure_http, because the render refuses a login wall over
        # cleartext — and a TLS cert pair is not what any assertion here is
        # about.
        web_terminals["auth"] = {"method": "password", "allow_insecure_http": True}
    if bundle:
        config["facility_knowledge"] = {"bundle_path": BUNDLE_PATH}
    if personas:
        web_terminals["personas"] = {
            "operator": {"project": "dls-operator", "project_path": "../dls-operator"},
            "physicist": {"project": "dls-physicist", "project_path": "../dls-physicist"},
        }
        web_terminals["users"] = [
            {"name": "alice", "index": 0, "persona": "operator"},
            {"name": "bob", "index": 1, "persona": "physicist"},
        ]
    return config


def _web_services(**kwargs) -> dict:
    """The parsed ``services:`` mapping of a rendered web overlay."""
    rendered = render_web_terminals(_web_config(**kwargs))["docker-compose.web.yml"]
    return yaml.safe_load(rendered)["services"]


def _env(service: dict) -> dict[str, str]:
    """A compose LIST-form ``environment:`` block as a mapping."""
    return dict(item.split("=", 1) for item in service["environment"])


def _audit_mounts(service: dict) -> list[str]:
    """Every bind this service holds whose source is under the audit zone."""
    return [
        mount
        for mount in service.get("volumes", [])
        if mount.split(":", 1)[0].removeprefix("./").startswith(f"{AUDIT_DIR_RELPATH}/")
    ]


def _worker_services(*, worker_count: int = 1, deployed: bool = True) -> dict:
    """The parsed ``services:`` mapping of a rendered dispatch-worker fragment.

    Rendered through the production injection (``_inject_project_metadata``),
    never a hand-built context: the container-side audit root and the bind
    source are exactly what is under test, and a hand-assembled context would
    assert against values the test itself supplied.
    """
    from jinja2 import Environment, FileSystemLoader

    templates_root = resources.files(osprey).joinpath("templates")
    env = Environment(loader=FileSystemLoader(str(templates_root)), autoescape=False)
    config: dict = {
        "project_name": "demo",
        "project_root": "/r/demo",
        "services": {"dispatch_worker": {"worker_count": worker_count}},
        "system": {"timezone": "UTC"},
    }
    if deployed:
        config["deployed_services"] = ["dispatch_worker"]
    rendered = env.get_template("services/dispatch_worker/docker-compose.yml.j2").render(
        _inject_project_metadata(config)
    )
    return yaml.safe_load(rendered)["services"]


def _service_fragment(service: str, *, deployed: bool = True) -> dict:
    """The parsed ``services:`` mapping of one rendered service fragment.

    Through the production injection, never a hand-built context, for the same
    reason :func:`_worker_services` is: the container-side audit root and the
    bind source are exactly what is under test.
    """
    from jinja2 import Environment, FileSystemLoader

    templates_root = resources.files(osprey).joinpath("templates")
    env = Environment(loader=FileSystemLoader(str(templates_root)), autoescape=False)
    config: dict = {
        "project_name": "demo",
        "project_root": "/r/demo",
        # An empty block per bundled service, not just this one: a template may
        # read a SIBLING's block with no fallback (bluesky_web derives the
        # bridge's URL from `services.bluesky.port`), and every per-service block
        # is read that way, so a context holding only the service under test
        # raises before any `| default(...)` applies.
        "services": {
            Path(str(path)).parent.name: {}
            for path in templates_root.glob("services/*/docker-compose*.yml.j2")
        },
        "system": {"timezone": "UTC"},
        # Present-but-empty rather than absent: Jinja is NOT strict here, but a
        # template reading `deployment.<key> | default(...)` still raises on an
        # undefined `deployment` before the filter runs.
        "deployment": {},
        "deployed_services": [service] if deployed else [],
    }
    rendered = env.get_template(f"services/{service}/docker-compose.yml.j2").render(
        _inject_project_metadata(config)
    )
    return yaml.safe_load(rendered)["services"]


# ---------------------------------------------------------------------------
# Web terminals: identity and path are one derivation
# ---------------------------------------------------------------------------


def test_every_user_is_told_its_own_identity():
    services = _web_services()

    for user in ("alice", "bob"):
        assert _env(services[f"web-{user}"])["OSPREY_AUDIT_IDENTITY"] == user


def test_the_identity_names_the_directory_it_writes_into():
    """The stamped identity and the mounted path are the same name.

    Not a tautology of the render: they are two lines of the compose document
    produced from one roster entry, and the whole point of deriving both from
    that entry is that a record cannot be filed under alice while being stamped
    bob.
    """
    services = _web_services()

    for user in ("alice", "bob"):
        env = _env(services[f"web-{user}"])
        identity = env["OSPREY_AUDIT_IDENTITY"]
        source, target = _audit_mounts(services[f"web-{user}"])[0].split(":")
        assert env["OSPREY_AUDIT_DIR"] == target
        assert source.rsplit("/", 1)[-1] == identity
        assert target.rsplit("/", 1)[-1] == identity


def test_the_audit_path_is_routed_through_the_shared_constant():
    """``var/audit`` is never re-spelled here.

    The writer inside the container, ``osprey reset --purge-audit`` and the
    hooks all resolve their audit root through ``AUDIT_DIR_RELPATH``. A second
    spelling in a template is a mount that shadows nothing while the records
    accumulate in the container's writable layer.
    """
    services = _web_services()

    source, target = _audit_mounts(services["web-alice"])[0].split(":")
    assert source == f"./{AUDIT_DIR_RELPATH}/alice"
    assert target == f"/app/dls-assistant/{AUDIT_DIR_RELPATH}/alice"


def test_the_target_is_computed_per_persona_from_its_own_project_dir():
    """Two personas built from different projects read their audit root at two
    different in-container paths, so one hardcoded target would mount the
    directory where only one of them writes."""
    services = _web_services(personas=True)

    assert _env(services["web-alice"])["OSPREY_AUDIT_DIR"] == (
        f"/app/dls-operator/{AUDIT_DIR_RELPATH}/alice"
    )
    assert _env(services["web-bob"])["OSPREY_AUDIT_DIR"] == (
        f"/app/dls-physicist/{AUDIT_DIR_RELPATH}/bob"
    )


# ---------------------------------------------------------------------------
# Web terminals: the isolation is the mount
# ---------------------------------------------------------------------------


def test_a_user_mounts_its_own_subdir_and_nothing_else_under_the_audit_zone():
    services = _web_services()

    for user, other in (("alice", "bob"), ("bob", "alice")):
        mounts = _audit_mounts(services[f"web-{user}"])
        assert len(mounts) == 1, mounts
        assert mounts[0].startswith(f"./{AUDIT_DIR_RELPATH}/{user}:")
        assert other not in mounts[0]


def test_no_container_binds_the_audit_root_itself():
    """Widening any of these to ``var/audit`` would hand one terminal the whole
    deployment's trail — every other user's records included."""
    services = _web_services(auth=True)

    for service in services.values():
        for mount in _audit_mounts(service):
            assert mount.split(":", 1)[0] != f"./{AUDIT_DIR_RELPATH}"


def test_the_audit_mount_is_read_write():
    """The container writes the records; a `:ro` bind would fail every write."""
    services = _web_services()

    assert _audit_mounts(services["web-alice"])[0].count(":") == 1


def test_nginx_gets_no_audit_mount():
    """The mount follows the middleware. nginx hosts no interface app and
    launches no MCP server — it is a stock image proxying to the ones that do."""
    services = _web_services(auth=True)

    assert _audit_mounts(services["nginx"]) == []
    assert "OSPREY_AUDIT_IDENTITY" not in _env(services["nginx"])


# ---------------------------------------------------------------------------
# Web terminals: unconditional on authentication
# ---------------------------------------------------------------------------


def test_the_user_audit_lines_do_not_depend_on_the_auth_method():
    """The mount follows the middleware, not the login wall: an unauthenticated
    deployment records exactly what an authenticated one does. This is also what
    keeps the ``auth.method: none`` render identical to the authenticated one
    everywhere except the auth seam itself."""
    without = _web_services()
    with_auth = _web_services(auth=True)

    for user in ("alice", "bob"):
        key = f"web-{user}"
        assert (
            _env(without[key])["OSPREY_AUDIT_IDENTITY"]
            == (_env(with_auth[key])["OSPREY_AUDIT_IDENTITY"])
        )
        assert _env(without[key])["OSPREY_AUDIT_DIR"] == _env(with_auth[key])["OSPREY_AUDIT_DIR"]
        assert _audit_mounts(without[key]) == _audit_mounts(with_auth[key])


# ---------------------------------------------------------------------------
# The bundle directory, named for the entrypoint's group step
# ---------------------------------------------------------------------------


def test_the_bundle_dir_is_named_when_a_bundle_is_mounted():
    """The entrypoint's root phase iterates exactly the directories
    ``OSPREY_AUDIT_DIR`` and ``OSPREY_FACILITY_BUNDLE_DIR`` name, so the value
    here has to be the bundle's actual mount target — not a second derivation of
    it that could name a path nothing is mounted at."""
    services = _web_services(bundle=True)

    service = services["web-alice"]
    bundle_targets = [
        mount.split(":")[1] for mount in service["volumes"] if BUNDLE_PATH in mount.split(":")[0]
    ]
    assert _env(service)["OSPREY_FACILITY_BUNDLE_DIR"] == bundle_targets[0]


def test_no_bundle_dir_without_a_bundle_mount():
    """A service with no bundle bound names no second directory, and the
    entrypoint's step skips what is unset — which is why a dispatch worker needs
    no special case there."""
    services = _web_services()

    assert "OSPREY_FACILITY_BUNDLE_DIR" not in _env(services["web-alice"])


# ---------------------------------------------------------------------------
# The auth sidecar
# ---------------------------------------------------------------------------


def test_the_sidecar_identity_is_the_fixed_name():
    """One service, not one per user: the users its login and denial events name
    are their SUBJECTS, never their writers, so a record filed under a username
    here would read as that user having written it."""
    services = _web_services(auth=True)

    assert _env(services["auth"])["OSPREY_AUDIT_IDENTITY"] == AUTH_SIDECAR_AUDIT_IDENTITY
    assert AUTH_SIDECAR_AUDIT_IDENTITY == "sidecar"


def test_the_sidecar_binds_its_own_fixed_subdir():
    services = _web_services(auth=True)

    service = services["auth"]
    source, target = _audit_mounts(service)[0].split(":")
    assert source == f"./{AUDIT_DIR_RELPATH}/{AUTH_SIDECAR_AUDIT_IDENTITY}"
    assert _env(service)["OSPREY_AUTH_AUDIT_DIR"] == target


def test_the_sidecar_reads_its_directory_under_the_auth_specific_name():
    """It is not an OSPREY project — one uvicorn app, no project root to
    resolve — and it is the one service excluded from the entrypoint's group
    step (no entrypoint, no ``user:``, runs as root). ``OSPREY_AUDIT_DIR`` is
    the variable that root phase iterates, so emitting it on an image with no
    root phase would be dead config."""
    services = _web_services(auth=True)

    assert "OSPREY_AUDIT_DIR" not in _env(services["auth"])
    assert "group_add" not in services["auth"]


def test_no_sidecar_service_means_no_sidecar_audit_mount():
    """With authentication off no sidecar is rendered at all, so nothing mounts
    its subdirectory — and the deploy provisions none."""
    services = _web_services()

    assert "auth" not in services
    rendered = render_web_terminals(_web_config())["docker-compose.web.yml"]
    assert AUTH_SIDECAR_AUDIT_IDENTITY not in rendered


# ---------------------------------------------------------------------------
# Dispatch workers: one subdirectory per worker
# ---------------------------------------------------------------------------


def test_each_worker_gets_its_own_identity_and_subdir():
    """One per WORKER, not one for the service: the workers are separate
    containers writing concurrently, and nothing reads their records as a single
    stream."""
    services = _worker_services(worker_count=3)

    for i in (1, 2, 3):
        service = services[f"dispatch-worker-{i}"]
        env = service["environment"]
        assert env["OSPREY_AUDIT_IDENTITY"] == f"dispatch-worker-{i}"
        assert env["OSPREY_AUDIT_DIR"] == f"/app/demo/{AUDIT_DIR_RELPATH}/dispatch-worker-{i}"
        mounts = _audit_mounts(service)
        assert mounts == [
            f"./{AUDIT_DIR_RELPATH}/dispatch-worker-{i}:"
            f"/app/demo/{AUDIT_DIR_RELPATH}/dispatch-worker-{i}"
        ]


def test_a_workers_identity_is_its_compose_service_key():
    """A record traces back to the container that wrote it with no lookup
    table — and the provisioning helper derives the host directory from the same
    rule, which is what the drift test below pins."""
    services = _worker_services(worker_count=2)

    for key, service in services.items():
        assert service["environment"]["OSPREY_AUDIT_IDENTITY"] == key


def test_provisioned_worker_identities_are_exactly_the_rendered_service_keys():
    """The drift that would matter: the template names the service key while
    Python provisions the directory, and a disagreement between the two is a
    bind mount whose host side nobody created — which a rootful daemon then
    creates root-owned, leaving the dropped worker process unable to write."""
    config = {
        "services": {"dispatch_worker": {"worker_count": 4}},
        "deployed_services": ["dispatch_worker"],
    }

    assert dispatch_worker_audit_identities(config) == list(_worker_services(worker_count=4))


def test_a_worker_names_no_bundle_dir():
    """It mounts no facility bundle, so the entrypoint's group step finds
    nothing else to iterate — no special case needed there."""
    services = _worker_services()

    assert "OSPREY_FACILITY_BUNDLE_DIR" not in services["dispatch-worker-1"]["environment"]


def test_no_worker_identities_when_the_worker_is_not_deployed():
    """A stack without the worker provisions nothing for it."""
    assert dispatch_worker_audit_identities({"services": {"dispatch_worker": {}}}) == []
    assert dispatch_worker_audit_identities({}) == []


@pytest.mark.parametrize("worker_count", [None, "", "two", 0])
def test_an_unusable_worker_count_falls_back_to_one(worker_count):
    """The template's own ``| default(1)`` for the same unset/unusable value —
    the two must agree, or the render and the provisioning disagree about how
    many workers exist."""
    config = {
        "services": {"dispatch_worker": {"worker_count": worker_count}},
        "deployed_services": ["dispatch_worker"],
    }

    assert dispatch_worker_audit_identities(config) == ["dispatch-worker-1"]


# ---------------------------------------------------------------------------
# Services with a FIXED identity: bluesky-web
#
# The mount follows the MIDDLEWARE. bluesky_web is the one bundled service
# template that runs one of the framework's interface apps in its own container
# (`uvicorn osprey.interfaces.bluesky_web.app:app`), so it carries
# WebAuthMiddleware and HttpAuditMiddleware — and the mutations the latter
# records are queue add/move/start/stop/abort, i.e. starting and aborting scans
# on the machine. Told no identity it records as the container's process
# account, into a path no bind covers.
# ---------------------------------------------------------------------------


def test_the_bluesky_web_sidecar_is_told_its_own_identity():
    services = _service_fragment("bluesky_web")

    assert services["bluesky-web"]["environment"]["OSPREY_AUDIT_IDENTITY"] == "bluesky-web"


def test_the_bluesky_web_identity_is_its_compose_service_key():
    """Same rule as a dispatch worker's: a record traces back to the container
    that wrote it with no lookup table."""
    services = _service_fragment("bluesky_web")

    for key, service in services.items():
        assert service["environment"]["OSPREY_AUDIT_IDENTITY"] == key


def test_the_bluesky_web_sidecar_binds_its_own_subdir_and_nothing_wider():
    """The isolation is the MOUNT: this container holds a bind to
    ``var/audit/bluesky-web`` and nothing else under the audit zone, so it can
    read neither a terminal user's records nor a worker's."""
    service = _service_fragment("bluesky_web")["bluesky-web"]

    mounts = _audit_mounts(service)
    assert len(mounts) == 1, mounts
    source, target = mounts[0].split(":")
    assert source == f"./{AUDIT_DIR_RELPATH}/bluesky-web"
    assert source.rsplit("/", 1)[-1] == service["environment"]["OSPREY_AUDIT_IDENTITY"]
    assert target.rsplit("/", 1)[-1] == service["environment"]["OSPREY_AUDIT_IDENTITY"]
    # Read-write: the container writes the records, and a `:ro` bind would fail
    # every write.
    assert mounts[0].count(":") == 1


def test_the_bluesky_web_mount_target_is_where_its_writer_actually_resolves():
    """NOT ``/app/<project>/var/audit``. This image runs no OSPREY project, so
    ``writer.audit_dir`` falls through to the working directory (WORKDIR /app) —
    and a mount at the project-rooted path would bind a directory nothing ever
    writes to while every record piled up in the container's writable layer,
    discarded at the next ``osprey up``."""
    service = _service_fragment("bluesky_web")["bluesky-web"]

    _, target = _audit_mounts(service)[0].split(":")
    assert target == f"/app/{AUDIT_DIR_RELPATH}/bluesky-web"
    assert not target.startswith("/app/demo/")


def test_the_bluesky_web_sidecar_names_no_audit_dir_variable():
    """``OSPREY_AUDIT_DIR`` is the contract a compose file offers the PROJECT
    image's entrypoint, whose root phase joins the group of each directory it
    names before dropping privileges. This image has no entrypoint and no
    privilege drop, so the variable would be config nothing reads — the same
    reasoning that keeps it off the auth sidecar."""
    service = _service_fragment("bluesky_web")["bluesky-web"]

    assert "OSPREY_AUDIT_DIR" not in service["environment"]
    assert "group_add" not in service


def test_provisioned_service_identities_are_exactly_the_rendered_service_keys():
    """The drift that would matter: the template names the service key while
    Python provisions the directory, and a disagreement between the two is a
    bind mount whose host side nobody created."""
    for service, identity in FIXED_SERVICE_AUDIT_IDENTITIES.items():
        assert service_audit_identities({"deployed_services": [service]}) == [identity]
        assert list(_service_fragment(service)) == [identity]


def test_no_service_identities_when_the_service_is_not_deployed():
    """A stack without it provisions nothing for it."""
    assert service_audit_identities({"deployed_services": ["mongodb"]}) == []
    assert service_audit_identities({}) == []


def test_the_build_path_provisions_every_deployed_services_subdir(tmp_path):
    """The service binds it, so something has to create it before the container
    runtime does."""
    _ensure_agent_data_structure(
        {"project_root": str(tmp_path), "deployed_services": list(FIXED_SERVICE_AUDIT_IDENTITIES)}
    )

    for identity in FIXED_SERVICE_AUDIT_IDENTITIES.values():
        target = audit_identity_dir(tmp_path, identity)
        assert target.is_dir()
        assert stat.S_IMODE(target.stat().st_mode) & stat.S_ISGID


def test_the_build_path_provisions_nothing_for_an_undeployed_service(tmp_path):
    _ensure_agent_data_structure({"project_root": str(tmp_path), "deployed_services": []})

    for identity in FIXED_SERVICE_AUDIT_IDENTITIES.values():
        assert not (tmp_path / AUDIT_DIR_RELPATH / identity).exists()


# ---------------------------------------------------------------------------
# Host-side provisioning
# ---------------------------------------------------------------------------


def test_a_new_audit_dir_gets_exactly_the_bits_the_mount_needs(tmp_path):
    """setgid + group-write, nothing wider: the records name what an operator
    asked an agent to do, and no consumer needs the ``other`` triad."""
    gid = ensure_audit_dir(tmp_path, "alice", relative_to=tmp_path)

    target = audit_identity_dir(tmp_path, "alice")
    assert target.is_dir()
    assert stat.S_IMODE(target.stat().st_mode) == 0o2770
    assert gid == target.stat().st_gid


def test_the_subdir_hangs_under_the_shared_audit_relpath(tmp_path):
    """Same constant the container's writer resolves through, so the directory
    the deploy provisions is the directory ``osprey reset --purge-audit``
    promises to keep."""
    assert audit_identity_dir(tmp_path, "alice") == tmp_path / AUDIT_DIR_RELPATH / "alice"


def test_records_written_by_any_uid_take_the_directorys_group(tmp_path):
    """The setgid half in action: a file created inside is group-owned by the
    DIRECTORY, whichever process wrote it — which is what keeps a container's
    records purgeable by the operator on the host."""
    ensure_audit_dir(tmp_path, "alice")

    target = audit_identity_dir(tmp_path, "alice")
    record = target / "mcp.jsonl"
    record.write_text("{}\n")

    assert record.stat().st_gid == target.stat().st_gid


def test_an_existing_dir_is_widened_not_replaced(tmp_path):
    """Inherit, never invent: the operator's own ``other`` triad survives, and
    nothing is chowned."""
    target = audit_identity_dir(tmp_path, "alice")
    target.mkdir(parents=True, mode=0o0705)

    ensure_audit_dir(tmp_path, "alice", relative_to=tmp_path)

    assert stat.S_IMODE(target.stat().st_mode) == 0o2775


def test_widening_an_existing_dir_is_reported_relatively(tmp_path, caplog):
    """One unwrapped line naming a relative path — the default build view
    carries exactly one absolute path, and a second wraps a normal terminal and
    buries it."""
    target = audit_identity_dir(tmp_path, "alice")
    target.mkdir(parents=True, mode=0o700)

    with caplog.at_level(logging.INFO, logger="deployment.compose"):
        ensure_audit_dir(tmp_path, "alice", relative_to=tmp_path)

    assert f"Audit dir {AUDIT_DIR_RELPATH}/alice: 0700 -> 2770" in caplog.text
    assert str(tmp_path) not in caplog.text
    emitted = [record.getMessage() for record in caplog.records]
    assert emitted and all(len(message) <= 80 for message in emitted), emitted


def test_a_freshly_created_dir_is_reported_at_nothing_louder_than_debug(tmp_path, caplog):
    """Every deploy creates these on a fresh repo. An INFO line per identity per
    ``osprey up`` about permissions nobody had is noise that trains operators to
    ignore the line that matters."""
    with caplog.at_level(logging.INFO, logger="deployment.compose"):
        ensure_audit_dir(tmp_path, "alice", relative_to=tmp_path)
        ensure_audit_dir(tmp_path, "alice", relative_to=tmp_path)

    assert "Audit dir" not in caplog.text


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses the directory permission this test relies on",
)
def test_provisioning_failure_is_not_fatal(tmp_path, caplog):
    """A deploy is not refused because one audit subdirectory could not be
    made — the records fail visibly at write time instead, which is a smaller
    failure than a stack that will not come up."""
    parent = tmp_path / "locked"
    parent.mkdir(mode=0o500)
    try:
        with caplog.at_level(logging.WARNING, logger="deployment.compose"):
            assert ensure_audit_dir(parent, "alice") is None
    finally:
        parent.chmod(0o700)  # so pytest's tmp_path cleanup can remove it

    assert "Could not provision audit directory" in caplog.text


def test_the_build_path_provisions_every_deployed_workers_subdir(tmp_path):
    """The worker binds it, so something has to create it before the container
    runtime does — root-owned, and unwritable by the dropped process inside."""
    _ensure_agent_data_structure(
        {
            "project_root": str(tmp_path),
            "services": {"dispatch_worker": {"worker_count": 2}},
            "deployed_services": ["dispatch_worker"],
        }
    )

    for i in (1, 2):
        target = audit_identity_dir(tmp_path, f"dispatch-worker-{i}")
        assert target.is_dir()
        assert stat.S_IMODE(target.stat().st_mode) & stat.S_ISGID


def test_the_build_path_provisions_nothing_for_an_undeployed_worker(tmp_path):
    _ensure_agent_data_structure(
        {"project_root": str(tmp_path), "services": {"dispatch_worker": {"worker_count": 2}}}
    )

    assert not (tmp_path / AUDIT_DIR_RELPATH / "dispatch-worker-1").exists()


def test_web_deploy_provisions_every_subdir_before_any_compose_invocation(monkeypatch, tmp_path):
    """Ordering is the point, exactly as it is for the bundle: left to the
    runtime, a rootful daemon creates each missing bind source owned by ROOT,
    and the container's dropped ``osprey`` process can then write none of the
    records it exists to write."""
    from osprey.deployment.web_terminals import provision

    from .web_terminals.test_provision import _sidecar_config, _stub_web_stack

    _stub_web_stack(monkeypatch, tmp_path)
    # _stub_web_stack neuters resolve_personas to an empty roster; this deploy
    # has to have one, since the roster IS the identity list.
    monkeypatch.setattr(
        provision,
        "resolve_personas",
        lambda *a, **kw: [{"name": "alice"}, {"name": "bob"}],
    )
    expected = ["alice", "bob", AUTH_SIDECAR_AUDIT_IDENTITY]
    # Sampled from INSIDE the run: the assertion is about ordering, so it
    # cannot be made after the fact.
    seen_at_first_compose: list[list[bool]] = []

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def _run(cmd, **kwargs):
        seen_at_first_compose.append(
            [audit_identity_dir(tmp_path, identity).is_dir() for identity in expected]
        )
        return _Completed()

    monkeypatch.setattr(provision.subprocess, "run", _run)

    provision.deploy_up_web_terminals(_sidecar_config("local"), [], False, {}, [])

    assert seen_at_first_compose, "no compose invocation ran to order against"
    assert seen_at_first_compose[0] == [True] * len(expected)
    for identity in expected:
        assert stat.S_IMODE(audit_identity_dir(tmp_path, identity).stat().st_mode) & stat.S_ISGID


def test_web_deploy_provisions_no_sidecar_subdir_with_auth_off(monkeypatch, tmp_path):
    """Nothing renders the sidecar service, so nothing mounts its subdirectory —
    and an empty directory nobody writes is a question an operator should not
    have to answer."""
    from osprey.deployment.web_terminals import provision

    from .web_terminals.test_provision import _sidecar_config, _stub_web_stack

    _stub_web_stack(monkeypatch, tmp_path)
    monkeypatch.setattr(provision, "resolve_personas", lambda *a, **kw: [{"name": "alice"}])

    provision.deploy_up_web_terminals(_sidecar_config("local", method="none"), [], False, {}, [])

    assert audit_identity_dir(tmp_path, "alice").is_dir()
    assert not audit_identity_dir(tmp_path, AUTH_SIDECAR_AUDIT_IDENTITY).exists()


# ---------------------------------------------------------------------------
# Reserved and unusable roster names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["sidecar", "dispatch-worker-1", "dispatch-worker-12"])
def test_a_roster_name_that_is_a_service_identity_is_refused(name):
    """The one place "your own subdirectory" must not be honoured literally: a
    user named for a service would get a read-write bind onto the records of the
    component that audits them."""
    config = _web_config(auth=True)
    config["modules"]["web_terminals"]["users"] = ["alice", name]

    with pytest.raises(ValueError, match="audit identity"):
        render_web_terminals(config)


def test_the_reserved_name_gate_holds_with_authentication_off():
    """Unconditional, unlike the username-charset gate: the name is a directory
    in every posture, so the collision is real in every posture too."""
    config = _web_config()
    config["modules"]["web_terminals"]["users"] = ["sidecar"]

    with pytest.raises(ValueError, match="audit identity"):
        render_web_terminals(config)


def test_an_ordinary_hyphenated_name_is_still_fine():
    """The gate is narrow: only the service identities themselves, not every
    name that happens to look service-ish."""
    config = _web_config()
    config["modules"]["web_terminals"]["users"] = ["dispatch-worker", "sidecar-ops", "alice"]

    services = _web_services()
    assert services  # the reference roster still renders
    render_web_terminals(config)


def test_a_roster_name_that_is_a_fixed_service_identity_is_refused():
    """The same protection reaches every service whose identity is fixed, not
    just the two the gate was born with: the reserved set is built from
    ``FIXED_SERVICE_AUDIT_IDENTITIES``, so a service that gains an identity
    gains this refusal in the same edit rather than in a follow-up nobody
    remembers."""
    assert FIXED_SERVICE_AUDIT_IDENTITIES, "nothing to protect — the mapping is empty"

    for identity in FIXED_SERVICE_AUDIT_IDENTITIES.values():
        config = _web_config()
        config["modules"]["web_terminals"]["users"] = ["alice", identity]

        with pytest.raises(ValueError, match="audit identity") as exc_info:
            render_web_terminals(config)
        assert identity in str(exc_info.value)


# ---------------------------------------------------------------------------
# A roster name is a path segment in EVERY auth posture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["../../etc", "..", ".", "a/b", "Alice", "j.doe"])
def test_a_name_that_is_not_one_path_segment_is_refused_with_auth_off(name):
    """``auth.method: none`` is the DEFAULT posture, and the render emits
    ``./var/audit/<name>:<container>/var/audit/<name>`` verbatim from the roster.

    So a name of ``../../etc`` was a bind whose host side resolved outside the
    repo, read-write, into a container that runs agent-generated code — and
    ``alice`` beside ``Alice`` was two terminals on one host subdirectory
    wherever the filesystem is case-insensitive, each able to rewrite the
    other's trail. The provisioning seam refuses these names, but it only
    declines to CREATE the source; the container runtime then creates it
    anyway when the container starts. The refusal has to be the renderer's.
    """
    config = _web_config()
    config["modules"]["web_terminals"]["users"] = ["alice", name]

    with pytest.raises(ValueError, match="audit subdirectory") as exc_info:
        render_web_terminals(config)
    assert repr(name) in str(exc_info.value)


def test_no_render_emits_an_audit_bind_that_leaves_the_audit_zone():
    """The property the gate exists for, asserted on the artifact itself: every
    audit bind source a render emits stays inside ``var/audit/``."""
    for kwargs in ({}, {"auth": True}, {"personas": True}):
        for service in _web_services(**kwargs).values():
            for mount in _audit_mounts(service):
                source = mount.split(":", 1)[0]
                assert ".." not in Path(source).parts, mount
                assert Path(source).resolve().is_relative_to(Path(AUDIT_DIR_RELPATH).resolve())


def test_the_render_and_the_provisioning_seam_agree_about_every_roster_name():
    """The disagreement that let this through: the seam refused an identity the
    renderer emitted anyway, and the renderer wins because its output is what
    compose reads. One charset, asked at both ends."""
    for name in ("../../etc", "..", "a/b", "Alice"):
        config = _web_config()
        config["modules"]["web_terminals"]["users"] = [name]

        with pytest.raises(ValueError):
            audit_identity_dir("/repo", name)
        with pytest.raises(ValueError):
            render_web_terminals(config)


# ---------------------------------------------------------------------------
# The provisioning seam refuses what it cannot safely make
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("identity", ["../../x", "a/b", "", ".", "Alice"])
def test_the_path_seam_refuses_an_identity_outside_the_charset(identity, tmp_path):
    """``audit_identity_dir`` is the seam every PROVISIONING caller goes
    through, and the identity is a path SEGMENT there. Validating at the seam
    rather than at each caller is what makes a traversal unreachable on the
    deploy path, which provisions before the render's own roster gates ever run.

    Scope, stated because an earlier version of this docstring overclaimed it:
    this seam covers what is CREATED, not what is mounted. A compose document is
    the renderer's output, and this helper never sees it — the render refuses the
    same names on its own account
    (``render._check_roster_audit_identities``, which uses the same charset), and
    that refusal is what keeps a traversing bind out of the document.
    """
    with pytest.raises(ValueError, match="audit identity"):
        audit_identity_dir(tmp_path, identity)


@pytest.mark.parametrize("identity", ["alice", "sidecar", "dispatch-worker-1", "a_b-2"])
def test_every_identity_the_framework_mints_passes_the_seam(identity, tmp_path):
    assert audit_identity_dir(tmp_path, identity) == tmp_path / AUDIT_DIR_RELPATH / identity


def test_a_refused_identity_creates_nothing_and_is_not_fatal(tmp_path, caplog):
    """Skipped, not raised: this helper is never the thing that fails a deploy —
    the render's roster gates and lint are, with messages that can explain a bad
    username. What matters is that nothing outside the audit zone is created on
    the way there."""
    with caplog.at_level(logging.WARNING, logger="deployment.compose"):
        assert ensure_audit_dir(tmp_path / "repo", "../../escaped") is None

    assert "Skipping audit directory provisioning" in caplog.text
    assert not (tmp_path / "escaped").exists()
    assert not (tmp_path / "repo").exists()
