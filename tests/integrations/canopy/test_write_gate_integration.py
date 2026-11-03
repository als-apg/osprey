"""A CANOPY write through the real FRONT and DATA, with the gates the sidecar renders.

A real :class:`CanopySidecar` starts ``canopy_mcp`` (FRONT) and ``canopy.main``
(DATA) over a ``va`` deployment in ``tmp_path``, with the fake connector
registered under the virtual accelerator's type in DATA by a
``sitecustomize`` (as in ``test_sidecar_smoke.py``). Every PUT goes to FRONT
the way the terminal's panel proxy sends it -- the access header, JSON, the
binding the heartbeat named -- so it crosses FRONT's exact relay, the write
key in the keys file the sidecar wrote, DATA's PV backend and the control
session, and lands in the agent record as ``audit`` entries.

Nothing about writes is rendered into DATA's environment: each PUT is judged on
the execution verdict DATA reads then.

* the switch allowed: the write is relayed and confirmed;
* the tier, the switch or the position refusing: 403 in the verdict's words,
  audited;
* a hold on any session the terminal runs (the sidecar's ``sessions.json``,
  the hold store it names to DATA): 403 ``session_held``, audited; no session,
  no hold;
* the shared card (oi-40): the audit identity the deploy renders for a shared
  roster entry is the card's name, and the write's entries are recorded under it.
"""

from __future__ import annotations

import json
import os
import shutil
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Marks the module as a CANOPY contract test: the mark is derived from imports.
import canopy  # noqa: F401
import pytest
import yaml
from tests._renders import tier_render

from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.integrations.canopy.backend import SURFACE
from osprey.integrations.canopy.sidecar import CanopySidecar
from osprey.runtime import session as session_module
from osprey_connectors import control_context, posture_store, session_hold, types
from osprey_connectors.identity import AUDIT_IDENTITY_ENV
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors_tests._harness._control_context_fixtures import _set_switch_to

#: How long the two servers get to come up.
READY_TIMEOUT_S = 120.0

PV = "TEST:SCALAR"
BINDING = "va@1"

#: The viewer's widget catalog, a copy of CANOPY's own at the proven commit.
IR_MAP = Path(__file__).parent / "fixtures" / "ir-map.json"

SITECUSTOMIZE = textwrap.dedent(
    """
    import os

    if os.environ.get("CANOPY_EPICS__BACKEND"):
        from osprey_connectors import types
        from osprey_connectors_tests._fake_connector import FakeConnector
        from osprey_connectors.factory import ConnectorFactory

        ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, FakeConnector)
    """
)


def _config(*, writes: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "control_system": {
            "type": types.VIRTUAL_ACCELERATOR,
            "connector": {types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0}},
        },
    }
    if writes is not None:
        config["control_system"]["writes"] = writes
    return config


def _put(
    sidecar: CanopySidecar, body: dict[str, Any], *, url: str | None = None, headers=None
) -> tuple[int, dict[str, Any]]:
    """PUT *body* to :data:`PV` on FRONT (or *url*) as the panel proxy sends it."""
    target = url or f"{sidecar.url}/hare/v1/pvs/{PV}"
    sent = {"Content-Type": "application/json"}
    sent.update(sidecar.auth_headers if headers is None else headers)
    request = urllib.request.Request(
        target, data=json.dumps(body).encode(), headers=sent, method="PUT"
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status, raw = int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    try:
        payload = json.loads(raw) if raw else {}
    except ValueError:
        payload = {"raw": raw.decode(errors="replace")}
    return status, payload


def _code_and_detail(payload: dict[str, Any]) -> tuple[Any, Any]:
    """The error envelope's ``code`` and ``detail``, wherever CANOPY nests them."""
    inner = payload.get("detail")
    if isinstance(inner, dict):
        return inner.get("code"), inner.get("detail", inner.get("message"))
    return payload.get("code"), inner


def audit_records(recorded_entries) -> list[tuple[str, dict[str, Any]]]:
    """Every CANOPY ``audit`` entry the test's spool zone holds, with its identity.

    The sidecar's children inherit the zone's ``OSPREY_SPOOL_DIR``, so DATA's
    control session writes there; the identity is the spool directory's.
    """
    return [
        (entry.identity, dict(entry.body))
        for entry in recorded_entries("audit")
        if entry.body.get("surface") == SURFACE
    ]


@pytest.fixture
def launch(tmp_path, monkeypatch, recorded_entries):
    """Build sidecars over a ``va`` deployment in ``tmp_path``; all are stopped afterwards."""
    for name in list(os.environ):
        if name.startswith(("CANOPY_", "OSPREY_CANOPY_", "OSPREY_CONTROL_TARGET", "EPICS_")):
            monkeypatch.delenv(name)
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        session_hold.HOLD_DIR_ENV_VAR,
        session_module.SESSION_KEY_ENV_VAR,
        "OSPREY_EXECUTION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)

    state_dir = tmp_path / "control_target"
    state_dir.mkdir()
    config_path = tmp_path / "config.yml"

    payload = tmp_path / "canopy_payload"
    viewer = payload / "viewer"
    viewer.mkdir(parents=True)
    (viewer / "index.html").write_text("<!doctype html><title>stub</title>", encoding="utf-8")
    shutil.copy(IR_MAP, viewer / "ir-map.json")

    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    # The connectors' test package too: the sitecustomize imports the fake from it.
    root = Path(__file__).resolve().parents[3]
    paths = [str(site), str(root), str(root / "packages" / "osprey-connectors" / "tests")]
    inherited = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([*paths, inherited] if inherited else paths))

    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(state_dir))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "test-operator")
    monkeypatch.chdir(tmp_path)
    record = state_dir / control_context.RECORD_FILENAME
    control_context.write_record(
        control_context.ControlContext(target="va", generation=1), path=record
    )
    control_context.invalidate_cache()
    # The switch on for the bound target, so the write verdict allows.
    _set_switch_to(record, {"va": "allowed"})
    control_context.invalidate_cache()

    sidecars: list[CanopySidecar] = []

    def start(
        config: dict[str, Any], *, hold_root: Path | None = None, live_sessions=None
    ) -> CanopySidecar:
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        sidecar = CanopySidecar(
            tmp_path / "agent-data",
            "",
            None,
            payload_dir=payload,
            hold_root=hold_root,
            live_sessions=live_sessions,
        )
        sidecars.append(sidecar)
        sidecar.preflight()
        sidecar.spawn()
        try:
            sidecar.wait_ready(READY_TIMEOUT_S)
        except RuntimeError as exc:
            pytest.fail(str(exc))
        return sidecar

    start.root = tmp_path
    start.record = record
    start.recorded = recorded_entries
    yield start
    for sidecar in sidecars:
        sidecar.stop()
    control_context.invalidate_cache()


def test_a_put_is_relayed_end_to_end_and_confirmed(launch):
    """FRONT's relay, the write key, DATA's backend and the session: one confirmed write."""
    sidecar = launch(_config(writes="allowed"))
    status, payload = _put(sidecar, {"value": 4.25, "binding": BINDING})
    assert status == 200, payload

    records = audit_records(launch.recorded)
    decisions = [(record["decision"], record.get("reason")) for _, record in records]
    assert decisions and all(decision == "allowed" for decision, _ in decisions), decisions
    assert any(PV in json.dumps(record) for _, record in records)


def _decisions(launch) -> list[tuple[str, Any]]:
    return [
        (record["decision"], record.get("reason")) for _, record in audit_records(launch.recorded)
    ]


def _refused(payload: dict[str, Any]) -> str:
    code, detail = _code_and_detail(payload)
    assert code == "pv_write_refused", payload
    return detail


def test_a_put_quoting_another_binding_is_refused_and_writes_nothing(launch):
    sidecar = launch(_config(writes="allowed"))
    status, payload = _put(sidecar, {"value": 1.0, "binding": "standin@9"})
    assert status == 409, payload
    assert _code_and_detail(payload) == (
        "pv_target_changed",
        refusal_phrase("switching", "simulator"),
    )
    assert _decisions(launch) == [("refused", "switching")]


@pytest.mark.parametrize("writes", [None, "never", True])
def test_a_tier_that_does_not_allow_writes_is_refused_in_the_verdicts_words(launch, writes):
    sidecar = launch(_config(writes=writes))
    status, payload = _put(sidecar, {"value": 4.25, "binding": BINDING})

    assert status == 403, payload
    assert _refused(payload) == refusal_phrase("tier_never", "simulator")
    assert _decisions(launch) == [("refused", "tier_never")]


def test_a_switch_turned_off_under_a_running_data_refuses_the_next_put(launch):
    """No restart: DATA reads the switch at the write."""
    sidecar = launch(_config(writes="allowed"))
    assert _put(sidecar, {"value": 1.0, "binding": BINDING})[0] == 200
    _set_switch_to(launch.record, {})
    control_context.invalidate_cache()

    status, payload = _put(sidecar, {"value": 2.0, "binding": BINDING})
    assert status == 403, payload
    assert _refused(payload) == refusal_phrase("switch_off", "simulator")
    assert sorted(_decisions(launch)) == [
        ("allowed", "write_attempt"),
        ("allowed", "write_landed"),
        ("refused", "switch_off"),
    ]


def test_at_asked_a_put_without_a_confirmation_is_refused(launch):
    _set_switch_to(launch.record, {"va": "asked"})
    control_context.invalidate_cache()
    sidecar = launch(_config(writes="allowed"))
    status, payload = _put(sidecar, {"value": 4.25, "binding": BINDING})

    assert status == 403, payload
    assert _refused(payload) == "This write needs your confirmation; nothing written."
    assert _decisions(launch) == [("refused", "approval_missing")]


def _hold_store(launch, *keys: str) -> Path:
    root = launch.root / "holds"
    for key in keys:
        session_hold.ensure_session(key, root=root)
    return root


def _raise(root: Path, key: str) -> None:
    assert session_hold.open_flag(
        key,
        flag_id="flag-1",
        source="watchdog:limits",
        reason="a limit was refused",
        prompt_ref="turn-7",
        prompt_excerpt="set the current to 900",
        root=root,
    )


def test_a_hold_on_any_of_the_terminals_sessions_refuses_the_put(launch):
    root = _hold_store(launch, "alice-1", "alice-2")
    _raise(root, "alice-2")
    sidecar = launch(
        _config(writes="allowed"), hold_root=root, live_sessions=lambda: ["alice-1", "alice-2"]
    )
    status, payload = _put(sidecar, {"value": 4.25, "binding": BINDING})

    assert status == 403, payload
    assert _refused(payload) == (
        refusal_phrase("session_held", "simulator") + " Raised by: watchdog:limits."
    )
    assert _decisions(launch) == [("refused", "session_held")]


def test_a_held_session_the_terminal_no_longer_runs_holds_nothing(launch):
    """No session, no hold: the terminal names none, so the held record is not asked."""
    root = _hold_store(launch, "alice-1")
    _raise(root, "alice-1")
    sidecar = launch(_config(writes="allowed"), hold_root=root, live_sessions=lambda: [])
    status, payload = _put(sidecar, {"value": 4.25, "binding": BINDING})

    assert status == 200, payload
    # The spool's order is not the write's: compare the entries as a set.
    assert sorted(_decisions(launch)) == [("allowed", "write_attempt"), ("allowed", "write_landed")]


def test_data_refuses_a_write_with_the_read_key_and_without_a_key(launch):
    """Only FRONT's write key may write on DATA; its read key and no key may not."""
    sidecar = launch(_config(writes="allowed"))
    read_key = (sidecar.key_dir / "front.read").read_text().strip()
    direct = f"{sidecar.data_url}/hare/v1/pvs/{PV}"
    body = {"value": 4.25, "binding": BINDING}

    assert _put(sidecar, body, url=direct, headers={})[0] == 401
    assert _put(sidecar, body, url=direct, headers={"X-API-Key": read_key})[0] == 403
    assert audit_records(launch.recorded) == []


def test_front_refuses_a_put_without_the_access_credential(launch):
    sidecar = launch(_config(writes="allowed"))
    status, _ = _put(sidecar, {"value": 4.25, "binding": BINDING}, headers={})
    assert status == 401
    assert audit_records(launch.recorded) == []


# -- the shared card (oi-40) -------------------------------------------------------


def _shared_card_identity(repo: Path, card: str) -> str:
    """The audit identity the deploy renders into a shared card's terminal."""
    project = repo / "build" / "dls-readwrite"
    project.mkdir(parents=True)
    (project / "config.yml").write_text(
        yaml.safe_dump({"project_name": "dls-readwrite", "web": {"panels": {"canopy": {}}}}),
        encoding="utf-8",
    )
    config = {
        "facility": {"name": "Demo Light Source"},
        "system": {"timezone": "UTC"},
        "registry": {"url": "git.dls.example.org:5050/physics/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "renders": {
                    "readwrite": tier_render(
                        "readwrite", "dls-readwrite", project_path="build/dls-readwrite"
                    )
                },
                "default_tier": "readwrite",
                "users": [
                    {"name": "alice", "index": 0, "tier": "readwrite"},
                    {"name": card, "index": 1, "tier": "readwrite", "access": "any"},
                ],
            }
        },
    }
    services = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])["services"]
    environment = services[f"web-{card}"]["environment"]
    if isinstance(environment, list):
        environment = dict(item.split("=", 1) for item in environment)
    return str(environment[AUDIT_IDENTITY_ENV])


def test_a_shared_cards_write_is_audited_under_the_cards_name(
    launch, monkeypatch, tmp_path, spool_for
):
    card = "control-room"
    identity = _shared_card_identity(tmp_path / "deploy", card)
    assert identity == card

    monkeypatch.setenv(AUDIT_IDENTITY_ENV, identity)
    # The card's own spool directory, as the deploy binds it for that name.
    spool_for(identity)
    sidecar = launch(_config(writes="allowed"))
    status, payload = _put(sidecar, {"value": 2.5, "binding": BINDING})
    assert status == 200, payload

    records = audit_records(launch.recorded)
    assert records
    for recorded_as, _record in records:
        assert recorded_as == card, recorded_as
    assert any(card in json.dumps(record) for _, record in records)
