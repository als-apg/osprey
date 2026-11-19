"""The dispatch bus's per-container key files, as ``osprey up`` provisions them.

Each case runs the real provisioning against a temporary repo root and then
reads the files back through the same settings readers the containers use, so
"the dispatcher accepts exactly the workers that hold keys" is checked end to
end rather than by inspecting strings.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from osprey.deployment import dispatch_bus_keys as keys
from osprey.dispatch.bus import settings
from osprey.dispatch.bus.keys import public_of


def _config(*, workers: int = 2, transport: str | None = "bus") -> dict:
    dispatcher: dict = {"port": 10010}
    if transport is not None:
        dispatcher["transport"] = transport
    return {
        "deployed_services": ["event_dispatcher", "dispatch_worker"],
        "services": {"event_dispatcher": dispatcher, "dispatch_worker": {"worker_count": workers}},
    }


def _env(root: Path, service: str) -> dict[str, str]:
    return keys._read(root / keys.key_file(service))


def _files(root: Path) -> list[str]:
    return sorted(p.name for p in root.glob(f"{keys.FILE_PREFIX}*"))


@pytest.mark.parametrize("transport", [None, "http"])
def test_a_deployment_off_the_bus_gets_no_files(tmp_path, transport):
    report = keys.ensure_dispatch_bus_keys(tmp_path, _config(transport=transport))
    assert report == keys.BusKeysReport()
    assert _files(tmp_path) == []
    assert keys.digest_env(tmp_path, _config(transport=transport)) == {}


def test_every_container_gets_its_own_file_and_only_its_own_keys(tmp_path):
    report = keys.ensure_dispatch_bus_keys(tmp_path, _config())

    names = [
        ".env.dispatch-bus.dispatch-worker-1",
        ".env.dispatch-bus.dispatch-worker-2",
        ".env.dispatch-bus.event-dispatcher",
    ]
    assert _files(tmp_path) == names
    assert sorted(report.minted) == names
    for name in names:
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o600

    dispatcher = _env(tmp_path, keys.DISPATCHER_SERVICE)
    worker = _env(tmp_path, "dispatch-worker-1")
    assert set(dispatcher) == {"DISPATCH_BUS_SECRET_KEY", "DISPATCH_BUS_WORKER_KEYS"}
    assert set(worker) == {"DISPATCH_BUS_SECRET_KEY", "DISPATCH_BUS_SERVER_KEY"}
    # No worker holds the dispatcher's secret or another worker's.
    secrets = {
        _env(tmp_path, s)["DISPATCH_BUS_SECRET_KEY"]
        for s in (keys.DISPATCHER_SERVICE, "dispatch-worker-1", "dispatch-worker-2")
    }
    assert len(secrets) == 3


def test_the_files_pair_up_as_the_containers_read_them(tmp_path):
    keys.ensure_dispatch_bus_keys(tmp_path, _config())

    server = settings.server_settings(
        {settings.BIND_ENV: "tcp://0.0.0.0:10050", **_env(tmp_path, keys.DISPATCHER_SERVICE)}
    )
    assert server is not None and server.keys is not None
    for name in ("dispatch-worker-1", "dispatch-worker-2"):
        worker = settings.worker_settings(
            {
                settings.CONNECT_ENV: "tcp://event-dispatcher:10050",
                settings.WORKER_NAME_ENV: name,
                **_env(tmp_path, name),
            }
        )
        assert worker is not None and worker.keys is not None
        assert worker.keys.server_public == server.keys.public
        assert server.keys.workers[worker.keys.public] == name


def test_a_redeploy_keeps_every_key_and_writes_nothing(tmp_path):
    keys.ensure_dispatch_bus_keys(tmp_path, _config())
    before = {name: (tmp_path / name).read_text() for name in _files(tmp_path)}

    report = keys.ensure_dispatch_bus_keys(tmp_path, _config())

    assert report == keys.BusKeysReport()
    assert {name: (tmp_path / name).read_text() for name in _files(tmp_path)} == before


def test_a_worker_added_is_minted_and_listed_and_one_removed_is_dropped(tmp_path):
    keys.ensure_dispatch_bus_keys(tmp_path, _config(workers=2))
    kept = _env(tmp_path, "dispatch-worker-1")["DISPATCH_BUS_SECRET_KEY"]

    grown = keys.ensure_dispatch_bus_keys(tmp_path, _config(workers=3))
    assert grown.minted == [".env.dispatch-bus.dispatch-worker-3"]
    assert ".env.dispatch-bus.event-dispatcher" in grown.written
    listed = _env(tmp_path, keys.DISPATCHER_SERVICE)["DISPATCH_BUS_WORKER_KEYS"]
    assert "dispatch-worker-3=" in listed
    assert _env(tmp_path, "dispatch-worker-1")["DISPATCH_BUS_SECRET_KEY"] == kept

    shrunk = keys.ensure_dispatch_bus_keys(tmp_path, _config(workers=1))
    assert shrunk.removed == [
        ".env.dispatch-bus.dispatch-worker-2",
        ".env.dispatch-bus.dispatch-worker-3",
    ]
    listed = _env(tmp_path, keys.DISPATCHER_SERVICE)["DISPATCH_BUS_WORKER_KEYS"]
    assert listed == f"dispatch-worker-1={public_of(kept)}"


def test_deleting_a_file_rotates_that_key_and_its_listing(tmp_path):
    keys.ensure_dispatch_bus_keys(tmp_path, _config())
    old = _env(tmp_path, "dispatch-worker-2")["DISPATCH_BUS_SECRET_KEY"]
    (tmp_path / keys.key_file("dispatch-worker-2")).unlink()

    report = keys.ensure_dispatch_bus_keys(tmp_path, _config())

    new = _env(tmp_path, "dispatch-worker-2")["DISPATCH_BUS_SECRET_KEY"]
    assert new != old
    assert report.minted == [".env.dispatch-bus.dispatch-worker-2"]
    listed = _env(tmp_path, keys.DISPATCHER_SERVICE)["DISPATCH_BUS_WORKER_KEYS"]
    assert public_of(new) in listed and public_of(old) not in listed


def test_a_dispatcher_rekeyed_hands_every_worker_its_new_public_key(tmp_path):
    keys.ensure_dispatch_bus_keys(tmp_path, _config())
    (tmp_path / keys.key_file(keys.DISPATCHER_SERVICE)).write_text("DISPATCH_BUS_SECRET_KEY=nope\n")

    report = keys.ensure_dispatch_bus_keys(tmp_path, _config())

    server_public = public_of(_env(tmp_path, keys.DISPATCHER_SERVICE)["DISPATCH_BUS_SECRET_KEY"])
    assert report.minted == [".env.dispatch-bus.event-dispatcher"]
    for name in ("dispatch-worker-1", "dispatch-worker-2"):
        assert _env(tmp_path, name)["DISPATCH_BUS_SERVER_KEY"] == server_public


def test_each_digest_moves_with_its_own_file_only(tmp_path):
    keys.ensure_dispatch_bus_keys(tmp_path, _config())
    before = keys.digest_env(tmp_path, _config())
    assert set(before) == {
        "OSPREY_DISPATCH_BUS_DIGEST_EVENT_DISPATCHER",
        "OSPREY_DISPATCH_BUS_DIGEST_DISPATCH_WORKER_1",
        "OSPREY_DISPATCH_BUS_DIGEST_DISPATCH_WORKER_2",
    }
    (tmp_path / keys.key_file("dispatch-worker-2")).unlink()
    keys.ensure_dispatch_bus_keys(tmp_path, _config())
    after = keys.digest_env(tmp_path, _config())

    assert (
        after["OSPREY_DISPATCH_BUS_DIGEST_DISPATCH_WORKER_1"]
        == (before["OSPREY_DISPATCH_BUS_DIGEST_DISPATCH_WORKER_1"])
    )
    for moved in ("DISPATCH_WORKER_2", "EVENT_DISPATCHER"):
        assert (
            after[f"OSPREY_DISPATCH_BUS_DIGEST_{moved}"]
            != (before[f"OSPREY_DISPATCH_BUS_DIGEST_{moved}"])
        )
