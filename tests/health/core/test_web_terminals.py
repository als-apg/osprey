"""Tests for the core ``web_terminals`` health category.

The category asks every web-terminal card and every dispatch worker which
account name its process runs under (``ca_user`` on ``/health``) and compares it
with the name the render intended. The intended name is derived from the same
config keys the compose templates read, so these tests pin both halves: the
address each row probes, and the status each ``/health`` answer maps to.
"""

from __future__ import annotations

import httpx

from osprey.deployment.web_terminals.ports import allocate_ports, base_ports_from_config
from osprey.health.core.web_terminals import CATEGORY, web_terminals
from osprey.health.models import CheckResult, Status
from osprey.port_layout import default_port, resolve_port_base


def _card_port(config: dict, index: int) -> int:
    wt = config["modules"]["web_terminals"]
    base_ports = base_ports_from_config(wt, base=resolve_port_base(config))
    return allocate_ports(base_ports, index)["web"]


def _worker_port(config: dict, i: int) -> int:
    base = resolve_port_base(config)
    return default_port("worker", 1, base=base) + (i - 1)


def _cfg(
    *,
    users=None,
    method: str | None = "password",
    worker: dict | None = None,
    deployed: list[str] | None = None,
) -> dict:
    cfg: dict = {}
    if users is not None:
        wt: dict = {"enabled": True, "users": users}
        if method is not None:
            wt["auth"] = {"method": method}
        cfg["modules"] = {"web_terminals": wt}
    if worker is not None:
        cfg["services"] = {"dispatch_worker": worker}
        cfg["deployed_services"] = deployed if deployed is not None else ["dispatch_worker"]
    return cfg


def _answers(by_port: dict[int, dict | int | Exception]):
    """A mock handler answering ``/health`` per port: a body, a status, or a raise."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        assert request.url.host == "127.0.0.1"
        answer = by_port.get(request.url.port)
        if answer is None:
            raise httpx.ConnectError("refused", request=request)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, int):
            return httpx.Response(answer, text="nope")
        return httpx.Response(200, json=answer)

    return handler


async def _run(config, handler) -> dict[str, CheckResult]:
    results = await web_terminals(config, transport=httpx.MockTransport(handler))()
    assert all(r.category == CATEGORY for r in results)
    names = [r.name for r in results]
    assert names == sorted(names)
    return {r.name: r for r in results}


_ALICE = {"name": "alice", "index": 0, "control_identity": "alice"}
_BOB = {"name": "bob-ops", "index": 1}


class TestCardExpectation:
    """A card expects its roster identity only behind a login wall."""

    async def test_walled_card_matching_identity_is_ok(self):
        cfg = _cfg(users=[_ALICE])
        rows = await _run(cfg, _answers({_card_port(cfg, 0): {"ca_user": "alice"}}))
        row = rows["web_terminals.alice"]
        assert row.status is Status.OK
        assert row.value == "alice"

    async def test_walled_card_mismatch_is_error(self):
        cfg = _cfg(users=[_ALICE])
        rows = await _run(cfg, _answers({_card_port(cfg, 0): {"ca_user": "osprey"}}))
        row = rows["web_terminals.alice"]
        assert row.status is Status.ERROR
        assert "alice" in row.message and "osprey" in row.message

    async def test_unwalled_card_expects_image_account(self):
        # auth.method token: no login wall, so compose emits no identity and the
        # card keeps the image account even though the roster names one.
        cfg = _cfg(users=[_ALICE], method="token")
        rows = await _run(cfg, _answers({_card_port(cfg, 0): {"ca_user": "osprey"}}))
        assert rows["web_terminals.alice"].status is Status.OK

    async def test_unwalled_card_answering_the_roster_identity_is_error(self):
        cfg = _cfg(users=[_ALICE], method="none")
        rows = await _run(cfg, _answers({_card_port(cfg, 0): {"ca_user": "alice"}}))
        assert rows["web_terminals.alice"].status is Status.ERROR

    async def test_card_without_identity_expects_osprey(self):
        cfg = _cfg(users=[_ALICE, _BOB])
        rows = await _run(
            cfg,
            _answers(
                {
                    _card_port(cfg, 0): {"ca_user": "alice"},
                    _card_port(cfg, 1): {"ca_user": "osprey"},
                }
            ),
        )
        assert rows["web_terminals.bob_ops"].status is Status.OK
        assert rows["web_terminals.alice"].status is Status.OK

    async def test_bare_string_roster_entries(self):
        cfg = _cfg(users=["carol"])
        rows = await _run(cfg, _answers({_card_port(cfg, 0): {"ca_user": "osprey"}}))
        assert rows["web_terminals.carol"].status is Status.OK


class TestMissingField:
    """An image older than control identity answers /health without ``ca_user``."""

    async def test_missing_field_with_identity_expected_is_error(self):
        cfg = _cfg(users=[_ALICE])
        rows = await _run(cfg, _answers({_card_port(cfg, 0): {"status": "healthy"}}))
        row = rows["web_terminals.alice"]
        assert row.status is Status.ERROR
        assert "ca_user" in row.message or "ca_user" in row.details

    async def test_missing_field_with_osprey_expected_is_ok(self):
        cfg = _cfg(users=[_BOB])
        rows = await _run(cfg, _answers({_card_port(cfg, 1): {"status": "healthy"}}))
        assert rows["web_terminals.bob_ops"].status is Status.OK


class TestSkipped:
    async def test_skipped_identity_is_warning_not_error(self):
        # A non-root start of an osprey-* identity leaves uid 1000 as osprey and
        # says so; that is a degraded start, not an image skew.
        cfg = _cfg(worker={"network": "host", "worker_count": 1})
        rows = await _run(
            cfg,
            _answers(
                {
                    _worker_port(cfg, 1): {
                        "ca_user": "osprey",
                        "control_identity_skipped": "non-root-start",
                    }
                }
            ),
        )
        row = rows["web_terminals.dispatch_worker_1"]
        assert row.status is Status.WARNING
        assert "non-root-start" in row.message or "non-root-start" in row.details


class TestWorkers:
    async def test_host_workers_expect_numbered_identity(self):
        cfg = _cfg(worker={"network": "host", "worker_count": 2})
        rows = await _run(
            cfg,
            _answers(
                {
                    _worker_port(cfg, 1): {"ca_user": "osprey-dispatch-1"},
                    _worker_port(cfg, 2): {"ca_user": "osprey-dispatch-1"},
                }
            ),
        )
        assert rows["web_terminals.dispatch_worker_1"].status is Status.OK
        assert rows["web_terminals.dispatch_worker_2"].status is Status.ERROR

    async def test_worker_port_keys_are_honoured(self):
        cfg = _cfg(
            worker={
                "network": "host",
                "worker_count": 2,
                "worker_port_base": 31000,
                "worker_port_stride": 10,
            }
        )
        rows = await _run(
            cfg,
            _answers(
                {
                    31000: {"ca_user": "osprey-dispatch-1"},
                    31010: {"ca_user": "osprey-dispatch-2"},
                }
            ),
        )
        assert rows["web_terminals.dispatch_worker_1"].status is Status.OK
        assert rows["web_terminals.dispatch_worker_2"].status is Status.OK

    async def test_worker_missing_field_is_error(self):
        # A worker always expects osprey-dispatch-<i>, so an old image is skew.
        cfg = _cfg(worker={"network": "host"})
        rows = await _run(cfg, _answers({_worker_port(cfg, 1): {"status": "ok"}}))
        assert rows["web_terminals.dispatch_worker_1"].status is Status.ERROR

    async def test_bridge_workers_are_skipped_not_probed(self):
        cfg = _cfg(worker={"worker_count": 2})

        def handler(request):  # pragma: no cover - must not be reached
            raise AssertionError(f"bridge worker probed at {request.url}")

        rows = await _run(cfg, handler)
        assert set(rows) == {
            "web_terminals.dispatch_worker_1",
            "web_terminals.dispatch_worker_2",
        }
        assert all(r.status is Status.SKIP for r in rows.values())

    async def test_worker_not_deployed_contributes_nothing(self):
        cfg = _cfg(worker={"network": "host"}, deployed=[])
        assert await _run(cfg, _answers({})) == {}


class TestProbeFailures:
    async def test_unreachable_is_warning(self):
        cfg = _cfg(users=[_ALICE])
        rows = await _run(cfg, _answers({}))
        row = rows["web_terminals.alice"]
        assert row.status is Status.WARNING
        assert str(_card_port(cfg, 0)) in row.details

    async def test_http_error_is_warning(self):
        cfg = _cfg(users=[_ALICE])
        rows = await _run(cfg, _answers({_card_port(cfg, 0): 503}))
        assert rows["web_terminals.alice"].status is Status.WARNING

    async def test_non_json_body_is_warning(self):
        cfg = _cfg(users=[_ALICE])

        def handler(_request):
            return httpx.Response(200, text="<html>")

        rows = await _run(cfg, handler)
        assert rows["web_terminals.alice"].status is Status.WARNING


class TestConfigShapes:
    async def test_none_config_yields_no_rows(self):
        assert await _run(None, _answers({})) == {}

    async def test_disabled_web_terminals_yields_no_rows(self):
        cfg = {"modules": {"web_terminals": {"enabled": False, "users": ["alice"]}}}
        assert await _run(cfg, _answers({})) == {}

    async def test_unknown_auth_method_is_one_config_row(self):
        cfg = _cfg(users=[_ALICE], method="kerberos")
        rows = await _run(cfg, _answers({}))
        assert set(rows) == {"web_terminals.auth"}
        assert rows["web_terminals.auth"].status is Status.WARNING

    async def test_cards_and_workers_together(self):
        cfg = _cfg(users=[_ALICE], worker={"network": "host"})
        rows = await _run(
            cfg,
            _answers(
                {
                    _card_port(cfg, 0): {"ca_user": "alice"},
                    _worker_port(cfg, 1): {"ca_user": "osprey-dispatch-1"},
                }
            ),
        )
        assert set(rows) == {"web_terminals.alice", "web_terminals.dispatch_worker_1"}
        assert all(r.status is Status.OK for r in rows.values())


class TestRegistration:
    def test_category_is_registered_and_config_dependent(self):
        from osprey.health.core import CORE_CATEGORY_NAMES, get_core_category_factory
        from osprey.health.records import CONFIG_DEPENDENT

        assert "web_terminals" in CORE_CATEGORY_NAMES
        assert get_core_category_factory("web_terminals") is web_terminals
        assert "web_terminals" in CONFIG_DEPENDENT
