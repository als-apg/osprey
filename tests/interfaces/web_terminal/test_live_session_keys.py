"""Which sessions a terminal runs now: the keys a session-writing sidecar is held with."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from osprey.interfaces.web_terminal.chat_session_pool import ChatSessionPool
from osprey.interfaces.web_terminal.routes.watchdog import live_session_keys, session_is_live


class _Pool:
    def __init__(self, *keys: str) -> None:
        self._keys = list(keys)

    def session_keys(self) -> list[str]:
        return list(self._keys)

    def get_session(self, key: str):
        return object() if key in self._keys else None

    def has_chat_key(self, _key: str) -> bool:
        return False


def _app(**state) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(**state))


def test_every_pool_is_named_once_and_sorted():
    app = _app(pty_registry=_Pool("b-term", "a-term"), operator_registry=_Pool("c-chat", "a-term"))

    assert live_session_keys(app) == ["a-term", "b-term", "c-chat"]


def test_a_named_key_is_one_session_is_live_accepts():
    app = _app(pty_registry=_Pool("a-term"), operator_registry=_Pool("c-chat"))

    assert all(session_is_live(app, key) for key in live_session_keys(app))


def test_no_registry_names_no_session():
    assert live_session_keys(_app()) == []


def test_the_chat_pool_names_a_chat_that_is_still_starting():
    pool = ChatSessionPool(factory=lambda *_args: None)  # type: ignore[arg-type,return-value]
    pool._sessions["running"] = object()  # type: ignore[assignment]
    pool._pending["starting"] = asyncio.Future(loop=asyncio.new_event_loop())

    assert sorted(pool.keys()) == ["running", "starting"]


class _Recorded:
    """A sidecar factory that records what it was built with, then refuses to start."""

    calls: list[tuple[tuple, dict]] = []

    def __init__(self, *args, **kwargs) -> None:
        type(self).calls.append((args, kwargs))
        raise RuntimeError("recorded")


def _launch(monkeypatch, tmp_path, *, session_writes: bool) -> dict:
    import dataclasses

    from osprey.interfaces.web_terminal import app as web_terminal_app
    from osprey.interfaces.web_terminal import operator_session
    from osprey.profiles.sidecar_panels import CANOPY_PANEL_ID, SIDECAR_PANELS

    entry = dataclasses.replace(
        SIDECAR_PANELS[CANOPY_PANEL_ID],
        factory_path=f"{__name__}:_Recorded",
        session_writes=session_writes,
    )
    monkeypatch.setitem(SIDECAR_PANELS, CANOPY_PANEL_ID, entry)
    monkeypatch.setattr(operator_session, "resolve_agent_data_root", lambda _app: str(tmp_path))
    _Recorded.calls = []
    hold_root = tmp_path / "holds"
    app = _app(
        sidecars={},
        panel_auth_headers={},
        sidecars_closing=False,
        sidecar_status={},
        panel_status_root=tmp_path / "status",
        session_hold_root=hold_root,
        pty_registry=_Pool("a-term"),
        operator_registry=_Pool(),
    )
    asyncio.run(web_terminal_app._launch_sidecar(app, CANOPY_PANEL_ID))
    ((_args, kwargs),) = _Recorded.calls
    return {"kwargs": kwargs, "hold_root": hold_root}


def test_a_session_writing_sidecar_is_given_the_hold_store_and_the_live_sessions(
    monkeypatch, tmp_path
):
    launched = _launch(monkeypatch, tmp_path, session_writes=True)

    kwargs = launched["kwargs"]
    assert kwargs["hold_root"] == launched["hold_root"]
    assert kwargs["live_sessions"]() == ["a-term"]


def test_any_other_sidecar_is_built_as_before(monkeypatch, tmp_path):
    assert _launch(monkeypatch, tmp_path, session_writes=False)["kwargs"] == {}
