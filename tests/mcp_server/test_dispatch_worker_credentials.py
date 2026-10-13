"""The dispatch worker holds its bearer in memory, not in ``os.environ``.

The worker starts agent runs in its own process tree, and the agent SDK builds
the child's environment as ``{**os.environ, **options.env}`` — an overlay, so a
name merely declined by the clean-env helper is inherited anyway. The only thing
that keeps the worker's bearer from the agent is that it is no longer in
``os.environ`` by the time a run is spawned. These tests pin that, and that the
process guards its memory against same-uid peers once it holds the bearer.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

_WORKER_TOKEN_ENV = "DISPATCH_WORKER_TOKEN"
_DISPATCHER_TOKEN_ENV = "EVENT_DISPATCHER_TOKEN"
_WORKER_TOKEN = "worker-bearer-value-of-some-length"
_LINUX = sys.platform.startswith("linux")


@pytest.fixture(autouse=True)
def _unpopulated_worker(monkeypatch: pytest.MonkeyPatch):
    """Each test starts from an empty holder and carriers it sets itself."""
    from osprey.mcp_server.dispatch_worker import credentials

    monkeypatch.delenv(_WORKER_TOKEN_ENV, raising=False)
    monkeypatch.delenv(_DISPATCHER_TOKEN_ENV, raising=False)
    credentials.reset_worker_credentials()
    yield
    credentials.reset_worker_credentials()


def _bearer(value: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=value)


def _start_worker() -> None:
    """Run the worker's entry point with the server itself stubbed out."""
    from osprey.mcp_server.dispatch_worker.__main__ import main

    with patch("osprey.mcp_server.dispatch_worker.__main__.uvicorn.run"):
        main()


class TestStartup:
    def test_startup_moves_the_bearers_out_of_environ(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from osprey.mcp_server.dispatch_worker import credentials

        monkeypatch.setenv(_WORKER_TOKEN_ENV, _WORKER_TOKEN)
        monkeypatch.setenv(_DISPATCHER_TOKEN_ENV, "dispatcher-bearer-value")

        _start_worker()

        assert _WORKER_TOKEN_ENV not in os.environ
        assert _DISPATCHER_TOKEN_ENV not in os.environ
        assert credentials.worker_credentials().worker_token == _WORKER_TOKEN

    @pytest.mark.skipif(not _LINUX, reason="PR_SET_DUMPABLE is Linux")
    def test_startup_makes_the_process_non_dumpable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from osprey.utils import sensitive_env

        monkeypatch.setenv(_WORKER_TOKEN_ENV, _WORKER_TOKEN)
        assert sensitive_env._prctl(sensitive_env.PR_SET_DUMPABLE, 1) == 0

        _start_worker()

        assert sensitive_env._prctl(sensitive_env.PR_GET_DUMPABLE, 0) == 0


class TestVerifyToken:
    """``/dispatch`` authenticates against the holder, never ``os.environ``."""

    def test_held_bearer_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from osprey.mcp_server.dispatch_worker import dispatch_api

        monkeypatch.setenv(_WORKER_TOKEN_ENV, _WORKER_TOKEN)
        _start_worker()

        dispatch_api._verify_token(_bearer(_WORKER_TOKEN))  # does not raise

    def test_other_bearer_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from osprey.mcp_server.dispatch_worker import dispatch_api

        monkeypatch.setenv(_WORKER_TOKEN_ENV, _WORKER_TOKEN)
        _start_worker()

        with pytest.raises(HTTPException) as refused:
            dispatch_api._verify_token(_bearer("not-the-worker-bearer-value"))
        assert refused.value.status_code in (401, 403)

    def test_unconfigured_worker_refuses_every_bearer(self) -> None:
        from osprey.mcp_server.dispatch_worker import dispatch_api

        with pytest.raises(HTTPException) as refused:
            dispatch_api._verify_token(_bearer(_WORKER_TOKEN))
        assert refused.value.status_code == 500

    def test_bearer_set_after_population_is_not_consulted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The holder is decided once; a later environment value is not a credential."""
        from osprey.mcp_server.dispatch_worker import dispatch_api

        monkeypatch.setenv(_WORKER_TOKEN_ENV, _WORKER_TOKEN)
        _start_worker()
        monkeypatch.setenv(_WORKER_TOKEN_ENV, "a-value-that-appeared-later")

        with pytest.raises(HTTPException):
            dispatch_api._verify_token(_bearer("a-value-that-appeared-later"))


class TestOutputScrub:
    def test_held_bearers_are_scrubbed_from_run_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Masking must still cover a value that is no longer in ``os.environ``."""
        from osprey.mcp_server.dispatch_worker import sdk_runner

        monkeypatch.setenv(_WORKER_TOKEN_ENV, _WORKER_TOKEN)
        monkeypatch.setenv(_DISPATCHER_TOKEN_ENV, "dispatcher-bearer-value")
        _start_worker()

        secrets = sdk_runner._secret_values()

        assert _WORKER_TOKEN in secrets
        assert "dispatcher-bearer-value" in secrets
