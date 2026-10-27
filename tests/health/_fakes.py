"""Stand-ins the health suites inject in place of subprocesses and providers."""

from __future__ import annotations

import asyncio
import time


class FakeProc:
    """Minimal stand-in for an ``asyncio`` subprocess.

    ``hang`` makes ``communicate`` outlast any probe deadline; ``killed`` and
    ``reaped`` record whether the caller killed the process and then waited
    on it.
    """

    def __init__(
        self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0, hang: bool = False
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode: int | None = returncode
        self._hang = hang
        self.killed = False
        self.reaped = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang:
            await asyncio.sleep(10)
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int | None:
        self.reaped = True
        return self.returncode


class StubRegistry:
    """Duck-typed stand-in for ``ProviderRegistry`` with a name→class map."""

    def __init__(self, mapping: dict[str, type], non_chat: frozenset[str] = frozenset()) -> None:
        self._mapping = mapping
        self._non_chat = non_chat

    def get_provider(self, name: str) -> type | None:
        return self._mapping.get(name)

    def is_chat(self, name: str) -> bool:
        return name not in self._non_chat


def fake_provider(result: tuple[bool, str] = (True, "ok"), sleep: float = 0.0) -> type:
    """Build a fake provider class returning *result* after an optional sleep."""

    class _Fake:
        def check_health(
            self,
            _api_key: str | None,
            _base_url: str | None,
            _timeout: float = 5.0,
            _model_id: str | None = None,
        ) -> tuple[bool, str]:
            if sleep:
                time.sleep(sleep)
            return result

    return _Fake


class SpyConnector:
    """Minimal async-``disconnect``-only stand-in for a control-system connector."""

    def __init__(self) -> None:
        self.disconnect_calls = 0

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
