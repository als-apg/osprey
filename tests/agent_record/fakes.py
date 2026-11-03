"""Test doubles for the agent record's readers.

:class:`FakeRecordClient` stands in for
:class:`~osprey.agent_record.client.RecordClient`: the same read methods, each
answering a canned :class:`~osprey.agent_record.client.ClientAnswer` and
recording the call, so a reader's tests can pin what it asked for and how it
renders a complete, an incomplete and an unavailable answer without a service.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from osprey.agent_record.client import READ_NAMES, ClientAnswer, RecordClient

__all__ = ["FakeCall", "FakeRecordClient"]

Canned = ClientAnswer | Callable[..., ClientAnswer]


@dataclass(frozen=True)
class FakeCall:
    """One read the fake was asked for: its method name and the arguments the
    caller passed, keyed by the real client's parameter names."""

    name: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)


def _bind(name: str) -> Callable[..., ClientAnswer]:
    signature = inspect.signature(getattr(RecordClient, name))

    def call(self: FakeRecordClient, *args: Any, **kwargs: Any) -> ClientAnswer:
        # Binding against the real signature refuses a call the client would.
        bound = signature.bind(self, *args, **kwargs).arguments
        bound.pop("self")
        return self._answer(name, dict(bound))

    call.__name__ = name
    return call


class FakeRecordClient:
    """A record client answering what the test says.

    Args:
        answers: Per read name, a :class:`ClientAnswer` or a callable taking
            the call's keyword arguments and returning one. A list answers
            one item per call, the last repeating.
        default: What a read with no canned answer answers; a complete,
            empty answer when ``None``.
        unavailable: When set, every read answers unavailable with this
            reason, whatever is canned.
        login: The login the fake stands for.
        viewer: The viewer it names.
    """

    def __init__(
        self,
        answers: Mapping[str, Canned | list[Canned]] | None = None,
        *,
        default: ClientAnswer | None = None,
        unavailable: str | None = None,
        login: str = "fake",
        viewer: str | None = None,
    ) -> None:
        self.answers: dict[str, Canned | list[Canned]] = dict(answers or {})
        self.default = default if default is not None else ClientAnswer(data=[], complete=True)
        self.unavailable = unavailable
        self.login = login
        self.viewer = viewer
        self.calls: list[FakeCall] = []

    def calls_to(self, name: str) -> list[Mapping[str, Any]]:
        """The keyword arguments of every call to *name*, in order."""
        return [c.kwargs for c in self.calls if c.name == name]

    def _answer(self, name: str, kwargs: dict[str, Any]) -> ClientAnswer:
        self.calls.append(FakeCall(name, kwargs))
        if self.unavailable is not None:
            return ClientAnswer.failed(self.unavailable)
        canned = self.answers.get(name, self.default)
        if isinstance(canned, list):
            index = len(self.calls_to(name)) - 1
            canned = canned[min(index, len(canned) - 1)]
        if callable(canned) and not isinstance(canned, ClientAnswer):
            return canned(**kwargs)
        return canned


for _name in READ_NAMES:
    setattr(FakeRecordClient, _name, _bind(_name))
