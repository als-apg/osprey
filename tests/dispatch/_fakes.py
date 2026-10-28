"""Stand-ins shared by the event-dispatcher suites."""

from __future__ import annotations


class FakeEntryPoint:
    """Minimal EntryPoint stand-in exposing .name and .load()."""

    def __init__(self, name: str, cls: type) -> None:
        self.name = name
        self._cls = cls

    def load(self) -> type:
        return self._cls
