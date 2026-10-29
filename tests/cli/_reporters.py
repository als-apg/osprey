"""A reporter that prints into a buffer, and a scoped install of any reporter.

The phase reporter is a process singleton, so a test that installs its own
reporter must put the previous one back. :func:`installed` does that through
``install_reporter``, which quiesces the outgoing reporter (stops its region and
its monitor thread) rather than leaving it running behind the restored one.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

from rich.console import Console

from osprey.cli.phase_reporter import PhaseReporter, install_reporter

_R = TypeVar("_R", bound=PhaseReporter)


class ConsoleReporter(PhaseReporter):
    """A real reporter whose console is a buffer, not the terminal.

    Overrides ``out()``, not ``emit()``: ``out()`` is the seam every
    echo-class primitive resolves per call, and
    :meth:`osprey.cli.phase_reporter.Phase.step` prints through ``emit`` on the
    same console, so one buffer holds both a verb's report lines and the
    ``  · name`` sub-steps hung off its phase. A fake that intercepted ``emit``
    would see nothing a renderer prints through ``out()``.
    """

    def __init__(self, console: Console, *, color: bool = False) -> None:
        super().__init__(color=color)
        self._console = console

    def out(self) -> Console:
        return self._console


@contextmanager
def installed(reporter: _R) -> Iterator[_R]:
    """Install *reporter* for the block, then reinstall whichever it replaced."""
    previous = install_reporter(reporter)
    try:
        yield reporter
    finally:
        install_reporter(previous)
