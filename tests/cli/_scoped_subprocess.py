"""Patch ``subprocess.run`` for one module without touching the stdlib module.

A patch target that names the module under test is *not* automatically scoped to
it. After a module does ``import subprocess``, ``<module>.subprocess`` **is** the
stdlib module object, so both of these mutate ``sys.modules["subprocess"]``:

    patch("subprocess.run")                       # obviously global
    patch("osprey.cli.claude_cmd.subprocess.run") # equally global, looks scoped

While either is active the fake is visible to every daemon thread and background
server in the worker, which is the leak this helper exists to avoid. What does
scope the fake is rebinding the *importing module's own* ``subprocess`` name to a
stand-in object, which is what :func:`patch_subprocess` does. The stand-in carries
the real exception classes across, so ``except subprocess.TimeoutExpired`` in the
code under test still catches.

This requires the module under test to import ``subprocess`` at module level. A
function-local ``import subprocess`` re-reads ``sys.modules`` on every call and
cannot be intercepted this way at all.

Usage — as a decorator, where the injected argument is the stand-in::

    @patch_subprocess("osprey.cli.claude_cmd", return_value=Mock(returncode=0))
    def test_launch(self, fake_subprocess, ...):
        ...
        fake_subprocess.run.assert_called_once()

or as a context manager::

    with patch_subprocess("osprey.cli.interactive_menu", side_effect=[ps, inspect]) as fake:
        ...
"""

import subprocess
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

#: Names copied from the real module onto the stand-in. The exception classes
#: must be the real ones or ``except subprocess.X`` in the code under test stops
#: catching; the rest are the attributes CLI code reads besides ``run``.
_CARRIED_OVER = (
    "TimeoutExpired",
    "SubprocessError",
    "CalledProcessError",
    "PIPE",
    "STDOUT",
    "DEVNULL",
    "Popen",
)


def patch_subprocess(target_module: str, **run_kwargs: Any):
    """Replace ``subprocess`` as seen by ``target_module`` with a stand-in.

    Args:
        target_module: Dotted path of the module under test, e.g.
            ``"osprey.cli.claude_cmd"``. Note this is the module that imports
            ``subprocess``, not the module that defines it.
        **run_kwargs: Passed to the ``run`` mock (``return_value``,
            ``side_effect``, ...).

    Returns:
        A patcher usable as either a decorator or a context manager. It supplies
        the stand-in; assert against its ``run`` attribute.
    """

    def _stand_in() -> SimpleNamespace:
        ns = SimpleNamespace(**{name: getattr(subprocess, name) for name in _CARRIED_OVER})
        ns.run = MagicMock(**run_kwargs)
        return ns

    return patch(f"{target_module}.subprocess", new_callable=_stand_in)
