"""Patch ``subprocess`` for one module without touching the stdlib module.

A patch target that names the module under test is *not* automatically scoped to
it. After a module does ``import subprocess``, ``<module>.subprocess`` **is** the
stdlib module object, so both of these mutate ``sys.modules["subprocess"]``:

    patch("subprocess.run")                       # obviously global
    patch("osprey.cli.chat_cmd.subprocess.run")   # equally global, looks scoped

While either is active the fake is visible to every daemon thread and background
server in the worker, which is the leak this helper exists to avoid. What does
scope the fake is rebinding the *importing module's own* ``subprocess`` name to a
stand-in object, which is what :func:`patch_subprocess` does. The stand-in carries
the real exception classes across, so ``except subprocess.TimeoutExpired`` in the
code under test still catches.

Both spawning entry points are covered. ``run`` is always a fresh ``MagicMock``;
``Popen`` becomes one only when the caller asks, via ``popen=``. That default is
deliberate: a globally-patched ``Popen`` swallows the ``git init`` the exemplar
repo fixture runs — and anything else spawning in this worker — so a test that
needs a fake child process must get it scoped, but a test that only cares about
``run`` must keep the real ``Popen`` working underneath it.

This requires the module under test to import ``subprocess`` at module level. A
function-local ``import subprocess`` re-reads ``sys.modules`` on every call and
cannot be intercepted this way at all.

Usage — as a decorator, where the injected argument is the stand-in::

    @patch_subprocess("osprey.cli.chat_cmd", return_value=Mock(returncode=0))
    def test_launch(self, fake_subprocess, ...):
        ...
        fake_subprocess.run.assert_called_once()

or as a context manager::

    with patch_subprocess("osprey.deployment.container_lifecycle", side_effect=[a, b]) as fake:
        ...

To fake a detached child, pass ``popen=`` — either a ready-made callable, or
``popen=True`` for a plain ``MagicMock`` whose ``return_value`` you configure::

    with patch_subprocess("osprey.cli.web_cmd", popen=True) as fake:
        fake.Popen.return_value.pid = 4321
        ...
        assert fake.Popen.call_args.args[0] == [...]
"""

import subprocess
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

#: Names copied from the real module onto the stand-in. The exception classes
#: must be the real ones or ``except subprocess.X`` in the code under test stops
#: catching; the rest are the attributes CLI code reads besides ``run``.
#: ``Popen`` is carried over so it keeps working by default, and is replaced
#: only when the caller passes ``popen=``.
_CARRIED_OVER = (
    "TimeoutExpired",
    "SubprocessError",
    "CalledProcessError",
    "PIPE",
    "STDOUT",
    "DEVNULL",
    "Popen",
)


def patch_subprocess(target_module: str, *, popen: Any = None, **run_kwargs: Any):
    """Replace ``subprocess`` as seen by ``target_module`` with a stand-in.

    Args:
        target_module: Dotted path of the module under test, e.g.
            ``"osprey.cli.chat_cmd"``. Note this is the module that imports
            ``subprocess``, not the module that defines it.
        popen: How to scope ``Popen``. ``None`` (the default) leaves the real
            stdlib ``Popen`` in place, so unrelated spawning — notably the
            ``git init`` behind the exemplar repo fixture — still works. ``True``
            substitutes a bare ``MagicMock``. Any other value is used as the
            replacement directly, which is how a test supplies its own recorder.
        **run_kwargs: Passed to the ``run`` mock (``return_value``,
            ``side_effect``, ...).

    Returns:
        A patcher usable as either a decorator or a context manager. It supplies
        the stand-in; assert against its ``run`` (and, when scoped, ``Popen``).
    """

    def _stand_in() -> SimpleNamespace:
        ns = SimpleNamespace(**{name: getattr(subprocess, name) for name in _CARRIED_OVER})
        ns.run = MagicMock(**run_kwargs)
        if popen is True:
            ns.Popen = MagicMock()
        elif popen is not None:
            ns.Popen = popen
        return ns

    return patch(f"{target_module}.subprocess", new_callable=_stand_in)
