"""The environment a generated wrapper script runs in as a real child process.

The wrapper imports osprey in the child (its serializer, its guards). In
production the child inherits the deployment's environment; in a test, this
checkout's own source tree stands in for it.
"""

from __future__ import annotations

import os
from pathlib import Path

#: This checkout's ``src/``, the tree the child imports osprey from.
SRC_ROOT = Path(__file__).resolve().parents[3] / "src"


def child_env(*front: str | os.PathLike[str]) -> dict[str, str]:
    """A copy of ``os.environ`` whose ``PYTHONPATH`` starts with *front*, then ``src/``.

    Args:
        front: Extra directories the child must find before ``src/``, such as a
            sandbox ``site`` holding stand-in modules.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [*map(str, front), str(SRC_ROOT), env.get("PYTHONPATH")])
    )
    return env
