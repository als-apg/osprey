"""Entry point for the Jupyter notebook sidecar subprocess.

On a dev host the sidecar's parent is the OSPREY web terminal process; a parent
poller thread exits the sidecar when that parent dies, since nothing else would
notice an orphaned Jupyter server. In a container the sidecar's parent is PID 1,
so the container's own lifecycle already bounds it and the poller is a no-op.
"""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> None:
    """Start the parent-death poller, then launch the Jupyter server.

    Args:
        argv: Command-line arguments for ``ServerApp``, excluding the program
            name. ``None`` lets ``ServerApp`` fall back to ``sys.argv[1:]``.
    """
    from ipykernel.parentpoller import ParentPollerUnix
    from jupyter_server.serverapp import ServerApp

    ParentPollerUnix(parent_pid=os.getppid()).start()
    ServerApp.launch_instance(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
