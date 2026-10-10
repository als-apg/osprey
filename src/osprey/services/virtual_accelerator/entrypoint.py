"""Virtual accelerator entrypoint.

Serves the simulator view a build writes under ``<data root>/simulator/``,
opened through :class:`~osprey_connectors.simulation.view.SimulatorView`:
one :class:`~osprey_connectors.simulation.composite.Composite` over the view,
served on Channel Access and PVAccess by one
:class:`~osprey.services.virtual_accelerator.serving.runner.ModelRunner`,
which also answers the model RPC. The calling thread is handed to the runner,
which blocks serving until the process is signalled.

Run contract (see docker/virtual-accelerator/README.md for the full version)::

    -v <render>/data:/data:ro                                 # the render's data root
    -v <repo>/var/agent_data/simulation:/state/simulation:ro  # the active scenarios
    -e VA_INSTANCE=virtual_accelerator                        # required
    -e VA_STATE_DIR=/state/simulation
    -p 5064:5064/tcp

Environment:

``VA_INSTANCE``
    **Required.** Which instance this process serves as, one of
    :data:`VA_INSTANCES`. It is written into the composite's log records and
    the model RPC's ``status`` reply. Missing or unknown refuses the boot.
``VA_DATA_DIR``
    The data root the view sits under; :data:`DEFAULT_DATA_DIR` when unset.
``VA_STATE_DIR``
    The directory holding the ``active_scenarios`` file ``osprey sim apply``
    writes; the composite re-reads it when it changes. Unset serves
    ``nominal`` alone.
``VA_POLL_INTERVAL_S``
    The period of the runner's own passes, in seconds, greater than zero;
    :data:`~osprey_connectors.simulation.DEFAULT_TICK_S` when unset.
``VA_MODEL_WRITE_TOKEN``
    The secret a model RPC write must present. Unset refuses every model
    write.

Health:

The runner rewrites its health record to :data:`HEALTH_FILE`,
``/run/osprey-va/health.json``, after every publishing pass; the compose
healthcheck reads that file. The ready line is the boot-time view only.

Importing this module loads neither the composite nor the server extension:
both are imported by :func:`main`.
"""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from typing import Any

from osprey_connectors.simulation.view import (
    ADDRESSES_FILE,
    SERVED_MODELS_FILE,
    VARIABLES_FILE,
    VIEW_RELPATH,
    NoSimulatorView,
    SimulatorView,
    ViewSchemaError,
)

#: The data root the view is read under when ``VA_DATA_DIR`` is unset.
DEFAULT_DATA_DIR = "/data"

#: The instances a process may serve as; the composite's own list names one
#: more, ``inprocess``, which is never a served instance.
VA_INSTANCES = ("virtual_accelerator", "live_standin")

#: The directory the model logs are appended in: the mount target of every
#: virtual accelerator compose block, whose host side is ``var/simulator/`` for
#: the virtual accelerator and ``var/simulator/standin/`` for the live stand-in.
LOG_DIR = Path("/var/simulator")

#: The health record the runner rewrites after every publishing pass: the one
#: path the compose healthcheck reads. Container-local, never a bind mount.
HEALTH_FILE = Path("/run/osprey-va/health.json")

# The line this process prints once the first publishing pass has published
# every served channel, and the marker everything that waits on that boot
# greps for: the image boot check
# (``scripts/va/build_and_boot_check.sh``), the container e2e fixtures, and
# anyone reading ``docker logs``. Both halves are load-bearing -- the marker
# is matched as a prefix, the channel count is read out of the remainder --
# so the whole line is one contract and is written out in exactly one place.
READY_MARKER = "virtual accelerator IOC serving PVs"


def _ready_line(channel_count: int) -> str:
    """The readiness announcement for a namespace of ``channel_count`` channels."""
    return f"{READY_MARKER}: {channel_count} channels"


def view_dir() -> Path:
    """The simulator view this process serves: ``$VA_DATA_DIR/simulator``."""
    data_dir = os.environ.get("VA_DATA_DIR", "").strip() or DEFAULT_DATA_DIR
    return Path(data_dir) / VIEW_RELPATH.name


def _resolve_instance() -> str:
    """Read ``VA_INSTANCE``, one of :data:`VA_INSTANCES`.

    Raises:
        SystemExit: The variable is unset, empty or names no served instance.
    """
    raw = os.environ.get("VA_INSTANCE", "").strip()
    if raw not in VA_INSTANCES:
        raise SystemExit(
            f"FATAL: VA_INSTANCE={raw!r} names no instance. Set it to one of "
            f"{', '.join(VA_INSTANCES)}."
        )
    return raw


def _resolve_tick_interval() -> float:
    """Read ``VA_POLL_INTERVAL_S`` as seconds greater than zero.

    Returns:
        The stated period, or
        :data:`~osprey_connectors.simulation.DEFAULT_TICK_S` when the variable
        is unset or empty.

    Raises:
        SystemExit: The value is not a number, or is zero or below.
    """
    from osprey_connectors.simulation import DEFAULT_TICK_S

    raw = os.environ.get("VA_POLL_INTERVAL_S", "").strip()
    if not raw:
        return DEFAULT_TICK_S
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(
            f"FATAL: VA_POLL_INTERVAL_S={raw!r} is not a number. Set it to seconds "
            f"greater than 0, or unset it for {DEFAULT_TICK_S}."
        ) from None
    if not value > 0:
        raise SystemExit(f"FATAL: VA_POLL_INTERVAL_S={raw!r} must be greater than 0.")
    return value


def _resolve_state_dir() -> Path | None:
    """Read ``VA_STATE_DIR``; ``None`` when unset or empty."""
    raw = os.environ.get("VA_STATE_DIR", "").strip()
    return Path(raw) if raw else None


def _resolve_model_write_token() -> str | None:
    """Resolve ``VA_MODEL_WRITE_TOKEN`` into the secret a model RPC write must
    present, or ``None``.

    ``None`` refuses every model write, and unset resolves to it: the model
    RPC reaches past the served namespace into the physics itself, so it is
    armed by a deployment that says so and by nothing else. Empty counts as
    unset because the compose passthrough sends ``""`` when the host var is
    absent, and an empty token is one an empty credential would match.

    Whitespace alone is unset for the same reason. Anything else is the token
    exactly as given, surrounding spaces included: it is matched byte for byte
    against what a client presents, so rewriting it here would refuse the very
    credential the deployment configured.
    """
    raw = os.environ.get("VA_MODEL_WRITE_TOKEN", "")
    return raw if raw.strip() else None


def _open_view(path: Path) -> SimulatorView:
    """Open the simulator view at ``path`` and read the documents this process serves.

    Raises:
        SystemExit: A document is absent, is not JSON or is from another
            schema; the message names the file and what to do.
    """
    current = path / ADDRESSES_FILE
    try:
        view = SimulatorView.open(path)
        for name in (SERVED_MODELS_FILE, VARIABLES_FILE):
            current = path / name
            view.document(name)
        view.models()
    except (NoSimulatorView, FileNotFoundError):
        raise SystemExit(
            f"FATAL: no simulator view file at {current}. Bind-mount the render's data "
            f"directory (<project>/build/data) to {DEFAULT_DATA_DIR}, or point "
            f"VA_DATA_DIR at it; `osprey build` writes the view."
        ) from None
    except ViewSchemaError as exc:
        raise SystemExit(f"FATAL: {exc}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"FATAL: {current} is not JSON: {exc}") from None
    return view


def _raise_keyboard_interrupt(signum: int, _frame: Any) -> None:
    """Signal handler: turn a stop signal into the interrupt the runner exits on."""
    raise KeyboardInterrupt(f"signal {signum}")


def _install_shutdown_signals() -> None:
    """Make SIGTERM behave exactly as Ctrl-C does.

    The runner's ``run()`` returns on one thing only: a ``KeyboardInterrupt``
    reaching the thread that called it. SIGINT raises one by Python's own
    default; SIGTERM -- what ``docker stop`` sends -- terminates the process
    outright unless a handler says otherwise. Pointing both at one handler
    makes a container stop leave through the runner's own exit. Installed only
    once the first pass has published: before that, dying at once is the right
    answer to a stop signal.
    """
    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


def _configure_logging() -> None:
    """Give the records this process drives a handler on stderr."""
    from osprey.utils.logger import configure_logging

    configure_logging()


def main() -> None:
    """Serve the simulator view until the process is signalled."""
    _configure_logging()

    instance = _resolve_instance()
    tick_interval_s = _resolve_tick_interval()
    state_dir = _resolve_state_dir()
    model_write_token = _resolve_model_write_token()
    view = _open_view(view_dir())
    served_models = view.served()

    print(f"Instance: {instance}", flush=True)
    print(f"Serving the simulator view at {view.path}", flush=True)
    print(f"Serving models: {', '.join(served_models)}", flush=True)
    print(
        f"Active scenarios from {state_dir}"
        if state_dir is not None
        else "Active scenarios: nominal (VA_STATE_DIR unset)",
        flush=True,
    )
    # Whether the model RPC will accept a write is operational state, and an
    # operator reads it out of these lines. The token behind it is a secret
    # and never joins them.
    print(
        "Model writes armed: VA_MODEL_WRITE_TOKEN set"
        if model_write_token
        else "Model writes disabled: VA_MODEL_WRITE_TOKEN unset",
        flush=True,
    )

    from osprey_connectors.simulation.composite import Composite

    composite = Composite(view, state_dir=state_dir, instance=instance, log_dir=LOG_DIR)

    # The runner module reaches the Channel Access server extension at import.
    # Constructing the runner creates the servers and starts serving.
    from osprey.services.virtual_accelerator.serving.runner import ModelRunner

    runner = ModelRunner(
        composite,
        view,
        model_write_token=model_write_token,
        tick_interval_s=tick_interval_s,
        instance=instance,
        health_file=HEALTH_FILE,
    )

    # The first publishing pass runs here, on the thread that runs the loop,
    # so the ready line is printed only once every served channel carries a
    # value the composite computed.
    error = runner.first_pass()
    if error is not None:
        raise SystemExit(f"FATAL: the first publishing pass failed: {error}")

    _install_shutdown_signals()
    print(_ready_line(len(view.channels())), flush=True)

    # `run()` blocks on the run loop and returns on a KeyboardInterrupt, which
    # the handlers above raise for SIGINT and SIGTERM alike; catching it here
    # covers a signal arriving between the loop's own try and this call.
    # Shutdown is process exit: a write already queued when the signal arrives
    # is never applied, so its client's put times out rather than being told
    # a value landed that did not.
    try:
        runner.run()
    except KeyboardInterrupt:
        pass
    print("virtual accelerator IOC stopped", flush=True)


if __name__ == "__main__":
    main()
