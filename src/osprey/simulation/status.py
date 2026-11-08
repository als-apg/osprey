"""Every served physics model's status on one control target.

:func:`target_model_statuses` builds the connector a control target selects and
reads each model's status through it: from the composite a simulator served in
process holds, or from each model's status channel on a simulator served by a
process of its own. ``osprey sim status`` and the ``simulator`` health category
read it the same way, so the two can never disagree about a model.
:func:`model_state` sorts a status text into one of three states: the model
serves (:data:`MODEL_OK`), names no deck and wires nothing, so it is not built
on purpose (:data:`MODEL_NO_DECK`), or failed (:data:`MODEL_FAILED`, its text
the error).

:func:`read_in_child` makes that read in a process of its own. A process has one
Channel Access context, and connecting to a target repoints it, closing every
channel the process holds. A caller sharing its process with other Channel
Access users, as the health suite does, must not connect to another target: its
neighbours' next reads would land on that machine. Running this module as a
script is the child's side.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "MODEL_FAILED",
    "MODEL_NO_DECK",
    "MODEL_OK",
    "STATUS_NO_DECK",
    "is_error",
    "model_state",
    "read_in_child",
    "target_model_statuses",
]

#: The status the composite gives a model that names no deck and wires no
#: channel. Spelled here rather than imported: the framework runs against
#: every connectors release down to its floor, and one older than the text
#: never sends it, so a match is never missed, only never made.
STATUS_NO_DECK = "no deck"

try:
    from osprey_connectors.simulation.status import is_error
except ImportError:  # a connectors release older than the predicate

    def is_error(status: str) -> bool:
        """Whether a model's status text is an error: any text but ``ok`` and ``no deck``."""
        return status not in ("ok", STATUS_NO_DECK)


#: The states :func:`model_state` sorts a status text into.
MODEL_OK = "ok"
MODEL_NO_DECK = "no_deck"
MODEL_FAILED = "failed"


def model_state(status: str) -> str:
    """The state a model's status text reports.

    Args:
        status: A model's status, as :func:`target_model_statuses` returns it.

    Returns:
        :data:`MODEL_OK` for ``ok``, :data:`MODEL_NO_DECK` for
        :data:`STATUS_NO_DECK`, and :data:`MODEL_FAILED` for any other text,
        which is the error the model failed with.
    """
    if is_error(status):
        return MODEL_FAILED
    return MODEL_NO_DECK if status == STATUS_NO_DECK else MODEL_OK


#: Prefixes the child's answer: a loaded logger may also write to its stdout.
_ANSWER_MARK = "OSPREY_MODEL_STATUSES "


async def target_model_statuses(
    section: dict, target: str | None, models: list[str], addresses: Mapping[str, str] | None
) -> dict[str, str]:
    """Each model's status, read through the connector built for ``target``.

    The simulator served in process answers from the composite it holds; any
    other connector reads each model's status channel, ``addresses[model]``.

    Args:
        section: The ``control_system:`` config section.
        target: The control target, or ``None`` for the deployment's own.
        models: The served physics models to report.
        addresses: Each model's status address, or ``None`` for a target
            served in process.

    Returns:
        Each model's status text, by model name.
    """
    from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors
    from osprey_connectors.simulation import model_status, read_model_status

    register_builtin_connectors()
    connector = await ConnectorFactory.create_control_system_connector(
        section, control_target=target
    )
    try:
        if addresses is None:
            return {model: model_status(connector, model) for model in models}
        return {model: await read_model_status(connector, addresses[model]) for model in models}
    finally:
        await connector.disconnect()


async def read_in_child(
    section: Mapping[str, Any],
    target: str,
    models: list[str],
    addresses: Mapping[str, str],
    *,
    config_path: Path | None,
    timeout_s: float,
) -> dict[str, str]:
    """:func:`target_model_statuses` for ``target``, run in a child process.

    Args:
        section: The ``control_system:`` config section.
        target: The control target.
        models: The served physics models to report.
        addresses: Each model's status address.
        config_path: The rendered config the child reads as ``CONFIG_FILE``,
            which decides the port of a gateway that states none.
        timeout_s: Seconds the child may take; it is killed after that.

    Returns:
        Each model's status text, by model name.

    Raises:
        TimeoutError: The child did not answer in time.
        RuntimeError: The child failed; the message holds its last output.
    """
    env = dict(os.environ)
    if config_path is not None:
        env["CONFIG_FILE"] = str(config_path)
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        # Safe path: the cwd is the operator's, and code found there must not run.
        "-P",
        "-m",
        "osprey.simulation.status",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    payload = {
        "section": section,
        "target": target,
        "models": models,
        "addresses": dict(addresses),
    }
    try:
        stdout, _ = await asyncio.wait_for(
            child.communicate(json.dumps(payload).encode("utf-8")), timeout_s
        )
    except BaseException:
        child.kill()
        await child.wait()
        raise
    text = stdout.decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        if line.startswith(_ANSWER_MARK):
            answer: dict[str, str] = json.loads(line.removeprefix(_ANSWER_MARK))
            return answer
    raise RuntimeError(text.strip()[-500:] or f"the status reader exited {child.returncode}")


def _main() -> None:
    """Read the statuses a JSON payload on stdin asks for; print them as one marked line.

    It leaves through ``os._exit``: tearing down a Channel Access context at
    interpreter exit can hang, and the answer is already out.
    """
    payload = json.load(sys.stdin)
    statuses = asyncio.run(
        target_model_statuses(
            payload["section"], payload["target"], payload["models"], payload["addresses"]
        )
    )
    sys.stdout.write(f"\n{_ANSWER_MARK}{json.dumps(statuses)}\n")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    _main()
