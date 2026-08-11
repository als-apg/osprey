"""Container entry point: ``python -m osprey.services.archiver_recorder``.

Startup is deliberately fail-fast. Everything the recorder cannot invent — the
config, the store, the credential, the channel list — is resolved here, before
the loop begins, so a misconfigured deployment says so in ``docker logs``
within seconds instead of running as a service that quietly writes nothing.

What is NOT fatal here is the control system being something other than the one
recorded: that is the ordinary state of a mock deployment, and the service is
supposed to sit there idling until someone flips it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from osprey.utils.logger import configure_logging

from .config import RecorderConfigError, load_settings, resolve_channel_addresses
from .recorder import Recorder
from .store import ArchiveWriter

logger = logging.getLogger(__name__)

#: Same convention as the dispatch worker and the bluesky bridge: the working
#: directory inside the image is not the project, so the config is named
#: explicitly and mounted.
DEFAULT_CONFIG_FILE = "/app/project/config.yml"


async def _run(recorder: Recorder) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # `docker stop` sends SIGTERM; without this the process dies where it
        # stands, mid-write. Asking the loop to stop instead lets the current
        # sample finish and the client close cleanly.
        loop.add_signal_handler(sig, recorder.stop)
    await recorder.run_forever()


def main() -> int:
    configure_logging()

    config_path = Path(os.environ.get("CONFIG_FILE", DEFAULT_CONFIG_FILE))
    try:
        settings = load_settings(config_path)
        addresses = resolve_channel_addresses()
    except RecorderConfigError as exc:
        logger.error("FATAL: %s", exc)
        return 1

    password = os.environ.get(settings.password_env, "")
    if not password:
        logger.error(
            "FATAL: %s is unset, so there is no credential for the archive store. "
            "`osprey deploy up` mints it into the project .env; deploy through it "
            "rather than running compose directly.",
            settings.password_env,
        )
        return 1

    writer = ArchiveWriter(settings, password)
    try:
        writer.connect()
    except ConnectionError as exc:
        logger.error("FATAL: %s", exc)
        return 1

    recorder = Recorder(
        settings=settings,
        addresses=addresses,
        writer=writer,
        config_path=config_path,
    )
    logger.info(
        "Archiver recorder started: %d channels available, enablement re-read from %s every %ds.",
        len(addresses),
        config_path,
        settings.poll_sec,
    )
    try:
        asyncio.run(_run(recorder))
    finally:
        writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
