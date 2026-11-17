"""Serve an alarm feed's folded snapshot over HTTP, read-only, from beside the broker.

Run it on a host that reaches the alarm server's Kafka broker::

    python -m osprey_connectors.alarm_state.relay \\
        --bootstrap-servers <broker>:<port> --topic Accelerator --host 0.0.0.0

A reader that cannot reach the broker, because only HTTP crosses the network
between them, names the relay's url under ``pv_directory.alarm_state`` with type
``osprey_alarm_relay``. The relay follows the topic as the Kafka backend does,
with no consumer group and nothing produced, and answers four ``GET`` routes:

``/``
    What the service is and its contact with the broker.
``/states``
    Every channel's alarm state, the ``OK`` ones included, with ``stale`` and
    ``last_contact``.
``/state?channel=<name>``
    One channel's state; ``"state": null`` when the feed holds none.
``/config?channel=<name>``
    One channel's place in the alarm tree, with its guidance and displays.

Every other method is refused, so nothing sent here can reach the broker. A feed
with no answer says ``{"unavailable": "<reason>"}`` with status 200, which keeps
the reason through a front end that replaces error bodies.

The server is asyncio streams and nothing more: one request per connection, a
bounded request head, no body read.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import signal
import sys
import urllib.parse
from collections.abc import Sequence
from contextlib import suppress

from osprey_connectors import __version__
from osprey_connectors.alarm_state._contract import AlarmState
from osprey_connectors.alarm_state._phoebus_kafka import KAFKA_CLIENT_MISSING, PhoebusAlarmState
from osprey_connectors.alarm_state._relay import (
    CHANNEL_PARAM,
    CONFIG_PATH,
    ROOT_PATH,
    SERVICE_NAME,
    STATE_PATH,
    STATES_PATH,
    config_payload,
    state_payload,
    states_payload,
    unavailable_payload,
)
from osprey_connectors.logger import get_logger
from osprey_connectors.service_answers import BackendSpec, Unavailable

logger = get_logger("alarm_relay")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8082
_ENV_PREFIX = "OSPREY_ALARM_RELAY_"

#: The longest request head read; a head with no end inside it is refused.
_MAX_HEAD_BYTES = 16 * 1024
#: Seconds a client has to send its request head.
_HEAD_TIMEOUT_S = 10.0
_REASONS = {200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed"}


async def serve(
    feed: AlarmState, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> asyncio.Server:
    """Start answering from *feed* on *host*:*port*; the caller starts and closes the feed."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            status, body = await _answer(feed, reader)
            payload = json.dumps(body, separators=(",", ":")).encode()
            head = (
                f"HTTP/1.1 {status} {_REASONS[status]}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\n"
                + ("Allow: GET\r\n" if status == 405 else "")
                + "Cache-Control: no-store\r\n"
                "Connection: close\r\n\r\n"
            )
            writer.write(head.encode("ascii") + payload)
            await writer.drain()
        except (ConnectionError, TimeoutError):
            pass  # the client left, or never finished its request
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()

    return await asyncio.start_server(handle, host, port, limit=_MAX_HEAD_BYTES)


async def _answer(feed: AlarmState, reader: asyncio.StreamReader) -> tuple[int, dict[str, object]]:
    """The status and JSON body one request gets."""
    try:
        async with asyncio.timeout(_HEAD_TIMEOUT_S):
            head = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
        return 400, {"error": "not an HTTP request this service reads"}
    parts = head.split(b"\r\n", 1)[0].decode("latin-1").split(" ")
    if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
        return 400, {"error": "not an HTTP request this service reads"}
    method, target = parts[0], parts[1]
    if method != "GET":
        return 405, {"error": "this service only answers GET"}
    url = urllib.parse.urlsplit(target)
    path = url.path
    if path == ROOT_PATH:
        return 200, await _root(feed)
    if path == STATES_PATH:
        return 200, states_payload(await feed.states(0))
    if path in (STATE_PATH, CONFIG_PATH):
        channels = urllib.parse.parse_qs(url.query).get(CHANNEL_PARAM, [])
        if len(channels) != 1 or not channels[0]:
            return 400, {"error": f"name one channel: {path}?{CHANNEL_PARAM}=<name>"}
        if path == STATE_PATH:
            return 200, state_payload(await feed.state_for(channels[0], 0))
        return 200, config_payload(await feed.config_for(channels[0], 0))
    routes = f"{ROOT_PATH}, {STATES_PATH}, {STATE_PATH} and {CONFIG_PATH}"
    return 404, {"error": f"no such route; this service answers {routes}"}


async def _root(feed: AlarmState) -> dict[str, object]:
    body: dict[str, object] = {"service": SERVICE_NAME, "version": __version__}
    evidence = await feed.probe(0)
    if isinstance(evidence, Unavailable):
        return body | unavailable_payload(evidence)
    return body | {"evidence": evidence}


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    def env(name: str, default: str | None = None) -> str | None:
        return os.environ.get(_ENV_PREFIX + name) or default

    parser = argparse.ArgumentParser(
        prog="python -m osprey_connectors.alarm_state.relay",
        description="Serve a Phoebus alarm server's state over HTTP, read-only. "
        f"Each option also reads {_ENV_PREFIX}<OPTION>.",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=env("BOOTSTRAP_SERVERS"),
        help="the Kafka broker, host:port",
    )
    parser.add_argument(
        "--topic", default=env("TOPIC"), help="the alarm configuration's topic, e.g. Accelerator"
    )
    parser.add_argument(
        "--host",
        default=env("HOST", DEFAULT_HOST),
        help=f"the address to listen on (default {DEFAULT_HOST}; 0.0.0.0 in a container)",
    )
    parser.add_argument(
        "--port", type=int, default=int(env("PORT", str(DEFAULT_PORT)) or DEFAULT_PORT)
    )
    arguments = parser.parse_args(argv)
    if not arguments.bootstrap_servers or not arguments.topic:
        parser.error("--bootstrap-servers and --topic are required")
    return arguments


async def _run(arguments: argparse.Namespace) -> None:
    feed = PhoebusAlarmState(
        BackendSpec(
            kind="alarm_state",
            type="phoebus_alarm_server",
            target="live",
            settings={
                "bootstrap_servers": arguments.bootstrap_servers,
                "topic": arguments.topic,
            },
            rehearsal=False,
        )
    )
    if importlib.util.find_spec("aiokafka") is None:
        raise SystemExit(KAFKA_CLIENT_MISSING)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    await feed.start()
    server = await serve(feed, host=arguments.host, port=arguments.port)
    logger.info(
        "alarm relay for topic %s on %s:%s", arguments.topic, arguments.host, arguments.port
    )
    try:
        await stop.wait()
    finally:
        server.close()
        await server.wait_closed()
        await feed.aclose()


def main(argv: Sequence[str] | None = None) -> None:
    """Run the relay until it is interrupted."""
    asyncio.run(_run(_arguments(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
