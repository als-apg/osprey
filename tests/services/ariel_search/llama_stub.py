"""A llama-server stand-in for the picture-embedding tests.

:class:`LlamaStub` is a real HTTP server on 127.0.0.1 that answers the two
routes the llama-cpp adapter uses:

* ``GET /v1/models`` answers the recorded ``models.json`` of
  ``tests/fixtures/llama_server`` (``data[0].id`` is the served alias);
* ``POST /v1/embeddings`` replays the recorded response whose file name is the
  sha256 of the request's content part, serialised canonically
  (``json.dumps(part, sort_keys=True, separators=(",", ":"))``, the key
  ``record.part_key`` writes). Content with no recording -- every rendition,
  which is always re-encoded -- gets a deterministic 2048-float vector seeded
  from that sha256, so the same picture always embeds to the same vector.

Switches (attributes, settable before or during a test) give the failures a
real server shows:

==================  ==========================================================
``status``          Every route answers this HTTP status (401, 404, ...).
``embed_status``    Only ``/v1/embeddings`` answers it, with ``embed_error``
                    as its message (400 'image input is not supported' is a
                    server started without ``--mmproj``).
``alias``           ``/v1/models`` lists this id instead; ``/v1/embeddings``
                    keeps answering 200, as a server started with another
                    ``--alias`` does.
``hang``            ``/v1/embeddings`` accepts and never answers (until stop).
``silent``          Every route accepts and never answers (until stop).
``malformed``       ``/v1/embeddings`` answers 200 with a body that is not JSON.
``short``           ``/v1/embeddings`` answers a vector of this length.
``zero_calls``      1-based numbers of embedding POSTs answered with a zero vector.
``refuse_after``    After this many embedding answers the server closes its
                    listening socket, so the next connection is refused.
==================  ==========================================================

``models_gets`` and ``embeddings`` count what reached the server.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "llama_server"

#: The alias the recorded server serves.
MODEL = json.loads((FIXTURES / "models.json").read_text())["data"][0]["id"]

#: The width of every vector the recorded model returns.
WIDTH = 2048


def part_key(part: dict[str, Any]) -> str:
    """sha256 of a content part, serialised canonically (the recordings' file names)."""
    canonical = json.dumps(part, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def seeded_vector(key: str, width: int = WIDTH) -> list[float]:
    """The deterministic vector answered for content with no recording."""
    rng = random.Random(int(key, 16))
    return [rng.uniform(-1.0, 1.0) for _ in range(width)]


class LlamaStub:
    """A llama-server stand-in on 127.0.0.1; see the module docstring for its switches."""

    def __init__(self) -> None:
        self.status: int | None = None
        self.embed_status: int | None = None
        self.embed_error = "image input is not supported"
        self.alias: str | None = None
        self.hang = False
        self.silent = False
        self.malformed = False
        self.short: int | None = None
        self.zero_calls: set[int] = set()
        self.refuse_after: int | None = None
        self.models_gets = 0
        self.embeddings: list[dict[str, Any]] = []
        self._release = threading.Event()
        self._lock = threading.Lock()
        self._stopped = False
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return None

            def _send(self, status: int, body: Any, *, raw: bytes | None = None) -> None:
                data = raw if raw is not None else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()

            def _stall(self) -> None:
                stub._release.wait()

            def do_GET(self) -> None:
                if stub.silent:
                    self._stall()
                    return
                if self.path != "/v1/models":
                    self._send(404, {"error": {"message": "not found"}})
                    return
                with stub._lock:
                    stub.models_gets += 1
                if stub.status is not None:
                    self._send(stub.status, {"error": {"code": stub.status}})
                    return
                body = json.loads((FIXTURES / "models.json").read_text())
                if stub.alias is not None:
                    body["data"][0]["id"] = stub.alias
                self._send(200, body)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                request = json.loads(self.rfile.read(length) or b"{}")
                if stub.silent:
                    self._stall()
                    return
                if self.path != "/v1/embeddings":
                    self._send(404, {"error": {"message": "not found"}})
                    return
                with stub._lock:
                    stub.embeddings.append(request)
                    number = len(stub.embeddings)
                if stub.hang:
                    self._stall()
                    return
                if stub.status is not None:
                    self._send(stub.status, {"error": {"code": stub.status}})
                elif stub.embed_status is not None:
                    self._send(
                        stub.embed_status,
                        {"error": {"code": stub.embed_status, "message": stub.embed_error}},
                    )
                elif stub.malformed:
                    self._send(200, None, raw=b"{not json")
                else:
                    self._send(200, stub._answer(request, number))
                if stub.refuse_after is not None and number >= stub.refuse_after:
                    stub.stop()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def _answer(self, request: dict[str, Any], number: int) -> dict[str, Any]:
        """The ``/v1/embeddings`` body for one request."""
        (item,) = request["input"]
        (part,) = item["content"]
        key = part_key(part)
        recorded = FIXTURES / "embeddings" / f"{key}.json"
        if number in self.zero_calls:
            vector = [0.0] * WIDTH
        elif recorded.exists() and self.short is None:
            return json.loads(recorded.read_text())
        else:
            vector = seeded_vector(key, self.short or WIDTH)
        return {
            "model": request.get("model"),
            "object": "list",
            "data": [{"index": 0, "object": "embedding", "embedding": vector}],
        }

    def stop(self) -> None:
        """Close the listening socket (the next connection is refused) and free stalled requests."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        self._release.set()
        self.server.shutdown()
        self.server.server_close()
