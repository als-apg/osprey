"""Providers whose modules take a set time to import, for timed-window tests.

The first lookup of a provider in a process imports its adapter module, which
for a real adapter (and LiteLLM behind it) costs seconds of CPU. A timed call
must not pay that import: the code resolves the class before its timer starts.
:class:`SlowProviders` registers providers whose module sleeps while it
imports, in a fresh provider registry, so a test can show the import ran
before the window and the window held only the call.

The class bodies answer at once and touch no network except
:data:`CHAT_PROBE`, whose health check lists models at the URL the test
serves with :func:`models_listing`.
"""

from __future__ import annotations

import http.server
import itertools
import json
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from osprey.models import provider_registry

_COUNTER = itertools.count()

_HEADER = """\
import time

time.sleep({sleep_s})

from osprey.models.providers.base import BaseProvider
from osprey.models.providers.health import HealthResult


class Provider(BaseProvider):
    name = {name!r}
    description = "a provider whose module sleeps while it imports"
    requires_api_key = False
    requires_base_url = False
    supports_proxy = False
"""

#: A chat route the semantic processor's health check probes by its model listing.
CHAT_PROBE = """
    models_probe = "bearer"

    def execute_completion(self, message, model_id, api_key, base_url, **kwargs):
        return "OK"
"""

#: A text-embedding route whose health check answers healthy.
TEXT_EMBEDDING = """
    def execute_embedding(self, texts, model_id, api_key=None, base_url=None,
                          dimensions=None, timeout=600.0):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def check_embedding_health(self, api_key, base_url, model_id=None, timeout=10.0):
        return HealthResult(True, "ok", None)
"""

#: An image-embedding route whose health check answers healthy.
IMAGE_EMBEDDING = """
    def execute_image_embedding(self, inputs, model_id, api_key=None, base_url=None,
                                dimensions=None, timeout=600.0):
        return [[1.0] + [0.0] * ((dimensions or 4) - 1) for _ in inputs]

    def check_embedding_health(self, api_key, base_url, model_id=None, timeout=10.0):
        return HealthResult(True, "ok", None)
"""


class SlowProviders:
    """Registers slow-importing providers in a fresh provider registry for one test.

    Args:
        tmp_path: Where the provider modules are written.
        monkeypatch: Swaps the registry singleton and puts *tmp_path* on
            ``sys.path`` for the test only.
    """

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._dir = tmp_path
        self._modules: dict[str, str] = {}
        self.registry = provider_registry.ProviderRegistry()
        monkeypatch.setattr(provider_registry, "_registry", self.registry)
        monkeypatch.syspath_prepend(str(tmp_path))

    def register(self, name: str, body: str, *, sleep_s: float) -> None:
        """Register *name* to a class with *body* in a module that sleeps *sleep_s* on import."""
        module = f"slow_provider_{next(_COUNTER)}_{name.replace('-', '_')}"
        source = _HEADER.format(sleep_s=sleep_s, name=name) + body
        (self._dir / f"{module}.py").write_text(source)
        self.registry.register_provider(name, module, "Provider")
        self._modules[name] = module

    def imported(self, name: str) -> bool:
        """Whether the module of the provider registered as *name* has been imported."""
        return self._modules[name] in sys.modules


@contextmanager
def models_listing(model_id: str) -> Iterator[str]:
    """Serve ``GET`` answering ``{"data": [{"id": model_id}]}``; yields the ``/v1`` base URL."""
    body = json.dumps({"data": [{"id": model_id}]}).encode()

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
