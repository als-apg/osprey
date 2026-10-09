"""A default deployment with no llama-server and no caption model runs clean.

The ARIEL config is the ``control-assistant`` preset as the build reads it, so
``image_caption``, ``image_embedding``, ``hybrid`` and
``ariel.attachments.view.enabled`` are on as shipped. It deviates from the
preset by one named override, ``ariel.ingestion`` (a ``generic_json`` file
source whose entry carries two sidecar PNGs, so no network and no
``allowed_origins``), besides the database URI. The test runs ``chdir``-ed into
``tmp_path`` (the preset's ``qmd_export`` mirror path is relative) with qmd
faked at its client boundary.

Every model server is in-test or closed: ``OLLAMA_HOST`` names an Ollama stub
(or a closed port), ``LLAMA_CPP_HOST`` a closed port (or the shared
``llama_stub``), the configured provider URLs and every container fallback are
closed local ports, so a developer's own Ollama or llama-server is never
reached and the outcome does not depend on the host.

Each case runs one two-picture ingest, two ``run_sync`` passes, ``osprey ariel
status`` and the MCP tools ``keyword_search``, ``hybrid_search``,
``capabilities`` and ``attachment_view``, and pins that a missing picture
service costs nothing but its own module: no exception, no ERROR log record, no
picture status key or row, one WARNING per picture module, and a status that
names each skipped module and its fix.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import psycopg
import pytest

import osprey
from osprey.models.providers import _local_server
from osprey.services.ariel_search import cli_operations as ops
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.migrations import image_embedding_target
from osprey.services.ariel_search.enhancement import _offload, availability
from osprey.services.ariel_search.search import image_lane
from osprey.services.ariel_search.search import qmd as qmd_module
from osprey.services.ariel_search.search.qmd import PICTURE_UNAVAILABLE_MESSAGE
from osprey.services.qmd import QMDSearchResult
from tests.mcp_server.conftest import get_tool_fn
from tests.services.ariel_search.llama_stub import FIXTURES

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.xdist_group("docker"),
    pytest.mark.timeout(300),
    pytest.mark.real_fetch,
]

PRESET = "control-assistant"
CAPTION = "image_caption"
EMBED = "image_embedding"
PICTURE_MODULES = (CAPTION, EMBED)
TEXT_EMBED = "text_embedding"
QMD_EXPORT = "qmd_export"
LANE = image_lane.TRACKER_KEY
ENTRY_ID = "degrade-1"
TITLE = "Beam lost at injection"
TEXT = "Beam lost at 14:02 after the kicker fired; orbit plot and tunnel temperature attached."
PICTURES = ("orbit_kick.png", "tunnel_temp.png")
TEXT_MODEL = "nomic-embed-text"
TEXT_DIMS = 768
#: An alias the llama-server stub serves that is not the preset's model.
OTHER_ALIAS = "some-other-embedding-model"
#: The vocabulary the control-assistant build ships at ``data/ariel/vocabulary.yml``.
VOCABULARY = (
    Path(osprey.__file__).parent
    / "templates"
    / "apps"
    / "control_assistant"
    / "data"
    / "ariel"
    / "vocabulary.yml"
)


# --- servers -----------------------------------------------------------------------


def _closed_url() -> str:
    """A local URL nothing listens on: a port bound once, then released."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


class OllamaStub:
    """An Ollama stand-in on 127.0.0.1 that serves the text embedding model only.

    ``GET /api/tags`` and ``GET /v1/models`` list ``nomic-embed-text``;
    ``POST /api/show`` answers it and 404s every other model (the caption model
    is not pulled); ``POST /api/embed`` (and the older ``/api/embeddings``)
    answer 768-float vectors.
    """

    def __init__(self) -> None:
        stub = self
        self.embeds = 0
        self.shows: list[str] = []

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return

            def _send(self, status: int, body: Any) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    return json.loads(raw or b"{}")
                except ValueError:
                    return {}

            def do_GET(self) -> None:
                if self.path == "/api/tags":
                    self._send(
                        200,
                        {
                            "models": [
                                {"name": f"{TEXT_MODEL}:latest", "model": f"{TEXT_MODEL}:latest"}
                            ]
                        },
                    )
                elif self.path == "/v1/models":
                    self._send(200, {"object": "list", "data": [{"id": f"{TEXT_MODEL}:latest"}]})
                elif self.path == "/api/version":
                    self._send(200, {"version": "0.0.0-stub"})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self) -> None:
                body = self._body()
                if self.path == "/api/show":
                    model = str(body.get("model") or body.get("name") or "")
                    stub.shows.append(model)
                    if model.split(":")[0] == TEXT_MODEL:
                        self._send(200, {"capabilities": ["embedding"]})
                    else:
                        self._send(404, {"error": f"model '{model}' not found"})
                elif self.path == "/api/embed":
                    texts = body.get("input")
                    count = len(texts) if isinstance(texts, list) else 1
                    stub.embeds += count
                    self._send(
                        200,
                        {
                            "model": body.get("model"),
                            "embeddings": [_vector(i) for i in range(count)],
                            "prompt_eval_count": 8 * count,
                        },
                    )
                elif self.path == "/api/embeddings":
                    stub.embeds += 1
                    self._send(200, {"embedding": _vector(0)})
                else:
                    self._send(404, {"error": "not found"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _vector(seed: int) -> list[float]:
    return [((i + seed) % 17) / 17.0 + 0.01 for i in range(TEXT_DIMS)]


@pytest.fixture
def ollama_stub() -> Iterator[OllamaStub]:
    stub = OllamaStub()
    yield stub
    stub.stop()


# --- fakes -------------------------------------------------------------------------


class _FakeQMDClient:
    """qmd faked at its client boundary: one hit, the entry's mirror file."""

    is_configured = True
    base_url = "http://127.0.0.1:8180"

    def __init__(self, mirror_file: str, title: str) -> None:
        self.mirror_file = mirror_file
        self.title = title
        self.queries = 0

    def is_available(self) -> bool:
        return True

    def query(self, collection: str | None, _text: str, **kwargs: Any) -> list[QMDSearchResult]:
        self.queries += 1
        hits = [
            QMDSearchResult(
                docid="#000000",
                file=self.mirror_file,
                collection=collection or "ariel",
                title=self.title,
                score=0.9,
                line=1,
                snippet=f"1: {TEXT}",
            )
        ]
        return hits[: kwargs.get("limit", len(hits))]


# --- config ------------------------------------------------------------------------


def _preset_ariel() -> dict[str, Any]:
    """The preset's ``ariel`` config as the build reads it, dotted keys expanded."""
    from osprey.cli.build_profile_archiver import _expand_dotted
    from osprey.cli.build_profile_merge import _resolve_extends
    from osprey.cli.build_profile_presets import _load_preset_raw

    raw, path = _load_preset_raw(PRESET)
    document = _resolve_extends(dict(raw), path)
    return copy.deepcopy(_expand_dotted(dict(document.get("config") or {}))["ariel"])


def _leaves(tree: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(tree, dict):
        out: dict[str, Any] = {}
        for key, value in tree.items():
            out.update(_leaves(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: tree}


def _write_source(tmp_path: Path) -> Path:
    """The generic_json file source: one entry, two relative sidecar PNGs.

    The preset's vocabulary file is laid down where the build ships it
    (``data/ariel/vocabulary.yml`` beside ``config.yml``).
    """
    vocabulary = tmp_path / "data" / "ariel" / "vocabulary.yml"
    vocabulary.parent.mkdir(parents=True, exist_ok=True)
    vocabulary.write_bytes(VOCABULARY.read_bytes())
    for name in PICTURES:
        (tmp_path / name).write_bytes((FIXTURES / name).read_bytes())
    source = tmp_path / "entries.json"
    source.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": ENTRY_ID,
                        "title": TITLE,
                        "text": TEXT,
                        "author": "operator",
                        "timestamp": "2026-09-01T10:00:00+00:00",
                        "attachments": [
                            {"url": name, "type": "image/png", "filename": name}
                            for name in PICTURES
                        ],
                    }
                ]
            }
        )
    )
    return source


def _deployment_config(uri: str, source: Path) -> dict[str, Any]:
    """The preset's ``ariel`` block with the database and the ingestion source set."""
    raw = _preset_ariel()
    raw["database"] = {"uri": uri}
    raw["ingestion"] = {"adapter": "generic_json", "source_url": str(source)}
    return raw


# --- observation -------------------------------------------------------------------


@dataclass
class Observed:
    """What one case saw, for the assertions and the switch-off comparisons."""

    raw: dict[str, Any]
    status_rows: dict[str, Any]
    raw_text: str
    captions: Any
    copy_rows: dict[str, tuple[str, str | None]]
    image_table: str
    image_rows: int | None
    pass_records: list[logging.LogRecord]
    status_doc: dict[str, Any]
    status_text: str
    keyword: dict[str, Any]
    hybrid_text: str | None
    hybrid: dict[str, Any] | None
    caps_after: dict[str, Any]
    view_kinds: set[str] = field(default_factory=set)
    errors: list[logging.LogRecord] = field(default_factory=list)


def _availability_warnings(records: list[logging.LogRecord], module: str) -> list[str]:
    """The availability tracker's WARNINGs naming *module*.

    The shared ``ariel`` logger also carries other modules' WARNINGs, and every
    record of the component-logger wrapper reports its own file as
    ``pathname``, so the tracker's records are told apart by the line only
    ``availability.report_unavailable`` writes:
    ``<module>: unavailable (<reason>)...; skipped until fixed (fix: ...)``.
    """
    pattern = re.compile(
        rf"^{re.escape(module)}: unavailable \(\w+\).*; skipped until fixed \(fix: "
    )
    return [
        r.getMessage()
        for r in records
        if r.levelno == logging.WARNING and pattern.match(r.getMessage())
    ]


def _db_state(uri: str, image_table: str) -> tuple[dict, str, Any, dict, int | None]:
    with psycopg.connect(uri) as conn:
        row = conn.execute(
            "SELECT enhancement_status, raw_text, attachment_captions FROM enhanced_entries"
            " WHERE entry_id = %s",
            (ENTRY_ID,),
        ).fetchone()
        assert row is not None, "the entry was not stored"
        files = conn.execute(
            "SELECT attachment_id, copy_status, rendition_sha256 FROM attachment_files"
            " WHERE entry_id = %s",
            (ENTRY_ID,),
        ).fetchall()
        exists = conn.execute("SELECT to_regclass(%s)", (image_table,)).fetchone()
        image_rows = None
        if exists is not None and exists[0] is not None:
            counted = conn.execute(f'SELECT count(*) FROM "{image_table}"').fetchone()
            image_rows = counted[0] if counted else 0
    return (
        row[0] or {},
        row[1],
        row[2],
        {f[0]: (f[1], f[2]) for f in files},
        image_rows,
    )


def _image_table_exists(uri: str, image_table: str) -> bool:
    with psycopg.connect(uri) as conn:
        row = conn.execute("SELECT to_regclass(%s)", (image_table,)).fetchone()
    return row is not None and row[0] is not None


def _status(config: dict[str, Any], *, as_json: bool) -> str:
    """``osprey ariel status`` on *config*; returns stdout."""
    from click.testing import CliRunner

    from osprey.cli.ariel import ariel_group

    with (
        patch("osprey.cli.ariel.get_config_value", return_value=config),
        patch("osprey.imaging.render.probe_render_worker", new=AsyncMock(return_value=True)),
    ):
        result = CliRunner().invoke(ariel_group, ["status", "--json"] if as_json else ["status"])
    assert result.exit_code == 0, result.output
    return result.stdout


def _skip_line(name: str, reason: str, config: dict[str, Any]) -> str:
    from osprey.cli.ariel import module_skip_line

    with patch("osprey.cli.ariel.get_config_value", return_value=config):
        return module_skip_line(name, reason, config)


def _register_tools() -> None:
    from osprey.mcp_server.ariel.tools import (  # noqa: F401
        attachment,
        capabilities,
        hybrid_search,
        keyword_search,
    )


def _mirror_hit(tmp_path: Path) -> tuple[str, str]:
    """The entry's mirror path (relative to the mirror root) and its title, as qmd names a hit."""
    root = tmp_path / "var" / "ariel_mirror"
    files = [p for p in root.rglob("*.md") if not p.name.startswith(".")]
    assert len(files) == 1, files
    heading = next(
        line[2:].strip() for line in files[0].read_text().splitlines() if line.startswith("# ")
    )
    return files[0].relative_to(root).as_posix(), heading


# --- fixtures ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Fresh process state, ``cwd`` in ``tmp_path``, every server URL closed by default."""
    from osprey.mcp_server.ariel.server_context import reset_ariel_context
    from osprey.utils.workspace import reset_config_cache

    availability.reset_availability()
    _offload.reset_offload_state()
    reset_ariel_context()
    reset_config_cache()
    qmd_module._reset_client_cache()
    image_lane._reset_state()
    _local_server.reset_cache()
    monkeypatch.chdir(tmp_path)

    closed = {"ollama": _closed_url(), "llama-cpp": _closed_url()}
    fallbacks = [_closed_url(), _closed_url()]
    # The api.providers entries point at closed ports: an env override names
    # the in-test server, and nothing ever reaches the host's own servers.
    monkeypatch.setattr(
        "osprey.models.config.get_provider_config",
        lambda name, *a, **k: {"base_url": closed[name]} if name in closed else {},
    )
    monkeypatch.setattr(
        _local_server, "container_fallback_urls", lambda base_url, default_port: list(fallbacks)
    )
    monkeypatch.setenv("OLLAMA_HOST", _closed_url())
    monkeypatch.setenv("LLAMA_CPP_HOST", _closed_url())
    yield
    reset_ariel_context()
    reset_config_cache()
    qmd_module._reset_client_cache()
    availability.reset_availability()
    _offload.reset_offload_state()
    image_lane._reset_state()
    _local_server.reset_cache()


# --- the case driver ---------------------------------------------------------------


async def _run_case(
    raw: dict[str, Any],
    uri: str,
    tmp_path: Path,
    monkeypatch,
    caplog,
    *,
    hybrid_calls: int = 1,
    clock: list[float] | None = None,
) -> Observed:
    """Two sync passes, status, and the MCP tools on *raw*; returns what was seen."""
    from osprey.mcp_server.ariel.server_context import initialize_ariel_context
    from osprey.mcp_server.ariel.tools.attachment import attachment_view
    from osprey.mcp_server.ariel.tools.capabilities import capabilities
    from osprey.mcp_server.ariel.tools.hybrid_search import hybrid_search
    from osprey.mcp_server.ariel.tools.keyword_search import keyword_search
    from osprey.services.ariel_search.service import create_ariel_service

    caplog.set_level(logging.DEBUG, logger="ariel")
    caplog.set_level(logging.INFO)
    config = ARIELConfig.from_dict(raw)
    image_table = image_embedding_target(
        config.get_enhancement_module_config(EMBED) or _preset_ariel()["enhancement_modules"][EMBED]
    ).table

    (tmp_path / "config.yml").write_text(json.dumps({"ariel": raw}))

    # -- two passes: ingest, then catch-up ---------------------------------------
    await ops.run_sync(raw)
    await ops.run_sync(raw)
    pass_records = list(caplog.records)

    status_rows, raw_text, captions, copy_rows, image_rows = _db_state(uri, image_table)

    # -- osprey ariel status -----------------------------------------------------
    status_doc = json.loads(await asyncio.to_thread(_status, raw, as_json=True))
    status_text = await asyncio.to_thread(_status, raw, as_json=False)

    # -- the MCP surface ---------------------------------------------------------
    hybrid_on = bool(raw["search_modules"]["hybrid"]["enabled"])
    qmd_client = _FakeQMDClient(*_mirror_hit(tmp_path)) if hybrid_on else None
    if qmd_client is not None:
        monkeypatch.setattr(qmd_module, "_resolve_client", lambda client: (qmd_client, True))
    initialize_ariel_context()
    _register_tools()

    hybrid_text: str | None = None
    hybrid_doc: dict[str, Any] | None = None
    view_kinds: set[str] = set()
    service = await create_ariel_service(config)
    try:
        with patch(
            "osprey.mcp_server.ariel.server_context.ARIELContext.service",
            new=AsyncMock(return_value=service),
        ):
            keyword = json.loads(await get_tool_fn(keyword_search)(query="beam"))
            if hybrid_on:
                for _ in range(hybrid_calls):
                    hybrid_text = await get_tool_fn(hybrid_search)(query="beam lost")
                    if clock is not None:
                        clock[0] += 120.0 / hybrid_calls
                assert hybrid_text is not None
                hybrid_doc = json.loads(hybrid_text)
            caps_after = json.loads(await get_tool_fn(capabilities)())
            for aid in copy_rows:
                result = await get_tool_fn(attachment_view)(attachment_id=aid)
                view_kinds |= {getattr(part, "type", "") for part in result.content}
    finally:
        await service.__aexit__(None, None, None)

    return Observed(
        raw=raw,
        status_rows=status_rows,
        raw_text=raw_text,
        captions=captions,
        copy_rows=copy_rows,
        image_table=image_table,
        image_rows=image_rows,
        pass_records=pass_records,
        status_doc=status_doc,
        status_text=status_text,
        keyword=keyword,
        hybrid_text=hybrid_text,
        hybrid=hybrid_doc,
        caps_after=caps_after,
        view_kinds=view_kinds,
        errors=[r for r in caplog.records if r.levelno >= logging.ERROR],
    )


def _health(observed: Observed, module: str) -> dict[str, Any] | None:
    return observed.status_doc["enhancement_modules"][module].get("health")


def _assert_runs_clean(observed: Observed) -> None:
    """No ERROR record; the entry is stored; its pictures are copied and viewable."""
    assert observed.errors == [], [r.getMessage() for r in observed.errors]
    assert TITLE in observed.raw_text and "kicker fired" in observed.raw_text
    assert len(observed.copy_rows) == len(PICTURES), observed.copy_rows
    for status, sha in observed.copy_rows.values():
        assert status == "copied"
        assert sha, "a copied picture carries its rendition"
    assert "image" in observed.view_kinds
    assert [e["entry_id"] for e in observed.keyword["entries"]] == [ENTRY_ID]


def _assert_picture_modules_skipped(observed: Observed, modules=PICTURE_MODULES) -> None:
    """No status key, no caption, no image row, one WARNING per enabled picture module."""
    for module in PICTURE_MODULES:
        assert module not in observed.status_rows, (module, observed.status_rows)
    assert not observed.captions, observed.captions
    assert not observed.image_rows, observed.image_rows
    for module in modules:
        warnings = _availability_warnings(observed.pass_records, module)
        assert len(warnings) == 1, (module, warnings)


def _assert_text_only_hybrid(observed: Observed) -> None:
    assert observed.hybrid is not None and observed.hybrid_text is not None
    assert [e["entry_id"] for e in observed.hybrid["entries"]] == [ENTRY_ID]
    assert "Picture search unavailable" in observed.hybrid_text
    assert PICTURE_UNAVAILABLE_MESSAGE in json.dumps(observed.hybrid, ensure_ascii=False)


# --- the main case: the shipped default most sites meet ------------------------------


async def _main_case(
    raw, scratch_database, tmp_path, monkeypatch, caplog, ollama_stub, **kwargs
) -> Observed:
    monkeypatch.setenv("OLLAMA_HOST", ollama_stub.url)
    return await _run_case(raw, scratch_database, tmp_path, monkeypatch, caplog, **kwargs)


async def test_default_deployment_without_picture_servers_runs_clean(
    scratch_database, ollama_stub, monkeypatch, caplog, tmp_path
):
    source = _write_source(tmp_path)
    raw = _deployment_config(scratch_database, source)

    # Exactly one override besides the database: the ingestion source.
    preset = _leaves(_preset_ariel())
    deployed = {
        k: v for k, v in _leaves(raw).items() if not k.startswith(("database.", "ingestion."))
    }
    assert deployed == preset
    assert raw["ingestion"] == {"adapter": "generic_json", "source_url": str(source)}

    observed = await _main_case(raw, scratch_database, tmp_path, monkeypatch, caplog, ollama_stub)

    _assert_runs_clean(observed)
    assert observed.status_rows[TEXT_EMBED]["status"] == "complete", observed.status_rows
    assert observed.status_rows[QMD_EXPORT]["status"] == "complete", observed.status_rows
    assert ollama_stub.embeds > 0
    _assert_picture_modules_skipped(observed)

    assert _health(observed, CAPTION)["reachable"] is False
    assert _health(observed, CAPTION)["reason"] == "model"
    assert _health(observed, EMBED)["reachable"] is False
    assert _health(observed, EMBED)["reason"] == "unreachable"
    assert _skip_line(CAPTION, "model", raw) in observed.status_text
    assert _skip_line(EMBED, "unreachable", raw) in observed.status_text
    assert "pull it" in observed.status_text
    assert "start llama-server" in observed.status_text

    _assert_text_only_hybrid(observed)
    assert observed.caps_after["attachments"]["picture_search"] is True
    assert observed.caps_after["attachments"]["picture_search_unavailable"] == "unreachable"


@pytest.mark.parametrize(
    "switch",
    [
        ("enhancement_modules", CAPTION),
        ("enhancement_modules", EMBED),
        ("search_modules", "hybrid"),
    ],
    ids=["caption-off", "embedding-off", "hybrid-off"],
)
async def test_each_module_switches_off_alone(
    switch, scratch_database, ollama_stub, monkeypatch, caplog, tmp_path
):
    """With one module off: no health entry for it, the rest as in the main case."""
    group, name = switch
    raw = _deployment_config(scratch_database, _write_source(tmp_path))
    raw[group][name]["enabled"] = False
    if name == "hybrid":
        # The preset's default mode is hybrid; a deployment that turns hybrid off
        # names another default, or the config is refused.
        raw["default_search_mode"] = "keyword"

    observed = await _main_case(raw, scratch_database, tmp_path, monkeypatch, caplog, ollama_stub)

    _assert_runs_clean(observed)
    assert observed.status_rows[TEXT_EMBED]["status"] == "complete"
    assert observed.status_rows[QMD_EXPORT]["status"] == "complete"
    on = [m for m in PICTURE_MODULES if m != name]
    _assert_picture_modules_skipped(observed, modules=on)

    modules = observed.status_doc["enhancement_modules"]
    if group == "enhancement_modules":
        assert "health" not in modules[name], modules[name]
        assert _availability_warnings(observed.pass_records, name) == []
    else:
        assert observed.status_doc["search_modules"]["hybrid"] is False
        assert observed.hybrid is None
    expected = {CAPTION: "model", EMBED: "unreachable"}
    if name == "hybrid":
        # With hybrid off nothing reads picture vectors: the module's own
        # health reason says so ahead of the unreachable server.
        expected[EMBED] = "no_reader"
    for module in on:
        health = _health(observed, module)
        assert health is not None and health["reachable"] is False
        assert health["reason"] == expected[module]
        assert _skip_line(module, expected[module], raw) in observed.status_text
    assert observed.caps_after["attachments"]["picture_search"] is (name not in (EMBED, "hybrid"))
    if name == CAPTION:
        _assert_text_only_hybrid(observed)


# --- second case: no Ollama either ---------------------------------------------------


async def test_no_ollama_either(scratch_database, monkeypatch, caplog, tmp_path):
    raw = _deployment_config(scratch_database, _write_source(tmp_path))

    observed = await _run_case(raw, scratch_database, tmp_path, monkeypatch, caplog)

    _assert_runs_clean(observed)
    text = observed.status_rows.get(TEXT_EMBED)
    assert text is not None and text["status"] == "failed", observed.status_rows
    assert observed.status_rows[QMD_EXPORT]["status"] == "complete"
    _assert_picture_modules_skipped(observed)

    assert _health(observed, CAPTION)["reason"] == "unreachable"
    assert _health(observed, EMBED)["reason"] == "unreachable"
    assert _skip_line(CAPTION, "unreachable", raw) in observed.status_text
    assert _skip_line(EMBED, "unreachable", raw) in observed.status_text
    _assert_text_only_hybrid(observed)
    assert observed.caps_after["attachments"]["picture_search_unavailable"] == "unreachable"


# --- third case: a llama-server serving another alias --------------------------------


async def test_llama_server_serving_another_model(
    scratch_database, ollama_stub, llama_stub, monkeypatch, caplog, tmp_path
):
    stub = llama_stub()
    stub.alias = OTHER_ALIAS
    monkeypatch.setenv("LLAMA_CPP_HOST", stub.url)
    raw = _deployment_config(scratch_database, _write_source(tmp_path))

    observed = await _main_case(raw, scratch_database, tmp_path, monkeypatch, caplog, ollama_stub)

    _assert_runs_clean(observed)
    _assert_picture_modules_skipped(observed)
    assert _health(observed, EMBED)["reason"] == "model"
    assert _health(observed, CAPTION)["reason"] == "model"
    assert observed.image_rows in (None, 0)
    _assert_text_only_hybrid(observed)
    assert observed.caps_after["attachments"]["picture_search_unavailable"] == "model"


# --- fourth case: a store without pgvector -------------------------------------------


async def test_store_without_pgvector(
    scratch_database, ollama_stub, llama_stub, monkeypatch, caplog, tmp_path
):
    from osprey.services.ariel_search.enhancement.image_embedding import (
        migration as image_migration,
    )
    from osprey.services.ariel_search.enhancement.text_embedding import (
        hnsw_migration,
    )
    from osprey.services.ariel_search.enhancement.text_embedding import (
        migration as text_migration,
    )

    # The preamble both vector migrations share, imported by name into each.
    for module in (text_migration, image_migration, hnsw_migration):
        monkeypatch.setattr(module, "pgvector_available", AsyncMock(return_value=False))
    stub = llama_stub()
    monkeypatch.setenv("LLAMA_CPP_HOST", stub.url)
    raw = _deployment_config(scratch_database, _write_source(tmp_path))
    config = ARIELConfig.from_dict(raw)
    image_table = image_embedding_target(config.get_enhancement_module_config(EMBED) or {}).table
    assert not _image_table_exists(scratch_database, image_table)

    observed = await _main_case(raw, scratch_database, tmp_path, monkeypatch, caplog, ollama_stub)

    assert not _image_table_exists(scratch_database, image_table)
    _assert_runs_clean(observed)
    # What text_embedding records without pgvector: the pass finds no embedding
    # table, says so in one WARNING and marks the entry done without a vector.
    assert observed.status_rows[TEXT_EMBED]["status"] == "complete", observed.status_rows
    assert any(
        r.levelno == logging.WARNING and "Embedding tables do not exist" in r.getMessage()
        for r in observed.pass_records
    )
    assert not _image_table_exists(
        scratch_database, f"text_embeddings_{TEXT_MODEL.replace('-', '_')}"
    )
    assert EMBED not in observed.status_rows
    assert not any(
        isinstance(v, dict) and v.get("status") == "failed" and k in PICTURE_MODULES
        for k, v in observed.status_rows.items()
    )
    assert len(_availability_warnings(observed.pass_records, EMBED)) == 1
    assert _health(observed, EMBED)["reachable"] is False
    assert _health(observed, EMBED)["reason"] == "config"
    line = _skip_line(EMBED, "config", raw)
    assert "osprey ariel migrate" in line
    assert line in observed.status_text
    _assert_text_only_hybrid(observed)


# --- fifth check: the picture lane warns once per process ----------------------------


async def test_ten_hybrid_searches_log_one_lane_warning(
    scratch_database, ollama_stub, monkeypatch, caplog, tmp_path
):
    clock = [1000.0]
    monkeypatch.setattr(image_lane, "_now", lambda: clock[0])
    raw = _deployment_config(scratch_database, _write_source(tmp_path))

    observed = await _main_case(
        raw,
        scratch_database,
        tmp_path,
        monkeypatch,
        caplog,
        ollama_stub,
        hybrid_calls=10,
        clock=clock,
    )

    assert clock[0] == pytest.approx(1120.0)
    _assert_runs_clean(observed)
    _assert_text_only_hybrid(observed)
    lane = _availability_warnings(list(caplog.records), LANE)
    assert len(lane) == 1, lane
