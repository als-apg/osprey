"""The ARIEL how-to snippets run against, or equal, the code they describe.

Pages read, each block found by its content (never by line number) through the
snippet-coverage scanner:

* ``how-to/ariel/data-ingestion.rst`` -- the three ``FacilityAdapter``
  snippets (sidecar names, entry fields, attachments) are executed, completed
  with the two abstract members and instantiated; the backoff formula is
  evaluated against the scheduler's own interval; the ``osprey build``
  advisory is the one ``ariel_ingestion_advisories`` writes; the ``UPDATE
  text_embeddings_<model>`` statement names the table the text-embedding
  migration creates and a column on it.
* ``how-to/ariel/search-modes.rst`` -- the ``extend_framework_registry``
  snippet is executed and merged through the registry loader, which keeps the
  framework's modules and adds the snippet's.
* ``how-to/ariel/web-interface.rst`` -- the ``create_app`` snippet is executed
  with a stand-in ``uvicorn``; the app has the ``/api/search`` route and the
  snippet binds the ARIEL panel's default address (``resolve_web_server_address``).
* ``how-to/ariel/index.rst`` and ``how-to/ariel/picture-search.rst`` -- the
  ``ollama pull`` tag is the caption model the preset ships.

A snippet runs in a fresh namespace whose ``__import__`` admits ``osprey`` and
the stand-ins the test hands it and refuses anything else, and whose builtins
lack ``open``, ``exec``, ``eval`` and ``compile``. That is an import allowlist
that catches undeclared dependencies, not a security boundary (an admitted
``osprey`` module exposes its own imports); the snippets are repo-reviewed
text.

A page that says something other than the code is recorded in ``_STALE`` with
the page, the code location and the PR that fixes it, so the gap is visible
and the suite stays green until the page is fixed.
"""

from __future__ import annotations

import ast
import builtins
import re
import types
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.port_layout import DEFAULT_PORT_BASE, layout_ports
from osprey.registry import loader
from osprey.registry.base import ExtendedRegistryConfig
from osprey.registry.web import resolve_web_server_address
from osprey.services.ariel_search.config import (
    ARIELConfig,
    DatabaseConfig,
    IngestionConfig,
    WatchConfig,
)
from osprey.services.ariel_search.database.migrations import model_to_table_name
from osprey.services.ariel_search.enhancement.text_embedding.migration import (
    HNSW_MIN_PGVECTOR,
    TextEmbeddingMigration,
)
from osprey.services.ariel_search.entry_fields import (
    EntryFieldDeclarationError,
    check_declarations,
    fetch_entry_field_options,
)
from osprey.services.ariel_search.ingestion.base import FacilityAdapter
from osprey.services.ariel_search.ingestion.metadata_attachment import _sidecar_names
from osprey.services.ariel_search.ingestion.scheduler import IngestionScheduler, StopReason
from osprey.services.ariel_search.models import FacilityEntryCreateRequest
from osprey.services.ariel_search.search.base import ParameterDescriptor
from tests._by_path import load_module

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _REPO_ROOT / "docs" / "source"
_SCRIPT = _REPO_ROOT / "scripts" / "docs" / "snippet_coverage.py"
_PRESET = _REPO_ROOT / "src" / "osprey" / "profiles" / "presets" / "demo.yml"

_DATA_INGESTION = "how-to/ariel/data-ingestion.rst"
_SEARCH_MODES = "how-to/ariel/search-modes.rst"
_WEB_INTERFACE = "how-to/ariel/web-interface.rst"
_INDEX = "how-to/ariel/index.rst"
_PICTURE_SEARCH = "how-to/ariel/picture-search.rst"

#: Blocks whose text differs from the code, as ``(page, what, page says)`` ->
#: ``(code location and its value, PR that fixes the page)``. A test that
#: compares such a block accepts the recorded page value. The stale-entry guard
#: fails once the page no longer says it when no fixing PR is recorded; it skips
#: while ``fixed_by`` is recorded and the page shows the fixed text. Drop the
#: entry after that PR merges.
_STALE: dict[tuple[str, str, str], tuple[str, int | None]] = {}


# -- reading blocks ----------------------------------------------------------------


@cache
def _scanner() -> types.ModuleType:
    """The coverage script, loaded by path; registered because it defines dataclasses."""
    return load_module("snippet_coverage", _SCRIPT, register=True)


@cache
def _blocks(page: str) -> tuple[Any, ...]:
    text = (_DOCS / page).read_text(encoding="utf-8")
    return tuple(_scanner().scan_page(text, page, _DOCS))


def _block(page: str, language: str, needle: str) -> str:
    """The text of the one *language* block on *page* that contains *needle*."""
    found = [b.text for b in _blocks(page) if b.language == language and needle in b.text]
    assert len(found) == 1, (
        f"{page}: expected one {language} block with {needle!r}, found {len(found)}"
    )
    return str(found[0])


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _exec_snippet(
    source: str,
    *,
    page: str,
    names: Mapping[str, Any] | None = None,
    stubs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run *source* in a fresh namespace holding *names*; return the namespace.

    Imports of ``osprey`` modules go to the real package; a module named in
    *stubs* is the stand-in given; any other import is refused. This catches
    a dependency the snippet does not declare; it is not a security boundary.
    """
    stand_ins = dict(stubs or {})
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in stand_ins:
            return stand_ins[name]
        if level == 0 and (name == "osprey" or name.startswith("osprey.")):
            return real_import(name, globals, locals, fromlist, level)
        raise ImportError(f"{page}: the snippet imports {name!r}, which this test does not admit")

    sandbox_builtins = {
        name: value
        for name, value in vars(builtins).items()
        if name not in ("open", "exec", "eval", "compile")
    }
    sandbox_builtins["__import__"] = guarded_import
    namespace: dict[str, Any] = {"__builtins__": sandbox_builtins, "__name__": "docs_snippet"}
    namespace.update(names or {})
    exec(compile(source, f"<{page}>", "exec"), namespace)
    return namespace


def _config(**ingestion: Any) -> ARIELConfig:
    """The ``ARIELConfig`` stand-in the ARIEL service tests build adapters from."""
    return ARIELConfig(
        database=DatabaseConfig(uri="postgresql://localhost/test"),
        ingestion=IngestionConfig(**ingestion) if ingestion else None,
    )


def _complete(snippet_class: type) -> type:
    """*snippet_class* with the two abstract members a real adapter supplies."""

    class Completed(snippet_class):
        @property
        def source_system_name(self) -> str:
            return "Docs Logbook"

        async def fetch_entries(self, *_args, **_kwargs):
            return
            yield

    return Completed


def _adapter_class(needle: str) -> type:
    source = _block(_DATA_INGESTION, "python", needle)
    namespace = _exec_snippet(
        source, page=_DATA_INGESTION, names={"FacilityAdapter": FacilityAdapter}
    )
    snippet_class = namespace["MyLogbookAdapter"]
    assert issubclass(snippet_class, FacilityAdapter)
    return _complete(snippet_class)


def test_the_sandbox_refuses_an_import_it_was_not_given() -> None:
    """Negative control: a snippet cannot reach a library the test did not admit."""
    with pytest.raises(ImportError, match="'socket'"):
        _exec_snippet("import socket", page=_WEB_INTERFACE)


# -- data-ingestion.rst: the adapter snippets ---------------------------------------


def test_sidecar_snippet_declares_the_name_ingestion_matches() -> None:
    adapter = _adapter_class("metadata_sidecar_names")(_config())
    assert adapter.metadata_sidecar_names == ("entry-meta.json",)
    assert _sidecar_names(adapter) == {"entry-meta.json"}
    # The page's "The default is ``("metadata.json",)``".
    assert FacilityAdapter.metadata_sidecar_names == ("metadata.json",)
    assert '``("metadata.json",)``' in (_DOCS / _DATA_INGESTION).read_text(encoding="utf-8")


async def test_entry_fields_snippet_declares_fields_the_service_accepts() -> None:
    adapter_class = _adapter_class("get_entry_field_descriptors")

    class WithLookup(adapter_class):  # type: ignore[misc, valid-type]
        async def _scans_on(self, day):
            return [types.SimpleNamespace(id=f"{day}-1", title="Morning scan")]

    adapter = WithLookup(_config())
    descriptors = adapter.get_entry_field_descriptors()
    assert all(isinstance(d, ParameterDescriptor) for d in descriptors)
    assert [d.name for d in descriptors] == ["book", "day", "scan"]
    for descriptor in descriptors:
        assert descriptor.to_dict()["name"] == descriptor.name
    by_name = {d.name: d for d in descriptors}
    assert by_name["scan"].depends_on == ("day",)
    assert by_name["scan"].to_dict()["depends_on"] == ["day"]
    # The check every form and write path reads the declarations through.
    assert check_declarations(descriptors, adapter=adapter.source_system_name) == descriptors
    # Through the service's own call, with its timeout wrapper.
    assert await fetch_entry_field_options(adapter, "book", {}) == []
    assert await fetch_entry_field_options(adapter, "scan", {"day": "2026-10-04"}) == [
        {"value": "2026-10-04-1", "label": "Morning scan"}
    ]


def test_an_unknown_field_type_is_refused() -> None:
    """Negative control: the declaration check the snippet passes does refuse a bad type."""
    adapter = _adapter_class("get_entry_field_descriptors")(_config())
    descriptors = adapter.get_entry_field_descriptors()
    broken = [
        *descriptors[:2],
        ParameterDescriptor(**{**vars(descriptors[2]), "param_type": "dropdown"}),
    ]
    with pytest.raises(EntryFieldDeclarationError, match="unknown type 'dropdown'"):
        check_declarations(broken, adapter=adapter.source_system_name)


async def test_attachments_snippet_publishes_every_file() -> None:
    adapter_class = _adapter_class("supports_attachments")
    posted: list[tuple[str, str]] = []

    class WithLogbook(adapter_class):  # type: ignore[misc, valid-type]
        async def _post_entry(self, request):
            posted.append(("entry", request.subject))
            return "E-17"

        async def _post_file(self, entry_id, path):
            posted.append((entry_id, path))

    adapter = WithLogbook(_config())
    assert adapter.supports_attachments is True
    assert _complete(FacilityAdapter)(_config()).supports_attachments is False
    request = FacilityEntryCreateRequest(
        subject="Beam dump", details="RF trip", attachment_paths=["/x/a.png", "/x/b.pdf"]
    )
    assert await adapter.create_entry(request) == "E-17"
    assert posted == [("entry", "Beam dump"), ("E-17", "/x/a.png"), ("E-17", "/x/b.pdf")]


# -- data-ingestion.rst: the backoff formula ------------------------------------------


def _formula() -> tuple[str, ast.Expression]:
    text = _block(_DATA_INGESTION, "literal", "backoff_multiplier ^")
    left, right = (part.strip() for part in text.split("=", 1))
    expression = ast.parse(right.replace("×", "*").replace("^", "**"), mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.Mult, ast.Pow, ast.Name, ast.Load)
    assert all(isinstance(node, allowed) for node in ast.walk(expression)), text
    return left, expression


def _evaluate(expression: ast.Expression, **values: float) -> float:
    return float(eval(compile(expression, "<formula>", "eval"), {"__builtins__": {}}, values))


async def _scheduler_intervals(base: int, multiplier: float, failures: int) -> list[float]:
    """The interval the scheduler computes after each of *failures* failed polls."""
    config = _config(
        adapter="generic_json",
        source_url="https://api.example.com/logbook",
        poll_interval_seconds=base,
        watch=WatchConfig(
            max_consecutive_failures=failures + 1,
            backoff_multiplier=multiplier,
            max_interval_seconds=3600,
        ),
    )
    scheduler = IngestionScheduler(config=config, repository=None)  # type: ignore[arg-type]
    seen: list[float] = []
    real_interval = scheduler._get_current_interval

    async def failing_poll(*_args, **_kwargs):
        raise RuntimeError("logbook unreachable")

    def recorded_interval() -> float:
        seen.append(real_interval())
        return 0.0  # poll again at once; the test reads the value, not the clock

    scheduler.poll_once = failing_poll  # type: ignore[method-assign]
    scheduler._get_current_interval = recorded_interval  # type: ignore[method-assign]
    assert await scheduler.run_forever() == StopReason.FAILURE_CAP
    return seen


async def test_backoff_formula_is_the_schedulers_interval() -> None:
    left, expression = _formula()
    assert left == "interval"
    seen = await _scheduler_intervals(60, 2.0, 3)
    assert seen == [
        _evaluate(
            expression,
            poll_interval_seconds=60,
            backoff_multiplier=2.0,
            consecutive_failures=n,
        )
        for n in (1, 2, 3)
    ]
    assert seen[-1] == 480


async def test_backoff_formula_with_another_multiplier_differs() -> None:
    """Negative control: the comparison sees a wrong multiplier."""
    _, expression = _formula()
    seen = await _scheduler_intervals(60, 2.0, 3)
    wrong = _evaluate(
        expression, poll_interval_seconds=60, backoff_multiplier=3.0, consecutive_failures=3
    )
    assert seen[-1] != wrong


# -- data-ingestion.rst: the build advisory -------------------------------------------


def _advisories(source_url: str, services: list[str]) -> list[str]:
    from osprey.cli.build_profile_reach import ariel_ingestion_advisories

    return ariel_ingestion_advisories(
        {"deployed_services": services, "ariel": {"ingestion": {"source_url": source_url}}}
    )


def test_build_advisory_is_the_line_the_build_prints() -> None:
    page_line = _flat(_block(_DATA_INGESTION, "literal", "source_url is"))
    url = re.search(r"source_url is (\S+), but", page_line)
    assert url, page_line
    advisories = _advisories(url.group(1), ["postgresql"])
    # build_cmd prints each advisory as ``⚠ {advisory}``.
    assert [f"⚠ {_flat(a)}" for a in advisories] == [page_line]


def test_build_advisory_is_silent_when_it_should_be() -> None:
    """Negative control: a mirroring service, or a local file, gets no line."""
    assert _advisories("https://api.example.com/logbook", ["ariel_sync"]) == []
    assert _advisories("data/logbook.json", ["postgresql"]) == []


# -- data-ingestion.rst: the input_key reset --------------------------------------------


class _RecordingConnection:
    """Records each statement; answers the pgvector probe with a version HNSW accepts."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    async def execute(self, sql: str, *_args: Any) -> Any:
        self.statements.append(sql)

        async def fetchone():
            # (installed_version, default_version) from pg_available_extensions.
            return (None, ".".join(map(str, HNSW_MIN_PGVECTOR)))

        return types.SimpleNamespace(fetchone=fetchone)


async def test_input_key_reset_names_the_migrations_table_and_column() -> None:
    sql = _block(_DATA_INGESTION, "sql", "input_key")
    match = re.fullmatch(r"UPDATE (text_embeddings_)<model> SET (\w+) = NULL;", sql.strip())
    assert match, sql
    prefix, column = match.groups()
    table = model_to_table_name("nomic-embed-text")
    assert table == "text_embeddings_nomic_embed_text"
    assert table.startswith(prefix)
    # The page's own example of a table name is the builder's.
    assert f"``{table}``" in (_DOCS / _DATA_INGESTION).read_text(encoding="utf-8")

    connection = _RecordingConnection()
    await TextEmbeddingMigration([("nomic-embed-text", 768)]).up(connection)  # type: ignore[arg-type]
    create = next(s for s in connection.statements if "CREATE TABLE" in s)
    assert f"CREATE TABLE IF NOT EXISTS {table} (" in create
    # A nullable column, so ``SET input_key = NULL`` is a legal statement.
    assert re.search(rf"^\s*{column}\s+TEXT NULL,", create, re.MULTILINE), create


def test_a_wrong_table_name_is_not_the_builders() -> None:
    """Negative control: the model name is not used verbatim."""
    assert model_to_table_name("nomic-embed-text") != "text_embeddings_nomic-embed-text"


# -- search-modes.rst: registering a search module ---------------------------------------


def test_registration_snippet_extends_the_framework_registry(monkeypatch) -> None:
    source = _block(_SEARCH_MODES, "python", "extend_framework_registry")
    namespace = _exec_snippet(source, page=_SEARCH_MODES)
    app_config = namespace["app_config"]
    assert isinstance(app_config, ExtendedRegistryConfig)

    monkeypatch.setattr(loader, "load_registry_from_path", lambda _path: app_config)
    merged, _excluded = loader.build_merged_configuration("my_app/registry.py")
    framework = loader.load_registry_from_module("osprey.registry.builtins")
    names = [r.name for r in merged.ariel_search_modules]
    assert names == [r.name for r in framework.ariel_search_modules] + ["my_search"]
    added = merged.ariel_search_modules[-1]
    assert added.module_path == "my_app.search.my_module"


# -- web-interface.rst: the programmatic app --------------------------------------------


def test_create_app_snippet_builds_the_search_app(monkeypatch) -> None:
    monkeypatch.delenv("OSPREY_ARIEL_PORT", raising=False)
    runs: list[tuple[Any, dict[str, Any]]] = []
    uvicorn = types.SimpleNamespace(run=lambda app, **kwargs: runs.append((app, kwargs)))
    source = _block(_WEB_INTERFACE, "python", "create_app")
    namespace = _exec_snippet(source, page=_WEB_INTERFACE, stubs={"uvicorn": uvicorn})

    from fastapi import FastAPI

    app = namespace["app"]
    assert isinstance(app, FastAPI)
    assert "/api/search" in app.openapi()["paths"]
    assert len(runs) == 1 and runs[0][0] is app
    kwargs = runs[0][1]

    host, port = resolve_web_server_address("ariel", {})
    assert kwargs["port"] == port == layout_ports(DEFAULT_PORT_BASE)["ariel"]
    if kwargs["host"] != host:
        assert (_WEB_INTERFACE, "uvicorn.run host", kwargs["host"]) in _STALE, (
            f"{_WEB_INTERFACE} binds {kwargs['host']}; the ARIEL panel's default address "
            f"binds {host}"
        )


def _page_says(page: str, what: str) -> str | None:
    """The value a ``_STALE`` entry's block holds today."""
    if (page, what) == (_WEB_INTERFACE, "uvicorn.run host"):
        source = _block(_WEB_INTERFACE, "python", "create_app")
        match = re.search(r'uvicorn\.run\(app, host="([^"]+)"', source)
        return match.group(1) if match else None
    raise AssertionError(f"no reader for the _STALE entry {page} / {what}")


@pytest.mark.parametrize(("key", "where"), list(_STALE.items()), ids=lambda v: str(v))
def test_stale_entries_are_still_stale(
    key: tuple[str, str, str], where: tuple[str, int | None]
) -> None:
    page, what, stale_value = key
    _code, fixed_by = where
    if _page_says(page, what) == stale_value:
        return
    if fixed_by is not None:
        pytest.skip(f"{page} fixed by #{fixed_by}; drop the _STALE entry once it has merged")
    pytest.fail(f"{page} no longer says {stale_value!r} for {what}: drop the _STALE entry")


# -- index.rst and picture-search.rst: the Ollama models --------------------------------


def _preset_ollama_models() -> set[str]:
    """The models the preset has Ollama serve: text embeddings and picture captions."""
    config = yaml.safe_load(_PRESET.read_text(encoding="utf-8"))["config"]
    prefix = "ariel.enhancement_modules."
    assert config[f"{prefix}text_embedding.provider"] == "ollama"
    assert config[f"{prefix}image_caption.provider"] == "ollama"
    return {
        *(model["name"] for model in config[f"{prefix}text_embedding.models"]),
        config[f"{prefix}image_caption.model.model_id"],
    }


def _pulled(page: str) -> list[str]:
    """The model of every ``ollama pull`` line in the page's bash blocks."""
    lines = [
        line.split("#", 1)[0].strip()
        for block in _blocks(page)
        if block.language == "bash"
        for line in block.text.splitlines()
        if line.strip().startswith("ollama pull")
    ]
    models = [re.fullmatch(r"ollama pull (\S+)", line) for line in lines]
    assert all(models), lines
    return [m.group(1) for m in models if m]


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        (_INDEX, ["nomic-embed-text", "nomic-embed-text", "qwen3-vl:4b"]),
        (_PICTURE_SEARCH, ["qwen3-vl:4b"]),
    ],
)
def test_ollama_pull_fetches_the_presets_models(page: str, expected: list[str]) -> None:
    pulled = _pulled(page)
    assert pulled == expected
    assert set(pulled) <= _preset_ollama_models()


def test_a_model_the_preset_does_not_name_is_caught() -> None:
    """Negative control: the preset's set is not everything."""
    assert "qwen3-vl:8b" not in _preset_ollama_models()
