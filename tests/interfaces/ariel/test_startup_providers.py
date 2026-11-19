"""The ARIEL web panel imports its configured providers before it serves.

A provider's first lookup in a process imports its adapter, seconds of CPU. The
panel's searches and its entry inspector make that lookup on the event loop,
where it would hold up the timers of other requests, so the lifespan resolves
the providers those paths will ask for before the first request, and no others.
Which search lanes ask is read from the registered modes' descriptors, so a
deployment's own mode counts as a built-in does.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

from tests.services.ariel_search.slow_provider import (
    CHAT_PROBE,
    IMAGE_EMBEDDING,
    TEXT_EMBEDDING,
    SlowProviders,
)

#: A provider whose module raises while it imports. Each attempt first adds a
#: line to the ``.tries`` file beside the module, so a test can count them.
RAISES_ON_IMPORT = """
import pathlib

with pathlib.Path(__file__).with_suffix(".tries").open("a") as tries:
    tries.write("import\\n")
raise OSError("the adapter needs a library this host lacks")
"""

#: A deployment registry adding two search modes: ``site_similar`` embeds the
#: query as the semantic mode does, and ``site_pictures`` takes the picture
#: lane as the hybrid mode does. Neither is named ``semantic`` or ``hybrid``.
SITE_SEARCH_REGISTRY = """
from osprey.registry.base import ArielSearchModuleRegistration, RegistryConfigProvider
from osprey.registry.helpers import extend_framework_registry


class SiteRegistry(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            ariel_search_modules=[
                ArielSearchModuleRegistration(
                    name="site_similar", module_path="site_similar_search", description="x"
                ),
                ArielSearchModuleRegistration(
                    name="site_pictures", module_path="site_pictures_search", description="x"
                ),
            ]
        )
"""

SITE_SIMILAR = """
import dataclasses

from osprey.services.ariel_search.search import semantic


def get_tool_descriptor():
    return dataclasses.replace(
        semantic.get_tool_descriptor(), name="site_similar_search", search_mode="site_similar"
    )
"""

SITE_PICTURES = """
import dataclasses

from osprey.services.ariel_search.search import qmd


def get_tool_descriptor():
    return dataclasses.replace(
        qmd.get_tool_descriptor(), name="site_pictures_search", search_mode="site_pictures"
    )
"""

#: A deployment registry whose initialization fails part-way: its enhancement
#: module raises, not with an ImportError, while it imports.
FAILING_REGISTRY = """
from osprey.registry.base import ArielEnhancementModuleRegistration, RegistryConfigProvider
from osprey.registry.helpers import extend_framework_registry


class SiteRegistry(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            ariel_enhancement_modules=[
                ArielEnhancementModuleRegistration(
                    name="site_tagger",
                    module_path="site_tagger_module",
                    class_name="SiteTagger",
                    description="x",
                    execution_order=90,
                )
            ]
        )
"""

FAILING_MODULE = 'raise RuntimeError("the site tagger needs a licence server")\n'


def _tries(tmp_path: Path) -> int:
    """How many times the module of :data:`RAISES_ON_IMPORT` has been imported."""
    tries = list(tmp_path.glob("*slow_broken.tries"))
    return len(tries[0].read_text().splitlines()) if tries else 0


def _open_service(config: object, **_kwargs: object) -> object:
    """A real search service on *config*, healthy, over a stand-in store."""
    from osprey.services.ariel_search.service import ARIELSearchService

    service = ARIELSearchService(config=config, pool=MagicMock(), repository=MagicMock())
    service.health_check = AsyncMock(return_value=(True, "Service healthy"))
    service.verify_search_model_table = AsyncMock()
    service.aclose = AsyncMock()
    service.repository.count_entries = AsyncMock(return_value=0)
    service.repository.get_last_ingestion = AsyncMock(return_value=None)
    return service


def _start(
    tmp_path: Path,
    ariel: dict,
    *,
    database: bool = True,
    files: dict[str, str] | None = None,
) -> object:
    """Run the real lifespan on a config carrying *ariel*; return the started app.

    The service the lifespan opens is a real one on the lifespan's own config;
    ``database=False`` makes the open refuse, as an unreachable database does.
    *files* are written into the project, which is on ``sys.path``; a
    ``site_registry.py`` among them is named as the deployment's registry.
    """
    from osprey.interfaces.ariel import create_app

    project = tmp_path / "project"
    project.mkdir()
    document: dict = {"ariel": {"database": {"uri": "postgresql://localhost/test"}, **ariel}}
    for name, source in (files or {}).items():
        (project / name).write_text(source)
    if files and "site_registry.py" in files:
        document["registry_path"] = "site_registry.py"
    config_file = project / "config.yml"
    config_file.write_text(yaml.dump(document))
    create = (
        AsyncMock(side_effect=_open_service)
        if database
        else AsyncMock(side_effect=ConnectionError("connection refused"))
    )
    try:
        with (
            patch.object(sys, "path", [str(project), *sys.path]),
            patch("osprey.services.ariel_search.create_ariel_service", create),
        ):
            app = create_app(config_file)
            with TestClient(app) as client:
                assert client.get("/health").status_code == 200
    finally:
        for name in files or {}:
            sys.modules.pop(Path(name).stem, None)
    return app


#: Every lane and module the panel can ask a provider of, each on its own provider.
FULL = {
    "search_modules": {
        "semantic": {"enabled": True, "model": "m", "provider": "slow-text"},
        "hybrid": {"enabled": True},
    },
    "enhancement_modules": {
        "image_embedding": {
            "enabled": True,
            "provider": "slow-image",
            "model": "img",
            "dimensions": 4,
        },
        "text_embedding": {
            "enabled": True,
            "provider": "slow-module",
            "models": [{"name": "m", "dimension": 3}],
        },
        "semantic_processor": {"enabled": False, "provider": "slow-disabled"},
    },
}


@pytest.fixture
def providers(tmp_path, monkeypatch) -> SlowProviders:
    """The four providers :data:`FULL` names.

    The import sleep is zero: the tests read whether a module was imported,
    not how long it took.
    """
    slow = SlowProviders(tmp_path, monkeypatch)
    slow.register("slow-text", TEXT_EMBEDDING, sleep_s=0)
    slow.register("slow-image", IMAGE_EMBEDDING, sleep_s=0)
    slow.register("slow-module", TEXT_EMBEDDING, sleep_s=0)
    slow.register("slow-disabled", CHAT_PROBE, sleep_s=0)
    return slow


def test_startup_imports_the_configured_providers_and_no_other(tmp_path, providers):
    _start(tmp_path, FULL)

    # The semantic and picture lanes' providers, and the text-embedding
    # module's, which the entry inspector builds.
    assert providers.imported("slow-text")
    assert providers.imported("slow-image")
    assert providers.imported("slow-module")
    assert not providers.imported("slow-disabled")


def test_a_deployment_search_mode_gets_its_providers_at_startup(tmp_path, providers):
    """The lanes follow the descriptors of the deployment's enabled modes, not mode names."""
    _start(
        tmp_path,
        {
            "embedding": {"provider": "slow-text"},
            "search_modules": {
                "semantic": {"enabled": False},
                "hybrid": {"enabled": False},
                "site_similar": {"enabled": True},
                "site_pictures": {"enabled": True},
            },
            "enhancement_modules": {
                "image_embedding": FULL["enhancement_modules"]["image_embedding"]
            },
        },
        files={
            "site_registry.py": SITE_SEARCH_REGISTRY,
            "site_similar_search.py": SITE_SIMILAR,
            "site_pictures_search.py": SITE_PICTURES,
        },
    )

    assert providers.imported("slow-text")
    assert providers.imported("slow-image")


@pytest.mark.parametrize(
    ("hybrid", "image_embedding"),
    [(False, True), (True, False)],
    ids=["no-mode-takes-pictures", "image-embedding-off"],
)
def test_the_picture_lane_provider_waits_for_a_mode_that_takes_pictures(
    tmp_path, providers, hybrid, image_embedding
):
    """The search lanes resolve the picture provider only when an enabled mode gets the lane.

    Read on the service alone, over the framework registry the panel's
    lifespan would have initialized: at panel startup the entry inspector's
    build of an enabled ``image_embedding`` module resolves the same provider.
    """
    from osprey.registry import get_registry
    from osprey.services.ariel_search.config import ARIELConfig

    ariel = {
        "database": {"uri": "postgresql://localhost/test"},
        "search_modules": {**FULL["search_modules"], "hybrid": {"enabled": hybrid}},
        "enhancement_modules": {
            "image_embedding": {
                **FULL["enhancement_modules"]["image_embedding"],
                "enabled": image_embedding,
            },
        },
    }
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump({"ariel": ariel}))
    get_registry(config_path=str(config_file)).initialize(silent=True)

    _open_service(ARIELConfig.from_dict(ariel)).resolve_search_providers()

    assert providers.imported("slow-text")
    assert not providers.imported("slow-image")


def test_a_panel_without_a_database_resolves_nothing(tmp_path, providers):
    """With no service the panel searches nothing and inspects nothing."""
    app = _start(tmp_path, FULL, database=False)

    assert app.state.ariel_service is None
    assert not providers.imported("slow-text")
    assert not providers.imported("slow-module")


def test_a_registry_that_fails_to_initialize_is_tried_once(tmp_path, providers):
    """The preload reads the lifespan's registry and does not initialize it again.

    Each lookup would retry the failed initialization and log its ERROR once
    more; the panel logs the failure once and resolves nothing.
    """
    failures: list[str] = []

    class _Count(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "Registry initialization failed" in record.getMessage():
                failures.append(record.getMessage())

    handler = _Count(level=logging.ERROR)
    registry_logger = logging.getLogger("registry")
    registry_logger.addHandler(handler)
    try:
        app = _start(
            tmp_path,
            FULL,
            files={"site_registry.py": FAILING_REGISTRY, "site_tagger_module.py": FAILING_MODULE},
        )
    finally:
        registry_logger.removeHandler(handler)

    assert app.state.ariel_service is not None
    assert len(failures) == 1, failures
    assert not providers.imported("slow-text")
    assert not providers.imported("slow-module")


def test_a_provider_that_fails_at_startup_is_reported_by_the_first_search(tmp_path, monkeypatch):
    slow = SlowProviders(tmp_path, monkeypatch)
    slow.register("slow-broken", RAISES_ON_IMPORT, sleep_s=0)

    app = _start(
        tmp_path,
        {
            "search_modules": {
                "semantic": {"enabled": True, "model": "m", "provider": "slow-broken"}
            }
        },
    )

    # Startup tried the provider once and the panel is up. The first search
    # tries again, nothing having been cached, and raises what it raises when
    # nothing tried before it.
    assert _tries(tmp_path) == 1
    with pytest.raises(RuntimeError, match=r"Provider 'slow-broken' could not be loaded.*OSError"):
        app.state.ariel_service._get_embedder()
    assert _tries(tmp_path) == 2
