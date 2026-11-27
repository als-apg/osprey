"""The vocabulary a build states is the file ARIEL reads, from every render.

The demo facility ships a vocabulary, so the build turns expansion on and states
the path. ARIEL reads the file at config parse and its tool server refuses to
start without it. Each render carries its own copy and its own ``config.yml``:
the deployment's render, where the project root is the directory above, and a
login's render, which is its own project root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from osprey.deployment.web_terminals.render_naming import RENDERS_KEY
from osprey.facility.settings import VOCABULARY_SOURCE
from osprey.mcp_server.ariel.server_context import ARIELContext
from osprey.port_layout import resolve_port_base
from osprey.services.ariel_search.config import ARIELConfig
from tests._builds import BuiltProject


def _config(render: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((render / "config.yml").read_text(encoding="utf-8"))
    return loaded


def _vocabulary_renders(built: BuiltProject) -> list[Path]:
    """The deployment's render and each login's, where the config turns the vocabulary on."""
    renders: Any = _config(built.build_dir)
    for part in RENDERS_KEY.split("."):
        renders = renders[part]
    candidates = [
        built.build_dir,
        *(built.repo / item["project_path"] for item in renders.values()),
    ]
    return [
        render
        for render in candidates
        if ((_config(render).get("ariel") or {}).get("vocabulary") or {}).get("enabled")
    ]


def test_the_stated_vocabulary_loads_from_every_render(built_demo: BuiltProject) -> None:
    renders = _vocabulary_renders(built_demo)
    assert built_demo.build_dir in renders
    assert len(renders) > 1, "no login's render turns the vocabulary on"

    for render in renders:
        config = _config(render)
        assert config["ariel"]["vocabulary"]["path"] == VOCABULARY_SOURCE

        parsed = ARIELConfig.from_dict(
            config["ariel"],
            (config.get("services") or {}).get("postgresql") or {},
            config_dir=render,
            base=resolve_port_base(config),
        )

        assert parsed.vocabulary_errors == [], render
        assert parsed.loaded_vocabulary is not None, render
        ARIELContext._validate_or_refuse(parsed)
