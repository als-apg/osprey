"""Graph mode configures the control-assistant preset through nothing but its name.

The three database-backed paradigms each carry a ``channel_finder.pipelines.<mode>``
block naming the file they read. The graph paradigm reads a store, and that store
is already declared — ``services.graphdb``, deployed by this preset or pointed at a
facility-hosted one through ``uri``. So ``pipeline_mode: graph`` plus that block is
the *entire* configuration: there is no ``pipelines.graph`` block, and graph mode
introduces no config key that the other modes do not already render.

That is a claim about the shape of the rendered file, so it is pinned as one here:
the graph render's key set is the in-context render's key set with the pipeline
block removed — nothing added, nothing else dropped. The corollary is pinned too:
the manager derives an ``enable_graph`` flag for every registered paradigm, and no
template reads this one. Should a template start reading it, the test that walks
the template tree fails, and the render matrix in ``scripts/config_key_manifest.yml``
(which exercises the ``enable_*`` branches) would need a fourth cell.

The mode-enumerating comments are checked against the registry rather than against
a literal list, so a fifth paradigm fails these tests instead of quietly leaving
the preset's prose one mode short.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
from osprey.cli.templates.manager import TemplateManager, _enable_flags

CONFIG_TEMPLATE = "apps/control_assistant/config.yml.j2"
README_TEMPLATE = "apps/control_assistant/README.md.j2"

#: Context sufficient to render the preset. Keys absent from it render empty,
#: which is why only the paradigm-selecting keys are spelled out per mode.
_BASE_CTX: dict[str, Any] = {
    "project_name": "demo",
    "app_display_name": "Demo Control Assistant",
    "facility_name": "Demo",
    "framework_version": "0.0.0",
    "default_provider": "anthropic",
    "default_model": "haiku",
    "current_python_env": "/usr/bin/python3",
    "selected_web_panels": [],
    "builtin_panels": [],
    "system": {"timezone": "UTC"},
    "osprey_labels": {"project_name": "demo", "project_root": "/tmp/demo"},
}


def _ctx(mode: str) -> dict[str, Any]:
    """Render context for one paradigm, with the flags derived as the manager derives them."""
    return {
        **_BASE_CTX,
        "channel_finder_mode": mode,
        "default_pipeline": mode,
        **_enable_flags(mode),
    }


def _render(template_path: str, mode: str) -> str:
    return TemplateManager().jinja_env.get_template(template_path).render(**_ctx(mode))


def _config(mode: str) -> dict[str, Any]:
    return yaml.safe_load(_render(CONFIG_TEMPLATE, mode))


def _dotted_keys(node: Any, prefix: str = "") -> set[str]:
    """Every dotted key path in a parsed config, mappings only."""
    keys: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}{key}"
            keys.add(path)
            keys |= _dotted_keys(value, f"{path}.")
    return keys


def test_graph_mode_selects_the_paradigm_by_name():
    channel_finder = _config("graph")["channel_finder"]
    assert channel_finder["pipeline_mode"] == "graph"


def test_graph_mode_renders_no_pipeline_block():
    pipelines = _config("graph")["channel_finder"].get("pipelines") or {}
    assert "graph" not in pipelines
    assert pipelines == {}, f"graph mode rendered a pipeline block: {sorted(pipelines)}"


def test_graph_mode_configuration_is_the_graph_store():
    """The store the paradigm reads is declared, and deployed by this preset."""
    config = _config("graph")
    assert "graphdb" in config["services"]
    assert "graphdb" in config["deployed_services"]


def test_graph_mode_adds_no_config_key():
    """The graph render is the in-context render minus the pipeline block."""
    graph_keys = _dotted_keys(_config("graph"))
    in_context_keys = _dotted_keys(_config("in_context"))
    expected = {key for key in in_context_keys if not key.startswith("channel_finder.pipelines.")}
    assert graph_keys == expected


def test_enable_graph_is_derived_but_unread():
    """The flag exists for every registered paradigm; no template consumes this one."""
    assert _enable_flags("graph")["enable_graph"] is True

    template_root = Path(TemplateManager().template_root)
    readers = [
        str(path.relative_to(template_root))
        for path in template_root.rglob("*")
        if path.is_file() and _mentions_enable_graph(path)
    ]
    assert readers == [], f"templates reading enable_graph: {readers}"


def _mentions_enable_graph(path: Path) -> bool:
    try:
        return "enable_graph" in path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False


@pytest.mark.parametrize("template_path", [CONFIG_TEMPLATE, README_TEMPLATE])
def test_pipeline_mode_comment_names_every_paradigm(template_path: str):
    """The comment beside `pipeline_mode:` enumerates the registry, not a subset."""
    rendered = _render(template_path, "in_context")
    comment = next(
        line for line in rendered.splitlines() if line.strip().startswith("pipeline_mode:")
    )
    missing = [mode for mode in VALID_CHANNEL_FINDER_MODES if f'"{mode}"' not in comment]
    assert missing == [], f"{template_path} names no paradigm {missing} in {comment.strip()!r}"
