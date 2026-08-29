"""The template context's port table: one base in, every framework port out.

``TemplateManager`` is the single place a render's ports are derived. It takes
whatever ``port_base`` its caller resolved — the layout default when there is
no config to resolve one from, the profile's own base when ``osprey build``
hands one down — and expands it into ``osprey_ports``, the slot-to-port
mapping every template reads instead of spelling a literal.

What these tests pin is that the expansion happens once, at the caller's base,
and that an out-of-range base handed in here is refused rather than quietly
becoming the default.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from osprey.cli.templates.manager import TemplateManager
from osprey.port_layout import DEFAULT_PORT_BASE, LAYOUT, layout_ports

# A base far from the default, so a port that came from the default instead of
# from the caller cannot pass by coincidence.
MOVED_BASE = 20000


def _context(**context: object) -> dict:
    """The template context a project render would be made with.

    Args:
        **context: Caller context, as ``osprey build`` passes it.

    Returns:
        The merged context ``TemplateManager`` hands the templates.
    """
    return TemplateManager()._project_context(
        "build", Path("/tmp/does-not-need-to-exist"), "control_assistant", dict(context), None
    )


def _dotted(config: dict, key: str) -> tuple[bool, object]:
    """Read a dotted key out of a rendered config.

    Args:
        config: The loaded ``config.yml``.
        key: Dotted path, as :data:`LAYOUT` spells an override key.

    Returns:
        ``(found, value)`` — ``(False, None)`` when the render wrote no such
        key, which is the ordinary case for a service this project does not
        deploy.
    """
    node: object = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return (False, None)
        node = node[part]
    return (True, node)


def test_bare_render_writes_integer_ports(tmp_path: Path) -> None:
    """Every framework port a bare render writes is a number.

    The programmatic entry point — no build profile, no resolved base — must
    still render a config a deployment can read: a port that came out as a
    string, an empty value or an unrendered Jinja expression would only fail
    later, at the point something tried to bind it.
    """
    render = TemplateManager().create_project(
        project_name="build",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    config = yaml.safe_load((render / "config.yml").read_text(encoding="utf-8"))

    written = {}
    for slot in LAYOUT:
        if slot.config_key is None:
            continue
        found, value = _dotted(config, slot.config_key)
        if found:
            written[slot.config_key] = value

    assert written, "the render wrote no framework port at all — the check is vacuous"
    assert all(isinstance(value, int) for value in written.values()), written


def test_no_caller_base_means_the_layout_default() -> None:
    """With nothing to resolve a base from, the whole table is at the default."""
    ctx = _context()

    assert ctx["port_base"] == DEFAULT_PORT_BASE
    assert ctx["osprey_ports"] == layout_ports(DEFAULT_PORT_BASE)


def test_the_callers_base_moves_every_slot() -> None:
    """A base handed down by the caller moves the whole block, not part of it.

    This is the rule the feature rests on: the ports come from the base the
    deployment actually resolved. A slot that kept the default here would be
    the one port a second deployment on the same host collides on.
    """
    ctx = _context(port_base=MOVED_BASE)

    assert ctx["osprey_ports"] == layout_ports(MOVED_BASE)
    assert ctx["osprey_ports"]["nginx"] == MOVED_BASE
    assert min(ctx["osprey_ports"].values()) >= MOVED_BASE


def test_an_out_of_range_base_is_refused_here_too() -> None:
    """A base the block cannot start at fails the render, not the deployment.

    The resolver refuses it at ``osprey build``; this pins that a caller
    reaching the manager by another road gets the same refusal rather than a
    render silently placed at the default base.
    """
    with pytest.raises(ValueError, match="deployment.port_base"):
        _context(port_base=1000)
