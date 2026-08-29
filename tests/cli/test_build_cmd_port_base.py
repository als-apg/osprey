"""``osprey build`` resolves the port base once, from the profile it is building.

A profile's ``config:`` is a flat bag of dotted keys, so ``deployment.port_base``
can arrive spelled three ways — dotted (what ``osprey set`` writes), nested, or
both at once. The build folds them into one deployment subtree, resolves a base
from it, and hands that scalar to the template manager, which expands it into
the ``osprey_ports`` table the templates read.

The rule these tests exist for: **a port is always derived from the base the
deployment actually resolved**. A base the build cannot honour stops the build;
it never falls back to the layout default, because a deployment published at
10000 when its author asked for something else is a silent collision on a host
that is already running one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli.build_cmd import _repo_render_context
from osprey.cli.build_profile_model import BuildProfile
from osprey.cli.templates.manager import TemplateManager
from osprey.port_layout import DEFAULT_PORT_BASE, layout_ports

# A base far from the default, so a port that came from the default instead of
# from the profile cannot pass by coincidence.
MOVED_BASE = 20000


def _build_context(tmp_path: Path, config: dict | None = None) -> dict:
    """The render context ``osprey build`` composes for a profile.

    Args:
        tmp_path: A scratch repo root; nothing is written to it.
        config: The profile's ``config:`` block, as parsed from YAML.

    Returns:
        The context handed to :class:`TemplateManager`.
    """
    profile = BuildProfile(name="port-base-test", config=dict(config or {}))
    return _repo_render_context(
        profile,
        repo_root=tmp_path,
        build_dir=tmp_path / "build",
        runtime_root=None,
        project_deps=[],
        skip_deps=True,
    )


def _ports(context: dict) -> dict[str, int]:
    """The port table the templates would read, given a build context.

    Args:
        context: A context from :func:`_build_context`.

    Returns:
        ``osprey_ports`` — slot name to host port.
    """
    ctx = TemplateManager()._project_context(
        "build", Path("/tmp/does-not-need-to-exist"), "control_assistant", context, None
    )
    return ctx["osprey_ports"]


def test_a_profile_naming_no_base_builds_at_the_default(tmp_path: Path) -> None:
    """The common case, and the one that must never change: a profile written
    before the key existed keeps rendering the block it always rendered."""
    context = _build_context(tmp_path)

    assert context["port_base"] == DEFAULT_PORT_BASE
    assert _ports(context) == layout_ports(DEFAULT_PORT_BASE)


def test_the_dotted_spelling_moves_the_block(tmp_path: Path) -> None:
    """``deployment.port_base: 20000`` — how ``osprey set`` writes it.

    The value arrives already an int (``osprey set`` routes through
    ``yaml.safe_load``), and the whole table moves with it.
    """
    context = _build_context(tmp_path, {"deployment.port_base": MOVED_BASE})

    assert context["port_base"] == MOVED_BASE
    assert _ports(context)["nginx"] == MOVED_BASE
    assert _ports(context) == layout_ports(MOVED_BASE)


def test_the_nested_spelling_resolves_the_same(tmp_path: Path) -> None:
    """A hand-written profile may nest the block instead of dotting the key.

    Both spellings address one subtree, so both must produce one answer —
    otherwise the base a deployment publishes at would depend on how its
    author happened to type it.
    """
    dotted = _build_context(tmp_path, {"deployment.port_base": MOVED_BASE})
    nested = _build_context(tmp_path, {"deployment": {"port_base": MOVED_BASE}})

    assert nested["port_base"] == dotted["port_base"] == MOVED_BASE


def test_a_deeper_key_wins_over_the_block_it_sits_under(tmp_path: Path) -> None:
    """A profile that spells both keeps the deeper one — the fold's own rule.

    Reading either key alone would answer wrong here, which is why the build
    goes through the subtree fold rather than looking the dotted key up.
    """
    context = _build_context(
        tmp_path,
        {"deployment": {"port_base": 30000}, "deployment.port_base": MOVED_BASE},
    )

    assert context["port_base"] == MOVED_BASE


def test_an_unusable_base_stops_the_build(tmp_path: Path) -> None:
    """A base below 1024 is refused, and nothing is rendered at the default.

    The refusal has to land here, before the render: a build that swallowed it
    would publish the deployment at 10000 while its author's profile says 1000,
    and the first sign would be a service answering on a port nobody named.
    """
    with pytest.raises(ValueError, match="deployment.port_base") as refusal:
        _build_context(tmp_path, {"deployment.port_base": 1000})

    assert "1000" in str(refusal.value)
    # Nothing was rendered: the refusal lands while the context is composed,
    # which is before the build writes anything at all.
    assert not (tmp_path / "build").exists()
