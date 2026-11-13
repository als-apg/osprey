"""Docs media show the operator chat, never the Claude Code terminal.

The hub's Agent tile can still be switched to the terminal, but no docs video
or screenshot may be recorded on it. These tests fail when a registry entry
names the terminal, or when the registries stop refusing it.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest
from docs.screenshots import capture, contact_sheet, video_layout
from docs.screenshots.recipes import REGISTRY, DocShot, validate_registry
from docs.screenshots.video_layout import Layout
from docs.screenshots.videos import VIDEOS, validate_videos


def test_every_registered_video_starts_in_the_chat() -> None:
    assert VIDEOS
    assert {video.name: video.layout.surface for video in VIDEOS} == {
        video.name: "chat" for video in VIDEOS
    }


def test_the_terminal_is_not_a_surface_a_video_may_name() -> None:
    assert video_layout.SURFACES == ("chat",)
    on_terminal = dataclasses.replace(
        VIDEOS[0], layout=Layout(surface="terminal", active=None, tiles=("artifacts",))
    )
    with pytest.raises(ValueError, match="layout surface 'terminal'"):
        validate_videos([on_terminal])


def test_a_recipe_cannot_name_the_agent_surface() -> None:
    """A recipe has no field to pick the terminal with."""
    assert "hub_agent" not in {field.name for field in dataclasses.fields(DocShot)}
    with pytest.raises(TypeError):
        DocShot(name="probe", environment="hermetic_hub", kind="static", hub_agent="terminal")


def test_a_hermetic_hub_recipe_boots_in_the_chat(tmp_path, monkeypatch) -> None:
    agents: list[str] = []

    class _Hub:
        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

    monkeypatch.setattr(contact_sheet, "hermetic_hub", _Hub)
    monkeypatch.setattr(
        contact_sheet,
        "capture_hub_view",
        lambda _browser, _hub, _theme, agent, _dest, **_kw: agents.append(agent),
    )
    monkeypatch.setattr(capture, "output_dir", lambda: tmp_path)
    recipes = [shot for shot in REGISTRY if shot.environment == "hermetic_hub" and not shot.hub]
    assert recipes
    for shot in recipes:
        capture._capture_hermetic_hub(object(), shot)
    assert set(agents) == {"chat"}


def test_a_hub_page_recipe_is_never_the_hub_itself() -> None:
    """A page recipe names a page beside the hub, so the Agent tile is not in it."""
    pages = [shot for shot in REGISTRY if shot.hub]
    assert pages
    assert all(shot.path != "/" for shot in pages)
    at_the_hub = dataclasses.replace(pages[0], path="/")
    with pytest.raises(ValueError, match="a hub recipe names a page other than '/'"):
        validate_registry([at_the_hub])


def test_only_the_popover_capture_opens_the_terminal_and_only_behind_a_crop() -> None:
    """The one capture that starts its session from the terminal crops it out.

    The control-target popover needs a session that holds the control-context
    record, which the chat only starts with a prompt. Its capture therefore
    opens the terminal off camera and may only write an element crop.
    """
    users = [
        name
        for name, fn in inspect.getmembers(capture, inspect.isfunction)
        if fn.__module__ == capture.__name__ and "TERMINAL_SURFACE_INIT" in inspect.getsource(fn)
    ]
    assert users == ["_capture_web_terminal_static"]

    behind = [
        shot
        for shot in REGISTRY
        if shot.environment == "tutorial_stack" and shot.kind == "static" and shot.panel is None
    ]
    assert [shot.name for shot in behind] == ["control_target_popover"]
    for shot in behind:
        assert shot.capture_mode == "element"
        assert shot.element_selector == ".ctc-popover:not([hidden])"


def test_the_popover_capture_refuses_a_full_page_shot(tmp_path) -> None:
    full_page = DocShot(name="probe", environment="tutorial_stack", kind="static")
    with pytest.raises(ValueError, match="must crop an element_selector"):
        capture._capture_web_terminal_static(object(), tmp_path, full_page)
