"""``web.panels.canopy.home_screen`` judged against the project's screen library.

The two commands that gate on the profile, ``osprey build`` and
``osprey validate``, call this producer beside the other profile-side checks,
once for the profile and once for each tier and app file. The key names the screen
CANOPY lands on by its library id; a build that rendered an id the library does
not hold would ship a panel that opens on nothing, so it is refused before
anything is written.

The key is inert while the profile does not select the CANOPY panel, and unset
(or empty) it means CANOPY's first screen, so neither case is judged. A value
that references the environment (it contains ``$``) is judged neither way: the
shell that runs the build is not the deployment's ``.env``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from osprey.integrations.canopy.library import HOME_SCREEN_KEY, find_home_screen

CANOPY_PANEL = "canopy"
#: The skills that go with the panel and leave the deployment with it.
CANOPY_SKILLS = ("canopy-panel-builder", "canopy-operator", "canopy-screens")
#: How many of the library's screens a refusal lists.
LISTED_SCREENS = 10


def canopy_home_screen_errors(
    config: Mapping[str, Any], selected_panels: Iterable[str] | None, library_root: Path
) -> list[str]:
    """One message per spelling of ``web.panels.canopy.home_screen`` the library does not hold.

    Args:
        config: The profile's merged ``config:`` block.
        selected_panels: The profile's ``web_panels``, as resolved.
        library_root: The project's screen library,
            ``<data root>/canopy/library``.

    Returns:
        list[str]: Messages naming the key as written, the id, the path looked
        in, one fix and up to ten screens the library holds.
    """
    if not isinstance(config, Mapping) or CANOPY_PANEL not in (selected_panels or ()):
        return []

    # Imported in-function: the reach registry behind `spelled_values` pulls
    # the whole service-resolution package in with it.
    from .build_profile_reach import spelled_values

    errors: list[str] = []
    for spelling, value in spelled_values(config, HOME_SCREEN_KEY):
        if value is None or value == "":
            continue
        if isinstance(value, str) and "$" in value:
            continue
        home = find_home_screen(library_root, value)
        if home.found:
            continue
        message = (
            f"The profile's config: block sets `{spelling}` to {value!r}, but {home.problem} "
            f"(looked in {home.looked_in}). {home.fix} {_holdings(home.screens)}"
        )
        errors.append(message)
    return errors


def _holdings(screens: tuple[str, ...]) -> str:
    if not screens:
        return "The library holds no screens."
    shown = ", ".join(repr(screen) for screen in screens[:LISTED_SCREENS])
    more = len(screens) - LISTED_SCREENS
    tail = f", and {more} more" if more > 0 else ""
    return f"The library holds {shown}{tail}."


def repo_home_screen_errors(
    profile: Any, repo_root: Path, overlays: tuple[Path, ...] = ()
) -> list[str]:
    """:func:`canopy_home_screen_errors` for a repo's profile and each of its tier and app files.

    Each layer file (:func:`~osprey.cli.build_cmd._layer_files`) is judged on
    the resolution the build renders it from, with the same overlays, and
    against the library under ``repo_root``'s data root; its messages are
    headed by the file's path. A layer that does not resolve on its own is not
    judged here: whether it resolves is the build's question, asked where it
    renders the layer, and ``osprey validate`` does not ask it of a repo.

    Args:
        profile: The repo profile's resolved ``BuildProfile``.
        repo_root: The directory holding ``profile.yml``.
        overlays: The host-variant overlays the profile was resolved with.
    """
    from osprey_connectors.errors import BuildProfileError

    from .build_cmd import _layer_files
    from .build_profile import resolve_build_document

    errors = profile_home_screen_errors(profile, repo_root)
    # A layer that inherits the profile's value inherits its refusal too, and
    # the one fix in profile.yml clears both, so it is reported once.
    inherited = set(errors)
    for _kind, layer in _layer_files(repo_root):
        try:
            layer_profile = resolve_build_document(layer, None, overlays).profile
        except BuildProfileError:
            continue
        heading = layer.relative_to(repo_root).as_posix()
        errors.extend(
            f"{heading}: {error}"
            for error in profile_home_screen_errors(layer_profile, repo_root)
            if error not in inherited
        )
    return errors


def profile_home_screen_errors(profile: Any, root: Path) -> list[str]:
    """:func:`canopy_home_screen_errors` for one resolved profile.

    The library is the one under the profile's data root, resolved against
    ``root``; a profile with no data root has no library to judge against.
    """
    data_root = profile.resolved_data_root(root)
    if data_root is None:
        return []
    return canopy_home_screen_errors(
        profile.config, profile.web_panels, data_root / "canopy" / "library"
    )


def without_unstaged_panel(profile: Any, repo_root: Path) -> Any:
    """*profile* without the CANOPY panel when the build stages no payload for it.

    With ``OSPREY_CANOPY_PAYLOAD`` unset (and staging not skipped) the image
    carries no CANOPY, so the panel, its skills and a ``default_panel`` naming it
    leave the render rather than ship broken. Any other profile is returned as is.
    """
    from dataclasses import replace

    from osprey.integrations.canopy.stage import panel_off

    if CANOPY_PANEL not in (profile.web_panels or ()) or not panel_off(repo_root):
        return profile
    dropped = replace(
        profile,
        web_panels=[panel for panel in profile.web_panels if panel != CANOPY_PANEL],
        default_panel=None if profile.default_panel == CANOPY_PANEL else profile.default_panel,
        skills=[skill for skill in profile.skills if skill not in CANOPY_SKILLS],
    )
    # Not a dataclass field, so `replace` does not carry it.
    dropped.inherited_preset = profile.inherited_preset
    return dropped
