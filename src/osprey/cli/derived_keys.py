"""The config keys the build renders, and the refusal of a second spelling.

``profile.yml`` is the whole declarative input: every key an operator may state
lives in its ``config:`` block, documented on the line that sets it. A handful
of keys are NOT the operator's to state — the framework template
(``templates/project/config.yml.j2``) writes them at build time from the
project layout, the port layout, ``providers.yml``, or a profile FIELD
(``provider:``, ``model:``, ``channel_finder_mode:``, ``default_panel:``,
``panel_presets:``). Those are the *derived* keys.

A ``config:`` entry for one of them is a second home for one fact, and the
losing copy is the silent one: the render overwrites it, so the profile says
``opus`` and the deployment runs whatever ``model:`` names, with nothing on
screen to say which won. :func:`derived_key_errors` refuses that spelling in
:meth:`osprey.cli.build_profile_model.BuildProfile.validate`, naming the field
to set instead.

Two rules make the match total:

* **Prefix-aware.** A member of :data:`DERIVED_KEYS` claims that key AND every
  key beneath it, so ``file_paths`` covers ``file_paths.docs_dir`` and
  ``channel_finder.pipelines`` covers every per-mode pipeline flag.
* **Every spelling.** ``config:`` may write a leaf as a whole dotted key, as a
  dotted prefix over a mapping, fully nested, or any mix — all of them reach
  the same rendered leaf (:func:`osprey.cli.build_profile_reach.spelled_values`
  makes the same point for a single key), so the walk here splits on ``.``
  at every level and reports the spelling as it was written.

``web.panels.<id>.enabled`` is deliberately absent: it is derived from
``web_panels:`` too, but a spelling that AGREES with the selection is accepted
there, so it stays with
:func:`osprey.cli.build_profile_panels.panel_selection_errors`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

__all__ = ["DERIVED_KEYS", "derived_key_errors", "is_derived_key"]


#: What each derived key is rendered from, phrased as the fix. Every member of
#: :data:`DERIVED_KEYS` has an entry; the mapping IS the key list.
_DERIVED_KEY_SOURCES: dict[str, str] = {
    # Project layout — the build knows where it is writing.
    "project_name": "the build takes it from the profile's `name:`",
    "project_root": "the build takes it from the repository it renders into",
    "build_dir": "the build takes it from the repository it renders into",
    "file_paths": "the build derives these paths from the project layout",
    "agent_data.base_dir": "the build derives it from the project layout",
    # The deployment's Python environment.
    "execution.environment.python": "the build derives it from the deployment's Python environment",
    "execution.environment.packages": (
        "the build derives it from the deployment's Python environment"
    ),
    "execution.environment.inherit_exclude": (
        "the build derives it from the deployment's Python environment"
    ),
    # The port layout.
    "artifact_server.port": "the build derives it from `deployment.port_base`",
    # Profile fields.
    "claude_code.provider": "the top-level `provider:` field sets it",
    "claude_code.default_model": "the top-level `model:` field sets it",
    "logbook.composition.provider": "the top-level `provider:` field sets it",
    "ariel.enhancement_modules.semantic_processor.provider": (
        "the top-level `provider:` field sets it"
    ),
    "ariel.enhancement_modules.semantic_processor.model.model_id": (
        "the top-level `model:` field sets it"
    ),
    "channel_finder.pipeline_mode": "the top-level `channel_finder_mode:` field sets it",
    "channel_finder.pipelines": "the top-level `channel_finder_mode:` field sets it",
    "web.default_panel": "the top-level `default_panel:` field sets it",
    "web.presets": "the top-level `panel_presets:` field sets it",
}

#: The dotted config keys the framework template owns. Each claims itself and
#: every key beneath it (see :func:`is_derived_key`).
DERIVED_KEYS: frozenset[str] = frozenset(_DERIVED_KEY_SOURCES)


def is_derived_key(dotted_key: str) -> bool:
    """Whether the build renders *dotted_key*, rather than the profile stating it.

    Args:
        dotted_key: A fully dotted config path, as a rendered ``config.yml``
            flattens to.

    Returns:
        ``True`` when *dotted_key* is a member of :data:`DERIVED_KEYS` or sits
        beneath one.
    """
    return _claiming_key(dotted_key) is not None


def derived_key_errors(config: Any) -> list[str]:
    """Refuse a ``config:`` block that spells a key the build renders.

    Args:
        config: The profile's ``config:`` block, whatever shape it parsed as —
            dotted keys, nested mappings, or a mix. A non-mapping is somebody
            else's refusal and yields nothing here.

    Returns:
        One error per offending spelling, naming the key, what supplies it, and
        the file to remove it from. Empty when the block states nothing derived.
    """
    errors: list[str] = []
    for spelling, dotted_key in _spelled_paths(config):
        claimed = _claiming_key(dotted_key)
        if claimed is None:
            continue
        named = dotted_key if spelling == dotted_key else f"{spelling} — {dotted_key}"
        errors.append(
            f"config: {named} is rendered by the build; "
            f"{_DERIVED_KEY_SOURCES[claimed]}. Remove it from profile.yml."
        )
    return errors


def _claiming_key(dotted_key: str) -> str | None:
    """The member of :data:`DERIVED_KEYS` that owns *dotted_key*, or ``None``."""
    segments = dotted_key.split(".")
    for cut in range(1, len(segments) + 1):
        candidate = ".".join(segments[:cut])
        if candidate in _DERIVED_KEY_SOURCES:
            return candidate
    return None


def _spelled_paths(
    node: Any, segments: tuple[str, ...] = (), written: tuple[str, ...] = ()
) -> Iterator[tuple[str, str]]:
    """Every path *node* writes, as ``(spelling as written, dotted path)``.

    Walks outermost-first and stops descending once a path is reported as
    derived, so a whole derived branch spelled nested (``file_paths:`` holding
    four leaves) is one refusal naming the branch, not four naming its leaves.
    """
    if not isinstance(node, Mapping):
        return
    for key, value in node.items():
        if not isinstance(key, str):
            continue
        here = (*segments, *key.split("."))
        spelling = ": ".join([*written, key])
        dotted = ".".join(here)
        yield spelling, dotted
        if _claiming_key(dotted) is None:
            yield from _spelled_paths(value, here, (*written, key))
