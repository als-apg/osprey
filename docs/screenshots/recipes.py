"""Declarative registry of documentation screenshots.

``REGISTRY`` is the single source of truth for every committed doc image and how
it is produced. Each :class:`DocShot` names the *environment* it captures from,
its *kind*, themes, viewport, sub-views, and capture mode. The runner
(:mod:`docs.screenshots.capture`) and the ``conf.py`` caption hook both read
this list; adding a doc image is one ``DocShot`` line here.

Three environments cover every real case:

* ``standalone_interface`` — boot a single interface ``create_app()`` on a free
  port (zero container, deterministic). A default target of ``make screenshots``.
* ``static_page`` — serve a committed HTML file (hand-authored architecture
  diagrams under ``docs/diagrams/``) from a throwaway local HTTP server and crop
  a figure element out of it. Zero app, zero container; also on by default.
* ``tutorial_stack`` — build the ``control-assistant`` tutorial project, bring up
  Postgres, and seed ARIEL via the product's own commands. Opt-in (``--stack``);
  agentic recipes (the web-terminal hero) additionally need ``--agentic``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Fixed output location; filenames keep parity with the consuming ``.rst`` figures.
OUTPUT_SUBDIR = "source/_static/screenshots"
MANIFEST_NAME = "manifest.json"

# Determinism anchor: seeded ARIEL logbook dates resolve their relative
# ``days_ago``/``at_time`` against this frozen instant, so ``--stack`` captures
# are byte-stable across runs. Passed to ``osprey sim apply --now``.
ANCHOR = "2024-03-18T12:00:00"

_ALLOWED_THEMES = frozenset({"light", "dark"})


@dataclass(frozen=True)
class SubView:
    """One captured view within a single environment (env → many output files).

    ARIEL, for example, is one running app whose ``#search``/``#browse``/
    ``#create``/``#status`` hash routes each become a separate PNG.
    """

    anchor: str
    """Hash route (``"#search"``) or activation selector navigated before capture."""

    out: str
    """Output filename stem, e.g. ``"ariel_search"`` → ``ariel_search.png``."""

    wait_selector: str | None = None
    """Optional selector to await (attached) after navigating, before shooting."""


@dataclass(frozen=True)
class DocShot:
    """A single declarative screenshot recipe."""

    name: str
    environment: Literal["standalone_interface", "static_page", "tutorial_stack"]
    kind: Literal["static", "agentic"]
    themes: tuple[str, ...] = ("light", "dark")
    viewport: tuple[int, int] = (1280, 800)

    # standalone_interface / tutorial_stack app boot
    app_factory: str | None = None
    """Dotted path to a ``create_app`` callable, e.g. ``"osprey.interfaces.artifacts.app:create_app"``."""
    path: str = "/"
    """URL path (relative to the app base) to capture."""

    # static_page
    source_file: str | None = None
    """Repo-relative path to a committed HTML file (``static_page`` environment).

    The runner serves the file's directory over a throwaway local HTTP server and
    navigates to the file with the usual ``?theme=`` parameter appended, so the
    page itself decides how a theme name maps onto its CSS (see
    ``docs/diagrams/retrieval_map.html`` for the ``data-theme`` shim). Combine
    with ``capture_mode="element"`` to crop a single ``<figure>`` out of a page
    holding several diagrams."""

    # env → many files
    subviews: tuple[SubView, ...] = ()
    """When non-empty, one running environment yields one PNG per sub-view."""

    # capture shaping
    capture_mode: Literal["full_page", "element"] = "full_page"
    element_selector: str | None = None
    """CSS selector for ``capture_mode="element"`` (the element is cropped)."""
    wait_selector: str | None = None
    """Selector to await (attached) after load, before capturing."""

    # agentic (tutorial_stack only)
    prompt: str | None = None
    """Operator prompt driven into a live terminal for ``kind="agentic"`` recipes."""
    wait_for: str | None = None
    """Substring an ``artifact_type`` must contain before the agentic capture (e.g. ``"plot"``)."""

    def output_names(self) -> list[str]:
        """Every base output name this recipe writes (theme suffix added by the runner)."""
        if self.subviews:
            return [sv.out for sv in self.subviews]
        return [self.name]


# ---------------------------------------------------------------------------
# The registry. Phase-2 recipe tasks append their DocShots here.
# ---------------------------------------------------------------------------

REGISTRY: list[DocShot] = [
    # Default (container-free) target of ``make screenshots``: an element crop of
    # the design-system theme switcher, light + dark, for the theming how-to.
    # ``channel_finder`` is the lightest interface that still mounts the switcher
    # in its own standalone header — its ``create_app()`` takes no required
    # argument and needs no workspace or backend, unlike ``artifacts``/``lattice``
    # which need a workspace_root to show anything useful. (ARIEL is no longer a
    # candidate: its header
    # delegates theming to the hub and no longer mounts the switcher itself.)
    DocShot(
        name="theme_switcher",
        environment="standalone_interface",
        kind="static",
        app_factory="osprey.interfaces.channel_finder.app:create_app",
        capture_mode="element",
        element_selector="osprey-theme-switcher",
        wait_selector="osprey-theme-switcher",
        themes=("light", "dark"),
    ),
    # Architecture diagrams: hand-authored SVG in a committed HTML page, one
    # PNG per theme, cropped per figure at 2x. Edit docs/diagrams/*.html, then
    # `make screenshots-<name>` regenerates the committed images.
    DocShot(
        name="retrieval_map",
        environment="static_page",
        kind="static",
        source_file="docs/diagrams/retrieval_map.html",
        capture_mode="element",
        element_selector="#fig-retrieval-map",
        wait_selector="#fig-retrieval-map svg",
        themes=("light", "dark"),
        viewport=(1400, 1000),
    ),
    DocShot(
        name="qmd_query_pipeline",
        environment="static_page",
        kind="static",
        source_file="docs/diagrams/retrieval_map.html",
        capture_mode="element",
        element_selector="#fig-qmd-query",
        wait_selector="#fig-qmd-query svg",
        themes=("light", "dark"),
        viewport=(1400, 1000),
    ),
    # ARIEL views (opt-in ``--stack``). One tutorial stack → four PNGs, one per
    # hash-routed view. Keyword mode is ARIEL's default, so ``#search`` shows the
    # keyword form + results with no extra activation (Postgres-only, no Ollama).
    # Single-theme (light) → no theme suffix, matching the committed filenames;
    # viewport 1280x900 matches the committed aspect so the .rst needs no re-layout.
    DocShot(
        name="ariel",
        environment="tutorial_stack",
        kind="static",
        themes=("light",),
        viewport=(1280, 900),
        subviews=(
            SubView("#search", "ariel_search", wait_selector="#view-search"),
            SubView("#browse", "ariel_browse", wait_selector="#view-browse"),
            SubView("#create", "ariel_create", wait_selector="#view-create"),
            SubView("#status", "ariel_status", wait_selector="#view-status"),
        ),
    ),
    # Web Terminal hero (opt-in ``--agentic``). Drives a live agent session on the
    # tutorial stack to produce a real beam-current plot, captured light + dark.
    DocShot(
        name="web_terminal_hero",
        environment="tutorial_stack",
        kind="agentic",
        themes=("light", "dark"),
        viewport=(1280, 800),
        prompt="Plot the storage ring beam current over the last hour.",
        wait_for="plot",
    ),
]


# ---------------------------------------------------------------------------
# Validation & selection (shared by the CLI, the runner, and the tests)
# ---------------------------------------------------------------------------


def validate_registry(registry: list[DocShot] | None = None) -> None:
    """Raise ``ValueError`` if any recipe is malformed or names collide.

    Rules:
      * recipe ``name`` values are unique, and so are all output filenames;
      * ``capture_mode="element"`` requires ``element_selector``;
      * ``kind="agentic"`` requires ``prompt`` and ``wait_for`` and lives on
        the ``tutorial_stack`` environment;
      * ``standalone_interface`` recipes require an ``app_factory``;
      * ``static_page`` recipes require a ``source_file``;
      * ``themes`` is a non-empty subset of ``{light, dark}``.
    """
    reg = REGISTRY if registry is None else registry
    seen_names: set[str] = set()
    seen_outputs: set[str] = set()
    for shot in reg:
        if shot.name in seen_names:
            raise ValueError(f"duplicate recipe name: {shot.name!r}")
        seen_names.add(shot.name)

        for out in shot.output_names():
            if out in seen_outputs:
                raise ValueError(f"duplicate output filename {out!r} (recipe {shot.name!r})")
            seen_outputs.add(out)

        if not shot.themes or not set(shot.themes) <= _ALLOWED_THEMES:
            raise ValueError(
                f"{shot.name!r}: themes {shot.themes!r} must be a non-empty subset of {sorted(_ALLOWED_THEMES)}"
            )

        if shot.capture_mode == "element" and not shot.element_selector:
            raise ValueError(f"{shot.name!r}: capture_mode='element' requires element_selector")

        if shot.kind == "agentic":
            if not shot.prompt or not shot.wait_for:
                raise ValueError(f"{shot.name!r}: agentic recipes require prompt and wait_for")
            if shot.environment != "tutorial_stack":
                raise ValueError(
                    f"{shot.name!r}: agentic recipes must use the tutorial_stack environment"
                )

        if shot.environment == "standalone_interface" and not shot.app_factory:
            raise ValueError(f"{shot.name!r}: standalone_interface recipes require an app_factory")

        if shot.environment == "static_page" and not shot.source_file:
            raise ValueError(f"{shot.name!r}: static_page recipes require a source_file")


# ---------------------------------------------------------------------------
# Manifest + caption provenance (consumed by docs/source/conf.py)
# ---------------------------------------------------------------------------

# Caption value for a recipe with no manifest entry yet — keeps the docs build
# working on a fresh clone (every ``|captured_<name>|`` is always defined).
CAPTION_PLACEHOLDER = "an unreleased build"


def manifest_path() -> Path:
    """Absolute path to the committed capture manifest."""
    return Path(__file__).parent.parent / OUTPUT_SUBDIR / MANIFEST_NAME


def load_manifest() -> dict:
    """Read the capture manifest, or ``{}`` when it is missing or malformed."""
    try:
        data = json.loads(manifest_path().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def caption_substitutions(manifest: dict | None = None) -> dict[str, str]:
    """Map each recipe's ``captured_<name>`` substitution to its caption value.

    The keys are driven by the recipe registry (not by manifest presence), so
    every ``|captured_<name>|`` is always defined and the docs build never fails
    on a missing manifest. The value is ``v<osprey_version>`` from the manifest
    for a captured recipe, or :data:`CAPTION_PLACEHOLDER` otherwise.
    """
    data = load_manifest() if manifest is None else manifest
    subs: dict[str, str] = {}
    for shot in REGISTRY:
        version = (data.get(shot.name) or {}).get("osprey_version")
        subs[f"captured_{shot.name}"] = f"v{version}" if version else CAPTION_PLACEHOLDER
    return subs


def is_enabled(shot: DocShot, *, stack: bool, agentic: bool) -> bool:
    """Whether a recipe runs given the opt-in flags.

    Default (no flags) runs the container-free recipes: ``standalone_interface``
    and ``static_page`` statics. ``--stack`` adds ``tutorial_stack`` static
    recipes; ``--agentic`` adds agentic recipes.
    """
    if shot.kind == "agentic":
        return agentic
    if shot.environment == "tutorial_stack":
        return stack
    return True  # standalone_interface / static_page static — always on


def select_recipes(
    *, stack: bool = False, agentic: bool = False, only: str | None = None
) -> list[DocShot]:
    """Return the recipes to run for the given flags (and optional ``--only`` name)."""
    selected = [s for s in REGISTRY if is_enabled(s, stack=stack, agentic=agentic)]
    if only is not None:
        selected = [s for s in selected if s.name == only]
    return selected
