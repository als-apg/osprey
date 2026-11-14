"""Declarative registry of documentation screenshots.

``REGISTRY`` is the single source of truth for every committed doc image and how
it is produced. Each :class:`DocShot` names the *environment* it captures from,
its *kind*, themes, viewport, sub-views, and capture mode. The runner
(:mod:`docs.screenshots.capture`) and the ``conf.py`` caption hook both read
this list; adding a doc image is one ``DocShot`` line here.

Four environments cover every real case:

* ``standalone_interface`` — boot a single interface ``create_app()`` on a free
  port (zero container, deterministic). A default target of ``make screenshots``.
* ``static_page`` — serve a committed HTML file (hand-authored architecture
  diagrams under ``docs/diagrams/``) from a throwaway local HTTP server and crop
  a figure element out of it. Zero app, zero container; also on by default.
* ``hermetic_hub`` — boot the real web terminal through
  :func:`docs.screenshots.contact_sheet.hermetic_hub`: a seeded workspace and a
  canned terminal transcript, with zero containers and no provider. A recipe
  is a stage of the operator chat, or (``hub``) a page of a hub seeded for it.
  On by default.
* ``tutorial_stack`` — build the ``demo`` tutorial project, bring up
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

_ALLOWED_THEMES = frozenset({"light", "dark"})

# The seeded session the ``agent_record`` recipe opens (docs/screenshots/pages.py).
HISTORY_SESSION = "3f8a1c02-0000-4000-8000-000000000001"

# The recorded session the ``activity_safety`` recipe opens
# (docs/screenshots/activity_session.json).
ACTIVITY_SESSION = "42f96113-629c-46fb-b02a-dce13f29b727"


@dataclass(frozen=True)
class SubView:
    """One captured view within a single environment (env → many output files).

    An anchor starting with ``#`` is a hash route; anything else is a selector
    clicked after load.
    """

    anchor: str
    """Hash route (``"#create"``) or a selector clicked before capture."""

    out: str
    """Output filename stem, e.g. ``"ariel_screen"`` → ``ariel_screen.png``."""

    wait_selector: str | None = None
    """Optional selector to await (attached) after navigating, before shooting."""


@dataclass(frozen=True)
class DocShot:
    """A single declarative screenshot recipe."""

    name: str
    environment: Literal["standalone_interface", "static_page", "hermetic_hub", "tutorial_stack"]
    kind: Literal["static", "agentic"]
    themes: tuple[str, ...] = ("light", "dark")
    viewport: tuple[int, int] = (1280, 800)

    # tutorial_stack static: what the live web terminal shows
    panel: str | None = None
    """The panel a ``tutorial_stack`` *static* recipe opens in the live web
    terminal (``/?open=<panel>``); its waits and sub-view clicks run inside that
    panel's frame (see :func:`docs.screenshots.capture._capture_web_terminal_panel`).
    Without it the recipe crops the control-target popover
    (:func:`docs.screenshots.capture._capture_web_terminal_static`). Either way
    it is an ``osprey web`` launch against the built tutorial repo, with no agent
    turn and no provider key needed."""
    init_sets: tuple[str, ...] = ()
    """Extra ``osprey init --set KEY=VALUE`` values for the tutorial stack
    (``tutorial_stack`` only). Recipes naming the same values share one stack;
    a recipe with values of its own gets a stack of its own, so what it turns
    on never shows in another recipe's image."""

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

    # hermetic_hub
    stage: str | None = None
    """A key of :data:`docs.screenshots.contact_sheet.STAGES`: the UI state the
    page is driven into before the shot (``hermetic_hub`` environment only)."""

    hub: str | None = None
    """Dotted path to a context manager yielding a
    :class:`~docs.screenshots.contact_sheet.HermeticHub` (``hermetic_hub``
    environment only). The recipe is then a page of that hub -- ``path``,
    sub-views and waits as for a standalone interface -- instead of a stage of
    the operator chat."""

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
    # the terminal's open ☰ More menu, light + dark, for the theming how-to.
    DocShot(
        name="display_menu",
        environment="hermetic_hub",
        kind="static",
        stage="more_menu",
        capture_mode="element",
        element_selector="#more-menu",
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
    DocShot(
        name="install_flow",
        environment="static_page",
        kind="static",
        source_file="docs/diagrams/install_flow.html",
        capture_mode="element",
        element_selector="#fig-install-flow",
        wait_selector="#fig-install-flow svg",
        themes=("light", "dark"),
        viewport=(1400, 1000),
    ),
    # Multi-user pages (container-free), each from the code a deployment serves
    # them with, on seeded state: docs/screenshots/pages.py.
    # The landing page ``osprey build`` renders for the demo preset's roster.
    DocShot(
        name="multi_user_landing",
        environment="standalone_interface",
        kind="static",
        app_factory="docs.screenshots.pages:landing_app",
        wait_selector=".landing-main",
        themes=("light", "dark"),
        viewport=(1280, 800),
    ),
    # The password sign-in page, reached from alice's card.
    DocShot(
        name="sign_in",
        environment="standalone_interface",
        kind="static",
        app_factory="docs.screenshots.pages:sign_in_app",
        path="/auth/login?user=alice&next=/u/alice/",
        wait_selector="main.login-card #password",
        themes=("light", "dark"),
        viewport=(1280, 800),
    ),
    # One session on the history page: prompt, thinking, tool calls with their
    # decisions and results, a subagent's steps, a refused write, the answer.
    DocShot(
        name="agent_record",
        environment="standalone_interface",
        kind="static",
        app_factory="docs.screenshots.pages:history_app",
        path=f"/history/session/{HISTORY_SESSION}",
        wait_selector=".transcript .entry.answer",
        themes=("light", "dark"),
        viewport=(1280, 420),
    ),
    # The Activity page's Safety view of one recorded session: who allowed each
    # call, the one approval the operator gave, and the calls a server gate checked.
    DocShot(
        name="activity_safety",
        environment="hermetic_hub",
        kind="static",
        hub="docs.screenshots.pages:activity_hub",
        path="/static/session.html",
        subviews=(
            SubView(
                anchor=f"#{ACTIVITY_SESSION}/safety",
                out="activity_safety",
                wait_selector="#inner table",
            ),
        ),
        themes=("light", "dark"),
        # The session alone, whole: the session list beside it names days
        # relative to today, and the viewport is tall enough that nothing scrolls.
        capture_mode="element",
        element_selector="#inner",
        viewport=(1280, 1700),
    ),
    # ARIEL (opt-in ``--stack``), in the live web terminal. One tutorial stack →
    # one PNG of the one screen: rail, list and the seeded logbook's newest
    # entry, clicked into the reader. Single-theme (light) → no theme suffix;
    # viewport 1280x900.
    DocShot(
        name="ariel",
        environment="tutorial_stack",
        kind="static",
        panel="ariel",
        themes=("light",),
        viewport=(1280, 900),
        wait_selector='#ariel-list [data-entry-id][aria-selected="true"]',
        subviews=(
            SubView(
                ".ariel-row >> nth=0",
                "ariel_screen",
                wait_selector="#ariel-reader-body .entry-detail",
            ),
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
    # The control-target popover (opt-in ``--stack``): the panel behind the
    # sidebar's control-system chip, captured live from the tutorial
    # deployment's web terminal.
    # The shot shows the full shape — the active machine's card with its writes
    # switch and a row per other machine — real roster, real states, nothing
    # staged. The
    # popover only exists open, and only renders banner-free once the session
    # holds the deployment's control-context record, so the flow lives in
    # capture._capture_web_terminal_static rather than in a subview click. That
    # flow starts the session from the Claude Code terminal, off camera: the
    # shot must stay an element crop of the popover.
    DocShot(
        name="control_target_popover",
        environment="tutorial_stack",
        kind="static",
        capture_mode="element",
        element_selector=".ctc-popover:not([hidden])",
        themes=("light", "dark"),
        viewport=(1280, 800),
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
      * ``panel`` and ``init_sets`` require the ``tutorial_stack`` environment,
        and every ``init_sets`` value is ``KEY=VALUE``;
      * a ``stage`` requires the ``hermetic_hub`` environment and must be a key
        of :data:`docs.screenshots.contact_sheet.STAGES`;
      * a ``hub`` requires the ``hermetic_hub`` environment, excludes a
        ``stage`` and names a ``path`` other than ``/``; without one, a ``hermetic_hub`` recipe is the operator chat
        at ``/`` with no sub-views;
      * ``themes`` is a non-empty subset of ``{light, dark}``.
    """
    from docs.screenshots.contact_sheet import STAGES

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

        if shot.panel is not None and shot.environment != "tutorial_stack":
            raise ValueError(f"{shot.name!r}: panel requires the tutorial_stack environment")

        if shot.init_sets and shot.environment != "tutorial_stack":
            raise ValueError(f"{shot.name!r}: init_sets requires the tutorial_stack environment")
        for pair in shot.init_sets:
            if "=" not in pair:
                raise ValueError(f"{shot.name!r}: init_sets value {pair!r} is not KEY=VALUE")

        if shot.environment == "static_page" and not shot.source_file:
            raise ValueError(f"{shot.name!r}: static_page recipes require a source_file")

        if shot.hub is not None:
            if shot.environment != "hermetic_hub":
                raise ValueError(f"{shot.name!r}: a hub requires the hermetic_hub environment")
            if shot.stage is not None:
                raise ValueError(f"{shot.name!r}: a hub recipe is a page, not a chat stage")
            if shot.path == "/":
                raise ValueError(f"{shot.name!r}: a hub recipe names a page other than '/'")
        elif shot.environment == "hermetic_hub" and (shot.path != "/" or shot.subviews):
            raise ValueError(
                f"{shot.name!r}: a hermetic_hub recipe with a path or sub-views requires a hub"
            )

        if shot.stage is not None:
            if shot.environment != "hermetic_hub":
                raise ValueError(f"{shot.name!r}: a stage requires the hermetic_hub environment")
            if shot.stage not in STAGES:
                raise ValueError(
                    f"{shot.name!r}: unknown stage {shot.stage!r}; known: {sorted(STAGES)}"
                )


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

    Default (no flags) runs the container-free recipes: ``standalone_interface``,
    ``static_page`` and ``hermetic_hub`` statics. ``--stack`` adds ``tutorial_stack`` static
    recipes; ``--agentic`` adds agentic recipes.
    """
    if shot.kind == "agentic":
        return agentic
    if shot.environment == "tutorial_stack":
        return stack
    return True  # standalone_interface / static_page / hermetic_hub static — always on


def select_recipes(
    *, stack: bool = False, agentic: bool = False, only: str | None = None
) -> list[DocShot]:
    """Return the recipes to run for the given flags (and optional ``--only`` name)."""
    selected = [s for s in REGISTRY if is_enabled(s, stack=stack, agentic=agentic)]
    if only is not None:
        selected = [s for s in selected if s.name == only]
    return selected
