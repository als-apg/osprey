"""Tests for the contact-sheet demo-workspace seed and canned transcript.

The seed's contract is that a hand-written workspace loads through the real
:class:`~osprey.stores.artifact_store.ArtifactStore` tolerant loader exactly as
if the product had written it — so the contact-sheet renderer can point a fresh
store at it with no live agent. The transcript's contract is that every line
fits the narrow terminal card.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest
from docs.screenshots.contact_sheet import (
    _PLOT_DRAWN_SELECTOR,
    ACCENT_EXCLUSIONS,
    CONTACT_SHEET_NAME,
    DEMO_PLOT_ARTIFACT_ID,
    DEMO_SESSION_ID,
    DEMO_TRANSCRIPT_PATH,
    EXTRA_VARIANTS,
    MAX_CARD_LINE_WIDTH,
    STAGED_VARIANTS,
    STAGES,
    TERMINAL_DIMS_SEAM,
    TRANSCRIPT_SENTINEL,
    VARIANTS,
    CapturedVariant,
    HermeticHub,
    Stage,
    _accent_override_css,
    _accent_override_var_names,
    _assert_accent_map_covers_tokens,
    _assert_fits_columns,
    _beam_current_plot_html,
    _effective_variants,
    _fake_session_line,
    _plot_frame_selector,
    _read_fitted_cols,
    _replay_shell_command,
    _variant_filename,
    _variant_label,
    _variant_url,
    _wait_for_plot_drawn,
    _write_fake_session,
    capture_hub_view,
    compose_contact_sheet,
    hermetic_hub,
    longest_transcript_line_width,
    seed_demo_workspace,
)
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from osprey.cli.project_utils import encode_claude_project_path
from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery
from osprey.stores.artifact_store import ArtifactStore


def _get(url: str) -> tuple[int, bytes]:
    """GET *url*, returning (status, body) — loopback only, short timeout.

    Both servers :func:`hermetic_hub` boots are ordinary interface apps, so both
    are gated by ``WebAuthMiddleware`` and refuse an uncredentialed request with
    ``401``. The browser-facing seam (``authorize_browser_context``, which the
    capture runner uses) does not apply here: this helper is a raw HTTP client,
    not a browser, and holds no cookie jar. It authenticates the way every other
    non-browser harness caller does — the operator-secret header, read from
    *this process's* credential holder, which is the very holder the in-process
    gate verifies against. The header path is exempt from the Origin check, so a
    plain ``urlopen`` with no ``Origin`` passes.
    """
    from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER
    from osprey.interfaces.web_auth import get_web_credentials

    request = urllib.request.Request(url)  # loopback
    request.add_header(OPERATOR_SECRET_HEADER, get_web_credentials().operator_secret)
    with urllib.request.urlopen(request, timeout=10) as resp:  # loopback
        return resp.status, resp.read()


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def test_seed_writes_index_and_content_files(tmp_path: Path) -> None:
    """Seeding writes artifacts.json plus one content file per entry."""
    entries = seed_demo_workspace(tmp_path)

    artifacts_dir = tmp_path / "artifacts"
    assert (artifacts_dir / "artifacts.json").is_file()
    assert len(entries) >= 3
    for entry in entries:
        content_path = artifacts_dir / entry.filename
        assert content_path.is_file(), f"missing content file for {entry.id}"
        assert content_path.stat().st_size == entry.size_bytes


def test_seed_entries_load_via_artifact_store(tmp_path: Path) -> None:
    """The hand-seeded index loads through ArtifactStore's tolerant loader."""
    seeded = seed_demo_workspace(tmp_path)

    store = ArtifactStore(workspace_root=tmp_path)
    loaded = store.list_entries()

    assert len(loaded) == len(seeded)
    assert {e.id for e in loaded} == {e.id for e in seeded}
    # Round-trip fidelity on the caption-bearing fields the renderer relies on.
    for want, got in zip(seeded, loaded, strict=True):
        assert got.title == want.title
        assert got.description == want.description
        assert got.artifact_type == want.artifact_type


def test_seed_entries_share_one_session_id(tmp_path: Path) -> None:
    """Every seeded artifact carries the single fixed demo session id."""
    seed_demo_workspace(tmp_path)

    store = ArtifactStore(workspace_root=tmp_path)
    session_ids = {e.session_id for e in store.list_entries()}

    assert session_ids == {DEMO_SESSION_ID}


def test_seed_covers_three_types_including_plotly(tmp_path: Path) -> None:
    """At least three artifact types, one a real Plotly HTML beam-current plot."""
    seed_demo_workspace(tmp_path)

    store = ArtifactStore(workspace_root=tmp_path)
    entries = store.list_entries()

    types = {e.artifact_type for e in entries}
    assert len(types) >= 3
    assert "plot_html" in types

    plot = next(e for e in entries if e.artifact_type == "plot_html")
    html = (tmp_path / "artifacts" / plot.filename).read_text(encoding="utf-8")
    assert "plotly-graph-div" in html  # a genuine Plotly render, not a stub
    assert "beam current" in html.lower()


def test_seed_descriptions_serve_as_captions(tmp_path: Path) -> None:
    """Simple-mode captions come from description, so every entry has one."""
    seed_demo_workspace(tmp_path)

    store = ArtifactStore(workspace_root=tmp_path)
    for entry in store.list_entries():
        assert entry.description.strip(), f"{entry.id} has no caption"


def test_seed_is_deterministic(tmp_path: Path) -> None:
    """Re-seeding produces byte-identical content (stable contact sheet)."""
    first = seed_demo_workspace(tmp_path / "a")
    second = seed_demo_workspace(tmp_path / "b")

    for e1, e2 in zip(first, second, strict=True):
        c1 = (tmp_path / "a" / "artifacts" / e1.filename).read_bytes()
        c2 = (tmp_path / "b" / "artifacts" / e2.filename).read_bytes()
        assert c1 == c2


# ---------------------------------------------------------------------------
# Transcript + line-width guard
# ---------------------------------------------------------------------------


def test_transcript_fits_terminal_card() -> None:
    """No transcript line exceeds the narrow (~370px) card's column budget."""
    assert longest_transcript_line_width() <= MAX_CARD_LINE_WIDTH


def test_transcript_has_prompt_computing_output_and_sentinel() -> None:
    """The canned exchange has a prompt, a computing line, output, and sentinel."""
    text = DEMO_TRANSCRIPT_PATH.read_text(encoding="utf-8")

    assert "> Plot ring current" in text  # user prompt
    assert "Computing" in text  # computing line
    assert "500.2 mA" in text  # tool output
    assert TRANSCRIPT_SENTINEL in text  # distinctive completion sentinel


def test_line_width_guard_ignores_ansi_codes() -> None:
    """Visible width strips ANSI SGR escapes rather than counting their bytes."""
    from docs.screenshots.contact_sheet import _visible_width

    assert _visible_width("\\033[1;32m__OSPREY_SHEET_READY__\\033[0m") == len(
        "__OSPREY_SHEET_READY__"
    )


# ---------------------------------------------------------------------------
# Hermetic hub launcher
# ---------------------------------------------------------------------------


def test_launcher_replay_command_expands_ansi_not_cat() -> None:
    """The PTY replay uses printf %b (renders ANSI), not a literal cat."""
    argv = _replay_shell_command(DEMO_TRANSCRIPT_PATH)

    assert argv[:2] == ["bash", "-c"]
    assert "printf '%b" in argv[2]  # expands \033[...m escapes
    assert "cat " in argv[2]
    assert str(DEMO_TRANSCRIPT_PATH.resolve()) in argv[2]


def test_launcher_boots_hub_and_serves_seeded_artifacts() -> None:
    """The launcher yields a live hub plus its embedded, seeded artifacts backend."""
    with hermetic_hub() as hub:
        # The hub page is live and renders.
        status, body = _get(hub.base_url + "/")
        assert status == 200
        assert body, "hub returned an empty page"

        # The embedded artifacts backend serves the seeded store.
        _, raw = _get(hub.artifact_url + "/api/artifacts")
        artifacts = json.loads(raw).get("artifacts", [])
        assert len(artifacts) >= 3
        titles = " ".join(a.get("title", "") for a in artifacts).lower()
        assert "beam current" in titles
        # Every seeded artifact carries the one fixed demo session id.
        assert {a.get("session_id") for a in artifacts} == {DEMO_SESSION_ID}

        # Session discovery resolves to the encoded project dir — never hand-built.
        expected = (
            Path.home() / ".claude" / "projects" / encode_claude_project_path(hub.project_dir)
        )
        assert hub.session_dir == expected


def test_launcher_cleans_up_project_and_session_dirs_on_exit() -> None:
    """Both the seeded project dir and its session dir are removed on exit."""
    with hermetic_hub() as hub:
        project_dir = hub.project_dir
        session_dir = hub.session_dir
        # Stand in for the fake session record a later capture step would write.
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "fake.jsonl").write_text("{}\n", encoding="utf-8")
        assert project_dir.exists()

    assert not project_dir.exists()
    assert not session_dir.exists()


# ---------------------------------------------------------------------------
# Capture-variant loop (browser-free unit coverage; the end-to-end capture is
# exercised by the module's CLI gate)
# ---------------------------------------------------------------------------


def test_capture_variants_cover_dark_and_light() -> None:
    """The default variant list is the full theme × UI-mode matrix."""
    for theme in ("dark", "light"):
        for mode in ("expert", "simple"):
            assert (theme, mode) in VARIANTS
    assert len(VARIANTS) == 4
    # Output filenames are unique across variants.
    names = [_variant_filename(theme, mode) for theme, mode in VARIANTS]
    assert len(names) == len(set(names))


def test_capture_variant_filename_and_url() -> None:
    """Mode/rail are absent from both filename and URL until one is set."""
    assert _variant_filename("dark", None) == "web_terminal_dark.png"
    assert _variant_filename("light", "simple") == "web_terminal_light_simple.png"
    assert (
        _variant_filename("dark", "expert", rail="top") == "web_terminal_dark_expert_rail-top.png"
    )

    assert _variant_url("http://h", "dark", None) == "http://h/?theme=dark"
    assert _variant_url("http://h", "light", "simple") == "http://h/?theme=light&mode=simple"
    assert _variant_url("http://h", "dark", "expert", rail="top").endswith("&rail=top")


def test_staged_variants_are_expert_and_named() -> None:
    """Every staged card is an Expert hub card naming a known stage, and its
    filename collides with no base or showcase card."""
    assert STAGED_VARIANTS
    for _theme, mode, stage in STAGED_VARIANTS:
        assert mode == "expert"
        assert stage in STAGES
    base = [_variant_filename(theme, mode) for theme, mode in VARIANTS]
    extra = [_variant_filename(theme, mode, rail=rail) for theme, mode, rail in EXTRA_VARIANTS]
    staged = [_variant_filename(theme, mode, stage=stage) for theme, mode, stage in STAGED_VARIANTS]
    names = base + extra + staged
    assert len(names) == len(set(names))
    assert (
        _variant_filename("dark", "expert", stage="customize_sheet")
        == "web_terminal_dark_expert_customize_sheet.png"
    )


def test_extra_variants_unique_filenames() -> None:
    """The showcase extras never collide with the base matrix or each other."""
    base = [_variant_filename(theme, mode) for theme, mode in VARIANTS]
    extra = [_variant_filename(theme, mode, rail=rail) for theme, mode, rail in EXTRA_VARIANTS]
    names = base + extra
    assert len(names) == len(set(names))
    # Every extra is either a retro theme or a top-rail card — the showcase
    # never silently duplicates a base 2×2 cell.
    for theme, _mode, rail in EXTRA_VARIANTS:
        assert theme.startswith("retro") or rail is not None


def test_capture_fake_session_is_discoverable(tmp_path: Path) -> None:
    """The fake JSONL parses as a session whose id equals DEMO_SESSION_ID."""
    path = _write_fake_session(tmp_path)
    assert path.name == f"{DEMO_SESSION_ID}.jsonl"

    info = SessionDiscovery._parse_session_file(path)
    assert info is not None
    assert info.session_id == DEMO_SESSION_ID
    assert info.first_message  # a non-empty first user message

    # It is valid JSONL.
    json.loads(path.read_text(encoding="utf-8").strip())
    assert json.loads(_fake_session_line())["sessionId"] == DEMO_SESSION_ID


def test_capture_line_width_guard() -> None:
    """The fitted-column guard passes when wide and raises when too narrow."""
    widest = longest_transcript_line_width()
    _assert_fits_columns(widest)  # exactly fits — no raise
    _assert_fits_columns(widest + 10)  # comfortably fits
    with pytest.raises(RuntimeError, match="wrap"):
        _assert_fits_columns(widest - 1)


class _FakePage:
    """A stand-in Playwright page, so the fitted-cols reader is testable headless.

    Records every script it is handed, so a test can assert WHAT the reader asks
    the page for and not merely what it answers.
    """

    def __init__(self, *, dims: object, term_width: int = 370, ready: bool = True) -> None:
        self.dims = dims
        self.term_width = term_width
        self.ready = ready
        self.scripts: list[str] = []

    # The reader under test passes the wait budget by keyword.
    def wait_for_function(self, expression: str, timeout: int | None = None) -> None:  # noqa: ARG002
        self.scripts.append(expression)
        if not self.ready:
            raise TimeoutError("the seam never reported a fitted size")

    def evaluate(self, expression: str) -> object:
        self.scripts.append(expression)
        return self.term_width if ".xterm" in expression else self.dims


class _FakeLocator:
    """A stand-in Playwright locator that records clicks and plot waits on its page."""

    def __init__(self, page: _FakeHubPage, frames: tuple[str, ...], selector: str) -> None:
        self.page = page
        self.frames = frames
        self.selector = selector

    @property
    def first(self) -> _FakeLocator:
        return self

    def click(self, timeout: int | None = None) -> None:  # noqa: ARG002
        self.page.events.append("click")

    def wait_for(self, state: str, timeout: int | None = None) -> None:  # noqa: ARG002
        self.page.events.append(f"plot:{state}")
        self.page.waits.append((self.frames, self.selector, state))
        if self.page.wait_error is not None:
            raise self.page.wait_error
        if not self.page.drawn.pop(0):
            raise PlaywrightTimeoutError("Timeout 30000ms exceeded.")


class _FakeFrame:
    """A stand-in frame locator carrying the chain of iframe selectors that led to it."""

    def __init__(self, page: _FakeHubPage, frames: tuple[str, ...]) -> None:
        self.page = page
        self.frames = frames

    def frame_locator(self, selector: str) -> _FakeFrame:
        return _FakeFrame(self.page, (*self.frames, selector))

    def locator(self, selector: str) -> _FakeLocator:
        return _FakeLocator(self.page, self.frames, selector)


class _FakeRequest:
    """A stand-in ``page.request`` that records the teardown restart."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def post(self, url: str, headers: dict[str, str] | None = None) -> None:  # noqa: ARG002
        self.events.append("restart")


class _FakeHubPage:
    """A stand-in Playwright page for :func:`capture_hub_view`, recording its steps in order.

    ``drawn`` holds one answer per plot wait: ``False`` makes that wait time out.
    """

    def __init__(self, *, drawn: list[bool], wait_error: Exception | None = None) -> None:
        self.events: list[str] = []
        self.waits: list[tuple[tuple[str, ...], str, str]] = []
        self.drawn = list(drawn)
        self.wait_error = wait_error
        self.context = object()
        self.request = _FakeRequest(self.events)

    def add_init_script(self, script: str) -> None:
        pass

    def goto(self, url: str, **kwargs: object) -> None:
        pass

    def wait_for_function(self, expression: str, **kwargs: object) -> None:
        pass

    def wait_for_timeout(self, ms: int) -> None:  # noqa: ARG002
        self.events.append("settle")

    def screenshot(self) -> bytes:
        self.events.append("shot")
        return b"png"

    def close(self) -> None:
        self.events.append("close")

    def frame_locator(self, selector: str) -> _FakeFrame:
        return _FakeFrame(self, (selector,))


class _FakeBrowser:
    """A stand-in Playwright browser that hands out one prepared page."""

    def __init__(self, page: _FakeHubPage) -> None:
        self.page = page

    def new_page(self, viewport: object = None) -> _FakeHubPage:  # noqa: ARG002
        return self.page


_ARTIFACTS_PANEL = 'iframe.panel-iframe[data-panel-id="artifacts"]'


def _drive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    page: _FakeHubPage,
    *,
    mode: str = "expert",
    stage: str | None = None,
) -> None:
    """Run :func:`capture_hub_view` against *page* with the live-hub waits stubbed out."""
    from docs.screenshots import contact_sheet as cs

    import osprey.interfaces._serving as serving

    monkeypatch.setattr(cs, "_wait_for_session_ready", lambda page, mode: None)
    monkeypatch.setattr(cs, "_read_fitted_cols", lambda page: None)
    monkeypatch.setattr(serving, "authorize_browser_context", lambda context: None)
    hub = HermeticHub("http://127.0.0.1:9", "http://127.0.0.1:9", tmp_path, tmp_path / "sessions")
    capture_hub_view(_FakeBrowser(page), hub, "light", mode, tmp_path / "shot.png", stage=stage)


def test_fitted_cols_reader_distinguishes_none_from_a_number() -> None:
    """``None`` (skip the wrap guard) and a number (enforce it) must stay distinct.

    :func:`capture_contact_sheet`'s per-variant check is
    ``if fitted_cols is not None: _assert_fits_columns(fitted_cols)``. That makes
    ``None`` a silent OFF switch for the whole wrap guard: a reader that drifted
    into answering ``None`` on the live path too would turn the guard off and
    nothing anywhere would go red — the contact sheet would just start rendering
    wrapped transcripts. So the live path is pinned to a real ``int`` and every
    skip path to ``None``, and the two cannot collapse into one another.
    """
    live = _read_fitted_cols(_FakePage(dims={"cols": 96, "rows": 24}))
    assert live == 96
    assert isinstance(live, int) and not isinstance(live, bool)

    # The three legitimate skips, each for a different reason.
    #  - the seam never reports a fitted size (no terminal on this page)
    assert _read_fitted_cols(_FakePage(dims=None, ready=False)) is None
    #  - a zero-width .xterm: Simple mode's hidden terminal, whose dims are
    #    xterm's 10x4 fallback rather than a fit
    assert _read_fitted_cols(_FakePage(dims={"cols": 96, "rows": 24}, term_width=0)) is None
    #  - the seam answers null (getTerminalDimensions() before initTerminal())
    assert _read_fitted_cols(_FakePage(dims=None)) is None
    #  - and a payload without a usable column count is a skip, never a guess
    assert _read_fitted_cols(_FakePage(dims={"rows": 24})) is None
    assert _read_fitted_cols(_FakePage(dims={"cols": 0, "rows": 24})) is None


def test_fitted_cols_reader_asks_the_window_seam_not_the_status_bar() -> None:
    """The size comes from ``terminal.js``, not from a status-bar readout.

    ``#term-dims`` is gone: the terminal size is a bar ITEM now, so it can be
    moved, folded into the overflow menu, or removed. Scraping the bar would make
    the wrap guard depend on a layout choice and go quiet the first time an
    operator's saved layout dropped the item.
    """
    page = _FakePage(dims={"cols": 96, "rows": 24})
    assert _read_fitted_cols(page) == 96
    asked = " ".join(page.scripts)
    assert TERMINAL_DIMS_SEAM in asked
    assert "term-dims" not in asked


def test_terminal_js_publishes_the_window_seam() -> None:
    """The JS side of the seam exists, under the exact name this module reads.

    The two halves live in different languages and neither import nor type-check
    the other, so a rename in ``terminal.js`` would otherwise surface only as a
    wrap guard that had quietly stopped guarding.
    """
    terminal_js = (
        Path(__file__).resolve().parents[2]
        / "src/osprey/interfaces/web_terminal/static/js/terminal.js"
    )
    source = terminal_js.read_text(encoding="utf-8")
    assert TERMINAL_DIMS_SEAM.removeprefix("window.") in source
    assert "getTerminalDimensions" in source


def test_capture_plot_artifact_id_is_seeded(tmp_path: Path) -> None:
    """The plot the capture loop selects actually exists in the seeded store."""
    seed_demo_workspace(tmp_path)
    store = ArtifactStore(workspace_root=tmp_path)
    plot = store.get_entry(DEMO_PLOT_ARTIFACT_ID)
    assert plot is not None
    assert plot.artifact_type == "plot_html"


# ---------------------------------------------------------------------------
# Contact-sheet composition (browser-free)
# ---------------------------------------------------------------------------


def test_compose_contact_sheet_is_self_contained(tmp_path: Path) -> None:
    """The composed sheet references only sibling PNGs — no external assets."""
    captured = [
        CapturedVariant("dark", None, None, "web_terminal_dark.png"),
        CapturedVariant("light", "simple", "teal", "web_terminal_light_simple_teal.png"),
    ]
    for cv in captured:
        (tmp_path / cv.filename).write_bytes(b"\x89PNG\r\n")  # placeholder image bytes

    sheet = compose_contact_sheet(tmp_path, captured)
    assert sheet.name == CONTACT_SHEET_NAME
    doc = sheet.read_text(encoding="utf-8")

    # One <img> per variant, pointing at the local sibling PNG.
    for cv in captured:
        assert f'src="{cv.filename}"' in doc
    # Labels surface theme, mode, and accent candidate.
    assert "Light · Simple · teal" in doc
    assert "Dark · Default" in doc
    # Provenance header is present.
    assert "rev " in doc
    # Fully self-contained: no external (remote) asset references.
    assert "http://" not in doc
    assert "https://" not in doc


def test_compose_variant_label_omits_empty_axes() -> None:
    """The label drops mode/accent when unset, includes them when present."""
    assert _variant_label(CapturedVariant("dark", None, None, "x.png")) == "Dark · Default"
    assert (
        _variant_label(CapturedVariant("light", "simple", "blue", "x.png"))
        == "Light · Simple · blue"
    )
    staged = CapturedVariant("dark", "expert", None, "x.png", stage="customize_sheet")
    assert _variant_label(staged).endswith("· Customize bars open")


# ---------------------------------------------------------------------------
# Accent A/B injection (browser-free)
# ---------------------------------------------------------------------------


def test_accent_map_covers_every_token_accent_var() -> None:
    """The self-check passes against the real generated tokens.css."""
    _assert_accent_map_covers_tokens()  # must not raise


def test_accent_map_flags_an_unhandled_token_var(monkeypatch) -> None:
    """A new /accent/ var outside the map and exclusions fails the run."""
    from docs.screenshots import contact_sheet as cs

    monkeypatch.setattr(cs, "_tokens_css_text", lambda: "--color-accent-brandnew: #fff;")
    with pytest.raises(RuntimeError, match="neither overrides nor excludes"):
        cs._assert_accent_map_covers_tokens()


def test_accent_map_flags_a_stale_override_key(monkeypatch) -> None:
    """An override key that no longer exists in tokens.css fails the run."""
    from docs.screenshots import contact_sheet as cs

    # tokens.css with only one of the many overridden vars present.
    monkeypatch.setattr(cs, "_tokens_css_text", lambda: "--color-accent: #fff;")
    with pytest.raises(RuntimeError, match="absent from tokens.css"):
        cs._assert_accent_map_covers_tokens()


def test_accent_cursor_var_is_excluded_not_overridden() -> None:
    """--ansi-cursor-accent is a documented exclusion, never in the override set."""
    assert "--ansi-cursor-accent" in ACCENT_EXCLUSIONS
    assert "--ansi-cursor-accent" not in _accent_override_var_names()


def test_accent_override_css_recomputes_rgba_composites() -> None:
    """Solid vars take the hex; alpha composites become rgba() of that colour."""
    css = _accent_override_css("dark", "teal")  # #4fd1c5 -> (79, 209, 197)
    assert ':root[data-theme="dark"]' in css
    assert "--color-accent: #4fd1c5;" in css
    assert "--border-accent: rgba(79, 209, 197, 0.15);" in css  # dark border alpha
    assert "--accent-tint-30: rgba(79, 209, 197, 0.30);" in css
    assert "--wt-accent-system-tint-04: rgba(79, 209, 197, 0.04);" in css
    # Light candidate uses the theme's heavier border alpha.
    assert "rgba(31, 98, 196, 0.25)" in _accent_override_css("light", "blue")


def test_accent_effective_variants_double_and_are_paired() -> None:
    """--accents doubles the variant count; each base variant's pair is adjacent."""
    base = _effective_variants(False)
    ab = _effective_variants(True)
    assert all(accent is None for _, _, accent in base)
    assert len(ab) == 2 * len(base)
    assert ("dark", "expert", "blue") in ab
    assert ("dark", "expert", "teal") in ab
    # Distinct output filenames per accent.
    names = [_variant_filename(t, m, a) for t, m, a in ab]
    assert len(names) == len(set(names))
    assert "web_terminal_dark_expert_teal.png" in names


# ---------------------------------------------------------------------------
# Hub capture: the plot is drawn before the shot (browser-free)
# ---------------------------------------------------------------------------


def test_plot_frame_selector_is_scoped_to_the_active_view() -> None:
    """Each view's preview frame is named by that view's own container."""
    assert _plot_frame_selector("simple") == "#simple-result-preview iframe"
    assert _plot_frame_selector("expert") == "#preview-content iframe.preview-iframe-light"
    assert _plot_frame_selector(None) == "#preview-content iframe.preview-iframe-light"


def test_drawn_plot_selector_needs_a_drawn_trace() -> None:
    """The selector matches only what Plotly creates, never the served markup."""
    assert ".js-plotly-plot" in _PLOT_DRAWN_SELECTOR
    assert "path.js-line" in _PLOT_DRAWN_SELECTOR
    assert "plotly-graph-div" not in _PLOT_DRAWN_SELECTOR
    served = _beam_current_plot_html()
    assert "plotly-graph-div" in served
    assert "js-plotly-plot" not in served
    assert "js-line" not in served


@pytest.mark.parametrize("mode", ["expert", "simple"])
def test_plot_wait_asks_the_active_preview_for_a_visible_trace(mode: str) -> None:
    """One wait, for a visible drawn trace, inside the active view's preview frame."""
    page = _FakeHubPage(drawn=[True])
    _wait_for_plot_drawn(page, mode, variant=f"theme=light, mode={mode}", moment="before the shot")
    assert page.waits == [
        ((_ARTIFACTS_PANEL, _plot_frame_selector(mode)), _PLOT_DRAWN_SELECTOR, "visible")
    ]


def test_plot_wait_names_the_variant_when_the_plot_never_draws() -> None:
    """A timeout becomes a RuntimeError naming the variant and the moment."""
    page = _FakeHubPage(drawn=[False])
    with pytest.raises(RuntimeError) as info:
        _wait_for_plot_drawn(
            page, "expert", variant="theme=dark, mode=expert", moment="before the shot"
        )
    message = str(info.value)
    assert "theme=dark, mode=expert" in message
    assert "before the shot" in message
    assert "did not draw" in message
    assert isinstance(info.value.__cause__, PlaywrightTimeoutError)


def test_plot_wait_does_not_swallow_other_errors() -> None:
    """A Playwright error other than a timeout propagates unchanged."""
    page = _FakeHubPage(drawn=[True], wait_error=PlaywrightError("frame detached"))
    with pytest.raises(PlaywrightError, match="frame detached") as info:
        _wait_for_plot_drawn(page, "expert", variant="theme=light", moment="before the shot")
    assert not isinstance(info.value, RuntimeError)


def test_hub_capture_writes_no_image_when_the_plot_never_draws(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A plot that never draws fails the capture, writes nothing, and still tears down."""
    page = _FakeHubPage(drawn=[False])
    with pytest.raises(RuntimeError, match="theme=light, mode=expert"):
        _drive(monkeypatch, tmp_path, page)
    assert "shot" not in page.events
    assert not (tmp_path / "shot.png").exists()
    assert "restart" in page.events
    assert "close" in page.events


def test_hub_capture_rechecks_the_plot_after_the_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The plot is checked again after the stage and settle, right before the shot."""
    from docs.screenshots import contact_sheet as cs

    monkeypatch.setitem(cs.STAGES, "probe", Stage("probe", lambda p: p.events.append("stage")))
    page = _FakeHubPage(drawn=[True, False])
    with pytest.raises(RuntimeError, match="stage=probe") as info:
        _drive(monkeypatch, tmp_path, page, stage="probe")
    assert "before the shot" in str(info.value)
    steps = [e for e in page.events if e not in ("restart", "close")]
    assert steps == ["click", "plot:visible", "stage", "settle", "plot:visible"]
    assert "shot" not in page.events


def test_hub_capture_shoots_after_both_plot_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With the plot drawn at both checks, the shot follows the second one."""
    from docs.screenshots import contact_sheet as cs

    monkeypatch.setitem(cs.STAGES, "probe", Stage("probe", lambda p: p.events.append("stage")))
    page = _FakeHubPage(drawn=[True, True])
    _drive(monkeypatch, tmp_path, page, stage="probe")
    assert (tmp_path / "shot.png").read_bytes() == b"png"
    steps = [e for e in page.events if e not in ("restart", "close")]
    assert steps == ["click", "plot:visible", "stage", "settle", "plot:visible", "shot"]
