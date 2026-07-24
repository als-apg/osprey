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
    ACCENT_EXCLUSIONS,
    CONTACT_SHEET_NAME,
    DEMO_PLOT_ARTIFACT_ID,
    DEMO_SESSION_ID,
    DEMO_TRANSCRIPT_PATH,
    MAX_CARD_LINE_WIDTH,
    TRANSCRIPT_SENTINEL,
    VARIANTS,
    CapturedVariant,
    _accent_override_css,
    _accent_override_var_names,
    _assert_accent_map_covers_tokens,
    _assert_fits_columns,
    _effective_variants,
    _fake_session_line,
    _replay_shell_command,
    _variant_filename,
    _variant_label,
    _variant_url,
    _write_fake_session,
    compose_contact_sheet,
    hermetic_hub,
    longest_transcript_line_width,
    seed_demo_workspace,
)

from osprey.cli.project_utils import encode_claude_project_path
from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery
from osprey.stores.artifact_store import ArtifactStore


def _get(url: str) -> tuple[int, bytes]:
    """GET *url*, returning (status, body) — loopback only, short timeout."""
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 (loopback)
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
    """Mode is absent from both filename and URL until one is set."""
    assert _variant_filename("dark", None) == "web_terminal_dark.png"
    assert _variant_filename("light", "simple") == "web_terminal_light_simple.png"

    assert _variant_url("http://h", "dark", None) == "http://h/?theme=dark"
    assert _variant_url("http://h", "light", "simple") == "http://h/?theme=light&mode=simple"


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
