"""Hermetic demo-workspace seeding for the web-terminal contact sheet.

The contact-sheet renderer boots the *real* web terminal in every theme/mode
variant so the redesign can be reviewed in one artifact. To make every variant
look like the design mockups without a live agent (no provider, no hardware, no
network), it points the workspace panel at a pre-seeded workspace: a directory
of demo artifacts plus a canned agent transcript replayed into the terminal's
PTY.

This module owns that seed. :func:`seed_demo_workspace` writes an
``artifacts/artifacts.json`` index and its content files into a caller-provided
directory, shaped exactly like a store the product wrote itself, so the workspace
panel's :class:`~osprey.stores.artifact_store.ArtifactStore` loads it through its
own tolerant loader — no live save path, no artifact-server auto-launch. Every
entry shares one fixed :data:`DEMO_SESSION_ID` so the session-scoped panels stay
populated for a single fake session. :data:`DEMO_TRANSCRIPT_PATH` holds the
canned ANSI exchange; :func:`longest_transcript_line_width` guards it against the
narrow (~370px) terminal card.

The seed reads like accelerator control-room work — a storage-ring beam-current
trace and an orbit-response summary — not a generic demo.

:func:`hermetic_hub` composes the seed into the full capture stack: a real
artifacts backend serving the seeded store, and the real web-terminal hub wired
to it with a canned PTY (no live agent, no provider, no hardware).
:func:`capture_contact_sheet` drives a headless browser over that stack once per
entry in :data:`VARIANTS`, then boots every supported subpanel's real app
(:data:`PANEL_SURFACES` — ARIEL, channels, lattice, knowledge, events, and the
three Bluesky scan panels) and captures each in the full dark/light ×
expert/simple matrix — always through the hub's own iframe URL shape
(``embedded=true`` + theme + mode), so each card is the panel exactly as the
terminal embeds it — writing one viewport PNG per card.
:func:`compose_contact_sheet` folds them into a single self-contained
``contact-sheet.html`` — a tab strip with one tab per surface (hub first), each
tab holding that surface's 2×2 matrix — the one-click artifact the user
reviews. ``python -m docs.screenshots.contact_sheet --out DIR``
runs it end to end and prints the sheet path; ``--accents`` additionally renders
each hub variant under both accent candidates (blue vs teal) so the pending
accent decision can be made from real output. This
module is import-cheap (plotly/numpy, the interface apps, and Playwright are
imported only where used).
"""

from __future__ import annotations

import html
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from unittest import mock

if TYPE_CHECKING:
    from collections.abc import Iterator

    from osprey.stores.artifact_store import ArtifactEntry

# ---------------------------------------------------------------------------
# Fixed identifiers — the seed is fully deterministic so the contact sheet is
# byte-stable across runs and the fake session can be matched by later tasks.
# ---------------------------------------------------------------------------

#: One session id shared by every seeded artifact. The hermetic launcher writes
#: a fake session record with this exact UUID so the Expert session hex renders
#: and every session-scoped panel resolves to the same populated session.
DEMO_SESSION_ID = "3f9a1c72-5e84-4b21-9d6a-8c0f2e1b7a44"

#: Stable artifact id of the seeded beam-current plot. The capture loop selects
#: this artifact so every variant's preview pane shows the real Plotly figure.
DEMO_PLOT_ARTIFACT_ID = "0a1b2c3d4e5f"

#: Absolute path to the committed canned transcript replayed into the PTY.
DEMO_TRANSCRIPT_PATH = Path(__file__).parent / "demo_transcript.txt"

#: Distinctive final line the transcript prints once playback is complete. The
#: capture step waits for this sentinel in the terminal before screenshotting,
#: so it must not collide with any real agent/tool output.
TRANSCRIPT_SENTINEL = "__OSPREY_SHEET_READY__"

#: Visible-column budget for a transcript line: the terminal card is ~370px, so
#: lines wider than this wrap and break the mockup-faithful layout.
MAX_CARD_LINE_WIDTH = 40

# Frozen index timestamp (kept near the docs capture anchor for tidiness).
_DEMO_UPDATED_ISO = "2024-03-18T11:59:00+00:00"

# Matches an ANSI SGR sequence in either its raw ``ESC[...m`` form or the
# printable ``\033[...m`` / ``\x1b[...m`` / ``\e[...m`` escapes the committed
# transcript stores (a player expands them at replay time). Used to compute a
# line's *visible* width — the number of columns it occupies in the terminal.
_ANSI_SGR = re.compile(r"(?:\x1b|\\033|\\x1b|\\e)\[[0-9;]*m")


def _visible_width(line: str) -> int:
    """Return the on-screen column count of one line (ANSI SGR codes removed)."""
    return len(_ANSI_SGR.sub("", line.rstrip("\n")))


def longest_transcript_line_width(path: Path | None = None) -> int:
    """Return the widest *visible* line in the canned transcript.

    The renderer asserts this stays within :data:`MAX_CARD_LINE_WIDTH` so the
    transcript never wraps in the narrow terminal card. Width is measured after
    stripping ANSI SGR escapes, since colour codes occupy no columns.
    """
    transcript = (path or DEMO_TRANSCRIPT_PATH).read_text(encoding="utf-8")
    return max((_visible_width(line) for line in transcript.splitlines()), default=0)


# ---------------------------------------------------------------------------
# Demo artifact content builders
# ---------------------------------------------------------------------------


def _beam_current_plot_html() -> str:
    """Render a storage-ring beam-current decay trace as standalone Plotly HTML.

    Deterministic (fixed RNG seed) so the plot is byte-stable. Emitted with
    ``include_plotlyjs=False`` to match the store's ``plot_html`` artifacts — the
    workspace panel supplies plotly.js.
    """
    import numpy as np
    import plotly.graph_objects as go

    minutes = np.linspace(0, 60, 600)
    lifetime_h = 8.4
    # Smooth exponential decay from a fresh fill, plus tiny measurement noise.
    current = 500.2 * np.exp(-(minutes / 60.0) / lifetime_h)
    rng = np.random.default_rng(1729)
    current = current + rng.normal(0.0, 0.02, current.size)

    fig = go.Figure()
    fig.add_scatter(
        x=minutes,
        y=current,
        mode="lines",
        name="SR beam current",
        line={"color": "#1f62c4", "width": 2},
    )
    fig.update_layout(
        title="Storage-ring beam current — last hour",
        xaxis_title="Minutes",
        yaxis_title="Current (mA)",
        template="plotly_white",
        margin={"l": 60, "r": 20, "t": 50, "b": 45},
    )
    # Pin the div id: Plotly otherwise mints a random UUID per render, which
    # would make the seeded plot non-deterministic across runs.
    return fig.to_html(include_plotlyjs=False, full_html=True, div_id="beam_current_last_hour")


def _orbit_summary_table_html() -> str:
    """A small orbit / BPM summary table (``table_html``)."""
    rows = [
        ("SR01C:BPM1", "-0.12", "0.08", "OK"),
        ("SR03C:BPM2", "0.31", "-0.04", "OK"),
        ("SR05C:BPM1", "0.02", "0.19", "OK"),
        ("SR07C:BPM3", "-0.44", "0.11", "WATCH"),
    ]
    body = "\n".join(
        f"    <tr><td>{name}</td><td>{x}</td><td>{y}</td><td>{flag}</td></tr>"
        for name, x, y, flag in rows
    )
    return (
        '<table class="artifact-table" border="0">\n'
        "  <thead>\n"
        "    <tr><th>BPM</th><th>x (mm)</th><th>y (mm)</th><th>Status</th></tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        f"{body}\n"
        "  </tbody>\n"
        "</table>\n"
    )


def _lifetime_note_markdown() -> str:
    """An operator-style analysis note (``markdown``)."""
    return (
        "# Beam current — last hour\n\n"
        "- Current now: **500.2 mA**\n"
        "- Fill decay is smooth; estimated lifetime **~8.4 h**.\n"
        "- Orbit stable: 48 BPMs, RMS orbit **42 µm**.\n"
        "- `SR07C:BPM3` reads slightly high — watch on the next scan.\n"
    )


def _orbit_response_stats_json() -> bytes:
    """A structured data artifact (``json``) with a compact stats summary."""
    data = {
        "measurement": "orbit_response",
        "beam_current_mA": 500.2,
        "lifetime_h": 8.4,
        "n_bpms": 48,
        "n_correctors": 96,
        "rms_orbit_um": 42,
        "worst_bpm": "SR07C:BPM3",
    }
    return json.dumps(data, indent=2).encode("utf-8")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def seed_demo_workspace(workspace_root: Path) -> list[ArtifactEntry]:
    """Seed *workspace_root* with a hermetic demo artifact store.

    Writes ``<workspace_root>/artifacts/`` with one content file per demo
    artifact and an ``artifacts.json`` index in the store's own envelope shape,
    so :class:`~osprey.stores.artifact_store.ArtifactStore` loads it through its
    tolerant loader. Four artifact types are seeded (Plotly plot, HTML table,
    markdown, JSON data); every entry carries :data:`DEMO_SESSION_ID` and its
    ``description`` doubles as the Simple-mode caption. Returns the seeded
    entries in index order.

    This deliberately does *not* go through :meth:`ArtifactStore.save_file`,
    which would trigger the artifact-server auto-launch — the whole point is a
    quiet, side-effect-free directory the renderer can point a fresh store at.
    """
    from osprey.stores.artifact_store import ArtifactEntry

    artifacts_dir = workspace_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # (id, artifact_type, title, description caption, stem, mime, content, extra)
    specs: list[tuple[str, str, str, str, str, str, bytes, dict]] = [
        (
            DEMO_PLOT_ARTIFACT_ID,
            "plot_html",
            "Storage-ring beam current — last hour",
            "Beam current over the last hour; smooth decay, lifetime ~8.4 h.",
            "beam_current_last_hour.html",
            "text/html",
            _beam_current_plot_html().encode("utf-8"),
            {"category": "visualization", "tool_source": "execute"},
        ),
        (
            "1a2b3c4d5e6f",
            "table_html",
            "BPM orbit summary",
            "Orbit readings across 48 BPMs; SR07C:BPM3 flagged to watch.",
            "bpm_orbit_summary.html",
            "text/html",
            _orbit_summary_table_html().encode("utf-8"),
            {"category": "table", "tool_source": "execute"},
        ),
        (
            "2a3b4c5d6e7f",
            "markdown",
            "Beam lifetime note",
            "Operator note: current, decay, lifetime, and the orbit outlier.",
            "beam_lifetime_note.md",
            "text/markdown",
            _lifetime_note_markdown().encode("utf-8"),
            {"tool_source": "execute"},
        ),
        (
            "3a4b5c6d7e8f",
            "json",
            "Orbit response stats",
            "Structured orbit-response stats: 48 BPMs, 96 correctors, 42 µm RMS.",
            "orbit_response_stats.json",
            "application/json",
            _orbit_response_stats_json(),
            {
                "category": "data",
                "tool_source": "orbit_response",
                "summary": {"n_bpms": 48, "n_correctors": 96, "rms_orbit_um": 42},
            },
        ),
    ]

    base = datetime.fromisoformat(_DEMO_UPDATED_ISO)
    entries: list[ArtifactEntry] = []
    for offset, (art_id, art_type, title, desc, stem, mime, content, extra) in enumerate(specs):
        filename = f"{art_id}_{stem}"
        (artifacts_dir / filename).write_bytes(content)
        timestamp = base.replace(second=offset, tzinfo=UTC).isoformat()
        data_file = extra.pop("data_file", filename if art_type == "json" else "")
        entries.append(
            ArtifactEntry(
                id=art_id,
                artifact_type=art_type,
                title=title,
                description=desc,
                filename=filename,
                mime_type=mime,
                size_bytes=len(content),
                timestamp=timestamp,
                session_id=DEMO_SESSION_ID,
                data_file=data_file,
                **extra,
            )
        )

    index = {
        "version": 1,
        "updated": _DEMO_UPDATED_ISO,
        "entry_count": len(entries),
        "entries": [e.to_dict() for e in entries],
    }
    (artifacts_dir / "artifacts.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return entries


def seed_demo_knowledge_bundle(workspace_root: Path) -> Path:
    """Seed ``<workspace_root>/knowledge/`` with a tiny OKF markdown bundle.

    An OKF bundle is just a directory tree of markdown documents, so a
    handful of control-room-flavoured pages is enough for the knowledge
    panel to render a populated tree + document instead of its guarded
    not-configured shell. Returns the bundle directory.
    """
    bundle = workspace_root / "knowledge"
    (bundle / "devices").mkdir(parents=True, exist_ok=True)
    (bundle / "index.md").write_text(
        "# Storage Ring Operations\n\n"
        "Facility knowledge for the demo storage ring: device references and\n"
        "operating notes used by the control-room assistant.\n",
        encoding="utf-8",
    )
    (bundle / "devices" / "bpm.md").write_text(
        "# Beam Position Monitors\n\n"
        "48 BPMs around the ring report the closed orbit. Typical RMS orbit\n"
        "is below 50 µm during user operations.\n",
        encoding="utf-8",
    )
    (bundle / "devices" / "correctors.md").write_text(
        "# Orbit Correctors\n\n"
        "96 corrector magnets (horizontal and vertical) hold the golden orbit.\n"
        "Setpoints end in `:SP`; readbacks end in `:RB`.\n",
        encoding="utf-8",
    )
    return bundle


# ---------------------------------------------------------------------------
# Hermetic capture stack
# ---------------------------------------------------------------------------


class HermeticHub(NamedTuple):
    """Handles the capture step needs from a live, hermetic web-terminal stack.

    ``base_url`` is the running hub; ``artifact_url`` its embedded artifacts
    backend. ``project_dir`` is the seeded, throwaway project the hub treats as
    its cwd; ``session_dir`` is where the hub's :class:`SessionDiscovery` looks
    for session transcripts (``~/.claude/projects/<encoded>/``), so the capture
    step writes its one fake session record there to make the Expert session hex
    render. All four are torn down when :func:`hermetic_hub` exits.
    """

    base_url: str
    artifact_url: str
    project_dir: Path
    session_dir: Path


def _replay_shell_command(transcript_path: Path) -> list[str]:
    """Shell argv that paints the canned transcript into the PTY, then idles.

    The committed transcript stores ANSI SGR codes as printable ``\\033[...m``
    escapes (so the file stays plain ASCII), so it is replayed with ``printf
    %b`` — which expands them to real escape bytes — rather than a bare ``cat``,
    which would print the escapes literally. The trailing ``sleep`` keeps the
    PTY (and its rendered output) alive for the screenshot.
    """
    quoted = shlex.quote(str(transcript_path.resolve()))
    replay = f"printf '%b\\n' \"$(cat {quoted})\"; sleep 3600"
    return ["bash", "-c", replay]


@contextmanager
def hermetic_hub() -> Iterator[HermeticHub]:
    """Boot the full hermetic capture stack; yield a :class:`HermeticHub`.

    Creates a throwaway project directory, seeds it with the demo artifact store
    (:func:`seed_demo_workspace`), then brings up two real servers on free ports:

    1. the artifacts backend — ``artifacts.create_app(workspace_root=<seeded>)``
       — serving the seeded store, and
    2. the web-terminal hub — ``web_terminal.create_app(...)`` — with a canned
       PTY that replays the transcript, its ``project_dir`` pointed at the seeded
       directory so session discovery resolves to a controlled location.

    The hub is wired to the pre-launched artifacts backend with the three patches
    the visual-regression harness uses (``_load_web_config`` → the seeded
    watch_dir, ``_load_panel_config`` → artifacts-only, ``_launch_artifact_server``
    → the already-running backend URL), so no extra server is spawned and no live
    agent is involved. On exit both servers stop and both the project directory
    and its ``~/.claude/projects/<encoded>/`` session directory are removed.
    """
    from osprey.cli.project_utils import encode_claude_project_path
    from osprey.interfaces._serving import run_app_server

    project_dir = Path(tempfile.mkdtemp(prefix="osprey-contact-sheet-")).resolve()
    session_dir = Path.home() / ".claude" / "projects" / encode_claude_project_path(project_dir)
    try:
        seed_demo_workspace(project_dir)
        shell_command = _replay_shell_command(DEMO_TRANSCRIPT_PATH)

        with ExitStack() as stack:
            from osprey.interfaces.artifacts.app import create_app as create_artifacts_app

            artifacts_app = create_artifacts_app(workspace_root=project_dir)
            artifact_url = stack.enter_context(run_app_server(artifacts_app))

            # Wire the hub to the already-running backend without spawning
            # another one (mirrors tests/interfaces/design_system/test_visual.py).
            app_mod = "osprey.interfaces.web_terminal.app"
            stack.enter_context(
                mock.patch(
                    f"{app_mod}._load_web_config", return_value={"watch_dir": str(project_dir)}
                )
            )
            stack.enter_context(
                mock.patch(f"{app_mod}._load_panel_config", return_value=({"artifacts"}, [], None))
            )
            stack.enter_context(
                mock.patch(
                    f"{app_mod}._launch_artifact_server",
                    side_effect=lambda a: setattr(a.state, "artifact_server_url", artifact_url),
                )
            )

            from osprey.interfaces.web_terminal.app import create_app as create_hub_app

            hub_app = create_hub_app(shell_command=shell_command, project_dir=project_dir)
            base_url = stack.enter_context(run_app_server(hub_app))

            yield HermeticHub(
                base_url=base_url,
                artifact_url=artifact_url,
                project_dir=project_dir,
                session_dir=session_dir,
            )
    finally:
        shutil.rmtree(project_dir, ignore_errors=True)
        shutil.rmtree(session_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Per-variant capture
# ---------------------------------------------------------------------------

#: The theme/mode variants captured, in output order: the full dark/light ×
#: expert/simple 2×2, so one sheet shows every shell the redesign ships. Each
#: tuple's mode drives an explicit ``&mode=`` on the capture URL (see
#: :func:`_variant_url`); the pre-paint ladder in mode-boot.js then honours it
#: over localStorage/SSR. :data:`_FULL_MATRIX` mirrors this set as the
#: completeness invariant the run asserts against.
VARIANTS: list[tuple[str, str | None]] = [
    ("dark", "expert"),
    ("dark", "simple"),
    ("light", "expert"),
    ("light", "simple"),
]

#: The four theme×mode cells a complete base contact sheet must contain. Kept as
#: an explicit invariant (not derived from :data:`VARIANTS`) so a regression that
#: drops a cell from ``VARIANTS`` fails the run loudly instead of shipping a
#: silently short sheet.
_FULL_MATRIX: frozenset[tuple[str, str | None]] = frozenset(
    {("dark", "expert"), ("dark", "simple"), ("light", "expert"), ("light", "simple")}
)

#: The same 2×2 as an ordered list, for capture loops that want deterministic
#: card order (dark row first, expert before simple) — used by the per-panel
#: sections, which always capture the full matrix.
_FULL_MATRIX_ORDERED: list[tuple[str, str]] = [
    ("dark", "expert"),
    ("dark", "simple"),
    ("light", "expert"),
    ("light", "simple"),
]


class CapturedVariant(NamedTuple):
    """One captured variant and the labels the contact sheet shows for it."""

    theme: str
    mode: str | None
    accent: str | None
    filename: str
    #: Which surface the card belongs to: ``"hub"`` for the full web-terminal
    #: shell (the original sheet), or a :data:`PANEL_SURFACES` id for one of
    #: the per-subpanel 2×2 sections.
    surface: str = "hub"


_TERMINAL_VIEWPORT = {"width": 1280, "height": 800}

# Generous timeouts: the hub boots real servers and loads xterm/plotly from a
# CDN, so first paint and the canned PTY replay can take a few seconds.
_NAV_TIMEOUT_MS = 30_000
_SETTLE_MS = 600  # let the theme swap + layout settle (mirrors the visual suite)
_PLOTLY_MS = 2_000  # Plotly draws async; give the preview time before shooting


def _variant_filename(theme: str, mode: str | None, accent: str | None = None) -> str:
    """Output PNG name for one variant (mode/accent suffixes only when set)."""
    stem = f"web_terminal_{theme}"
    if mode is not None:
        stem += f"_{mode}"
    if accent is not None:
        stem += f"_{accent}"
    return f"{stem}.png"


def _variant_url(base_url: str, theme: str, mode: str | None) -> str:
    """Hub URL carrying ``?theme=`` and, when set, ``&mode=``."""
    url = f"{base_url}/?theme={theme}"
    if mode is not None:
        url += f"&mode={mode}"
    return url


def _fake_session_line() -> str:
    """One JSONL line for the fake session record the terminal label reads from.

    Session discovery keys off the *filename* (``<session_id>.jsonl``), but a
    single valid ``user`` line also makes the record show up in the session
    picker with a sensible first message, exactly like a real Claude transcript.
    """
    return json.dumps(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": "Plot the storage ring beam current over the last hour.",
            },
            "sessionId": DEMO_SESSION_ID,
            "timestamp": _DEMO_UPDATED_ISO,
        }
    )


def _write_fake_session(session_dir: Path) -> Path:
    """Write ``<session_dir>/<DEMO_SESSION_ID>.jsonl`` and return its path.

    Written *after* the PTY has spawned (so the hub's pre-spawn snapshot does not
    include it) — the hub then discovers it as a new session and pushes the id to
    the terminal label. Filename stem equals :data:`DEMO_SESSION_ID` so the active
    session matches the seeded artifacts' ``session_id``.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"{DEMO_SESSION_ID}.jsonl"
    path.write_text(_fake_session_line() + "\n", encoding="utf-8")
    return path


def _wait_for_session_ready(page, mode: str | None) -> None:
    """Block until the published fake session has reached the terminal chrome.

    Expert renders the session hex into ``#terminal-label`` (``Session 3f9a1c72``),
    so it waits for that exact hex — the strong assertion that session discovery
    ran and pushed the id, unchanged from the theme-only renderer.

    Simple mode's shell density pass (Task 5.4) hides ``#terminal-label`` and
    surfaces a ``Connected`` state in a sibling ``.terminal-label-simple`` span,
    so keying the wait on hex *visibility* would be unreliable. But the JS still
    writes ``Session <hex>`` into the (now hidden) ``#terminal-label`` on
    ``session_info`` in both modes, so Simple keys off that same write via
    ``textContent`` — which is populated regardless of CSS visibility — detected
    as the label moving off its static ``Session`` placeholder. That proves
    session discovery ran without depending on the Simple ``Connected`` chrome,
    and never weakens the Expert hex wait.
    """
    if mode == "simple":
        page.wait_for_function(
            "() => { const l = document.getElementById('terminal-label');"
            " if (!l) return false; const t = (l.textContent || '').trim();"
            " return t !== '' && t !== 'Session'; }",
            timeout=_NAV_TIMEOUT_MS,
        )
    else:
        page.wait_for_function(
            "(h) => { const l = document.getElementById('terminal-label');"
            " return !!l && (l.textContent || '').includes(h); }",
            arg=DEMO_SESSION_ID[:8],
            timeout=_NAV_TIMEOUT_MS,
        )


def _plot_row_selector(mode: str | None) -> str:
    """CSS for the demo plot's list row in the artifacts panel, scoped by mode.

    Expert and Simple both render a ``[data-id]`` row for every artifact, but
    into different (and mutually hidden) containers: Expert's ``#sidebar-body``
    and Simple's ``#simple-list-body``. Both containers exist in the DOM in
    either mode — only one is visible — so the selector must name the *active*
    view's container, or the click would target a hidden row and time out.
    """
    container = "#simple-list-body" if mode == "simple" else "#sidebar-body"
    return f'{container} [data-id="{DEMO_PLOT_ARTIFACT_ID}"]'


def _assert_fits_columns(fitted_cols: int) -> None:
    """Fail the run if the transcript would wrap at ``fitted_cols`` columns.

    The terminal card is narrow (~370px); a transcript line wider than the
    terminal's fitted column count wraps and breaks the mockup-faithful layout,
    so this raises rather than silently producing a wrong-looking screenshot.
    """
    widest = longest_transcript_line_width()
    if widest > fitted_cols:
        raise RuntimeError(
            f"Demo transcript is {widest} columns wide but the terminal card fits "
            f"only {fitted_cols}; it will wrap. Shorten docs/screenshots/demo_transcript.txt."
        )


def _read_fitted_cols(page) -> int | None:
    """Read the terminal's fitted column count from the ``#term-dims`` readout.

    The status bar shows ``<cols>×<rows>`` (updated by app.js); returns the column
    count, or ``None`` when the terminal is not the visible surface so the wrap
    guard is skipped. In Simple mode the terminal card hosts the operator CHAT
    (the xterm container is CSS-hidden), so ``.xterm`` measures zero wide and
    xterm reports its ``10×4`` zero-container fallback rather than a real fitted
    width — reading that as a fit would spuriously fail the guard. Expert mode
    renders the terminal at its real (narrow) width and still enforces the guard,
    so the transcript width stays protected where it is actually shown.
    """
    try:
        page.wait_for_function(
            "() => { const e = document.getElementById('term-dims');"
            " return !!e && /\\d/.test(e.textContent || ''); }",
            timeout=10_000,
        )
    except Exception:
        return None
    # A zero-width .xterm means the terminal isn't laid out as the visible surface
    # (Simple mode's inactive stacked tab); its dims are xterm's fallback, not a
    # real fit — skip the guard rather than measure a collapsed card.
    term_width = page.evaluate(
        "() => { const t = document.querySelector('.xterm');"
        " return t ? Math.round(t.getBoundingClientRect().width) : 0; }"
    )
    if not term_width:
        return None
    text = page.locator("#term-dims").inner_text()
    match = re.match(r"\s*(\d+)", text)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Accent A/B override injection
# ---------------------------------------------------------------------------
#
# The final blue-vs-teal accent decision is deferred to the accent-decision-gate;
# ``--accents`` renders each variant under both candidates so a human can compare
# them side by side. The candidate hexes live ONLY here (docs/ is outside the
# hygiene scan's hardcoded-colour scope) — they are proposals to look at, not
# tokens. Each candidate is applied as a complete per-theme override of every
# ``/accent/``-namespaced custom property, injected into every frame so the hub
# chrome and the panel iframes recolour together.

#: Candidate accents, per captured theme: the solid accent hex plus its
#: on-accent foreground. ``blue`` is the current azure working accent; ``teal``
#: is the OSPREY brand teal drawn from core.json's teal family.
_ACCENT_CANDIDATES: dict[str, dict[str, dict[str, str]]] = {
    "blue": {
        "dark": {"accent": "#6e9fff", "on": "#111217"},
        "light": {"accent": "#1f62c4", "on": "#fafbfc"},
    },
    "teal": {
        "dark": {"accent": "#4fd1c5", "on": "#111217"},
        "light": {"accent": "#0d7377", "on": "#fafbfc"},
    },
}

#: Per-theme alpha for ``--border-accent`` (dark hairlines are fainter).
_THEME_BORDER_ALPHA: dict[str, float] = {"dark": 0.15, "light": 0.25}

#: Alpha steps of the ``--accent-tint-NN`` and ``--wt-accent-system-tint-NN``
#: composite ladders (NN is the alpha in hundredths).
_ACCENT_TINT_SUFFIXES = ("04", "06", "08", "10", "12", "20", "25", "30")
_ACCENT_SYSTEM_TINT_SUFFIXES = ("04",)

#: ``/accent/``-namespaced vars deliberately NOT overridden. ``--ansi-cursor-accent``
#: is a background-derived terminal cursor colour, not part of the accent family.
ACCENT_EXCLUSIONS = frozenset({"--ansi-cursor-accent"})

_ACCENT_VAR_RE = re.compile(r"--[a-z0-9-]*accent[a-z0-9-]*")


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Parse ``#rrggbb`` into an ``(r, g, b)`` int triple."""
    v = value.lstrip("#")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def _accent_override_var_names() -> set[str]:
    """The exact set of custom-property names the override CSS assigns."""
    names = {"--color-accent", "--color-accent-light", "--color-on-accent", "--border-accent"}
    names |= {f"--accent-tint-{s}" for s in _ACCENT_TINT_SUFFIXES}
    names |= {f"--wt-accent-system-tint-{s}" for s in _ACCENT_SYSTEM_TINT_SUFFIXES}
    return names


def _accent_override_css(theme: str, accent: str) -> str:
    """Return a ``[data-theme]``-scoped override remapping every accent var.

    Solid vars take the candidate hex; the alpha composites (border + both tint
    ladders) are recomputed as ``rgba()`` of the candidate colour, preserving the
    original alphas. Scoped with ``:root[data-theme=…]`` so it outranks the base
    theme block regardless of stylesheet order.
    """
    spec = _ACCENT_CANDIDATES[accent][theme]
    r, g, b = _hex_to_rgb(spec["accent"])
    lines = [
        f"  --color-accent: {spec['accent']};",
        f"  --color-accent-light: {spec['accent']};",
        f"  --color-on-accent: {spec['on']};",
        f"  --border-accent: rgba({r}, {g}, {b}, {_THEME_BORDER_ALPHA[theme]});",
    ]
    for suffix in _ACCENT_TINT_SUFFIXES:
        lines.append(f"  --accent-tint-{suffix}: rgba({r}, {g}, {b}, {int(suffix) / 100:.2f});")
    for suffix in _ACCENT_SYSTEM_TINT_SUFFIXES:
        lines.append(
            f"  --wt-accent-system-tint-{suffix}: rgba({r}, {g}, {b}, {int(suffix) / 100:.2f});"
        )
    body = "\n".join(lines)
    return f':root[data-theme="{theme}"] {{\n{body}\n}}'


def _tokens_css_text() -> str:
    """Read the generated design-system ``tokens.css`` (the accent source of truth)."""
    import osprey.interfaces.design_system as design_system

    path = Path(design_system.__file__).parent / "static" / "css" / "tokens.css"
    return path.read_text(encoding="utf-8")


def _assert_accent_map_covers_tokens() -> None:
    """Fail the run unless the override map accounts for every accent var in tokens.css.

    Enumerates the ``/accent/``-namespaced custom properties actually present in
    the generated ``tokens.css`` and asserts each is either overridden by the A/B
    map or explicitly excluded — so a token refactor that adds a new accent var
    can never silently ship a half-recoloured contact sheet. Also guards the
    reverse: an override key that no longer exists in tokens.css.
    """
    present = set(_ACCENT_VAR_RE.findall(_tokens_css_text()))
    overridden = _accent_override_var_names()
    unexpected = present - overridden - set(ACCENT_EXCLUSIONS)
    if unexpected:
        raise RuntimeError(
            "tokens.css has /accent/ var(s) the accent A/B map neither overrides nor "
            f"excludes: {sorted(unexpected)}. Extend _ACCENT_* or ACCENT_EXCLUSIONS."
        )
    missing = overridden - present
    if missing:
        raise RuntimeError(
            f"accent override map targets var(s) absent from tokens.css: {sorted(missing)}."
        )


def _inject_accent(page, theme: str, accent: str) -> None:
    """Inject the candidate accent override into every frame (hub + panels)."""
    css = _accent_override_css(theme, accent)
    for frame in page.frames:
        try:
            frame.add_style_tag(content=css)
        except Exception:
            # about:blank / not-yet-navigated frames reject injection — harmless.
            pass


def _effective_variants(accents: bool) -> list[tuple[str, str | None, str | None]]:
    """Expand :data:`VARIANTS` into (theme, mode, accent) rows.

    Without ``--accents`` each base variant is captured once (accent ``None``);
    with it, each is captured once per candidate in :data:`_ACCENT_CANDIDATES`,
    kept adjacent per base variant so the sheet reads as an A/B pair.
    """
    if not accents:
        return [(theme, mode, None) for theme, mode in VARIANTS]
    return [
        (theme, mode, candidate) for theme, mode in VARIANTS for candidate in _ACCENT_CANDIDATES
    ]


def _assert_all_variants_captured(captured: list[CapturedVariant], accents: bool) -> None:
    """Fail the run unless every expected variant produced a captured card.

    Two guards, so the sheet can never silently ship short:

    * Every ``(theme, mode, accent)`` row :func:`_effective_variants` asked for
      must appear in ``captured`` — catches a variant that was dropped or
      deduplicated between the capture loop and composition.
    * The captured ``(theme, mode)`` cells must cover the full
      :data:`_FULL_MATRIX` 2×2 — catches a ``VARIANTS`` regression that removes a
      theme or mode cell (e.g. the Simple rows) before the loop runs. This runs
      unconditionally: the cell set collapses the accent axis away, so it holds
      in ``--accents`` mode too, where the first guard alone cannot catch the
      regression (both ``expected`` and ``got`` derive from ``VARIANTS`` and
      shrink in lockstep).
    """
    expected = set(_effective_variants(accents))
    got = {(cv.theme, cv.mode, cv.accent) for cv in captured}
    missing = expected - got
    if missing:
        raise RuntimeError(
            f"contact sheet is missing variant(s) {sorted(missing)}: captured "
            f"{len(captured)} of {len(expected)} expected."
        )
    absent = _FULL_MATRIX - {(cv.theme, cv.mode) for cv in captured}
    if absent:
        raise RuntimeError(
            "contact sheet must cover the full dark/light × expert/simple 2×2; "
            f"missing cell(s) {sorted(absent)}. Restore them to VARIANTS."
        )


def _capture_variant(
    browser,
    hub: HermeticHub,
    theme: str,
    mode: str | None,
    out_dir: Path,
    accent: str | None = None,
) -> CapturedVariant:
    """Drive one theme/mode variant to a viewport PNG; return its metadata."""
    session_file = hub.session_dir / f"{DEMO_SESSION_ID}.jsonl"
    # Ensure a clean slate so this variant's pre-spawn snapshot never already
    # contains the fake record (a stale one would never be seen as "new").
    session_file.unlink(missing_ok=True)

    page = browser.new_page(viewport=_TERMINAL_VIEWPORT)
    try:
        page.goto(
            _variant_url(hub.base_url, theme, mode),
            wait_until="domcontentloaded",
            timeout=_NAV_TIMEOUT_MS,
        )
        page.locator("#welcome-dismiss").click(timeout=_NAV_TIMEOUT_MS)

        # Wait for the canned transcript's sentinel to land in the terminal
        # (xterm's DOM renderer keeps the text in .xterm-rows). This also implies
        # the PTY has spawned and the hub's session snapshot has been taken.
        page.wait_for_function(
            "(s) => { const r = document.querySelector('.xterm-rows');"
            " return !!r && (r.textContent || '').includes(s); }",
            arg=TRANSCRIPT_SENTINEL,
            timeout=_NAV_TIMEOUT_MS,
        )

        # Now publish the fake session so the hub discovers it and the terminal
        # header updates; wait for it to reach the label (Expert: the hex;
        # Simple: the Task 5.4 "Connected" state — see _wait_for_session_ready).
        _write_fake_session(hub.session_dir)
        _wait_for_session_ready(page, mode)

        # Guard against a transcript that wraps at the real fitted width.
        fitted_cols = _read_fitted_cols(page)
        if fitted_cols is not None:
            _assert_fits_columns(fitted_cols)

        # Reveal the beam-current plot so every variant shows a real Plotly
        # figure: Expert clicks it into the preview pane, Simple promotes it to
        # the latest-result card. The workspace panel is a same-origin iframe and
        # each row carries ``data-id``; select the plot by its stable id, scoped
        # to the active view so a hidden twin row can't swallow the click.
        panel = page.frame_locator('iframe.panel-iframe[data-panel-id="artifacts"]')
        panel.locator(_plot_row_selector(mode)).first.click(timeout=_NAV_TIMEOUT_MS)

        # Anchor on the Plotly root actually appearing in the nested preview
        # iframe, with the 2s as a cap — best-effort, since a not-yet-drawn plot
        # should not fail the whole run (the settle below still gives it time).
        # Expert's preview iframe is class-tagged; Simple renders the plot into
        # the latest-result card's #simple-result-preview.
        try:
            if mode == "simple":
                preview = panel.frame_locator("#simple-result-preview iframe")
            else:
                preview = panel.frame_locator("iframe.preview-iframe-light")
            preview.locator(".plotly-graph-div").first.wait_for(
                state="attached", timeout=_PLOTLY_MS
            )
        except Exception:
            pass

        # Recolour the chrome for this accent candidate (all frames now exist).
        if accent is not None:
            _inject_accent(page, theme, accent)

        page.wait_for_timeout(_SETTLE_MS)

        png = page.screenshot()
        dest = out_dir / _variant_filename(theme, mode, accent)
        dest.write_bytes(png)
        return CapturedVariant(theme=theme, mode=mode, accent=accent, filename=dest.name)
    finally:
        page.close()


# ---------------------------------------------------------------------------
# Per-subpanel 2×2 sections
# ---------------------------------------------------------------------------
#
# Every supported subpanel is captured standalone in the full dark/light ×
# expert/simple matrix, one section per panel, using the same hermetic
# app-boot approach as the visual-regression suite: each panel is a real
# FastAPI app served by ``run_app_server``; with no live backend behind it,
# each renders its genuine (stable) shell state. The hub section above stays
# the seeded, transcript-driven showcase.


class PanelSurface(NamedTuple):
    """One standalone subpanel captured as its own 2×2 section."""

    id: str
    title: str
    path: str  # request path, may carry a fixed query (e.g. embedded=true)
    boot: str  # key into _PANEL_BOOTS
    wait_selector: str | None = None


#: Boot one hermetic app per surface family. Each callable takes the seeded
#: workspace and returns a context manager yielding the server's base URL.
def _boot_ariel(workspace: Path):
    from osprey.interfaces._serving import run_app_server
    from osprey.interfaces.ariel.app import create_app

    # No config passed: the app degrades to its DB-less mode (UI renders,
    # search disabled) — deterministic without a live Postgres.
    return run_app_server(create_app())


def _boot_channel_finder(workspace: Path):
    from osprey.interfaces._serving import run_app_server
    from osprey.interfaces.channel_finder.app import create_app

    return run_app_server(create_app(project_cwd=str(workspace)))


def _boot_lattice(workspace: Path):
    from osprey.interfaces._serving import run_app_server
    from osprey.interfaces.lattice_dashboard.app import create_app

    return run_app_server(create_app(workspace_root=workspace / "lattice_ws"))


def _boot_okf(workspace: Path):
    from osprey.interfaces._serving import run_app_server
    from osprey.interfaces.okf_panel.app import create_app

    return run_app_server(create_app(str(workspace / "knowledge")))


def _boot_dispatch_dashboard(workspace: Path):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    import osprey.interfaces.design_system as design_system_pkg
    from osprey.dispatch.dashboard import render_dashboard_html
    from osprey.interfaces._serving import run_app_server

    # The dashboard is rendered HTML mounted into the event-dispatcher server
    # in production; stand up the same minimal wrapper the visual suite uses.
    app = FastAPI()
    static_dir = Path(design_system_pkg.__file__).parent / "static"
    app.mount("/design-system", StaticFiles(directory=static_dir), name="design-system")

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return str(render_dashboard_html(facility_name="Demo Facility", pv_strip_prefix="SR:"))

    return run_app_server(app)


def _boot_bluesky_panels(workspace: Path):
    from osprey.interfaces._serving import run_app_server
    from osprey.services.bluesky_panels.app import app as bluesky_panels_app

    return run_app_server(bluesky_panels_app)


_PANEL_BOOTS = {
    "ariel": _boot_ariel,
    "channel_finder": _boot_channel_finder,
    "lattice": _boot_lattice,
    "okf": _boot_okf,
    "dispatch": _boot_dispatch_dashboard,
    "bluesky": _boot_bluesky_panels,
}

#: The supported subpanels, in sheet order. Every surface is captured with
#: ``embedded=true`` (see :func:`_capture_panel_variant`) — the exact URL shape
#: the hub's panel-manager gives its iframes — so each card shows the panel as
#: the terminal actually presents it, not its standalone chrome.
PANEL_SURFACES: list[PanelSurface] = [
    PanelSurface("ariel", "ARIEL — logbook search", "/", "ariel"),
    PanelSurface("channels", "Channels — channel finder", "/", "channel_finder"),
    PanelSurface("lattice", "Lattice — machine dashboard", "/", "lattice"),
    PanelSurface("knowledge", "Knowledge — OKF browser", "/", "okf", wait_selector="#tree"),
    PanelSurface("events", "Events — dispatch dashboard", "/", "dispatch"),
    PanelSurface(
        "plan", "Plan — Bluesky scan plans", "/plan/", "bluesky", wait_selector="#plan-tree"
    ),
    PanelSurface(
        "results",
        "Results — Bluesky run viewer",
        "/results/",
        "bluesky",
        wait_selector="#run-picker",
    ),
]


def _capture_panel_variant(
    browser,
    base_url: str,
    surface: PanelSurface,
    theme: str,
    mode: str,
    out_dir: Path,
) -> CapturedVariant:
    """Capture one subpanel surface in one theme/mode cell.

    The query string mirrors panel-manager.js's iframe URL construction
    (``embedded=true`` + ``theme`` + ``mode``), so every card shows the panel
    exactly as the hub embeds it — same app, same chrome-stripped variant.
    """
    separator = "&" if "?" in surface.path else "?"
    url = f"{base_url}{surface.path}{separator}embedded=true&theme={theme}&mode={mode}"

    page = browser.new_page(viewport=_TERMINAL_VIEWPORT)
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS)
        if surface.wait_selector:
            page.wait_for_selector(surface.wait_selector, state="attached", timeout=_NAV_TIMEOUT_MS)
        page.wait_for_timeout(_SETTLE_MS)

        applied_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
        applied_mode = page.evaluate("document.documentElement.getAttribute('data-ui-mode')")
        if applied_theme != theme or applied_mode != mode:
            raise RuntimeError(
                f"panel {surface.id}: requested {theme}/{mode} but page applied "
                f"{applied_theme}/{applied_mode}"
            )

        png = page.screenshot()
        dest = out_dir / f"panel_{surface.id}_{theme}_{mode}.png"
        dest.write_bytes(png)
        return CapturedVariant(
            theme=theme, mode=mode, accent=None, filename=dest.name, surface=surface.id
        )
    finally:
        page.close()


def capture_panel_sections(out_dir: Path, browser) -> list[CapturedVariant]:
    """Capture every :data:`PANEL_SURFACES` entry in the full 2×2; return cards.

    Boots one hermetic app per surface (surfaces sharing a boot key still get
    their own short-lived server — boots are cheap and isolation keeps a
    failure local), seeds the throwaway workspace the boots need, and asserts
    afterwards that every surface produced all four theme×mode cells.
    """
    captured: list[CapturedVariant] = []
    workspace = Path(tempfile.mkdtemp(prefix="osprey-panel-sheet-")).resolve()
    try:
        seed_demo_knowledge_bundle(workspace)
        for surface in PANEL_SURFACES:
            with _PANEL_BOOTS[surface.boot](workspace) as base_url:
                for theme, mode in _FULL_MATRIX_ORDERED:
                    captured.append(
                        _capture_panel_variant(browser, base_url, surface, theme, mode, out_dir)
                    )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    for surface in PANEL_SURFACES:
        cells = {(cv.theme, cv.mode) for cv in captured if cv.surface == surface.id}
        missing = set(_FULL_MATRIX_ORDERED) - cells
        if missing:
            raise RuntimeError(
                f"panel section '{surface.id}' is missing cell(s) {sorted(missing)}."
            )
    return captured


def capture_contact_sheet(out_dir: Path, *, accents: bool = False) -> Path:
    """Capture every variant and compose the review sheet.

    Boots the hermetic stack once, shares a single headless browser across all
    variants, writes one viewport PNG per variant into ``out_dir``, then composes
    them into ``contact-sheet.html``. With ``accents`` each base variant is
    captured under both accent candidates (see :func:`_effective_variants`). The
    transcript width is pre-checked against the design budget before any browser
    launches, and re-checked against the real fitted column count per variant;
    ``accents`` additionally asserts the override map covers every accent var in
    the generated tokens.css. Returns the path to the composed sheet.
    """
    # Cheap pre-flight against the design budget, before booting anything.
    _assert_fits_columns(MAX_CARD_LINE_WIDTH)
    if accents:
        _assert_accent_map_covers_tokens()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from docs.screenshots.capture import chromium_context

    captured: list[CapturedVariant] = []
    with chromium_context() as browser:
        with hermetic_hub() as hub:
            for theme, mode, accent in _effective_variants(accents):
                captured.append(_capture_variant(browser, hub, theme, mode, out_dir, accent))
        _assert_all_variants_captured(captured, accents)
        # Per-subpanel 2×2 sections (accent A/B stays a hub-only concern).
        captured.extend(capture_panel_sections(out_dir, browser))
    return compose_contact_sheet(out_dir, captured)


# ---------------------------------------------------------------------------
# Contact-sheet composition
# ---------------------------------------------------------------------------

CONTACT_SHEET_NAME = "contact-sheet.html"


def _git_rev() -> str:
    """Short git revision of this checkout, or ``"unknown"`` if undeterminable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _variant_label(cv: CapturedVariant) -> str:
    """Human variant name, e.g. ``"Dark · Simple · teal"`` (accent/mode optional)."""
    parts = [cv.theme.capitalize(), (cv.mode or "default").capitalize()]
    if cv.accent:
        parts.append(cv.accent)
    return " · ".join(parts)


def _tab_label(surface_id: str, title: str) -> str:
    """Short tab-strip label for a surface: the part before the ``" — "`` dash."""
    if surface_id == "hub":
        return "Hub"
    return title.split(" — ", 1)[0]


def compose_contact_sheet(out_dir: Path, captured: list[CapturedVariant]) -> Path:
    """Write a single self-contained ``contact-sheet.html``; return its path.

    One TAB per surface (hub first, then each subpanel in :data:`PANEL_SURFACES`
    order), each tab holding that surface's fixed dark/light × expert/simple 2×2
    card grid — click through the tab strip instead of scrolling a long page.
    The tabs are pure CSS (hidden radio inputs + labels), so the page stays fully
    self-contained (inline CSS, no JS, no external assets, sibling PNGs only) and
    theme-aware so it reads in light or dark. A header stamps the capture time
    and git revision for provenance.
    """
    out_dir = Path(out_dir)
    generated_utc = datetime.now(UTC).isoformat(timespec="seconds")
    git_rev = _git_rev()

    def _card(cv: CapturedVariant) -> str:
        return f"""      <figure class="card">
        <figcaption class="label">{html.escape(_variant_label(cv))}</figcaption>
        <a href="{html.escape(cv.filename)}" target="_blank" rel="noopener">
          <img src="{html.escape(cv.filename)}" alt="{html.escape(_variant_label(cv))}"
               loading="lazy" width="1280" height="800">
        </a>
      </figure>"""

    # Group cards per surface, in first-seen order (hub first, then each
    # subpanel in PANEL_SURFACES order — the capture loops guarantee this).
    section_titles = {"hub": "Web terminal — hub shell (seeded workspace)"}
    section_titles.update({s.id: s.title for s in PANEL_SURFACES})
    seen_order: list[str] = []
    by_surface: dict[str, list[CapturedVariant]] = {}
    for cv in captured:
        if cv.surface not in by_surface:
            seen_order.append(cv.surface)
            by_surface[cv.surface] = []
        by_surface[cv.surface].append(cv)

    radios: list[str] = []
    tabs: list[str] = []
    sections: list[str] = []
    tab_css: list[str] = []
    for index, surface_id in enumerate(seen_order):
        title = section_titles.get(surface_id, surface_id)
        checked = " checked" if index == 0 else ""
        radios.append(
            f'  <input class="tab-radio" type="radio" name="surface" '
            f'id="tab-{html.escape(surface_id)}"{checked}>'
        )
        tabs.append(
            f'    <label for="tab-{html.escape(surface_id)}">'
            f"{html.escape(_tab_label(surface_id, title))}</label>"
        )
        cards = "\n".join(_card(cv) for cv in by_surface[surface_id])
        sections.append(
            f"""  <section class="surface" id="sec-{html.escape(surface_id)}">
    <h2>{html.escape(title)}</h2>
    <div class="grid">
{cards}
    </div>
  </section>"""
        )
        # Per-surface tab wiring: show the checked surface, highlight its label.
        tab_css.append(
            f"#tab-{surface_id}:checked ~ #sec-{surface_id} {{ display: block; }}\n"
            f'#tab-{surface_id}:checked ~ .tabs label[for="tab-{surface_id}"] {{\n'
            f"  background: #1f62c4; border-color: #1f62c4; color: #fafbfc;\n"
            f"}}"
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OSPREY Web Terminal — contact sheet</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px;
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    background: #f4f5f5; color: #1a1c1f;
  }}
  header {{ margin-bottom: 16px; }}
  h1 {{ font-size: 18px; margin: 0 0 4px; }}
  .prov {{ font-size: 12px; opacity: 0.7; }}
  .tab-radio {{ position: absolute; opacity: 0; pointer-events: none; }}
  .tabs {{
    position: sticky; top: 0; z-index: 2;
    display: flex; flex-wrap: wrap; gap: 6px;
    padding: 8px 0 12px; background: #f4f5f5;
  }}
  .tabs label {{
    padding: 5px 14px; border-radius: 999px; cursor: pointer;
    font-size: 13px; font-weight: 600;
    background: #fafbfc; border: 1px solid #d8d9da;
  }}
  .tabs label:hover {{ border-color: #1f62c4; }}
  .surface {{ display: none; }}
  h2 {{
    font-size: 15px; margin: 4px 0 12px;
    padding-bottom: 6px; border-bottom: 1px solid #d8d9da;
  }}
  .grid {{
    display: grid; gap: 20px;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}
  @media (max-width: 1100px) {{
    .grid {{ grid-template-columns: minmax(0, 1fr); }}
  }}
  .card {{
    margin: 0; padding: 12px; border-radius: 6px;
    background: #fafbfc; border: 1px solid #d8d9da;
  }}
  .label {{ font-weight: 600; margin-bottom: 8px; }}
  .card img {{
    display: block; width: 100%; height: auto;
    border-radius: 4px; border: 1px solid #d8d9da;
  }}
  {chr(10).join(tab_css)}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #111217; color: #e6e7e8; }}
    .tabs {{ background: #111217; }}
    .tabs label {{ background: #181b1f; border-color: #2c3235; }}
    h2 {{ border-color: #2c3235; }}
    .card {{ background: #181b1f; border-color: #2c3235; }}
    .card img {{ border-color: #2c3235; }}
  }}
</style>
</head>
<body>
  <header>
    <h1>OSPREY Web Terminal — contact sheet</h1>
    <div class="prov">{len(captured)} variant(s) · captured {html.escape(generated_utc)} · rev {html.escape(git_rev)}</div>
  </header>
{chr(10).join(radios)}
  <nav class="tabs">
{chr(10).join(tabs)}
  </nav>
{chr(10).join(sections)}
</body>
</html>
"""
    path = out_dir / CONTACT_SHEET_NAME
    path.write_text(doc, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m docs.screenshots.contact_sheet --out DIR``."""
    import argparse

    from docs.screenshots.capture import ScreenshotSkip

    parser = argparse.ArgumentParser(
        prog="python -m docs.screenshots.contact_sheet",
        description="Capture the web-terminal redesign in every theme/mode variant.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Directory to write the per-variant PNGs and contact-sheet.html into.",
    )
    parser.add_argument(
        "--accents",
        action="store_true",
        help="Render each variant under both accent candidates (blue vs teal) for A/B review.",
    )
    args = parser.parse_args(argv)

    # Booting the real hub emits chatty INFO (config loads, watcher, etc.); keep
    # the CLI's own output readable.
    import logging

    logging.getLogger("osprey").setLevel(logging.WARNING)

    try:
        sheet = capture_contact_sheet(args.out, accents=args.accents)
    except ScreenshotSkip as exc:
        # Absent browser / runtime — a clean one-line notice, not a traceback.
        print(f"contact sheet skipped: {exc}", file=sys.stderr)
        return 1

    print(f"Contact sheet: {sheet.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
