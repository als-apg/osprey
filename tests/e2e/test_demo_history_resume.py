"""A seeded ``demo-history`` session resumes in the pinned CLI and keeps working.

The ``demo-history`` scenario lays down conversations the deployment never had:
transcripts authored by :mod:`osprey.agent_runner.transcript_author`, written by
the hub's lay-down into the directory Claude Code keeps its sessions in. The
model-free probe (``scripts/transcript_resume_probe.py``) proves the pinned
build reads such a transcript; this test proves a real turn runs on one.

Every e2e init withholds ``demo-history`` from the preset's default set, so the
test opts in explicitly: it activates ``[rf-thermal, demo-history]`` on the
built project, writes the seed file for the host identity through
:func:`~osprey.simulation.apply.write_scenario_sessions` at the anchor the
activation persisted, and lays it down with the hub's own
:func:`~osprey.interfaces.web_terminal.scenario_sessions.lay_down_scenario_sessions`.
It then resumes session 1 (``sector-7-beam-loss``, the 14:32 beam loss
yesterday) with the pinned CLI:

1. asked for the beam current now, the turn calls the controls read tool and
   answers with a value;
2. asked to replot the 14:32 window from the archive, it calls the archiver for
   the DCCT current and finds the ~18 mA drop.

A PR that bumps the pinned CLI carries ``full-ci`` so this test runs on it (see
``scripts/README.md``).

Run with:
    uv run pytest tests/e2e/test_demo_history_resume.py -v
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
import yaml

from tests.e2e.sdk_helpers import (
    HAS_SDK,
    agent_data_dir,
    init_project,
    render_dir,
    run_sdk_query,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.harness_benchmark,
    pytest.mark.requires_als_apg,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
]

#: The preset's default set with ``demo-history`` opted back in.
SCENARIOS = ["rf-thermal", "demo-history"]

#: Session 1 of ``demo-history``: the sector 7 beam loss at 14:32 yesterday.
SESSION_1 = "sector-7-beam-loss"

DCCT = "SR:DIAG:DCCT:01:CURRENT:RB"
CHANNEL_READ = "mcp__controls__channel_read"
ARCHIVER_READ = "mcp__controls__archiver_read"

CURRENT_PROMPT = "What is the beam current right now?"
REPLOT_PROMPT = (
    "Pull the stored current from the archiver again for the 14:32 window yesterday "
    "and replot it. How big was the drop?"
)

#: A current reading in the answer: a number with ``mA`` after it.
_CURRENT_VALUE = re.compile(r"\d+(?:\.\d+)?\s*mA\b")

#: The drop, as the archive holds it: about 18 mA, from 500 to about 482 mA.
_DROP = re.compile(r"\b(?:1[5-9]|2[0-1])(?:\.\d+)?\s*mA\b|\b4[78]\d(?:\.\d+)?\s*mA\b")


def _lay_down_session_1(repo: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Activate the set, seed the host identity, lay it down; return session 1's key."""
    from osprey.agent_record.identities import host_identity
    from osprey.interfaces.web_terminal.scenario_sessions import lay_down_scenario_sessions
    from osprey.simulation.apply import (
        SCENARIO_SESSIONS_FILENAME,
        apply_scenarios,
        scenario_sessions_dir,
        write_scenario_sessions,
    )
    from osprey_connectors import posture_store
    from osprey_connectors.simulation.state import read_active_state
    from osprey_connectors.workspace import resolve_simulation_state_dir

    render = render_dir(repo)
    config_path = render / "config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    host = host_identity(config)
    assert host, "the built config records no deployment.host_identity"

    # The logbook is not part of what this test asks, so no ARIEL is needed.
    result = apply_scenarios(repo, SCENARIOS, seed_logbook=False)
    assert "demo-history" in result.active, result.active
    _, epoch = read_active_state(resolve_simulation_state_dir(config, repo))
    assert epoch is not None, "the activation persisted no anchor"
    tz = ZoneInfo(((config.get("system") or {}).get("timezone")) or "UTC")
    anchor = datetime.fromtimestamp(epoch, UTC).astimezone(tz)

    written = write_scenario_sessions(config, repo, result.active, anchor)
    seed = scenario_sessions_dir(config, repo, host) / SCENARIO_SESSIONS_FILENAME
    assert seed in written, f"no seed file for the host identity {host}: {written}"
    document = json.loads(seed.read_text(encoding="utf-8"))
    session = next(s for s in document["sessions"] if s["slug"] == SESSION_1)
    key = session["key"]

    # The hub resolves its seed and stores from the acting identity's state
    # dir; bind it to the host's for the lay-down only, so the agent's own
    # processes see the environment a test run normally gives them.
    app = SimpleNamespace(
        state=SimpleNamespace(
            project_cwd=str(render),
            panel_status_root=agent_data_dir(repo),
            config_path=config_path,
        )
    )
    with monkeypatch.context() as m:
        m.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(seed.parent))
        posture_store.invalidate_cache()
        report = lay_down_scenario_sessions(app)
    posture_store.invalidate_cache()
    assert key in report.laid and not report.incomplete, report
    return key


@pytest.mark.flaky(reruns=2)  # live model turns; absorb a rare stochastic miss
@pytest.mark.asyncio
async def test_demo_history_resume_runs_a_turn_on_a_seeded_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume the seeded beam-loss session, read the current, replot the loss."""
    from osprey.agent_runner.project_paths import claude_project_dir

    repo = init_project(
        tmp_path,
        "demo_history_resume",
        template="control_assistant",
        provider="als-apg",
        model="claude-haiku-5-5",
    )
    sessions_dir = claude_project_dir(render_dir(repo))
    try:
        key = _lay_down_session_1(repo, monkeypatch)
        assert (sessions_dir / f"{key}.jsonl").is_file(), (
            f"the lay-down wrote no transcript for {key} under {sessions_dir}"
        )

        current = await run_sdk_query(
            repo, CURRENT_PROMPT, max_turns=10, max_budget_usd=1.0, resume=key
        )
        text = (current.result.result if current.result is not None else None) or ""
        assert current.result is not None, "no ResultMessage for the resumed turn"
        assert not current.result.is_error, f"the resumed turn ended in error: {text}"
        reads = [t for t in current.tool_traces if t.name == CHANNEL_READ]
        assert reads, f"the resumed turn never called {CHANNEL_READ}. Tools: {current.tool_names}"
        assert any(DCCT in str(t.input) for t in reads), (
            f"no {CHANNEL_READ} call read {DCCT}: {[t.input for t in reads]}"
        )
        assert _CURRENT_VALUE.search(text), f"the answer states no current value: {text!r}"

        session_id = current.result.session_id or key
        replot = await run_sdk_query(
            repo, REPLOT_PROMPT, max_turns=15, max_budget_usd=1.5, resume=session_id
        )
        replot_text = (replot.result.result if replot.result is not None else None) or ""
        assert replot.result is not None, "no ResultMessage for the replot turn"
        assert not replot.result.is_error, f"the replot turn ended in error: {replot_text}"
        archive = [
            t
            for t in replot.tool_traces
            if t.name == ARCHIVER_READ and DCCT in str(t.input) and not t.is_error
        ]
        assert archive, (
            f"the replot turn never read {DCCT} from the archiver. "
            f"Tools: {replot.tool_names}; archiver inputs: "
            f"{[t.input for t in replot.tool_traces if t.name == ARCHIVER_READ]}"
        )
        assert _DROP.search(replot_text), f"the replot answer names no ~18 mA drop: {replot_text!r}"
    finally:
        # The run uses the developer's Claude Code state; take this render's
        # session directory (keyed by its tmp path) back out of it.
        shutil.rmtree(sessions_dir, ignore_errors=True)
