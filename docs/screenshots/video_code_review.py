"""The code review demo: its storyboard and the probes only it needs.

The video opens on the Channels page, and the operator looks up the reviewer's instructions
through ⌘K: they open under Settings → Safety, and the deployment's one rule is that a script
moving a corrector first reads the beam mode. The operator asks for a corrector scan, which the
agent writes without the check, since the rule is the reviewer's and not the agent's. The review
blocks it before anyone is asked, the agent adds the check, and the corrected script reaches the
operator with the verdict on its card.

The Settings drawer's first-time warning is acknowledged before recording
(:func:`acknowledge_settings`), so the video never shows it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from docs.screenshots import video_chat
from docs.screenshots.video_beats import (
    Beat,
    click,
    focus,
    hold,
    prompt,
    shown,
    spotlight,
    wait_for,
    wait_until,
)
from docs.screenshots.video_probes import POLL_MS, ProbeFailed
from docs.screenshots.video_take import TYPE_DELAY_MS

if TYPE_CHECKING:
    from docs.screenshots.video_take import Take, TakeContext

CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"
BEAM_MODE = "SR:BEAM:USER_MODE"
PROMPT_SCAN = f"Write a Python script that steps {CORRECTOR} from 0 to 1 in 10 steps, and run it."

_PRESET = (
    Path(__file__).parents[2] / "src" / "osprey" / "profiles" / "presets" / "control-assistant.yml"
)
_PROFILE = yaml.safe_load(_PRESET.read_text(encoding="utf-8"))
HOOKS = [*_PROFILE["hooks"], "code-review"]
# No Bluesky on this stack: with it, a scan is a plan, which the reviewer does not read.
SKILLS = [skill for skill in _PROFILE["skills"] if "bluesky" not in skill]
INIT_SETS = (
    "config.code_review.enabled=true",
    "config.code_review.level=block",
    "config.claude_code.servers.bluesky.enabled=false",
    "bluesky=null",
    "bluesky_web=null",
    f"hooks={json.dumps(HOOKS, separators=(',', ':'))}",
    f"skills={json.dumps(SKILLS, separators=(',', ':'))}",
)

# The way to the reviewer's rules: ⌘K, a search, and the palette's row for them, which opens
# Settings → Safety on the rules' detail view.
PALETTE_INPUT = ".command-palette-input"
PALETTE_ROW = ".command-palette-item"
PALETTE_ACTIVE = ".command-palette-item--active"
SEARCH = "code review"
ROW_TEXT = "Code review rules"
RULE = "#tab-safety .prompts-detail-view .osprey-md-rendered li"
CLOSE_DRAWER = "#settings-drawer .drawer-close-btn"
DRAWER_BUDGET_MS = 15_000
PALETTE_BUDGET_MS = 10_000
# The Settings drawer's first-time warning, which the take acknowledges off camera.
WARNING = ".settings-warning-dialog"

# The palette's chord, as the page itself decides it (palette-boot.js): ⌘K on macOS, Ctrl+K
# elsewhere.
CHORD_JS = """
() => {
  const platform = navigator.userAgentData?.platform || navigator.platform || '';
  return /Mac|iPhone|iPad|iPod/i.test(platform) ? 'Meta+k' : 'Control+k';
}
"""

# Whether the palette's search field has the keyboard.
PALETTE_FOCUSED_JS = f"""
() => !!document.activeElement?.matches('{PALETTE_INPUT}')
"""

# The palette's active row, by its text.
ACTIVE_JS = f"""
() => document.querySelector('{PALETTE_ACTIVE}')?.textContent ?? null
"""

# Acknowledge the Settings warning for this server session as its Proceed button does
# (settings-warning-gate.js), so the drawer opens without it.
ACKNOWLEDGE_JS = """
async () => {
  const { scopedStorageKey } = await import('/design-system/js/storage-scope.js');
  const health = await (await fetch('/health', { cache: 'no-store' })).json();
  if (!health.session_id) return null;
  localStorage.setItem(scopedStorageKey('osprey-settings-warning-ack'), health.session_id);
  return health.session_id;
}
"""

# Where the drawer is: it slides in and out on a transform, so `open` alone says nothing about
# whether its tabs are on screen yet.
DRAWER_JS = """
() => {
  const drawer = document.getElementById('settings-drawer');
  const box = drawer.getBoundingClientRect();
  const still = getComputedStyle(drawer).transform;
  return {
    open: drawer.hasAttribute('open'),
    shown: box.left >= 0 && box.right <= innerWidth + 0.5
      && (still === 'none' || still === 'matrix(1, 0, 0, 1, 0, 0)'),
    gone: box.left >= innerWidth - 0.5,
  };
}
"""

CARD = "#operator-container .op-card-reason"
ALLOW = "#operator-container .op-btn-primary"
VERDICT = "Code review: pass"

REVIEW_BUDGET_MS = 420_000
RUN_BUDGET_MS = 180_000

# The approved script, one subtitle per row; a row's beats share its subtitle.
CAPTIONS = (
    "This deployment has code review on: a second AI agent reads every Python script the agent "
    "wants to run.",
    "Its instructions are in Settings. The operator looks them up.",
    "Its one rule: a script that moves a corrector must first read the beam mode, and stop while "
    "users have beam.",
    "The operator asks for a corrector scan. The agent was never told the rule.",
    "The agent writes the script. Before the operator is asked, the reviewer reads it.",
    "Blocked: the script never reads the beam mode. The reasons go back to the agent; the "
    "operator is not asked.",
    "The agent adds the check and sends the script again. This time the reviewer passes it.",
    "Only now is the operator asked, and the card shows the reviewer's verdict.",
    "Approved. The scan reads the beam mode first, then steps the corrector.",
    "Every review, pass or block, is recorded in the audit trail.",
)
(CHANNELS, LOOK_UP, THE_RULE, ASK, REVIEW, BLOCKED, RESUBMIT, ON_CARD, RAN, AUDIT) = CAPTIONS

# Move the switch for the session's machine to `asked`, as the header chip does.
ARM_JS = """
async (sessionId) => {
  const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
  const read = await (await fetch(`/api/terminal/posture${query}`)).json();
  const send = (epoch) => fetch('/api/terminal/posture', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId, target: read.control_target, state: 'asked', epoch, waived: false,
    }),
  });
  let answer = await send(read.switch_epoch);
  if (answer.status === 409) {
    const body = await answer.json();
    answer = await send(body.detail?.epoch ?? null);
  }
  return { status: answer.status, target: read.control_target };
}
"""


def acknowledge_settings(ctx: TakeContext) -> None:
    """Off camera: acknowledge the Settings warning, so ⌘K opens the drawer straight away."""
    if not ctx.page.evaluate(ACKNOWLEDGE_JS):
        raise ProbeFailed("acknowledge", "no server session to acknowledge the Settings warning")


def _chat_on_screen(take: Take) -> None:
    """Fail the take unless the operator chat has its composer on screen."""
    if not video_chat.mounted(take.page):
        raise ProbeFailed("chat", "the take left the operator chat")


#: When each take armed, the start of the hook records it reads.
_MARKS: dict[int, datetime] = {}


def _arm(take: Take) -> None:
    _chat_on_screen(take)
    _MARKS[id(take)] = datetime.now(UTC)
    answer = take.page.evaluate(ARM_JS, video_chat.session_id(take.page))
    if answer.get("status") != 200:
        raise ProbeFailed("arm", f"the write switch did not move: {answer}")


def _refused(take: Take) -> bool:
    """Whether the review has refused a run since the arm, by the approval hook's own record.

    The agent words what it was told its own way, so the screen is no witness: the hook logs
    ``refused`` for a call the review blocked ahead of the prompt.
    """
    _chat_on_screen(take)
    records = take.hook_log(_MARKS[id(take)])
    return any(r.get("hook") == "approval" and r.get("status") == "refused" for r in records)


def _show_blocked(take: Take, ms: int) -> None:
    """Bring the blocked call into view and hold it."""
    video_chat.scroll_to_end(take.page)
    take.page.wait_for_timeout(ms)


def _corrected_card(take: Take, ms: int) -> None:
    """The script on the card reads the beam mode; then hold the card."""
    card = take.page.locator("#operator-container .op-card-summary").first.inner_text()
    detail = take.page.locator("#operator-container .op-card-detail")
    text = card + (detail.first.text_content() or "" if detail.count() else "")
    if BEAM_MODE not in text:
        raise ProbeFailed("verdict", f"the script on the card does not read {BEAM_MODE}")
    take.page.wait_for_timeout(ms)


def _drawer(take: Take, name: str, settled: str) -> None:
    """Poll the drawer until its *settled* state; the Settings warning on screen fails it."""
    for _ in range(DRAWER_BUDGET_MS // POLL_MS):
        if shown(take.page, WARNING):
            raise ProbeFailed(name, "the Settings warning is on camera")
        state = take.page.evaluate(DRAWER_JS)
        if state[settled] and state["open"] == (settled == "shown"):
            return
        take.page.wait_for_timeout(POLL_MS)
    raise ProbeFailed(name, f"the Settings drawer did not settle {settled}: {state}")


def _open_palette(take: Take) -> None:
    """Press the palette's chord and wait for its search field to take the keyboard."""
    take.page.keyboard.press(take.page.evaluate(CHORD_JS))
    for _ in range(PALETTE_BUDGET_MS // POLL_MS):
        if take.page.evaluate(PALETTE_FOCUSED_JS):
            return
        take.page.wait_for_timeout(POLL_MS)
    raise ProbeFailed("palette", "the command palette did not open with the keyboard on it")


def _search(take: Take) -> None:
    take.page.keyboard.type(SEARCH, delay=TYPE_DELAY_MS)


def _choose_row(take: Take) -> None:
    """Arrow down to the rules' row, as an operator would, and press Enter on it."""
    rows = take.page.locator(PALETTE_ROW).count()
    for _ in range(rows + 1):
        if ROW_TEXT in (take.page.evaluate(ACTIVE_JS) or ""):
            take.page.keyboard.press("Enter")
            return
        take.page.keyboard.press("ArrowDown")
        take.page.wait_for_timeout(POLL_MS)
    raise ProbeFailed("reveal", f"no palette row {ROW_TEXT!r} became active")


def _wait_done(take: Take) -> Any:
    for _ in range(RUN_BUDGET_MS // POLL_MS):
        if video_chat.ready(take.page):
            return
        take.page.wait_for_timeout(POLL_MS)
    raise ProbeFailed("ran", f"the turn did not end:\n{take.tail()}")


CODE_REVIEW = (
    hold("channels", CHANNELS, 4_000),
    Beat("palette", LOOK_UP, _open_palette),
    Beat("search", LOOK_UP, _search),
    wait_for("row", LOOK_UP, PALETTE_ROW, budget_ms=PALETTE_BUDGET_MS, text=ROW_TEXT),
    Beat("reveal", LOOK_UP, _choose_row),
    Beat("settings-open", LOOK_UP, lambda take: _drawer(take, "settings-open", "shown")),
    wait_for("rules-shown", LOOK_UP, RULE, budget_ms=DRAWER_BUDGET_MS, text=BEAM_MODE),
    spotlight("rule", THE_RULE, RULE, text=BEAM_MODE, ms=8_000),
    click("close", ASK, CLOSE_DRAWER),
    Beat("settings-closed", ASK, lambda take: _drawer(take, "settings-closed", "gone")),
    Beat("arm", ASK, _arm),
    focus("focus", ASK),
    prompt("ask", ASK, PROMPT_SCAN),
    wait_until(
        "review",
        REVIEW,
        _refused,
        what="no call refused by the review",
        budget_ms=REVIEW_BUDGET_MS,
        fast_forward=True,
        refuse=((CARD, "the operator was asked before the review blocked a script"),),
    ),
    Beat("blocked", BLOCKED, lambda take: _show_blocked(take, 4_000)),
    wait_for(
        "resubmit",
        RESUBMIT,
        CARD,
        budget_ms=REVIEW_BUDGET_MS,
        text=VERDICT,
        fast_forward=True,
    ),
    Beat("verdict", ON_CARD, lambda take: _corrected_card(take, 5_000)),
    click("allow", RAN, ALLOW, text="Allow"),
    Beat("ran", RAN, _wait_done, fast_forward=True),
    hold("result", AUDIT, 4_000),
)
