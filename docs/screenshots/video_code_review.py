"""The code review demo: its storyboard and the probes only it needs.

The video opens on that rule, in the reviewer's instructions under Settings → Safety. The
deployment ships one rule, that a script moving a corrector first reads the beam mode. The
operator asks for a corrector scan, which the agent writes without the check, since the rule is
the reviewer's and not the agent's. The review blocks it before anyone is asked, the agent adds
the check, and the corrected script reaches the operator with the verdict on its card.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from docs.screenshots import video_chat
from docs.screenshots.video_beats import Beat, click, focus, hold, prompt, spotlight, wait_for
from docs.screenshots.video_probes import POLL_MS, ProbeFailed

if TYPE_CHECKING:
    from docs.screenshots.video_take import Take

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

# The way to the reviewer's rules: ☰ More → Settings → Safety → the rules' card.
MORE = "#more-btn"
SETTINGS = "#more-settings"
SAFETY_TAB = '#settings-drawer .drawer-tab[data-tab="tab-safety"]'
RULES_CARD = '#tab-safety .prompts-card[data-name="code-review/rules"]'
RULE = "#tab-safety .osprey-md-rendered li"
CLOSE_DRAWER = "#settings-drawer .drawer-close-btn"
DRAWER_BUDGET_MS = 15_000

CARD = "#operator-container .op-card-reason"
ALLOW = "#operator-container .op-btn-primary"
VERDICT = "Code review: pass"

REVIEW_BUDGET_MS = 420_000
RUN_BUDGET_MS = 180_000
HOLD_MS = 4_000

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


def _blocked_first(take: Take) -> None:
    """The review refused a run before the card, by the approval hook's own record.

    The agent words what it was told its own way, so the screen is no witness: the hook logs
    ``refused`` for a call the review blocked ahead of the prompt.
    """
    _chat_on_screen(take)
    records = take.hook_log(_MARKS[id(take)])
    refused = [r for r in records if r.get("hook") == "approval" and r.get("status") == "refused"]
    if not refused:
        raise ProbeFailed(
            "blocked", f"the approval hook refused nothing before the card:\n{take.tail()}"
        )


def _corrected_card(take: Take) -> None:
    """The script on the card reads the beam mode."""
    card = take.page.locator("#operator-container .op-card-summary").first.inner_text()
    detail = take.page.locator("#operator-container .op-card-detail")
    text = card + (detail.first.text_content() or "" if detail.count() else "")
    if BEAM_MODE not in text:
        raise ProbeFailed("corrected", f"the script on the card does not read {BEAM_MODE}")


def _wait_done(take: Take) -> Any:
    for _ in range(RUN_BUDGET_MS // POLL_MS):
        if video_chat.ready(take.page):
            return
        take.page.wait_for_timeout(POLL_MS)
    raise ProbeFailed("ran", f"the turn did not end:\n{take.tail()}")


CODE_REVIEW = (
    click("more", None, MORE),
    click("settings", None, SETTINGS),
    click("safety", None, SAFETY_TAB),
    wait_for("rules-listed", None, RULES_CARD, budget_ms=DRAWER_BUDGET_MS),
    click("rules", None, RULES_CARD),
    wait_for("rules-shown", None, RULE, budget_ms=DRAWER_BUDGET_MS, text=BEAM_MODE),
    spotlight(
        "rule",
        "The code reviewer's one rule: read the beam mode before moving a corrector.",
        RULE,
        text=BEAM_MODE,
        ms=HOLD_MS,
    ),
    click("close", None, CLOSE_DRAWER),
    Beat("arm", None, _arm),
    focus("focus", None),
    prompt("ask", None, PROMPT_SCAN),
    wait_for(
        "review",
        "The reviewer reads the script before anyone is asked.",
        CARD,
        budget_ms=REVIEW_BUDGET_MS,
        text=VERDICT,
        fast_forward=True,
    ),
    Beat(
        "blocked", "The review blocked the first script. The agent added the check.", _blocked_first
    ),
    hold("blocked-hold", None, HOLD_MS),
    Beat(
        "verdict", "The operator sees only the corrected script, with the verdict.", _corrected_card
    ),
    hold("verdict-hold", None, HOLD_MS),
    click("allow", None, ALLOW, text="Allow"),
    Beat("ran", "Approved. The scan runs.", _wait_done, fast_forward=True),
    hold("result", None, HOLD_MS),
)
