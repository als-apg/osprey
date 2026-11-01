"""The code review demo: its storyboard and the probes only it needs.

The deployment ships one rule, that a script moving a corrector first reads the beam mode. The
operator asks for a corrector scan, which the agent writes without the check, since the rule is
the reviewer's and not the agent's. The review blocks it before anyone is asked, the agent adds
the check, and the corrected script reaches the operator with the verdict on its card.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from docs.screenshots import video_chat
from docs.screenshots.video_beats import Beat, click, focus, hold, prompt, wait_for
from docs.screenshots.video_probes import POLL_MS, ProbeFailed

if TYPE_CHECKING:
    from docs.screenshots.video_take import Take

CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"
BEAM_MODE = "SR:BEAM:USER_MODE"
PROMPT_SCAN = f"Write a Python script that steps {CORRECTOR} from 0 to 1 in 10 steps, and run it."

_PRESET = (
    Path(__file__).parents[2] / "src" / "osprey" / "profiles" / "presets" / "control-assistant.yml"
)
HOOKS = [*yaml.safe_load(_PRESET.read_text(encoding="utf-8"))["hooks"], "code-review"]
INIT_SETS = (
    # The deployment offers the Simple view only: Expert is not on this stack at all.
    "config.web.expert_view=false",
    "config.code_review.enabled=true",
    "config.code_review.level=block",
    f"hooks={json.dumps(HOOKS, separators=(',', ':'))}",
)

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


def _simple_view(take: Take) -> None:
    """Fail the take unless the page is the Simple view with its composer on screen."""
    if not video_chat.mounted(take.page):
        raise ProbeFailed("simple-view", "the take left the Simple view")


def _arm(take: Take) -> None:
    _simple_view(take)
    answer = take.page.evaluate(ARM_JS, video_chat.session_id(take.page))
    if answer.get("status") != 200:
        raise ProbeFailed("arm", f"the write switch did not move: {answer}")


def _blocked_first(take: Take) -> None:
    """The first card must follow a block, or the take shows nothing of the review stopping code."""
    _simple_view(take)
    shown = take.screen()
    if "blocked" not in shown.lower():
        raise ProbeFailed("blocked", f"no block on screen before the card:\n{take.tail()}")


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
    Beat(
        "arm", "The deployment wrote one rule: read the beam mode before moving a corrector.", _arm
    ),
    hold("rule", None, HOLD_MS),
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
    Beat("blocked", "The first script was blocked. The agent added the check.", _blocked_first),
    hold("blocked-hold", None, HOLD_MS),
    Beat(
        "verdict", "The operator sees only the corrected script, with the verdict.", _corrected_card
    ),
    hold("verdict-hold", None, HOLD_MS),
    click("allow", None, ALLOW, text="Allow"),
    Beat("ran", "Approved. The scan runs.", _wait_done, fast_forward=True),
    hold("result", None, HOLD_MS),
)
