"""The code review route: what the Settings drawer's Review tab shows.

``GET /api/code-review`` answers every operator: whether code review is on, its
level and time limit, the reviewer's rules file and its text, and ``warning`` --
one sentence naming what a review will do while the section or the rules file
cannot be used, and the fix. ``can_change`` says whether this terminal is the
admin tier (``web.config_panel.enabled``), read the way the safety-switch route
reads it, and ``switch`` is code review's runtime safety switch, ``None`` when
the profile leaves review off (:mod:`osprey.interfaces.web_terminal.safety_switches`).
The admin moves it through the safety-switch route, audited there.

The review consumers, which run in the agent's processes, read the switch from
the terminal's own socket instead
(:mod:`osprey.interfaces.web_terminal.code_review_switch`).

Everything is read through the reviewer's own functions in
:mod:`osprey.services.code_review.settings`, uncached, so the tab shows the
section and the file the next review acts on, and refuses the files a review
would refuse. The warning follows what a review does with each failure:
an unusable section is a ``config`` failure with no level, which refuses
(``review.decide``); an unusable prompt, an empty one included, is a ``prompt``
failure, which refuses under ``block`` and lets the code run unreviewed under ``flag``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from osprey.cli.profile_conventions import RESERVED_PATH_CHANNELS
from osprey.interfaces.web_terminal.safety_switches import CODE_REVIEW_SWITCH, switches_of

logger = logging.getLogger(__name__)

router = APIRouter()

#: Where a ``code_review`` key is changed: the channel the protected set names
#: for ``config.yml``, so the fix the tab gives and the refusal a write gets agree.
_SECTION_CHANNEL = RESERVED_PATH_CHANNELS["config.yml"]

#: The warning for a section a review cannot read: every review refuses.
SECTION_UNUSABLE = (
    "The code_review section cannot be read ({reason}), so every review fails and the "
    "agent's Python and notebook code is blocked; correct it in {channel} and rebuild."
)

#: The warning for an unusable rules file under ``level: block``.
RULES_UNUSABLE_BLOCK = (
    "The rules file cannot be used ({reason}), so every review fails and the agent's "
    "Python and notebook code is blocked; restore {path} or point code_review.prompt "
    "at a readable file under data/."
)

#: The warning for an unusable rules file under ``level: flag``.
RULES_UNUSABLE_FLAG = (
    "The rules file cannot be used ({reason}), so every review fails and the agent's "
    "code runs unreviewed; restore {path} or point code_review.prompt at a readable "
    "file under data/."
)


def _section_settings(config: Mapping) -> Any:
    """The section's settings whether it is on or off, or the defaults where it is malformed.

    An off section is not validated by a running session, so a value it holds
    is shown as written where it parses and replaced by the default where not.
    """
    from osprey.services.code_review.settings import (
        ReviewConfigError,
        ReviewSettings,
        parse_review_section,
    )

    try:
        return parse_review_section(config.get("code_review", {}))
    except ReviewConfigError:
        return ReviewSettings()


def _switch_row(app: Any) -> dict[str, Any] | None:
    """Code review's switch as the safety-switch route serves it, or ``None``."""
    switches = switches_of(app)
    return switches.row(CODE_REVIEW_SWITCH) if switches is not None else None


@router.get("/api/code-review")
async def get_code_review(request: Request) -> dict[str, Any]:
    """The code review section as the next review reads it, for every operator."""
    from osprey.services.code_review.settings import (
        ReviewConfigError,
        load_review_settings,
        read_review_config,
        resolve_prompt,
    )
    from osprey_connectors.workspace import repo_root_for_config

    config_path: Path | None = getattr(request.app.state, "config_path", None)
    body: dict[str, Any] = {
        "can_change": bool(getattr(request.app.state, "config_panel_enabled", False)),
        "switch": _switch_row(request.app),
        "warning": None,
        "rules": None,
    }
    try:
        config = read_review_config(config_path) if config_path else {}
        running = load_review_settings(config)
    except ReviewConfigError as exc:
        logger.warning("Code review section cannot be read: %s", exc)
        body["warning"] = SECTION_UNUSABLE.format(reason=exc, channel=_SECTION_CHANNEL)
        body.update(enabled=None, level=None, time_limit=None, prompt=None)
        return body

    shown = running if running.enabled else _section_settings(config)
    body.update(
        enabled=running.enabled,
        level=shown.level,
        time_limit=shown.time_limit,
        prompt=shown.prompt,
    )
    if config_path is None:
        return body
    try:
        rules = resolve_prompt(shown, repo_root_for_config(config_path), config)
    except ReviewConfigError as exc:
        # Off, an unusable file is no review's concern; on, it fails every one.
        if running.enabled:
            template = RULES_UNUSABLE_FLAG if running.level == "flag" else RULES_UNUSABLE_BLOCK
            body["warning"] = template.format(reason=exc, path=shown.prompt)
        return body
    body["rules"] = rules
    return body
