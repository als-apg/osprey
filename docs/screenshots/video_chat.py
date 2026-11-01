"""Probes on the Simple view's operator chat, the counterpart of the xterm probes.

The selectors are the ones the chat's own browser tests use
(``tests/interfaces/web_terminal/test_chat_browser.py``): the console mounts in
``#operator-container``, the composer is its ``.op-input-area textarea``, a turn
in flight marks the container ``.streaming``, and a failed turn shows an
``.op-error-block`` or an ``.op-footer-error``. The chat's session id is the
tab's session pointer in ``localStorage`` (``osprey-pty-session``, with a
``--<scope>`` suffix on a scoped page); the session's Claude Code transcript is
named after it.
"""

from __future__ import annotations

from typing import Any

COMPOSER = "#operator-container .op-input-area textarea"

# The page is in the Simple view and its composer is on screen.
MOUNTED_JS = """
() => document.documentElement.dataset.uiMode === 'simple'
  && !!document.querySelector('#operator-container .op-input-area textarea')
"""

# The composer takes input: on screen, enabled, and no turn in flight.
READY_JS = """
() => {
  const box = document.querySelector('#operator-container .op-input-area textarea');
  const host = document.querySelector('#operator-container');
  return !!box && box.getClientRects().length > 0 && !box.disabled
    && !!host && !host.classList.contains('streaming');
}
"""

FOCUS_JS = """
() => document.activeElement === document.querySelector(
  '#operator-container .op-input-area textarea')
"""

TEXT_JS = """
() => {
  const messages = document.querySelector('#operator-container .op-messages');
  return messages ? messages.innerText : '';
}
"""

ERRORED_JS = """
() => !!document.querySelector(
  '#operator-container .op-error-block, #operator-container .op-footer-error')
"""

POINTER_JS = """
() => {
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key === 'osprey-pty-session' || key.startsWith('osprey-pty-session--')) {
      return localStorage.getItem(key);
    }
  }
  return null;
}
"""


def mounted(page: Any) -> bool:
    return bool(page.evaluate(MOUNTED_JS))


def ready(page: Any) -> bool:
    return bool(page.evaluate(READY_JS))


def focused(page: Any) -> bool:
    return bool(page.evaluate(FOCUS_JS))


def text(page: Any) -> str:
    """The chat transcript as shown on screen."""
    return str(page.evaluate(TEXT_JS) or "")


def errored(page: Any) -> bool:
    """Whether a turn on screen ended in an error."""
    return bool(page.evaluate(ERRORED_JS))


def session_id(page: Any) -> str | None:
    """The chat's session id: the tab's session pointer, once it has one."""
    value = page.evaluate(POINTER_JS)
    return str(value) if value else None
