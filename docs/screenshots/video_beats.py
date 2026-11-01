"""The beat primitives a docs video's storyboard is built from.

Each primitive returns a :class:`Beat`: one storyboard step with its name, its
caption and whether the encoder fast-forwards it. Played on a
:class:`~docs.screenshots.video_take.Take`, a beat stamps its span on the
timeline and shows its caption while it runs, and every check that gives up
raises :class:`~docs.screenshots.video_probes.ProbeFailed` naming the beat, so
a failed take says where it failed.

A wait on the screen (:func:`wait_for`, :func:`wait_gone`) also watches the
agent while it polls: an API error ending the turn, an unanswered terminal
dialog and a stall each fail the beat at once, and so does any selector named
in ``refuse``. A probe only one video needs is a custom beat: a
``Beat(name, caption, run)`` with the video's own ``run(take)``, or a plain
``run(take)`` callable that opens its own steps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from docs.screenshots import video_check_stall, video_overlay, video_probes
from docs.screenshots.video_probes import ProbeFailed

if TYPE_CHECKING:
    from docs.screenshots.video_take import Take

# Whether a visible element matches the selector and, when given, holds the text.
_SHOWN_JS = """
([selector, text]) => [...document.querySelectorAll(selector)].some(
  (el) => el.getClientRects().length > 0 && (!text || el.textContent.includes(text))
)
"""


@dataclass(frozen=True)
class Beat:
    """One storyboard step: ``run(take)`` inside a timeline step of this name."""

    name: str
    caption: str | None
    run: Callable[[Take], Any]
    fast_forward: bool = False

    def __call__(self, take: Take) -> None:
        with take.step(self.name, self.caption, self.fast_forward):
            self.run(take)


def shown(page: Any, selector: str, text: str | None = None) -> bool:
    """Whether a visible element matches *selector* (and contains *text*)."""
    return bool(page.evaluate(_SHOWN_JS, [selector, text]))


def focus(name: str, caption: str | None) -> Beat:
    """Glide the cursor to the terminal and click it."""
    return Beat(name, caption, lambda take: take.focus_terminal())


def prompt(name: str, caption: str | None, text: str) -> Beat:
    """Type *text* into the focused terminal and submit it."""
    return Beat(name, caption, lambda take: take.submit(name, text))


def hold(name: str, caption: str | None, ms: int) -> Beat:
    """Hold the screen still for *ms* milliseconds."""
    return Beat(name, caption, lambda take: take.page.wait_for_timeout(ms))


def click(name: str, caption: str | None, selector: str, *, text: str | None = None) -> Beat:
    """Glide the cursor to the first visible *selector* (holding *text*) and click it."""

    def run(take: Take) -> None:
        locator = take.page.locator(selector)
        if text is not None:
            locator = locator.filter(has_text=text)
        if not shown(take.page, selector, text):
            raise ProbeFailed(name, f"nothing to click: no visible {selector}")
        take.click(locator.first)

    return Beat(name, caption, run)


def spotlight(
    name: str, caption: str | None, selector: str, *, text: str | None = None, ms: int
) -> Beat:
    """Outline the first visible *selector* (holding *text*) for *ms* milliseconds.

    The outline is the video's own overlay, drawn over the page and cleared when
    the beat ends; the page under it is not touched.
    """

    def run(take: Take) -> None:
        if not shown(take.page, selector, text):
            raise ProbeFailed(name, f"nothing to spotlight: no visible {selector}")
        locator = take.page.locator(selector)
        if text is not None:
            locator = locator.filter(has_text=text)
        target = locator.first
        target.scroll_into_view_if_needed()
        box = target.bounding_box()
        if not box:
            raise ProbeFailed(name, f"{selector} has no box on screen")
        video_overlay.set_spotlight(take.page, box)
        try:
            take.page.wait_for_timeout(ms)
        finally:
            video_overlay.set_spotlight(take.page, None)

    return Beat(name, caption, run)


def _wait(
    name: str,
    caption: str | None,
    done: Callable[[Any], bool],
    what: str,
    budget_ms: int,
    fast_forward: bool,
    refuse: tuple[tuple[str, str], ...],
    dialogs: tuple[str, ...],
) -> Beat:
    def run(take: Take) -> None:
        watch = take.watch(name, dialogs)
        for _ in range(max(1, -(-budget_ms // video_probes.POLL_MS))):
            if done(take.page):
                return
            for selector, why in refuse:
                if shown(take.page, selector):
                    raise ProbeFailed(name, why)
            if take.turn() == "errored":
                raise ProbeFailed(name, "the agent's turn ended in an API error")
            watch()
            take.page.wait_for_timeout(video_probes.POLL_MS)
        if done(take.page):
            return
        raise ProbeFailed(
            name,
            f"{what} within {budget_ms} ms; last lines:\n{take.tail()}",
        )

    return Beat(name, caption, run, fast_forward)


def wait_for(
    name: str,
    caption: str | None,
    selector: str,
    *,
    budget_ms: int,
    text: str | None = None,
    fast_forward: bool = False,
    refuse: tuple[tuple[str, str], ...] = (),
    dialogs: tuple[str, ...] = video_check_stall.NOTICE_DIALOGS,
) -> Beat:
    """Wait until a visible *selector* (holding *text*) is on screen.

    ``refuse`` pairs a selector with the failure it means: any of them on
    screen fails the beat before the budget runs out.
    """
    what = f"no {selector}" + (f" with {text!r}" if text else "")
    return _wait(
        name,
        caption,
        lambda page: shown(page, selector, text),
        what,
        budget_ms,
        fast_forward,
        refuse,
        dialogs,
    )


def wait_gone(
    name: str,
    caption: str | None,
    selector: str,
    *,
    budget_ms: int,
    dialogs: tuple[str, ...] = video_check_stall.NOTICE_DIALOGS,
) -> Beat:
    """Wait until no visible element matches *selector*."""
    return _wait(
        name,
        caption,
        lambda page: not shown(page, selector),
        f"{selector} still on screen",
        budget_ms,
        False,
        (),
        dialogs,
    )
