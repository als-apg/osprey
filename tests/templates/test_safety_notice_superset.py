"""The packaged safety notice is the superset; the scaffolded copy adds only its
"this file is yours to edit" preamble.

Two files carry the same safety copy for two different readers. The packaged one
(``templates/modules/web_terminals/notices/working-safely.md``) is what a landing
page renders when a deployment lists no notices of its own, so it is the copy an
operator sees when nobody made a choice. The other
(``templates/apps/control_assistant/data/landing/working-safely.md``) is written
into a facility's profile by ``osprey init`` as a seed the facility then owns and
edits, which is why it cannot be generated at render time.

Two hand-kept copies drift, and this one had: the seed grew a section the
packaged copy never got, so the deployment that made no choice got the shorter
warning. The invariant that keeps them together is stated here rather than left
to whoever edits one of them next — the seed is the packaged text with the
preamble paragraph inserted, and nothing else.

Editing either file means editing both. The failure below names the paragraph
that differs.
"""

from __future__ import annotations

from pathlib import Path

import osprey

_TEMPLATES = Path(osprey.__file__).resolve().parent / "templates"

#: Shipped in the wheel, rendered by `_packaged_notice()` when a config lists no
#: `landing.notices` of its own.
PACKAGED_NOTICE = _TEMPLATES / "modules" / "web_terminals" / "notices" / "working-safely.md"

#: Copied verbatim into a facility's profile by `osprey init`, then owned by the
#: facility.
SEED_NOTICE = _TEMPLATES / "apps" / "control_assistant" / "data" / "landing" / "working-safely.md"

#: The one paragraph the seed carries and the packaged copy must not: it tells a
#: facility how to add notices of its own, which is advice for whoever edits the
#: profile, not for the operator reading the rendered page.
PREAMBLE_OPENER = "This file is yours to edit."


def _paragraphs(path: Path) -> list[str]:
    """One entry per blank-line-separated block of *path*, in order.

    Comparing blocks rather than lines means a failure names the paragraph that
    diverged instead of the first line that did, which is the difference between
    a diff a reader can act on and one they have to reconstruct.
    """
    text = path.read_text(encoding="utf-8")
    assert text.strip(), f"empty notice: {path}"
    return [block for block in text.split("\n\n") if block.strip()]


def test_the_seed_is_the_packaged_notice_plus_its_preamble() -> None:
    """The whole invariant, in one comparison."""
    packaged = _paragraphs(PACKAGED_NOTICE)
    seed = _paragraphs(SEED_NOTICE)

    preambles = [block for block in seed if block.startswith(PREAMBLE_OPENER)]
    assert len(preambles) == 1, (
        f"expected exactly one {PREAMBLE_OPENER!r} paragraph in {SEED_NOTICE.name}, "
        f"found {len(preambles)}"
    )

    assert [block for block in seed if not block.startswith(PREAMBLE_OPENER)] == packaged, (
        "the scaffolded seed and the packaged notice have drifted apart. They carry "
        "the same safety copy for two readers, and the seed may add nothing to it "
        "but its editing preamble — fold the change into both."
    )


def test_the_packaged_notice_carries_no_editing_preamble() -> None:
    """The packaged copy is not a file anyone edits: it is read out of the wheel,
    and a deployment that wants its own notices lists them instead."""
    assert PREAMBLE_OPENER not in PACKAGED_NOTICE.read_text(encoding="utf-8")
