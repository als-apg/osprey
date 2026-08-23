"""Guards for the ``Plain Language`` block in ``control-operator.md.j2``.

The block exists so the control assistant's prose stays readable to operators
working under time pressure, many of whom read English as a second language.
It is distilled from ASD-STE100 Simplified Technical English and the Google
developer documentation style guide -- neither text is vendored, because
ASD-STE100 is proprietary and the Google guide is CC BY 4.0.

Two things are pinned:

* The block survives into the rendered artifact. It is guidance in a shipped
  prompt, so nothing fails loudly if a template refactor drops it.
* The block obeys its own rules. Guidance that violates the sentence-length
  and filler-word limits it sets teaches the model the limits are decorative.
"""

import re

from osprey.cli.templates.manager import TemplateManager

#: Lines the block must carry, verbatim.
PLAIN_LANGUAGE_MARKERS = (
    "# Plain Language",
    "One idea per sentence.",
    "No noun stack longer than three words",
    "Facility vocabulary is not jargon.",
    "https://developers.google.com/style",
)

#: The limit the block sets for itself.
MAX_WORDS_PER_SENTENCE = 25

#: Words the block tells the agent to cut. The bullet that *defines* the list
#: necessarily contains all of them, so the scan skips that one line -- see
#: ``_prose_lines``.
FILLER_WORDS = ("simply", "just", "easy", "obviously", "please")

#: Start of the bullet that defines ``FILLER_WORDS``.
FILLER_BULLET_PREFIX = "- Cut the filler words:"


def _plain_language_block(text: str) -> str:
    """Return the ``# Plain Language`` section, without its heading."""
    _, _, after = text.partition("# Plain Language")
    block, _, _ = after.partition("\n# ")
    return block


def _prose_lines(block: str) -> list[str]:
    """Lines of the block that the self-check applies to."""
    return [
        line for line in block.splitlines() if not line.strip().startswith(FILLER_BULLET_PREFIX)
    ]


def _sentences(prose: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace.

    Decimals such as ``4.0`` are safe: the dot is not followed by a space.
    """
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", prose) if s.strip()]


def _word_count(sentence: str) -> int:
    """Count words, ignoring markdown punctuation and standalone dashes."""
    tokens = re.sub(r'[*_`"()\[\],;:.!?]', " ", sentence).split()
    return len([t for t in tokens if t not in {"—", "-", "→"}])


def _render_output_style(tmp_path) -> str:
    """Scaffold a project selecting the style, return the rendered artifact."""
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name="plain-language-style",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
        artifacts={"hooks": ["memory-guard"], "output_styles": ["control-operator"]},
    )
    return (project_dir / ".claude" / "output-styles" / "control-operator.md").read_text()


def test_plain_language_block_reaches_the_rendered_style(tmp_path):
    """Every marker survives into the artifact the agent actually loads."""
    rendered = _render_output_style(tmp_path)
    for marker in PLAIN_LANGUAGE_MARKERS:
        assert marker in rendered, f"missing from rendered output style: {marker!r}"


def test_plain_language_block_keeps_its_own_sentence_limit(tmp_path):
    """The guidance obeys the 25-word ceiling it sets."""
    block = _plain_language_block(_render_output_style(tmp_path))
    too_long = [
        (sentence, _word_count(sentence))
        for sentence in _sentences(" ".join(_prose_lines(block)))
        if _word_count(sentence) > MAX_WORDS_PER_SENTENCE
    ]
    assert not too_long, f"sentences over {MAX_WORDS_PER_SENTENCE} words: {too_long}"


def test_plain_language_block_uses_no_filler_words(tmp_path):
    """The guidance avoids the filler it tells the agent to cut."""
    block = _plain_language_block(_render_output_style(tmp_path))
    prose = " ".join(_prose_lines(block)).lower()
    found = [word for word in FILLER_WORDS if re.search(rf"\b{word}\b", prose)]
    assert not found, f"filler words in the plain-language block: {found}"
