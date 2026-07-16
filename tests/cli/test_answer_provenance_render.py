"""Static render guard for the answer-provenance / verify-first doctrine.

Renders the shipped orchestrator prompt surfaces through the real build
pipeline and asserts the verify-first provenance rule is present, coherent
(no leftover that a reader could take as contradicting it), and free of any
facility-specific source vocabulary:

* the ``control-operator`` output-style (the behaviorally load-bearing carrier,
  rendered into ``settings.json`` ``outputStyle``), and
* both orchestrator personas — ``CLAUDE.md`` (control) and its ARIEL variant.

This is the CI-enforceable half of the change (SC1/SC2/SC7): deterministic and
no API key. The behavioral companion (``test_answer_provenance_scenario``) is
local/advisory. See ``.claude/plans/answer-provenance-transparency/``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_cmd import build

# Facility-specific *source identifiers* that must never leak into the shared,
# facility-agnostic core prompts (SC7). ``concept_id`` / ``texkey`` are literal
# ALS source-idiom tokens; the third pattern catches a hard-coded PV / channel
# address like ``SR:C01-MG:PS1`` (uppercase run, colon, more address). Generic
# English ("a channel address", "an entry ID") intentionally does NOT match.
_FACILITY_ID_PATTERNS = (
    r"concept_id",
    r"texkey",
    r"\b[A-Z][A-Z0-9]{1,}:[A-Z0-9][\w:-]*\b",
)


def _build_preset(preset: str, dest: Path) -> Path:
    """Render a preset through the real build pipeline; return the project dir."""
    runner = CliRunner()
    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            preset,
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(dest),
        ],
    )
    assert result.exit_code == 0, result.output
    return dest / "smoke"


def _markdown_section(text: str, heading: str) -> str:
    """Body of a top-level ``# heading`` section, up to the next top-level ``# ``.

    ``## sub`` headings inside the section are retained (they do not start with
    ``"# "``), so this cleanly isolates one section's edited prose.
    """
    body: list[str] = []
    capturing = False
    for line in text.splitlines():
        if line.strip() == heading:
            capturing = True
            continue
        if capturing and line.startswith("# ") and line.strip() != heading:
            break
        if capturing:
            body.append(line)
    return "\n".join(body)


def _assert_no_facility_identifier(region: str, where: str) -> None:
    for pattern in _FACILITY_ID_PATTERNS:
        match = re.search(pattern, region)
        assert match is None, (
            f"facility-specific source identifier {match.group(0)!r} "
            f"(pattern {pattern!r}) leaked into {where}"
        )


@pytest.fixture(scope="module")
def control_project(tmp_path_factory) -> Path:
    return _build_preset("control-assistant", tmp_path_factory.mktemp("ctrl"))


@pytest.fixture(scope="module")
def ariel_project(tmp_path_factory) -> Path:
    return _build_preset("ariel-standalone", tmp_path_factory.mktemp("ariel"))


@pytest.fixture(scope="module")
def control_output_style(control_project: Path) -> str:
    path = control_project / ".claude" / "output-styles" / "control-operator.md"
    assert path.exists(), f"output-style not materialized: {path}"
    return path.read_text(encoding="utf-8")


class TestControlOperatorOutputStyle:
    """SC1 — the load-bearing output-style states verify-first coherently."""

    def test_states_verify_first_and_up_front_flag(self, control_output_style: str) -> None:
        style = control_output_style
        # Verify-first is the stated default posture (FR1).
        assert "Verify first" in style
        assert "lead with that result" in style
        # A plain, up-front not-tool-backed flag for the residual case (FR2).
        assert "not a tool-cited source" in style

    def test_forbids_answer_then_verify_trailer(self, control_output_style: str) -> None:
        # The FR3 anti-pattern must be named and forbidden, not merely implied.
        style = control_output_style
        assert "I can check if you want" in style
        assert "default first action" in style

    def test_ships_conditional_provenance_summary_shape(self, control_output_style: str) -> None:
        # FR5: a substantive multi-tool answer closes with an explicit
        # provenance summary (sources + a confidence/scope note); a trivial
        # answer stays terse. Asserts the *conditional shape* rule, not three
        # fixed footer labels (the doctrine describes an outcome, not a format).
        style = control_output_style
        assert "Answer shape" in style
        assert "stays terse" in style
        assert "provenance summary" in style
        assert "confidence" in style.lower()

    def test_no_contradictory_leftover(self, control_output_style: str) -> None:
        # CF-1/FR6: the old absolute prohibition ("fill gaps with textbook
        # physics") was *rewritten into* the ordered posture, not left beside a
        # rule that tells the agent how to use general knowledge with a flag.
        assert "textbook physics" not in control_output_style

    def test_epistemic_block_is_facility_agnostic(self, control_output_style: str) -> None:
        # SC7 — scoped to the edited section so the pre-existing archiver
        # formatting example (# Data Formatting) cannot false-trip the guard.
        epistemic = _markdown_section(control_output_style, "# Epistemic Discipline")
        assert epistemic.strip(), "Epistemic Discipline section not found in render"
        assert "Verify first" in epistemic
        _assert_no_facility_identifier(epistemic, "the control-operator epistemic block")


class TestPersonaParity:
    """SC2 — both orchestrator personas carry the verify-first posture."""

    def test_control_persona_carries_posture(self, control_project: Path) -> None:
        claude = (control_project / "CLAUDE.md").read_text(encoding="utf-8")
        assert "Control System Assistant" in claude  # right persona rendered
        paragraph = self._posture_region(
            claude, r"\*\*Verify first, and flag anything you can't\.\*\*", r"\n\n"
        )
        assert "lead with that result" in paragraph
        assert "up front" in paragraph
        assert "not an afterthought" in paragraph
        _assert_no_facility_identifier(paragraph, "the control CLAUDE.md posture paragraph")

    def test_ariel_persona_carries_posture(self, ariel_project: Path) -> None:
        claude = (ariel_project / "CLAUDE.md").read_text(encoding="utf-8")
        assert "Logbook Research Assistant" in claude  # ARIEL persona rendered
        assert "Control System Assistant" not in claude
        bullet = self._posture_region(
            claude, r"- \*\*Verify first; flag anything not tool-backed\.\*\*", r"\n- \*\*"
        )
        assert "lead with the sourced result" in bullet
        assert "[entry_id]" in bullet  # spoken in ARIEL's own citation idiom (FR7)
        assert "I can search" in bullet  # answer-then-verify trailer forbidden
        _assert_no_facility_identifier(bullet, "the ARIEL CLAUDE.md posture bullet")

    @staticmethod
    def _posture_region(text: str, start: str, stop: str) -> str:
        match = re.search(rf"{start}.*?(?={stop})", text, re.S)
        assert match is not None, f"posture region not found (anchor {start!r})"
        return match.group(0)
