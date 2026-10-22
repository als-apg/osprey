"""The simulation engine speaks generic words; a facility's names stay in its own files.

A facility binds its records -- a monitor's mode, its attenuation, its pilot
tone, the cleaning of a stored bunch -- onto the engine's generic readout
attributes and commands in its models file. None of those facility words may
appear in an engine module, so the engine never grows a facility's notion of
one of its devices.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import osprey

ENGINES = Path(osprey.__file__).parent / "simulation" / "engines"

#: A facility's names for a monitor's settings and a bunch's cleaning, matched
#: case-insensitively as a word or as a snake_case or camelCase part, never
#: inside another word (``flatten`` carries no attenuation).
FACILITY_WORDS = re.compile(
    r"(?i)(?<![a-z])(?:pilot|atten(?:uat\w*)?\b|atten_|bunch[_ ]?clean\w*|"
    r"bpm[_ ]?mode|low[_ ]?current|high[_ ]?current)"
)


def _engine_modules() -> list[Path]:
    return sorted(ENGINES.glob("*.py"))


def test_there_are_engine_modules_to_scan():
    assert any(path.name == "pyat_faults.py" for path in _engine_modules())


@pytest.mark.parametrize("path", _engine_modules(), ids=lambda path: path.name)
def test_an_engine_module_names_no_facility_word(path: Path):
    hits = [
        f"{path.name}:{number}: {line.strip()}"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if FACILITY_WORDS.search(line)
    ]
    assert hits == []


@pytest.mark.parametrize(
    "text",
    ["pilot_tone", "PilotTone", "attenuation", "atten_db", "bunch_cleaning", "BPM mode"],
)
def test_the_pattern_catches_each_word(text: str):
    assert FACILITY_WORDS.search(text)


@pytest.mark.parametrize("text", ["flatten", "pattern", "readout_tone_amplitude", "bpm_read"])
def test_the_pattern_spares_generic_words(text: str):
    assert not FACILITY_WORDS.search(text)
