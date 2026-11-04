"""The names of a model's pyAML view come from the facility file alone."""

from __future__ import annotations

from importlib import metadata
from typing import Any, NoReturn

import pytest

from osprey.facility.records import FacilityRecords
from osprey.facility.views.pyaml import view_names


def _raiser(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise AssertionError("engine loaded")


def test_view_names_load_no_engine(
    built_control_assistant: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = FacilityRecords(built_control_assistant.facility)
    unpatched = view_names(records, "SR")
    monkeypatch.setattr(metadata, "entry_points", _raiser)
    patched = view_names(records, "SR")
    assert patched.magnets == unpatched.magnets
    assert patched.bpms == unpatched.bpms
    assert patched.arrays == unpatched.arrays
