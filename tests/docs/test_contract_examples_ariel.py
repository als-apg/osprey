"""The ARIEL contract page's JSON shapes are read by the models that produce them.

``docs/source/reference/contracts/ariel.rst`` shows the web API's responses as
files under ``docs/source/reference/contracts/examples/`` (``ariel-*.json``),
included with ``literalinclude``. Each file is checked against the code that
writes that shape:

* the three entry-field error envelopes are the bodies
  ``_entry_field_error_response`` returns for the exception each one names,
  with the status the page gives;
* the health, search and status responses validate as their pydantic models
  and serialize back to themselves. The models accept unknown keys and fill
  missing defaults, so the round trip, not ``model_validate`` alone, is what
  shows a key the model does not have or a field the example leaves out;
* the status response's fields are ``ARIELStatusResult``'s, and its embedding
  table rows are ``EmbeddingTableInfo``'s, both ways;
* the capabilities document is what ``get_capabilities`` returns for the
  configuration the page names;
* the attachment summary field list on the page is ``SUMMARY_KEYS``, with
  ``?`` on exactly the keys a summary may leave out.

That every ``examples/`` file is on a page is checked by
``test_contract_examples.py``.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "docs/source/reference/contracts/ariel.rst"
EXAMPLES = REPO_ROOT / "docs/source/reference/contracts/examples"


def _example(name: str) -> Any:
    return json.loads((EXAMPLES / name).read_text())


def _round_trips(model: type[BaseModel], document: dict[str, Any]) -> bool:
    """Whether ``document`` validates as ``model`` and serializes back unchanged."""
    return model.model_validate(document).model_dump(mode="json") == document


# --- entry-field error envelopes -----------------------------------------------------


def _invalid_value() -> Exception:
    from osprey.services.ariel_search.entry_fields import coerce_entry_value

    try:
        coerce_entry_value(_BOOK, "engineering")
    except Exception as exc:
        return exc
    raise AssertionError("an undeclared choice was accepted")


def _options_unavailable() -> Exception:
    from osprey.services.ariel_search.entry_fields import EntryFieldOptionsUnavailable

    return EntryFieldOptionsUnavailable("scan")


def _misdeclared() -> Exception:
    from osprey.services.ariel_search.entry_fields import check_declarations

    try:
        check_declarations([_BOOK, _BOOK], adapter="Generic JSON")
    except Exception as exc:
        return exc
    raise AssertionError("a field declared twice was accepted")


def _book() -> Any:
    from osprey.services.ariel_search.search.base import ParameterDescriptor

    return ParameterDescriptor(
        name="book",
        label="Logbook",
        description="The logbook the entry goes to",
        param_type="select",
        default=None,
        options=[
            {"value": "operations", "label": "Operations"},
            {"value": "physics", "label": "Physics"},
        ],
    )


_BOOK = _book()


@pytest.mark.parametrize(
    ("name", "raise_it", "status"),
    [
        ("ariel-entry-field-invalid.json", _invalid_value, 422),
        ("ariel-entry-field-options-unavailable.json", _options_unavailable, 502),
        ("ariel-entry-fields-misdeclared.json", _misdeclared, 500),
    ],
)
def test_entry_field_envelope_is_the_routes_body(name: str, raise_it: Any, status: int) -> None:
    from osprey.interfaces.ariel.api.routes import _entry_field_error_response

    response = _entry_field_error_response(raise_it())
    assert _example(name) == json.loads(response.body)
    assert response.status_code == status
    after = PAGE.read_text().split(f"examples/{name}", 1)[1]
    stated = re.search(r"^\s+is (\d{3}):", after, re.MULTILINE)
    assert stated, f"the page gives {name} no status"
    assert int(stated.group(1)) == status


def test_options_timeout_on_the_page_is_the_code() -> None:
    from osprey.services.ariel_search.entry_fields import OPTIONS_TIMEOUT_SECONDS

    assert f"longer than {OPTIONS_TIMEOUT_SECONDS:g} seconds" in " ".join(PAGE.read_text().split())


# --- health, search and status responses ---------------------------------------------


def test_health_response_round_trips() -> None:
    from osprey.interfaces.ariel.api.schemas import HealthResponse
    from osprey.interfaces.ariel.app import CONFIG_STATUS_OK, HEALTH_OK

    example = _example("ariel-health-response.json")
    assert _round_trips(HealthResponse, example)
    assert example["status"] == "healthy"
    assert example["message"] == HEALTH_OK
    assert example["config_status"] == CONFIG_STATUS_OK


def test_search_response_round_trips() -> None:
    from osprey.interfaces.ariel.api.schemas import SearchResponse

    example = _example("ariel-search-response.json")
    assert _round_trips(SearchResponse, example)
    assert len(example["search_modes_used"]) == 1, "a completed search names the one mode it ran"
    assert example["total_results"] == len(example["entries"])


def test_status_response_round_trips_and_is_the_status_result() -> None:
    from osprey.interfaces.ariel.api.schemas import EmbeddingTableStatus, StatusResponse
    from osprey.services.ariel_search.models import ARIELStatusResult, EmbeddingTableInfo

    example = _example("ariel-status-response.json")
    assert _round_trips(StatusResponse, example)
    assert list(example) == [f.name for f in dataclasses.fields(ARIELStatusResult)]
    assert list(StatusResponse.model_fields) == [
        f.name for f in dataclasses.fields(ARIELStatusResult)
    ]
    table_fields = [f.name for f in dataclasses.fields(EmbeddingTableInfo)]
    assert list(EmbeddingTableStatus.model_fields) == table_fields
    assert all(list(row) == table_fields for row in example["embedding_tables"])


@pytest.mark.parametrize(
    ("name", "model_name"),
    [
        ("ariel-health-response.json", "HealthResponse"),
        ("ariel-search-response.json", "SearchResponse"),
        ("ariel-status-response.json", "StatusResponse"),
    ],
)
def test_a_planted_key_breaks_the_round_trip(name: str, model_name: str) -> None:
    from osprey.interfaces.ariel.api import schemas

    model = getattr(schemas, model_name)
    planted = {**_example(name), "not_a_field": 1}
    assert not _round_trips(model, planted)
    missing = _example(name)
    missing.pop(next(key for key in missing if not model.model_fields[key].is_required()))
    assert not _round_trips(model, missing)


def test_an_entry_without_its_required_times_is_refused() -> None:
    from osprey.interfaces.ariel.api.schemas import EntryResponse

    entry = _example("ariel-search-response.json")["entries"][0]
    for key in ("timestamp", "created_at", "updated_at"):
        with pytest.raises(ValidationError):
            EntryResponse.model_validate({k: v for k, v in entry.items() if k != key})


# --- capabilities --------------------------------------------------------------------

#: The configuration the page names for its capabilities example.
_CAPABILITIES_ARIEL = {
    "database": {"uri": "postgresql://localhost:5432/ariel"},
    "search_modules": {
        "keyword": {"enabled": True},
        "semantic": {"enabled": True, "model": "nomic-embed-text"},
    },
}


def test_capabilities_is_what_get_capabilities_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osprey.registry import get_registry
    from osprey.services.ariel_search.capabilities import SHARED_PARAMETERS, get_capabilities
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.search import image_lane

    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump({"ariel": _CAPABILITIES_ARIEL}))
    get_registry(config_path=str(config_file))
    # Process state: the reason an earlier picture search in this process failed.
    monkeypatch.setattr(image_lane, "last_unavailable_reason", lambda: None)

    example = _example("ariel-capabilities.json")
    assert example == get_capabilities(ARIELConfig.from_dict(_CAPABILITIES_ARIEL)), (
        "regenerate examples/ariel-capabilities.json: print "
        "json.dumps(get_capabilities(ARIELConfig.from_dict(_CAPABILITIES_ARIEL)), indent=2) "
        "after get_registry(config_path=<a config.yml holding {'ariel': _CAPABILITIES_ARIEL}>)"
    )
    assert [p["name"] for p in example["shared_parameters"]] == [p.name for p in SHARED_PARAMETERS]
    assert not example["vocabulary"]["enabled"], "the page says the example has no vocabulary"
    modes = [mode["name"] for mode in example["categories"]["direct"]["modes"]]
    assert modes == ["keyword", "semantic"]
    page = " ".join(PAGE.read_text().split())
    assert "with the ``keyword`` and ``semantic`` modes enabled and no vocabulary" in page


# --- attachment summary field list ---------------------------------------------------

_SUMMARY_BLOCK = re.compile(r"\{(attachment_id\??,[^}]*)\}")


def _page_summary_fields() -> list[str]:
    match = _SUMMARY_BLOCK.search(PAGE.read_text())
    assert match, "the attachment summary block is gone from ariel.rst"
    return [name.strip() for name in match.group(1).split(",")]


def _minimal_summary_keys() -> list[str]:
    """The keys of a summary built from the least an attachment item can carry."""
    from osprey.services.ariel_search.attachments.summaries import (
        build_attachment_summary_pairs,
    )

    entry = {"entry_id": "1", "attachments": [{"filename": "beam.png", "type": "image/png"}]}
    pairs = build_attachment_summary_pairs(entry, None, None, (), file_source=False)
    return list(pairs[0][0])


def _summary_problems(fields: list[str]) -> list[str]:
    from osprey.services.ariel_search.attachments.summaries import SUMMARY_KEYS

    problems = []
    names = [name.removesuffix("?") for name in fields]
    if names != list(SUMMARY_KEYS):
        problems.append(f"names {names} are not SUMMARY_KEYS {list(SUMMARY_KEYS)}")
    always = [name for name in fields if not name.endswith("?")]
    if always != _minimal_summary_keys():
        problems.append(f"unmarked {always} are not the always-present {_minimal_summary_keys()}")
    return problems


def test_attachment_summary_fields_are_the_summary_keys() -> None:
    assert _summary_problems(_page_summary_fields()) == []


def test_a_wrong_summary_field_list_is_caught() -> None:
    fields = _page_summary_fields()
    assert _summary_problems([*fields, "display_url?"])
    assert _summary_problems([name.removesuffix("?") for name in fields])


def test_the_web_attachment_adds_two_fields_to_the_summary() -> None:
    """The page's list is the agent's summary; the web panel's model adds two fields."""
    from osprey.interfaces.ariel.api.schemas import AttachmentResponse
    from osprey.services.ariel_search.attachments.summaries import SUMMARY_KEYS

    assert list(AttachmentResponse.model_fields) == [
        *SUMMARY_KEYS,
        "display_url",
        "skip_reason_text",
    ]
