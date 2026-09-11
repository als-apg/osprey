"""The entry form's field names and their readers, pinned against each other.

Two failure modes live here, and neither raises anything at runtime:

* the form input and the JavaScript that reads it can drift apart. The submit
  handler builds its payload with ``formData.get(<name>)``, so a renamed input
  simply yields ``null`` — credentials stop being sent and the publish fails
  with a 401 that names nothing; and
* a fixed ``<option>`` list can encode one facility's vocabulary in a field the
  API takes as free text. An operator whose site runs different shifts or files
  under different logbook names then cannot enter the value at all.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_STATIC = Path(__file__).resolve().parents[3] / "src/osprey/interfaces/ariel/static"


@pytest.fixture(scope="module")
def index_html() -> str:
    return (_STATIC / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def entries_form_js() -> str:
    return (_STATIC / "js" / "entries-form.js").read_text(encoding="utf-8")


#: Credential fields, as the multipart endpoint's ``Form(...)`` parameters spell
#: them. The form must post these names for the values to arrive at all.
CREDENTIAL_FIELDS = ("auth_user", "auth_password")


@pytest.mark.parametrize("field", CREDENTIAL_FIELDS)
def test_the_credential_input_is_named_as_the_api_reads_it(field, index_html):
    assert f'name="{field}"' in index_html, (
        f"no input posts {field}; the endpoint's Form parameter goes unfilled"
    )


@pytest.mark.parametrize("field", CREDENTIAL_FIELDS)
def test_the_submit_handler_reads_the_name_the_form_posts(field, entries_form_js):
    assert f"formData.get('{field}')" in entries_form_js, (
        f"the submit handler does not read {field}; a renamed input yields null silently"
    )


def test_no_vendor_named_credential_field_survives(index_html, entries_form_js):
    """The names were one logbook product's. Every adapter fills the same fields."""
    for source, text in (("index.html", index_html), ("entries-form.js", entries_form_js)):
        assert "olog_" not in text, f"{source} still names one logbook product in a field name"


@pytest.mark.parametrize("field_id", ("entry-shift", "entry-logbook"))
def test_the_free_form_fields_are_not_fixed_option_lists(field_id, index_html):
    """The API takes both as free text; the form must not be narrower than the API."""
    match = re.search(rf"<(\w+)[^>]*\bid=\"{field_id}\"", index_html)
    assert match, f"{field_id} is not in the form any more"
    assert match.group(1) == "input", (
        f"{field_id} is a <{match.group(1)}>; a fixed option list locks out a facility "
        f"whose shift or logbook names differ from the ones shipped"
    )
