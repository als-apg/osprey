"""A ``bind.state_fields`` key may be a dotted path into the state the backend returns.

``step.digest`` binds ``state["step"]["digest"]``. The approval hook resolves the
key when it builds the stamp, and the tool server's ``check_approved_call`` hands
back the stamped value under the field's name. A path that crosses a value that
is not a mapping leaves the field unbound, as an absent key does.
"""

from __future__ import annotations

import pytest

from osprey.apps.manifest import StateField
from osprey.apps.sdk import check_approved_call
from tests.apps import test_sdk_check_approved_call as approved_call
from tests.apps.test_sdk_check_approved_call import DIGEST, FULL_NAME, TOOL, VALUE
from tests.hooks.conftest import import_hook

#: The rendered deployment the approved-call tests run in, a fixture.
deployment = approved_call.deployment

FIELD = "approved_step"
STATE = {"step": {"digest": "abc123", "index": 3}, "readback": 4.0, "mode": "idle"}


@pytest.fixture(scope="module")
def approval():
    return import_hook("osprey_approval")


def _entry(key: str) -> dict:
    return {"bind": {"state_fields": [{"field": FIELD, "key": key}]}}


def _bound(approval, key: str, state=STATE) -> dict:
    return approval._app_binding(_entry(key), {}, state, "va")


def test_the_manifest_takes_a_dotted_key():
    assert StateField(field=FIELD, key="step.digest").key == "step.digest"


def test_a_nested_key_binds_the_nested_value(approval):
    assert _bound(approval, "step.digest")[FIELD] == "abc123"
    assert _bound(approval, "step.index")[FIELD] == 3


def test_a_flat_key_still_binds(approval):
    assert _bound(approval, "readback")[FIELD] == 4.0


@pytest.mark.parametrize("key", ["mode.digest", "readback.value", "step.missing", "absent.x"])
def test_a_path_through_a_scalar_or_a_gap_is_unbound(approval, key):
    assert FIELD not in _bound(approval, key)


def test_a_nested_key_naming_a_mapping_is_refused(approval):
    with pytest.raises(approval._AppRefused):
        _bound(approval, "step")


def test_a_nested_binding_checks_on_the_tool_server(approval, deployment):
    config = approved_call._config()
    entry = config["apps"][approved_call.APP]["approval"][TOOL]
    entry["bind"]["state_fields"] = [{"field": FIELD, "key": "step.digest"}]
    deployment.write_config(config)

    stamped = _bound(approval, "step.digest")[FIELD]
    with deployment.stamps.approved(
        FULL_NAME,
        approver="human",
        approved_target="va",
        approved_digest=DIGEST,
        **{FIELD: stamped},
    ):
        state = check_approved_call(
            TOOL,
            {**VALUE, "approved_target": "va", "approved_digest": DIGEST, FIELD: stamped},
        )
    assert state.state_fields == {FIELD: "abc123"}
