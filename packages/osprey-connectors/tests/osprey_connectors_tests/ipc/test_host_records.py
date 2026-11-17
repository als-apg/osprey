"""The init payload and the post-connect report are records that cross as the pinned bytes.

:class:`~osprey_connectors.ipc.HostInit` is sent as
``model_dump(exclude_none=True)`` and :class:`~osprey_connectors.ipc.HostReport`
as ``model_dump(by_alias=True)``, so each crosses the pipe as the plain mapping
the frame has always carried: the init frames are the ones
``test_frame_bytes`` pins, and a report is written as the same mapping, keyed
and ordered as the child has always written it.
"""

import pytest

from osprey_connectors.ipc import HostInit, HostReport
from osprey_connectors.ipc import _frames as frames
from osprey_connectors_tests.ipc.test_frame_bytes import _PINNED, _REPORT, _RID, _SECTION, _sent

_CONFIG_FILE = "/srv/project/config.yml"

_INITS = {
    "init": HostInit(control_system=_SECTION, target="live"),
    "init/config_file": HostInit(control_system=_SECTION, target="live", config_file=_CONFIG_FILE),
    "init/execution_mode": HostInit(
        control_system=_SECTION, target="live", execution_mode="readonly"
    ),
    "init/config_file+execution_mode": HostInit(
        control_system=_SECTION,
        target="live",
        config_file=_CONFIG_FILE,
        execution_mode="readonly",
    ),
}

#: The pinned report, with a mode a child can install.
_WIRE_REPORT = {**_REPORT, "mode": "addr_list"}


@pytest.mark.parametrize("name", list(_INITS))
async def test_an_init_record_is_sent_as_the_pinned_init_frame(name, monkeypatch):
    monkeypatch.setattr(frames, "new_request_id", lambda: _RID)
    payload = _INITS[name].model_dump(exclude_none=True)

    sent = await _sent(lambda proxy: proxy._request("init", payload, timeout_s=30.0))

    assert sent == _PINNED[f"request:{name}"]


def test_a_report_is_written_as_the_mapping_the_child_has_always_written():
    report = HostReport.model_validate(_WIRE_REPORT)
    written = report.model_dump(by_alias=True)

    assert list(written) == list(_WIRE_REPORT)
    assert frames.encode_result(_RID, written) == frames.encode_result(_RID, _WIRE_REPORT)


def test_a_report_keeps_the_wire_key_of_epics_configured():
    report = HostReport.model_validate(_WIRE_REPORT)

    assert report.epics_configured is True
    assert "_epics_configured" in report.model_dump(by_alias=True)


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "gateway"},
        {"selected_role": "pva"},
        {"writes_enabled": "yes"},
        {"pid": "4242"},
        {"surprise": True},
    ],
    ids=["mode", "selected_role", "writes_enabled", "pid", "extra-key"],
)
def test_a_reply_that_is_no_report_is_refused(change):
    with pytest.raises(ValueError):
        HostReport.model_validate({**_WIRE_REPORT, **change})


def test_a_reply_missing_a_field_is_refused():
    partial = dict(_WIRE_REPORT)
    del partial["transport"]

    with pytest.raises(ValueError):
        HostReport.model_validate(partial)
