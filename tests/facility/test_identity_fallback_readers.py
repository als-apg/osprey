"""Every reader that falls back to the facility file's identity refuses a ``/3`` file.

A render without a usable ``data/facility_facts.json`` reads its facts from the
identity its facility file records, and the build and deploy verbs take the
facility's display name from the same record. A facility file whose header this
reader does not read is handled exactly like one that cannot be read: each
caller refuses in its own channel, and the refusal names the rebuild.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

from osprey.apps import sdk
from osprey.facility.views.facts import read_facts
from osprey_facility.errors import FacilityFileError
from tests._facility_file import write_facility_document

pytestmark = pytest.mark.foreign_header

V3 = "osprey.facility.facility/3"


def _expected(path: Path) -> str:
    return (
        f"{path} is osprey.facility.facility/3; this reader reads "
        "osprey.facility.facility/4 -- rebuild with osprey build"
    )


@pytest.fixture
def v3_render(tmp_path: Path) -> tuple[Path, Path]:
    """A render with no facts file and a ``/3`` facility file; (root, facility file)."""
    path = write_facility_document(tmp_path, {"identity": {"code": "lab"}}, schema=V3)
    return tmp_path, path


def test_read_facts_without_a_facts_file_refuses_a_v3_facility_file(v3_render) -> None:
    root, path = v3_render

    with pytest.raises(FacilityFileError) as raised:
        read_facts(root, "lab")

    assert str(raised.value) == _expected(path)


def test_the_pyaml_view_probe_refuses_a_v3_facility_file(v3_render) -> None:
    from osprey.mcp_server.python_executor.server import pyaml_view_present

    root, path = v3_render

    with pytest.raises(FacilityFileError) as raised:
        pyaml_view_present(root)

    assert str(raised.value) == _expected(path)


def test_the_measure_tool_registration_refuses_a_v3_facility_file(v3_render) -> None:
    from osprey.mcp_server.python_executor.tools.pyaml_measure import advertised

    root, path = v3_render

    with pytest.raises(FacilityFileError) as raised:
        advertised(root)

    assert str(raised.value) == _expected(path)


def test_the_measure_arguments_refuse_a_v3_facility_file_as_a_validation_error(
    v3_render,
) -> None:
    from osprey.mcp_server.python_executor.tools._pyaml_measure_args import (
        resolve_measure_args,
    )

    root, path = v3_render

    with pytest.raises(ToolError) as raised:
        resolve_measure_args(
            "orbit", None, None, None, None, config_path=root / "config.yml", timeout=30.0
        )

    envelope = json.loads(str(raised.value))
    assert envelope["error_type"] == "validation_error"
    assert envelope["error_message"] == _expected(path)
    assert any("osprey build" in step for step in envelope["suggestions"])


def test_the_sdk_read_facts_refuses_a_v3_facility_file_as_a_value_error(v3_render) -> None:
    root, path = v3_render

    with pytest.raises(ValueError) as raised:
        sdk.read_facts(root)

    assert str(raised.value) == _expected(path)


def test_the_closed_orbit_models_refuse_a_v3_facility_file_as_a_value_error(v3_render) -> None:
    root, path = v3_render

    with pytest.raises(ValueError) as raised:
        sdk.closed_orbit_models(root)

    assert str(raised.value) == _expected(path)


def test_the_dispatch_build_step_refuses_a_v3_facility_file(tmp_path: Path) -> None:
    from ruamel.yaml import YAML

    from osprey.cli.build_cmd import _inject_dispatch
    from osprey.cli.build_profile import DispatchConfig

    project = tmp_path / "project"
    project.mkdir()
    profile = tmp_path / "profile"
    profile.mkdir()
    YAML().dump(
        {"project_name": "lab", "deployed_services": [], "services": {}}, project / "config.yml"
    )
    path = write_facility_document(project, {"identity": {"code": "lab"}}, schema=V3)
    dispatch = DispatchConfig(
        triggers="tutorial_triggers.yml",
        worker_count=1,
        workspace_mode="isolated",
        max_concurrent_runs=2,
        max_queue_depth=50,
        dispatcher_port=8020,
        worker_port_base=9190,
        timeout_sec=300,
        channel_strip_prefix="ERF:",
    )

    with pytest.raises(FacilityFileError) as raised:
        _inject_dispatch(dispatch, profile_dir=profile, project_path=project)

    assert str(raised.value) == _expected(path)
