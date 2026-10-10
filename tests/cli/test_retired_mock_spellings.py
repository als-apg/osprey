"""Every spelling of the retired ``mock`` type is refused, naming the new one.

The simulator in process is ``control_system.type: virtual_accelerator`` with
``control_system.connector.virtual_accelerator.serving: in_process``. A profile,
an ``osprey set`` line or a connector build that still says ``mock`` is refused
at the site it reaches first, and each refusal quotes the same sentence, so an
operator reads one fix wherever they meet it. The ``serving`` leaf is refused
too where no reader can honour it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_profile import _parse_profile
from osprey.cli.build_profile_deploy import limits_block_errors
from osprey.cli.set_cmd import set as set_command
from osprey.connectors.types import retired_type_message
from osprey.errors import BuildProfileError

#: The fix every refusal of the retired name names.
NEW_SPELLING = "control_system.connector.virtual_accelerator.serving: in_process"


def test_the_connector_shorthand_is_refused() -> None:
    with pytest.raises(BuildProfileError) as caught:
        _parse_profile({"name": "x", "data": "data", "connector": "mock"})

    assert str(caught.value) == retired_type_message("mock")
    assert NEW_SPELLING in str(caught.value)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"control_system.type": "mock"}, id="dotted"),
        pytest.param({"control_system": {"type": "mock"}}, id="nested"),
    ],
)
def test_a_retired_type_value_is_refused(config: dict) -> None:
    errors = limits_block_errors(config)

    assert len(errors) == 1
    assert retired_type_message("mock") in errors[0]


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            {
                "control_system.type": "virtual_accelerator",
                "control_system.connector.mock.response_delay_ms": 10,
            },
            id="dotted-leaf",
        ),
        pytest.param(
            {"control_system": {"type": "epics", "connector": {"mock": {}, "epics": {}}}},
            id="empty-nested-block",
        ),
    ],
)
def test_a_leftover_retired_block_is_refused(config: dict) -> None:
    """A leftover block would count as a real machine when ``live`` is derived."""
    errors = limits_block_errors(config)

    assert len(errors) == 1
    assert "`control_system.connector.mock` block" in errors[0]
    assert NEW_SPELLING in errors[0]


def test_a_serving_value_outside_the_two_venues_is_refused() -> None:
    errors = limits_block_errors(
        {
            "control_system.type": "virtual_accelerator",
            "control_system.connector.virtual_accelerator.serving": "container",
        }
    )

    assert len(errors) == 1
    assert "'container'" in errors[0]
    assert "served | in_process" in errors[0]


def test_in_process_on_a_deployment_of_another_type_is_refused() -> None:
    """On an ``epics`` deployment ``va`` is the served container (and the leaf
    would otherwise flip the host child that states its own type)."""
    errors = limits_block_errors(
        {
            "control_system.type": "epics",
            "control_system.connector.virtual_accelerator.serving": "in_process",
        }
    )

    assert len(errors) == 1
    assert "on this deployment `va` is the served container" in errors[0]


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            {
                "control_system.type": "virtual_accelerator",
                "control_system.connector.virtual_accelerator.serving": "in_process",
            },
            id="in-process",
        ),
        pytest.param({"control_system.type": "virtual_accelerator"}, id="served"),
        pytest.param(
            {"control_system.connector.virtual_accelerator.serving": "in_process"}, id="no-type"
        ),
    ],
)
def test_the_new_spellings_pass(config: dict) -> None:
    assert limits_block_errors(config) == []


@pytest.mark.parametrize(
    "pair",
    ["connector=mock", "config.control_system.type=mock"],
)
def test_osprey_set_refuses_the_retired_type_before_writing(tmp_path: Path, pair: str) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    profile = repo / "profile.yml"
    profile.write_text("name: x\ndata: data\n", encoding="utf-8")

    result = CliRunner().invoke(set_command, ["--repo", str(repo), pair])

    assert result.exit_code != 0
    assert NEW_SPELLING in result.output
    assert profile.read_text(encoding="utf-8") == "name: x\ndata: data\n"
