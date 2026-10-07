"""``osprey sim apply`` reaches a running mock on its next read.

The mock serves the build's simulator view through the composite, which reads
the active scenario set from the state file ``sim apply`` writes. A composite
built before the apply therefore serves the newly active scenario on its very
next operation, with no restart and no rebuild.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.sim import sim_group
from osprey_connectors.control_system.mock_connector import simulation_state_dir
from osprey_connectors.simulation.composite import Composite
from tests._builds import BuiltProject
from tests.cli._lifecycle_build import stub_build
from tests.fixtures.lifecycle_repo import build_exemplar_repo

#: The BPM whose polarity ``bpm-polarity`` inverts, on the plane the test reads.
BPM_Y = "SR:DIAG:BPM:17:POSITION:Y"

#: A fixed clock, so two reads of one channel differ only by the active scenario.
T0 = 1_760_000_000.0


@pytest.fixture(autouse=True)
def _contain_env_written_by_the_cli():
    """Keep what ``sim apply`` loads into the environment inside the test that ran it."""
    before = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(before)


def _stage(built: BuiltProject, tmp_path: Path) -> tuple[Path, Path]:
    """A deployment repo whose render carries the demo's simulator view.

    Returns:
        The repo and its simulator view.
    """
    repo = build_exemplar_repo(tmp_path / "repo")
    config = {
        "control_system": {
            "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}}
        },
    }
    build = stub_build(repo, config=yaml.safe_dump(config))
    prefix = "data/simulator/"
    for name, data in built.outputs[0].files.items():
        if name.startswith(prefix):
            target = build / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return repo, build / "data" / "simulator"


@pytest.mark.slow
def test_bpm_polarity_flips_the_reading_on_the_next_operation(
    built_control_assistant: BuiltProject, tmp_path: Path
) -> None:
    repo, view = _stage(built_control_assistant, tmp_path)
    mock = Composite(view, state_dir=simulation_state_dir(view), clock=lambda: T0, model_log=False)
    before = mock.get(BPM_Y)

    result = CliRunner().invoke(
        sim_group, ["apply", "--repo", str(repo), "bpm-polarity", "--no-seed", "--yes"]
    )

    assert result.exit_code == 0, result.output
    assert before != 0.0
    assert mock.get(BPM_Y) == pytest.approx(-before)
