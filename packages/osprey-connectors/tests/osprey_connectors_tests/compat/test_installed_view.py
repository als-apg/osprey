"""The view contract skips a release with no view writer only where it may."""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey_connectors_tests.compat import contracts


def test_a_release_without_the_writer_is_not_applicable(tmp_path: Path) -> None:
    summary = contracts.check_installed_view(tmp_path, tmp_path, require_view=False)
    assert summary.startswith("simulator view: not applicable for this release")


def test_a_release_that_must_write_the_view_and_ships_no_writer_breaks(tmp_path: Path) -> None:
    with pytest.raises(contracts.ContractError, match="must write the view"):
        contracts.check_installed_view(tmp_path, tmp_path, require_view=True)


@pytest.mark.parametrize("require_view", [False, True])
def test_a_release_with_the_writer_is_always_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, require_view: bool
) -> None:
    writer = tmp_path / contracts.VIEW_WRITER_RELPATH
    writer.parent.mkdir(parents=True)
    writer.touch()
    monkeypatch.setattr(contracts, "_osprey_cli", lambda: Path("osprey"))
    monkeypatch.setattr(contracts, "render_deployment", lambda workdir, cli: workdir / "build")
    with pytest.raises(contracts.ContractError, match="NoSimulatorView"):
        contracts.check_installed_view(tmp_path, tmp_path, require_view=require_view)
