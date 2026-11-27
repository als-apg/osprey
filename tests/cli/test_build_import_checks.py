"""``osprey build`` runs the import checks ``osprey app check`` runs, from one function.

One application breaks each import rule once: an import outside the boundary, a name
``osprey.apps.sdk`` lacks, a name ``osprey_facility.kinds`` lacks, and a module
``osprey_connectors`` lacks. The build, under ``--skip-deps`` as :func:`invoke_build`
always runs it, refuses before its first artifact with all four findings; the check
prints the same four.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.app_cmd import app
from tests.apps._contract import CONTRACT
from tests.cli._lifecycle_build import invoke_build, stub_build

SERVER = (
    "import osprey.cli\n"
    "from osprey.apps.sdk import no_such_name\n"
    "from osprey_facility.kinds import no_such_kind\n"
    "import osprey_connectors.no_such_module\n"
)
FINDINGS = (
    "apps/knob/knob_app/server.py:1: osprey.cli:",
    "apps/knob/knob_app/server.py:2: osprey.apps.sdk.no_such_name:",
    "apps/knob/knob_app/server.py:3: osprey_facility.kinds.no_such_kind:",
    "apps/knob/knob_app/server.py:4: osprey_connectors.no_such_module:",
)
RULES = ("[import_boundary]", "[import_missing_name]", "[import_missing_module]")


def _knob(repo: Path) -> None:
    """One listed application whose package breaks every import rule once."""
    app_dir = repo / "apps" / "knob"
    (app_dir / "knob_app").mkdir(parents=True)
    (app_dir / "app.yml").write_text(
        f"contract: {CONTRACT}\n"
        "name: knob\n"
        "containers:\n  knob-service:\n    runtime: osprey\n"
        "    command: [python, -m, server]\n    health: /health\n",
        encoding="utf-8",
    )
    (app_dir / "knob_app" / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "knob_app" / "server.py").write_text(SERVER, encoding="utf-8")
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    profile["apps"] = {"knob": {}}
    (repo / "profile.yml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _assert_four_findings(output: str) -> None:
    for finding in FINDINGS:
        assert finding in output, output
    for rule in RULES:
        assert rule in output, output
    assert output.count("[import_missing_name]") == 2, output
    assert "4 violation(s)." in output, output


def test_build_refuses_every_import_finding_before_any_artifact(
    lifecycle_repo: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _knob(lifecycle_repo)
    reached: list[bool] = []

    def stop(*_args: Any, **_kwargs: Any) -> None:
        reached.append(True)
        raise RuntimeError("build_deployment_packages was reached")

    def no_vendor(*_args: Any, **_kwargs: Any) -> None:
        reached.append(True)
        raise RuntimeError("fetch_app_vendor was reached")

    monkeypatch.setattr("osprey.cli.build_cmd.build_deployment_packages", stop)
    monkeypatch.setattr("osprey.interfaces.vendor.fetch_app_vendor", no_vendor)
    build = stub_build(lifecycle_repo)
    before = _tree(build)

    result = invoke_build(lifecycle_repo)  # --skip-deps --skip-lifecycle

    assert result.exit_code != 0, result.output
    assert not reached, caplog.text
    assert _tree(build) == before
    _assert_four_findings(result.output + caplog.text)


def test_app_check_reports_the_same_four_findings(runner: CliRunner, lifecycle_repo: Path) -> None:
    _knob(lifecycle_repo)

    result = runner.invoke(app, ["check", "--repo", str(lifecycle_repo)])

    assert result.exit_code == 1, result.output
    _assert_four_findings(result.output)
