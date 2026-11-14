"""The measure tool's resolver and the launch-time context read the facts, never the registry.

What a measurement kind is and how it behaves is written into the facts by
``osprey build``. The executor's ``resolve_measure_args`` and the context the
web terminal regenerates at every launch (``config_derived_context``, which
writes the approval hook's measurement block) read that rendered block and
import neither the kind registry (``osprey.facility.kinds``) nor the pyAML view
writer. Both run in a child interpreter with only ``src/`` on its path,
because the test process has imported both modules long before.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from osprey.facility.render import render_facility_outputs
from tests.facility._pyaml_trees import built_document, measured_tree, with_correctors, with_rf

_SRC = str(Path(__file__).resolve().parents[3] / "src")

#: The modules a runtime reader of the kind block never loads.
_BUILD_ONLY = ("osprey.facility.kinds", "osprey.facility.views.pyaml")

_CHILD = """
import json, sys
from pathlib import Path

render = Path(sys.argv[1])
before = set(sys.modules)
from osprey.cli.templates.claude_code import config_derived_context
from osprey.mcp_server.python_executor.tools._pyaml_measure_args import resolve_measure_args

args = resolve_measure_args(
    "orm", "SR", None, None, None, config_path=render / "config.yml", timeout=60
)
context = config_derived_context({"project_name": "demo"}, render)
print(json.dumps({
    "added": sorted(set(sys.modules) - before),
    "kind": args.kind,
    "measurement": sorted(context["measurement"]),
}))
"""


@pytest.fixture(scope="module")
def render(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("no-registry-import")
    tree = with_correctors(with_rf(measured_tree()))
    tree["measurement/SR.yaml"] |= {"kinds": ["orm", "trm"], "corrector_delta": 1.0e-5}
    document, facility = built_document(root / "tree", tree)
    rendered = root / "render"
    rendered.mkdir()
    render_facility_outputs(rendered, document, {}, facility)
    (rendered / "config.yml").write_text("project_name: demo\n", encoding="utf-8")
    return rendered


def test_the_resolver_and_the_launch_context_import_no_registry(render: Path):
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, str(render)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": _SRC},
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])

    assert report["kind"] == "orm"
    assert report["measurement"] == ["LINE", "SR"]
    assert [module for module in report["added"] if module in _BUILD_ONLY] == []
