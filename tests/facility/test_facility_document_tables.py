"""The demo's role views read the ``/4`` document's ``signals`` table as ``/3`` gave them.

The facts view and the channel-finder role descriptions read the document's
``signals`` table as it stands and give what the ``/3`` build gave
(``golden/demo_role_views.json``, captured from that build). The tables
themselves are the description's, and are held in the package suite
(``packages/osprey-facility/tests/osprey_facility_tests/test_facility_document_tables.py``).

The golden is never rewritten to make a test pass. To re-capture it byte for
byte from the ``/3`` build, run from the repo root::

    git worktree add --detach .claude/scratch/role-views 8f22637c06
    cp tests/facility/golden/demo_role_views.json .claude/scratch/role-views/golden.json
    (cd .claude/scratch/role-views && uv run python - <<'PY'
    import json, tempfile
    from pathlib import Path
    from osprey_facility.records import FacilityRecords
    from osprey.facility.views.channel_finder import _signal_descriptions
    from tests._builds import build_demo
    golden = Path("golden.json")
    captured = json.loads(golden.read_text(encoding="utf-8"))
    built = build_demo(Path(tempfile.mkdtemp()))
    facts = built.build_dir / "data" / "facility_facts.json"
    captured["facts"] = json.loads(facts.read_bytes())
    records = FacilityRecords(built.facility)
    captured["channel_finder_signal_descriptions"] = _signal_descriptions(records)
    golden.write_text(json.dumps(captured, indent=2) + "\n", encoding="utf-8")
    PY
    ) && cp .claude/scratch/role-views/golden.json tests/facility/golden/demo_role_views.json
    git worktree remove --force .claude/scratch/role-views
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey_facility.records import FacilityRecords

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

GOLDEN = Path(__file__).resolve().parent / "golden" / "demo_role_views.json"


def _without_configuration_digests(facts: dict[str, Any]) -> dict[str, Any]:
    """``facts`` with each measurement model's configuration digest taken out.

    A pyAML configuration carries numbers pyAT computes, such as ``alphac``,
    whose last digits differ between platforms, so its digest cannot be frozen.
    """
    for model in facts["measurement_models"].values():
        model.pop("sha256")
    return facts


def test_the_demo_role_views_are_those_of_the_base_build(built_demo: BuiltProject) -> None:
    from osprey.facility.views.channel_finder import _signal_descriptions

    reference = json.loads(GOLDEN.read_text(encoding="utf-8"))
    facts = json.loads((built_demo.build_dir / "data" / "facility_facts.json").read_bytes())
    descriptions = _signal_descriptions(FacilityRecords(built_demo.facility))

    for model in facts["measurement_models"].values():
        written = (built_demo.build_dir / model["path"]).read_bytes()
        assert model["sha256"] == hashlib.sha256(written).hexdigest()
    assert _without_configuration_digests(facts) == _without_configuration_digests(
        reference["facts"]
    )
    assert descriptions == reference["channel_finder_signal_descriptions"]
