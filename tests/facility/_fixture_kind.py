"""The fixture kind's deployment package, the facility tree that lists it, and its install.

``fixtures/fixture_kind/`` holds a deployment package that registers the
measurement kind ``fixture_kind`` and the engine ``fixture_ring`` through their
entry-point groups. Every test that builds a deployment of it reads the package,
the tree and the wheel stand-in from here, so they cannot drift apart.
"""

from __future__ import annotations

import copy
import importlib
import sys
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from osprey.apps.sdk.testing import measured_tree, model, with_correctors, with_rf

#: The deployment package's directory: its import package and its dist-info.
PACKAGE_DIR = Path(__file__).resolve().parent / "fixtures" / "fixture_kind"

#: The kind, its engine and the distribution that declares both.
KIND = "fixture_kind"
ENGINE = "fixture_ring"
DISTRIBUTION = "fixture-kind-deployment"

#: Where the deployment keeps the package's source, as its profile names it.
SOURCE = "./kinds/fixture_kind"

#: The package's ``pyproject.toml`` in the deployment.
PYPROJECT = f"""\
[project]
name = "{DISTRIBUTION}"
version = "0.1"
dependencies = []

[project.entry-points."osprey.facility.kinds"]
{KIND} = "fixture_kind_deployment:FIXTURE_KIND"

[project.entry-points."osprey.simulation.engines"]
{ENGINE} = "fixture_kind_deployment.engine"
"""

#: The facility-added signal role the bunch-clean enable carries.
FIXTURE_ROLE = {
    "name": "fixture_role",
    "aliases": ["bunch clean enable"],
    "description": "Enables the bunch cleaning ahead of a shot.",
}

#: The instruments the measurement file names, each with its channel; the RF status is absent.
INSTRUMENTS = {
    "inject_request": "INJ:REQ",
    "inject_busy": "INJ:BUSY",
    "clean_enable": "CLEAN:EN",
    "beam_current": "BEAM:I",
}

#: The kind's sub-block: a trigger sample, two ordered settings, a chain of three steps.
BLOCK: dict[str, Any] = {
    "trigger_sample": 12,
    "settings": [
        {
            "id": "clean",
            "label": "Bunch clean",
            "description": "Bunch cleaning per device",
            "scope": "device",
            "signal": "fixture_role",
            "timeout_s": 2.0,
        },
        {
            "id": "current",
            "label": "Beam current",
            "description": "The stored current the shot aims for",
            "scope": "global",
            "address": "BEAM:I",
            "timeout_s": 2.0,
        },
    ],
    "steps": [
        {"id": "settle", "label": "Settle", "description": "Wait for the ring", "default": True},
        {
            "id": "inject",
            "label": "Inject",
            "description": "Request one shot",
            "default": True,
            "command": {
                "address": "INJ:REQ",
                "value": 1,
                "timeout_s": 5.0,
                "done": {"address": "INJ:BUSY", "value": 0},
            },
        },
        {
            "id": "clean",
            "label": "Clean",
            "description": "Clean the bunches",
            "default": False,
            "command": {"address": "INJ:REQ", "value": 2, "timeout_s": 5.0},
        },
    ],
}

#: What the kind's facts writer writes from ``BLOCK``.
FACTS = {
    "chain": ["settle", "inject", "clean"],
    "commands": ["INJ:REQ", "INJ:REQ"],
    "trigger_sample": 12,
    "settings": ["clean", "current"],
}


def read_entry_points_afresh() -> None:
    """Drop the suite's once-per-session reading of the entry points, if it holds one."""
    getattr(metadata.entry_points, "cache_clear", lambda: None)()


def install(monkeypatch: pytest.MonkeyPatch, *sites: Path) -> None:
    """Put each site and its dist-info on ``sys.path`` for the test's duration."""
    for site in sites:
        monkeypatch.syspath_prepend(str(site))
    for module in [name for name in sys.modules if name.startswith("fixture_kind_")]:
        monkeypatch.delitem(sys.modules, module)
    importlib.invalidate_caches()
    read_entry_points_afresh()


def fake_wheel(source: Path, out_dir: Path) -> Path:
    """Stand in for the wheel build: the package and its entry points, zipped as a wheel."""
    wheel = out_dir / "fixture_kind_deployment-0.1-py3-none-any.whl"
    info = "fixture_kind_deployment-0.1.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path in sorted((source / "fixture_kind_deployment").glob("*.py")):
            archive.write(path, path.relative_to(source).as_posix())
        for name in ("METADATA", "entry_points.txt"):
            archive.write(PACKAGE_DIR / info / name, f"{info}/{name}")
    return wheel


def fixture_tree(block: dict[str, Any] | None = None) -> dict[str, Any]:
    """The measured tree with SR on the fixture engine and the fixture kind listed.

    SR keeps ``trm``; LINE stays on pyat with the built-in ``orm``. SR gains the
    instruments' channels, the bunch-clean enable carrying ``fixture_role``, and
    the injection request wired as a record the engine fires.
    """
    tree = with_correctors(with_rf(measured_tree()))
    tree["signals.yaml"] = [dict(FIXTURE_ROLE)]
    tree["records/channels.yaml"] += [
        {"id": "INJ:REQ", "role": "setpoint", "on": {"place": "SR"}},
        {"id": "INJ:BUSY", "on": {"place": "SR"}},
        {"id": "CLEAN:EN", "role": "setpoint", "signal": "fixture_role", "on": {"place": "SR"}},
        {"id": "BEAM:I", "unit": "mA", "on": {"place": "SR"}},
    ]
    sr = model(tree, "SR")
    sr["engine"] = ENGINE
    sr["wiring"].append({"address": "INJ:REQ", "engine": {"fire": "inject"}})
    measurement = tree["measurement/SR.yaml"]
    measurement["kinds"] = ["trm", KIND]
    measurement["groups"]["series_monitors"] = "SR/BPM"
    measurement["instruments"] |= INSTRUMENTS
    measurement |= {
        "corrector_delta": 1.0e-5,
        "kind_blocks": {KIND: copy.deepcopy(BLOCK if block is None else block)},
    }
    return tree
