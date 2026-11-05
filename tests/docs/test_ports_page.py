"""Every ``profile.yml`` example on the ports page loads and moves the base.

``docs/source/reference/ports.rst`` shows how to move a deployment's port
block. Its examples are what a reader pastes into ``profile.yml``, so each one
has to load through the real profile loader and resolve the base the page
states. Two drifts this catches: an example written as a nested
``config: {deployment: {...}}`` block, which loads but replaces the whole
``deployment`` subtree, and a ``dispatch:`` example that leaves out its one
required key, which the loader refuses.

Each block is laid over a minimal profile (the keys every profile file needs)
in a scratch directory and loaded with :func:`load_profile`. A trigger file the
block names is supplied there, copied from the bundled one, because the page
names the reader's own file.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey.profiles.triggers as bundled_triggers
from osprey.cli.build_profile_load import config_port_base, load_profile
from osprey.port_layout import DEFAULT_PORT_BASE

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The page this sweep reads, relative to the repo root.
_PAGE = "docs/source/reference/ports.rst"

#: The base the page's examples move the deployment to.
_STATED_BASE = 20000

#: The keys every profile file needs before it loads.
_MINIMAL_PROFILE: dict[str, Any] = {
    "name": "Ports page example",
    "project_name": "ports-example",
    "data": "data",
    "provider": "anthropic",
}

#: A bundled trigger file to stand in for the reader's own.
_BUNDLED_TRIGGERS = Path(bundled_triggers.__file__).parent / "tutorial_triggers.yml"

_DIRECTIVE = re.compile(r"^\.\. code-block:: yaml\s*$")


def _yaml_blocks() -> list[tuple[int, str]]:
    """``(line number, body)`` for every YAML code block on the page."""
    lines = (_REPO_ROOT / _PAGE).read_text(encoding="utf-8").splitlines()
    blocks: list[tuple[int, str]] = []
    for n, line in enumerate(lines):
        if not _DIRECTIVE.match(line):
            continue
        body: list[str] = []
        for following in lines[n + 1 :]:
            if following.strip() and not following.startswith(" "):
                break
            body.append(following)
        blocks.append((n + 1, "\n".join(body)))
    return blocks


def _load_over_minimal(block: dict[str, Any], tmp_path: Path):
    """Write the block over a minimal profile into *tmp_path* and load it."""
    (tmp_path / "data").mkdir()
    # The page names the reader's own trigger file, which the loader looks for
    # beside the profile; supply it there.
    triggers = (block.get("dispatch") or {}).get("triggers")
    if isinstance(triggers, str):
        shutil.copyfile(_BUNDLED_TRIGGERS, tmp_path / triggers)
    profile = tmp_path / "profile.yml"
    profile.write_text(yaml.safe_dump({**_MINIMAL_PROFILE, **block}), encoding="utf-8")
    return load_profile(profile)


def test_the_page_has_yaml_examples() -> None:
    """Pin the parse, so a moved page fails here rather than passing vacuously."""
    assert len(_yaml_blocks()) >= 2, f"expected the two profile.yml examples on {_PAGE}"


_BLOCKS = _yaml_blocks()


@pytest.mark.parametrize(("line", "body"), _BLOCKS, ids=[f"block{n}" for n in range(len(_BLOCKS))])
def test_each_example_loads_and_resolves_the_stated_base(
    line: int, body: str, tmp_path: Path
) -> None:
    """The loader accepts the example, and the deployment lands on its base."""
    block = yaml.safe_load(body)
    assert isinstance(block, dict), f"{_PAGE}:{line} is not a profile.yml mapping"
    config = block.get("config") or {}
    nested = sorted(k for k, v in config.items() if "." not in k and isinstance(v, dict))
    assert not nested, (
        f"{_PAGE}:{line} writes config key(s) {nested} as nested YAML; a nested "
        "block replaces the whole subtree, so write dotted keys"
    )

    profile = _load_over_minimal(block, tmp_path)

    assert _STATED_BASE != DEFAULT_PORT_BASE
    assert config_port_base(profile.config) == _STATED_BASE, (
        f"{_PAGE}:{line} does not move the deployment to {_STATED_BASE}"
    )
    if "dispatch" in block:
        assert profile.dispatch is not None
        assert profile.dispatch.worker_port_stride == block["dispatch"]["worker_port_stride"]
