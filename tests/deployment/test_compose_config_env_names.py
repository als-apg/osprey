"""Every compose template that names its config as ``CONFIG_FILE`` names it as ``OSPREY_CONFIG`` too.

The connectors package locates a process's config by ``OSPREY_CONFIG`` alone
(:func:`osprey_connectors.config.locate_config`); some framework readers still
read ``CONFIG_FILE``. A container handed only ``CONFIG_FILE`` would have its
connector lookups fall back to the working directory, which in these images
holds no config, and the limits validator would then come up as no validator:
limits checking off, without an error. So a service that states one states the
other, with the same value, in the same environment block.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TEMPLATES = Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates"

#: One environment entry: ``KEY: value`` in a mapping, ``- KEY=value`` in a list.
_ENTRY = re.compile(r"^\s*(?:-\s*)?(CONFIG_FILE|OSPREY_CONFIG)\s*[:=]\s*(\S.*?)\s*$")

#: The start of a compose service's ``environment:`` block.
_ENVIRONMENT = re.compile(r"^(\s*)environment:\s*$")


def _blocks(text: str) -> list[dict[str, list[str]]]:
    """The two variables' values in each ``environment:`` block of a template, in order."""
    blocks: list[dict[str, list[str]]] = []
    indent: int | None = None
    for line in text.splitlines():
        opened = _ENVIRONMENT.match(line)
        if opened:
            indent = len(opened.group(1))
            blocks.append({"CONFIG_FILE": [], "OSPREY_CONFIG": []})
            continue
        if indent is None or not line.strip() or line.lstrip().startswith(("#", "{%", "{#", "{{")):
            continue
        if len(line) - len(line.lstrip()) <= indent:
            indent = None
            continue
        entry = _ENTRY.match(line)
        if entry:
            blocks[-1][entry.group(1)].append(entry.group(2))
    return blocks


def _compose_templates() -> list[Path]:
    return sorted(TEMPLATES.rglob("docker-compose*.j2"))


def test_the_scan_sees_the_templates_that_set_config_file():
    """The scan finds the blocks it judges, so an empty pass cannot read as a clean one."""
    setting = [
        path
        for path in _compose_templates()
        if any(block["CONFIG_FILE"] for block in _blocks(path.read_text(encoding="utf-8")))
    ]
    assert len(setting) >= 3, setting


@pytest.mark.parametrize(
    "template", _compose_templates(), ids=lambda path: str(path.relative_to(TEMPLATES))
)
def test_config_file_is_never_named_without_osprey_config(template: Path):
    for index, block in enumerate(_blocks(template.read_text(encoding="utf-8"))):
        assert sorted(block["CONFIG_FILE"]) == sorted(
            value for value in block["OSPREY_CONFIG"] if value in block["CONFIG_FILE"]
        ), (
            f"environment block {index} sets CONFIG_FILE to {block['CONFIG_FILE']} but "
            f"OSPREY_CONFIG to {block['OSPREY_CONFIG']}: set OSPREY_CONFIG to the same path"
        )


def test_a_block_naming_only_config_file_is_refused():
    """The rule, on a block that breaks it, so the check is not vacuous."""
    blocks = _blocks(
        "services:\n  svc:\n    environment:\n      CONFIG_FILE: /app/config.yml\n      TZ: UTC\n"
    )
    assert blocks == [{"CONFIG_FILE": ["/app/config.yml"], "OSPREY_CONFIG": []}]
