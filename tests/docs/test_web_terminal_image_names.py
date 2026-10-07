"""The multi-user page states the registry-mode web-terminal image names.

Those names exist only in ``resolve_personas``. A facility pipeline pushes what
the page says and the deploy host pulls what the function says, so this module
holds the two equal: a rename in either one fails here.
"""

import re
import textwrap
from pathlib import Path

import yaml

from osprey.deployment.web_terminals.personas import (
    effective_image_source,
    resolve_image_tag,
    resolve_personas,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PAGE = _REPO_ROOT / "docs" / "source" / "how-to" / "web-terminal" / "multi-user" / "index.rst"
_FACILITY_PAGE = _REPO_ROOT / "docs" / "source" / "how-to" / "deploy-a-facility.rst"

_LABEL = "multi-user-registry-images"
_HEADING = re.compile(r"^[^\n]+\n[-=]{3,}\n", re.MULTILINE)
_BLOCK = re.compile(
    r"^\.\. code-block:: (?P<lang>\w+)\n\n(?P<body>(?:(?:   .*)?\n)+)", re.MULTILINE
)


def _section(text: str) -> str:
    start = text.index(f".. _{_LABEL}:\n")
    own_title = _HEADING.search(text, start)
    assert own_title is not None
    following = _HEADING.search(text, own_title.end())
    return text[start : following.start() if following else len(text)]


def _blocks(section: str, lang: str) -> list[str]:
    return [textwrap.dedent(m.group("body")) for m in _BLOCK.finditer(section) if m["lang"] == lang]


def _page_section() -> str:
    return _section(_PAGE.read_text(encoding="utf-8"))


def test_the_name_patterns_are_what_resolve_personas_spells():
    """Both image-name patterns on the page are the resolver's own output."""
    section = _page_section()
    wt = {
        "image_tag": "<tag>",
        "default_persona": "base",
        "users": [
            {"name": "a", "index": 0, "persona": "<persona>"},
            {"name": "b", "index": 1},
        ],
        "personas": {
            "base": {"project": "demo-base", "project_path": "build/demo-base"},
            "<persona>": {"project": "demo-other", "project_path": "build/demo-other"},
        },
    }
    registry_cfg = {"url": "<registry.url>"}
    resolved = {e["name"]: e["image"] for e in resolve_personas(wt, registry_cfg, "demo")}
    for image in resolved.values():
        assert f"``{image}``" in section

    no_catalog = {"image_tag": "<tag>", "users": [{"name": "c", "index": 0}]}
    (only,) = {e["image"] for e in resolve_personas(no_catalog, registry_cfg, "demo")}
    assert only == resolved["b"]


def test_the_worked_example_resolves_to_the_images_the_page_lists():
    """The worked example's listed images are what the resolver produces for its config."""
    section = _page_section()
    (yaml_body,) = _blocks(section, "yaml")
    (text_body,) = _blocks(section, "text")
    cfg = yaml.safe_load(yaml_body)
    wt = cfg["modules"]["web_terminals"]
    assert effective_image_source(wt) == "registry"
    resolved = {e["name"]: e["image"] for e in resolve_personas(wt, cfg["registry"], "demo")}
    listed = dict(line.split() for line in text_body.splitlines() if line.strip())
    assert resolved == listed


def test_the_tag_default_the_page_states_is_the_resolver_default():
    """The page's default tag is the resolver's default tag."""
    assert resolve_image_tag({}) == "latest"
    assert "``latest``" in _page_section()


def test_the_facility_page_points_at_the_contract():
    """The facility walk-through links the image-name contract."""
    assert f":ref:`{_LABEL}`" in _FACILITY_PAGE.read_text(encoding="utf-8")
