"""The multi-user page states the registry-mode web-terminal image names.

Those names exist only in ``resolve_cards``. A facility pipeline pushes what
the page says and the deploy host pulls what the function says, so this module
holds the two equal: a rename in either one fails here.
"""

import re
import textwrap
from pathlib import Path

import yaml

from osprey.deployment.web_terminals.render_naming import (
    RENDER_KIND_TIER,
    RENDERS_KEY,
    derived_renders,
)
from osprey.deployment.web_terminals.roster import (
    effective_image_source,
    resolve_cards,
    resolve_image_tag,
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


def test_the_name_patterns_are_what_resolve_cards_spells():
    """Both image-name patterns on the page are the resolver's own output."""
    section = _page_section()
    wt = {
        "image_tag": "<tag>",
        "default_tier": "base",
        "users": [
            {"name": "a", "index": 0, "tier": "<render>"},
            {"name": "b", "index": 1},
        ],
        "renders": {
            "base": {
                "kind": RENDER_KIND_TIER,
                "project": "demo-base",
                "project_path": "build/demo-base",
            },
            "<render>": {
                "kind": RENDER_KIND_TIER,
                "project": "demo-other",
                "project_path": "build/demo-other",
            },
        },
    }
    registry_cfg = {"url": "<registry.url>"}
    resolved = {e["name"]: e["image"] for e in resolve_cards(wt, registry_cfg)}
    for image in resolved.values():
        assert f"``{image}``" in section


def test_the_worked_example_resolves_to_the_images_the_page_lists(tmp_path):
    """The worked example's listed images are what the resolver produces for its config."""
    section = _page_section()
    (yaml_body,) = _blocks(section, "yaml")
    (text_body,) = _blocks(section, "text")
    cfg = yaml.safe_load(yaml_body)
    wt = cfg["modules"]["web_terminals"]
    assert effective_image_source(wt) == "registry"
    # The build writes one render per tier file of the repo into the catalog
    # of a deployment whose web terminals are on; the example's repo carries a
    # tier file for every tier its roster names.
    tiers = {wt["default_tier"], *(u["tier"] for u in wt.get("users", []) if "tier" in u)}
    (tmp_path / "tiers").mkdir()
    for tier in tiers:
        (tmp_path / "tiers" / f"{tier}.yml").write_text("{}\n", encoding="utf-8")
    enabled = {**cfg, "modules": {"web_terminals": {**wt, "enabled": True}}}
    renders = wt.setdefault("renders", {})
    for key, entry in derived_renders(enabled, "demo", tmp_path).items():
        renders.setdefault(key.removeprefix(f"{RENDERS_KEY}."), {}).update(entry)
    resolved = {e["name"]: e["image"] for e in resolve_cards(wt, cfg["registry"])}
    listed = dict(line.split() for line in text_body.splitlines() if line.strip())
    assert resolved == listed


def test_the_tag_default_the_page_states_is_the_resolver_default():
    """The page's default tag is the resolver's default tag."""
    assert resolve_image_tag({}) == "latest"
    assert "``latest``" in _page_section()


def test_the_facility_page_points_at_the_contract():
    """The facility walk-through links the image-name contract."""
    assert f":ref:`{_LABEL}`" in _FACILITY_PAGE.read_text(encoding="utf-8")
