"""The CA-file mount and the own-image build gate are each written once.

Both used to be spelled out in every template that needed them — the CA bind
in three, the build gate and its comment in six — so a fix to one copy (the
``$`` escaping, the gate's null-block rule) could miss the others. They now live
in ``services/_ca_bundles.j2`` and ``services/_own_image.j2``; this pins that
no template grows its own copy back, and what each macro renders.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

_TEMPLATE_ROOT = Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates"
_COMPOSE_TEMPLATES = sorted(_TEMPLATE_ROOT.rglob("docker-compose*.yml.j2"))


def _env() -> Environment:
    return Environment(loader=FileSystemLoader(str(_TEMPLATE_ROOT)))


def test_the_sweep_sees_the_compose_templates() -> None:
    names = {path.parent.name for path in _COMPOSE_TEMPLATES}
    assert {"archiver_recorder", "dispatch_worker", "web_terminals", "optimizer"} <= names


@pytest.mark.parametrize("template", _COMPOSE_TEMPLATES, ids=lambda p: p.parent.name)
def test_no_compose_template_spells_its_own_ca_bind(template: Path) -> None:
    text = template.read_text(encoding="utf-8")
    assert "for ca_bundle in" not in text, f"{template.parent.name}: use ca.mounts()"


@pytest.mark.parametrize("template", _COMPOSE_TEMPLATES, ids=lambda p: p.parent.name)
def test_no_compose_template_repeats_the_build_gate_comment(template: Path) -> None:
    text = template.read_text(encoding="utf-8")
    assert "The build renders only while this service runs the image" not in text, (
        f"{template.parent.name}: use own.build()"
    )


def test_ca_mounts_render_each_path_read_only_at_the_same_path() -> None:
    rendered = (
        _env()
        .from_string(
            '{% import "services/_ca_bundles.j2" as ca %}volumes:{{- ca.mounts(paths, indent=2) }}'
        )
        .render(paths=["/etc/ssl/a$b.pem", "/srv/ca bundle.pem"])
    )

    assert yaml.safe_load(rendered) == {
        "volumes": [
            {"type": "bind", "source": p, "target": p, "read_only": True}
            for p in ("/etc/ssl/a$$b.pem", "/srv/ca bundle.pem")
        ]
    }


@pytest.mark.parametrize("paths", [None, []])
def test_ca_mounts_render_nothing_for_no_paths(paths: list[str] | None) -> None:
    template = '{% import "services/_ca_bundles.j2" as ca %}a:{{- ca.mounts(paths) }}'
    assert _env().from_string(template).render(paths=paths) == "a:"


@pytest.mark.parametrize(
    ("image", "builds"), [(None, True), ("", True), ("own:1", True), ("site/other:2", False)]
)
def test_own_image_build_renders_only_for_the_packaged_image(
    image: str | None, builds: bool
) -> None:
    template = (
        '{% import "services/_own_image.j2" as own %}image: x'
        "{%- call own.build(image, 'own:1') %}\nbuild: y{%- endcall %}\nnext: z"
    )
    rendered = _env().from_string(template).render(image=image)

    assert rendered == ("image: x\nbuild: y\nnext: z" if builds else "image: x\nnext: z")
