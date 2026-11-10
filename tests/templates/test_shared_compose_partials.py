"""The CA-file mount and the own-image build gate are each written once.

Both used to be spelled out in every template that needed them — the CA bind
in three, the build gate and its comment in six, the gate's predicate again in
the three templates that elect one builder among services sharing an image — so
a fix to one copy (the ``$`` escaping, the gate's null-block rule) could miss
the others. They now live in ``services/_ca_bundles.j2`` and
``services/_own_image.j2``; this pins that no template grows its own copy back,
and what each macro renders.
"""

from __future__ import annotations

import re
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


#: The own-image predicate in any spelling a template has used: ``not <x>.image or``
#: and ``<x>.image == osprey_images.<key>``. Only ``_own_image.j2`` may state it.
_OWN_IMAGE_PREDICATE = re.compile(r"not\s+[\w.\[\]]*\.image\s+or\b|\.image\s*==\s*osprey_images\b")


@pytest.mark.parametrize("template", _COMPOSE_TEMPLATES, ids=lambda p: p.parent.name)
def test_no_compose_template_restates_the_own_image_predicate(template: Path) -> None:
    text = template.read_text(encoding="utf-8")
    assert not _OWN_IMAGE_PREDICATE.search(text), (
        f"{template.parent.name}: gate the build with own.build() or own.elect()"
    )
    assert "The build renders only while this service runs the image" not in text, (
        f"{template.parent.name}: use own.build()"
    )


def test_the_partial_states_the_predicate_once() -> None:
    text = (_TEMPLATE_ROOT / "services" / "_own_image.j2").read_text(encoding="utf-8")
    assert text.count("not image or image == own_image") == 1


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


_ELECT = (
    '{% import "services/_own_image.j2" as own %}'
    "{% set ns = namespace(key=none) %}{{ own.elect(ns, keys, services, 'own:1') }}{{ ns.key }}"
)


@pytest.mark.parametrize(
    ("services", "elected"),
    [
        ({"a": {}, "b": {}}, "a"),
        ({"a": None, "b": {}}, "a"),
        ({"a": {"image": "site/other:2"}, "b": {"image": "own:1"}}, "b"),
        ({"a": {"image": "site/other:2"}, "b": {}}, "b"),
        ({"a": {"image": "site/other:2"}, "b": {"image": "site/x:3"}}, "None"),
    ],
)
def test_elect_picks_the_first_key_running_the_packaged_image(services: dict, elected: str) -> None:
    rendered = _env().from_string(_ELECT).render(keys=["a", "b"], services=services)

    assert rendered == elected


def test_elect_never_reads_a_key_after_the_builder() -> None:
    # A later key with no block at all would fail to render if it were read.
    rendered = _env().from_string(_ELECT).render(keys=["a", "b"], services={"a": {}})

    assert rendered == "a"


#: The builder election as bluesky and virtual_accelerator spelled it inline
#: before ``own.elect`` (qmd's was its one-key case), kept here as the oracle.
_INLINE_ELECTION = (
    "{% set b = namespace(key=none) %}{% for c in keys %}"
    "{% if b.key is none and (not services[c].image or services[c].image == own_image) %}"
    "{% set b.key = c %}{% endif %}{% endfor %}{{ b.key }}"
)
_IMAGE_KINDS = {"unset": {}, "null": None, "own": {"image": "OWN"}, "foreign": {"image": "x/y:1"}}


@pytest.mark.parametrize(
    ("first", "second"),
    [(a, None) for a in _IMAGE_KINDS] + [(a, b) for a in _IMAGE_KINDS for b in _IMAGE_KINDS],
)
@pytest.mark.parametrize(
    ("keys", "own_image"),
    [
        (["virtual_accelerator", "live_standin"], "osprey-va"),
        (["bluesky", "bluesky_va"], "osprey-bluesky-bridge"),
        (["qmd"], "osprey-qmd"),
    ],
    ids=["va", "bluesky", "qmd"],
)
def test_elect_matches_the_inline_predicate_it_replaced(
    keys: list[str], own_image: str, first: str, second: str | None
) -> None:
    def block(kind: str) -> dict | None:
        spec = _IMAGE_KINDS[kind]
        return None if spec is None else {"image": own_image} if kind == "own" else dict(spec)

    services = {keys[0]: block(first)}
    if second is not None and len(keys) > 1:
        services[keys[1]] = block(second)
    elect_keys = list(services)
    context = {"keys": elect_keys, "services": services, "own_image": own_image}
    elected = (
        _env()
        .from_string(
            '{% import "services/_own_image.j2" as own %}'
            "{% set ns = namespace(key=none) %}"
            "{{ own.elect(ns, keys, services, own_image) }}{{ ns.key }}"
        )
        .render(context)
    )

    assert elected == _env().from_string(_INLINE_ELECTION).render(context)
