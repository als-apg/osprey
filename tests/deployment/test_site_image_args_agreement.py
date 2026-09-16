"""The two renderers of the site's build args hand a build the same thing.

The site's build settings — a CA for a TLS-intercepting proxy, pip's proxy
bypass list, an internal package index — reach an image two ways. The three
images OSPREY builds from an argv of its own get them as ``--build-arg`` flags
(:func:`osprey.deployment.container_lifecycle.site_image_build_args`); a managed
service image, which ``docker compose build`` builds from its own rendered
context, gets them as the ``args:`` block of its compose fragment
(:func:`osprey.deployment.compose_generator._stage_site_image_args_for_context`
and the ``build_args`` macro it feeds).

The two must hand a build the same ARG names carrying the same values for one
resolved config. Nothing downstream compares them, so a drift between the two
spellings surfaces only as an image built without the site's CA or index — at
the layer that installs it, on the host that builds it, long after the change
that caused it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

from osprey.deployment import compose_generator, container_lifecycle

#: The packaged service templates — where the macro under test lives.
SERVICES_DIR = Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates" / "services"

MACROS = "_site_image_args.j2"


@pytest.fixture
def no_site_env(monkeypatch):
    """A host that exports none of the axis overrides.

    The overrides are named after the build args themselves (PIP_INDEX_URL and
    kin), so a developer machine that happens to export one for its own tooling
    would otherwise leak into every assertion here.
    """
    for name in ("OSPREY_SITE_CA", "PIP_NO_PROXY", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("OSPREY_OFFLINE", raising=False)


def _site_config(tmp_path, **extra):
    """A config declaring every site axis, with a real CA file to stage."""
    ca = tmp_path / "site-ca.pem"
    ca.write_text("-----BEGIN CERTIFICATE-----\n")
    config = {
        "project_name": "myfacility",
        "images": {
            "site_ca": str(ca),
            "pip_no_proxy": ".internal.example.org",
            "pip_index_url": "https://mirror.example.org/simple",
            "pip_extra_index_url": "https://wheels.example.org/simple",
        },
    }
    config.update(extra)
    return config


def _build_args(cmd):
    """The ``--build-arg`` values in *cmd*, as a ``NAME -> value`` mapping."""
    return dict(
        arg.split("=", 1) for flag, arg in zip(cmd, cmd[1:], strict=False) if flag == "--build-arg"
    )


def _context(root, name):
    """A build context in the shape a render leaves behind.

    A directory holding a ``Dockerfile``, which is what the compose-side
    resolver keys on: a pure-image service builds nothing and gets none of it.
    """
    context = Path(root) / name
    context.mkdir(parents=True, exist_ok=True)
    (context / "Dockerfile").write_text("FROM scratch\n")
    return context


def _rendered_text(mapping):
    """The macro's output for *mapping*, rendered as a template would render it."""
    env = Environment(loader=FileSystemLoader(str(SERVICES_DIR)), autoescape=False)
    return env.get_template(MACROS).module.build_args(mapping)


def _rendered_args(mapping):
    """The macro's output for *mapping*, read back the way compose would.

    The splice reproduces the call site exactly: the macro's default ``indent``
    is the column an entry sits at under a six-column ``args:`` key, and the
    macro's leading newline is the one the ``{{-`` at the call site eats.
    """
    document = "services:\n  s:\n    build:\n      context: .\n      args:"
    parsed = yaml.safe_load(document + _rendered_text(mapping) + "\n")
    return parsed["services"]["s"]["build"]["args"] or {}


def test_both_renderers_carry_the_same_arg_names_and_values(no_site_env, tmp_path):
    """One resolved config, two renderers, one set of ARG names and values."""
    config = _site_config(tmp_path)
    argv_context = _context(tmp_path, "argv-context")
    compose_context = _context(tmp_path, "compose-context")

    argv = _build_args(container_lifecycle.site_image_build_args(config, argv_context))
    staged = compose_generator._stage_site_image_args_for_context(config, str(compose_context))
    rendered = _rendered_args(staged)

    expected = {
        # The staged NAME, never the host path: COPY cannot reach outside a
        # build context, so both forms name the file copied in beside them.
        "OSPREY_SITE_CA": container_lifecycle.SITE_CA_CONTEXT_FILENAME,
        "PIP_NO_PROXY": ".internal.example.org",
        "PIP_INDEX_URL": "https://mirror.example.org/simple",
        "PIP_EXTRA_INDEX_URL": "https://wheels.example.org/simple",
    }
    assert argv == expected
    assert staged == expected
    assert rendered == expected

    # The value both forms render is only true if the file is there.
    for context in (argv_context, compose_context):
        bundle = context / container_lifecycle.SITE_CA_CONTEXT_FILENAME
        assert bundle.read_text() == "-----BEGIN CERTIFICATE-----\n"


def test_the_two_renderers_order_their_args_deliberately_and_differently(no_site_env, tmp_path):
    """Order carries no meaning — a build arg is addressed by name.

    The argv follows the axis table so a reader can match a flag to its
    declaration; the macro sorts by name so a render is a function of its
    context alone. Both are deliberate, and neither is the other's bug.
    """
    config = _site_config(tmp_path)
    argv = container_lifecycle.site_image_build_args(config, _context(tmp_path, "argv-context"))
    staged = compose_generator._stage_site_image_args_for_context(
        config, str(_context(tmp_path, "compose-context"))
    )

    assert list(_build_args(argv)) == list(compose_generator.SITE_IMAGE_AXES)

    # Off the rendered TEXT, not the parsed mapping: the parse would hand back
    # whatever order the loader chose, which is not the order the macro wrote.
    rendered_names = [
        line.split(":", 1)[0].strip()
        for line in _rendered_text(staged).splitlines()
        if line.strip()
    ]
    assert rendered_names == sorted(rendered_names)
    assert set(rendered_names) == set(compose_generator.SITE_IMAGE_AXES)


def test_only_the_project_image_is_told_the_deployment_is_offline(no_site_env, tmp_path):
    """The one flag the two forms do not share is the one no service would read.

    Offline is a property of the image that SERVES the vendored web assets:
    only the project recipe declares that ARG and runs the vendoring step, so a
    service fragment carrying it would be handing a build a setting its
    Dockerfile has no line for.
    """
    config = _site_config(tmp_path, offline=True)
    argv = _build_args(
        container_lifecycle.site_image_build_args(config, _context(tmp_path, "argv-context"))
    )
    staged = compose_generator._stage_site_image_args_for_context(
        config, str(_context(tmp_path, "compose-context"))
    )

    assert argv["OSPREY_OFFLINE"] == "1"
    assert "OSPREY_OFFLINE" not in staged
    assert "OSPREY_OFFLINE" not in _rendered_args(staged)


def test_a_deployment_that_declares_nothing_builds_and_renders_as_it_always_did(
    no_site_env, tmp_path
):
    """No site settings means the argv and the fragment these axes were added to."""
    config = {"project_name": "x"}
    context = _context(tmp_path, "context")

    assert container_lifecycle.site_image_build_args(config, context) == []
    staged = compose_generator._stage_site_image_args_for_context(config, str(context))
    assert staged == {}
    assert _rendered_text(staged) == ""
    assert _rendered_args(staged) == {}
