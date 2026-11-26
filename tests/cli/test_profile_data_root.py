"""Tests for the deployment's declared sources replacing the bundle's data tree.

A deployment's render sources its ``data/`` from the declared sources beside
the profile (the ``facility/`` tree and the ``web/`` content), never from
``apps/<data_bundle>/data/``; no profile key names them. That is a full
replacement, not a layer, and the sources are content rather than templates:

* a stray ``.j2`` file lands byte-identical, extension intact
* no file of the bundle's packaged ``data/`` reaches the render, whatever its
  shape
* the build-time pipeline is otherwise untouched: the benchmark query step runs
  against the deployment's facility tree, and the project/ mirror applied later
  still wins
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

from osprey.cli.build_profile_emit import emit_layer_delta_yaml
from osprey.cli.build_profile_presets import list_tiers
from osprey.cli.templates.scaffolding import copy_template_data
from osprey.facility.views.limits import limits_document
from osprey_facility.build import build_facility
from osprey_facility.records import FacilityRecords
from tests._builds import FRAMEWORK_CHECKS
from tests._preset_data import bundle_source_root


def _write_profile(profile_dir: Path, **extra) -> Path:
    """Materialize a profile directory carrying its own copy of the bundle's sources.

    The copy starts byte-identical to what ``osprey init`` lays down from the
    demo preset (its ``facility/`` and ``web/``) so the benchmark query sources
    the build copies from are present; individual tests then mutate it to make
    the replacement observable.
    """
    profile_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle_source_root(), profile_dir, dirs_exist_ok=True)

    profile: dict = {
        "name": "Data Root Test",
        "project_name": "data-root-test",
        "extends": "demo",
        "provider": "cborg",
        "model": "claude-haiku-4-5",
        "channel_finder_mode": "in_context",
        # The preset deploys a virtual accelerator, which serves the project's
        # own channels and so refuses a build whose sources stage none. These
        # tests replace those sources wholesale — that is their subject — so the
        # inherited block is dropped rather than fed.
        "virtual_accelerator": None,
        # With no accelerator to stand up, the baseline moves off the stand-in
        # the preset selects and onto the connector that needs no service.
        "config": {
            "control_system.type": "virtual_accelerator",
            "control_system.connector.virtual_accelerator.serving": "in_process",
        },
    }
    extra_config = dict(extra.pop("config", {}) or {})
    profile.update(extra)
    profile["config"].update(extra_config)
    path = profile_dir / "profile.yml"
    path.write_text(yaml.dump(profile, default_flow_style=False))
    _materialize_layers(profile_dir, profile["name"])
    return path


def _materialize_layers(profile_dir: Path, profile_name: str) -> None:
    """Write the bundled logins beside ``profile.yml``, as ``osprey init`` does.

    The inherited roster names tiers and shared logins, and each resolves only
    to a file in the repo's ``tiers/``.
    """
    for kind, names in (("tiers", list_tiers()),):
        (profile_dir / kind).mkdir(exist_ok=True)
        for name in names:
            (profile_dir / kind / f"{name}.yml").write_text(
                emit_layer_delta_yaml(
                    preset_name=f"{kind}/{name}",
                    profile_name=profile_name,
                    profile_filename=f"{kind}/{name}.yml",
                ),
                encoding="utf-8",
            )


def _build(profile_path: Path) -> Path:
    """Run `osprey build` against the repo holding ``profile_path`` and return its build/.

    ``osprey build`` is zero-argument: it always renders the repo it is run
    against (``--repo``) into that repo's own ``build/``, so there is no output
    directory or project name to choose — the repo's directory (``profile_path``'s
    parent) decides both.
    """
    from click.testing import CliRunner

    from osprey.cli.main import cli

    repo = profile_path.parent
    result = CliRunner().invoke(
        cli,
        ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"],
    )
    assert result.exit_code == 0, (
        f"build failed (exit={result.exit_code})\n"
        f"--- output ---\n{result.output}\n"
        f"--- exception ---\n{result.exception}"
    )
    return repo / "build"


#: Where the build reads the in_context mode's benchmark queries from.
QUERIES_SOURCE = Path("facility/benchmarks/channel_finder/in_context_queries.json")

#: The flat channel database the build writes from the facility tree, which the
#: project/ mirror overrides.
FLAT_DB = Path("data/channel_databases/in_context.json")

STRAY_J2 = b"{{ facility_name }} raw braces\n{% if channel_finder_mode %}kept{% endif %}\n"
PROFILE_QUERIES = '[{"user_query": "profile", "targeted_pv": []}]\n'
MIRROR_BODY = '{"channels": {"MIRROR:CH": {"description": "mirrored"}}}\n'


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """One profile carrying every test's mark, built once; read only.

    Each mark is a file that one claim reads and no other build step does. A
    build that fails errors all four tests, with its output.
    """
    profile_dir = tmp_path_factory.mktemp("data-root") / "profile"
    profile_path = _write_profile(profile_dir)
    facility = profile_dir / "facility"
    (facility / "facility_notes.md.j2").write_bytes(STRAY_J2)
    # A deployment-only file proves the deployment's tree is the source.
    (facility / "facility_marker.txt").write_text("profile tree\n")
    # Queries recognizably NOT the bundle's.
    (profile_dir / QUERIES_SOURCE).write_text(PROFILE_QUERIES)
    # The mirror and the database the build writes differ, so only the mirror
    # landing last can pass.
    mirrored = profile_dir / "ops" / "project" / FLAT_DB
    mirrored.parent.mkdir(parents=True)
    mirrored.write_text(MIRROR_BODY)
    return profile_dir, _build(profile_path)


class TestProfileDataIsContentNotTemplates:
    """The deployment's sources are copied verbatim — no Jinja pass."""

    def test_stray_j2_survives_byte_identical(self, built: tuple[Path, Path]) -> None:
        _, project_dir = built

        landed = project_dir / "data" / "facility" / "facility_notes.md.j2"
        assert landed.exists(), "stray .j2 was dropped from the deployment's facility tree"
        assert landed.read_bytes() == STRAY_J2, "stray .j2 was rendered instead of copied verbatim"
        assert not (project_dir / "data" / "facility" / "facility_notes.md").exists(), (
            ".j2 extension was stripped — the profile tree went through the render path"
        )


class TestFullReplacement:
    """No bundle file reaches a project built from the deployment's sources."""

    def test_bundle_files_absent_from_build(self, built: tuple[Path, Path]) -> None:
        profile_dir, project_dir = built
        assert (project_dir / "data" / "facility" / "facility_marker.txt").exists(), (
            "deployment-only file did not land — the deployment's tree was not the source"
        )
        # The build writes the limits database itself, from the deployment's
        # `facility/limits.yaml`.
        limits = json.loads((project_dir / "data" / "channel_limits.json").read_text())
        facility_dir = profile_dir / "facility"
        assert limits == limits_document(
            FacilityRecords(
                build_facility(
                    facility_dir, project_name=project_dir.name, extra_checks=FRAMEWORK_CHECKS
                )
            )
        ), "the limits database is not the view of the deployment's limits.yaml"

    @pytest.mark.parametrize("bundle_data_at_top", [True, False])
    def test_both_bundle_derivation_branches_are_bypassed(
        self, tmp_path: Path, bundle_data_at_top: bool
    ) -> None:
        """The render reads the deployment root, never the bundle's packaged ``data/``.

        A bundle can hold its data at ``apps/<bundle>/data`` or nested deeper;
        parametrizing over the shape shows neither is read.
        """
        template_root = tmp_path / "templates"
        bundle_dir = template_root / "apps" / "demo"
        if bundle_data_at_top:
            bundle_data = bundle_dir / "data"
        else:
            bundle_data = bundle_dir / "pkg" / "data"
        bundle_data.mkdir(parents=True)
        (bundle_data / "from_bundle.txt").write_text("bundle\n")

        repo_root = tmp_path / "deployment"
        facility = repo_root / "facility"
        (facility / "nested").mkdir(parents=True)
        (facility / "from_profile.txt").write_text("profile\n")
        (facility / "nested" / "seed.json.j2").write_text("{{ not_rendered }}\n")

        project_dir = tmp_path / "project"
        (project_dir / "data").mkdir(parents=True)

        copy_template_data(
            template_root,
            project_dir,
            "demo_pkg",
            "demo",
            {},
            jinja_env=Environment(loader=FileSystemLoader(str(template_root))),
            repo_root=repo_root,
        )

        rendered = project_dir / "data" / "facility"
        assert (rendered / "from_profile.txt").read_text() == "profile\n"
        assert (rendered / "nested" / "seed.json.j2").read_text() == "{{ not_rendered }}\n"
        assert not any(path.name == "from_bundle.txt" for path in project_dir.rglob("*")), (
            "bundle data leaked into the render"
        )


class TestBuildPipelineOrderingHolds:
    """Materialization and the mirror behave as for a bundle-sourced tree."""

    def test_benchmark_queries_come_from_the_profile_tree(self, built: tuple[Path, Path]) -> None:
        profile_dir, project_dir = built
        queries_src = profile_dir / QUERIES_SOURCE

        queries = project_dir / "data" / "benchmarks" / "queries.json"
        assert queries.read_text() == PROFILE_QUERIES, (
            "benchmark queries were not copied from the deployment's facility tree"
        )
        assert not (project_dir / "data" / "facility" / "benchmarks" / "channel_finder").exists(), (
            "channel_finder/ subtree reached the render from the deployment's tree"
        )
        assert queries_src.exists(), "the deployment's own tree was pruned"

    def test_project_mirror_wins_over_the_deployment_sources(
        self, built: tuple[Path, Path]
    ) -> None:
        """The project/ mirror applies after the sources land, so it wins."""
        _, project_dir = built

        assert (project_dir / "data" / "facility" / "facility_marker.txt").exists(), (
            "data/ was not sourced from the deployment's tree — the ordering claim is untested"
        )
        landed = project_dir / FLAT_DB
        assert landed.read_text() == MIRROR_BODY, (
            "project/ mirror did not win over what the build writes from the facility tree"
        )
