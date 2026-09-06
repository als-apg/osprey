"""What an attached render is told about the host it hangs off.

A persona built with no deployment in its own repo is told its host's facts by
reading the deployment the profile describes — :func:`_template_host_config`.
That reading is where the ports come from: the panel URLs and service endpoints
a persona is handed are derived from it, so a port missing there is a port no
persona ever learns, and a port taken from the layout's own default rather than
from the deployment's base is a port nothing answers on.

An explicit profile spells the service blocks it deploys but need not spell a
host port for each one — the port is a derivation of the deployment's base, not
information the operator holds. The build fills those in for the deploying
render, and this reading gets the SAME fill, so the two agree by construction.

The base here is deliberately two blocks away from the layout default: a number
that came from :data:`~osprey.port_layout.DEFAULT_PORT_BASE` instead of from the
profile cannot pass any assertion below by coincidence.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path
from typing import Any

import pytest

from osprey.cli import build_cmd
from osprey.cli.build_profile_model import BuildProfile
from osprey.cli.build_profile_ports import layout_port_fill
from osprey.cli.build_profile_reach import reach_override_errors, spelled_values
from osprey.cli.templates.manager import TemplateManager
from osprey.port_layout import DEFAULT_PORT_BASE, layout_ports

#: The base every profile in this module deploys at.
MOVED_BASE = 20000

#: A port no layout row can produce at any base, so a profile that spells it is
#: visibly the author of it.
AUTHORED_PORT = 31234


def _host_reading(profile: BuildProfile) -> dict[str, Any]:
    """The rendered config :func:`_template_host_config` reads for *profile*.

    Runs the real render into a throwaway repo, exactly as a build with no
    deployment of its own does.
    """
    with tempfile.TemporaryDirectory() as scratch:
        repo_root = Path(scratch)
        render_dir = repo_root / "build" / "proj"
        render_dir.mkdir(parents=True)
        shared = build_cmd._SharedRenderInputs(
            repo_root=repo_root,
            build_dir=repo_root / "build",
            runtime_root=None,
            project_deps=[],
            skip_deps=True,
            manager=TemplateManager(),
            va_manifests={},
            va_reported=set(),
        )
        return build_cmd._template_host_config(
            shared,
            profile,
            project_name="proj",
            render_dir=render_dir,
            profile_dir=repo_root,
            context={
                "port_base": MOVED_BASE,
                "project_root": str(repo_root),
                "dependencies": [],
                "channel_finder_mode": "hierarchical",
            },
            artifacts=None,
        )


def _profile(**config: Any) -> BuildProfile:
    """An attached profile deploying at :data:`MOVED_BASE` with *config* over it."""
    return BuildProfile(
        name="persona",
        channel_finder_mode="hierarchical",
        config={"deployment.port_base": MOVED_BASE, **config},
    )


class TestTheHostReadingCarriesTheLayoutPorts:
    """The reading is told the ports the deployment publishes."""

    def test_a_deployed_block_without_a_port_gets_the_layout_one(self) -> None:
        """A profile that deploys a service and names no port is not told none.

        ``services.bluesky`` is the sharp case: nothing in the rendered
        template writes that block, so the two ports below can only have come
        from the layout fill this reading applies.
        """
        reading = _host_reading(_profile(**{"services.bluesky.path": "./services/bluesky"}))

        ports = layout_ports(MOVED_BASE)
        assert reading["services"]["bluesky"]["port"] == ports["bluesky"]
        assert reading["services"]["bluesky"]["tiled_port"] == ports["tiled"]

    def test_the_ports_come_from_the_profiles_base_not_the_layout_default(self) -> None:
        """Nothing in the reading is left at the base this deployment moved off."""
        reading = _host_reading(_profile(**{"services.bluesky.path": "./services/bluesky"}))

        block = reading["services"]["bluesky"]
        abandoned = range(DEFAULT_PORT_BASE, DEFAULT_PORT_BASE + 1000)
        assert block["port"] not in abandoned
        assert block["tiled_port"] not in abandoned

    def test_a_port_the_profile_spells_is_left_alone(self) -> None:
        """The fill is fill-if-absent: an authored port outranks the layout."""
        reading = _host_reading(
            _profile(
                **{
                    "services.bluesky.path": "./services/bluesky",
                    "services.bluesky.port": AUTHORED_PORT,
                }
            )
        )

        assert reading["services"]["bluesky"]["port"] == AUTHORED_PORT

    def test_a_service_the_profile_does_not_deploy_is_not_invented(self) -> None:
        """No block appears in the reading because the layout has a row for it."""
        reading = _host_reading(_profile())

        assert "bluesky" not in reading.get("services", {})

    def test_the_dispatch_bands_base_is_filled_like_any_other_row(self) -> None:
        """``worker_port_base`` is not special-cased out of the fill.

        It is the one layout row under ``services.`` whose key is a band base
        rather than a bound port, so the question was whether filling it says
        something a render does not. It does not: the dispatch injector writes
        the same key from the ``dispatch:`` block for every deploying render,
        bridge topology included, and the number here is the one the compose
        templates already fall back to (``osprey_ports.worker``). A profile
        that spells the block without deploying the pair gets that same number
        rather than a hole.
        """
        reading = _host_reading(
            _profile(**{"services.dispatch_worker.path": "./services/dispatch_worker"})
        )

        assert (
            reading["services"]["dispatch_worker"]["worker_port_base"]
            == layout_ports(MOVED_BASE)["worker"]
        )

    def test_the_reading_agrees_with_the_fill_key_for_key(self) -> None:
        """The reading and the deploying render run ONE derivation.

        Both sites call :func:`layout_port_fill` with the profile's own
        ``config:`` and the base it resolves, so every key the fill produces is
        in the reading with the fill's value. A persona told a different number
        than its host publishes is the failure this pins.
        """
        profile = _profile(
            **{
                "services.bluesky.path": "./services/bluesky",
                "services.graphdb.uri": "bolt://store.example:7687",
            }
        )
        reading = _host_reading(profile)

        filled = layout_port_fill(profile.config, build_cmd._profile_port_base(profile))
        assert filled, "the fixture profile must actually exercise the fill"
        for key, port in filled.items():
            _, service, leaf = key.split(".")
            assert reading["services"][service][leaf] == port


class TestAPresetKeyAPersonaInherits:
    """An inherited preset spelling is the host's value, not a second copy.

    Every shipped preset states its whole rendered configuration under
    ``config:``, and a persona ``extends:`` one — so a persona's merged
    ``config:`` already spells the projected keys its preset carries. Those
    spellings are where the host's value came from, so refusing them would
    refuse every persona built from a preset that names a projected service.
    """

    #: Projected keys the control-assistant preset states itself. Read off the
    #: resolved preset rather than hardcoded, so the test moves with the file.
    PROJECTED = ("services.postgresql.username", "services.postgresql.database_name")

    @staticmethod
    def _preset_config() -> dict[str, Any]:
        from osprey.cli.build_profile_resolve import resolve_build_profile

        profile, _preset_dir = resolve_build_profile(None, "control-assistant")
        return dict(profile.config)

    def test_an_inherited_preset_key_is_not_a_reach_override(self) -> None:
        config = self._preset_config()
        projected = {key: spelled_values(config, key)[0][1] for key in self.PROJECTED}
        assert all(projected.values()), (
            "the preset must actually spell these keys, or this proves nothing"
        )

        assert reach_override_errors(config, projected) == []

    def test_a_persona_delta_that_contradicts_the_host_is_still_refused(self) -> None:
        """The invariant is agreement, so a disagreeing spelling still fails."""
        config = self._preset_config()
        projected = {key: spelled_values(config, key)[0][1] for key in self.PROJECTED}
        moved = dict(projected)
        moved["services.postgresql.username"] = "someone-else"

        errors = reach_override_errors(config, moved)

        assert len(errors) == 1
        assert "services.postgresql.username" in errors[0]
        assert "someone-else" in errors[0]


@pytest.mark.skipif(
    not any(f.name == "data_bundle" for f in dataclasses.fields(BuildProfile)),
    reason="`data_bundle` has left the profile schema, so the reading cannot depend on it",
)
class TestTheReadingDoesNotOpenAnAppTemplate:
    """The host reading renders the framework's own ``config.yml`` template.

    While the field still exists, the sharpest statement of that is a profile
    naming a bundle that is not on disk: reading through the bundle raised
    ``Template ... not found`` before the reading stopped consulting it.
    """

    def test_a_bundle_name_that_is_not_on_disk_still_renders(self) -> None:
        profile = _profile(**{"services.bluesky.path": "./services/bluesky"})
        profile = dataclasses.replace(profile, data_bundle="no_such_app_template")

        reading = _host_reading(profile)

        assert reading["services"]["bluesky"]["port"] == layout_ports(MOVED_BASE)["bluesky"]
