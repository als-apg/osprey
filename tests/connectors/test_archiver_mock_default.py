"""An unset ``archiver.type`` must resolve to the mock archiver, never to EPICS.

A config that omits the ``archiver:`` section is under-specified rather than a
declaration that a facility archiver exists. Defaulting such a config to
``epics_archiver`` sent it straight into ``ValueError: archiver URL is
required`` on the first ``archiver_read``, so the archive row of
:func:`~osprey_connectors.service_targets.service_targets` resolves it to the
mock archiver, on every target — the same fail-closed shape
``control_system.type`` uses. The table is pure and logs nothing: the build's
archiver pairing, not a runtime warning, is what tells an operator.

The shipped presets are pinned here too: the fallback is a safety net, not a
substitute for a config that states what it uses.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey_connectors import connector_types
from osprey_connectors.archiver import ArchiveCoverage, ChannelArchive
from osprey_connectors.archiver._mock_archiver_connector import MockArchiverConnector
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.service_answers import Unavailable
from osprey_connectors.service_targets import service_targets
from osprey_connectors_tests._harness import isolated_connector_registries
from tests.facility.served_tree import in_process_config, served_tree

#: Every bundled preset that grants ``archiver_read`` an approval policy, and
#: so has to say which archiver that tool reaches. The framework template
#: renders no ``archiver`` block at all — this comes from the preset's own
#: ``config:``.
PRESETS_WITH_ARCHIVER = ["hello-world", "demo"]


def _cfg(preset: str) -> dict:
    """The preset's resolved ``config:`` block, dotted keys folded in."""
    profile, _profile_dir = resolve_build_profile(None, preset)
    return _expand_dotted(profile.config)


class _LiveArchiverTripwire(ChannelArchive):
    """Registered as EPICS so a regressed fallback fails loudly instead of dialing out."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "archiver.type fallback selected the EPICS archiver; "
            "an unset key must resolve to the mock archiver"
        )

    async def history(  # pragma: no cover - never built
        self,
        addresses: Sequence[str],  # noqa: ARG002 - never built
        **_kwargs: object,
    ) -> pd.DataFrame | Unavailable:
        raise AssertionError

    async def coverage(  # pragma: no cover - never built
        self,
        addresses: Sequence[str],  # noqa: ARG002 - never built
        *,
        timeout_s: float | None = None,  # noqa: ARG002 - never built
    ) -> Mapping[str, ArchiveCoverage] | Unavailable:
        raise AssertionError


@pytest.fixture(autouse=True)
def registered_archivers():
    """Mock plus an EPICS tripwire, so 'not epics_archiver' is proven, not assumed."""
    with isolated_connector_registries(clear=True):
        ConnectorFactory.register_service(
            "archive", connector_types.MOCK_ARCHIVER, MockArchiverConnector
        )
        ConnectorFactory.register_service(
            "archive", connector_types.EPICS_ARCHIVER, _LiveArchiverTripwire
        )
        yield


@pytest.fixture
def rendered_view(tmp_path, monkeypatch):
    """The render of a built tree as the working directory, its config beside its view.

    A config section that names no view reaches the one beside the loaded
    config, as a rendered deployment does.
    """
    view = served_tree(tmp_path, readings=["BEAM:CURRENT"])
    render = view.parent.parent
    (render / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    monkeypatch.chdir(render)
    return view


async def _the_mock_on_every_target(config: object) -> None:
    """*config*'s archive row: the mock on every target, built and serving the view."""
    table = service_targets(config)
    for target in connector_types.CONTROL_TARGETS:
        spec = table.archive[target]
        assert spec is not None, target
        assert spec.type == connector_types.MOCK_ARCHIVER
        archive = await ConnectorFactory.create_service(spec)
        assert isinstance(archive, MockArchiverConnector)
        end = datetime(2024, 1, 1, tzinfo=UTC)
        frame = await archive.history(
            ["BEAM:CURRENT"], start_at=end - timedelta(minutes=1), end_at=end
        )
        assert isinstance(frame, pd.DataFrame)
        assert set(frame["address"]) == {"BEAM:CURRENT"}
        await archive.aclose()
    # The mock synthesizes: it holds no target's history.
    assert table.history_targets() == frozenset()


@pytest.mark.usefixtures("rendered_view")
class TestArchiverTypeFallback:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({}, id="no-archiver-section"),
            pytest.param({"archiver": {}}, id="empty-section"),
            # A commented-out or emptied YAML value parses as None; it must take
            # the same fail-closed path as a missing key rather than raising later.
            pytest.param({"archiver": {"type": None}}, id="blank-type"),
            pytest.param(None, id="no-config-at-all"),
        ],
    )
    async def test_an_unset_type_opens_the_mock(self, config, caplog):
        with caplog.at_level(logging.WARNING):
            await _the_mock_on_every_target(config)

        # The table is pure: the fallback is no longer announced at run time.
        assert "archiver.type" not in caplog.text

    @pytest.mark.asyncio
    async def test_explicit_type_is_honoured(self, rendered_view):
        config = {
            "archiver": {
                "type": connector_types.MOCK_ARCHIVER,
                connector_types.MOCK_ARCHIVER: in_process_config(rendered_view, sample_rate_hz=1.0),
            }
        }

        await _the_mock_on_every_target(config)

        spec = service_targets(config).archive[connector_types.TARGET_LIVE]
        assert spec is not None
        assert spec.settings["sample_rate_hz"] == 1.0


class TestShippedPresetsDeclareAnArchiver:
    @pytest.mark.parametrize("preset", PRESETS_WITH_ARCHIVER)
    def test_preset_names_the_archiver_its_approval_policy_covers(self, preset: str):
        # A preset that grants archiver_read an approval policy has to say which
        # archiver that tool reaches rather than leaning on the fallback.
        config = _cfg(preset)

        assert "archiver_read" in config["approval"]["tools"], (
            f"{preset} no longer grants archiver_read a policy — drop it from this matrix"
        )
        assert config["archiver"]["type"], f"{preset} grants archiver_read but names no archiver"

    def test_the_onboarding_preset_ships_the_mock_archiver(self):
        """hello-world runs no archive service, so it states the mock explicitly."""
        assert _cfg("hello-world")["archiver"]["type"] == connector_types.MOCK_ARCHIVER
