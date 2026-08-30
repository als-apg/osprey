"""The shared stand-in predicates — the derivations every reader agrees on.

:func:`live_standin_active` is the ``standin`` target's deployed-container
gate: the target eligibility check, the recorder's enablement gate and the
build's gateway derivation all ask it whether the endpoint a session would dial
is this deployment's own stand-in. These tests pin its three conjuncts, and in
particular the two cases where a loopback endpoint is *not* a stand-in: an SSH
tunnel into a real gateway, and a stale port left behind after a deployment
went live.

:func:`archive_belongs_to_standin` answers the other question — whose history
the deployment's store holds — from the deployment's shape rather than from any
endpoint. Its tests pin that it needs both the stand-in and the recorder, and
that it never reads an endpoint at all.
"""

from __future__ import annotations

import pytest

from osprey_connectors.standin import (
    ARCHIVER_RECORDER_BLOCK_KEY,
    ARCHIVER_RECORDER_SERVICE,
    LIVE_STANDIN_PORT_KEY,
    archive_belongs_to_standin,
    live_standin_active,
    live_standin_port,
)

STANDIN_PORT = 5074


def config_with_standin(port: object = STANDIN_PORT) -> dict:
    """A rendered config whose services block carries a stand-in on *port*."""
    return {
        "control_system": {"type": "virtual_accelerator"},
        "services": {"live_standin": {"path": "./services/virtual_accelerator", "port": port}},
    }


#: Distinguishes "caller said nothing" from a deployed_services value that *is*
#: ``None`` — one of the shapes the predicate has to answer False for.
UNSET = object()


def config_with_recorder(port: object = STANDIN_PORT, deployed: object = UNSET) -> dict:
    """:func:`config_with_standin` plus a ``deployed_services`` list.

    *deployed* defaults to a deployment that runs the recorder beside the
    stand-in — the shape the predicate is about.
    """
    config = config_with_standin(port)
    config["deployed_services"] = (
        ["virtual_accelerator", "mongodb", ARCHIVER_RECORDER_SERVICE]
        if deployed is UNSET
        else deployed
    )
    return config


class TestActive:
    """Endpoints that are the deployment's own stand-in."""

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LocalHost", "::1", "127.0.0.5"])
    def test_loopback_host_on_the_stated_port_is_the_standin(self, host: str) -> None:
        """Every spelling of "this host" reaches the same verdict."""
        assert live_standin_active(
            config_with_standin(), endpoint_host=host, endpoint_port=STANDIN_PORT
        )

    def test_a_persona_render_carrying_only_the_projected_port_is_the_standin(self) -> None:
        """No ``deployed_services`` conjunct: the projected port is the whole
        evidence, so an attached render answers as the single-user one does."""
        persona_render = {"services": {"live_standin": {"port": STANDIN_PORT}}}

        assert live_standin_active(
            persona_render, endpoint_host="127.0.0.1", endpoint_port=STANDIN_PORT
        )

    def test_the_port_may_be_stated_as_text(self) -> None:
        """A YAML-quoted port still names the port it names."""
        assert live_standin_active(
            config_with_standin("5074"), endpoint_host="127.0.0.1", endpoint_port=STANDIN_PORT
        )


class TestNotActive:
    """Endpoints that are a real machine, and must be labelled as one."""

    def test_no_services_block_at_all(self) -> None:
        """A deployment that stood no stand-in up has none to claim."""
        assert not live_standin_active(
            {"control_system": {"type": "epics"}},
            endpoint_host="127.0.0.1",
            endpoint_port=STANDIN_PORT,
        )

    def test_ssh_tunnel_to_localhost_without_a_standin_block(self) -> None:
        """A forwarded real gateway is loopback and nothing else — still LIVE."""
        tunnelled = {"services": {"openobserve": {"port": 5080}}}

        assert not live_standin_active(tunnelled, endpoint_host="localhost", endpoint_port=5064)

    def test_stale_port_after_the_deployment_went_live(self) -> None:
        """The label follows the endpoint, never a leftover services block."""
        assert not live_standin_active(
            config_with_standin(), endpoint_host="127.0.0.1", endpoint_port=5064
        )

    def test_non_loopback_host_on_the_matching_port(self) -> None:
        """An SSH-tunnel-style named gateway is off this host, so it is not ours."""
        assert not live_standin_active(
            config_with_standin(), endpoint_host="cagw.example.com", endpoint_port=STANDIN_PORT
        )

    @pytest.mark.parametrize("host", ["", "   ", "not a host", "127.0.0.1:5074", "[::1]"])
    def test_unreadable_host_fails_toward_live_machine(self, host: str) -> None:
        """A host this module cannot read is a machine it cannot vouch for."""
        assert not live_standin_active(
            config_with_standin(), endpoint_host=host, endpoint_port=STANDIN_PORT
        )

    def test_unresolved_endpoint_port(self) -> None:
        """No port dialled means no endpoint to match against."""
        assert not live_standin_active(
            config_with_standin(), endpoint_host="127.0.0.1", endpoint_port=None
        )


class TestPort:
    """The port accessor the build and the recorder gate read on its own."""

    def test_present(self) -> None:
        assert live_standin_port(config_with_standin()) == STANDIN_PORT

    def test_absent(self) -> None:
        assert live_standin_port({"services": {}}) is None

    def test_absent_from_a_config_with_no_services_section(self) -> None:
        assert live_standin_port({"control_system": {"type": "epics"}}) is None

    @pytest.mark.parametrize("value", ["", None, "not-a-port", True, {"port": 5074}, [5074]])
    def test_values_that_name_no_port(self, value: object) -> None:
        """Including ``true``, which asks for a stand-in without saying where."""
        assert live_standin_port(config_with_standin(value)) is None

    def test_the_dotted_key_is_the_one_the_build_projects(self) -> None:
        """Pinned because the reach contract projects this exact spelling."""
        assert LIVE_STANDIN_PORT_KEY == "services.live_standin.port"


class TestArchiveBelongsToStandin:
    """Whose history the deployment's store holds, read from its shape."""

    def test_a_standin_recorded_by_this_deployment(self) -> None:
        """The archive belongs to the machine it records, and that is the stand-in."""
        assert archive_belongs_to_standin(config_with_recorder())

    def test_the_recorder_may_be_the_only_deployed_service(self) -> None:
        """Nothing else in the list matters; the recorder is what writes the store."""
        assert archive_belongs_to_standin(config_with_recorder(deployed=["archiver_recorder"]))

    def test_no_standin_means_the_archive_is_the_facilitys(self) -> None:
        """A deployment recording a machine it did not stand in for records that machine."""
        assert not archive_belongs_to_standin(
            {
                "control_system": {"type": "epics"},
                "deployed_services": ["archiver_recorder"],
            }
        )

    def test_a_services_block_without_a_standin(self) -> None:
        """The recorder alone proves nothing about which machine it samples."""
        assert not archive_belongs_to_standin(
            {
                "services": {"openobserve": {"port": 5080}},
                "deployed_services": ["archiver_recorder"],
            }
        )

    def test_a_standin_with_no_recorder_deployed(self) -> None:
        """No recorder, no store of this deployment's own — nothing to belong."""
        assert not archive_belongs_to_standin(
            config_with_recorder(deployed=["virtual_accelerator", "mongodb"])
        )

    def test_no_deployed_services_key_at_all(self) -> None:
        """A stand-in on its own does not make an archive."""
        assert not archive_belongs_to_standin(config_with_standin())

    def test_an_empty_deployed_services_list(self) -> None:
        """An attached render lists nothing, and this one was told nothing either:
        no recorder is named in either spelling, so nothing says the store holds a
        stand-in's history."""
        assert not archive_belongs_to_standin(config_with_recorder(deployed=[]))

    def test_a_persona_render_carrying_only_the_projected_port(self) -> None:
        """A persona of a host that does not record: the port reached it, and there
        was no recorder to project."""
        persona_render = {"services": {"live_standin": {"port": STANDIN_PORT}}}

        assert not archive_belongs_to_standin(persona_render)

    @pytest.mark.parametrize(
        "deployed",
        ["archiver_recorder", {"archiver_recorder": True}, None, 1],
        ids=["bare-string", "mapping", "null", "int"],
    )
    def test_a_deployed_services_value_that_is_not_a_list(self, deployed: object) -> None:
        """Including the bare string, whose substring would otherwise answer True."""
        assert not archive_belongs_to_standin(config_with_recorder(deployed=deployed))

    @pytest.mark.parametrize("value", ["", None, "not-a-port", True, {"port": 5074}, [5074]])
    def test_a_port_value_that_names_no_port(self, value: object) -> None:
        """The same reading of the same key as ``live_standin_port``: a key that
        cannot be dialled is not a deployment saying it built a stand-in."""
        assert not archive_belongs_to_standin(config_with_recorder(value))

    def test_the_port_may_be_stated_as_text(self) -> None:
        """A YAML-quoted port still names the port it names."""
        assert archive_belongs_to_standin(config_with_recorder("5074"))

    def test_a_config_with_no_sections_at_all(self) -> None:
        assert not archive_belongs_to_standin({})

    def test_the_recorder_service_name_is_the_deployed_services_entry(self) -> None:
        """Pinned because the recorder's compose entry is registered under it."""
        assert ARCHIVER_RECORDER_SERVICE == "archiver_recorder"

    # The projected spelling: the same fact, in the render that is told it.
    # A web-terminal persona is an attached render: it lists no services, so
    # `deployed_services` cannot carry this fact, and the build projects the
    # recorder's own block from the host's render instead (the
    # `archiver_recorder` Reach Contract). The persona reads the host's store,
    # so it has to reach the host's answer.

    def test_a_persona_told_the_recorder_block_answers_like_its_host(self) -> None:
        """The parity case: no deployed_services, both projected facts, True."""
        assert archive_belongs_to_standin(
            {
                "services": {
                    "live_standin": {"port": STANDIN_PORT},
                    "archiver_recorder": {"path": "./services/archiver_recorder"},
                },
                "deployed_services": [],
            }
        )

    def test_the_block_alone_still_needs_a_standin(self) -> None:
        """Recording something is not recording a stand-in."""
        assert not archive_belongs_to_standin(
            {"services": {"archiver_recorder": {"path": "./services/archiver_recorder"}}}
        )

    def test_an_empty_recorder_block_says_nothing(self) -> None:
        """A null or empty stanza declares no service, the same coercion the
        templates make, so it is not a deployment saying it records."""
        config = config_with_standin()
        config["services"]["archiver_recorder"] = {}

        assert not archive_belongs_to_standin(config)

    @pytest.mark.parametrize(
        "block",
        [None, "", "./services/archiver_recorder", True, ["path"]],
        ids=["null", "empty-string", "string", "bool", "list"],
    )
    def test_a_recorder_block_that_is_not_a_mapping(self, block: object) -> None:
        """The injector writes a mapping and the projection copies a leaf into
        one; anything else is not that block."""
        config = config_with_standin()
        config["services"]["archiver_recorder"] = block

        assert not archive_belongs_to_standin(config)

    def test_the_block_key_is_the_one_the_build_projects(self) -> None:
        """Pinned because the reach contract projects a leaf of this exact block."""
        assert ARCHIVER_RECORDER_BLOCK_KEY == "services.archiver_recorder"
