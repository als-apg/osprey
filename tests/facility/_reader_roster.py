"""The roster of every reader of the built ``facility.json``.

One entry per function that reads the facility document, or checks one it
holds: the function, what it does with a document it cannot read (its
channel), and the node id of the test that hands it a ``/3`` document and
proves that channel. The reader-coverage test holds the roster complete and
each node id collected; the facility-json contract page lists its direct
readers.

A task that makes a reader check its header adds that reader's line here, and
touches only its own lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: What a reader does with a document whose header it does not read.
#: ``refuse``: it raises, naming the rebuild. ``report-corrupt``: it answers
#: with a corrupt-source report the operator sees. ``degrade-absent``: it
#: carries on as if no document were there.
Channel = Literal["refuse", "report-corrupt", "degrade-absent"]

#: Every channel a roster entry may name.
CHANNELS: frozenset[str] = frozenset({"refuse", "report-corrupt", "degrade-absent"})


@dataclass(frozen=True)
class Reader:
    """One reader of the facility document.

    Attributes:
        function: The reader's qualified name, ``module.function`` or
            ``module.Class.method``.
        channel: What it does with a document it cannot read.
        test: The node id of the test that hands it a ``/3`` document.
    """

    function: str
    channel: Channel
    test: str


ROSTER: tuple[Reader, ...] = (
    Reader(
        "osprey_facility.reading.load_facility",
        "refuse",
        "tests/facility/test_reading.py::test_a_v3_document_is_refused",
    ),
    Reader(
        "osprey_facility.reading.check_header",
        "refuse",
        "tests/facility/test_reading.py::test_check_header_refuses_a_v3_document",
    ),
    Reader(
        "osprey.runtime.guarded_run._facility_commands",
        "refuse",
        "tests/runtime/test_command_write.py::test_a_v3_facility_definition_means_no_command_fires",
    ),
    Reader(
        "osprey.simulation.model_request._corrector_delta",
        "refuse",
        "tests/simulation/test_response_kind.py::TestFacilityFile::"
        "test_a_v3_facility_file_is_refused_naming_the_rebuild",
    ),
    Reader(
        "osprey.mcp_server.python_executor.tools.pyaml_measure._facility",
        "refuse",
        "tests/mcp_server/python_executor/test_pyaml_measure_tool.py::"
        "test_a_v3_facility_file_is_refused_before_any_write",
    ),
    Reader(
        "osprey.mcp_server.python_executor.tools._pyaml_measure_result.measure_labels",
        "refuse",
        "tests/mcp_server/python_executor/test_pyaml_measure_result.py::"
        "test_labels_refuse_a_v3_facility_file_naming_the_rebuild",
    ),
    Reader(
        "osprey.apps.sdk.testing._deployment.RenderedTree.read",
        "refuse",
        "tests/apps/test_sdk_testing_fixture.py::"
        "test_a_render_whose_facility_file_is_v3_is_refused_naming_the_rebuild",
    ),
    Reader(
        "osprey_facility.records.FacilityRecords.from_file",
        "refuse",
        "tests/facility/test_records.py::test_from_file_refuses_a_v3_document",
    ),
    Reader(
        "osprey.apps.sdk._model.model_members",
        "refuse",
        "tests/apps/test_sdk_model.py::TestModelMembers::test_a_v3_facility_file_is_refused",
    ),
    Reader(
        "osprey.services.ariel_search.ingestion.facility_references.load_facility_records",
        "degrade-absent",
        "tests/services/ariel_search/test_facility_references.py::"
        "test_a_v3_facility_file_means_no_facility_records_and_a_warning",
    ),
    Reader(
        "osprey.interfaces.okf_panel.app._load_facility_records",
        "degrade-absent",
        "tests/interfaces/okf_panel/test_app.py::"
        "test_concept_with_a_v3_facility_file_marks_links_unresolved",
    ),
    Reader(
        "osprey.facility.views.facts.parsed_facility_file",
        "refuse",
        "tests/facility/test_facts_view.py::test_the_shared_parse_refuses_a_v3_document",
    ),
    Reader(
        "osprey.channel_roster.sources.read_facility_roster",
        "report-corrupt",
        "tests/channel_roster/test_sources_facility.py::TestTheHeader::"
        "test_a_v3_document_is_a_corrupt_source_naming_the_rebuild",
    ),
    Reader(
        "osprey.integrations.canopy.screen_guide._guide_facility",
        "degrade-absent",
        "tests/cli/test_canopy_screens_skill.py::"
        "test_a_v3_facility_file_is_not_used_and_the_warning_names_the_rebuild",
    ),
    Reader(
        "osprey.cli.templates.claude_code._read_facility",
        "degrade-absent",
        "tests/cli/test_claude_code_templates.py::"
        "test_a_v3_facility_file_is_not_used_and_the_warning_names_the_rebuild",
    ),
    Reader(
        "osprey.utils.facility._recorded_identity",
        "refuse",
        "tests/utils/test_facility_identity.py::"
        "test_a_v3_facility_file_is_refused_naming_the_rebuild",
    ),
    Reader(
        "osprey.utils.facility.facility_name",
        "refuse",
        "tests/utils/test_facility_identity.py::"
        "test_the_facility_name_of_a_v3_facility_file_is_refused_naming_the_rebuild",
    ),
    Reader(
        "osprey.runtime._facility_channels",
        "refuse",
        "tests/runtime/test_settle.py::test_a_v3_facility_file_is_refused_naming_the_rebuild",
    ),
    Reader(
        "osprey.mcp_server.channel_finder_hierarchical.server_context.ChannelFinderHierContext.initialize",
        "refuse",
        "tests/interfaces/test_channel_finder_facility_name.py::"
        "test_a_v3_facility_file_refuses_the_pipeline_context_naming_the_rebuild",
    ),
    Reader(
        "osprey.mcp_server.channel_finder_middle_layer.server_context.ChannelFinderMLContext.initialize",
        "refuse",
        "tests/interfaces/test_channel_finder_facility_name.py::"
        "test_a_v3_facility_file_refuses_the_pipeline_context_naming_the_rebuild",
    ),
    Reader(
        "osprey.mcp_server.channel_finder_in_context.server_context.ChannelFinderICContext.initialize",
        "refuse",
        "tests/interfaces/test_channel_finder_facility_name.py::"
        "test_a_v3_facility_file_refuses_the_pipeline_context_naming_the_rebuild",
    ),
    Reader(
        "osprey.interfaces.channel_finder.app._create_lifespan",
        "refuse",
        "tests/interfaces/test_channel_finder_facility_name.py::"
        "test_a_v3_facility_file_refuses_the_app_start_naming_the_rebuild",
    ),
    Reader(
        "osprey.deployment.web_terminals.artifacts.resolve_render_inputs",
        "refuse",
        "tests/deployment/web_terminals/test_render_facility_name.py::"
        "test_a_v3_build_facility_file_refuses_the_deploy_naming_the_rebuild",
    ),
    Reader(
        "osprey.facility.views.facts.read_facts",
        "refuse",
        "tests/facility/test_identity_fallback_readers.py::"
        "test_read_facts_without_a_facts_file_refuses_a_v3_facility_file",
    ),
    Reader(
        "osprey.mcp_server.python_executor.server.pyaml_view_present",
        "refuse",
        "tests/facility/test_identity_fallback_readers.py::"
        "test_the_pyaml_view_probe_refuses_a_v3_facility_file",
    ),
    Reader(
        "osprey.mcp_server.python_executor.tools.pyaml_measure.advertised",
        "refuse",
        "tests/facility/test_identity_fallback_readers.py::"
        "test_the_measure_tool_registration_refuses_a_v3_facility_file",
    ),
    Reader(
        "osprey.mcp_server.python_executor.tools._pyaml_measure_args._facts",
        "refuse",
        "tests/facility/test_identity_fallback_readers.py::"
        "test_the_measure_arguments_refuse_a_v3_facility_file_as_a_validation_error",
    ),
    Reader(
        "osprey.cli.templates.claude_code.config_derived_context",
        "refuse",
        "tests/cli/test_claude_code_templates.py::"
        "test_a_v3_facility_file_refuses_the_context_naming_the_rebuild",
    ),
    Reader(
        "osprey.apps.sdk._model.closed_orbit_models",
        "refuse",
        "tests/facility/test_identity_fallback_readers.py::"
        "test_the_closed_orbit_models_refuse_a_v3_facility_file_as_a_value_error",
    ),
    Reader(
        "osprey.apps.sdk._facts.read_facts",
        "refuse",
        "tests/facility/test_identity_fallback_readers.py::"
        "test_the_sdk_read_facts_refuses_a_v3_facility_file_as_a_value_error",
    ),
    Reader(
        "osprey.cli.build_injectors._inject_dispatch",
        "refuse",
        "tests/facility/test_identity_fallback_readers.py::"
        "test_the_dispatch_build_step_refuses_a_v3_facility_file",
    ),
    Reader(
        "osprey_panels.cli._graph",
        "refuse",
        "packages/osprey-panels/tests/osprey_panels_tests/test_cli.py::"
        "test_a_facility_3_document_is_refused_with_the_rebuild_hint",
    ),
)
