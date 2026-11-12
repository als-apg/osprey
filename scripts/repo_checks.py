"""The framework's checks: what holds for the repository as a whole and what
judges the framework's code, each declared with everything the check command
does with it.

``scripts/check.py`` collects this module first and every other
``repo_checks.py`` after it; this module's docstring in ``check.py`` names the
four kinds and the two tiers. A check another package owns is declared in that
package's own ``repo_checks.py``.

Standard library only, for the reason ``scripts/check.py`` gives.
"""

from __future__ import annotations

import ast
import dataclasses
import glob
import importlib.util
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType

import check_kit as kit
from check_kit import (
    BOTH,
    CLEAN,
    FAILED_EXIT,
    FULL,
    NOT_TOUCHED,
    PATHS,
    SPENDS,
    UV_RUN,
    WHOLE,
    Check,
    Context,
    EnvironmentFault,
    Lane,
    Refused,
    Selection,
    Venue,
)


@dataclass(frozen=True)
class UnitSelection:
    """The one statement of the parallel pytest sweep."""

    #: The paths the whole sweep collects.
    roots: tuple[str, ...]
    #: Each path the sweep leaves out, with the check or lane that runs it.
    ignores: tuple[tuple[str, str], ...]
    marker: str
    quick_marker: str
    #: Shard name to (paths, extra ignores); the shards cover the roots exactly once.
    #: An ignore holding a glob character is passed as ``--ignore-glob``.
    shards: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]
    dist: tuple[str, ...]
    #: Subset name to the test files and node IDs it runs in place of the
    #: roots. Selected with ``--shard``, but outside the partition.
    subsets: dict[str, tuple[str, ...]]


#: The tests that pin this declaration and the repository-wide rules, and
#: every test that reads the whole tests tree: a change anywhere in the tree
#: can break them, so no change selection may leave them out.
INVARIANT_TESTS = (
    "tests/docs/test_shipped_identity_literals.py",
    "tests/scripts/test_ruff_file_selection.py",
    "tests/scripts/test_check.py",
    "tests/scripts/test_changelog_fragments.py",
    "tests/apps/test_no_trajectory_service.py",
    "tests/services/ariel_search/test_store_deadline_guard.py",
    "tests/interfaces/test_js_module_tests.py",
    "tests/interfaces/test_page_ready_guard.py",
    "tests/interfaces/test_page_ready_pages.py",
    "tests/deployment/test_ci_workflow_wiring.py",
    "tests/deployment/test_e2e_scenario_pin.py",
    "tests/deployment/test_retired_spellings_at_parse_point.py",
    "tests/deployment/web_terminals/test_roster_entries.py",
    "tests/infrastructure/test_import_time_audit.py",
    "tests/infrastructure/test_stand_in_canaries.py",
    "tests/test_offline_build_env.py",
    "tests/test_api_marker_placement.py",
    "tests/test_graphdb_container.py",
    "tests/test_mongo_container.py",
    "tests/test_no_legacy_symbols.py",
    "tests/test_wait_ceilings.py",
)

LIVE_CA_VENUE = "linux-x86_64"
DARWIN_VENUE = "darwin"

#: The readonly gate's warm-up module: which stdlib lookups load native code
#: lazily depends on the OS and the interpreter.
READONLY_WARMUPS_TEST_FILE = "tests/runtime/test_raw_put_block_readonly_warmups.py"

#: The bundled applications, each with its own panel and Vitest tests.
PRESET_APPS_ROOT = "src/osprey/profiles/presets/apps/"

#: The template `osprey app new` copies, with its own panel and Vitest tests.
SCAFFOLD_APP_ROOT = "src/osprey/apps/scaffold-app/"

#: The scope of ``tsc``: the include globs of ``tsconfig.json`` and their inputs.
TSC_SCOPE = (
    "src/osprey/interfaces/vendor-globals.d.ts",
    "src/osprey/interfaces/**/static/js/**/*.js",
    "src/osprey/interfaces/bluesky_web/panels/**/*.js",
    f"{PRESET_APPS_ROOT}*/panel/**/*.js",
    f"{PRESET_APPS_ROOT}*/tests/**/*.js",
    f"{PRESET_APPS_ROOT}*/tests/**/*.mjs",
    f"{SCAFFOLD_APP_ROOT}panel/**/*.js",
    f"{SCAFFOLD_APP_ROOT}tests/**/*.js",
    f"{SCAFFOLD_APP_ROOT}tests/**/*.mjs",
    "src/osprey/interfaces/optimizer_web/panels/**/*.js",
    "tests/**/*.js",
    "tests/**/*.mjs",
    "tsconfig.json",
    "src/osprey/interfaces/web_terminal/static/vendor/dock-engine.d.mts",
    "package*.json",
)

#: The ruff checks print GitHub annotations when they run on GitHub Actions.
GITHUB_FORMAT = ("--output-format=github",)

#: The documentation's directory, where its build runs.
DOCS_DIR = "docs"

#: The quick tier's bound for the parallel sweep; the full tier's is the check's own.
#: It stops a sweep that outlives its last test (a teardown hang), never a slow
#: one, so it sits at about twice the sweep's runtime on a loaded host.
QUICK_PYTEST_BOUND_S = 1800

#: Where the parallel sweep finds the tests of a changed source file, first match
#: wins. ``package`` maps the nearest package, or failing that its nearest parent,
#: whose directory under ``tests/`` holds a test module, never across a directory
#: with its own ``pyproject.toml``; ``file`` maps a top-level
#: ``scripts/<name>.py`` to ``tests/scripts/test_<name>.py``.
#: A source file under a root whose rule finds nothing widens the sweep.
TEST_MIRRORS: tuple[tuple[str, str], ...] = (
    ("src/osprey/", "package"),
    ("scripts/", "file"),
)

#: The test directories of the ``rest-a`` shard.
_REST_A = ("tests/facility", "tests/deployment")

#: The test directories of the ``rest-c`` shard. None of them uses a session
#: container another shard uses (agent_record's ClickHouse is its own; the
#: shared Mongo, Neo4j and Postgres sessions stay in ``rest-b``), so no
#: container is started in two jobs.
_REST_C = (
    "tests/interfaces",
    "tests/mcp_server",
    "tests/scripts",
    "tests/hooks",
    "tests/agent_record",
    "tests/infrastructure",
    "tests/docs",
    "tests/templates",
)

#: The tests/cli modules of the ``cli-a`` shard, by the first letter of the name
#: after ``test_``: a-h with tests/cli/templates is about half the directory's
#: test time. ``cli-b`` takes every other top-level module, so a new module
#: lands in one shard by its name; tests/deployment/test_ci_workflow_wiring.py
#: holds that every tests/cli module is in exactly one.
_CLI_A_LETTERS = "a-h"

#: What the macOS cell runs: the test files, and the tests of a file (a node
#: ID, ``path::name`` or ``path::Class::name``, every parametrization of it),
#: whose outcome can differ on macOS through a named mechanism, one per group
#: below. Everything else only repeats ubuntu, so a new test file runs on
#: ubuntu alone until it is added here. The list was drawn by reading every
#: test file the cell ran, the source each reaches for a darwin branch, and
#: the CI logs for tests that skip on ubuntu and pass on macOS;
#: tests/scripts/test_check.py holds that each entry exists and that a test
#: file comparing the platform to darwin is listed.
_MACOS_RUNS = (
    # The test, or the code it reaches, reads the platform and takes its darwin branch.
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_write_authorization.py",
    "tests/agent_record/test_host_record.py::test_a_busy_forward_skips_the_round",
    "tests/agent_record/test_host_record.py::test_a_deployment_without_a_host_identity_is_refused",
    "tests/agent_record/test_host_record.py::test_a_drained_read_forwards_first_in_a_host_launchs_children",
    "tests/agent_record/test_host_record.py::test_a_failed_first_pass_answers_503_so_no_one_waits_on_it",
    "tests/agent_record/test_host_record.py::test_a_foreign_port_that_is_not_ready_is_refused_at_the_health_bound",
    "tests/agent_record/test_host_record.py::test_a_forward_resumes_from_its_own_position",
    "tests/agent_record/test_host_record.py::test_a_forwarded_entry_is_scrubbed_of_the_build_secrets",
    "tests/agent_record/test_host_record.py::test_a_host_root_that_does_not_exist_yet_is_empty_not_an_error",
    "tests/agent_record/test_host_record.py::test_a_lease_written_under_one_zone_is_live_under_another",
    "tests/agent_record/test_host_record.py::test_a_live_pid_with_another_start_time_is_a_dead_lease",
    "tests/agent_record/test_host_record.py::test_a_live_port_that_is_not_ours_is_never_spawned_over",
    "tests/agent_record/test_host_record.py::test_a_live_stack_marker_is_never_spawned_under",
    "tests/agent_record/test_host_record.py::test_a_missing_host_login_names_the_build_and_the_run_time_mount",
    "tests/agent_record/test_host_record.py::test_a_missing_service_file_names_the_build_and_the_run_time_mount",
    "tests/agent_record/test_host_record.py::test_a_missing_service_file_spawns_nothing",
    "tests/agent_record/test_host_record.py::test_a_prefix_must_be_a_relative_directory_path",
    "tests/agent_record/test_host_record.py::test_a_reuser_that_outlives_the_starter_leaves_no_process",
    "tests/agent_record/test_host_record.py::test_a_start_time_reads_the_same_under_any_time_zone",
    "tests/agent_record/test_host_record.py::test_a_takeover_files_nothing_twice_and_reads_no_sibling",
    "tests/agent_record/test_host_record.py::test_an_unhealthy_port_that_turns_healthy_is_reused",
    "tests/agent_record/test_host_record.py::test_forward_for_drain_reads_the_rendered_config",
    "tests/agent_record/test_host_record.py::test_shutdown_waits_for_the_first_pass_before_closing_the_backend",
    "tests/agent_record/test_host_record.py::test_the_clickhouse_refusal_creates_no_file",
    "tests/agent_record/test_host_record.py::test_the_idle_stop_is_decided_under_the_ensure_lock",
    "tests/agent_record/test_host_record.py::test_the_lease_watch_skips_a_round_while_an_ensure_holds_the_lock",
    "tests/agent_record/test_host_record.py::test_the_process_is_gone_when_the_last_holder_returns",
    "tests/agent_record/test_host_record.py::test_the_repr_keeps_the_token_out",
    "tests/agent_record/test_host_record.py::test_the_service_reads_only_the_host_root_on_the_host",
    "tests/agent_record/test_host_record.py::test_the_service_stops_itself_when_its_holder_dies",
    "tests/agent_record/test_host_record.py::test_the_starter_waits_out_its_own_childs_first_pass",
    "tests/agent_record/test_host_record.py::test_the_store_is_group_writable",
    "tests/agent_record/test_host_record.py::test_two_concurrent_ensures_share_one_process",
    "tests/agent_record/test_host_record.py::test_without_a_stack_marker_nothing_is_forwarded",
    "tests/agent_record/test_host_record_integration.py::test_a_host_chat_session_records_without_a_container",
    "tests/agent_record/test_host_record_integration.py::test_a_host_query_records_without_a_container",
    "tests/agent_record/test_host_record_integration.py::test_a_host_session_keeps_its_record_through_up_and_down",
    "tests/agent_record/test_host_record_integration.py::test_a_host_web_session_records_without_a_container",
    "tests/agent_record/test_host_record_integration.py::test_a_launcher_uses_the_stacks_record",
    "tests/agent_record/test_host_record_integration.py::test_two_concurrent_host_sessions_share_one_record_and_leave_none",
    "tests/cli/test_failure_replay.py",
    "tests/connectors/ipc/test_host_child.py",
    "tests/connectors/test_pva_venue_guard.py",
    "tests/deployment/test_docker_desktop.py",
    "tests/deployment/test_process_table.py",
    "tests/deployment/test_runtime_detection.py",
    "tests/deployment/test_subprocess_capture.py",
    "tests/imaging/test_render_worker.py::test_the_worker_entry_point_hardens_before_its_handshake",
    "tests/imaging/test_render_worker.py::test_worker_process_is_hardened_and_imports_no_service_code",
    "tests/imaging/test_render_worker.py::test_worker_refuses_a_picture_between_the_pixel_limit_and_twice_it",
    "tests/imaging/test_render_worker.py::test_worker_renders_the_large_jpegs",
    "tests/imaging/test_render_worker.py::test_worker_round_trip_and_clean_exit_on_end_of_input",
    "tests/infrastructure/test_ports.py",
    "tests/integrations/canopy/test_proven_commits.py::test_the_venue_is_this_interpreters_platform",
    "tests/interfaces/web_terminal/test_chat_process_tree.py",
    "tests/interfaces/web_terminal/test_handoff_process_tree.py",
    "tests/interfaces/web_terminal/test_jupyter_sidecar.py::test_a_sidecar_that_dies_later_is_reported_once_with_its_stderr_tail",
    "tests/interfaces/web_terminal/test_jupyter_sidecar.py::test_a_stop_during_wait_ready_ends_the_wait",
    "tests/interfaces/web_terminal/test_jupyter_sidecar.py::test_an_early_exit_surfaces_the_stderr_tail",
    "tests/interfaces/web_terminal/test_jupyter_sidecar.py::test_checkpoints_obey_the_same_confinement_as_the_contents_api",
    "tests/interfaces/web_terminal/test_jupyter_sidecar.py::test_on_exit_registered_after_the_death_fires_at_once",
    "tests/interfaces/web_terminal/test_jupyter_sidecar.py::test_raw_files_obey_the_same_confinement_as_the_contents_api",
    "tests/interfaces/web_terminal/test_jupyter_sidecar.py::test_stop_ends_a_member_of_the_sidecars_group_that_ignores_sigterm",
    "tests/interfaces/web_terminal/test_jupyter_sidecar.py::test_stop_leaves_no_process_and_removes_the_tempdir",
    "tests/interfaces/web_terminal/test_jupyter_sidecar.py::test_stop_names_what_its_group_stop_could_not_prove_gone",
    "tests/interfaces/web_terminal/test_jupyter_sidecar.py::test_stop_reports_no_exit_and_fires_no_callback",
    "tests/interfaces/web_terminal/test_jupyter_sidecar.py::test_stop_returns_while_an_orphan_still_holds_the_stderr_pipe",
    "tests/interfaces/web_terminal/test_jupyter_sidecar.py::test_the_exit_watcher_reports_without_reaping",
    "tests/interfaces/web_terminal/test_jupyter_sidecar.py::test_the_root_redirects_to_jupyterlab_not_the_server_landing_page",
    "tests/interfaces/web_terminal/test_jupyter_sidecar.py::test_the_runtime_dir_is_outside_the_shared_root",
    "tests/interfaces/web_terminal/test_jupyter_sidecar.py::test_the_sidecar_exits_when_its_parent_is_killed",
    "tests/interfaces/web_terminal/test_supervised_child.py",
    "tests/mcp_server/test_dispatch_worker_credentials.py",
    "tests/runtime/test_raw_put_block_readonly_warmups.py",
    "tests/services/code_review/test_switch.py",
    "tests/utils/test_facility_zone.py",
    "tests/va/test_live_ca_venue_guard.py::TestTheVenueIsPresentWhereItIsRequired::test_the_live_channel_access_venue_is_installed",
    # The readonly gate over real native loads: dlopen paths, library names and the
    # stdlib's lazy loads differ on darwin.
    "tests/mcp_server/python_executor/test_model_response_matrix.py::TestSolve::test_the_matrix_equals_the_plug_in_called_directly",
    "tests/runtime/test_raw_put_block_no_client.py::test_a_renamed_libca_is_refused_by_symbol",
    "tests/runtime/test_raw_put_block_no_client.py::test_cffi_dlopen_is_gated",
    "tests/runtime/test_raw_put_block_no_client.py::test_osprey_layer_imports_and_client_frameworks_are_refused",
    "tests/runtime/test_raw_put_block_readonly_loader.py",
    "tests/services/python_executor/execution/test_readonly_guard.py::test_guard_holds_against_installed_libraries",
    "tests/services/python_executor/execution/test_readonly_guard.py::test_readonly_aioca_caget_still_reads",
    "tests/services/python_executor/execution/test_readonly_guard.py::test_readonly_prewarms_ctypes_util",
    "tests/services/python_executor/execution/test_readonly_guard.py::test_readonly_prewarms_platform_processor",
    "tests/services/python_executor/execution/test_readonly_guard.py::test_readonly_refuses_the_installed_pvaccess_writes",
    # Native code built per platform: EPICS libca and soft IOCs, pyAT tracking,
    # DuckDB, p4p and libzmq, the claude binary.
    "packages/osprey-connectors/tests/osprey_connectors_tests/ipc/test_pool_authorization_soft_ioc.py",
    "packages/osprey-connectors/tests/osprey_connectors_tests/ipc/test_pool_soft_ioc.py",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_ca_bulk_path.py::TestOverTheWire::test_a_second_connector_on_another_route_repoints_the_first_and_it_comes_back",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_ca_bulk_path.py::TestOverTheWire::test_ioc_kill_and_restart_keeps_one_pair_and_delivers_each_value_once",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_ca_bulk_path.py::TestOverTheWire::test_subscribe_unsubscribe_subscribe_and_two_subscribers",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_ca_bulk_path.py::TestOverTheWire::test_values_limits_labels_and_one_record_per_address",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_ca_wait_stale.py::TestOverTheWire::test_a_kill_and_restart_reports_stale_exactly_once",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_ca_wait_stale.py::TestOverTheWire::test_a_one_shot_of_an_unknown_pv_on_an_idle_connector_is_not_stale",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_ca_wait_stale.py::TestOverTheWire::test_a_server_that_stays_gone_reports_stale_once_after_the_silent_window",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_ca_wait_stale.py::TestOverTheWire::test_a_udp_search_connector_never_reports_stale",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_ca_wait_stale.py::TestOverTheWire::test_names_the_ioc_does_not_serve_never_report_stale",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_pva_disconnect.py::TestLiveDisconnect::test_get_metadata_of_a_subscribed_address_awaits_its_first_update",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_pva_disconnect.py::TestLiveDisconnect::test_get_metadata_of_a_subscribed_unserved_address_times_out",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_pva_disconnect.py::TestLiveDisconnect::test_server_drop_reports_false_then_true_on_the_same_monitor",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_soft_ioc_fixture.py",
    "tests/agent_record/test_embedded_backend.py",
    "tests/agent_runner/test_task_tools_real_cli.py",
    "tests/agent_runner/test_unrendered_mcp_servers.py",
    "tests/connectors/test_epics_ca_softioc.py",
    "tests/connectors/test_pva_live_fixture.py",
    "tests/connectors/test_write_door_libca.py",
    "tests/facility/test_check_a.py::test_chromaticity_replays_the_model_file",
    "tests/facility/test_check_a.py::test_dispersion_replays_the_model_file",
    "tests/facility/test_check_a.py::test_family_response_replays_the_model_file",
    "tests/facility/test_check_a.py::test_orbit_response_replays_the_model_file",
    "tests/facility/test_check_a.py::test_ring_state_replays_the_model_file",
    "tests/facility/test_check_a.py::test_the_engine_builds_the_model_on_the_deck_the_build_wrote",
    "tests/facility/test_check_a.py::test_transport_dispersion_replays_the_model_file",
    "tests/facility/test_check_a.py::test_transport_orbit_response_replays_the_model_file",
    "tests/facility/test_check_a.py::test_tunes_replay_the_model_file",
    "tests/facility/test_check_a_hardware.py::test_line_dispersion_hardware_is_the_plug_in_twiss_dispersion",
    "tests/facility/test_check_a_hardware.py::test_line_orbit_hardware_is_the_plug_in_orbit_through_the_build_calibration",
    "tests/facility/test_mml_fixtures_wellformed.py::TestSyntheticExport::test_the_synthetic_fixture_regenerates_byte_identically",
    "tests/facility/test_mml_fixtures_wellformed.py::TestSyntheticExport::test_the_synthetic_ring_boots",
    "tests/facility/test_mml_synthetic_fixture.py",
    "tests/facility/test_response_check.py::test_a_measured_export_prints_its_wrapped_positions_then_one_line_and_exits_0",
    "tests/facility/test_response_check.py::test_spear3_holds_its_counts",
    "tests/interfaces/web_terminal/test_audit_routes.py::test_only_the_entries_this_identity_wrote_are_served",
    "tests/interfaces/web_terminal/test_process_tree.py",
    "tests/mcp_server/channel_finder_middle_layer/test_run_sql.py::test_an_umbrella_family_and_its_member_family_share_a_channel",
    "tests/mcp_server/channel_finder_middle_layer/test_run_sql.py::test_query_returns_columns_rows_and_count",
    "tests/mcp_server/workspace/test_sandbox_executor.py",
    "tests/runtime/test_jupyter_kernel_shell_rearm.py",
    "tests/scripts/test_bpm_polarity_picture.py::test_a_rerun_writes_the_same_bytes_and_the_committed_picture_is_the_same_picture",
    "tests/services/bluesky_bridge/test_document_plane.py",
    "tests/services/channel_finder/graph_index/test_open.py",
    "tests/services/channel_finder/graph_index/test_writer.py",
    "tests/services/python_executor/test_client_packages_cover_connectors.py::test_every_shipped_library_is_a_client_stem_or_reviewed",
    "tests/simulation/test_fidelity.py",
    "tests/simulation/test_lattice.py",
    # A real venv: its layout, its base interpreter and the wheels it installs.
    "tests/cli/test_freeze_base_environment.py",
    "tests/test_offline_build_env.py::test_a_build_installs_osprey_with_no_route_to_an_index",
    # Real children signalled, killed or reaped: process groups, signal delivery,
    # fork, ptys and the event loop's child watcher.
    "packages/osprey-connectors/tests/osprey_connectors_tests/process/test_process.py",
    "tests/cli/test_agent_record_cmd.py::test_a_clickhouse_project_with_no_stack_is_refused_naming_the_fix",
    "tests/cli/test_agent_record_cmd.py::test_export_on_a_no_stack_project_runs_catch_up_and_leaves_no_process",
    "tests/cli/test_agent_record_cmd.py::test_health_on_a_no_stack_project_answers_from_the_record_and_leaves_no_process",
    "tests/cli/test_build_lifecycle_drain.py",
    "tests/cli/test_build_swap.py::TestServedFilesSurviveABuild::test_the_served_files_are_intact_after_a_build",
    "tests/cli/test_build_swap.py::TestSwapRecovery::test_a_leftover_that_cannot_even_be_moved_names_the_path_and_the_reason",
    "tests/cli/test_build_swap.py::TestSwapRecovery::test_a_present_build_wins_over_leftovers",
    "tests/cli/test_build_swap.py::TestSwapRecovery::test_an_incoming_render_is_adopted_when_build_is_gone",
    "tests/cli/test_build_swap.py::TestSwapRecovery::test_an_outgoing_render_is_adopted_when_nothing_else_is_left",
    "tests/cli/test_build_swap.py::TestSwapRecovery::test_an_undeletable_leftover_is_moved_aside_rather_than_left_in_the_way",
    "tests/cli/test_build_swap.py::test_a_killed_build_leaves_a_build_that_can_still_stop_the_stack",
    "tests/cli/test_feedback_cmd.py::test_export_resolves_the_referenced_steps_and_leaves_no_process",
    "tests/cli/test_feedback_cmd.py::test_list_on_a_no_stack_project_sees_every_users_pending_lines_and_leaves_no_process",
    "tests/cli/test_health_cmd.py::TestNoHang::test_subprocess_no_hang_hung_plugin_exits",
    "tests/cli/test_host_launch_record.py::test_the_supervisor_restarts_the_record_stopped_mid_session",
    "tests/cli/test_host_launch_record.py::test_web_ends_the_lease_when_stopped_by_sigterm",
    "tests/cli/test_host_launch_record.py::test_web_stop_ends_the_lease_and_leaves_no_record",
    "tests/cli/test_host_launch_record.py::test_web_stop_leaves_a_record_another_session_still_holds",
    "tests/connectors/ipc/test_pool.py::test_a_child_starts_on_first_use_and_is_shared_by_its_key",
    "tests/connectors/ipc/test_pool.py::test_a_child_that_exits_before_answering_init_is_refused_at_the_init_stage",
    "tests/connectors/ipc/test_pool.py::test_a_child_that_never_finishes_connecting_hits_the_start_timeout",
    "tests/connectors/ipc/test_pool.py::test_a_killed_child_fails_its_call_once_and_the_next_call_respawns",
    "tests/connectors/ipc/test_pool.py::test_a_project_env_file_puts_no_epics_variable_back_into_a_child",
    "tests/connectors/ipc/test_pool.py::test_a_wedged_child_is_killed_and_replaced",
    "tests/connectors/ipc/test_pool.py::test_children_do_not_outlive_a_parent_that_is_killed",
    "tests/connectors/ipc/test_pool.py::test_close_during_a_start_leaves_nothing_running",
    "tests/connectors/ipc/test_pool.py::test_close_stops_every_child",
    "tests/connectors/ipc/test_pool.py::test_close_with_a_call_in_flight_fails_it_as_stopped",
    "tests/connectors/ipc/test_pool.py::test_failed_starts_leave_no_unhandled_asyncio_errors",
    "tests/connectors/ipc/test_pool.py::test_with_kill_on_write_timeout_a_write_that_misses_its_deadline_kills_its_child",
    "tests/hooks/test_watchdog_trigger.py",
    "tests/imaging/test_render_isolated.py",
    "tests/imaging/test_render_loop_lifetime.py",
    "tests/infrastructure/test_ci_diagnostics.py::test_a_capture_past_an_armed_timeout_leaves_the_worker_alive",
    "tests/infrastructure/test_ci_diagnostics.py::test_a_capture_whose_first_call_times_out_is_marked_incomplete",
    "tests/infrastructure/test_ci_diagnostics.py::test_snapshot_is_taken_before_the_module_fixture_tears_down",
    "tests/infrastructure/test_nested_pytest.py",
    "tests/integration/test_mcp_handshake_budget.py",
    "tests/integration/test_target_switch_mock_pair.py",
    "tests/integration/test_watchdog_non_operator.py::TestWatchedPrompts::test_positive_control_without_the_skip_stamp_a_verdict_is_recorded",
    "tests/integration/test_watchdog_non_operator.py::test_the_settle_wait_sees_a_lingering_child",
    "tests/integrations/canopy/test_sidecar.py",
    "tests/interfaces/test_watchdog_e2e.py::test_a_stop_a_running_command_outlives_keeps_the_dialog_until_continue",
    "tests/interfaces/web_terminal/test_pty_manager.py::TestPtyChildWaits",
    "tests/interfaces/web_terminal/test_pty_manager.py::TestPtyRegistry",
    "tests/interfaces/web_terminal/test_pty_manager.py::TestPtySession::test_is_alive_never_reaps_the_child",
    "tests/interfaces/web_terminal/test_pty_manager.py::TestPtySession::test_resize",
    "tests/interfaces/web_terminal/test_pty_manager.py::TestPtySession::test_sigwinch_delivered_on_resize",
    "tests/interfaces/web_terminal/test_pty_manager.py::TestPtySession::test_start_with_custom_dimensions",
    "tests/interfaces/web_terminal/test_pty_manager.py::TestPtySession::test_start_with_cwd",
    "tests/interfaces/web_terminal/test_pty_manager.py::TestPtySession::test_started_commands_lists_only_groups_of_their_own",
    "tests/interfaces/web_terminal/test_pty_manager.py::TestPtySession::test_started_commands_of_a_dead_child_is_empty",
    "tests/interfaces/web_terminal/test_pty_manager.py::TestPtySession::test_terminate_cleans_up",
    "tests/interfaces/web_terminal/test_pty_manager.py::TestPtySession::test_terminate_ends_a_helper_that_outlives_the_hang_up",
    "tests/interfaces/web_terminal/test_pty_manager.py::TestPtySession::test_terminate_ends_a_started_process_that_ignores_sigterm",
    "tests/interfaces/web_terminal/test_pty_manager.py::TestPtySession::test_terminate_ends_and_logs_the_process_groups_the_child_started",
    "tests/interfaces/web_terminal/test_pty_manager.py::TestPtySession::test_terminate_ends_what_the_child_started_when_the_first_process_table_read_fails",
    "tests/interfaces/web_terminal/test_pty_manager.py::TestPtySession::test_terminate_of_a_dead_child_looks_for_nothing_it_started",
    "tests/interfaces/web_terminal/test_pty_manager.py::TestPtySession::test_terminate_stops_the_childs_own_group_including_a_member_that_ignores_the_hang_up",
    "tests/interfaces/web_terminal/test_pty_manager.py::TestPtySession::test_terminate_warns_and_ends_its_own_group_when_the_table_stays_unreadable",
    "tests/interfaces/web_terminal/test_pty_manager.py::TestPtySession::test_write_and_read",
    "tests/mcp_server/python_executor/test_local_run_cancel.py",
    "tests/mcp_server/python_executor/test_wrapper_interrupt.py",
    "tests/mcp_server/test_connector_host_handshake_exit_code.py",
    "tests/mcp_server/test_switch_lifecycle.py::TestConcurrentSwitches::test_two_switches_serialize_and_the_second_sees_the_first",
    "tests/mcp_server/test_switch_lifecycle.py::TestDraining::test_the_drain_deadline_kills_the_child_and_names_the_switch",
    "tests/mcp_server/test_switch_lifecycle.py::TestFailedSwitchLeavesThePreviousTargetActive::test_a_failed_probe_kills_the_candidate_and_keeps_the_previous_child",
    "tests/mcp_server/test_switch_lifecycle.py::TestFailedSwitchLeavesThePreviousTargetActive::test_an_underivable_target_is_refused_before_any_spawn",
    "tests/mcp_server/test_switch_lifecycle.py::TestFailedSwitchLeavesThePreviousTargetActive::test_an_unknown_target_name_is_refused",
    "tests/mcp_server/test_switch_lifecycle.py::TestFailedSwitchLeavesThePreviousTargetActive::test_the_configured_probe_timeout_bounds_the_probe",
    "tests/mcp_server/test_switch_lifecycle.py::TestNoChildRefusal::test_the_next_call_is_served_by_the_restarted_child",
    "tests/mcp_server/test_switch_lifecycle.py::TestNoChildState::test_a_child_that_dies_outside_a_switch_leaves_the_session_without_one",
    "tests/mcp_server/test_switch_lifecycle.py::TestRespawn::test_a_same_target_respawn_replaces_the_process_without_a_generation_bump",
    "tests/mcp_server/test_switch_lifecycle.py::TestRespawn::test_invalidating_without_a_connector_host_is_unchanged",
    "tests/mcp_server/test_switch_lifecycle.py::TestStartupSweep::test_a_dead_servers_report_is_swept_and_its_recorded_children_are_not_signalled",
    "tests/mcp_server/test_switch_lifecycle.py::TestSuccessfulSwitch::test_a_reconcile_asked_to_launch_brings_up_the_first_child_on_the_record",
    "tests/mcp_server/test_switch_lifecycle.py::TestSuccessfulSwitch::test_a_reconcile_with_no_child_adopts_and_spawns_nothing",
    "tests/mcp_server/test_switch_lifecycle.py::TestSuccessfulSwitch::test_the_new_child_serves_reads_on_its_own_target",
    "tests/mcp_server/test_switch_lifecycle.py::TestSuccessfulSwitch::test_the_previous_child_process_actually_exits",
    "tests/pyaml_cs_osprey/test_run_tool.py::TestRunLock::test_a_second_run_on_the_target_is_busy_and_never_calls_the_tool",
    "tests/pyaml_cs_osprey/test_run_tool.py::TestSigintAbort::test_sigint_abort_stops_script",
    "tests/pyaml_cs_osprey/test_run_tool.py::TestSigintAbort::test_sigint_during_span_aborts",
    "tests/pyaml_cs_osprey/test_run_tool.py::TestSigintAbort::test_sigint_in_nested_run_aborts_outer",
    "tests/pyaml_cs_osprey/test_run_tool.py::TestSigintAbort::test_sigint_stops_a_helper_threads_tool_through_its_callback",
    "tests/pyaml_cs_osprey/test_run_tool.py::TestSigintAbort::test_sigint_without_callback_raises_with_what_it_moved",
    "tests/runtime/test_guarded_run_cases.py",
    "tests/runtime/test_kernel_connector_pool.py::TestPoolCleanup::test_a_forked_child_exits_without_touching_the_parents_pool",
    "tests/runtime/test_kernel_connector_pool.py::TestRealChildren::test_an_in_process_deployment_reads_through_a_real_child",
    "tests/runtime/test_kernel_connector_pool.py::TestRealChildren::test_reads_share_a_child_and_a_generation_move_respawns_it",
    "tests/runtime/test_kernel_connector_pool.py::TestRealChildrenFollowTheSwitch::test_writes_follow_the_switch_through_fresh_children",
    "tests/runtime/test_run_lock.py",
    "tests/scripts/test_live_ca_gate.py",
    "tests/scripts/test_run_bounded.py",
    "tests/services/ariel_search/test_search_offload.py::test_a_forked_child_gets_its_own_pool",
    "tests/services/ariel_search/test_search_offload.py::test_the_pool_does_not_outlive_its_process",
    "tests/services/python_executor/execution/test_net_guard.py",
    "tests/utils/test_profile_env_writes.py::TestConcurrentAppends::test_two_processes_both_land_their_keys",
    # Real children spawned and read back: exec, inherited descriptors, environment, exit status.
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_record_lock.py",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_session_hold.py",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_set_switch.py::test_all_off_racing_a_non_person_write_leaves_the_switch_empty",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_set_switch.py::test_the_gesture_waits_for_the_lock_and_refuses_after_a_bounded_wait",
    "tests/agent_runner/test_agent_shell_env.py",
    "tests/agent_runner/test_claude_state.py::test_the_pinned_cli_prints_the_notice_in_the_recognised_form",
    "tests/cli/test_build_lifecycle_report.py",
    "tests/cli/test_init_verb.py::test_enclosing_git_repository_is_left_alone",
    "tests/cli/test_init_verb.py::test_env_is_seeded_from_the_shell",
    "tests/cli/test_init_verb.py::test_git_status_is_clean_from_birth",
    "tests/cli/test_init_verb.py::test_in_place_into_an_empty_clone",
    "tests/cli/test_init_verb.py::test_refuses_to_nest_inside_another_deployment",
    "tests/cli/test_scaffold_cmd.py::TestClaimRefusals::test_a_skill_containing_an_escaping_symlink_is_refused",
    "tests/cli/test_scaffold_cmd.py::TestClaimRefusals::test_a_symlinked_artifact_is_refused",
    "tests/cli/test_verify_script_run.py",
    "tests/connectors/test_mock_journal.py",
    "tests/deployment/test_reset_keeps_record_volume.py::test_a_live_lease_refuses_every_reset_and_names_the_session",
    "tests/hooks/test_approval_hook.py::test_a_measurement_on_a_held_target_is_asked_with_both_approved_fields",
    "tests/hooks/test_approval_hook.py::test_the_prompt_binds_the_target_whatever_holds_its_lock",
    "tests/infrastructure/test_run_temp_root.py",
    "tests/mcp_server/python_executor/test_record_protected_roots.py",
    "tests/mcp_server/test_sandbox_child_env_allowlist.py::test_executor_child_does_not_reload_the_env_chain",
    "tests/mcp_server/test_sandbox_guard_characterization.py",
    "tests/scripts/test_cli_tool_inventory.py::test_probe_of_a_silent_build_reports_a_timeout",
    "tests/scripts/test_cli_tool_inventory.py::test_the_probe_hands_its_environment_to_the_build",
    "tests/scripts/test_pinned_claude.py",
    "tests/services/python_executor/test_fs_guard.py",
    "tests/services/python_executor/test_runtime_guard.py",
    "tests/services/python_executor/test_secret_zone_guard.py::test_a_secret_spelled_in_another_case_is_refused",
    "tests/services/python_executor/test_secret_zone_guard.py::test_executor_refuses_secret_opens",
    "tests/services/python_executor/test_secret_zone_guard.py::test_executor_still_reads_the_config",
    "tests/services/python_executor/test_secret_zone_guard.py::test_visualization_sandbox_refuses_env",
    "tests/services/python_executor/test_secret_zone_guard.py::test_visualization_sandbox_still_reads_project_data",
    "tests/services/python_executor/test_wrapper_holds_lock.py",
    "tests/services/watchdog/test_record_only.py",
    "tests/test_version.py",
    "tests/va/test_va_build_check_builder.py",
    # Real sockets: binds and reuse, the ephemeral port range, AF_UNIX path length and modes.
    "tests/cli/test_web_bind.py::TestBusyPortFallback::test_declared_port_still_hard_fails",
    "tests/cli/test_web_bind.py::TestBusyPortFallback::test_explicit_port_flag_still_hard_fails",
    "tests/cli/test_web_bind.py::TestBusyPortFallback::test_unspecified_port_auto_moves_off_busy_default",
    "tests/cli/test_web_bind.py::TestCompanionProbeAttribution::test_companion_port_held_by_the_roster_is_attributed",
    "tests/cli/test_web_bind.py::TestCompanionProbeAttribution::test_companion_port_held_without_a_roster_stays_foreign",
    "tests/cli/test_web_cmd.py::TestPortZero::test_a_busy_default_moves_without_releasing_the_socket",
    "tests/cli/test_web_cmd.py::TestPortZero::test_port_zero_serves_on_the_announced_port",
    "tests/cli/test_web_cmd.py::TestPreflightCompanionPortCollision::test_held_artifact_port_reported",
    "tests/deployment/test_host_ports.py::TestExternalConflict::test_listener_detected_then_cleared",
    "tests/deployment/test_host_ports.py::TestHistoryOriginSlot::test_a_listener_already_on_the_slot_stops_the_deploy",
    "tests/infrastructure/test_server_launcher.py::TestPortIsBindableAgainstARealKernel::test_a_reserved_ipv4_port_is_bindable",
    "tests/infrastructure/test_server_launcher.py::TestPortIsBindableAgainstARealKernel::test_a_reserved_ipv6_port_is_bindable",
    "tests/infrastructure/test_server_launcher.py::TestPortIsBindableAgainstARealKernel::test_an_ipv4_listener_holds_its_own_address",
    "tests/infrastructure/test_server_launcher.py::TestPortIsBindableAgainstARealKernel::test_an_ipv6_listener_holds_its_own_address",
    "tests/infrastructure/test_server_launcher.py::TestPortIsBindableAsksTheKernelToBind::test_the_address_family_follows_the_host_spelling",
    "tests/interfaces/test_serving.py::test_a_held_port_is_refused",
    "tests/interfaces/test_serving.py::test_port_zero_listens_on_the_port_the_socket_reports",
    "tests/interfaces/web_terminal/test_code_review_switch.py::test_closing_removes_the_socket_and_unpublishes_it",
    "tests/interfaces/web_terminal/test_code_review_switch.py::test_only_the_owner_can_reach_the_socket",
    "tests/interfaces/web_terminal/test_code_review_switch.py::test_the_socket_answers_on_without_a_registry",
    "tests/interfaces/web_terminal/test_code_review_switch.py::test_the_socket_serves_the_row_each_connection_reads",
    "tests/mcp_server/test_endpoint_prober.py",
    "tests/services/ariel_search/test_attachment_fetch.py::test_fetch_file_source_closes_every_descriptor",
    "tests/services/ariel_search/test_attachment_fetch.py::test_fetch_file_source_directory_leaf_is_not_a_regular_file",
    "tests/services/ariel_search/test_attachment_fetch.py::test_fetch_file_source_errno_maps_to_outcome",
    "tests/services/ariel_search/test_attachment_fetch.py::test_fetch_file_source_fifo_leaf_does_not_block",
    "tests/services/ariel_search/test_attachment_fetch.py::test_fetch_file_source_file_as_intermediate_is_source_gone",
    "tests/services/ariel_search/test_attachment_fetch.py::test_fetch_file_source_mode_000_directory_is_source_refused",
    "tests/services/ariel_search/test_attachment_fetch.py::test_fetch_file_source_mode_000_file_is_source_refused",
    "tests/services/ariel_search/test_attachment_fetch.py::test_fetch_file_source_refuses_dot_dot_absolute_and_empty_parts",
    "tests/services/ariel_search/test_attachment_fetch.py::test_fetch_file_source_symlinked_intermediate_directory_is_not_a_regular_file",
    "tests/services/ariel_search/test_attachment_fetch.py::test_fetch_file_source_symlinked_leaf_is_not_a_regular_file",
    # The live file-watch backend: FSEvents on darwin, inotify on Linux.
    "tests/interfaces/artifacts/test_store_watcher.py::TestStoreWatcher::test_an_external_save_reaches_the_broadcast_under_the_live_backend",
    "tests/interfaces/web_terminal/test_file_watcher.py",
    # Timezones from the system tzdata, or set through TZ and tzset.
    "tests/cli/test_build_profile_timezone.py",
    "tests/connectors/test_archiver_timezone.py",
    "tests/simulation/test_series_primitives.py::TestDailyOccurrences::test_positions_are_box_tz_independent",
    # Filesystem semantics macOS spells differently: case-insensitive APFS,
    # /var -> /private/var, symlinks, hard links, modes and renames.
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_agent_dir_container_anchor.py",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_control_context.py::test_a_writer_created_state_dir_is_exactly_mode_2770",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_control_context.py::test_a_written_record_is_group_readable_and_not_owner_only",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_control_context.py::test_an_existing_state_dir_keeps_the_mode_it_was_provisioned_with",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_control_context.py::test_every_level_the_writer_creates_is_exactly_mode_2770",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_control_context.py::test_every_state_dir_file_is_written_group_readable_mode",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_control_context.py::test_the_record_lands_group_readable_under_an_owner_only_umask",
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_safety_file.py",
    "tests/agent_record/test_spool_consumer.py::test_a_fifo_and_a_symlink_are_quarantined",
    "tests/agent_record/test_transcript_tailer.py::test_a_symlink_never_escapes_the_root",
    "tests/agent_record/test_transcript_tailer.py::test_a_symlinked_root_is_refused",
    "tests/agent_record/test_transcript_tailer.py::test_only_jsonl_regular_files_are_read",
    "tests/audit/test_posture_store.py::TestDegradation::test_an_unreadable_record_refuses",
    "tests/audit/test_posture_store.py::TestSignatureCache::test_a_same_size_rewrite_inside_one_clock_tick_is_still_seen",
    "tests/audit/test_posture_store.py::TestSignatureCache::test_a_turn_off_lands_on_a_process_already_running",
    "tests/audit/test_posture_store.py::TestSignatureCache::test_turning_it_on_is_seen_too",
    "tests/cli/test_code_review_build_check.py::test_a_prompt_linked_out_of_the_source_zone_is_refused",
    "tests/connectors/test_posture_store_contract.py",
    "tests/deployment/test_agent_record_render.py::test_the_build_provisions_the_spool_and_record_dir_group_shared",
    "tests/deployment/test_archive_dir.py",
    "tests/deployment/test_compose_generator.py::TestBuildPathProvisionsTheControlTree::test_a_fresh_build_creates_the_control_target_root_at_exactly_2770",
    "tests/deployment/test_compose_generator.py::TestBuildPathProvisionsTheControlTree::test_a_symlink_planted_at_a_control_target_identity_name_is_not_chmodded_through",
    "tests/deployment/test_compose_generator.py::TestBuildPathProvisionsTheControlTree::test_a_symlink_planted_at_the_control_target_root_is_not_provisioned_through",
    "tests/deployment/test_compose_generator.py::TestBuildPathProvisionsTheControlTree::test_a_symlink_planted_at_the_control_tree_marker_is_not_written_through",
    "tests/deployment/test_compose_generator.py::TestControlTargetPathSeam::test_a_control_target_identity_swapped_to_a_symlink_after_the_guard_is_refused",
    "tests/deployment/test_compose_generator.py::TestControlTargetPathSeam::test_a_control_target_identity_under_a_symlinked_tree_root_is_refused",
    "tests/deployment/test_compose_generator.py::TestControlTargetPathSeam::test_a_control_target_tree_swapped_to_a_symlink_after_the_guard_is_refused",
    "tests/deployment/test_compose_generator.py::TestControlTargetPathSeam::test_the_control_target_root_and_identity_dir_are_provisioned_at_2770",
    "tests/deployment/test_compose_generator.py::TestEnsureGroupSharedDirExact::test_a_directory_this_call_creates_lands_at_2770_either_way",
    "tests/deployment/test_compose_generator.py::TestEnsureGroupSharedDirExact::test_an_existing_0755_dir_is_narrowed_to_exactly_2770",
    "tests/deployment/test_guarded_run_mounts.py",
    "tests/deployment/test_reset_scoping.py::test_the_refusal_points_at_a_recorded_path_that_really_is_there",
    "tests/deployment/test_reset_scoping.py::test_two_spellings_of_one_repo_are_one_checkout",
    "tests/deployment/web_terminals/test_auth_credentials.py::test_a_link_planted_at_the_temporary_name_is_never_followed",
    "tests/deployment/web_terminals/test_auth_credentials.py::test_a_purge_holds_the_lock_a_concurrent_append_must_wait_for",
    "tests/deployment/web_terminals/test_auth_credentials.py::test_a_re_mint_is_one_critical_section_a_concurrent_append_waits_for",
    "tests/deployment/web_terminals/test_auth_credentials.py::test_an_unwritable_root_reports_missing_without_raising",
    "tests/deployment/web_terminals/test_auth_credentials.py::test_purge_preserves_mode_and_leaves_no_temporary_file",
    "tests/deployment/web_terminals/test_auth_credentials.py::test_set_auth_password_raises_on_a_write_failure",
    "tests/deployment/web_terminals/test_auth_credentials.py::test_unwritable_env_auth_reports_missing_secrets_without_raising",
    "tests/deployment/web_terminals/test_auth_credentials.py::test_unwritable_env_auth_reports_missing_without_raising",
    "tests/deployment/web_terminals/test_bundle_mount.py::test_a_wider_existing_mode_is_not_narrowed_either",
    "tests/deployment/web_terminals/test_bundle_mount.py::test_existing_bundle_dir_keeps_its_own_other_triad",
    "tests/deployment/web_terminals/test_bundle_mount.py::test_new_bundle_dir_gets_exactly_the_bits_sharing_needs",
    "tests/deployment/web_terminals/test_bundle_mount.py::test_new_files_inherit_the_directorys_group",
    "tests/deployment/web_terminals/test_bundle_mount.py::test_no_chmod_when_the_mode_already_satisfies_sharing",
    "tests/deployment/web_terminals/test_bundle_mount.py::test_provisioning_failure_is_not_fatal",
    "tests/deployment/web_terminals/test_render_images.py::test_delta_symlinked_out_of_the_profile_is_rejected",
    "tests/deployment/web_terminals/test_render_images.py::test_symlinked_tiers_directory_is_rejected",
    "tests/hooks/test_write_approval_stamp.py::test_a_link_planted_at_the_stamps_temporary_name_is_never_followed",
    "tests/hooks/test_write_verdict_twin.py::test_a_link_at_the_record_or_the_directory_is_held_on_both_sides",
    "tests/interfaces/test_session_store.py::test_directory_that_cannot_be_created_warns_at_construction",
    "tests/interfaces/test_session_store.py::test_saved_file_is_owner_only",
    "tests/interfaces/test_session_store.py::test_unreadable_file_loads_empty_with_one_warning",
    "tests/interfaces/test_session_store.py::test_unwritable_directory_keeps_retrying",
    "tests/interfaces/test_session_store.py::test_unwritable_directory_warns_once_and_never_raises",
    "tests/interfaces/web_terminal/test_chat_attachments.py::test_the_sweep_does_not_follow_a_symlink",
    "tests/interfaces/web_terminal/test_files_route_concealment.py::TestContentReturns404::test_case_variant_spellings_cannot_bypass_the_guard",
    "tests/interfaces/web_terminal/test_protected_writes_integration.py::TestTheGateIsDiscriminating::test_a_knowledge_page_linked_into_the_facility_tree_is_refused",
    "tests/interfaces/web_terminal/test_scaffold_gallery_service.py::TestCreateArtifact::test_create_artifact_hook_executable",
    "tests/interfaces/web_terminal/test_scaffold_gallery_service.py::TestLinkedWritesAreJudgedOnTheResolvedFile::test_a_link_onto_a_reserved_file_is_shown_read_only",
    "tests/interfaces/web_terminal/test_scaffold_gallery_service.py::TestLinkedWritesAreJudgedOnTheResolvedFile::test_create_artifact_refuses_a_link_that_lands_on_a_reserved_file",
    "tests/interfaces/web_terminal/test_scaffold_gallery_service.py::TestRestoreRefusesBodiesThatEscapeTheStore::test_a_body_reached_through_a_symlinked_directory_is_not_restored",
    "tests/interfaces/web_terminal/test_scaffold_gallery_service.py::TestRestoreRefusesBodiesThatEscapeTheStore::test_a_fifo_body_is_refused_under_its_own_reason",
    "tests/interfaces/web_terminal/test_scaffold_gallery_service.py::TestRestoreRefusesBodiesThatEscapeTheStore::test_a_hard_linked_body_is_refused",
    "tests/interfaces/web_terminal/test_scaffold_gallery_service.py::TestRestoreRefusesBodiesThatEscapeTheStore::test_a_symlinked_body_is_not_restored",
    "tests/interfaces/web_terminal/test_scaffold_gallery_service.py::TestRestoreRefusesReservedRecords::test_restore_reserved_skips_a_record_resolving_onto_a_reserved_file",
    "tests/interfaces/web_terminal/test_scaffold_gallery_service.py::TestScanUntrackedJudgesTheResolvedFile::test_a_link_onto_an_ordinary_file_is_still_listed",
    "tests/interfaces/web_terminal/test_scaffold_gallery_service.py::TestScanUntrackedJudgesTheResolvedFile::test_an_entry_linked_onto_a_reserved_file_is_not_listed",
    "tests/interfaces/web_terminal/test_scaffold_gallery_service.py::TestUnclaimDeletesTheStoredPath::test_a_stored_path_in_the_protected_set_is_refused",
    "tests/interfaces/web_terminal/test_scaffold_gallery_service.py::TestUnoverrideFrameworkRestore::test_a_restore_that_fails_keeps_the_release_and_says_so",
    "tests/interfaces/web_terminal/test_scaffold_gallery_service.py::TestVolumeMode::test_concurrent_claims_do_not_lose_records",
    "tests/interfaces/web_terminal/test_scaffold_gallery_service.py::TestWritesThatLeaveTheProjectAreRefused::test_delete_untracked_refuses_a_link_that_lands_outside_the_project",
    "tests/mcp_server/test_phoebus_tools.py::test_panel_lookup_accepts_a_file_url_from_the_caller",
    "tests/mcp_server/test_phoebus_tools.py::test_panel_lookup_follows_a_symlinked_display_mount",
    "tests/mcp_server/test_phoebus_tools.py::test_panel_lookup_matches_a_config_relative_registration",
    "tests/mcp_server/test_phoebus_tools.py::test_panel_lookup_matches_a_file_url_registration",
    "tests/mcp_server/test_phoebus_tools.py::test_panel_lookup_names_the_panel_that_opens_the_path",
    "tests/mcp_server/test_phoebus_tools.py::test_panel_lookup_normalises_dot_segments",
    "tests/mcp_server/workspace/test_transcript_reader.py",
    "tests/runtime/test_guarded_run_dir.py",
    "tests/runtime/test_guarded_run_regular_files.py",
    "tests/services/archive/test_run.py",
    "tests/services/ariel_search/test_qmd_writer.py::TestMirrorPath::test_path_round_trips_to_entry_id",
    "tests/services/ariel_search/test_qmd_writer.py::TestMirrorPath::test_stays_inside_mirror_root",
    "tests/services/ariel_search/test_qmd_writer.py::TestWriteEntry::test_hostile_ids_do_not_collide",
    "tests/services/ariel_search/test_qmd_writer.py::TestWriteEntry::test_hostile_ids_write_and_reread",
    "tests/services/ariel_search/test_qmd_writer.py::TestWriteEntry::test_no_temp_files_left_behind",
    "tests/services/bluesky_bridge/test_substrate_devices.py::TestWriteDevicesFile::test_leaves_no_temp_file_behind",
    "tests/services/bluesky_bridge/test_substrate_devices.py::TestWriteDevicesFile::test_written_file_is_readable_by_a_container_user",
    "tests/services/channel_finder/feedback/test_pending_store.py::test_concurrent_captures",
    "tests/services/code_review/test_settings.py::test_resolve_absolute_prompt_inside_the_source_zone",
    "tests/services/code_review/test_settings.py::test_resolve_default_prompt_anchors_at_the_repo_root",
    "tests/services/code_review/test_settings.py::test_resolve_prompt_in_another_source_entry",
    "tests/services/code_review/test_settings.py::test_resolve_refuses_absolute_prompt_outside_the_repo",
    "tests/services/code_review/test_settings.py::test_resolve_refuses_dotdot_out_of_the_source_zone",
    "tests/services/code_review/test_settings.py::test_resolve_refuses_prompt_under_an_absolute_agent_data_root",
    "tests/services/code_review/test_settings.py::test_resolve_refuses_symlink_out_of_the_source_zone",
    "tests/services/code_review/test_settings.py::test_resolve_refuses_unreadable_prompt",
    "tests/services/code_review/test_settings.py::test_unreadable_config_file_is_refused",
    "tests/services/test_bundle_path_resolution.py",
    "tests/services/watchdog/test_agent_engine.py::TestDenyCallbacks::test_read_refuses_a_link_out_of_the_references",
    "tests/stores/test_artifact_manifest.py::test_a_directory_or_a_fifo_under_an_artifact_name_is_not_collected",
    "tests/stores/test_artifact_manifest.py::test_a_hard_link_to_a_file_outside_is_neither_collected_nor_read",
    "tests/stores/test_artifact_manifest.py::test_a_link_in_the_folder_to_a_file_outside_is_neither_collected_nor_read",
    "tests/stores/test_artifact_manifest.py::test_a_manifest_that_is_a_link_collects_nothing",
    "tests/stores/test_artifact_manifest.py::test_an_artifacts_folder_that_is_a_link_collects_nothing",
    "tests/stores/test_artifact_sha256_dedup.py::test_two_threads_saving_the_same_bytes_give_one_artifact",
    "tests/stores/test_artifact_two_file_entries.py::TestOutsideStoreDirectoryIsNeverUnlinked::test_symlinked_companion_escaping_the_store_survives",
    "tests/test_e2e_chart_value_reading_build.py",
)

UNIT_SELECTION = UnitSelection(
    # An explicit path replaces testpaths, so the connectors package's own
    # suite runs only where it is named.
    roots=("tests/", "packages/osprey-connectors/tests"),
    ignores=(
        ("tests/e2e", "e2e-tests"),
        ("tests/va/test_apply_fault.py", "live-ca"),
        ("tests/va/test_lume_pva_seam.py", "live-ca"),
        ("tests/va/test_timed_passes_live.py", "live-ca"),
        ("tests/va/test_injection_live.py", "live-ca"),
        ("tests/integration/test_build_boot.py", "build-boot-smoke"),
        ("tests/integration/test_preset_static.py", "static-checks"),
        ("tests/integration/test_preset_build.py", "static-checks"),
    ),
    # A `canopy` test needs CANOPY installed and refuses to skip; the CANOPY
    # integration's `canopy-contract` check runs them.
    marker="not pty and not wallclock and not canopy",
    quick_marker="not pty and not wallclock and not canopy and not slow",
    shards={
        "cli-a": (("tests/cli",), (f"tests/cli/test_[!{_CLI_A_LETTERS}]*",)),
        "cli-b": (("tests/cli",), ("tests/cli/templates", f"tests/cli/test_[{_CLI_A_LETTERS}]*")),
        "rest-a": (_REST_A, ()),
        "rest-b": (
            ("tests/", "packages/osprey-connectors/tests"),
            ("tests/cli", *_REST_A, *_REST_C),
        ),
        "rest-c": (_REST_C, ()),
    },
    subsets={"macos": _MACOS_RUNS},
    # A dead worker ends the sweep with its report instead of being replaced.
    dist=("--dist", "loadgroup", "--max-worker-restart=0"),
)

#: The module that owns the JavaScript module-to-test mapping.
GUARD_MODULE = "tests/interfaces/test_js_module_tests.py"

#: Suffixes whose change reaches no test runner.
INERT_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".woff", ".woff2", ".ttf", ".pdf"}
)

JS_SUFFIXES = (".js", ".mjs", ".cjs")

#: Files whose change reaches every Vitest file.
VITEST_WIDE_NAMES = frozenset({"vitest.config.js", "package.json", "package-lock.json"})
VITEST_SETUP = "tests/vitest.setup.mjs"

#: The interface every other interface imports through the design-system alias.
SHARED_INTERFACE = "design_system"


# ---------------------------------------------------------------------------
# Selection


def _guard() -> ModuleType | None:
    """The module that owns the JavaScript module-to-test mapping, loaded by
    path, or ``None`` while the tree holds no such module."""
    if not (kit.ROOT / GUARD_MODULE).is_file():
        return None
    spec = importlib.util.spec_from_file_location("js_module_tests", kit.ROOT / GUARD_MODULE)
    if spec is None or spec.loader is None:
        raise Refused(f"cannot load {GUARD_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hosted_outside_sweep(path: str) -> bool:
    """Whether the parallel sweep leaves *path* to another check or to a lane."""
    for ignored, _host in UNIT_SELECTION.ignores:
        if path == ignored or path.startswith(ignored.rstrip("/") + "/"):
            return True
    return path.startswith("tests/pty/")


def _is_pytest_file(name: str) -> bool:
    return (name.startswith("test_") and name.endswith(".py")) or name.endswith("_test.py")


def _sweep_root(path: str) -> str | None:
    """The ``UNIT_SELECTION.roots`` tree *path* lies under, without its slash, if any."""
    for root in UNIT_SELECTION.roots:
        root = root.rstrip("/")
        if path.startswith(root + "/"):
            return root
    return None


def _holds_test_module(directory: str, files: Sequence[str]) -> bool:
    """Whether *directory* holds a test file the sweep would collect."""
    prefix = directory + "/"
    return any(
        name.startswith(prefix) and _is_pytest_file(name.rsplit("/", 1)[-1]) for name in files
    )


#: Why a change widens a runner to its whole selection, one rule per cause.
WIDENS_ROOT_CONFTEST = "a root conftest reaches every test below it"
WIDENS_UNREADABLE = "cannot read what it reaches"
WIDENS_HOOK = "defines {hook}, which acts on the whole session"
WIDENS_IMPORTED = "imported by {importer}"
WIDENS_UNMIRRORED = "a Python file outside the test mirrors"
WIDENS_PROJECT = "the project's dependencies reach every test"
WIDENS_NO_TEST_MODULE = "its test directory holds no test module"
WIDENS_ROOT_PACKAGE = "the root test package reaches every test below it"
WIDENS_SWEEP_HELPER = "a helper under a sweep root that framework tests import"
WIDENS_SWEEP_CONFTEST = "a conftest under a sweep root other than tests/, which no rule scopes"
WIDENS_UNPLACED = "a file under packages/ or tests/ that no rule places"
WIDENS_NO_MIRROR = "no tested package mirrors it"
WIDENS_BUNDLED_PROJECT = "a bundled project outside the test mirrors"
WIDENS_NO_MAPPING = "no JavaScript module-to-test mapping in this tree"
WIDENS_SHARED_INTERFACE = "the shared interface reaches every interface"
WIDENS_UNTESTED_INTERFACE = "its interface has no tests"
WIDENS_WIDE_INPUT = "a shared JavaScript input reaches every test"


def _whole(path: str, rule: str) -> Selection:
    return Selection(WHOLE, reason=f"{path}: {rule}")


def _statements(node: ast.AST) -> Iterator[ast.AST]:
    """Every statement below *node*; an import is never inside an expression."""
    for field in ("body", "orelse", "finalbody", "handlers", "cases"):
        for child in getattr(node, field, ()):
            yield child
            yield from _statements(child)


def _imported_modules(path: str, tree: ast.Module) -> Iterator[str]:
    """The files *tree*, read from *path*, may import, as repository paths."""
    parts = PurePosixPath(path).parts
    roots = [""] if parts[0] != "packages" or len(parts) < 2 else ["", f"packages/{parts[1]}/"]
    here = PurePosixPath(path).parent
    for node in _statements(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield from (root + alias.name.replace(".", "/") + ".py" for root in roots)
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").replace(".", "/")
            if node.level:
                base = here
                for _ in range(node.level - 1):
                    base = base.parent
                bases = [(base / module).as_posix() if module else base.as_posix()]
            else:
                bases = [root + module for root in roots]
            for base in bases:
                yield base + ".py"
                yield from (f"{base}/{alias.name}.py" for alias in node.names)


def _runs(path: str, reached: set[str]) -> bool:
    """Whether importing the files *reached* runs *path*.

    A package's ``__init__.py`` runs for any module imported below it.
    """
    if path.endswith("/__init__.py"):
        package = path.removesuffix("/__init__.py")
        return package + ".py" in reached or any(r.startswith(package + "/") for r in reached)
    return path in reached


def _importers(
    path: str,
    files: Sequence[str],
    beside: str | None = None,
    sources: dict[str, tuple[str, set[str] | None]] | None = None,
) -> Iterator[str]:
    """The Python files under ``tests/`` or ``packages/`` whose imports run *path*.

    Files below *beside*, a directory that already reaches *path*, are skipped.
    *sources* keeps each file's text and imports across calls of one scan.
    """
    sources = {} if sources is None else sources
    stem = PurePosixPath(path).stem
    word = PurePosixPath(path).parent.name if stem == "__init__" else stem
    for importer in files:
        if not importer.endswith(".py") or not importer.startswith(("tests/", "packages/")):
            continue
        if importer == path or (beside is not None and importer.startswith(beside + "/")):
            continue
        if importer not in sources:
            try:
                sources[importer] = ((kit.ROOT / importer).read_text(encoding="utf-8"), None)
            except (OSError, UnicodeDecodeError):
                sources[importer] = ("", set())
        text, reached = sources[importer]
        if word not in text:
            continue
        if reached is None:
            try:
                reached = set(_imported_modules(importer, ast.parse(text)))
            except (SyntaxError, ValueError):
                reached = set()
            sources[importer] = (text, reached)
        if _runs(path, reached):
            yield importer


def _importer_selection(
    path: str, files: Sequence[str], sources: dict[str, tuple[str, set[str] | None]]
) -> list[str]:
    """The tests that import a changed *path* test module or package, at any remove.

    A test module importer runs itself; a conftest or ``__init__.py`` importer
    its directory; a helper selects nothing itself, but its own importers count.
    *sources* is the scan's cache, shared by every changed path of one selection.
    """
    chosen: list[str] = []
    seen = {path}
    pending = [path]
    swept = [name for name in files if _sweep_root(name) is not None]
    while pending:
        target = pending.pop()
        directory, name = target.rsplit("/", 1)
        beside = directory if name in ("conftest.py", "__init__.py") else None
        for importer in _importers(target, swept, beside, sources):
            if importer in seen:
                continue
            seen.add(importer)
            if _hosted_outside_sweep(importer):
                continue
            pending.append(importer)
            leaf = importer.rsplit("/", 1)[-1]
            if _is_pytest_file(leaf):
                chosen.append(importer)
            elif leaf in ("conftest.py", "__init__.py"):
                chosen.append(importer.rsplit("/", 1)[0])
    return chosen


def _conftest_selection(path: str, files: Sequence[str]) -> str | Selection:
    """The directory a changed *path* conftest reaches, or why it reaches the sweep.

    pytest scopes a conftest to its own subtree, so one that only serves fixtures
    to it selects that directory. One that a module outside imports, or that
    defines a hook, acts beyond it.
    """
    directory = path.rsplit("/", 1)[0]
    root = _sweep_root(path)
    if root is None or directory == root:
        return _whole(path, WIDENS_ROOT_CONFTEST)
    if root != "tests":
        return _whole(path, WIDENS_SWEEP_CONFTEST)
    try:
        tree = ast.parse((kit.ROOT / path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return _whole(path, WIDENS_UNREADABLE)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
            "pytest_"
        ):
            return _whole(path, WIDENS_HOOK.format(hook=node.name))
    importer = next(_importers(path, files, directory), None)
    if importer is not None:
        return _whole(path, WIDENS_IMPORTED.format(importer=importer))
    if not _holds_test_module(directory, files):
        return _whole(path, WIDENS_NO_TEST_MODULE)
    return directory


def _mirror_root(path: str) -> tuple[str, str] | None:
    """The ``TEST_MIRRORS`` entry whose source root holds *path*, if any."""
    return next((entry for entry in TEST_MIRRORS if path.startswith(entry[0])), None)


def _bundled_project(path: str, files: Sequence[str]) -> str | None:
    """The directory below *path*'s mirror root holding its own ``pyproject.toml``, if any.

    Such a directory is a separate distribution, so its tests are its own, never a
    parent package's mirror.
    """
    entry = _mirror_root(path)
    if entry is None or entry[1] != "package":
        return None
    known = set(files)
    top = entry[0].rstrip("/")
    directory = PurePosixPath(path).parent
    while directory.as_posix().startswith(top + "/"):
        if f"{directory}/pyproject.toml" in known:
            return directory.as_posix()
        directory = directory.parent
    return None


def _mirror(path: str, files: Sequence[str]) -> str | None:
    """The tests that mirror *path* under its ``TEST_MIRRORS`` rule, if they exist.

    Under ``package`` the climb starts at the nearest directory that holds
    ``__init__.py`` and stops at the first ancestor whose ``tests/`` mirror
    holds a test module; directories on the way up need not be packages.
    """
    entry = _mirror_root(path)
    if entry is None:
        return None
    root, rule = entry
    if rule == "file":
        relative = PurePosixPath(path).relative_to(root)
        if len(relative.parts) != 1 or relative.suffix != ".py":
            return None
        mirror = f"tests/{root.rstrip('/')}/test_{relative.stem}.py"
        return mirror if mirror in set(files) else None
    if _bundled_project(path, files) is not None:
        return None
    known = set(files)
    top = PurePosixPath(root.rstrip("/"))
    package = PurePosixPath(path).parent
    while package != top and f"{package}/__init__.py" not in known:
        package = package.parent
    while package != top:
        mirror = "tests/" + package.relative_to(top).as_posix()
        if _holds_test_module(mirror, files):
            return mirror
        package = package.parent
    return None


def _paths(chosen: Iterable[str]) -> Selection:
    unique = tuple(dict.fromkeys(chosen))
    return Selection(PATHS, unique) if unique else Selection(NOT_TOUCHED)


def _select_pytest(
    paths: Sequence[str], files: Sequence[str], shaping: frozenset[str] = frozenset()
) -> Selection:
    chosen: list[str] = []
    sources: dict[str, tuple[str, set[str] | None]] = {}
    for path in paths:
        name = path.rsplit("/", 1)[-1]
        suffix = PurePosixPath(path).suffix
        if _hosted_outside_sweep(path):
            continue
        if suffix == ".py":
            root = _sweep_root(path)
            if root is not None and _is_pytest_file(name):
                chosen.append(path)
                chosen.extend(_importer_selection(path, files, sources))
                continue
            if root == "tests" and name == "__init__.py":
                directory = PurePosixPath(path).parent.as_posix()
                if directory == "tests":
                    return _whole(path, WIDENS_ROOT_PACKAGE)
                if not _holds_test_module(directory, files):
                    return _whole(path, WIDENS_NO_TEST_MODULE)
                chosen.append(directory)
                chosen.extend(_importer_selection(path, files, sources))
                continue
            if path in shaping:
                chosen.append("tests/scripts")
                continue
            if name == "conftest.py":
                reached = _conftest_selection(path, files)
                if isinstance(reached, Selection):
                    return reached
                chosen.append(reached)
                continue
            if root is not None and root != "tests":
                # A non-test module under any other sweep root (the connectors
                # test package's helpers and inits) is imported by framework
                # tests too, so it widens.
                return _whole(path, WIDENS_SWEEP_HELPER)
            if _mirror_root(path) is None:
                return _whole(path, WIDENS_UNMIRRORED)
        elif name in ("pyproject.toml", "uv.lock"):
            return _whole(path, WIDENS_PROJECT)
        elif suffix in JS_SUFFIXES or suffix in INERT_SUFFIXES:
            continue
        elif path.startswith("tests/") and len(PurePosixPath(path).parts) > 2:
            directory = "tests/" + PurePosixPath(path).parts[1]
            if not _holds_test_module(directory, files):
                return _whole(path, WIDENS_NO_TEST_MODULE)
            chosen.append(directory)
            continue
        elif path.startswith(("packages/", "tests/")):
            return _whole(path, WIDENS_UNPLACED)
        elif not path.startswith("src/osprey/"):
            continue
        mirror = _mirror(path, files)
        if mirror is None:
            if _bundled_project(path, files) is not None:
                return _whole(path, WIDENS_BUNDLED_PROJECT)
            return _whole(path, WIDENS_NO_MIRROR)
        chosen.append(mirror)
    return _paths(chosen)


def _select_vitest(
    paths: Sequence[str], files: Sequence[str], _shaping: frozenset[str] = frozenset()
) -> Selection:
    guard = _guard()
    if guard is None:
        # Without the mapping no JavaScript change can be placed, so any one
        # widens Vitest to its whole selection.
        reaches = next(
            (
                path
                for path in paths
                if path.endswith(JS_SUFFIXES)
                or path.rsplit("/", 1)[-1] in VITEST_WIDE_NAMES
                or path == VITEST_SETUP
            ),
            None,
        )
        return Selection(NOT_TOUCHED) if reaches is None else _whole(reaches, WIDENS_NO_MAPPING)
    chosen: list[str] = []
    for path in paths:
        name = path.rsplit("/", 1)[-1]
        if path.endswith(guard.TEST_SUFFIXES):
            if path.startswith(
                ("tests/", guard.INTERFACES_ROOT, PRESET_APPS_ROOT, SCAFFOLD_APP_ROOT)
            ):
                chosen.append(path)
        elif guard.modules([path]):
            interface = guard.interface_of(path)
            if interface == SHARED_INTERFACE:
                return _whole(path, WIDENS_SHARED_INTERFACE)
            own = guard.covering_tests(path, files)
            if own:
                chosen += own
                continue
            fallback = f"{guard.TESTS_ROOT}{interface}/"
            if not any(f.startswith(fallback) and f.endswith(guard.TEST_SUFFIXES) for f in files):
                return _whole(path, WIDENS_UNTESTED_INTERFACE)
            chosen.append(fallback)
        elif (
            name in VITEST_WIDE_NAMES
            or path == VITEST_SETUP
            or path.endswith(JS_SUFFIXES)
            and path.startswith(("tests/", "src/"))
        ):
            return _whole(path, WIDENS_WIDE_INPUT)
    return _paths(chosen)


# ---------------------------------------------------------------------------
# Commands


def ignore_flag(path: str) -> str:
    """The pytest flag that leaves *path* out: ``--ignore-glob`` for a pattern."""
    return f"--ignore-glob={path}" if any(c in path for c in "*?[") else f"--ignore={path}"


def unit_argv(
    tier: str,
    targets: Sequence[str] = (),
    shard: str | None = None,
    unit: UnitSelection = UNIT_SELECTION,
) -> list[str]:
    """The parallel sweep's selection: paths, ignores and marker expression."""
    if shard in unit.subsets:
        if targets:
            raise ValueError(f"the {shard} subset names its own tests; it takes no paths")
        roots, shard_ignores = unit.subsets[shard], ()
    else:
        roots, shard_ignores = unit.shards[shard] if shard else (unit.roots, ())
    ignores = [*shard_ignores, *(path for path, _host in unit.ignores)]
    marker = unit.quick_marker if tier == "quick" else unit.marker
    return [
        *(targets or roots),
        *(ignore_flag(path) for path in ignores),
        "-m",
        marker,
    ]


def _pytest_command(
    check: Check,
    selection: Selection,
    context: Context,
    flags: Sequence[str],
    written: Sequence[str],
) -> list[list[str]]:
    bound = QUICK_PYTEST_BOUND_S if context.tier == "quick" else check.bound_s
    selected = unit_argv(context.tier, selection.paths, context.shard)
    argv = [*check.argv, *selected, *flags, *UNIT_SELECTION.dist, *written]
    return [kit.bounded(bound, argv)]


#: Finds the test files that carry the ``wallclock`` marker; the root conftest
#: holds its answer to what pytest marks.
WALLCLOCK_FILES_MODULE = "tests/_wallclock_files.py"


def wallclock_unit_files() -> list[str]:
    """The files the image-free wall-clock budgets live in, read off the tree."""
    # The scanner is this checkout's code; the tree it reads is the root's.
    spec = importlib.util.spec_from_file_location(
        "wallclock_files", Path(__file__).resolve().parents[1] / WALLCLOCK_FILES_MODULE
    )
    if spec is None or spec.loader is None:
        raise Refused(f"cannot load {WALLCLOCK_FILES_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.marked_files(kit.ROOT, UNIT_SELECTION.roots)


def _wallclock_unit_command(
    check: Check,
    _selection: Selection,
    _context: Context,
    flags: Sequence[str],
    written: Sequence[str],
) -> list[list[str]]:
    """Only the marked files, selected by the marker: collecting the whole tree
    to pick a handful of tests cost about a minute. No file found falls back
    to the tree, where the marker selects nothing and the run says so."""
    files = wallclock_unit_files() or ["tests/", "--ignore=tests/e2e", "--ignore=tests/integration"]
    argv = [*check.argv, *files, "-m", "wallclock", *flags, *written]
    return [kit.bounded(check.bound_s, argv)]


#: The connectors wheel's import walk, shared with release-connectors.yml so the
#: wheel a connectors tag publishes passes the check every pull request runs.
#: Run with ``-I`` by the bare venv's interpreter: neither the checkout nor
#: ``PYTHONPATH`` can lend it a module the wheel does not ship.
LEAN_IMPORT_WALK = "scripts/ci/connectors_lean_import_walk.py"

#: Run by a venv holding the framework, connectors and pyaml-cs-osprey wheels.
PYAML_WHEEL_PROBE = """\
import importlib, pkgutil
import pyaml_cs_osprey
def _raise(name):
    raise ImportError(f'failed to walk {name}')
modules = [m.name for m in pkgutil.walk_packages(pyaml_cs_osprey.__path__, 'pyaml_cs_osprey.', onerror=_raise)]
for name in modules:
    importlib.import_module(name)
print(f'PYAML WHEEL OK ({len(modules)} modules)')
"""

#: Run by a venv holding only the connectors and optimizer wheels. The base is
#: what the framework imports (models, environment files), so it must load from
#: the two sibling wheels alone and pull in none of the ``service`` stack; the app
#: itself must refuse with the install hint.
OPTIMIZER_BASE_PROBE = """\
import importlib, sys
for name in ('osprey_optimizer', 'osprey_optimizer.models', 'osprey_optimizer.models.api',
             'osprey_optimizer.models.config', 'osprey_optimizer.environments.spec',
             'osprey_optimizer.environments.connector_env', 'osprey_optimizer.environments.interface'):
    importlib.import_module(name)
bad = sorted({m.split('.')[0] for m in sys.modules} & {'osprey', 'xopt', 'torch', 'botorch', 'gpytorch', 'polars', 'redis', 'fastapi', 'sklearn'})
assert not bad, f'the optimizer base loaded: {bad}'
try:
    import osprey_optimizer.api
except ImportError as exc:
    assert 'osprey-optimizer[service]' in str(exc), exc
else:
    raise AssertionError('osprey_optimizer.api imported without the service extra')
print('OPTIMIZER BASE WHEEL OK')
"""

#: The package build, in order; each step needs the one before it. ``{scratch}``
#: is a fresh temporary directory, and a ``dist/`` glob is expanded when the
#: step runs.
PACKAGE_STEPS: tuple[tuple[str, ...], ...] = (
    ("uv", "build"),
    ("uvx", "twine", "check", "dist/*"),
    ("uv", "build", "--package", "osprey-connectors"),
    ("uv", "venv", "{scratch}/lean", "--python", "3.11"),
    (
        "uv",
        "pip",
        "install",
        "--python",
        "{scratch}/lean/bin/python",
        "dist/osprey_connectors-*.whl",
    ),
    ("{scratch}/lean/bin/python", "-I", LEAN_IMPORT_WALK),
    ("uv", "build", "--package", "osprey-optimizer"),
    ("uv", "venv", "{scratch}/optimizer", "--python", "3.11"),
    (
        *("uv", "pip", "install", "--python", "{scratch}/optimizer/bin/python"),
        *("dist/osprey_connectors-*.whl", "dist/osprey_optimizer-*.whl"),
    ),
    ("{scratch}/optimizer/bin/python", "-c", OPTIMIZER_BASE_PROBE),
    ("uv", "build", "--package", "pyaml-cs-osprey"),
    ("uv", "venv", "{scratch}/pyaml", "--python", "3.11"),
    (
        *("uv", "pip", "install", "--python", "{scratch}/pyaml/bin/python"),
        *("dist/osprey_framework-*.whl", "dist/osprey_connectors-*.whl"),
        "dist/pyaml_cs_osprey-*.whl",
    ),
    ("{scratch}/pyaml/bin/python", "-c", PYAML_WHEEL_PROBE),
)


def _package_command(*_run: object) -> list[list[str]]:
    return [list(step) for step in PACKAGE_STEPS]


# ---------------------------------------------------------------------------
# Checks implemented here

#: Prints the keys of the documentation's ``redirects`` table, space separated.
REDIRECT_KEYS_PROBE = (
    *UV_RUN,
    "python",
    "-c",
    "import os, runpy; os.chdir('docs/source'); "
    "print(' '.join(runpy.run_path('conf.py')['redirects']))",
)

DOCS_HTML = "docs/build/html"


def missing_redirect_stubs(keys: Iterable[str], html_root: Path) -> list[str]:
    """The redirect keys whose stub page the documentation build did not write."""
    return [key for key in keys if not (html_root / f"{key}.html").is_file()]


def _check_redirect_stubs(_context: Context) -> int:
    status, out = kit.capture(REDIRECT_KEYS_PROBE)
    if status != 0:
        kit.say("docs: cannot read the redirects table of docs/source/conf.py")
        return FAILED_EXIT
    missing = missing_redirect_stubs(out.split(), kit.ROOT / DOCS_HTML)
    if missing:
        kit.say(f"docs: redirect stubs missing from the build: {' '.join(missing)}")
        return FAILED_EXIT
    return CLEAN


def _expand(step: Sequence[str], scratch: str) -> list[str]:
    """Fill ``{scratch}`` and expand a ``dist/`` glob against the repository root."""
    argv: list[str] = []
    for part in step:
        part = part.replace("{scratch}", scratch)
        if part.startswith("dist/") and "*" in part:
            found = sorted(glob.glob(part, root_dir=kit.ROOT))
            argv += found or [part]
        else:
            argv.append(part)
    return argv


def _check_package(_context: Context) -> int:
    # dist/ outlives a branch switch and `uv build` only adds to it, so anything
    # left there would be checked as this tree's.
    shutil.rmtree(kit.ROOT / "dist", ignore_errors=True)
    with tempfile.TemporaryDirectory(prefix="package-check-") as scratch:
        for step in PACKAGE_STEPS:
            status = kit.run(_expand(step, scratch))
            if status != 0:
                return status
    return CLEAN


#: Lists every tracked path a ``.gitignore`` rule matches.
TRACKED_IGNORED = ("git", "ls-files", "-ci", "--exclude-standard")


def _check_tracked_ignored(_context: Context) -> int:
    status, out = kit.capture(TRACKED_IGNORED)
    if status != 0:
        kit.say(f"tracked-ignored: {shlex.join(TRACKED_IGNORED)} exited {status}")
        return FAILED_EXIT
    ignored = out.split()
    if ignored:
        kit.say("tracked-ignored: tracked files matched by .gitignore (narrow the rule or untrack")
        kit.say(f"tracked-ignored: the file): {' '.join(ignored)}")
        return FAILED_EXIT
    return CLEAN


#: Lists every tracked prose file, NUL-separated.
TRACKED_DOCS = ("git", "ls-files", "-z", "--", "*.md", "*.rst")

#: Hand-typed parallel sweeps a document may keep, keyed by (path, command
#: text) and valued with the reason; a key whose document holds no such line is
#: reported.
SWEEP_LINE_EXCEPTIONS: dict[tuple[str, str], str] = {
    (
        "plugins/osprey/skills/release/SKILL.md",
        "pytest tests/e2e/ -v --tb=short -n 4 --dist loadfile",
    ): (
        "the e2e lane's invocation is declared only in ci.yml's e2e-tests job"
        " until that lane is a check"
    ),
}

#: A ``pytest`` command word: bare, ``uv run pytest`` or ``python -m pytest``.
_PYTEST_WORD = re.compile(r"(?<![\w./-])pytest(?=[\s`'\"]|$)")

#: An xdist flag: ``-n <count>``, ``-n<count>``, ``--numprocesses`` or ``--dist``.
_XDIST_FLAG = re.compile(
    r"(?<![\w-])(?:-n\s*(?:\d+|auto|logical)(?![\w-])|--numprocesses\b|--dist\b)"
)


def _logical_lines(text: str) -> Iterable[tuple[int, str]]:
    """Yield each line with its 1-based number, a ``\\``-continued run joined into one."""
    start, held = 0, list[str]()
    for number, line in enumerate(text.splitlines(), 1):
        if not held:
            start = number
        if line.rstrip().endswith("\\"):
            held.append(line.rstrip()[:-1])
            continue
        held.append(line)
        yield start, " ".join(held)
        held = []
    if held:
        yield start, " ".join(held)


def hand_typed_sweeps(
    texts: Mapping[str, str],
    exceptions: Mapping[tuple[str, str], str] = SWEEP_LINE_EXCEPTIONS,
) -> list[str]:
    """Name every line that types a ``pytest`` command with an xdist flag.

    A line naming ``scripts/check.py`` runs a declared check and passes; a line
    holding an exception's command text passes, and an exception that matches
    no line, its document gone or unread included, is reported as stale.
    """
    found: list[str] = []
    used: set[tuple[str, str]] = set()
    for path in sorted(texts):
        held = [key for key in exceptions if key[0] == path]
        for number, line in _logical_lines(texts[path]):
            if "scripts/check.py" in line:
                continue
            if not (_PYTEST_WORD.search(line) and _XDIST_FLAG.search(line)):
                continue
            excepted = [key for key in held if key[1] in line]
            if excepted:
                used.update(excepted)
                continue
            found.append(
                f"{path}:{number}: a hand-typed parallel sweep; "
                "name scripts/check.py (quick, full, run pytest)"
            )
    stale = sorted(key for key in exceptions if key not in used)
    found += [f"stale exception: {path}: {text}" for path, text in stale]
    return found


def read_doc_texts(paths: Iterable[str], root: Path) -> tuple[dict[str, str], list[str]]:
    """Return each existing file's UTF-8 text, and the sorted paths that are not UTF-8."""
    texts: dict[str, str] = {}
    unreadable: list[str] = []
    for path in paths:
        file = root / path
        if not file.is_file():
            continue
        try:
            texts[path] = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            unreadable.append(path)
    return texts, sorted(unreadable)


def tracked_doc_texts() -> tuple[dict[str, str], list[str]] | None:
    """Read every tracked ``*.md`` and ``*.rst`` file; None when git fails."""
    status, out = kit.capture(TRACKED_DOCS)
    if status != 0:
        return None
    return read_doc_texts((path for path in out.split("\0") if path), kit.ROOT)


def _check_hand_typed_sweeps(_context: Context) -> int:
    read = tracked_doc_texts()
    if read is None:
        kit.say(f"hand-typed-sweeps: {shlex.join(TRACKED_DOCS)} failed")
        return FAILED_EXIT
    texts, unreadable = read
    findings = [f"{path}: not UTF-8 text" for path in unreadable]
    findings += hand_typed_sweeps(texts, SWEEP_LINE_EXCEPTIONS)
    for finding in findings:
        kit.say(f"hand-typed-sweeps: {finding}")
    return FAILED_EXIT if findings else CLEAN


#: Run in the documentation directory; needs the docs extra, as the build does.
DOCS_CLEAN = (*UV_RUN, "make", "clean")


def _prune_bytecode(_context: Context) -> int:
    # A package whose sources were deleted still imports, as a namespace
    # package, while its directory holds a __pycache__; a test that the
    # package is gone then fails here and passes on a fresh checkout.
    package = kit.ROOT / "src" / "osprey"
    for cache in sorted(package.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)
    emptied = sorted((path for path in package.rglob("*") if path.is_dir()), reverse=True)
    for directory in emptied:
        try:
            directory.rmdir()
        except OSError:
            continue
    return CLEAN


def _clean_docs_build(_context: Context) -> int:
    # The build tree outlives a branch switch and the build only adds to it, so
    # a page left there would be judged as this tree's.
    kit.say(f"docs: {shlex.join(DOCS_CLEAN)}")
    return kit.run(DOCS_CLEAN, kit.ROOT / DOCS_DIR)


# ---------------------------------------------------------------------------
# Needs and venues

#: Importable only in a venv synced with the docs extra.
DOCS_EXTRA_PROBE = (*UV_RUN, "python", "-c", "import myst_parser")

#: Importable only in a venv synced with the virtual-accelerator extra, on its
#: platform. The modules are the ones the live gate requires.
LIVE_CA_PROBE = (*UV_RUN, "python", "-c", "import pcaspy, p4p, lume_pva_apg")

#: Prints the directory whose ``claude`` reports the pin, or the product's
#: refusal on stderr.
CLAUDE_PIN_PROBE = (*UV_RUN, "python", "scripts/pinned_claude.py")

#: The compare of ``node_modules`` with ``package-lock.json``, stdlib only.
JS_DEPS_CURRENT = Path(__file__).resolve().with_name("js_deps_current.py")

#: Below this a run can die halfway through its temp tree; one figure, no knob.
FREE_SPACE_FLOOR_GIB = 5

#: How long each docker command of the storage probe may take.
DOCKER_PROBE_TIMEOUT_S = 60

#: The docker group as the tests spell it, read as text: collecting the
#: selection to ask would cost more than the probe it arms.
DOCKER_GROUP_PATTERN = r'mark\.xdist_group\(\s*"docker"'

#: ``--no-sync``: a plain ``uv run`` is what would hide a stale install, or install it.
INSTALLED_VERSIONS_PROBE = (*UV_RUN, "--no-sync", "python", "scripts/installed_versions.py")

#: Runs only in a venv synced with the dev extra. actionlint disables its
#: shellcheck rule without a word when shellcheck is absent and still exits 0, so
#: this probe is what keeps the check from passing over unread ``run:`` blocks.
WORKFLOW_LINT_PROBE = (*UV_RUN, "shellcheck", "--version")


def _need_uv() -> str | None:
    if shutil.which("uv"):
        return None
    return "uv is not on PATH: install uv, then run uv sync --extra dev"


def _need_gitleaks() -> str | None:
    if shutil.which("gitleaks"):
        return None
    return (
        "gitleaks is not on PATH: run brew install gitleaks, "
        "or install a release from https://github.com/gitleaks/gitleaks/releases"
    )


def _need_workflow_linters() -> str | None:
    if kit.capture(WORKFLOW_LINT_PROBE)[0] == 0:
        return None
    return "the workflow linters are not installed: run uv sync --extra dev"


def _probe(argv: Sequence[str], timeout: float | None = None) -> tuple[int, str, str]:
    """Run a probe and return its exit status, stdout and stderr."""
    try:
        done = subprocess.run(
            list(argv), cwd=kit.ROOT, capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError:
        return 127, "", ""
    return done.returncode, done.stdout, done.stderr


def _need_claude_pin() -> str | None:
    """Met when a ``claude`` reporting the pin is found; its directory then goes
    first on ``PATH``, so every check this command starts runs it."""
    status, out, err = _probe(CLAUDE_PIN_PROBE)
    if status != 0:
        return " ".join(err.split()) or f"{shlex.join(CLAUDE_PIN_PROBE)} exited {status}"
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines or not Path(lines[-1]).is_dir():
        return f"{shlex.join(CLAUDE_PIN_PROBE)} printed no directory"
    directory = lines[-1]
    os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
    kit.say(f"claude-pin: {directory} first on PATH")
    return None


def _opt_in_js_check(context: Context) -> int:
    """Open the test that fetches and runs the toolchain ``osprey app check`` ships."""
    os.environ["OSPREY_JS_CHECK"] = "1"
    return _prune_bytecode(context)


def _js_deps_current() -> ModuleType:
    """The lockfile compare, loaded by path: scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("js_deps_current", JS_DEPS_CURRENT)
    if spec is None or spec.loader is None:
        raise Refused(f"cannot load {JS_DEPS_CURRENT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _need_node() -> str | None:
    """Met when ``node_modules`` holds what ``package-lock.json`` pins; runs
    ``npm ci`` when it does not."""
    if not (shutil.which("node") and shutil.which("npm")):
        return "the Node toolchain is not installed: install Node, then run npm ci"
    first_difference = _js_deps_current().first_difference
    difference = first_difference(kit.ROOT)
    if difference is None:
        return None
    kit.say(f"node: {difference}; npm ci")
    status = kit.run(("npm", "ci"))
    if status != 0:
        return f"npm ci failed (exit {status}): {difference}"
    again = first_difference(kit.ROOT)
    if again is not None:
        return f"node_modules still differs from package-lock.json after npm ci: {again}"
    return None


def _device(path: str | Path) -> int:
    return os.stat(path).st_dev


def _need_free_space() -> str | None:
    """Met when the temp directory and the checkout each hold the floor; each
    device is read once."""
    try:
        places: dict[int, str | Path] = {}
        for place in (tempfile.gettempdir(), kit.ROOT):
            places.setdefault(_device(place), place)
        for path in places.values():
            free = shutil.disk_usage(path).free
            if free < FREE_SPACE_FLOOR_GIB * 2**30:
                raise EnvironmentFault(
                    "free space",
                    f"{free // 2**20} MiB free on {path}, floor {FREE_SPACE_FLOOR_GIB} GiB",
                )
    except OSError as error:
        raise EnvironmentFault("free space", str(error)) from error
    return None


def _need_docker_storage() -> str | None:
    """Met when a volume can be created and removed, and on a machine with no
    daemon to ask, whose Docker tests skip. It starts no container."""
    if not shutil.which("docker"):
        return None
    try:
        if _probe(("docker", "info"), DOCKER_PROBE_TIMEOUT_S)[0] != 0:
            return None
    except subprocess.TimeoutExpired:
        return None
    volume = f"osprey-check-probe-{os.getpid()}"
    for verb in ("create", "rm"):
        try:
            status, _out, err = _probe(("docker", "volume", verb, volume), DOCKER_PROBE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            fact = f"did not answer in {DOCKER_PROBE_TIMEOUT_S} s"
        else:
            if status == 0:
                continue
            fact = err.strip().rpartition(": ")[2] or f"exited {status}"
        raise EnvironmentFault("docker storage", f"volume {verb} {fact}")
    return None


def _docker_group_needs(
    roots: Callable[[Context], Iterable[str]],
) -> Callable[[Selection, Context], tuple[str, ...]]:
    """The hook of a check whose whole selection is *roots*: it needs Docker
    storage when a selected file, or a conftest above one, carries the group."""

    def needs(selection: Selection, context: Context) -> tuple[str, ...]:
        paths = selection.paths if selection.state == PATHS else roots(context)
        targets = [path.partition("::")[0].rstrip("/") for path in paths]
        above = {
            conftest
            for target in targets
            for parent in PurePosixPath(target).parents
            if (kit.ROOT / (conftest := str(parent / "conftest.py"))).is_file()
        }
        status, out, _err = _probe(
            ("git", "grep", "--untracked", "-lE", DOCKER_GROUP_PATTERN, "--", *targets, *above)
        )
        hit = status == 0 and any(line.endswith(".py") for line in out.splitlines())
        return ("docker-storage",) if hit else ()

    return needs


def _leading_paths(words: Sequence[str]) -> list[str]:
    """The paths a pytest command line names before its first flag."""
    paths: list[str] = []
    for word in words:
        if word.startswith("-"):
            break
        paths.append(word)
    return paths


def _pytest_paths(argv: Sequence[str]) -> list[str]:
    """The paths *argv* hands pytest."""
    return _leading_paths(argv[list(argv).index("pytest") + 1 :])


def _unit_roots(context: Context) -> list[str]:
    """What the parallel sweep collects for the run's tier and shard."""
    return _leading_paths(unit_argv(context.tier, (), context.shard))


def _stale_installs() -> list[list[str]]:
    """The reader's rows. It prints none on a checkout with no environment yet:
    the check's own ``uv run`` creates that one current."""
    rows = (line.split("\t") for line in _probe(INSTALLED_VERSIONS_PROBE)[1].splitlines())
    return [row for row in rows if len(row) == 3]


def _need_install_current() -> str | None:
    """Met when the install is what the tree describes; re-syncs it when it is not."""
    stale = _stale_installs()
    if not stale:
        return None
    for dist, installed, described in stale:
        kit.say(f"install-current: {dist} installed {installed}, tree describes {described}")
    reinstall = [word for row in stale for word in ("--reinstall-package", row[0])]
    # --inexact: a plain sync removes the extras this environment was given.
    status = kit.run(("uv", "sync", "--frozen", "--inexact", *reinstall))
    still = _stale_installs() or (stale if status != 0 else [])
    if not still:
        return None
    dist, installed, described = still[0]
    raise EnvironmentFault(
        "install currency",
        f"{dist} installed {installed}, tree describes {described} after uv sync",
    )


def _need_history() -> str | None:
    if kit.git("rev-parse", "--is-shallow-repository").strip() == "true":
        return "this clone is shallow: run git fetch --unshallow"
    return None


def _need_docs_extra() -> str | None:
    if kit.capture(DOCS_EXTRA_PROBE)[0] == 0:
        return None
    return "the docs extra is not installed: run uv sync --extra docs"


def _need_graphviz() -> str | None:
    if shutil.which("dot"):
        return None
    return "graphviz is not on PATH: the docs build draws its diagrams with dot; install graphviz"


def _need_va_extra() -> str | None:
    if kit.capture(LIVE_CA_PROBE)[0] == 0:
        return None
    return "the virtual-accelerator extra is not installed: run uv sync --extra virtual-accelerator"


#: Each need to its probe: ``None`` when met, else the refusal.
NEEDS = {
    "uv": _need_uv,
    "gitleaks": _need_gitleaks,
    "node": _need_node,
    "claude-pin": _need_claude_pin,
    "history": _need_history,
    "docs-extra": _need_docs_extra,
    "graphviz": _need_graphviz,
    "workflow-linters": _need_workflow_linters,
    "va-extra": _need_va_extra,
    "free-space": _need_free_space,
    "docker-storage": _need_docker_storage,
    "install-current": _need_install_current,
}


def _on_live_ca_venue() -> bool:
    """The platforms the virtual-accelerator extra's marker installs on."""
    return sys.platform == "linux" and platform.machine() in ("x86_64", "aarch64")


def _on_darwin_venue() -> bool:
    return sys.platform == "darwin"


VENUES = {
    LIVE_CA_VENUE: Venue(_on_live_ca_venue, "needs linux x86_64 or aarch64"),
    DARWIN_VENUE: Venue(_on_darwin_venue, "needs macOS"),
}


# ---------------------------------------------------------------------------
# The checks

#: The wall-clock suite; its leading paths are what its Docker hook reads.
WALLCLOCK_ARGV = (*UV_RUN, "pytest", "tests/integration", "-m", "wallclock")

CHECKS: tuple[Check, ...] = (
    Check("lockfile", "invariant", BOTH, ("uv", "lock", "--check"), needs=("uv",)),
    Check(
        "ruff",
        "invariant",
        BOTH,
        (*UV_RUN, "ruff", "check", "."),
        needs=("uv",),
        fix=((*UV_RUN, "ruff", "check", ".", "--fix"),),
        annotate=GITHUB_FORMAT,
    ),
    Check(
        "ruff-format",
        "invariant",
        BOTH,
        (*UV_RUN, "ruff", "format", "--check", "."),
        needs=("uv",),
        fix=((*UV_RUN, "ruff", "format", "."),),
    ),
    Check(
        "config-keys",
        "invariant",
        BOTH,
        (*UV_RUN, "python", "scripts/check_config_keys.py", "--back-test"),
        needs=("uv", "history"),
    ),
    Check(
        "changelog",
        "invariant",
        BOTH,
        ("python3", "scripts/changelog_fragments.py", "check", "--base", "{base}"),
        needs=("base",),
        worktree=(
            (
                "python3",
                "scripts/changelog_fragments.py",
                "check",
                "--base",
                "{base}",
                "--worktree",
            ),
        ),
    ),
    Check(
        "plugin-version",
        "invariant",
        BOTH,
        ("python3", "scripts/plugin_version.py", "check", "--base", "{base}"),
        needs=("base",),
        worktree=(
            ("python3", "scripts/plugin_version.py", "check", "--base", "{base}", "--worktree"),
        ),
    ),
    # Implemented in Python: git lists what is both tracked and ignored.
    Check("tracked-ignored", "invariant", BOTH, (), builtin=_check_tracked_ignored),
    Check(
        "gitleaks",
        "invariant",
        BOTH,
        ("gitleaks", "git", "--log-opts={base}..HEAD", "--no-banner", "--redact"),
        needs=("base", "gitleaks"),
        # ``git --staged`` scans the index, ``git --pre-commit`` the unstaged
        # diff, and ``dir`` one untracked path: ``dir`` given several paths or
        # none scans the whole directory.
        worktree=(
            ("gitleaks", "git", "--log-opts={base}..HEAD", "--no-banner", "--redact"),
            ("gitleaks", "git", "--staged", "--no-banner", "--redact"),
            ("gitleaks", "git", "--pre-commit", "--no-banner", "--redact"),
            ("gitleaks", "dir", "{untracked}", "--no-banner", "--redact"),
        ),
    ),
    Check(
        "invariant-tests",
        "invariant",
        BOTH,
        (*UV_RUN, "pytest", *INVARIANT_TESTS),
        needs=("free-space", "uv"),
        report="junit-pytest",
    ),
    Check(
        "mypy",
        "static",
        BOTH,
        (*UV_RUN, "python", "scripts/mypy_gate.py"),
        scope=("*.py", "pyproject.toml", "uv.lock"),
        needs=("uv",),
    ),
    Check(
        "import-contracts",
        "static",
        BOTH,
        (*UV_RUN, "lint-imports"),
        scope=("*.py", "pyproject.toml", "uv.lock"),
        needs=("uv",),
    ),
    Check(
        "private-imports",
        "static",
        BOTH,
        (*UV_RUN, "ruff", "check", "--preview", "--select=PLC2701", "."),
        scope=("*.py", "pyproject.toml"),
        needs=("uv",),
        annotate=GITHUB_FORMAT,
    ),
    Check(
        "tsc",
        "static",
        BOTH,
        ("npm", "run", "typecheck"),
        scope=TSC_SCOPE,
        needs=("free-space", "node"),
    ),
    Check(
        "eslint",
        "static",
        BOTH,
        ("npm", "run", "lint"),
        scope=("*.js", "*.mjs", "*.ts", "eslint.config.js", "tools/eslint/**", "package*.json"),
        needs=("free-space", "node"),
    ),
    Check(
        "prettier",
        "static",
        BOTH,
        ("npm", "run", "format:check"),
        scope=("*.js", "*.mjs", ".prettierrc.json", ".prettierignore", "package*.json"),
        needs=("free-space", "node"),
        fix=(("npm", "run", "format"),),
    ),
    Check(
        "actionlint",
        "static",
        BOTH,
        (*UV_RUN, "actionlint"),
        scope=(".github/workflows/**",),
        needs=("uv", "workflow-linters"),
    ),
    # Implemented in Python: a contributor instruction that runs the parallel
    # sweep names a declared check, so its flags have one home.
    Check(
        "hand-typed-sweeps",
        "static",
        BOTH,
        (),
        scope=("*.md", "*.rst"),
        builtin=_check_hand_typed_sweeps,
    ),
    # The parallel sweep: its selection is UNIT_SELECTION, a range picks its
    # tests through TEST_MIRRORS, and a run whose paths the marker expression
    # empties is named, not failed.
    Check(
        "pytest",
        "tests",
        BOTH,
        (*UV_RUN, "pytest"),
        needs=("free-space", "uv", "install-current", "claude-pin"),
        selected_needs=_docker_group_needs(_unit_roots),
        bound_s=5400,
        report="junit-pytest",
        tier_flags={
            "quick": ("-n", "auto", "--maxfail=1", "--tb=line", "-q"),
            "full": ("-n", "auto", "--tb=short", "--timeout=600", "--timeout-method=signal"),
        },
        empty_exit=5,
        named_suffixes=(".py",),
        named_node_ids=True,
        named_dirs=_is_pytest_file,
        selector=_select_pytest,
        command=_pytest_command,
        prepare=_prune_bytecode,
        shards=(*UNIT_SELECTION.shards, *UNIT_SELECTION.subsets),
        exec_flags=True,
    ),
    Check(
        "pytest-pty",
        "tests",
        FULL,
        (*UV_RUN, "pytest", "tests/pty/", "-m", "pty"),
        scope=("tests/pty/**", "src/osprey/cli/**"),
        needs=("free-space", "uv", "install-current"),
        bound_s=600,
        report="junit-pytest",
        tier_flags={"full": ("--tb=short",)},
        prepare=_prune_bytecode,
        exec_flags=True,
    ),
    # A wall-clock budget timed beside loaded workers measures the workers, so
    # the sweep deselects the marker and this suite runs alone.
    Check(
        "pytest-wallclock",
        "tests",
        FULL,
        WALLCLOCK_ARGV,
        scope=("tests/integration/**", "*.py", "pyproject.toml", "uv.lock"),
        needs=("free-space", "uv", "install-current"),
        selected_needs=_docker_group_needs(lambda _context: _pytest_paths(WALLCLOCK_ARGV)),
        bound_s=900,
        report="junit-pytest",
        tier_flags={"full": ("--tb=short",)},
        exec_flags=True,
    ),
    # The image-free budgets: every wallclock test under the sweep's roots
    # outside tests/integration, whose budgets need the sidecar image and are
    # timed by pytest-wallclock.
    # The files are found by a scan of the tree when the check runs, so a budget
    # added anywhere is timed alone with no new wiring, and the run collects
    # those files rather than the whole tree.
    Check(
        "pytest-wallclock-unit",
        "tests",
        FULL,
        (*UV_RUN, "pytest"),
        scope=(
            *(f"{root.rstrip('/')}/**" for root in UNIT_SELECTION.roots),
            "*.py",
            "pyproject.toml",
            "uv.lock",
        ),
        needs=("free-space", "uv", "install-current"),
        bound_s=900,
        report="junit-pytest",
        tier_flags={"full": ("--tb=short",)},
        exec_flags=True,
        command=_wallclock_unit_command,
    ),
    # The readonly gate refuses every shared-library load once user code starts,
    # so a stdlib lookup that loads native code lazily must be warmed before it
    # closes. The sweep runs the warm-ups on the tree's interpreter; this runs
    # them under 3.14 on macOS in a throwaway environment.
    Check(
        "readonly-warmups-py314",
        "tests",
        FULL,
        (
            *UV_RUN,
            "--isolated",
            "--python",
            "3.14",
            "--extra",
            "dev",
            "pytest",
            READONLY_WARMUPS_TEST_FILE,
            "-v",
        ),
        scope=(READONLY_WARMUPS_TEST_FILE, "src/osprey/runtime/**"),
        needs=("free-space", "uv"),
        venue=DARWIN_VENUE,
        bound_s=900,
        report="junit-pytest",
    ),
    Check(
        "live-ca",
        "tests",
        FULL,
        (*UV_RUN, "python", "scripts/va/live_ca/gate.py", "--pva"),
        scope=(
            "tests/va/test_apply_fault.py",
            "tests/va/test_lume_pva_seam.py",
            "tests/va/test_timed_passes_live.py",
            "tests/va/test_injection_live.py",
            "scripts/va/live_ca/**",
            "src/osprey/services/virtual_accelerator/**",
        ),
        needs=("free-space", "uv", "va-extra", "install-current"),
        venue=LIVE_CA_VENUE,
    ),
    # A range picks its files through the JavaScript module-to-test mapping.
    Check(
        "vitest",
        "tests",
        BOTH,
        ("npx", "vitest", "run"),
        needs=("free-space", "node"),
        report="junit-vitest",
        named_suffixes=(".test.js", ".test.mjs"),
        selector=_select_vitest,
    ),
    # The unit lanes leave this test closed, having no reason to fetch a JS toolchain.
    Check(
        "app-check-js",
        "tests",
        FULL,
        (
            *UV_RUN,
            "--extra",
            "dev",
            "pytest",
            "tests/cli/test_app_cmd.py",
            "tests/cli/test_app_new.py",
            "-k",
            "pinned_toolchain",
        ),
        scope=(
            "src/osprey/apps/js_check/**",
            "src/osprey/cli/app_cmd.py",
            f"{PRESET_APPS_ROOT}orbit-bump/**",
            f"{SCAFFOLD_APP_ROOT}**",
            "tests/cli/test_app_cmd.py",
            "tests/cli/test_app_new.py",
            "tests/apps/fixtures/js_check_app/**",
        ),
        needs=("free-space", "uv", "node", "install-current"),
        bound_s=600,
        report="junit-pytest",
        prepare=_opt_in_js_check,
    ),
    # The build runs from a clean tree, and every redirect key must leave its
    # stub page in it.
    Check(
        "docs",
        "build",
        FULL,
        (*UV_RUN, "make", "html", "SPHINXOPTS=-W --keep-going"),
        scope=("docs/**", "*.rst", "*.md"),
        needs=("free-space", "uv", "docs-extra", "graphviz"),
        workdir=DOCS_DIR,
        prepare=_clean_docs_build,
        builtin=_check_redirect_stubs,
    ),
    # A link no reader can follow fails the run; an address that answers a
    # reader and not a checker is in `linkcheck_ignore` in docs/source/conf.py.
    Check(
        "linkcheck",
        "build",
        FULL,
        (*UV_RUN, "make", "linkcheck"),
        scope=("docs/**",),
        needs=("free-space", "uv", "docs-extra"),
        workdir=DOCS_DIR,
    ),
    # Implemented in Python: PACKAGE_STEPS, run in order.
    Check(
        "package",
        "build",
        FULL,
        (),
        scope=("pyproject.toml", "packages/**", "src/**", "MANIFEST*"),
        needs=("free-space", "uv"),
        command=_package_command,
        builtin=_check_package,
    ),
)

#: The CI jobs that call the check command; every other required job is a lane.
CALLER_JOBS = ("test", "serial-suites", "lint", "docs", "package", "frontend-js")

#: The paths of the scan stack: a range that touches one needs the container lanes.
SCAN_STACK = (
    "src/osprey/services/bluesky_bridge/**",
    "src/osprey/interfaces/bluesky_web/**",
    "src/osprey/templates/services/bluesky**",
    "src/osprey/mcp_server/bluesky/**",
    "tests/e2e/test_bluesky**",
    "tests/e2e/test_tiled_roundtrip**",
    "tests/e2e/test_grid_scan_roundtrip**",
    "tests/e2e/_orm_stack**",
    "tests/e2e/_queue_drive**",
)

#: The modules ``stack-e2e`` runs, one matrix cell each.
_STACK_E2E_MODULES = tuple(
    f"tests/e2e/{name}"
    for name in (
        "test_bluesky_deploy.py",
        "test_bluesky_web_deploy.py",
        "test_va_substrate_equivalence.py",
        "test_bluesky_queue_e2e.py",
        "test_archiver_world_e2e.py",
        "test_tiled_roundtrip.py",
        "test_bluesky_catalog_e2e.py",
        "test_bluesky_sandbox_escape_e2e.py",
        "test_optimizer_deploy.py",
        "test_execution_environment_parity.py",
        "test_deploy_writeback_e2e.py",
        "test_jupyter_panel_lifecycle.py",
        "test_graph_reseed_stamp.py",
        "web_terminals/test_auth_perimeter.py",
        "web_terminals/test_terminal_auth_multiuser_e2e.py",
    )
)

_PLUGIN_TREES = ("plugins/**", ".claude-plugin/**", ".agents/plugins/**")


def _e2e(
    job: str, *tests: str, why: str = "containers", prefix: str | None = None, id: str = ""
) -> Lane:
    return Lane(job, why, tuple(f"tests/e2e/{name}" for name in tests), prefix, id=id)


#: ``dispatch-stack-e2e``'s cells, each its own spending lane: the cell's
#: ``lane:``, its ``name:`` and its module. The release gate and
#: ``full_ci_on_main`` then require every cell by its own rendered name.
_DISPATCH_STACK_CELLS = (
    ("dispatch-deploy-e2e", "Deploy", "test_dispatch_deploy.py"),
    ("single-user-narrowing-e2e", "Single-User Narrowing", "test_single_user_narrowing_e2e.py"),
    ("dispatch-overlay-e2e", "Overlay Visibility", "test_dispatch_overlay_visibility.py"),
    ("host-record-e2e", "Host Record", "test_host_record_e2e.py"),
    ("scan-agentic-e2e", "Scan Stack Agentic", "test_plan_stack_agentic.py"),
)


def _scan(job: str, *tests: str, why: str = "containers", prefix: str | None = None) -> Lane:
    named = tuple(f"tests/e2e/{name}" for name in tests)
    # SCAN_STACK's globs reach the other scan-stack lanes' modules.
    return Lane(job, why, (*SCAN_STACK, *named), prefix, modules=named)


def _va(job: str, *tests: str) -> Lane:
    return Lane(job, "containers", tuple(f"tests/va/e2e/{name}" for name in tests))


LANES: tuple[Lane, ...] = (
    Lane(
        "dependency-floor",
        "dependency floor",
        (
            "pyproject.toml",
            "uv.lock",
            "tests/connectors/test_archiver_timezone.py",
            "tests/connectors/test_mock_connector.py",
            "tests/connectors/test_mya_registration.py",
            "packages/osprey-connectors/tests/osprey_connectors_tests/test_archiver_timerange.py",
            "packages/osprey-connectors/tests/osprey_connectors_tests/test_doocs_archiver_connector.py",
            "packages/osprey-connectors/tests/osprey_connectors_tests/test_epics_archiver_connector.py",
            "packages/osprey-connectors/tests/osprey_connectors_tests/test_mya_archiver_connector.py",
            "tests/connectors/test_simulation_integration.py",
            "tests/utils/test_timeseries.py",
        ),
    ),
    Lane(
        "framework-at-connectors-floor",
        "installs the published connectors floor",
        (
            "src/osprey/**",
            "packages/pyaml-cs-osprey/**",
            "packages/osprey-connectors/tests/osprey_connectors_tests/compat/**",
            "tests/connectors/**",
            "tests/registry/**",
            "scripts/ci/connectors_compat_lane.py",
            "scripts/ci/connectors_compat_matrix.py",
            "pyproject.toml",
            "uv.lock",
        ),
    ),
    Lane(
        "optimizer",
        "search stack and redis",
        ("packages/osprey-optimizer/**", "tests/optimizer/**", "pyproject.toml", "uv.lock"),
    ),
    Lane(
        "static-checks",
        "builds a project",
        ("tests/integration/test_preset_static.py", "tests/integration/test_preset_build.py"),
    ),
    Lane(
        "visual-theming",
        "browser",
        (
            "src/osprey/interfaces/**",
            "tests/interfaces/**/*.py",
            "tests/interfaces/design_system/baselines/**",
            "tests/docs/screenshots/**",
        ),
    ),
    Lane(
        "visual-theming-agent",
        "browser",
        (
            "src/osprey/interfaces/**",
            "tests/interfaces/**/*.py",
            "tests/interfaces/design_system/baselines/**",
        ),
    ),
    Lane("build-boot-smoke", "matrix", ("tests/integration/test_build_boot.py",)),
    Lane("e2e-tests", SPENDS, ("tests/e2e/**",), "E2E Tests"),
    Lane(
        "e2e-no-model",
        "containers",
        ("tests/e2e/**",),
        modules=tuple(
            f"tests/e2e/{name}"
            for name in (
                "test_scripted_model.py",
                "claude_code/test_safety_gates_scripted.py",
                "test_dispatch_deploy_render.py",
                "web_terminals/test_scaffold_render_roundtrip.py",
                "web_terminals/test_skills_overlay_discovery.py",
                "web_terminals/test_loopback_chokepoint.py",
                "web_terminals/test_session_restart_e2e.py",
                "web_terminals/test_terminal_auth_e2e.py",
                "web_terminals/test_render_permissions_scripted.py",
                "claude_code/test_chat_approval_gates_scripted.py",
                "claude_code/test_chat_plan_mode_scripted.py",
                "claude_code/test_chat_attachment_read_scripted.py",
                "test_watchdog_agent_scripted.py",
                "test_watchdog_hold_scripted.py",
            )
        ),
    ),
    Lane("deploy-e2e", "containers", ("src/osprey/**",)),
    *(
        _e2e(
            "dispatch-stack-e2e",
            module,
            why=SPENDS,
            prefix=f"Dispatch Stack E2E — {name}",
            id=lane,
        )
        for lane, name, module in _DISPATCH_STACK_CELLS
    ),
    Lane(
        "stack-e2e",
        "containers",
        (
            *SCAN_STACK,
            *_STACK_E2E_MODULES,
            "src/osprey/templates/services/optimizer/**",
            "packages/osprey-optimizer/**",
        ),
        # SCAN_STACK's globs reach the scan-stack modules other lanes run.
        modules=_STACK_E2E_MODULES,
    ),
    _scan(
        "orm-roundtrip-e2e",
        "test_orm_roundtrip.py",
        "test_grid_scan_roundtrip.py",
        "test_bump_roundtrip.py",
        "test_preflight_probe.py",
    ),
    _e2e("agent-record-e2e", "test_agent_record_e2e.py"),
    _e2e(
        "nextcloud-talk-bridge-e2e",
        "test_nextcloud_talk_bridge_e2e.py",
        why=SPENDS,
        prefix="Nextcloud Talk Bridge E2E",
    ),
    _e2e(
        "gchat-bridge-e2e",
        "test_gchat_bridge_e2e.py",
        "fixtures/test_gchat_fixture_smoke.py",
        why=SPENDS,
        prefix="Google Chat Bridge E2E",
    ),
    _e2e(
        "teams-slack-bridge-e2e",
        "test_teams_bridge_e2e.py",
        "fixtures/test_teams_fixture_smoke.py",
        "test_slack_bridge_e2e.py",
        "fixtures/test_slack_fixture_smoke.py",
        why="loopback services",
    ),
    _e2e("dockerfile-e2e", "test_dockerfile_e2e.py", "_dockerfile_image.py"),
    _e2e(
        "dockerfile-agent-e2e",
        "test_dockerfile_agent_e2e.py",
        "_dockerfile_image.py",
        why=SPENDS,
        prefix="Dockerfile Agent E2E",
    ),
    _e2e("privilege-split-e2e", "test_privilege_split.py", "test_audit_two_container.py"),
    _e2e(
        "agentic-per-preset",
        "test_preset_agentic.py",
        why=SPENDS,
        prefix="Tier 3 — Agentic flow",
    ),
    _e2e("multi-user-deploy-lifecycle-e2e", "test_deploy_lifecycle.py"),
    _e2e("full-chain-auth-e2e", "test_full_chain_auth.py"),
    _e2e("two-shape-boot-e2e", "test_two_shape_boot.py"),
    Lane(
        "qmd-sidecar-e2e",
        "containers",
        (
            "src/osprey/templates/services/qmd/**",
            "tests/_container_support.py",
            "tests/e2e/test_deploy_lifecycle.py",
            "tests/integration/_qmd_ariel_support.py",
            "tests/integration/_qmd_okf_support.py",
            "tests/integration/test_qmd_ariel_resync_paths.py",
            "tests/integration/test_qmd_ariel_search_sidecar.py",
            "tests/integration/test_qmd_models_mount.py",
            "tests/integration/test_qmd_okf_search_sidecar.py",
        ),
        # test_deploy_lifecycle.py is in scope, and the job runs its `-m qmd` half;
        # multi-user-deploy-lifecycle-e2e hosts the module and runs the rest.
        modules=(
            "tests/integration/test_qmd_ariel_resync_paths.py",
            "tests/integration/test_qmd_ariel_search_sidecar.py",
            "tests/integration/test_qmd_models_mount.py",
            "tests/integration/test_qmd_okf_search_sidecar.py",
        ),
    ),
    Lane(
        "qmd-cache-warmup",
        "main-branch cache",
        ("src/osprey/templates/services/qmd/**", ".github/actions/build-qmd-image/**"),
    ),
    _e2e(
        "target-switch-agentic-e2e",
        "test_target_switch_agentic.py",
        why=SPENDS,
        prefix="Control-Target Switch Agentic E2E",
    ),
    _va(
        "va-live-e2e",
        "test_approval_smoke.py",
        "test_limits_enforcement.py",
        "test_orbit_response.py",
        "test_full_sweep.py",
        "test_finder_live_reads.py",
        "test_scenario_reload.py",
        "test_sim_apply_va.py",
        "test_va_put_surface.py",
        "test_planted_journal.py",
        "test_substrate_parity.py",
        "test_chain_boot.py",
        "test_pyaml_measure_guarded.py",
        "test_mml_trees_boot.py",
        "test_check_b_served.py",
        "test_bluesky_lanes.py",
        "test_target_switch.py",
        "test_bench_ioc_live.py",
        "test_live_standin.py",
        "test_per_target_posture.py",
        "test_model_status_va.py",
        "test_ca_loop_exit.py",
    ),
    Lane("plugin-smoke", "plugin install", _PLUGIN_TREES),
    Lane(
        "benchmarks",
        "on-demand benchmark",
        ("tests/e2e/claude_code/test_channel_finder_mcp_benchmarks.py",),
        modules=(
            "tests/e2e/claude_code/test_channel_finder_mcp_benchmarks.py",
            "tests/e2e/tier2/test_ariel_picture_run_id.py",
        ),
    ),
)

#: The lane that runs every ``tests/e2e`` test module no other host claims.
SHARED_E2E_LANE = "e2e-tests"

#: The host of the modules no CI job runs.
OFF_CI = "off-ci"

#: The ``tests/e2e`` test modules no CI job runs, each with the reason.
E2E_OFF_CI: dict[str, str] = {
    "tests/e2e/test_watchdog_live.py": "spends a direct-route key per call; run by hand",
    "tests/e2e/tier2/test_chart_value_reading.py": "deploys a full stack; benchmark board only",
}


def is_e2e_module(path: str) -> bool:
    """Whether *path* is a test module of ``tests/e2e``, at any depth."""
    return path.startswith("tests/e2e/") and kit.matches(path, "test_*.py")


def e2e_hosts(lanes: Iterable[Lane], files: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Each ``tests/e2e`` test module of *files* to the hosts that claim it.

    A lane claims the modules its ``modules`` globs match, or its scope's where
    it names none; :data:`E2E_OFF_CI` claims its own. A module nothing claims
    is :data:`SHARED_E2E_LANE`'s, so the shared lane's ``--ignore`` list is
    every other module, and a module two hosts claim breaks the invariant that
    each runs in exactly one place.
    """
    claims = [
        (lane.id, lane.scope if lane.modules is None else lane.modules)
        for lane in lanes
        if lane.id != SHARED_E2E_LANE
    ]
    hosts = {}
    for path in sorted(filter(is_e2e_module, files)):
        named = [job for job, globs in claims if any(kit.matches(path, g) for g in globs)]
        if path in E2E_OFF_CI:
            named.append(OFF_CI)
        hosts[path] = tuple(named) or (SHARED_E2E_LANE,)
    return hosts


def e2e_shared_ignores(lanes: Iterable[Lane], files: Iterable[str]) -> list[str]:
    """The modules the shared lane leaves to another host, sorted."""
    return [path for path, hosts in e2e_hosts(lanes, files).items() if hosts != (SHARED_E2E_LANE,)]


#: What ``list --json`` prints beside the checks and the lanes.
LISTED = {
    "unit_selection": dataclasses.asdict(UNIT_SELECTION),
    "invariant_tests": list(INVARIANT_TESTS),
}
