"""Promotion and disposition needles for the deploy path's INFO narration.

One question per test, asked of the two instruments together:

* **printed** -- did the fact reach the default view, as the verb's own output?
  Echo-class lines go through :mod:`osprey.cli.output`, which resolves
  ``current_reporter().out()`` per call, so the capture reporter below is what
  reads them.
* **painted / recorded** -- did the *old* log form stay out of the terminal
  while its record stayed in the sink? That pair is the honesty half: a
  promotion that quietly deleted the record would pass a "not in the terminal"
  assertion for entirely the wrong reason.

Every absence assertion here carries an armed witness (see
:func:`assert_promoted`), because a console nothing can reach satisfies every
``not in`` there is.

Tasks 3.3, 3.4 and 3.7 append their own sections below; the fixtures and
helpers above the first section marker are shared.

A WARNING is a third case with two states. With no reporter installed the
altitude gate passes it and it stays on the *painted* side; while a reporter
owns the terminal the gate drops it there too, and anything an operator must
read arrives promoted through ``warn_fact`` instead — the unshown-mint warning
below is pinned in both states.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from rich.console import Console

from osprey.cli import build_cmd, build_persistence, deploy_cmd, output
from osprey.cli.build_persistence import _apply_conventions
from osprey.cli.main import cli
from osprey.cli.phase_reporter import PhaseReporter, install_reporter
from osprey.cli.profile_conventions import PROJECT_MIRROR_DIR
from osprey.deployment import compose_generator, container_lifecycle, wheel_build
from osprey.deployment.web_terminals import (
    auth_credentials,
    env_production,
    persona_images,
    postup_hooks,
    provision,
    seeding,
)
from osprey.simulation import apply
from tests.cli.conftest import TerminalProbe

# The witness logger and its marker. An ERROR is above the altitude gate on
# every path -- WARNING no longer is while a reporter is installed, which these
# tests' capture reporter counts as -- so it MUST reach the probe console; if
# it does not, the console is not armed and nothing else this module asserts
# about absence means anything.
_WITNESS_LOGGER = "osprey.lifecycle_promotion_witness"
_WITNESS = "PROMOTIONWITNESSMARKER"


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------


class _CaptureReporter(PhaseReporter):
    """A reporter whose console is a buffer, so printed lines are readable.

    ``out()`` is the seam every echo-class primitive resolves per call, and
    :meth:`osprey.cli.phase_reporter.Phase.step` prints through ``emit`` on the
    same console -- so one buffer holds both a verb's report lines and the
    ``  · name`` sub-steps hung off its phase.
    """

    def __init__(self, console: Console) -> None:
        super().__init__(color=False)
        self._console = console

    def out(self) -> Console:
        return self._console


class Printed:
    """What the operator saw, plus the phase machinery to hang steps off."""

    def __init__(self, reporter: _CaptureReporter, buffer: StringIO) -> None:
        self.reporter = reporter
        self._buffer = buffer

    @property
    def text(self) -> str:
        """Everything printed so far, soft-wrapped exactly as a terminal gets it."""
        return self._buffer.getvalue()

    @property
    def flowed(self) -> str:
        """:attr:`text` with every run of whitespace collapsed to one space.

        The capture console is 100 columns wide, so a long report line wraps.
        Assert a multi-word fragment against this rather than against
        :attr:`text`.
        """
        return " ".join(self.text.split())

    def open_phase(self, title: str = "Starting services") -> None:
        """Open a phase, so ``report_step`` has somewhere to hang its lines."""
        self.reporter.phase(title)


@pytest.fixture
def printed() -> Iterator[Printed]:
    """Install a capture reporter for the duration of one test."""
    # Imported here for the same reason conftest does it: the theme module
    # pulls in the CLI package, and this conftest runs for all of tests/cli.
    from osprey.cli.styles import osprey_theme

    buffer = StringIO()
    console = Console(
        file=buffer,
        width=100,
        force_terminal=False,
        legacy_windows=False,
        theme=osprey_theme,
    )
    reporter = _CaptureReporter(console)
    previous = install_reporter(reporter)
    try:
        yield Printed(reporter, buffer)
    finally:
        install_reporter(previous)


@pytest.fixture
def default_altitude(
    terminal_probe: TerminalProbe, monkeypatch: pytest.MonkeyPatch
) -> TerminalProbe:
    """A default (``-v``-less) run's logging state, with the probe watching it.

    Invoking the bare group runs the real callback -- ``configure_logging``,
    the handler console swap, and ``install_gate`` -- so what the handler does
    with a record here is what it does under ``osprey up``.
    ``load_project_dotenv`` is stubbed because it writes an ancestor ``.env``
    straight into ``os.environ``, which would outlive the test.
    """
    monkeypatch.setattr("osprey.utils.config.load_project_dotenv", lambda *a, **k: None)
    result = CliRunner().invoke(cli, [])
    assert result.exit_code == 0, result.output
    logging.getLogger().setLevel(logging.INFO)
    return terminal_probe


def assert_promoted(probe: TerminalProbe, printed: Printed, fragment: str) -> None:
    """The promotion needle, all three halves plus the witness.

    Args:
        probe: The terminal probe, armed by ``default_altitude``.
        printed: The capture reporter's buffer.
        fragment: A distinctive piece of the fact, short enough to survive the
            capture console's wrapping.
    """
    assert fragment in printed.flowed, f"not in the default view: {fragment!r}"
    assert fragment not in probe.rendered_text, f"still painted by the log handler: {fragment!r}"
    assert any(fragment in message for message in probe.messages), (
        f"no record kept for: {fragment!r}"
    )

    # Armed witness, in the same test: an ERROR is above the gate on every
    # path, so its absence would mean the probe console was never reachable
    # and the absence assertion above proved nothing.
    logging.getLogger(_WITNESS_LOGGER).error(_WITNESS)
    assert _WITNESS in probe.rendered_text, "the probe console is not armed"


def assert_warned(probe: TerminalProbe, promoted: str, fragment: str) -> None:
    """The warning needle: the ``warn_fact`` counterpart to :func:`assert_promoted`.

    A promoted FACT goes through the reporter, so its needle is the reporter's
    buffer. A promoted WARNING goes to the trouble stream instead, so reading
    ``printed`` for one would pass vacuously against an empty buffer. Everything
    else is the same three halves plus the witness.

    Args:
        probe: The terminal probe, armed by ``default_altitude``.
        promoted: The run's stderr, from ``capsys``.
        fragment: A distinctive piece of the warning, short enough to survive
            the console's wrapping once whitespace is collapsed.
    """
    flowed = " ".join(promoted.split())
    assert fragment in flowed, f"not in the promoted block: {fragment!r}\n{flowed}"
    assert fragment not in probe.rendered_text, f"still painted by the log handler: {fragment!r}"
    assert any(fragment in message for message in probe.messages), (
        f"no record kept for: {fragment!r}"
    )

    # Armed witness, as in assert_promoted: without it the absence assertion
    # above would also pass on a probe console nothing could ever reach.
    logging.getLogger(_WITNESS_LOGGER).error(_WITNESS)
    assert _WITNESS in probe.rendered_text, "the probe console is not armed"


def assert_sub_step(printed: Printed, name: str) -> None:
    """A ``sub-step`` row's needle: the phase record carries ``· name``."""
    assert re.search(rf"·\s+{re.escape(name)}", printed.flowed), (
        f"no sub-step line for: {name!r}\n{printed.flowed}"
    )


# ---------------------------------------------------------------------------
# --- Task 3.2: container_lifecycle promotions ---
# ---------------------------------------------------------------------------


BLUESKY_VA = {"deployed_services": ["bluesky", "virtual_accelerator"]}

#: A project that runs ARIEL's own store, which is what makes the staged
#: bring-up do anything at all (see ``_ariel_store_deployed``).
ARIEL_PROJECT = {"deployed_services": ["postgresql"], "ariel": {"database": "ariel"}}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project directory holding an empty ``.env``, the provisioners' target."""
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    return env_path


class TestMintedSecretsAreReported:
    """Anything this deploy created that the operator now owns."""

    def test_a_minted_auth_token_names_its_keys_and_its_file(
        self, default_altitude, printed, project, monkeypatch
    ):
        monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)
        monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)

        container_lifecycle._ensure_service_tokens(
            {"deployed_services": ["event_dispatcher"]}, False, project
        )

        assert_promoted(default_altitude, printed, "minted 2 service auth token(s)")
        output.flush_ledger()
        assert "EVENT_DISPATCHER_TOKEN" in printed.flowed
        assert "keep them secret" in printed.flowed

    def test_an_adopted_volume_credential_names_the_variables_never_the_values(
        self, default_altitude, printed, project
    ):
        store = container_lifecycle._VOLUME_INITIALIZED_VARS["MONGO_ROOT_PASSWORD"]
        project.write_text("MONGO_ROOT_PASSWORD=freshmint\n", encoding="utf-8")

        container_lifecycle._adopt_original_credentials(
            project,
            [("MONGO_ROOT_PASSWORD", store)],
            {"MONGO_ROOT_PASSWORD": "theoriginal"},
        )

        assert_promoted(default_altitude, printed, "Adopted 1 pre-existing data volume(s)")
        assert "MONGO_ROOT_PASSWORD" in printed.flowed
        assert "theoriginal" not in printed.flowed

    def test_a_minted_qserver_keypair_says_both_halves_are_secret(
        self, default_altitude, printed, project, monkeypatch
    ):
        monkeypatch.delenv(container_lifecycle._QSERVER_ZMQ_PRIVATE_KEY_VAR, raising=False)
        monkeypatch.delenv(container_lifecycle._QSERVER_ZMQ_PUBLIC_KEY_VAR, raising=False)

        container_lifecycle._ensure_bluesky_control_plane_keys(
            {"deployed_services": ["bluesky"]}, project
        )

        assert_promoted(default_altitude, printed, "bluesky control-socket keypair minted")
        output.flush_ledger()
        assert "treat both halves as secrets" in printed.flowed

    def test_a_derived_public_half_says_it_was_not_written_to_disk(
        self, default_altitude, printed, project, monkeypatch
    ):
        """The private half lives only in this shell, so nothing lands on disk."""
        private = container_lifecycle._derive_qserver_keypair("")[
            container_lifecycle._QSERVER_ZMQ_PRIVATE_KEY_VAR
        ]
        monkeypatch.setenv(container_lifecycle._QSERVER_ZMQ_PRIVATE_KEY_VAR, private)
        monkeypatch.delenv(container_lifecycle._QSERVER_ZMQ_PUBLIC_KEY_VAR, raising=False)

        container_lifecycle._ensure_bluesky_control_plane_keys(
            {"deployed_services": ["bluesky"]}, project
        )

        assert_promoted(default_altitude, printed, "for this deploy only")
        assert container_lifecycle._QSERVER_ZMQ_PUBLIC_KEY_VAR in printed.flowed

    def test_generated_curve_certificates_say_when_they_take_effect(
        self, default_altitude, printed, project, monkeypatch
    ):
        monkeypatch.setattr(
            container_lifecycle, "_generate_bluesky_curve_certs", lambda paths: None
        )

        container_lifecycle._ensure_bluesky_document_plane_certs(
            {"deployed_services": ["bluesky"]}, project
        )

        assert_promoted(default_altitude, printed, "bluesky CURVE certificates")
        output.flush_ledger()
        assert "bridge and queueserver recreated" in printed.flowed


class TestDegradationsAreReported:
    """A deployment that will not do what it looks like it does says so."""

    def test_a_mock_control_system_reports_the_browse_only_deployment(
        self, default_altitude, printed, project
    ):
        container_lifecycle._ensure_bluesky_substrate_env(
            {**BLUESKY_VA, "control_system": {"type": "mock"}}, project
        )

        assert_promoted(default_altitude, printed, "this deployment is browse-only")
        assert "osprey set connector=virtual_accelerator" in printed.flowed

    def test_a_pinned_worker_image_reports_the_build_it_skipped(
        self, default_altitude, printed, monkeypatch
    ):
        monkeypatch.setattr(container_lifecycle, "resolve_project_name", lambda config: "demo")

        container_lifecycle._build_project_image(
            {"deployed_services": ["dispatch_worker"]},
            False,
            {"OSPREY_WORKER_IMAGE": "ghcr.io/example/worker:1"},
        )

        assert_promoted(default_altitude, printed, "Dispatch worker uses image")
        assert "ghcr.io/example/worker:1" in printed.flowed
        assert "Skipping the demo:local build." in printed.flowed

    def test_an_empty_service_roster_says_nothing_was_started(
        self, default_altitude, printed, tmp_path
    ):
        container_lifecycle._start_stack({}, [], tmp_path)

        assert_promoted(default_altitude, printed, "No services are configured for this project")
        assert "Skipping osprey up." in printed.flowed


class TestAutonomousHostChangesAreReported:
    """What the deploy wrote, or decided, without being asked to."""

    def test_derived_plan_devices_name_the_file_they_came_from(
        self, default_altitude, printed, project, monkeypatch
    ):
        monkeypatch.setattr(
            "osprey.services.bluesky_bridge.substrate_devices.derive_substrate_env",
            lambda project_dir: {"BLUESKY_EPICS_SETPOINTS": "SR:C01:COR"},
        )

        container_lifecycle._ensure_bluesky_substrate_env(
            {**BLUESKY_VA, "control_system": {"type": "virtual_accelerator"}}, project
        )

        assert_promoted(default_altitude, printed, "bluesky plan devices auto-configured")
        output.flush_ledger()
        assert "channel_limits.json" in printed.flowed
        assert "BLUESKY_EPICS_SETPOINTS" in printed.flowed

    def test_a_local_env_override_is_reported_by_name(self, default_altitude, printed, tmp_path):
        from osprey.utils.dotenv import ENV_SHARED_FILENAME

        (tmp_path / ENV_SHARED_FILENAME).write_text("ARIEL_DB_PASSWORD=shared\n", encoding="utf-8")
        (tmp_path / ".env").write_text("ARIEL_DB_PASSWORD=mine\n", encoding="utf-8")

        container_lifecycle._report_chain_overrides(tmp_path)

        assert_promoted(default_altitude, printed, "ARIEL_DB_PASSWORD")
        assert "the local file winning" in printed.flowed
        assert "mine" not in printed.flowed.split("ARIEL_DB_PASSWORD")[-1]

    def test_dev_mode_reports_the_variable_it_set_for_the_containers(
        self, default_altitude, printed, tmp_path, monkeypatch
    ):
        _stub_start_stack_preflights(monkeypatch)

        with pytest.raises(_StopBeforeTheBuild):
            container_lifecycle._start_stack(
                {"deployed_services": ["event_dispatcher"]}, [], tmp_path, dev_mode=True
            )

        assert_promoted(default_altitude, printed, "dev mode: DEV_MODE")


class TestDestructiveActionsAreReported:
    """Anything the operator would want to have been told before it happened."""

    def test_the_archive_rebuild_says_history_is_discarded(
        self, default_altitude, printed, archiver_stubs, tmp_path
    ):
        archiver_stubs["state"] = _seed_state("MISMATCH")

        container_lifecycle._stage_archiver_store(
            {"deployed_services": ["archiver_store"]}, ["compose.yml"], {}, tmp_path
        )

        assert_promoted(default_altitude, printed, "Rebuilding the base series")
        assert "--keep-archiver-base" in printed.flowed

    def test_a_label_sweep_says_how_many_containers_it_will_stop(
        self, default_altitude, printed, tmp_path, monkeypatch
    ):
        _stub_label_sweep(monkeypatch, container_ids=["abc123", "def456"])

        container_lifecycle._down_by_label({}, tmp_path)

        assert_promoted(default_altitude, printed, "stopping the 2 container(s) labelled")
        assert "volumes and the compose network are kept" in printed.flowed

    def test_finding_nothing_to_stop_states_the_limit_of_the_sweep(
        self, default_altitude, printed, tmp_path, monkeypatch
    ):
        _stub_label_sweep(monkeypatch, container_ids=[])

        container_lifecycle._down_by_label({}, tmp_path)

        assert_promoted(default_altitude, printed, "nothing to stop")
        # The sweep's limit is a caveat for the transcript, not the phase
        # column: the record is kept, the default view stays one line.
        assert any("may still be running" in message for message in default_altitude.messages), (
            "the log sinks lost the sweep's limit"
        )
        assert "may still be running" not in printed.flowed


class TestSubStepRows:
    """The disposition table's ``sub-step`` rows, at the default altitude."""

    def test_a_matching_fingerprint_says_why_the_seed_did_not_run(
        self, default_altitude, printed, archiver_stubs, tmp_path
    ):
        archiver_stubs["state"] = _seed_state("MATCH")
        printed.open_phase()

        container_lifecycle._stage_archiver_store(
            {"deployed_services": ["archiver_store"]}, ["compose.yml"], {}, tmp_path
        )

        assert_sub_step(printed, "archive already seeded, skipping the base seed")

    def test_the_base_seed_sets_the_expectation_before_it_starts(
        self, default_altitude, printed, archiver_stubs, tmp_path
    ):
        printed.open_phase()

        container_lifecycle._stage_archiver_store(
            {"deployed_services": ["archiver_store"]}, ["compose.yml"], {}, tmp_path
        )

        assert_sub_step(printed, "seeding the archive base: 2 channels over 7 days")
        assert "(minutes on a first deploy)" in printed.flowed

    def test_the_finished_base_seed_reports_what_it_wrote(
        self, default_altitude, printed, archiver_stubs, tmp_path
    ):
        printed.open_phase()

        container_lifecycle._stage_archiver_store(
            {"deployed_services": ["archiver_store"]}, ["compose.yml"], {}, tmp_path
        )

        assert_sub_step(printed, "archive base: seeded 10 documents")

    def test_the_reseed_closes_on_the_scenarios_it_put_back(
        self, default_altitude, printed, tmp_path, monkeypatch
    ):
        _stub_reapply(monkeypatch, active=("nominal", "bpm_dropout"), archiver_describe="rewrote 3")
        printed.open_phase()

        container_lifecycle._reapply_active_scenarios(
            {}, tmp_path, _engine("nominal", "bpm_dropout")
        )

        assert_sub_step(printed, "scenarios re-applied: nominal, bpm_dropout")
        assert "(rewrote 3)" in printed.flowed

    def test_a_skipped_archive_rewrite_is_not_appended_to_the_reseed_line(
        self, default_altitude, printed, tmp_path, monkeypatch
    ):
        """Row 1's condition: the parenthetical is the rewrite's counts or nothing.

        ``skipped`` already says why nothing happened, in words that do not fit
        on the end of a step name -- and a ``None`` archiver never ran at all.
        """
        _stub_reapply(monkeypatch, active=("nominal",), archiver_describe=None)
        printed.open_phase()

        container_lifecycle._reapply_active_scenarios({}, tmp_path, _engine("nominal"))

        assert_sub_step(printed, "scenarios re-applied: nominal")
        assert "(" not in printed.flowed.split("scenarios re-applied")[-1]

    def test_no_archiver_at_all_leaves_the_reseed_line_bare(
        self, default_altitude, printed, tmp_path, monkeypatch
    ):
        _stub_reapply(monkeypatch, active=("nominal",), archiver=None)
        printed.open_phase()

        container_lifecycle._reapply_active_scenarios({}, tmp_path, _engine("nominal"))

        assert_sub_step(printed, "scenarios re-applied: nominal")
        assert "(" not in printed.flowed.split("scenarios re-applied")[-1]

    def test_the_seeded_logbook_reports_its_count(
        self, default_altitude, printed, tmp_path, monkeypatch
    ):
        _stub_ariel_stage(monkeypatch, seeded=12)
        printed.open_phase()

        container_lifecycle._stage_ariel_store(ARIEL_PROJECT, ["compose.yml"], {}, tmp_path)

        assert_sub_step(printed, "logbook seeded: 12 entries")


class TestDemotedRows:
    """Appendix A, plus the *legacy* rows: emitted, but below the default view."""

    def test_present_curve_certificates_are_a_debug_line(self, project, caplog):
        paths = container_lifecycle._bluesky_curve_paths(project.parent)
        for role in ("proxy_secret", "publisher_secret", "proxy_public", "publisher_public"):
            paths[role].parent.mkdir(parents=True, exist_ok=True)
            paths[role].write_text("key", encoding="utf-8")

        with caplog.at_level(logging.DEBUG):
            container_lifecycle._ensure_bluesky_document_plane_certs(
                {"deployed_services": ["bluesky"]}, project
            )

        assert _levels_for(caplog, "already present") == {logging.DEBUG}

    def test_having_no_machine_model_to_reseed_is_a_debug_line(self, tmp_path, caplog):
        with caplog.at_level(logging.DEBUG):
            container_lifecycle._reapply_active_scenarios({}, tmp_path, None)

        assert _levels_for(caplog, "no scenarios to re-apply") == {logging.DEBUG}

    def test_the_legacy_down_compose_file_listing_is_a_debug_line(
        self, tmp_path, monkeypatch, caplog
    ):
        _stub_legacy_down(monkeypatch, tmp_path, compose_files=["docker-compose.yml"])

        # `deploy_down` exits the process on the runtime's status, which is 0
        # here; the narration under test has already been emitted by then.
        with caplog.at_level(logging.DEBUG), pytest.raises(SystemExit):
            container_lifecycle.deploy_down(tmp_path / "config.yml")

        assert _levels_for(caplog, "Using existing compose files") == {logging.DEBUG}
        assert _levels_for(caplog, "- docker-compose.yml") == {logging.DEBUG}

    def test_the_legacy_down_rerender_is_a_debug_line(self, tmp_path, monkeypatch, caplog):
        _stub_legacy_down(monkeypatch, tmp_path, compose_files=[])

        with caplog.at_level(logging.DEBUG), pytest.raises(SystemExit):
            container_lifecycle.deploy_down(tmp_path / "config.yml")

        assert _levels_for(caplog, "No existing compose files found") == {logging.DEBUG}

    def test_the_legacy_detached_restart_line_is_a_debug_line(self, tmp_path, monkeypatch, caplog):
        _stub_legacy_restart(monkeypatch, tmp_path)

        with caplog.at_level(logging.DEBUG):
            container_lifecycle.deploy_restart(tmp_path / "config.yml", detached=True)

        assert _levels_for(caplog, "Running in detached mode") == {logging.DEBUG}


class TestVerboseKeepsThePrimaryOutput:
    """Criterion 3, for this module: ``-v`` adds the transcript, loses nothing."""

    def test_a_promoted_fact_is_printed_under_verbose_too(
        self, terminal_probe, printed, project, monkeypatch
    ):
        monkeypatch.setattr("osprey.utils.config.load_project_dotenv", lambda *a, **k: None)
        assert CliRunner().invoke(cli, ["-v"]).exit_code == 0

        container_lifecycle._ensure_bluesky_substrate_env(
            {**BLUESKY_VA, "control_system": {"type": "mock"}}, project
        )

        # Printed by the renderer, AND painted from the record: under -v the
        # transcript is what was asked for, so the fact appears in both.
        assert "this deployment is browse-only" in printed.flowed
        assert "browse-only" in terminal_probe.rendered_text


class TestPromotedCopyStyle:
    """The copy guard's rules, reapplied where the guard cannot see.

    ``_report_fact`` takes the finished line, so the literals at its call sites
    are invisible to ``tests/cli/test_printed_copy_style.py`` (which matches on
    the printer's own name). These facts are printed at an operator, so the
    same rules hold: this walk is what holds them.
    """

    def _literals(self) -> list[tuple[int, str]]:
        source = Path(container_lifecycle.__file__).read_text(encoding="utf-8")
        found: list[tuple[int, str]] = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
            if name not in {"_report_fact", "_report_step"}:
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.append((node.lineno, arg.value))
                elif isinstance(arg, ast.JoinedStr):
                    found.append(
                        (
                            node.lineno,
                            "".join(
                                part.value
                                for part in arg.values
                                if isinstance(part, ast.Constant) and isinstance(part.value, str)
                            ),
                        )
                    )
        return found

    def test_the_walk_reaches_the_promoted_lines(self):
        """A rename that hides these call sites fails here, not silently."""
        literals = self._literals()
        assert len(literals) >= 14, f"only {len(literals)} promoted literals found"

    def test_no_promoted_line_carries_an_em_dash(self):
        offences = [f"{line}: {text}" for line, text in self._literals() if "—" in text]
        assert not offences, "Em-dash in a promoted line:\n  " + "\n  ".join(offences)

    def test_no_promoted_line_carries_in_house_jargon(self):
        from tests.cli.test_printed_copy_style import _JARGON

        offences = [
            f"{line}: {term!r} -> {plain}"
            for line, text in self._literals()
            for term, plain in _JARGON.items()
            if re.search(rf"\b{re.escape(term)}\b", text.lower())
        ]
        assert not offences, "In-house jargon in a promoted line:\n  " + "\n  ".join(offences)


# ---------------------------------------------------------------------------
# Task 3.2 stubs
# ---------------------------------------------------------------------------


class _StopBeforeTheBuild(RuntimeError):
    """Sentinel: ``_start_stack`` reached the image build, so the test is done."""


def _stub_start_stack_preflights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reduce ``_start_stack`` to the dev-mode branch and stop at the build.

    Every preflight between the roster check and the ``DEV_MODE`` line asks the
    host a question this test has no answer for; each is covered by its own
    module's tests.
    """
    for name in (
        "_check_shared_disk_preflight",
        "_preflight_host_ports",
        "_preflight_archiver_pymongo",
        "_preflight_stale_store_volumes",
        "_ensure_bluesky_substrate_env",
        "_ensure_bluesky_control_plane_keys",
        "_ensure_bluesky_document_plane_certs",
        "_preflight_env_chain_drift",
        "_preflight_env_shadowing",
    ):
        monkeypatch.setattr(container_lifecycle, name, lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: set())
    monkeypatch.setattr(container_lifecycle, "_compose_provider", lambda config: None)

    def _stop(*args, **kwargs):
        raise _StopBeforeTheBuild

    monkeypatch.setattr(container_lifecycle, "_build_project_image", _stop)


def _stub_label_sweep(monkeypatch: pytest.MonkeyPatch, *, container_ids: list[str]) -> None:
    """Answer the label sweep's ``ps`` with ``container_ids`` and nothing else."""
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )

    def _run(cmd, **kwargs):
        stdout = "\n".join(container_ids) + ("\n" if container_ids else "")
        return subprocess.CompletedProcess(list(cmd), 0, stdout=stdout, stderr="")

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _run)


def _seed_state(name: str):
    """The archiver seeder's fingerprint verdict, by ``SeedState`` member name."""
    from osprey.simulation import archiver_seed

    return getattr(archiver_seed.SeedState, name)


@pytest.fixture
def archiver_stubs(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Reduce ``_stage_archiver_store`` to the lines this module owns.

    The store connection, the fingerprint comparison and the base rewrite are
    each covered by their own module's tests; what is under test here is which
    of them says something, and at what altitude.
    """
    from osprey.simulation import apply as apply_mod
    from osprey.simulation import archiver_seed as seed_mod

    state: dict = {"state": _seed_state("ABSENT")}

    monkeypatch.setattr(
        container_lifecycle,
        "_archiver_store_connection",
        lambda config, project_dir: {
            "password": "pw",
            "password_env": "MONGO_PW",
            "username": "root",
            "host": "localhost",
            "port": 27017,
        },
    )
    monkeypatch.setattr(container_lifecycle, "runtime_env", lambda config, env: dict(env))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle, "compose_base_cmd", lambda *a, **k: ["docker"])
    monkeypatch.setattr(container_lifecycle, "_env_file_args", lambda *a, **k: [])
    monkeypatch.setattr(container_lifecycle, "run_captured", lambda *a, **k: None)
    monkeypatch.setattr(
        container_lifecycle,
        "_archiver_seed_inputs",
        lambda config, project_dir: (
            [{"address": "SR:BPM1:X"}, {"address": "SR:BPM2:X"}],
            None,
            {},
        ),
    )
    monkeypatch.setattr(container_lifecycle, "_wait_for_archiver_store", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "_reapply_active_scenarios", lambda *a, **k: None)

    knobs = SimpleNamespace(
        retention_days=7, hot_span_hours=6, hot_cadence_sec=1, tail_cadence_sec=60
    )
    monkeypatch.setattr(
        seed_mod, "SeedKnobs", SimpleNamespace(from_config=staticmethod(lambda config: knobs))
    )
    monkeypatch.setattr(seed_mod, "seed_fingerprint", lambda *a, **k: "fp")
    # The real comparison object, not a stand-in: the knob diff is rendered from
    # its `differences` and recorded from its `describe()`, so a fake carrying
    # only one of the two would let the other half of that pairing rot.
    monkeypatch.setattr(
        seed_mod,
        "compare_fingerprint",
        lambda collection, fingerprint: seed_mod.FingerprintComparison(
            state=state["state"], differences=(("retention_days", 5, 7),)
        ),
    )
    monkeypatch.setattr(
        seed_mod,
        "seed_base",
        lambda *a, **k: SimpleNamespace(describe=lambda: "seeded 10 documents x 2 channels"),
    )

    class _Collection:
        def drop(self) -> None:
            pass

    class _Connection:
        def __enter__(self):
            return _Collection()

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(apply_mod, "archiver_collection", lambda store: _Connection())
    return state


def _stub_reapply(
    monkeypatch: pytest.MonkeyPatch,
    *,
    active: tuple[str, ...],
    archiver_describe: str | None = None,
    archiver: object = ...,
) -> None:
    """Hand ``_reapply_active_scenarios`` a finished ``ApplyResult``.

    Args:
        active: Scenario names the re-apply put back.
        archiver_describe: The rewrite's one-line summary, or ``None`` for a
            result that was skipped and says why.
        archiver: Pass ``None`` for a result whose rewrite never ran at all.
    """
    from osprey.simulation import apply as apply_mod

    if archiver is ...:
        archiver = SimpleNamespace(
            skipped=None if archiver_describe else "the archive has not been seeded yet",
            describe=lambda: archiver_describe or "archive not rewritten",
        )
    result = SimpleNamespace(active=active, archiver=archiver)

    monkeypatch.setattr(apply_mod, "apply_scenarios", lambda *a, **k: result)
    monkeypatch.setattr(apply_mod, "persisted_scenario_anchor", lambda config, project_dir: None)


def _engine(*active: str) -> SimpleNamespace:
    """A machine model that reports ``active`` as its live scenario set."""
    return SimpleNamespace(active_scenarios=lambda: list(active))


def _stub_ariel_stage(monkeypatch: pytest.MonkeyPatch, *, seeded: int) -> None:
    """Reduce ``_stage_ariel_store`` to its logbook seed."""
    from osprey.simulation import apply as apply_mod

    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle, "compose_base_cmd", lambda *a, **k: ["docker"])
    monkeypatch.setattr(container_lifecycle, "_env_file_args", lambda *a, **k: [])
    monkeypatch.setattr(container_lifecycle, "runtime_env", lambda config, env: dict(env))
    monkeypatch.setattr(container_lifecycle, "run_captured", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "_ariel_store_config", lambda *a, **k: {})
    monkeypatch.setattr(container_lifecycle, "_wait_for_ariel_store", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "_migrate_ariel_store", lambda *a, **k: None)
    monkeypatch.setattr(apply_mod, "seed_active_logbook", lambda *a, **k: seeded)


def _stub_legacy_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, compose_files: list[str]
) -> None:
    """Reduce the *legacy* ``deploy_down`` to its compose-file narration.

    ``tmp_path`` becomes the working directory because the captured `down`
    spools a file under ``var/logs/`` relative to it.
    """
    from osprey.deployment import compose_generator

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(container_lifecycle, "load_project_config", lambda *a, **k: {})
    monkeypatch.setattr(container_lifecycle, "_web_terminals_enabled", lambda config: False)
    monkeypatch.setattr(
        compose_generator, "find_existing_compose_files", lambda *a, **k: list(compose_files)
    )
    monkeypatch.setattr(
        container_lifecycle, "prepare_compose_files", lambda *a, **k: ({}, ["rendered.yml"])
    )
    monkeypatch.setattr(container_lifecycle, "resolve_repo_root", lambda *a, **k: Path("."))
    monkeypatch.setattr(container_lifecycle, "_compose_provider", lambda config: None)
    monkeypatch.setattr(container_lifecycle, "compose_base_cmd", lambda *a, **k: ["docker"])
    monkeypatch.setattr(container_lifecycle, "_env_file_args", lambda *a, **k: [])
    monkeypatch.setattr(container_lifecycle, "runtime_env", lambda config, env: dict(env))
    monkeypatch.setattr(
        container_lifecycle, "with_plain_progress", lambda cmd: list(cmd) if cmd else ["docker"]
    )
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(list(cmd), 0),
    )


def _stub_legacy_restart(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reduce the *legacy* ``deploy_restart`` to its detached-mode line."""
    _stub_legacy_down(monkeypatch, tmp_path, compose_files=["docker-compose.yml"])
    monkeypatch.setattr(
        container_lifecycle, "_ensure_bluesky_control_plane_keys", lambda *a, **k: None
    )
    # `restart` probes the runtime where `down` does not, and raises its own
    # error rather than going through the stubbed `get_runtime_command`. Without
    # this the narration under test is unreachable on any host with no container
    # runtime installed.
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))


def _levels_for(caplog: pytest.LogCaptureFixture, fragment: str) -> set[int]:
    """The levels every captured record carrying ``fragment`` was emitted at."""
    levels = {record.levelno for record in caplog.records if fragment in record.getMessage()}
    assert levels, f"nothing was logged carrying: {fragment!r}"
    return levels


# ---------------------------------------------------------------------------
# --- Task 3.4: the remaining deploy-path sites ---
# ---------------------------------------------------------------------------
#
# The disposition table's rows for ``compose_generator``, ``wheel_build``,
# ``deploy_cmd``, ``build_persistence`` and ``build_cmd``, plus the two rows in
# ``simulation/apply.py`` the reseed absorbs. Stubs for this section sit in the
# labelled block at the very bottom of the module.


class TestPromotedFactsOnTheMiscDeployPath:
    """Appendix C's rows for these files: printed, unpainted, still recorded."""

    def test_the_env_file_a_start_verb_seeds_is_reported(
        self, default_altitude, printed, monkeypatch, tmp_path
    ):
        """``osprey up`` writing the operator's secret store is not a detail."""
        _offer_the_env_seed(monkeypatch, answer=True)
        repo = tmp_path / "repo"
        repo.mkdir()

        deploy_cmd.ensure_repo_env(repo, {"claude_code": {"provider": _SEED_PROVIDER}})

        assert (repo / ".env").is_file(), "the seed never happened"
        assert_promoted(default_altitude, printed, "(mode 0600) with")

    def test_a_runtime_root_build_says_why_no_compose_was_rendered(self, default_altitude, printed):
        """The build silently produced no compose files; that has to be said."""
        assert build_cmd._render_compose_files(None, "/srv/osprey", dev_mode=False) is None

        assert_promoted(default_altitude, printed, "Compose files not rendered")

    def test_the_one_env_write_outside_build_is_reported(self, default_altitude, printed, tmp_path):
        """The build's only write into the operator's own file, named."""
        _repo_with_a_channel_manifest(tmp_path)

        build_cmd._wire_build_derived_env(tmp_path, tmp_path / "build")

        assert (tmp_path / ".env").is_file()
        assert_promoted(default_altitude, printed, "at the generated channel manifest")


class TestSubStepRowsOnTheMiscDeployPath:
    """Rows 22, 36 and 37: a ``  · name`` line under the open phase."""

    def test_the_dev_wheel_build_reports_one_step(self, printed, monkeypatch, tmp_path):
        """Row 22. Under ``--dev`` this is the proof the flag was honored."""
        printed.open_phase("Rendering the configuration")
        _stub_a_successful_wheel_build(monkeypatch)

        assert wheel_build._build_dev_wheel_cached(tmp_path) is not None

        assert_sub_step(printed, "built the osprey wheel from the local checkout")

    def test_seeding_a_profile_context_dir_is_a_step(self, printed, tmp_path):
        """Row 36: the build wrote into the operator's profile, outside build/."""
        printed.open_phase("Rendering the configuration")
        project_path = _rendered_project(tmp_path, "context-seed")
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()

        applied = _apply_conventions(profile_dir, project_path, ["alice"])

        assert applied.seeded_users == ["alice"]
        assert_sub_step(printed, "seeded 1 empty context dir(s) in the profile")

    def test_mirroring_onto_the_project_root_is_a_step(self, printed, tmp_path):
        """Row 37: the build wrote onto the project root, outside build/."""
        printed.open_phase("Rendering the configuration")
        project_path = _rendered_project(tmp_path, "mirror-step")
        profile_dir = tmp_path / "profile"
        (profile_dir / PROJECT_MIRROR_DIR).mkdir(parents=True)
        (profile_dir / PROJECT_MIRROR_DIR / "runbook.md").write_text("# ops\n", encoding="utf-8")

        _apply_conventions(profile_dir, project_path, None)

        assert (project_path / "runbook.md").is_file()
        assert_sub_step(printed, "mirrored 1 file(s) onto the project root")


class TestTheBuildsIdentityLine:
    """Row 24 and the two rows it absorbs, read off one real build."""

    def test_the_three_identity_fields_arrive_as_one_line(self, built_exemplar):
        assert re.search(r"profile .+? \(bundle \S+, tier \d\)", built_exemplar.flowed), (
            built_exemplar.flowed
        )

    def test_the_indented_key_value_spelling_is_gone(self, built_exemplar):
        """Rows 25 and 26: three lines for three fields, retired."""
        for retired in ("Data bundle:", "Tier:"):
            assert retired not in built_exemplar.flowed, retired
            assert not [m for m in built_exemplar.records(logging.INFO) if retired in m], retired

    def test_the_identity_line_lands_before_the_first_phase(self, built_exemplar):
        """It labels the wait, so it cannot arrive after the wait it labels."""
        identity = built_exemplar.flowed.index("profile ")
        first_phase = built_exemplar.flowed.index("→ ")
        assert identity < first_phase, built_exemplar.flowed

    def test_the_localhost_bind_posture_is_reported(self, built_exemplar):
        """Appendix C's ``compose_generator`` row, on the same run.

        A security posture the build chose on the operator's behalf, so it is
        printed rather than logged. The record is kept for the sinks.
        """
        assert "bound to localhost only" in built_exemplar.flowed
        assert any("bound to localhost only" in m for m in built_exemplar.records(logging.INFO))


class TestAbsorbedRowsSayNothingOfTheirOwn:
    """Rows whose fact another line already carries: the call is gone."""

    def test_the_build_no_longer_counts_its_compose_files_at_info(self, built_exemplar):
        """Row 27: the render phase's ``compose files`` step says it instead."""
        assert not [m for m in built_exemplar.records(logging.INFO) if "compose file(s)" in m]
        assert any("compose file(s)" in m for m in built_exemplar.records(logging.DEBUG))
        assert re.search(r"·\s+compose files", built_exemplar.flowed), built_exemplar.flowed

    def test_the_build_no_longer_announces_the_tree_it_rendered(self, built_exemplar):
        """Row 28: the closing card is the "it worked" surface, so this is gone.

        Read off a real build, so the absence is the absence of a line that had
        every chance to fire: this run rendered, and said nothing about where.
        """
        assert not [m for m in built_exemplar.records(logging.INFO) if "✓ Rendered" in m]
        assert "✓ Rendered" not in built_exemplar.flowed

    def test_the_card_header_names_the_build_that_finished(self, tmp_path):
        """Row 28's absorber, asserted on the card rather than on its caller.

        The header names the deployment, not the tree: the operator asked for
        this build by name and reads the card by the same name. The path is in
        neither, which is the point of retiring the line that carried it.
        """
        from osprey.cli.summary_card import format_summary_card

        repo = tmp_path / "beamline-ops"

        assert format_summary_card(repo, "built")[0] == "beamline-ops — built"
        assert str(repo) not in format_summary_card(repo, "built")[0]

    def test_the_demo_seed_hint_is_not_a_line_of_its_own(self, built_exemplar):
        """Row 29: a next step belongs in the card's ``next`` row, once.

        Source-level as well as behavioral, because the hint fired only for a
        profile shipping simulation scenarios: an exemplar without them would
        satisfy the run assertion without the call site ever going away.
        """
        source = Path(build_cmd.__file__).read_text(encoding="utf-8")
        assert "Seed the demo logbook" not in source
        assert not [m for m in built_exemplar.records(logging.INFO) if "sim apply nominal" in m]

    def test_the_service_list_is_no_longer_announced(self, built_exemplar):
        """Row 11: ``osprey status`` is where the list is read."""
        assert "Deployed services:" not in built_exemplar.flowed
        assert not [m for m in built_exemplar.records(logging.DEBUG) if "Deployed services:" in m]

    def test_the_wheel_build_is_not_announced_before_it_runs(self, monkeypatch, tmp_path, caplog):
        """Row 21: announce-then-confirm, for a build memoized to one shot."""
        _stub_a_successful_wheel_build(monkeypatch)

        with caplog.at_level(logging.DEBUG):
            wheel_build._build_dev_wheel_cached(tmp_path)

        assert not [r for r in caplog.records if "Building osprey wheel" in r.getMessage()]

    def test_the_clean_no_longer_announces_itself(self):
        """Row 9: the two steps below it name what they remove as they remove it."""
        source = Path(compose_generator.__file__).read_text(encoding="utf-8")
        assert "Cleaning up deployment" not in source

    def test_the_missing_env_remedy_rides_on_its_warning(self, tmp_path, caplog):
        """Row 12: the remedy is the warning's second half, not a fact of its own."""
        with caplog.at_level(logging.DEBUG):
            assert compose_generator.compose_env_file_args(tmp_path) == []

        remedies = [r for r in caplog.records if ".env.example" in r.getMessage()]
        assert len(remedies) == 1, remedies
        assert remedies[0].levelno == logging.WARNING
        assert "No .env file found" in remedies[0].getMessage()

    def test_activating_scenarios_says_nothing_of_its_own(self):
        """Row 2: the reseed's closing step and ``osprey sim apply`` both say it."""
        source = Path(apply.__file__).read_text(encoding="utf-8")
        assert "Activated scenarios" not in source

    def test_the_archive_rewrite_counts_are_not_logged_twice(self):
        """Row 3: row 1's parenthetical and ``sim apply``'s echo carry them."""
        source = Path(apply.__file__).read_text(encoding="utf-8")
        assert "logger.info(result.describe())" not in source


class TestDemotedRowsOnTheMiscDeployPath:
    """Appendix A's rows for these files, plus row 23."""

    def test_the_per_context_wheel_copy_is_a_debug_line(self, monkeypatch, tmp_path, caplog):
        """Row 23: fires once per build context; as a step it would repeat."""
        _stub_a_successful_wheel_build(monkeypatch)
        out_dir = tmp_path / "context"
        out_dir.mkdir()

        with caplog.at_level(logging.DEBUG):
            assert wheel_build._copy_local_framework_for_override(str(out_dir)) is True

        assert _levels_for(caplog, "Copied osprey wheels") == {logging.DEBUG}

    def test_the_shadowed_artifact_notice_is_a_debug_line(self, tmp_path, caplog):
        """Row 35: one line per overridden artifact, inside the render loop.

        The shadow is real, not simulated: the profile's copy lands on top of a
        framework file that was already there, which is the condition the notice
        reports on.
        """
        project_path = _rendered_project(tmp_path, "shadow-notice")
        framework_rule = project_path / ".claude" / "rules" / "safety.md"
        framework_rule.parent.mkdir(parents=True, exist_ok=True)
        framework_rule.write_text("# shipped\n", encoding="utf-8")
        profile_dir = tmp_path / "profile"
        (profile_dir / "rules").mkdir(parents=True)
        (profile_dir / "rules" / "safety.md").write_text("# facility\n", encoding="utf-8")

        with caplog.at_level(logging.DEBUG):
            _apply_conventions(profile_dir, project_path, None)

        assert framework_rule.read_text(encoding="utf-8") == "# facility\n"
        assert _levels_for(caplog, "profile overrides framework rule") == {logging.DEBUG}

    def test_the_kernel_template_render_is_a_debug_line(self, monkeypatch, tmp_path, caplog):
        """Rows 40 and 41: a per-file path inside a loop, and its announcement."""
        source_dir, out_dir = _a_kernel_template(tmp_path)
        # The render itself is jinja's business; only what it narrates is under
        # test, and a real render needs the loader root the pipeline sets up.
        monkeypatch.setattr(compose_generator, "render_template", lambda *a, **k: None)

        with caplog.at_level(logging.DEBUG):
            compose_generator.render_kernel_templates(str(source_dir), {}, str(out_dir))

        assert _levels_for(caplog, "Rendered kernel template") == {logging.DEBUG}

    def test_the_legacy_clean_narration_is_debug_lines(self, monkeypatch, tmp_path, caplog):
        """Rows 10, 38 and 39: raw argv, and a bare completion with no phase."""
        _stub_the_legacy_clean(monkeypatch, tmp_path)

        with caplog.at_level(logging.DEBUG):
            compose_generator.clean_deployment(
                ["docker-compose.yml"], config={}, repo_root=tmp_path
            )

        assert _levels_for(caplog, "Running: ") == {logging.DEBUG}
        assert _levels_for(caplog, "Cleanup completed") == {logging.DEBUG}


class TestMiscDeployCopyStyle:
    """The copy guard's rules, reapplied where the guard cannot see.

    ``_report_fact`` and ``report_step`` take the finished line, so their call
    sites are invisible to ``tests/cli/test_printed_copy_style.py`` (which
    matches on the printer's own name). These lines are printed at an operator,
    so the same rules hold.
    """

    #: Every module Task 3.4 promoted or stepped a line in.
    _MODULES = (compose_generator, wheel_build, deploy_cmd, build_persistence, build_cmd)

    def _literals(self) -> list[tuple[str, int, str]]:
        found: list[tuple[str, int, str]] = []
        for module in self._MODULES:
            source = Path(module.__file__).read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
                if name not in {"_report_fact", "report_step"}:
                    continue
                for arg in node.args:
                    text = _prose_of(arg)
                    if text:
                        found.append((module.__name__, node.lineno, text))
        return found

    def test_the_walk_reaches_the_promoted_lines(self):
        """A rename that hides these call sites fails here, not silently."""
        literals = self._literals()
        modules = {module for module, _, _ in literals}
        assert len(literals) >= 6, f"only {len(literals)} literals found: {literals}"
        assert len(modules) >= 4, f"only {len(modules)} modules matched: {modules}"

    def test_no_promoted_line_carries_an_em_dash(self):
        offences = [f"{mod}:{line}: {text}" for mod, line, text in self._literals() if "—" in text]
        assert not offences, "Em-dash in a promoted line:\n  " + "\n  ".join(offences)

    def test_no_promoted_line_carries_in_house_jargon(self):
        from tests.cli.test_printed_copy_style import _JARGON

        offences = [
            f"{mod}:{line}: {term!r} -> {plain}"
            for mod, line, text in self._literals()
            for term, plain in _JARGON.items()
            if re.search(rf"\b{re.escape(term)}\b", text.lower())
        ]
        assert not offences, "In-house jargon in a promoted line:\n  " + "\n  ".join(offences)


# ---------------------------------------------------------------------------
# Task 3.4 stubs
# ---------------------------------------------------------------------------

#: The provider whose secret the ``.env`` seed offer harvests from the shell.
_SEED_PROVIDER = "anthropic"


class _BuildRun:
    """One real ``osprey build``: what it printed, and what it logged."""

    def __init__(self, text: str, records: list[logging.LogRecord]) -> None:
        self._text = text
        self._records = records

    @property
    def flowed(self) -> str:
        """Everything printed, whitespace collapsed (the console wraps at 100)."""
        return " ".join(self._text.split())

    def records(self, at_least: int) -> list[str]:
        """The messages of every record emitted at ``at_least`` or above."""
        return [r.getMessage() for r in self._records if r.levelno >= at_least]


@pytest.fixture(scope="module")
def built_exemplar(tmp_path_factory: pytest.TempPathFactory) -> _BuildRun:
    """One build of the exemplar repo, with its output and its records kept.

    Module-scoped because the build is the expensive part and every assertion
    reading it asks about the same run. ``--skip-deps``/``--skip-lifecycle``
    drop the venv install and the profile's shell phases: neither carries a
    disposition row, and both cost minutes.
    """
    from osprey.cli.styles import osprey_theme
    from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

    repo = build_exemplar_repo(tmp_path_factory.mktemp("misc-deploy") / EXEMPLAR_DIRNAME)

    buffer = StringIO()
    console = Console(
        file=buffer, width=100, force_terminal=False, legacy_windows=False, theme=osprey_theme
    )
    sink = _RecordSink()
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(sink)
    root.setLevel(logging.DEBUG)
    previous_cwd = Path.cwd()
    os.chdir(repo)
    previous_reporter = install_reporter(_CaptureReporter(console))
    try:
        result = CliRunner().invoke(build_cmd.build, ["--skip-deps", "--skip-lifecycle"])
    finally:
        install_reporter(previous_reporter)
        os.chdir(previous_cwd)
        root.removeHandler(sink)
        root.setLevel(previous_level)

    assert result.exit_code == 0, result.output
    return _BuildRun(buffer.getvalue(), sink.records)


class _RecordSink(logging.Handler):
    """Every record the build emitted, kept whole so level and text stay paired."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _prose_of(node: ast.expr) -> str:
    """The literal prose in ``node``, or empty for anything that is not prose."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
    return ""


def _offer_the_env_seed(monkeypatch: pytest.MonkeyPatch, *, answer: bool) -> None:
    """Put ``ensure_repo_env`` in the state where it offers to seed a ``.env``."""
    from osprey.build.claude_code_resolver import provider_auth_secret_env

    secret_var = provider_auth_secret_env(_SEED_PROVIDER, None)
    assert secret_var, "the seed offer needs a provider with a secret variable"
    monkeypatch.setenv(secret_var, "sk-from-the-shell")
    monkeypatch.setattr(deploy_cmd, "_stdin_is_a_terminal", lambda: True)
    monkeypatch.setattr(deploy_cmd.click, "confirm", lambda *a, **k: answer)


def _repo_with_a_channel_manifest(repo: Path) -> None:
    """A repo whose build produced the manifest the ``.env`` pointer names."""
    from osprey.services.virtual_accelerator.manifest.build import MANIFEST_FILENAME

    simulation = repo / "build" / "data" / "simulation"
    simulation.mkdir(parents=True)
    (simulation / MANIFEST_FILENAME).write_text('{"channels": []}', encoding="utf-8")


def _stub_a_successful_wheel_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``python -m build`` produce a wheel without running a build.

    The memo is reset on the way in: it is process-wide, so a stubbed wheel
    left in it would be handed to whatever asks next.
    """
    wheel_build._reset_wheel_build_cache()

    def _fake_build(argv, **kwargs):
        outdir = Path(argv[argv.index("--outdir") + 1])
        (outdir / "osprey-0.0.0-py3-none-any.whl").write_bytes(b"")
        return subprocess.CompletedProcess(list(argv), 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_build)
    monkeypatch.setattr(wheel_build, "_wheel_base_requirements", lambda path: [])


def _rendered_project(tmp_path: Path, name: str) -> Path:
    """A project tree the convention application can be run against."""
    from osprey.cli.templates.manager import TemplateManager

    return TemplateManager().create_project(
        project_name=name, output_dir=tmp_path, data_bundle="hello_world"
    )


def _a_kernel_template(tmp_path: Path) -> tuple[Path, Path]:
    """A source tree holding one kernel template, and somewhere to render it."""
    source_dir = tmp_path / "service"
    (source_dir / "kernels").mkdir(parents=True)
    (source_dir / "kernels" / "kernel.json.j2").write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    return source_dir, out_dir


def _stub_the_legacy_clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reduce the *legacy* ``clean_deployment`` to its narration."""
    monkeypatch.setattr(compose_generator, "run_captured", lambda *a, **k: None)
    monkeypatch.setattr(compose_generator, "get_runtime_command", lambda config: "docker")
    monkeypatch.setattr(compose_generator, "runtime_env", lambda config, env: dict(env))
    monkeypatch.setattr(compose_generator, "resolve_repo_root", lambda config: tmp_path)
    # `clean_deployment` imports these two INSIDE the function, so they never
    # enter this module's namespace and patching it here would miss them. The
    # deferred import resolves the attribute off `container_lifecycle` at call
    # time, which is what makes patching them there take effect. Left unpatched,
    # `_compose_provider` probes for a real runtime and raises.
    monkeypatch.setattr(container_lifecycle, "_compose_provider", lambda config: None)
    monkeypatch.setattr(container_lifecycle, "_env_file_args", lambda *a, **k: [])


# ---------------------------------------------------------------------------
# --- Task 3.3: the web_terminals package ---
# ---------------------------------------------------------------------------
#
# Appendix C's rows for ``auth_credentials``, ``env_production``,
# ``postup_hooks`` and ``provision``, plus the disposition table's rows 13-20
# for the same package. Stubs for this section sit in the labelled block at the
# very bottom of the module.
#
# Row 15 (``· web endpoint reachable``) is asserted here through ``printed``,
# and again in ``tests/deployment/test_web_terminals_capture.py`` through the
# reporter's step list -- that one pins the ORDER of the bounce and its payoff,
# which a single-fragment needle cannot state.


class TestWebTerminalSecretsAreReported:
    """The credentials and secret files this deploy created, moved or replaced.

    Names only, in every one of them: a hash identifies a credential
    generation, a signing secret authenticates a session, and neither may
    travel to a log sink or to a terminal. Each test asserts the variable IS
    named and the value is NOT.
    """

    def test_provisioned_credentials_name_the_variables_never_the_hash(
        self, default_altitude, printed, tmp_path
    ):
        result = auth_credentials.ensure_auth_credentials(["alice"], tmp_path, echo=_swallow)

        assert_promoted(default_altitude, printed, "Provisioned web-terminal auth credential(s)")
        assert "OSPREY_AUTH_PW_HASH_ALICE" in printed.flowed
        assert "(gitignored, 0600)" in printed.flowed
        assert _stored_value(result.env_auth_path, "OSPREY_AUTH_PW_HASH_ALICE") not in printed.text

    def test_provisioned_session_secrets_name_the_variables_never_the_values(
        self, default_altitude, printed, tmp_path
    ):
        result = auth_credentials.ensure_auth_session_secrets(tmp_path)

        assert_promoted(default_altitude, printed, "Provisioned web-terminal auth secret(s)")
        for var in auth_credentials.SESSION_SECRET_VARS:
            assert var in printed.flowed
            assert _stored_value(result.env_auth_path, var) not in printed.text

    def test_a_removed_credential_names_the_user_and_the_file(
        self, default_altitude, printed, tmp_path
    ):
        """``osprey users remove`` deletes a login. That is not a detail."""
        auth_credentials.ensure_auth_credentials(["alice"], tmp_path, echo=_swallow)

        assert auth_credentials.purge_auth_credentials("alice", tmp_path) is True

        assert_promoted(default_altitude, printed, "Removed web-terminal auth credential(s)")
        assert "'alice'" in printed.flowed
        assert ".env.auth" in printed.flowed

    def test_a_rotated_password_names_the_variable_never_the_password(
        self, default_altitude, printed, tmp_path
    ):
        """Every live session of that user dies here, so the rotation is said."""
        auth_credentials.ensure_auth_credentials(["alice"], tmp_path, echo=_swallow)

        auth_credentials.set_auth_password("alice", "a-chosen-password", tmp_path)

        assert_promoted(default_altitude, printed, "Replaced web-terminal auth credential")
        assert "OSPREY_AUTH_PW_HASH_ALICE" in printed.flowed
        assert "a-chosen-password" not in printed.text

    def test_a_generated_users_env_names_its_sources_and_its_variables(
        self, default_altitude, printed, tmp_path
    ):
        """A secret file the deploy wrote, at a mode it chose. Names, not values."""
        (tmp_path / ".env").write_text("CBORG_API_KEY=llm-secret\n", encoding="utf-8")

        env_production.ensure_env_production(_LOCAL_WEB_CONFIG, tmp_path)

        assert_promoted(default_altitude, printed, "(mode 0600)")
        assert "CBORG_API_KEY" in printed.flowed
        assert "llm-secret" not in printed.text

    def test_a_renamed_secrets_file_names_both_paths(self, default_altitude, printed, tmp_path):
        """The deploy moved the operator's secrets to a different filename."""
        (tmp_path / env_production.LEGACY_USERS_ENV_FILENAME).write_text("A=1\n", encoding="utf-8")

        assert env_production.migrate_users_env(tmp_path) == tmp_path / ".env.users"

        assert_promoted(default_altitude, printed, ".env.production renamed")
        assert env_production.LEGACY_USERS_ENV_FILENAME in printed.flowed
        assert ".env.users" in printed.flowed

    def test_a_superseded_leftover_secrets_file_says_where_it_went(
        self, default_altitude, printed, tmp_path
    ):
        """The other half of the migration: a leftover secrets file set aside.

        Set aside rather than deleted, so the line names the path the operator
        can go and read -- and the promotion is what tells them a second
        secret-bearing file is now sitting in the repo root.
        """
        (tmp_path / env_production.LEGACY_USERS_ENV_FILENAME).write_text("A=1\n", encoding="utf-8")
        (tmp_path / ".env.users").write_text("B=2\n", encoding="utf-8")

        assert env_production.migrate_users_env(tmp_path) == tmp_path / ".env.users"

        assert_promoted(default_altitude, printed, ".env.production set aside as")
        assert env_production.SUPERSEDED_USERS_ENV_FILENAME in printed.flowed
        assert not (tmp_path / env_production.LEGACY_USERS_ENV_FILENAME).exists()
        assert (tmp_path / env_production.SUPERSEDED_USERS_ENV_FILENAME).is_file()


class TestWebTerminalHostChangesAreReported:
    """What the web-terminal deploy changed on the host, or decided for it."""

    def test_enabling_systemd_linger_is_reported(self, default_altitude, printed, monkeypatch):
        """A logind setting that outlives the deploy, and the operator's login."""
        _stub_linger(monkeypatch, enable_returncode=0)

        postup_hooks.enable_linger({}, {})

        assert_promoted(default_altitude, printed, "Enabled systemd linger for deployuser")
        assert "(podman persistence)" in printed.flowed

    def test_the_host_port_bounce_says_what_it_is_about_to_restart(
        self, default_altitude, printed, monkeypatch, tmp_path
    ):
        """Remediation this deploy performed on its own, with the argv it ran."""
        _stub_the_host_port_bounce(monkeypatch, tmp_path, answers=[False, True])

        postup_hooks.warn_if_web_stack_unreachable(
            {"modules": {"web_terminals": {"nginx_port": 8080}}},
            attempts=1,
            delay=0,
            web_cmd=["docker", "compose", "-f", "web.yml"],
            run_env={},
        )

        assert_promoted(default_altitude, printed, "is not reachable from this host yet")
        assert "docker compose -f web.yml restart" in printed.flowed

    def test_a_bounce_that_did_not_help_still_reaches_the_operator(
        self, default_altitude, printed, monkeypatch, tmp_path, capsys
    ):
        """The regression this class exists for, on the one path that had no test.

        A deploy whose bounce fixed nothing used to end its run with
        ``logger.warning``. The altitude gate drops WARNING while a lifecycle
        reporter owns the terminal, and the root logger carries no other handler,
        so the only text that named the remedy was emitted and then discarded.
        The operator saw ``bounced the web stack ...`` and then the endpoint
        table, which reads as success. Both stubbed probes fail, which is the
        shape of a bounce that changed nothing.
        """
        _stub_the_host_port_bounce(monkeypatch, tmp_path, answers=[False, False])

        postup_hooks.warn_if_web_stack_unreachable(
            {"modules": {"web_terminals": {"nginx_port": 8080}}},
            attempts=1,
            delay=0,
            web_cmd=["docker", "compose", "-f", "web.yml"],
            run_env={},
        )

        promoted = capsys.readouterr().err
        assert_warned(default_altitude, promoted, "not reachable from this host")
        assert "Enable host networking" in " ".join(promoted.split())
        # The bounce's success step is the only other thing that could have
        # spoken, and it must not have: the endpoint is still dark.
        assert "web endpoint reachable" not in printed.flowed

    def test_a_disabled_forwarder_is_named_as_the_cause_and_skips_the_bounce(
        self, default_altitude, monkeypatch, tmp_path, capsys
    ):
        """Docker Desktop was asked and said no, so nothing here is a guess.

        The bounce is skipped because no restart can re-register a port through
        a forwarder that is switched off, and the copy states the cause rather
        than asking the operator to go and check it.
        """
        _stub_the_host_port_bounce(monkeypatch, tmp_path, answers=[False])
        monkeypatch.setattr(postup_hooks, "host_networking_enabled", lambda: False)
        restarts: list[list[str]] = []
        monkeypatch.setattr(
            postup_hooks, "run_captured", lambda cmd, **kwargs: restarts.append(cmd)
        )

        postup_hooks.warn_if_web_stack_unreachable(
            {"modules": {"web_terminals": {"nginx_port": 8080}}},
            attempts=1,
            delay=0,
            web_cmd=["docker", "compose", "-f", "web.yml"],
            run_env={},
        )

        assert restarts == [], f"a bounce that cannot help was attempted anyway: {restarts}"
        promoted = capsys.readouterr().err
        assert_warned(default_altitude, promoted, "Host networking is turned off")
        assert "Enable host networking" in " ".join(promoted.split())

    def test_a_pinned_sidecar_image_reports_the_build_it_skipped(
        self, default_altitude, printed, tmp_path, monkeypatch
    ):
        """The deployment supplies the image, so nothing local was built."""
        monkeypatch.chdir(tmp_path)

        provision.build_auth_sidecar_image(_PINNED_SIDECAR_CONFIG, False, {})

        assert_promoted(default_altitude, printed, "Skipping the local auth-sidecar build")
        assert "ghcr.io/example/auth:2" in printed.flowed


class TestTheUnshownMintWarningStillReachesTheTerminal:
    """The pinned minted-password warning, re-pinned to its promoted form.

    It is promoted through ``warn_fact``: the renderer prints the ``⚠`` block
    (the altitude gate keeps the raw WARNING record off the terminal while a
    reporter is installed), and the record stays for the sinks. The fragments
    are the ones ``tests/deployment/test_up_as_built.py`` pins, asserted here
    against what a default-altitude run keeps in its records.
    """

    def test_the_pinned_fragments_survive_at_the_default_altitude(self, default_altitude, tmp_path):
        result = auth_credentials.ensure_auth_credentials(["alice"], tmp_path, echo=_swallow)

        provision._report_unshown_mints(result)

        painted = " ".join(default_altitude.rendered_text.split())
        for fragment in (
            "Minted a login password",
            "did NOT print it",
            "osprey users passwd",
            "OSPREY_AUTH_PW_<USER>",
        ):
            assert fragment in painted, fragment
            assert any(fragment in message for message in default_altitude.messages), fragment

    def test_the_password_itself_is_in_neither_stream(
        self, default_altitude, printed, tmp_path, capsys
    ):
        """The point of the warning: the plaintext went nowhere, and is gone."""
        shown: list[str] = []
        result = auth_credentials.ensure_auth_credentials(["alice"], tmp_path, echo=shown.append)

        provision._report_unshown_mints(result)

        assert result.minted == ("alice",)
        # With a reporter installed the promoted block is the warning's only
        # terminal form, on the trouble stream; the record is for the sinks.
        promoted = " ".join(capsys.readouterr().err.split())
        assert "alice" in promoted
        assert any("alice" in message for message in default_altitude.messages)
        for line in shown:
            assert line not in promoted
            assert line not in printed.text
            assert line not in default_altitude.rendered_text


class TestWebTerminalSubStepRows:
    """The disposition table's ``sub-step`` rows for this package."""

    def test_a_bounce_that_worked_reports_the_endpoint_as_reachable(
        self, printed, monkeypatch, tmp_path
    ):
        """Row 15: without it the bounce never says whether it worked."""
        _stub_the_host_port_bounce(monkeypatch, tmp_path, answers=[False, True])
        printed.open_phase()

        postup_hooks.warn_if_web_stack_unreachable(
            {"modules": {"web_terminals": {"nginx_port": 8080}}},
            attempts=1,
            delay=0,
            web_cmd=["docker", "compose"],
            run_env={},
        )

        assert_sub_step(printed, "web endpoint reachable")

    def test_the_seed_loop_reports_its_counts(self, printed, monkeypatch, tmp_path):
        """Row 16: counts, not an announcement, now that the loop is quiet."""
        _stub_the_seed_loop(monkeypatch, tmp_path, outcomes={"alice": True, "bob": True})
        printed.open_phase()

        seeding.seed_user_containers(_seed_config(["alice", "bob"]))

        assert_sub_step(printed, "seeded 2/2 user contexts")

    def test_a_short_seed_is_legible_on_the_count_line_alone(self, printed, monkeypatch, tmp_path):
        """Row 16's whole reason for being a count: ``<n>`` seeded of ``<total>``."""
        _stub_the_seed_loop(monkeypatch, tmp_path, outcomes={"alice": True, "bob": None})
        printed.open_phase()

        seeding.seed_user_containers(_seed_config(["alice", "bob"]))

        assert_sub_step(printed, "seeded 1/2 user contexts")

    def test_a_container_that_was_not_ready_names_the_user_it_cost(
        self, printed, monkeypatch, tmp_path, caplog
    ):
        """Rows 16 and 17's joint condition: the count alone would not say WHO.

        The not-ready branch is excluded from the systemic check by design, so
        without this warning a demoted per-user line would hide a real
        degradation behind a number.
        """
        _stub_the_seed_loop(monkeypatch, tmp_path, outcomes={"alice": True, "bob": None})
        printed.open_phase()

        with caplog.at_level(logging.WARNING):
            seeding.seed_user_containers(_seed_config(["alice", "bob"]))

        assert _levels_for(caplog, "No container was running") == {logging.WARNING}
        assert "bob" in caplog.text
        assert "alice" not in caplog.text

    def test_a_fully_ready_roster_warns_about_nobody(self, printed, monkeypatch, tmp_path, caplog):
        """The warning is the exception, not a line every deploy carries."""
        _stub_the_seed_loop(monkeypatch, tmp_path, outcomes={"alice": True, "bob": True})
        printed.open_phase()

        with caplog.at_level(logging.WARNING):
            seeding.seed_user_containers(_seed_config(["alice", "bob"]))

        assert "No container was running" not in caplog.text


class TestWebTerminalAbsorbedRowsSayNothingOfTheirOwn:
    """Rows 13, 14, 19 and 20: an announcement the step line already carries."""

    def test_the_smoke_check_is_not_announced_before_it_runs(self, tmp_path, caplog):
        """Rows 13 and 14, on a real script: the step reports it WITH its exit."""
        _write_a_verify_script(tmp_path)

        with caplog.at_level(logging.DEBUG):
            postup_hooks.run_verify_script(str(tmp_path), {})

        assert not [r for r in caplog.records if "Running post-up smoke check" in r.getMessage()]
        assert not [r for r in caplog.records if "completed (exit 0)" in r.getMessage()]

    def test_the_smoke_checks_step_still_carries_the_script_and_its_exit(self, printed, tmp_path):
        """Rows 13 and 14's absorber, on the same code path."""
        _write_a_verify_script(tmp_path)
        printed.open_phase()

        postup_hooks.run_verify_script(str(tmp_path), {})

        assert_sub_step(printed, "smoke check verify.sh: exit 0")

    def test_the_persona_image_build_is_not_announced(self):
        """Row 19: the live build region carries the progress in between."""
        source = Path(persona_images.__file__).read_text(encoding="utf-8")
        assert "Building persona image" not in source

    def test_the_auth_sidecar_build_is_not_announced(self, monkeypatch, tmp_path, caplog):
        """Row 20, source-level AND on a real build: the step says it instead."""
        source = Path(provision.__file__).read_text(encoding="utf-8")
        assert "Building auth sidecar image" not in source

        _stub_the_sidecar_build(monkeypatch, tmp_path)
        with caplog.at_level(logging.DEBUG):
            provision.build_auth_sidecar_image(_LOCAL_SIDECAR_CONFIG, False, {})

        assert not [r for r in caplog.records if "auth sidecar image" in r.getMessage()]


class TestWebTerminalDemotedRows:
    """Rows 17 and 18: per-user chatter, emitted below the default view."""

    def test_a_not_ready_container_is_a_debug_line(self, monkeypatch, tmp_path, caplog):
        """Row 17. Safe only because row 16's count and its warning cover it."""
        monkeypatch.setattr(seeding, "_container_exists", lambda *a, **k: False)

        with caplog.at_level(logging.DEBUG):
            outcome = seeding._seed_one_user(
                "docker", "alice", "dls", "", "/skills", tmp_path, env=None
            )

        assert outcome is None
        assert _levels_for(caplog, "(skipped alice: container not ready)") == {logging.DEBUG}

    def test_a_seeded_user_is_a_debug_line(self, monkeypatch, tmp_path, caplog):
        """Row 18: one line per user; row 16's step carries the count."""
        _stub_one_users_seed(monkeypatch)

        with caplog.at_level(logging.DEBUG):
            outcome = seeding._seed_one_user(
                "docker", "alice", "dls", "", "/skills", tmp_path, env=None
            )

        assert outcome is True
        assert _levels_for(caplog, "seeded alice") == {logging.DEBUG}


class TestWebTerminalCopyStyle:
    """The copy guard's rules, reapplied where the guard cannot see.

    ``report_fact`` and ``report_step`` take the finished line, so their call
    sites are invisible to ``tests/cli/test_printed_copy_style.py`` (which
    matches on the printer's own name). These lines are printed at an operator,
    so the same rules hold.
    """

    #: Every module Task 3.3 promoted or stepped a line in.
    _MODULES = (
        auth_credentials,
        env_production,
        persona_images,
        postup_hooks,
        provision,
        seeding,
    )

    def _literals(self) -> list[tuple[str, int, str]]:
        found: list[tuple[str, int, str]] = []
        for module in self._MODULES:
            source = Path(module.__file__).read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
                if name not in {"report_fact", "_report_step"}:
                    continue
                for arg in node.args:
                    text = _prose_of(arg)
                    if text:
                        found.append((module.__name__, node.lineno, text))
        return found

    def test_the_walk_reaches_the_promoted_lines(self):
        """A rename that hides these call sites fails here, not silently."""
        literals = self._literals()
        modules = {module for module, _, _ in literals}
        assert len(literals) >= 12, f"only {len(literals)} literals found: {literals}"
        assert len(modules) >= 5, f"only {len(modules)} modules matched: {modules}"

    def test_no_promoted_line_carries_an_em_dash(self):
        offences = [f"{mod}:{line}: {text}" for mod, line, text in self._literals() if "—" in text]
        assert not offences, "Em-dash in a promoted line:\n  " + "\n  ".join(offences)

    def test_no_promoted_line_carries_in_house_jargon(self):
        from tests.cli.test_printed_copy_style import _JARGON

        offences = [
            f"{mod}:{line}: {term!r} -> {plain}"
            for mod, line, text in self._literals()
            for term, plain in _JARGON.items()
            if re.search(rf"\b{re.escape(term)}\b", text.lower())
        ]
        assert not offences, "In-house jargon in a promoted line:\n  " + "\n  ".join(offences)


# ---------------------------------------------------------------------------
# Task 3.3 stubs
# ---------------------------------------------------------------------------


def _swallow(_line: str) -> None:
    """The mint sink on a non-terminal stdout: a password goes nowhere."""


def _stored_value(env_auth_path: Path, var: str) -> str:
    """The value ``.env.auth`` actually holds for ``var``.

    Read back off the file rather than fabricated, so the "never printed"
    assertions are about the real secret this run generated.
    """
    from osprey.utils.dotenv import parse_dotenv_file

    value = parse_dotenv_file(env_auth_path).get(var, "")
    assert value, f"nothing was stored for {var}"
    return value


#: A local-mode web-terminal deployment, the only mode ``.env.users`` generates in.
_LOCAL_WEB_CONFIG = {
    "facility": {"name": "Demo", "prefix": "dls", "timezone": "UTC"},
    "llm": {"provider": "cborg", "api_key_env_var": "CBORG_API_KEY"},
    "modules": {"web_terminals": {"enabled": True, "image_source": "local"}},
}

#: Local mode with ``auth.image`` pinned: nothing to build, and that is the fact.
_PINNED_SIDECAR_CONFIG = {
    "project_name": "demo",
    "modules": {
        "web_terminals": {
            "image_source": "local",
            "auth": {"method": "password", "image": "ghcr.io/example/auth:2"},
        }
    },
}

#: Local mode with nothing pinned: the build runs, and reports one step.
_LOCAL_SIDECAR_CONFIG = {
    "project_name": "demo",
    "modules": {"web_terminals": {"image_source": "local", "auth": {"method": "password"}}},
}


def _stub_linger(monkeypatch: pytest.MonkeyPatch, *, enable_returncode: int) -> None:
    """Reduce ``enable_linger`` to its narration: podman, ``loginctl`` present."""
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(postup_hooks.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(postup_hooks.getpass, "getuser", lambda: "deployuser")

    def _run(argv, **kwargs):
        # `show-user` reports linger off, so the enable below actually runs.
        returncode = 0 if argv[1] == "show-user" else enable_returncode
        return SimpleNamespace(returncode=returncode, stdout="Linger=no", stderr="")

    monkeypatch.setattr(postup_hooks.subprocess, "run", _run)


def _stub_the_host_port_bounce(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, answers: list[bool]
) -> None:
    """A Docker Desktop host whose port answers only after the bounce."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(postup_hooks, "on_docker_desktop", lambda config: True)
    # None is "could not read the setting", which is the state that keeps the
    # self-heal bounce these tests are about. A definite False would skip it.
    monkeypatch.setattr(postup_hooks, "host_networking_enabled", lambda: None)
    replies = iter(answers)
    monkeypatch.setattr(
        postup_hooks, "_host_port_answers", lambda url, attempts, delay: next(replies)
    )
    monkeypatch.setattr(postup_hooks, "run_captured", lambda *a, **k: None)


def _stub_the_sidecar_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reduce the sidecar image build to its narration."""
    monkeypatch.chdir(tmp_path)
    context = tmp_path / "build" / "auth"
    context.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker"])
    monkeypatch.setattr(
        provision, "_materialize_auth_build_context", lambda repo_root, dev_mode: context
    )
    monkeypatch.setattr(provision, "run_captured", lambda *a, **k: None)


def _write_a_verify_script(project_root: Path) -> None:
    """The scaffolded post-up smoke check, exiting 0 as its convention says."""
    scripts = project_root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "verify.sh").write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")


def _seed_config(users: list[str]) -> dict:
    """A roster the seed loop can resolve personas for."""
    return {
        "project_name": "demo",
        "facility": {"name": "Demo", "prefix": "dls", "timezone": "UTC"},
        "modules": {"web_terminals": {"enabled": True, "users": users}},
    }


def _stub_the_seed_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, outcomes: dict[str, bool | None]
) -> None:
    """Drive ``seed_user_containers``' loop by per-user outcome.

    ``_seed_one_user`` is the seam: ``True`` seeded, ``False`` ready but failed,
    ``None`` container not ready. Stubbing it rather than the runtime keeps the
    counts under the test's control, which is what rows 16 and 17 are about.
    """
    monkeypatch.chdir(tmp_path)
    context = tmp_path / "build" / "docker" / "web-terminal-context"
    context.mkdir(parents=True, exist_ok=True)
    (context / "base.md").write_text("# base\n", encoding="utf-8")
    monkeypatch.setattr(seeding, "get_runtime_command", lambda config=None: ["docker", "compose"])
    monkeypatch.setattr(seeding, "runtime_env", lambda config, **kw: {})
    monkeypatch.setattr(seeding, "_seed_one_user", lambda runtime, user, *a, **k: outcomes[user])


def _stub_one_users_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reduce ``_seed_one_user`` to its narration, for a container that is up."""
    monkeypatch.setattr(seeding, "_container_exists", lambda *a, **k: True)
    monkeypatch.setattr(seeding, "_container_seed_owner", lambda *a, **k: "1000:1000")
    monkeypatch.setattr(seeding, "_resolve_extra_md", lambda user, context_dir: "")
    monkeypatch.setattr(seeding, "_seed_claude_md", lambda *a, **k: None)
    monkeypatch.setattr(seeding, "_seed_skills", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# --- Task 3.7: the last two report-through-logger sites ---
# ---------------------------------------------------------------------------
#
# Both were reports the logger carried because there was nowhere else to put
# them. Each now has a renderer shape of its own -- the knob diff is a block of
# facts about one thing (``section``), the port report is a refusal (``fail``)
# -- and each keeps the record it always had.
#
# The port report needs a second instrument. ``fail`` always writes to stderr,
# which is ``styles.err_console`` and NOT the reporter's console, so
# ``printed`` cannot see it: ``printed_err`` below is the seam for that half.


class TestTheArchiveKnobDiff:
    """The diff an operator reads before a rebuild discards their history."""

    def test_the_changed_knobs_reach_the_default_view(
        self, default_altitude, printed, archiver_stubs, tmp_path
    ):
        archiver_stubs["state"] = _seed_state("MISMATCH")

        container_lifecycle._stage_archiver_store(
            {"deployed_services": ["archiver_store"]}, ["compose.yml"], {}, tmp_path
        )

        assert_promoted(
            default_altitude, printed, "The archive's knobs changed since it was seeded"
        )
        assert "retention_days 5 -> 7" in printed.flowed

    def test_the_diff_is_a_block_of_rows_under_its_heading(
        self, default_altitude, printed, archiver_stubs, tmp_path
    ):
        """One row per knob that moved, indented under the heading -- the shape
        every other block of facts the CLI prints uses."""
        archiver_stubs["state"] = _seed_state("MISMATCH")

        container_lifecycle._stage_archiver_store(
            {"deployed_services": ["archiver_store"]}, ["compose.yml"], {}, tmp_path
        )

        lines = [line.rstrip() for line in printed.text.splitlines() if line.strip()]
        heading = lines.index("The archive's knobs changed since it was seeded:")
        assert lines[heading + 1].startswith("  retention_days")
        assert lines[heading + 1].endswith("5 -> 7")

    def test_the_old_logged_diff_stays_out_of_the_terminal(
        self, default_altitude, printed, archiver_stubs, tmp_path
    ):
        """`describe()`'s own spelling carries a colon the printed rows do not,
        so it is the fragment that tells the two forms apart."""
        archiver_stubs["state"] = _seed_state("MISMATCH")

        container_lifecycle._stage_archiver_store(
            {"deployed_services": ["archiver_store"]}, ["compose.yml"], {}, tmp_path
        )

        assert "retention_days: 5 -> 7" not in default_altitude.rendered_text
        assert any("retention_days: 5 -> 7" in message for message in default_altitude.messages), (
            "the log sinks lost the diff"
        )
        logging.getLogger(_WITNESS_LOGGER).error(_WITNESS)
        assert _WITNESS in default_altitude.rendered_text, "the probe console is not armed"

    def test_a_matching_fingerprint_prints_no_diff(
        self, default_altitude, printed, archiver_stubs, tmp_path
    ):
        """Nothing moved, so there is nothing to say about what moved."""
        archiver_stubs["state"] = _seed_state("MATCH")

        container_lifecycle._stage_archiver_store(
            {"deployed_services": ["archiver_store"]}, ["compose.yml"], {}, tmp_path
        )

        assert "knobs changed" not in printed.flowed


@pytest.fixture
def printed_err(monkeypatch: pytest.MonkeyPatch) -> Iterator[Printed]:
    """Capture what the trouble primitives put on stderr.

    ``output.fail`` resolves ``styles.err_console`` per call, so standing a
    buffered console in its place is the whole instrument. Reuses ``Printed``
    for its ``flowed`` reader: a failure's cause lines wrap exactly as a report
    line does.
    """
    from osprey.cli import styles
    from osprey.cli.styles import osprey_theme

    buffer = StringIO()
    console = Console(
        file=buffer,
        width=100,
        force_terminal=False,
        legacy_windows=False,
        theme=osprey_theme,
    )
    monkeypatch.setattr(styles, "err_console", console)
    yield Printed(_CaptureReporter(console), buffer)


def _stub_one_port_conflict(monkeypatch: pytest.MonkeyPatch, *, host_network: bool = False) -> None:
    """Hand the preflight one external conflict to report."""
    from osprey.deployment.host_ports import PortConflict

    conflict = PortConflict(
        host_port=8080,
        bind_address="127.0.0.1",
        service="web-terminal",
        kind="external",
        holder="another process",
        remedy="modules.web_terminals.nginx_port",
        host_network=host_network,
    )
    monkeypatch.setattr(container_lifecycle, "parse_host_port_bindings", lambda files: [])
    monkeypatch.setattr(container_lifecycle, "resolve_project_name", lambda config: "demo")
    monkeypatch.setattr(container_lifecycle, "find_port_conflicts", lambda *a, **k: [conflict])


class TestTheHostPortConflictReport:
    """The preflight's refusal, in the shape every other refusal wears."""

    def test_the_report_lands_on_stderr_as_a_failure(
        self, default_altitude, printed_err, monkeypatch
    ):
        _stub_one_port_conflict(monkeypatch)

        with pytest.raises(RuntimeError):
            container_lifecycle._preflight_host_ports({}, ["compose.yml"])

        assert "✗ Host port preflight found 1 conflict" in printed_err.flowed
        assert "port 8080 (127.0.0.1)" in printed_err.flowed
        assert "modules.web_terminals.nginx_port" in printed_err.flowed

    def test_the_remedy_is_the_line_that_says_what_to_do(
        self, default_altitude, printed_err, monkeypatch
    ):
        """The report's closing hint becomes the ``→`` line, which is what makes
        a remedy findable in a scrollback full of causes."""
        _stub_one_port_conflict(monkeypatch)

        with pytest.raises(RuntimeError):
            container_lifecycle._preflight_host_ports({}, ["compose.yml"])

        assert "→ Change the listed config keys to free ports before deploying." in (
            printed_err.flowed
        )

    def test_the_report_is_no_longer_painted_by_the_log_handler(
        self, default_altitude, printed_err, monkeypatch
    ):
        _stub_one_port_conflict(monkeypatch)

        with pytest.raises(RuntimeError):
            container_lifecycle._preflight_host_ports({}, ["compose.yml"])

        assert "Host port preflight found" not in default_altitude.rendered_text
        logging.getLogger(_WITNESS_LOGGER).warning(_WITNESS)
        assert _WITNESS in default_altitude.rendered_text, "the probe console is not armed"

    def test_the_record_is_still_the_whole_report(self, default_altitude, printed_err, monkeypatch):
        """A log file reads exactly what it read before: the report as one
        record, spacer lines and all."""
        _stub_one_port_conflict(monkeypatch)

        with pytest.raises(RuntimeError):
            container_lifecycle._preflight_host_ports({}, ["compose.yml"])

        assert any(
            "Host port preflight found 1 conflict" in message and "port 8080" in message
            for message in default_altitude.messages
        ), "the log sinks lost the report"

    def test_the_host_network_note_survives_as_a_cause_line(
        self, default_altitude, printed_err, monkeypatch
    ):
        """The report's middle paragraphs are the failure's cause, so a
        multi-paragraph report keeps every paragraph."""
        _stub_one_port_conflict(monkeypatch, host_network=True)

        with pytest.raises(RuntimeError):
            container_lifecycle._preflight_host_ports({}, ["compose.yml"])

        assert "bind these ports directly" in printed_err.flowed

    def test_the_abort_is_unchanged(self, default_altitude, printed_err, monkeypatch):
        """`fail` only prints. The caller's own RuntimeError is still what stops
        the deploy, and it still carries the sentence it always carried."""
        _stub_one_port_conflict(monkeypatch)

        with pytest.raises(RuntimeError) as raised:
            container_lifecycle._preflight_host_ports({}, ["compose.yml"])

        assert "host port preflight failed: 1 conflict" in str(raised.value)

    def test_no_conflicts_reports_nothing_and_returns(
        self, default_altitude, printed_err, monkeypatch
    ):
        monkeypatch.setattr(container_lifecycle, "parse_host_port_bindings", lambda files: [])
        monkeypatch.setattr(container_lifecycle, "resolve_project_name", lambda config: "demo")
        monkeypatch.setattr(container_lifecycle, "find_port_conflicts", lambda *a, **k: [])

        container_lifecycle._preflight_host_ports({}, ["compose.yml"])

        assert printed_err.text == ""
