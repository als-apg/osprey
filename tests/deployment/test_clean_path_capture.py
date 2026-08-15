"""The clean/rebuild path spools its compose output instead of the terminal.

``clean_deployment`` runs two compose invocations back to back. Both used to
inherit this process's stdio, so a rebuild buried the phase lines under compose's
own progress stream. They now go through :func:`run_captured`, and each announces
itself as a step on the open phase.

Two properties matter beyond the routing itself and are asserted per site:

* ``check=False`` — a clean over a deployment that was never up exits non-zero,
  and that has always been tolerated. Capturing must not promote it to a
  failure.
* ``repo_root=`` — the spool lands under the deployment repo's ``var/logs/``,
  not under whatever directory the operator happened to stand in.
"""

import logging

import pytest

from osprey.cli.phase_reporter import PhaseReporter, install_reporter
from osprey.deployment import compose_generator


class _RecordingPhase:
    """A stand-in for :class:`osprey.cli.phase_reporter.Phase` that records steps."""

    def __init__(self):
        self.steps = []

    def step(self, name):
        self.steps.append(name)


class _RecordingReporter(PhaseReporter):
    """A reporter whose only job is to hand ``clean_deployment`` an open phase.

    Subclasses the real reporter rather than duck-typing it, and must keep doing
    so: the reporter's terminal-ownership hooks (``stop_rendering`` today, and
    the no-ops the deploy call sites invoke unconditionally) are defined on the
    base class, so a standalone fake breaks on each one as it arrives.

    ``current_phase`` is a read-only property on the base, hence ``_phase``.
    """

    def __init__(self, phase):
        super().__init__(color=False)
        self._phase = phase


@pytest.fixture(autouse=True)
def stub_runtime(monkeypatch):
    """Answer the runtime probe from a fixture instead of the host's daemon.

    ``clean_deployment`` builds both compose commands from
    :func:`get_runtime_command`, which detects a *running* runtime and raises
    when it finds none. Nothing below asserts on the detection — the argv checks
    read the subcommand tail — so on a machine with no daemon these tests would
    fail for the host's state rather than for the routing under test.
    """
    monkeypatch.setattr(
        compose_generator, "get_runtime_command", lambda config=None: ["docker", "compose"]
    )


@pytest.fixture
def captured_runs(monkeypatch):
    """Replace ``run_captured`` in compose_generator's namespace, recording calls."""
    calls = []

    def _fake_run_captured(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        return None

    monkeypatch.setattr(compose_generator, "run_captured", _fake_run_captured)
    return calls


@pytest.fixture
def phase():
    """Install a recording phase as the reporter's open phase."""
    recording = _RecordingPhase()
    previous = install_reporter(_RecordingReporter(recording))
    yield recording
    install_reporter(previous)


@pytest.fixture
def clean_run(captured_runs, phase, tmp_path):
    """Run ``clean_deployment`` pinned to ``tmp_path`` as the deployment repo."""
    compose_generator.clean_deployment(
        ["docker-compose.yml"], {"project_name": "myproj", "project_root": str(tmp_path)}
    )
    return captured_runs


def test_clean_runs_exactly_two_captured_commands(clean_run):
    assert len(clean_run) == 2


def test_down_site_argv_ends_with_the_down_subcommand(clean_run):
    cmd, _ = clean_run[0]
    assert cmd[-3:] == ["down", "--volumes", "--remove-orphans"]


def test_rmi_site_argv_ends_with_the_rmi_subcommand(clean_run):
    cmd, _ = clean_run[1]
    assert cmd[-3:] == ["down", "--rmi", "all"]


def test_each_site_spools_under_its_own_descriptive_name(clean_run):
    names = [kwargs["spool_name"] for _, kwargs in clean_run]
    assert names == ["compose-clean-down", "compose-clean-rmi"]


def test_each_site_pins_the_spool_to_the_deployment_repo(clean_run, tmp_path):
    for _, kwargs in clean_run:
        assert kwargs["repo_root"] == tmp_path.absolute()


def test_neither_site_promotes_a_tolerated_non_zero_exit_to_a_failure(clean_run):
    """Both invocations tolerated non-zero before capture; they still must."""
    assert [kwargs["check"] for _, kwargs in clean_run] == [False, False]


def test_each_site_still_receives_the_pinned_compose_environment(clean_run):
    for _, kwargs in clean_run:
        assert kwargs["env"]["COMPOSE_PROJECT_NAME"] == "myproj"


def test_each_site_announces_itself_as_a_step_on_the_open_phase(clean_run, phase):
    assert len(phase.steps) == 2
    assert phase.steps[0] != phase.steps[1]


def test_clean_runs_when_no_phase_is_open(captured_runs, tmp_path):
    """No open phase is the normal case for a library caller — not an error."""
    previous = install_reporter(_RecordingReporter(None))
    try:
        compose_generator.clean_deployment(["docker-compose.yml"], {"project_root": str(tmp_path)})
    finally:
        install_reporter(previous)

    assert len(captured_runs) == 2


def test_production_mode_line_is_not_emitted_at_info(tmp_path, caplog):
    """It fires once per service build context on every production build."""
    with caplog.at_level(logging.INFO):
        compose_generator._stage_dev_wheel_for_context(str(tmp_path), dev_mode=False)

    assert "Production mode" not in caplog.text


def test_production_mode_line_is_still_available_at_debug(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG):
        compose_generator._stage_dev_wheel_for_context(str(tmp_path), dev_mode=False)

    assert "Production mode: Containers will install osprey from PyPI" in caplog.text
