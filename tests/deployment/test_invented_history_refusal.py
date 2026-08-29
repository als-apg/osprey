"""Deploy-time refusal of a stack whose archive would be fabricated.

``osprey build`` refuses to write the pairing and the MCP server refuses to
start on it, but a deploy reads the project's ``config.yml`` as it stands now —
including one edited after the build. It is also the moment the pairing stops
being a file and becomes a running stack other people trust, which is why
``_refuse_invented_history`` runs before any branch of ``deploy_up`` has touched
the host, and again on the restart path (``config.yml`` is bind-mounted, so a
restart is how an edit takes effect).

The rule itself — which spelling is live in which kind of config, and why an
unset ``archiver.type`` counts as the mock — is proven in
``tests/connectors/test_honesty_rule.py``. Asserted here: that this surface asks
it, that it raises rather than warns, and that what it says is actionable by
whoever ran the deploy.
"""

import pytest

from osprey.deployment.container_lifecycle import _refuse_invented_history

VA_MOCK_NESTED = {
    "control_system": {"type": "virtual_accelerator", "writes_enabled": True},
    "archiver": {"type": "mock_archiver"},
    "deployed_services": ["virtual_accelerator", "bluesky"],
}
VA_ARCHIVER_UNSET = {"control_system": {"type": "virtual_accelerator"}}

# Top-level dotted lines a rendered config.yml's readers never honour — see the
# bypass regressions in tests/connectors/test_honesty_rule.py. Each of these
# deploys a VA whose live archiver is the mock, behind a line that looks like it
# said otherwise.
INERT_FLAT_ARCHIVER = VA_MOCK_NESTED | {"archiver.type": "mongodb_archiver"}
INERT_FLAT_ARCHIVER_ONLY = VA_ARCHIVER_UNSET | {"archiver.type": "mongodb_archiver"}
INERT_FLAT_CONTROL_SYSTEM = VA_MOCK_NESTED | {"control_system.type": "mock"}


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(VA_MOCK_NESTED, id="nested-mock"),
        pytest.param(VA_ARCHIVER_UNSET, id="archiver-unset"),
        pytest.param(INERT_FLAT_ARCHIVER, id="inert-flat-archiver-over-mock"),
        pytest.param(INERT_FLAT_ARCHIVER_ONLY, id="inert-flat-archiver-only"),
        pytest.param(INERT_FLAT_CONTROL_SYSTEM, id="inert-flat-control-system"),
    ],
)
def test_a_deploy_that_would_fabricate_its_past_is_refused(config):
    with pytest.raises(RuntimeError) as excinfo:
        _refuse_invented_history(config)

    assert "virtual_accelerator" in str(excinfo.value)


def test_the_refusal_names_the_file_to_fix_and_the_ways_out():
    """The reader is at a terminal that just declined to deploy; the message has
    to be the whole diagnosis."""
    with pytest.raises(RuntimeError) as excinfo:
        _refuse_invented_history(VA_MOCK_NESTED)
    message = str(excinfo.value)

    assert "config.yml" in message
    assert "va_archiver" in message
    assert "'mock'" in message


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            {
                "control_system": {"type": "virtual_accelerator"},
                "archiver": {"type": "mongodb_archiver"},
                "deployed_services": ["virtual_accelerator", "mongodb", "archiver-recorder"],
            },
            id="va-with-its-store",
        ),
        pytest.param(
            {"control_system": {"type": "mock"}, "archiver": {"type": "mock_archiver"}},
            id="honestly-storeless",
        ),
        pytest.param(
            {"control_system": {"type": "epics"}, "archiver": {"type": "epics_archiver"}},
            id="hardware",
        ),
        pytest.param(
            {"control_system.type": "virtual_accelerator", "archiver": {"type": "mock_archiver"}},
            id="inert-flat-control-system-only",
        ),
        pytest.param({}, id="says-nothing"),
    ],
)
def test_every_other_deploy_proceeds_untouched(config):
    """The last case is the mirror of the rule rather than a hole: with no
    `control_system:` section the connector factory falls back to the mock, so
    that deployment is a mock machine with a mock archive — honest, whatever the
    inert top-level line was aiming at."""
    assert _refuse_invented_history(config) is None


# ---------------------------------------------------------------------------
# The message names the machine it refused
#
# The pairing covers every type with no recorded past, so a deployment
# baselined on the live stand-in reaches this refusal too — and the operator
# reading it has no virtual accelerator to go looking for. A message naming one
# would send them to a `control_system:` section that says something else.
# ---------------------------------------------------------------------------

STANDIN_MOCK_NESTED = {
    "control_system": {"type": "live_standin", "writes_enabled": True},
    # The stand-in is a second container of the virtual-accelerator service,
    # so this is what a deployment standing one up actually lists.
    "deployed_services": ["virtual_accelerator", "bluesky"],
    "archiver": {"type": "mock_archiver"},
}
STANDIN_ARCHIVER_UNSET = {"control_system": {"type": "live_standin"}}


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(STANDIN_MOCK_NESTED, id="standin-nested-mock"),
        pytest.param(STANDIN_ARCHIVER_UNSET, id="standin-archiver-unset"),
    ],
)
def test_a_stand_in_is_refused_under_its_own_name(config):
    with pytest.raises(RuntimeError) as excinfo:
        _refuse_invented_history(config)
    message = str(excinfo.value)

    assert "'live_standin'" in message
    # Not the machine this deployment does not run. The shared reason sentence
    # names both kinds in prose; what must not appear is the *type* the
    # operator would then go hunting for in `control_system:`.
    assert "'virtual_accelerator'" not in message


def test_the_virtual_accelerator_is_still_named_when_it_is_the_one_refused():
    """The type comes from the config rather than from a literal, so the case
    that motivated the rule keeps the message it had."""
    with pytest.raises(RuntimeError) as excinfo:
        _refuse_invented_history(VA_MOCK_NESTED)
    message = str(excinfo.value)

    assert "'virtual_accelerator'" in message
    assert "'live_standin'" not in message


def test_an_attached_project_deploying_no_va_container_is_still_refused():
    """Keyed on what the deployment claims about itself, not on
    ``deployed_services``: a project that deploys no VA of its own still points
    its agent at one."""
    with pytest.raises(RuntimeError):
        _refuse_invented_history(
            {
                "control_system": {"type": "virtual_accelerator"},
                "archiver": {"type": "mock_archiver"},
                "deployed_services": [],
            }
        )


# ---------------------------------------------------------------------------
# Wired into the paths that start containers, before either touches the host
# ---------------------------------------------------------------------------


@pytest.fixture
def lying_project(monkeypatch, tmp_path):
    """A project whose config.yml pairs a VA with the mock, and a runtime that
    records any attempt to reach it."""
    from osprey.deployment import container_lifecycle

    reached: list[list[str]] = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (dict(VA_MOCK_NESTED), ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "run",
        lambda cmd, env=None, check=False: reached.append(cmd),
    )
    return container_lifecycle, reached, tmp_path


def test_deploy_up_aborts_before_the_host_is_touched(lying_project):
    container_lifecycle, reached, tmp_path = lying_project

    with pytest.raises(RuntimeError, match="virtual_accelerator"):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    assert reached == []


def test_deploy_restart_aborts_too(lying_project):
    """config.yml is bind-mounted, so a restart is how an edit into the pairing
    would take effect."""
    container_lifecycle, reached, tmp_path = lying_project

    with pytest.raises(RuntimeError, match="virtual_accelerator"):
        container_lifecycle.deploy_restart(str(tmp_path / "config.yml"))

    assert reached == []
