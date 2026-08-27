"""Render-level tests for the bluesky service's compose template.

The queue backend is a four-container arrangement (bridge, RE Manager, Redis,
optionally Tiled) whose safety properties live entirely in how they are wired:
Redis reachable only from an internal network, no published port on either new
service, CurveZMQ keys on both 0MQ planes, and the limits DB present wherever
the reference monitor runs. None of that can be asserted by reading Python —
it is a property of the *rendered* compose file, so these tests render the
packaged template with production-shaped contexts and assert against the
parsed YAML.

Rendering the packaged template (not a built project's copy) is deliberate
here: this is the artifact `_inject_bluesky` copies, so it is the one whose
contract the rest of the feature depends on.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

# Rooted at the templates/ PROJECT root, not services/, because service
# templates import the shared axis macros as "services/_*.j2" — the spelling
# compose_generator's own loader resolves. Template names below stay relative
# to services/ via the second search path.
_TEMPLATES_ROOT = Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates"
TEMPLATE_DIR = _TEMPLATES_ROOT / "services"
_LOADER_ROOTS = [str(_TEMPLATES_ROOT), str(TEMPLATE_DIR)]
BLUESKY_TEMPLATE = "bluesky/docker-compose.yml.j2"


def _image_defaults(project_name: str) -> dict[str, str]:
    """The image map ``_inject_project_metadata`` injects, for hand-built ctx.

    The bridge's image line renders its innermost fallback from this mapping,
    so a context assembled by hand still has to carry it. Taken from the
    production helper rather than restated, so these renders follow the
    registry and tag axes instead of pinning a name the generator may not
    produce any more.
    """
    from osprey.deployment.compose_generator import resolve_image_defaults

    return resolve_image_defaults({"project_name": project_name})


#: The finished channel-limits bind mount a deployment whose config is read
#: from the REPO ROOT renders. Both halves are computed host-side by
#: ``resolve_limits_mount`` and reach the template as finished strings, so a
#: hand-built context has to carry the same shape the generator produces —
#: source anchored on the compose project directory, target on the container's
#: project root. Restated here rather than imported because what these renders
#: pin is what the TEMPLATE does with the strings; where the strings come from
#: is pinned in ``tests/deployment/test_compose_generator.py``.
REPO_ROOT_LIMITS_MOUNT: dict[str, str] = {
    "source": "./data/channel_limits.json",
    "target": "/app/project/data/channel_limits.json",
}

#: The same mount for a config read from the build zone: only the SOURCE moves,
#: because only it resolves against the repo root.
BUILD_ZONE_LIMITS_MOUNT: dict[str, str] = {
    "source": "./build/data/channel_limits.json",
    "target": "/app/project/data/channel_limits.json",
}

#: Where the worker reads its device document, and the one bind that puts a
#: file there. The source is a literal staged path (``_stage_bluesky_devices``
#: owns the basename), so unlike the limits mount it is spelled in the template
#: and pinned here verbatim.
DEVICES_FILE_TARGET = "/app/project/data/bluesky_devices.yml"
DEVICES_MOUNT = f"./build/services/bluesky/bluesky_devices.yml:{DEVICES_FILE_TARGET}:ro"


def _render(
    *,
    tiled_enabled: bool = False,
    writes_enabled: bool = False,
    deployed_services: list[str] | None = None,
    plan_dir: str | None = None,
    excluded_plans: str | None = None,
    devices_present: bool = False,
    limits_mount: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Render the packaged bluesky compose template and parse the result.

    Mirrors ``compose_generator.render_template``'s context contract: the whole
    config dict plus the ``osprey_labels``/``osprey_version`` metadata that
    ``_inject_project_metadata`` supplies.

    Two of those keys are computed by the generator rather than configured, and
    a hand-built context has to TYPE them or it pins a render no deploy can
    reach:

    ``bluesky_devices``
        The real boolean ``_stage_bluesky_devices`` returns — True only when a
        device file actually landed in the build context. Always present on a
        production render, so it is always present here too; *devices_present*
        chooses which of the two answers this render carries.
    ``limits_mount``
        The finished bind mount ``resolve_limits_mount`` computes. It exists
        exactly when the key it derives from names a path, and a writable
        deployment can never reach the template without it — so it is injected
        whenever *writes_enabled* is on, defaulting to the repo-root shape.
        Passing an explicit dict renders another entry point's spelling.
    """
    # The generator precomputes each lane's posture onto its service entry
    # (the template cannot resolve a target itself); a direct render supplies it.
    bluesky: dict[str, Any] = {
        "port": 8090,
        "tiled_enabled": tiled_enabled,
        "writes_enabled": writes_enabled,
    }
    if plan_dir is not None:
        bluesky["plan_dir"] = plan_dir
    if excluded_plans is not None:
        bluesky["excluded_plans"] = excluded_plans

    # ``lane_keys`` is seeded with 'bluesky' unconditionally (the template's
    # lane axis), so every render defines lane 1's containers whatever
    # deployed_services says — and every context therefore has to carry the
    # matching services block.
    deployed = list(deployed_services or ["bluesky"])
    assert "bluesky" in deployed, "lane 1 always renders; a context omitting it is not reachable"

    context: dict[str, Any] = {
        "osprey_labels": {
            "project_name": "proj",
            "project_root": "/tmp/proj",
        },
        "osprey_images": _image_defaults("proj"),
        "osprey_version": "2026.8.1",
        "system": {"timezone": "UTC"},
        "deployment": {},
        "deployed_services": deployed,
        "control_system": {"writes_enabled": writes_enabled},
        "services": {"bluesky": bluesky, "virtual_accelerator": {"port": 5064}},
        "bluesky_devices": devices_present,
    }
    if limits_mount is not None:
        context["limits_mount"] = limits_mount
    elif writes_enabled:
        context["limits_mount"] = REPO_ROOT_LIMITS_MOUNT
    env = Environment(loader=FileSystemLoader(_LOADER_ROOTS))
    return yaml.safe_load(env.get_template(BLUESKY_TEMPLATE).render(context))


@pytest.fixture
def rendered() -> dict[str, Any]:
    """A stock render: bridge-only deploy, no Tiled, writes disabled."""
    return _render()


# ---------------------------------------------------------------------------
# Service presence and shape.
# ---------------------------------------------------------------------------


def test_queueserver_and_redis_services_exist(rendered: dict[str, Any]) -> None:
    assert set(rendered["services"]) == {"bluesky-bridge", "queueserver", "bluesky-redis"}


def _manager_argv(rendered: dict[str, Any]) -> str:
    """The `start-re-manager` command line, as one string.

    The command is a `sh -c` form (it resolves the startup script's in-image
    path at container start), so the arguments live in the shell string rather
    than in a list.
    """
    command = rendered["services"]["queueserver"]["command"]
    assert command[:2] == ["sh", "-c"]
    return command[2]


def test_queueserver_runs_the_re_manager_on_the_bridge_image(rendered: dict[str, Any]) -> None:
    """One image, two entrypoints — the worker namespace and the bridge must
    share the osprey code they disagree about at their peril."""
    queueserver = rendered["services"]["queueserver"]
    bridge = rendered["services"]["bluesky-bridge"]
    assert queueserver["image"] == bridge["image"]
    assert "exec start-re-manager" in _manager_argv(rendered)


def test_queueserver_has_no_build_block(rendered: dict[str, Any]) -> None:
    """Two services building the same tag race each other; the bridge owns the
    build (same rule the dispatch worker follows)."""
    assert "build" not in rendered["services"]["queueserver"]


def test_startup_code_is_loaded_as_a_script_never_as_a_module(
    rendered: dict[str, Any],
) -> None:
    """The two flags are NOT interchangeable. Upstream's `load_startup_module`
    plain-imports, while `load_startup_script` execs with `__name__` patched to
    `"__main__"` — and qserver_startup.py wires everything under an
    `if __name__ == "__main__":` guard. The module form would import it, run
    nothing, and leave the worker with an empty namespace and no error."""
    argv = _manager_argv(rendered)
    assert "--startup-script" in argv
    assert "--startup-module" not in argv


def test_startup_script_path_is_resolved_from_the_installed_package(
    rendered: dict[str, Any],
) -> None:
    """Hardcoding a site-packages path would break on any interpreter bump and
    would miss the `--dev` wheel overlay. `$$` is compose's escape for a
    literal `$`, so the subshell runs in the container, not at render time."""
    argv = _manager_argv(rendered)
    assert "$$(python -c" in argv
    assert "osprey.services.bluesky_bridge.qserver_startup" in argv
    assert "m.__file__" in argv


def test_queueserver_points_at_the_internal_redis(rendered: dict[str, Any]) -> None:
    assert "--redis-addr bluesky-redis:6379" in _manager_argv(rendered)


def test_plan_list_persistence_is_disabled(rendered: dict[str, Any]) -> None:
    """PERSISTENCE only. With `--startup-script` (not a startup DIRECTORY)
    upstream leaves its aux_dir None, so the default ENVIRONMENT_OPEN would make
    every environment open attempt a write for no benefit; NEVER keeps that off
    the path. Nothing is lost — the manager downloads the real plan/device list
    from the worker on every environment open
    (`_load_existing_plans_and_devices_from_worker`), so this flag does not
    affect what can be enqueued. It is independent of the mandatory
    `--existing-plans-devices` PATH asserted below."""
    assert "--update-existing-plans-devices NEVER" in _manager_argv(rendered)


def test_manager_is_given_the_mandatory_existing_plans_devices_path(
    rendered: dict[str, Any],
) -> None:
    """Omitting this path does not degrade the manager — it kills it.

    Upstream's `start_manager` treats an unset
    ``existing_plans_and_devices_path`` as fatal (``logger.error(...); return
    1``), so a command without this flag exits immediately and the container
    restart-loops forever, with the bridge reporting ``manager_unreachable``
    and no plan ever running. That is exactly what shipped until the
    container e2e (``tests/e2e/test_bluesky_queue_e2e.py``) actually deployed
    the stack: this render suite asserted the argv we WROTE, which upstream is
    free to reject.

    Only the DIRECTORY has to exist — a missing FILE is a startup warning
    ("populated after RE worker environment is opened the first time"), which
    is why ``NEVER`` above stays correct alongside it. ``/app/qserver`` is
    guaranteed to exist because Docker creates the parent directory of the
    ``user_group_permissions.yaml`` single-file bind mount, asserted here
    together so the path and the thing that guarantees it cannot drift apart.
    """
    argv = _manager_argv(rendered)
    assert "--existing-plans-devices /app/qserver/existing_plans_and_devices.yaml" in argv
    assert (
        "./build/services/bluesky/user_group_permissions.yaml:"
        "/app/qserver/user_group_permissions.yaml:ro"
        in rendered["services"]["queueserver"]["volumes"]
    ), "the mount that guarantees /app/qserver exists is gone; the path above now has no directory"


def test_existing_plans_devices_flag_avoids_upstreams_own_misspelling(
    rendered: dict[str, Any],
) -> None:
    """A NEGATIVE pin for a flag nobody wrote — because upstream will tell them to.

    When the flag is missing, upstream's error message reads:

        The path to the list of existing plans and devices
        (--existing-plans-and-devices) is not specified.

    That flag does not exist. argparse accepts only ``--existing-plans-devices``
    (no second "and"), matching ``--update-existing-plans-devices``. Copying the
    error message verbatim — the obvious thing to do — yields
    ``start-re-manager: error: unrecognized arguments`` and the SAME restart
    loop, one step further along. This pin exists so the next person meets a
    failing test instead of that loop.
    """
    assert "--existing-plans-and-devices" not in _manager_argv(rendered)


def test_queueserver_depends_on_redis_being_healthy(rendered: dict[str, Any]) -> None:
    depends = rendered["services"]["queueserver"]["depends_on"]
    assert depends["bluesky-redis"] == {"condition": "service_healthy"}


def test_redis_persists_to_a_named_volume(rendered: dict[str, Any]) -> None:
    """A queue that survives a bridge restart but not a Redis restart would be
    a half-kept promise."""
    redis = rendered["services"]["bluesky-redis"]
    assert "--appendonly" in redis["command"]
    assert redis["volumes"] == ["bluesky_queueserver_redis:/data"]
    assert "bluesky_queueserver_redis" in rendered["volumes"]


# ---------------------------------------------------------------------------
# Isolation: the bridge is the only route to the manager.
# ---------------------------------------------------------------------------


def test_neither_new_service_publishes_a_port(rendered: dict[str, Any]) -> None:
    """A published control socket would be a second route to plan execution,
    beside the bridge's launch-token gate."""
    assert "ports" not in rendered["services"]["queueserver"]
    assert "ports" not in rendered["services"]["bluesky-redis"]


def test_redis_is_on_the_internal_network_only(rendered: dict[str, Any]) -> None:
    assert rendered["services"]["bluesky-redis"]["networks"] == ["bluesky-internal"]


def test_internal_network_is_declared_internal(rendered: dict[str, Any]) -> None:
    assert rendered["networks"]["bluesky-internal"] == {"internal": True}


def test_bridge_and_queueserver_are_dual_homed(rendered: dict[str, Any]) -> None:
    """Both need osprey-network (Tiled, the VA, the bluesky-web sidecar) as well as
    the internal network — which is exactly why key auth, not network
    placement, is what protects the manager's sockets."""
    for service in ("bluesky-bridge", "queueserver"):
        assert set(rendered["services"][service]["networks"]) == {
            "osprey-network",
            "bluesky-internal",
        }


# ---------------------------------------------------------------------------
# CurveZMQ: two independent keypairs, fail-closed passthrough.
# ---------------------------------------------------------------------------


def test_control_plane_keys_are_split_across_the_two_services(
    rendered: dict[str, Any],
) -> None:
    """The manager holds the private key, the bridge only the public one."""
    manager_env = rendered["services"]["queueserver"]["environment"]
    bridge_env = rendered["services"]["bluesky-bridge"]["environment"]

    # Exact guard syntax is pinned separately; what matters here is which
    # container holds which half.
    assert "BLUESKY_QSERVER_ZMQ_PRIVATE_KEY" in manager_env["QSERVER_ZMQ_PRIVATE_KEY_FOR_SERVER"]
    assert "QSERVER_ZMQ_PRIVATE_KEY_FOR_SERVER" not in bridge_env
    assert "BLUESKY_QSERVER_ZMQ_PUBLIC_KEY" in bridge_env["QSERVER_ZMQ_PUBLIC_KEY"]


def test_document_plane_uses_certificate_paths_not_key_strings(
    rendered: dict[str, Any],
) -> None:
    """``bluesky.callbacks.zmq``'s ServerCurve/ClientCurve take filesystem
    paths, so these env vars name certificates — the names and the meaning are
    ``qserver_startup``'s published contract on the publisher side."""
    bridge_env = rendered["services"]["bluesky-bridge"]["environment"]
    manager_env = rendered["services"]["queueserver"]["environment"]

    # Bridge binds the proxy: server secret, plus the FOLDER of client public
    # keys it will accept (ServerCurve.client_public_keys is a directory).
    assert bridge_env["BLUESKY_ZMQ_CURVE_SECRET_KEY"] == "/app/curve/proxy.key_secret"
    assert bridge_env["BLUESKY_ZMQ_CURVE_CLIENT_PUBLIC_KEYS"] == "/app/curve/clients"
    # Publisher side holds the mirror image.
    assert manager_env["BLUESKY_ZMQ_CURVE_SECRET_KEY"] == "/app/curve/publisher.key_secret"
    assert manager_env["BLUESKY_ZMQ_CURVE_SERVER_PUBLIC_KEY"] == "/app/curve/proxy.key"


def test_document_plane_secrets_do_not_cross_containers(rendered: dict[str, Any]) -> None:
    """Each side mounts only the directory holding the secret it is entitled
    to: splitting by directory (not by individual file) is what makes that
    enforceable rather than merely conventional."""
    bridge = rendered["services"]["bluesky-bridge"]
    queueserver = rendered["services"]["queueserver"]

    assert "./data/.runtime/bluesky_curve/bridge:/app/curve:ro" in bridge["volumes"]
    assert "./data/.runtime/bluesky_curve/queueserver:/app/curve:ro" in queueserver["volumes"]
    # Neither container can even see the other's certificate directory.
    assert not any("bluesky_curve/queueserver" in str(v) for v in bridge["volumes"])
    assert not any("bluesky_curve/bridge" in str(v) for v in queueserver["volumes"])


def test_document_plane_certificates_are_mounted_read_only(rendered: dict[str, Any]) -> None:
    for service in ("bluesky-bridge", "queueserver"):
        mounts = [v for v in rendered["services"][service]["volumes"] if "/app/curve" in str(v)]
        assert mounts and all(str(m).endswith(":ro") for m in mounts)


@pytest.mark.parametrize(
    "service,var",
    [
        ("queueserver", "QSERVER_ZMQ_PRIVATE_KEY_FOR_SERVER"),
        ("queueserver", "QSERVER_ZMQ_PUBLIC_KEY"),
        ("bluesky-bridge", "QSERVER_ZMQ_PUBLIC_KEY"),
    ],
)
def test_control_plane_keys_fail_the_deploy_when_unminted(
    rendered: dict[str, Any], service: str, var: str
) -> None:
    """Every control-plane key must carry compose's `:?` guard.

    This is the difference between a loud deploy failure and a silently
    PLAINTEXT control socket. Upstream reads an empty key as "encryption off"
    (config.py's `key or None`), so a bare `${VAR}` — let alone `${VAR:-}` —
    turns a missing mint into a manager that accepts unauthenticated commands.
    The queueserver is dual-homed onto osprey-network, so that would put plan
    execution within reach of anything on that network, around the bridge's
    launch-token gate entirely.

    Regression guard with a specific failure in mind: if the mint in
    container_lifecycle.py is ever removed, renamed, or made conditional, this
    keeps the template from quietly falling back to plaintext.
    """
    value = rendered["services"][service]["environment"][var]
    assert value.startswith("${") and ":?" in value, (
        f"{service}.{var} must use the fail-closed ${{VAR:?...}} form, got {value!r}"
    )
    assert ":-" not in value


def test_control_plane_guards_name_the_variables_the_deploy_actually_mints() -> None:
    """The `:?` guards are only fail-CLOSED if they guard the names
    `osprey up` mints — a guard on a misspelled variable would abort
    every deploy instead, which is a different bug wearing the same clothes."""
    from osprey.deployment.container_lifecycle import (
        _QSERVER_ZMQ_PRIVATE_KEY_VAR,
        _QSERVER_ZMQ_PUBLIC_KEY_VAR,
    )

    rendered = _render()
    manager_env = rendered["services"]["queueserver"]["environment"]
    bridge_env = rendered["services"]["bluesky-bridge"]["environment"]

    assert manager_env["QSERVER_ZMQ_PRIVATE_KEY_FOR_SERVER"].startswith(
        "${" + _QSERVER_ZMQ_PRIVATE_KEY_VAR + ":?"
    )
    for env in (manager_env, bridge_env):
        assert env["QSERVER_ZMQ_PUBLIC_KEY"].startswith("${" + _QSERVER_ZMQ_PUBLIC_KEY_VAR + ":?")


# ---------------------------------------------------------------------------
# Document-plane addressing.
# ---------------------------------------------------------------------------


def test_proxy_addresses_are_bind_form_on_the_bridge(rendered: dict[str, Any]) -> None:
    """The bridge runs the proxy and binds BOTH sockets; the out-side stays on
    loopback because only the bridge's own dispatcher subscribes."""
    bridge_env = rendered["services"]["bluesky-bridge"]["environment"]
    assert bridge_env["BLUESKY_ZMQ_PUBLISH_ADDR"] == "tcp://*:5567"
    assert bridge_env["BLUESKY_ZMQ_SUBSCRIBE_ADDR"] == "tcp://127.0.0.1:5568"


def test_publisher_connects_to_the_bridge_in_side(rendered: dict[str, Any]) -> None:
    manager_env = rendered["services"]["queueserver"]["environment"]
    assert manager_env["BLUESKY_ZMQ_PUBLISH_ADDR"] == "tcp://bluesky-bridge:5567"
    # The manager never subscribes — it only publishes.
    assert "BLUESKY_ZMQ_SUBSCRIBE_ADDR" not in manager_env


def test_bridge_reaches_the_manager_by_service_name(rendered: dict[str, Any]) -> None:
    bridge_env = rendered["services"]["bluesky-bridge"]["environment"]
    assert bridge_env["QSERVER_ZMQ_CONTROL_ADDRESS"] == "tcp://queueserver:60615"
    assert "--zmq-control-addr 'tcp://*:60615'" in _manager_argv(rendered)


# ---------------------------------------------------------------------------
# User-group permissions: without them the manager rejects everything.
# ---------------------------------------------------------------------------


def test_manager_is_given_a_user_group_permissions_file(rendered: dict[str, Any]) -> None:
    """`bluesky-queueserver-api` sends the `primary` group by default. With no
    permissions file the manager builds only `root`, and every bridge request
    fails with "Unknown user group: 'primary'" — plans that scanned correctly
    and exist in the namespace still cannot be enqueued."""
    argv = _manager_argv(rendered)
    assert "--user-group-permissions /app/qserver/user_group_permissions.yaml" in argv
    assert (
        "./build/services/bluesky/user_group_permissions.yaml:"
        "/app/qserver/user_group_permissions.yaml:ro"
        in rendered["services"]["queueserver"]["volumes"]
    )


def test_shipped_permissions_pass_upstream_and_admit_catalog_plans() -> None:
    """Run the file through queueserver's OWN loader rather than asserting on
    its YAML shape: the shape is upstream's contract, not ours, and a
    schema-valid file that still resolved to an empty allowed list would look
    identical to a correct one from the outside.

    Also pins the two things that actually break a deploy: `primary` must
    exist (the API client's default group — its absence is the "Unknown user
    group" rejection), and ordinary catalog plans must survive the root
    filter while private names do not.
    """
    from bluesky_queueserver.manager.profile_ops import (
        load_allowed_plans_and_devices,
        load_user_group_permissions,
    )

    permissions = load_user_group_permissions(
        str(TEMPLATE_DIR / "bluesky" / "user_group_permissions.yaml")
    )
    assert {"root", "primary"} <= set(permissions["user_groups"])

    allowed_plans, allowed_devices = load_allowed_plans_and_devices(
        existing_plans={
            "grid_scan": {"name": "grid_scan"},
            "orm": {"name": "orm"},
            "_private": {"name": "_private"},
        },
        existing_devices={"corr1": {"name": "corr1"}, "bpm1": {"name": "bpm1"}},
        user_group_permissions=permissions,
    )
    assert sorted(allowed_plans["primary"]) == ["grid_scan", "orm"]
    assert sorted(allowed_devices["primary"]) == ["bpm1", "corr1"]


def test_shipped_permissions_allow_only_preview_plan() -> None:
    """`function_execute` runs arbitrary callables in the worker namespace,
    outside the plan path and outside the connector's reference monitor, so
    this must stay closed except for one deliberate, read-only exception.

    Proved with queueserver's own `check_if_function_allowed` rather than by
    reading the YAML shape, so the assertion tracks upstream's actual
    resolution (root-as-preliminary-filter, allow/forbid pattern matching)
    instead of our assumptions about it. `preview_plan_in_namespace` and
    `collect_channel_moves` are sibling worker-namespace callables that share
    `preview_plan`'s prefix; both must stay denied, which also pins that the
    entry is an exact-match literal, not an unanchored substring or regex."""
    from bluesky_queueserver.manager.profile_ops import (
        check_if_function_allowed,
        load_user_group_permissions,
    )

    permissions = load_user_group_permissions(
        str(TEMPLATE_DIR / "bluesky" / "user_group_permissions.yaml")
    )
    for group in ("root", "primary"):
        assert check_if_function_allowed(
            "preview_plan", group_name=group, user_group_permissions=permissions
        )
        for other in ("preview_plan_in_namespace", "collect_channel_moves"):
            assert not check_if_function_allowed(
                other, group_name=group, user_group_permissions=permissions
            )


# ---------------------------------------------------------------------------
# Config, limits and device-file wiring.
# ---------------------------------------------------------------------------


def test_queueserver_gets_the_config_mount_and_config_file(rendered: dict[str, Any]) -> None:
    """The reference monitor runs in this container now, so it needs the same
    config.yml the bridge reads — and CONFIG_FILE to find it, since CWD is the
    image WORKDIR, not the project dir."""
    queueserver = rendered["services"]["queueserver"]
    assert queueserver["environment"]["CONFIG_FILE"] == "/app/project/config.yml"
    assert (
        "./build/services/bluesky/config.yml:/app/project/config.yml:ro" in queueserver["volumes"]
    )


def test_limits_db_is_mounted_read_only_when_writes_are_enabled() -> None:
    """Without the DB the empty-DB failsafe blocks every write, so a
    writes-enabled deploy that skipped this mount could not run a plan at all."""
    rendered = _render(writes_enabled=True)
    assert (
        "./data/channel_limits.json:/app/project/data/channel_limits.json:ro"
        in rendered["services"]["queueserver"]["volumes"]
    )


def test_limits_db_reaches_both_containers_that_enforce_limits() -> None:
    """The bridge and the manager mount the SAME file, from one computed pair.

    Two containers now hold a reference monitor, and the template consumes the
    same ``limits_mount`` strings for both. Asserting them together is what
    proves the mount cannot drift between the two: a bridge checking one file
    while the worker that actually issues the puts checks another would enforce
    limits nobody configured.
    """
    rendered = _render(writes_enabled=True)
    expected = "./data/channel_limits.json:/app/project/data/channel_limits.json:ro"
    for service in ("bluesky-bridge", "queueserver"):
        assert expected in rendered["services"][service]["volumes"], service


def test_limits_db_renders_the_build_zone_source_unchanged() -> None:
    """Only the SOURCE moves when the deployed config is read from ``build/``.

    The template makes no path decision of its own — it interpolates the two
    halves it is handed. This pins that: hand it the build-zone pair and the
    render carries it verbatim, with the container-side target untouched
    (the connector resolves the configured relative path against the container
    project root either way, so a prefixed target would miss the mount).
    """
    rendered = _render(writes_enabled=True, limits_mount=BUILD_ZONE_LIMITS_MOUNT)
    expected = "./build/data/channel_limits.json:/app/project/data/channel_limits.json:ro"
    for service in ("bluesky-bridge", "queueserver"):
        assert expected in rendered["services"][service]["volumes"], service


def test_limits_db_renders_an_absolute_operator_path_verbatim_on_both_sides() -> None:
    """An absolute path is operator-owned and mounted at the identical path.

    ``resolve_limits_mount`` never rewrites it, so what the template must not do
    is re-anchor it — prefixing either half would point the bind at a file that
    is not there.
    """
    absolute = {"source": "/srv/facility/limits.json", "target": "/srv/facility/limits.json"}
    rendered = _render(writes_enabled=True, limits_mount=absolute)
    assert (
        "/srv/facility/limits.json:/srv/facility/limits.json:ro"
        in rendered["services"]["queueserver"]["volumes"]
    )


def test_limits_db_is_absent_on_a_read_only_deploy(rendered: dict[str, Any]) -> None:
    assert not any(
        "channel_limits.json" in str(v) for v in rendered["services"]["queueserver"]["volumes"]
    )


def test_device_file_env_and_mount_arrive_together_when_one_was_staged() -> None:
    """The env var names a container path; the mount is what puts a file there.

    They are two halves of one decision (``_stage_bluesky_devices``'s return
    value gates both), and either half alone is a worker that fails its own
    load: an env var pointing at nothing, or a file no one reads.
    """
    queueserver = _render(devices_present=True)["services"]["queueserver"]
    assert queueserver["environment"]["BLUESKY_DEVICES_FILE"] == DEVICES_FILE_TARGET
    assert DEVICES_MOUNT in queueserver["volumes"]


def test_device_file_env_and_mount_are_both_absent_when_nothing_was_staged(
    rendered: dict[str, Any],
) -> None:
    """Browse-only is the fail-closed direction and must be silent.

    Naming a file the mount does not carry would fail the worker's load rather
    than degrade to a worker that can browse plans and run none, which is what
    a deployment with no device set is supposed to get.
    """
    queueserver = rendered["services"]["queueserver"]
    assert "BLUESKY_DEVICES_FILE" not in (queueserver["environment"] or {})
    assert not any("bluesky_devices" in str(volume) for volume in queueserver["volumes"])


def test_device_file_is_not_gated_on_the_va_being_co_deployed() -> None:
    """A facility on real EPICS has no VA container but still has devices.

    The device set now comes from an authored file rather than from PV
    spellings the container infers, but the claim is the one the substrate env
    carried before it: gating the wiring on the VA would leave every real-EPICS
    worker with no devices at all.
    """
    queueserver = _render(devices_present=True, deployed_services=["bluesky"])["services"][
        "queueserver"
    ]
    assert queueserver["environment"]["BLUESKY_DEVICES_FILE"] == DEVICES_FILE_TARGET
    assert DEVICES_MOUNT in queueserver["volumes"]


def test_only_the_queueserver_is_given_the_device_file() -> None:
    """The bridge is a facade over the manager and builds no devices of its own.

    Handing it the file would give the deployment two device sets built from
    one document by two processes, and the bridge's copy would be the one no
    plan ever runs against.
    """
    rendered = _render(devices_present=True)
    bridge = rendered["services"]["bluesky-bridge"]
    assert "BLUESKY_DEVICES_FILE" not in (bridge["environment"] or {})
    assert not any("bluesky_devices" in str(volume) for volume in bridge["volumes"])


def test_the_retired_substrate_variables_appear_nowhere_in_the_render() -> None:
    """The three retired ``BLUESKY_EPICS`` names are gone from the whole contract.

    A leftover passthrough would be dead config that still looks live to an
    operator reading the compose file — and, worse, the worker no longer reads
    those names, so a deployment setting them would come up browse-only with no
    indication of why.
    """
    for devices_present in (False, True):
        rendered = _render(devices_present=devices_present)
        assert "BLUESKY_EPICS" not in yaml.safe_dump(rendered)


def test_plan_dirs_and_exclusions_reach_the_worker() -> None:
    """The worker builds its namespace from the same catalog the operator
    composes against — including the exclusions, so a hidden plan is not
    reachable by name through a hand-built queue item."""
    rendered = _render(plan_dir="/opt/facility/plans", excluded_plans="orm")
    queueserver = rendered["services"]["queueserver"]
    assert queueserver["environment"]["BLUESKY_PLAN_DIRS"] == "/app/project/plans"
    assert queueserver["environment"]["BLUESKY_EXCLUDED_PLANS"] == "orm"
    assert "/opt/facility/plans:/app/project/plans:ro" in queueserver["volumes"]


def test_no_plan_dir_omits_the_mount_and_env(rendered: dict[str, Any]) -> None:
    queueserver = rendered["services"]["queueserver"]
    assert "BLUESKY_PLAN_DIRS" not in queueserver["environment"]
    assert not any(str(v).endswith(":/app/project/plans:ro") for v in queueserver["volumes"])


# ---------------------------------------------------------------------------
# Conditional co-deployment blocks.
# ---------------------------------------------------------------------------


def test_va_co_deploy_adds_the_healthcheck_dependency_and_ca_env() -> None:
    rendered = _render(deployed_services=["bluesky", "virtual_accelerator"])
    queueserver = rendered["services"]["queueserver"]
    assert queueserver["depends_on"]["virtual-accelerator"] == {"condition": "service_healthy"}
    assert queueserver["environment"]["EPICS_CA_NAME_SERVERS"] == "virtual-accelerator:5064"
    assert queueserver["environment"]["EPICS_CA_AUTO_ADDR_LIST"] == "NO"


def test_bridge_only_deploy_never_names_the_va(rendered: dict[str, Any]) -> None:
    """compose errors outright on a depends_on naming an undefined service."""
    queueserver = rendered["services"]["queueserver"]
    assert "virtual-accelerator" not in queueserver["depends_on"]
    assert "EPICS_CA_NAME_SERVERS" not in queueserver["environment"]


def test_tiled_env_is_gated_on_tiled_enabled(rendered: dict[str, Any]) -> None:
    assert "BLUESKY_TILED_URI" not in rendered["services"]["queueserver"]["environment"]


def test_tiled_enabled_gives_the_worker_the_writer_credentials() -> None:
    """The startup module owns the TiledWriter subscription — documents are
    persisted from the process that produces them."""
    rendered = _render(tiled_enabled=True)
    manager_env = rendered["services"]["queueserver"]["environment"]
    assert manager_env["BLUESKY_TILED_URI"] == "http://tiled:8000"
    assert manager_env["BLUESKY_TILED_API_KEY"] == "${BLUESKY_TILED_API_KEY}"
    # Both volumes coexist once Tiled is on.
    assert set(rendered["volumes"]) == {"bluesky_queueserver_redis", "bluesky_tiled_catalog"}


# ---------------------------------------------------------------------------
# Healthcheck semantics.
# ---------------------------------------------------------------------------


def test_manager_healthcheck_probes_responsiveness_not_environment_state(
    rendered: dict[str, Any],
) -> None:
    """`qserver ping` succeeds with the RE worker environment CLOSED — the
    correct reading on a mock deployment, which is browse-only but healthy.
    A probe that required an open environment would mark every mock deploy
    unhealthy."""
    healthcheck = rendered["services"]["queueserver"]["healthcheck"]
    assert "qserver ping" in healthcheck["test"][1]
    for token in ("environment_open", "status", "queue_start"):
        assert token not in healthcheck["test"][1]


def test_redis_has_a_healthcheck_so_depends_on_can_wait_for_it(
    rendered: dict[str, Any],
) -> None:
    assert rendered["services"]["bluesky-redis"]["healthcheck"]["test"] == [
        "CMD",
        "redis-cli",
        "ping",
    ]


# ---------------------------------------------------------------------------
# Pins and overrides.
# ---------------------------------------------------------------------------


def test_redis_image_is_pinned_and_overridable(rendered: dict[str, Any]) -> None:
    image = rendered["services"]["bluesky-redis"]["image"]
    assert image.startswith("${OSPREY_BLUESKY_REDIS_IMAGE:-")
    assert "redis:7.4-alpine" in image


def test_redis_image_honours_a_config_override() -> None:
    context_image = "my-registry/redis:7.4"
    env = Environment(loader=FileSystemLoader(_LOADER_ROOTS))
    rendered = yaml.safe_load(
        env.get_template(BLUESKY_TEMPLATE).render(
            {
                "osprey_labels": {
                    "project_name": "proj",
                    "project_root": "/tmp/proj",
                },
                "osprey_images": _image_defaults("proj"),
                "system": {"timezone": "UTC"},
                "deployment": {},
                "deployed_services": ["bluesky"],
                "services": {"bluesky": {"redis_image": context_image}},
            }
        )
    )
    assert context_image in rendered["services"]["bluesky-redis"]["image"]


def _dockerfile_queueserver_pin() -> str:
    dockerfile = (TEMPLATE_DIR / "bluesky" / "Dockerfile").read_text(encoding="utf-8")
    match = re.search(r'^ARG BLUESKY_QUEUESERVER_VERSION="([^"]+)"', dockerfile, re.MULTILINE)
    assert match, "Dockerfile must pin bluesky-queueserver via ARG BLUESKY_QUEUESERVER_VERSION"
    return match.group(1)


def _locked_queueserver_version() -> str:
    lock = (Path(__file__).resolve().parents[2] / "uv.lock").read_text(encoding="utf-8")
    match = re.search(
        r'^\[\[package\]\]\nname = "bluesky-queueserver"\nversion = "([^"]+)"',
        lock,
        re.MULTILINE,
    )
    assert match, "bluesky-queueserver not found in uv.lock"
    return match.group(1)


def test_dockerfile_pins_the_queueserver_version() -> None:
    """The manager the image runs and the client the bridge talks to must be a
    deliberately chosen pair, not whatever the transitive resolve produced."""
    dockerfile = (TEMPLATE_DIR / "bluesky" / "Dockerfile").read_text(encoding="utf-8")
    assert _dockerfile_queueserver_pin()
    assert 'pip install --no-cache-dir "bluesky-queueserver==$BLUESKY_QUEUESERVER_VERSION"' in (
        dockerfile
    )


def test_dockerfile_queueserver_pin_matches_the_lockfile() -> None:
    """The image's manager and the bridge's client must be the SAME version.

    Nothing bumps this literal automatically — dependabot has no docker
    ecosystem configured here and would not parse an ARG in any case — so a
    `uv lock` that moves bluesky-queueserver would otherwise leave the image
    behind, and the resulting manager/client skew would surface as protocol
    errors in a container, not as a failing test. This is the check that makes
    "manually maintained" safe.
    """
    assert _dockerfile_queueserver_pin() == _locked_queueserver_version(), (
        "Dockerfile's BLUESKY_QUEUESERVER_VERSION and uv.lock's bluesky-queueserver "
        "have drifted — update the Dockerfile ARG to match the lockfile"
    )


# ---------------------------------------------------------------------------
# The two `pip check` gates, and the asymmetry between them.
#
# Found by the real-container e2e (tests/e2e/test_bluesky_queue_e2e.py): the
# deps layer holds the last RELEASED framework beside the WORKING TREE's
# dependency pins, so a strict `pip check` there fails EVERY `--dev` build for
# as long as any pin has moved since the last release. The wheel layer, which
# installs the local wheel the local pins actually belong to, is where the
# check means something on that path — so it stays unconditional. These pins
# exist because that asymmetry is invisible in a green unit-test run and only
# shows up as a failed image build.
#
# For reference: the sibling bluesky_web Dockerfile has NEVER had a
# deps-layer `pip check` — wheel-layer-only is the house shape, and this
# template is the one that diverged from it.
# ---------------------------------------------------------------------------

_DEV_BUILD_ARG = "OSPREY_DEV"


def _dockerfile_layers() -> tuple[str, str]:
    """Split the Dockerfile into (deps layer, wheel layer) source text."""
    dockerfile = (TEMPLATE_DIR / "bluesky" / "Dockerfile").read_text(encoding="utf-8")
    marker = "# ── wheel layer ─"
    assert marker in dockerfile, "the Dockerfile's wheel-layer section marker moved"
    deps, _, wheel = dockerfile.partition(marker)
    deps_start = deps.index("# ── deps layer ─")
    return deps[deps_start:], wheel


def test_deps_layer_pip_check_is_skipped_only_on_dev_builds() -> None:
    """The deps-layer `pip check` still runs — for non-dev builds only.

    Asserted as one literal rather than "a pip check appears somewhere": the
    failure mode this guards against is the check being deleted outright (the
    strict release-build gate silently gone) or the guard being widened to skip
    it always.
    """
    deps, _wheel = _dockerfile_layers()
    guarded = f'&& {{ [ "${_DEV_BUILD_ARG}" = "1" ] || pip check ; }}'
    assert guarded in deps, (
        "the deps layer must still run `pip check`, guarded so ONLY dev builds "
        f"skip it; expected the literal {guarded!r}"
    )
    # Negative control: no UNGUARDED `pip check` in this layer, which would put
    # the dev-build failure straight back.
    for line in deps.splitlines():
        stripped = line.strip()
        if "pip check" in stripped and stripped.startswith("&&"):
            assert _DEV_BUILD_ARG in stripped, (
                f"unguarded `pip check` in the deps layer would fail every dev build: {line!r}"
            )


def test_wheel_layer_pip_check_stays_unconditional() -> None:
    """The wheel layer's `pip check` carries NO dev exemption.

    This is the check that actually gates a dev image: it runs after the local
    wheel is installed, so the pins it validates are the ones the image ships.
    Weakening it the same way as the deps layer would leave a `--dev` build
    with no dependency-consistency gate at all.
    """
    _deps, wheel = _dockerfile_layers()
    assert "&& pip check ;" in wheel, "the wheel layer must run `pip check`"
    assert _DEV_BUILD_ARG not in wheel, (
        "the wheel layer's `pip check` must not be exempted on dev builds — it is "
        "the only dependency gate a --dev image has"
    )


def test_dev_guard_keys_on_the_build_arg_the_compose_template_passes() -> None:
    """The Dockerfile's dev guard and the compose template's dev build-arg agree.

    Keyed across the two files rather than within one: a rename on either side
    would silently DETACH the guard (the deps-layer check would go back to
    running on dev builds and fail them all), and neither file's own tests
    would notice.
    """
    deps, _wheel = _dockerfile_layers()
    assert f'ARG {_DEV_BUILD_ARG}=""' in deps, (
        f"the Dockerfile no longer declares ARG {_DEV_BUILD_ARG}"
    )

    env = Environment(loader=FileSystemLoader(_LOADER_ROOTS))
    rendered = yaml.safe_load(
        env.get_template(BLUESKY_TEMPLATE).render(
            {
                "osprey_labels": {
                    "project_name": "proj",
                    "project_root": "/tmp/proj",
                },
                "osprey_images": _image_defaults("proj"),
                "system": {"timezone": "UTC"},
                "deployment": {},
                "deployed_services": ["bluesky"],
                "services": {"bluesky": {}},
                "dev_mode": True,
            }
        )
    )
    build_args = rendered["services"]["bluesky-bridge"]["build"]["args"]
    assert build_args.get(_DEV_BUILD_ARG) == "1", (
        f"`osprey up --dev` must pass {_DEV_BUILD_ARG}=1, which is the value "
        f"the Dockerfile's guard compares against: {build_args}"
    )

    # Negative control: a non-dev render must NOT set it, or the deps-layer
    # check would be skipped on release builds too.
    non_dev = yaml.safe_load(
        env.get_template(BLUESKY_TEMPLATE).render(
            {
                "osprey_labels": {
                    "project_name": "proj",
                    "project_root": "/tmp/proj",
                },
                "osprey_images": _image_defaults("proj"),
                "system": {"timezone": "UTC"},
                "deployment": {},
                "deployed_services": ["bluesky"],
                "services": {"bluesky": {}},
            }
        )
    )
    assert _DEV_BUILD_ARG not in non_dev["services"]["bluesky-bridge"]["build"]["args"], (
        "a non-dev render must not set the dev build-arg — the strict deps-layer "
        "`pip check` is the release build's gate"
    )


# ---------------------------------------------------------------------------
# Regression guards on invariants this task must not disturb.
# ---------------------------------------------------------------------------


def test_bridge_port_bind_stays_loopback_and_token_stays_fail_closed(
    rendered: dict[str, Any],
) -> None:
    bridge = rendered["services"]["bluesky-bridge"]
    assert bridge["ports"] == ["127.0.0.1:8090:8090"]
    assert bridge["environment"]["BLUESKY_LAUNCH_TOKEN"] == "${BLUESKY_LAUNCH_TOKEN}"


def test_template_renders_valid_yaml_in_every_gating_combination() -> None:
    """The conditional blocks multiply: a combination that renders broken YAML
    would only surface at `osprey up` on somebody's machine.

    The device file is its own axis, independent of every other one — it is
    staged from an authored document or derived from the limits database, so it
    can be present on a read-only bridge-only deploy and absent on a writable
    one with the VA co-deployed. Both keys the generator computes are typed on
    every combination, which is also what a real render always hands the
    template.
    """
    for tiled in (False, True):
        for writes in (False, True):
            for deployed in (["bluesky"], ["bluesky", "virtual_accelerator"]):
                for plan_dir in (None, "/opt/facility/plans"):
                    for devices in (False, True):
                        parsed = _render(
                            tiled_enabled=tiled,
                            writes_enabled=writes,
                            deployed_services=deployed,
                            plan_dir=plan_dir,
                            devices_present=devices,
                        )
                        assert "queueserver" in parsed["services"]
                        assert "bluesky-redis" in parsed["services"]
                        queueserver = parsed["services"]["queueserver"]
                        assert (
                            "BLUESKY_DEVICES_FILE" in (queueserver["environment"] or {})
                        ) is devices
                        assert (
                            any("bluesky_devices" in str(v) for v in queueserver["volumes"])
                        ) is devices
                        assert (
                            any("channel_limits" in str(v) for v in queueserver["volumes"])
                        ) is writes


# ---------------------------------------------------------------------------
# The two computed context keys, through the real generator.
#
# Everything above renders a hand-built context, which pins what the TEMPLATE
# does with the values but not that the values ever arrive. These go through
# `prepare_compose_files` — the production entry point — so the strings the
# assertions above assume are the strings a deployment actually gets, and a
# generator that stopped setting either key fails here rather than silently
# rendering a mount-less stack.
# ---------------------------------------------------------------------------

_LIMITS_RELPATH = "data/channel_limits.json"
_DEVICES_RELPATH = "data/bluesky_devices.yml"

#: A device document the worker's own loader accepts in full, so the staging
#: step copies it rather than refusing the render.
_DEVICES_DOCUMENT = """\
settables:
  - name: COR:H:01
    setpoint: COR:H:01:SP
    readback: COR:H:01:RB
readables:
  - name: BPM:01:X
    pv: BPM:01:X
"""


def _lane_config(config_dir: Path, *, writes_enabled: bool) -> dict:
    """The project config both entry points render from.

    One dict for both, because the ONLY thing the two entry points differ in is
    where the file holding it sits — which is the whole point of the pair.
    """
    return {
        "project_name": "lane-fixture",
        "build_dir": str(config_dir / "build") if config_dir.name != "build" else str(config_dir),
        "system": {"timezone": "UTC"},
        "deployment": {},
        "deployed_services": ["bluesky"],
        "services": {
            "bluesky": {
                "path": "./services/bluesky",
                "port": 8090,
                "devices_file": _DEVICES_RELPATH,
            }
        },
        "control_system": {
            "type": "epics",
            "writes_enabled": writes_enabled,
            "limits_checking": {"enabled": True, "database_path": _LIMITS_RELPATH},
        },
    }


def _bluesky_repo(root: Path, *, config_dir: Path, writes_enabled: bool) -> Path:
    """A deployment repo that renders the bluesky lane; returns its config path.

    *config_dir* is the entry point under test. The repo root is what ``osprey
    up`` loads for a repo-root deployment; ``<repo>/build`` is what it loads
    once a build's atomic swap has landed — and in that shape the repo root
    still holds the authored config and the service templates, which is why the
    root config is written either way.

    The limits database and the device document are staged beside the config
    being loaded, because that is the directory their relative paths are
    authored against (``_render_anchor_dir`` / ``resolve_limits_mount`` both
    anchor on ``config_dir``).
    """
    from ruamel.yaml import YAML

    from osprey.cli.build_injectors import _copy_service_templates

    yaml_rt = YAML()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "config.yml").open("w", encoding="utf-8") as handle:
        yaml_rt.dump(_lane_config(root, writes_enabled=writes_enabled), handle)

    # Reads the repo-root config to decide which service template directories
    # the repo needs, so it runs after that file exists.
    _copy_service_templates(root)

    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yml"
    if config_dir != root:
        with config_path.open("w", encoding="utf-8") as handle:
            yaml_rt.dump(_lane_config(config_dir, writes_enabled=writes_enabled), handle)

    for relative, contents in ((_LIMITS_RELPATH, "{}"), (_DEVICES_RELPATH, _DEVICES_DOCUMENT)):
        staged = config_dir / relative
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text(contents, encoding="utf-8")
    return config_path


def _rendered_lane(config_path: Path) -> dict:
    """Run the real render and parse the bluesky compose file it produced."""
    from osprey.deployment.compose_generator import prepare_compose_files

    _, compose_files = prepare_compose_files(str(config_path))
    lane = [path for path in compose_files if "bluesky" in path]
    assert lane, f"the bluesky lane rendered no compose file: {compose_files}"
    return yaml.safe_load(Path(lane[0]).read_text(encoding="utf-8"))


def test_a_repo_root_deployment_really_mounts_the_unprefixed_limits_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end for the commonest entry point: config at the repo root.

    The bind source is resolved by compose against the pinned project directory
    — the repo root — so a config read from there needs no prefix at all. This
    is the assertion that proves the strings the hand-built contexts above use
    are the strings the generator actually produces.
    """
    config_path = _bluesky_repo(tmp_path, config_dir=tmp_path, writes_enabled=True)

    monkeypatch.chdir(tmp_path)
    queueserver = _rendered_lane(config_path)["services"]["queueserver"]

    assert (
        f"{REPO_ROOT_LIMITS_MOUNT['source']}:{REPO_ROOT_LIMITS_MOUNT['target']}:ro"
        in queueserver["volumes"]
    )


def test_a_build_zone_deployment_really_mounts_the_build_prefixed_limits_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deploy-time shape: ``osprey up`` loads ``<repo>/build/config.yml``.

    Same configured value, same container target, a source one directory
    deeper — because the compose project directory stays the repo root while
    the config (and the data beside it) moved into the build zone. A render
    that kept the repo-root spelling here would bind a path that does not
    exist, which the container runtime silently creates as an empty directory.
    """
    config_path = _bluesky_repo(tmp_path, config_dir=tmp_path / "build", writes_enabled=True)

    monkeypatch.chdir(tmp_path)
    queueserver = _rendered_lane(config_path)["services"]["queueserver"]

    assert (
        f"{BUILD_ZONE_LIMITS_MOUNT['source']}:{BUILD_ZONE_LIMITS_MOUNT['target']}:ro"
        in queueserver["volumes"]
    )


def test_the_real_render_context_always_carries_a_boolean_bluesky_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``bluesky_devices`` is typed by the generator, not defaulted by the template.

    The template carries ``| default(false)`` as a belt for hand-built contexts,
    and a belt that is load-bearing is a bug waiting to happen: it would turn a
    generator that stopped staging — or stopped reporting that it staged — into
    a silently browse-only deployment instead of a failure. So the production
    render is asserted to hand the template a REAL boolean every time.
    """
    from osprey.deployment import compose_generator

    contexts: list[dict] = []
    real = compose_generator.render_template

    def recording(template_path, config, out_dir):
        # Keyed on the template, because the root services declaration renders
        # from the bare config and carries no staging decision at all — only
        # the per-service renders do.
        if "bluesky" in str(template_path):
            contexts.append(config)
        return real(template_path, config, out_dir)

    monkeypatch.setattr(compose_generator, "render_template", recording)

    config_path = _bluesky_repo(tmp_path, config_dir=tmp_path, writes_enabled=False)
    monkeypatch.chdir(tmp_path)
    rendered = _rendered_lane(config_path)

    assert contexts, "the bluesky lane rendered no context at all"
    for context in contexts:
        assert "bluesky_devices" in context, sorted(context)
        assert context["bluesky_devices"] is True, (
            "this repo authors a valid device file, so the staging step must "
            f"report that one landed: {context['bluesky_devices']!r}"
        )
    # And the flag is not bookkeeping: it is what put the mount in the file.
    assert DEVICES_MOUNT in rendered["services"]["queueserver"]["volumes"]
