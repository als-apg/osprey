"""The Bluesky plan-lane axis across the deploy surface.

A *lane* is a full Bluesky stack — bridge + RE Manager + its own Redis — bound
at render time to the control-system target it serves. Every project rendered
before this feature has exactly one, and the second lane is opt-in
(``bluesky.second_lane``). Four resources that were single-set are now
per-lane: the document-plane CURVE certificates, the launch token, Redis, and
the host-port map. Tiled is the one shared component and stays on lane 1.

Two claims are tested here, and the first one is the anchor:

1. **A single-lane deployment renders byte-for-byte what it rendered before.**
   Not "parses to the same YAML" — literally the same bytes. The goldens under
   ``goldens/bluesky_single_lane/`` were produced from the template as it stood
   before the lane axis existed, so a diff against them is a diff against
   history. Every existing project is single-lane, so this is the whole
   compatibility surface of the change.

2. **A two-lane deployment isolates the two lanes.** Separate bridges,
   managers, Redis instances, internal networks, CURVE certificate
   directories, launch tokens and host ports — because a shared resource on
   any one of those axes lets either lane's bridge drive the other lane's
   manager, which is exactly the confusion between "which machine am I talking
   to" that the run-time target switch exists to make explicit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

# Rooted at the templates/ PROJECT root, not services/, because service
# templates import the shared axis macros as "services/_*.j2" — the spelling
# compose_generator's own loader resolves. Same two-root loader as
# tests/cli/test_bluesky_compose_render.py, so both suites render the packaged
# template the way the deployment does.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATES_ROOT = _REPO_ROOT / "src" / "osprey" / "templates"
_LOADER_ROOTS = [str(_TEMPLATES_ROOT), str(_TEMPLATES_ROOT / "services")]
BLUESKY_TEMPLATE = "bluesky/docker-compose.yml.j2"

GOLDEN_DIR = Path(__file__).parent / "goldens" / "bluesky_single_lane"

#: The literal ``_inject_bluesky`` writes into the live-serving lane's config
#: block. Restated here as a PREFIX rather than imported whole: the test cares
#: that compose is handed a required variable (``:?``), not about the wording
#: of the operator message that follows it.
CA_NAME_SERVERS_REQUIRED_PREFIX = "${EPICS_CA_NAME_SERVERS:?"


def _image_defaults(project_name: str) -> dict[str, str]:
    """The image map ``_inject_project_metadata`` injects, for hand-built ctx.

    Taken from the production helper rather than restated, so these renders
    follow the registry and tag axes instead of pinning a name the generator
    may not produce any more.
    """
    from osprey.deployment.compose_generator import resolve_image_defaults

    return resolve_image_defaults({"project_name": project_name})


def _lane_block(
    port: int,
    *,
    tiled_enabled: bool = False,
    tiled_port: int | None = None,
    plan_dir: str | None = None,
    excluded_plans: str | None = None,
    env: list[str] | None = None,
    target: str | None = None,
    ca_name_servers: str | None = None,
) -> dict[str, Any]:
    """One ``services.<lane>`` block, spelled the way ``_inject_bluesky`` writes it.

    Keys absent from a single-lane render (``target``, ``ca_name_servers``) are
    omitted unless asked for — that omission is what the byte-identity claim
    rests on.
    """
    block: dict[str, Any] = {"path": "./services/bluesky", "port": port}
    if tiled_enabled:
        block["tiled_enabled"] = True
        block["tiled_port"] = tiled_port if tiled_port is not None else 8091
    if plan_dir is not None:
        block["plan_dir"] = plan_dir
    if excluded_plans is not None:
        block["excluded_plans"] = excluded_plans
    if env is not None:
        block["env"] = env
    if target is not None:
        block["target"] = target
    if ca_name_servers is not None:
        block["ca_name_servers"] = ca_name_servers
    return block


#: The finished channel-limits bind mount ``resolve_limits_mount`` computes for
#: a deployment whose config is read from the repo root. Both halves reach the
#: template as strings the generator already resolved, so a context that leaves
#: the key out is not a render any deployment can produce — a writable one can
#: never reach the template without it.
LIMITS_MOUNT: dict[str, str] = {
    "source": "./data/channel_limits.json",
    "target": "/app/project/data/channel_limits.json",
}

#: The one staged device document, and the bind that carries it. The source is
#: literal (the staging step owns the basename), which is what makes the
#: two-lane claim below checkable: both lanes render this same string.
DEVICES_FILE_TARGET = "/app/project/data/bluesky_devices.yml"
DEVICES_MOUNT = f"./build/services/bluesky/bluesky_devices.yml:{DEVICES_FILE_TARGET}:ro"


def _context(
    *,
    lanes: dict[str, dict[str, Any]],
    deployed_services: list[str],
    writes_enabled: bool = False,
    va_port: int = 5064,
    devices_present: bool = False,
) -> dict[str, Any]:
    """Mirror ``compose_generator.render_template``'s context contract.

    Two keys are computed by the generator rather than configured, and both are
    typed here for the same reason: a production render always carries them, so
    a context that omits one pins a render no deploy can reach.

    ``bluesky_devices`` is the real boolean ``_stage_bluesky_devices`` returns —
    ONE value for the whole file, not one per lane, because both lanes declare
    the same service ``path`` and therefore stage into one directory.
    ``limits_mount`` is injected whenever writes are enabled, which is exactly
    when the template mounts it.
    """
    services: dict[str, Any] = dict(lanes)
    services["virtual_accelerator"] = {"port": va_port}
    context: dict[str, Any] = {
        "osprey_labels": {
            "project_name": "proj",
            "project_root": "/tmp/proj",
            "repo_id": "abc123def456",
        },
        "osprey_images": _image_defaults("proj"),
        "osprey_version": "2026.8.1",
        "system": {"timezone": "UTC"},
        "deployment": {},
        "deployed_services": deployed_services,
        "control_system": {"writes_enabled": writes_enabled},
        "services": services,
        "bluesky_devices": devices_present,
    }
    if writes_enabled:
        context["limits_mount"] = LIMITS_MOUNT
    return context


def _render_text(context: dict[str, Any]) -> str:
    """Render the packaged bluesky compose template to raw text."""
    env = Environment(loader=FileSystemLoader(_LOADER_ROOTS), keep_trailing_newline=True)
    return env.get_template(BLUESKY_TEMPLATE).render(context)


def _render(context: dict[str, Any]) -> dict[str, Any]:
    """Render and parse the packaged bluesky compose template."""
    return yaml.safe_load(_render_text(context))


# ---------------------------------------------------------------------------
# The pinned single-lane contexts. Named, because the goldens are named for
# them and a regenerated golden has to come from the same context.
# ---------------------------------------------------------------------------


def _single_lane_contexts() -> dict[str, dict[str, Any]]:
    """Every single-lane shape whose rendered bytes are pinned.

    ``minimal`` is the plainest deploy the injector can produce — bridge only,
    no Tiled, no VA, reads only. ``full`` turns on every optional axis at once
    (co-deployed VA, Tiled, writes, a facility plan directory, exclusions and
    a host-env passthrough), because the branches those flags open are where a
    lane-parameterized template is most likely to drift.

    Both stage no device file. That is not an omission: these contexts are
    reused as the canonical single-lane render by
    ``tests/deployment/test_bluesky_substrate_env.py``, whose browse-only test
    renders them as-is and asserts the device wiring is absent. The staged
    shape is pinned there instead, beside the staging step that produces it —
    and as parsed YAML by the device tests in this module and in
    ``tests/cli/test_bluesky_compose_render.py``.
    """
    return {
        "minimal": _context(
            lanes={"bluesky": _lane_block(8090)},
            deployed_services=["bluesky"],
        ),
        "full": _context(
            lanes={
                "bluesky": _lane_block(
                    8090,
                    tiled_enabled=True,
                    plan_dir="/facility/plans",
                    excluded_plans="pkg.a:pkg.b",
                    env=["HTTP_PROXY", "NO_PROXY"],
                )
            },
            deployed_services=["bluesky", "virtual_accelerator"],
            writes_enabled=True,
        ),
    }


@pytest.mark.parametrize("name", sorted(_single_lane_contexts()))
def test_single_lane_render_is_byte_identical_to_the_pinned_shape(name: str) -> None:
    """A single-lane render must reproduce its pinned shape exactly.

    The goldens were first produced from the template BEFORE the lane axis was
    introduced, so what they pin is a before/after equality rather than a
    self-consistency check. Byte equality rather than parsed equality on
    purpose: a rendered compose file is also read by humans and diffed by
    operators, and a reshuffled-but-equivalent document is a change they have
    to review.

    **Update discipline** — a failure here means a template edit moved a
    single-lane render, which is never on its own a reason to hand-edit a
    golden. Regenerate the pair from the same contexts, in the SAME reviewed
    change as the template edit that moved them::

        PYTHONPATH=src ./.venv/bin/python tests/deployment/test_lane_compose.py

    then account for every changed byte. The device-file rewrite is the second
    deliberate move these carry: the three retired ``BLUESKY_EPICS`` passthrough
    variables left both containers, and the channel-limits bind became the pair
    of strings the generator computes host-side — the same file, spelled once
    at render time instead of twice in the template.
    """
    # Final-newline count is normalized on both sides: the repo's
    # end-of-file-fixer hook owns the goldens' trailing newline, which the
    # renderer does not reproduce. Every other byte still has to match.
    golden = (GOLDEN_DIR / f"{name}.yml").read_text(encoding="utf-8")
    rendered = _render_text(_single_lane_contexts()[name])
    assert rendered.rstrip("\n") + "\n" == golden.rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# The two-lane render
# ---------------------------------------------------------------------------

VA_BASELINE_LANES = {
    "bluesky": _lane_block(8090, tiled_enabled=True, target="va"),
    "bluesky_live": _lane_block(
        8190,
        target="live",
        ca_name_servers=f"{CA_NAME_SERVERS_REQUIRED_PREFIX}set it to <host>:<port>}}",
    ),
}


@pytest.fixture
def two_lane() -> dict[str, Any]:
    """A VA-baseline deployment with a live second lane, as the injector writes it.

    Lane 1 (``bluesky``) serves the deployment baseline — a co-deployed Virtual
    Accelerator — and owns Tiled; lane 2 (``bluesky_live``) serves the facility
    and carries the required-variable gateway address.
    """
    return _render(
        _context(
            lanes=VA_BASELINE_LANES,
            deployed_services=["bluesky", "bluesky_live", "virtual_accelerator"],
        )
    )


def _env(rendered: dict[str, Any], service: str) -> dict[str, Any]:
    return rendered["services"][service].get("environment") or {}


def test_each_lane_renders_its_own_bridge_manager_and_redis(two_lane: dict[str, Any]) -> None:
    """A lane is a whole stack, so two lanes are six containers, not four.

    Lane 1 keeps its historical service keys (``bluesky-bridge``,
    ``queueserver``, ``bluesky-redis``) because renaming them would recreate
    every existing project's containers for nothing.
    """
    assert set(two_lane["services"]) == {
        "bluesky-bridge",
        "queueserver",
        "bluesky-redis",
        "bluesky-live-bridge",
        "bluesky-live-queueserver",
        "bluesky-live-redis",
        "tiled",
    }


def test_exactly_one_tiled_and_it_belongs_to_lane_one(two_lane: dict[str, Any]) -> None:
    """Tiled is the one SHARED component: a catalog per lane, not a lane per catalog.

    Lane 2's config block carries no tiled keys at all (the injector omits
    them), so the second lane's containers must be silent about Tiled rather
    than pointing at lane 1's — a second writer into one catalog would merge
    two machines' run documents into one history.
    """
    assert [name for name in two_lane["services"] if name == "tiled"] == ["tiled"]
    assert two_lane["services"]["tiled"]["ports"] == ["127.0.0.1:8091:8000"]
    for service in ("bluesky-live-bridge", "bluesky-live-queueserver"):
        assert "BLUESKY_TILED_URI" not in _env(two_lane, service)
    assert _env(two_lane, "bluesky-bridge")["BLUESKY_TILED_URI"] == "http://tiled:8000"


def test_only_the_second_lanes_bridge_publishes_its_derived_port(
    two_lane: dict[str, Any],
) -> None:
    """Each lane publishes exactly one host port: its bridge's."""
    published = {
        name: service.get("ports")
        for name, service in two_lane["services"].items()
        if service.get("ports")
    }
    assert published == {
        "bluesky-bridge": ["127.0.0.1:8090:8090"],
        "bluesky-live-bridge": ["127.0.0.1:8190:8190"],
        "tiled": ["127.0.0.1:8091:8000"],
    }


def test_the_live_lane_addresses_a_required_gateway_variable(two_lane: dict[str, Any]) -> None:
    """`${VAR:?}`, never a bare passthrough, and never the VA's address.

    Compose interpolates an unset BARE reference to the empty string, so a
    live lane wired that way would come up looking healthy while searching for
    PVs at nowhere. Both of the lane's containers build devices against this
    address, so both must carry the refusing form.
    """
    for service in ("bluesky-live-bridge", "bluesky-live-queueserver"):
        addressing = _env(two_lane, service)["EPICS_CA_NAME_SERVERS"]
        assert addressing.startswith(CA_NAME_SERVERS_REQUIRED_PREFIX)
        assert "virtual-accelerator" not in addressing


def test_the_va_lane_keeps_the_co_deployed_accelerator_addressing(
    two_lane: dict[str, Any],
) -> None:
    """The baseline lane's CA wiring is unchanged: the VA container, by name."""
    for service in ("bluesky-bridge", "queueserver"):
        assert _env(two_lane, service)["EPICS_CA_NAME_SERVERS"] == "virtual-accelerator:5064"


def test_only_the_va_lane_waits_on_the_virtual_accelerator(two_lane: dict[str, Any]) -> None:
    """A lane that never talks to the VA has no reason to be ordered after it.

    Both containers of the lane, not just its bridge: the RunEngine worker
    lives in the manager, so the manager carries the same iocInit guard and
    therefore the same lane half of it.
    """
    for service in ("bluesky-bridge", "queueserver"):
        assert "virtual-accelerator" in two_lane["services"][service]["depends_on"], service
    for service in ("bluesky-live-bridge", "bluesky-live-queueserver"):
        assert "virtual-accelerator" not in two_lane["services"][service]["depends_on"], service


def test_each_lane_carries_its_own_launch_token_variable(two_lane: dict[str, Any]) -> None:
    """The token ARMS a launch, so a shared one would let an approval be replayed.

    The name inside the container is the same for both — the image is
    lane-agnostic — and only the host variable filling it differs.
    """
    assert _env(two_lane, "bluesky-bridge")["BLUESKY_LAUNCH_TOKEN"] == "${BLUESKY_LAUNCH_TOKEN}"
    assert (
        _env(two_lane, "bluesky-live-bridge")["BLUESKY_LAUNCH_TOKEN"]
        == "${BLUESKY_LIVE_LAUNCH_TOKEN}"
    )


def test_each_lane_carries_its_own_control_socket_keypair(two_lane: dict[str, Any]) -> None:
    """One keypair across both lanes would let either bridge drive either queue.

    Both halves keep their fail-closed ``:?`` guard on both lanes: an empty
    value runs the RE manager's control socket in plaintext, which is not a
    supported mode on any lane.
    """
    lane_one = _env(two_lane, "queueserver")
    lane_two = _env(two_lane, "bluesky-live-queueserver")
    assert lane_one["QSERVER_ZMQ_PRIVATE_KEY_FOR_SERVER"].startswith(
        "${BLUESKY_QSERVER_ZMQ_PRIVATE_KEY:?"
    )
    assert lane_two["QSERVER_ZMQ_PRIVATE_KEY_FOR_SERVER"].startswith(
        "${BLUESKY_LIVE_QSERVER_ZMQ_PRIVATE_KEY:?"
    )
    assert _env(two_lane, "bluesky-live-bridge")["QSERVER_ZMQ_PUBLIC_KEY"].startswith(
        "${BLUESKY_LIVE_QSERVER_ZMQ_PUBLIC_KEY:?"
    )


def test_each_lane_mounts_its_own_curve_certificate_directory(two_lane: dict[str, Any]) -> None:
    """Distinct directories, because one shared pair authenticates either publisher.

    With a single set, a plan running on one machine could inject run
    documents into the other machine's history — the proxy would accept it,
    since the credential is the same.
    """
    mounts = {
        service: [v for v in two_lane["services"][service]["volumes"] if "/app/curve" in v]
        for service in (
            "bluesky-bridge",
            "queueserver",
            "bluesky-live-bridge",
            "bluesky-live-queueserver",
        )
    }
    assert mounts == {
        "bluesky-bridge": ["./data/.runtime/bluesky_curve/bridge:/app/curve:ro"],
        "queueserver": ["./data/.runtime/bluesky_curve/queueserver:/app/curve:ro"],
        "bluesky-live-bridge": ["./data/.runtime/bluesky_live_curve/bridge:/app/curve:ro"],
        "bluesky-live-queueserver": [
            "./data/.runtime/bluesky_live_curve/queueserver:/app/curve:ro"
        ],
    }


@pytest.fixture
def two_lane_with_devices() -> dict[str, Any]:
    """The same two-lane deployment, rendered after a device file was staged.

    ``bluesky_devices`` is one boolean for the whole render — the staging step
    runs once per lane against the same service directory and reaches the same
    decision — so there is no per-lane variant of this fixture to write.
    """
    return _render(
        _context(
            lanes=VA_BASELINE_LANES,
            deployed_services=["bluesky", "bluesky_live", "virtual_accelerator"],
            devices_present=True,
        )
    )


def test_both_lanes_read_the_one_staged_device_file(
    two_lane_with_devices: dict[str, Any],
) -> None:
    """The device document is SHARED, unlike every per-lane resource above.

    Both lanes declare the same service ``path``, so the staging step writes
    one file into one build context; a mount naming a per-lane path would point
    the second lane at a file nothing ever writes, and the worker would fail
    its own load rather than come up browse-only. Any split between the two
    machines' device sets therefore lives INSIDE the document, not in this
    mount.
    """
    managers = ("queueserver", "bluesky-live-queueserver")
    mounts = {
        manager: [
            volume
            for volume in two_lane_with_devices["services"][manager]["volumes"]
            if "bluesky_devices" in str(volume)
        ]
        for manager in managers
    }
    assert mounts == {
        "queueserver": [DEVICES_MOUNT],
        "bluesky-live-queueserver": [DEVICES_MOUNT],
    }

    named = {
        manager: _env(two_lane_with_devices, manager).get("BLUESKY_DEVICES_FILE")
        for manager in managers
    }
    assert named == {
        "queueserver": DEVICES_FILE_TARGET,
        "bluesky-live-queueserver": DEVICES_FILE_TARGET,
    }


def test_neither_lanes_bridge_is_given_the_device_file(
    two_lane_with_devices: dict[str, Any],
) -> None:
    """Devices are built by the managers; a bridge is a facade over one."""
    for bridge in ("bluesky-bridge", "bluesky-live-bridge"):
        service = two_lane_with_devices["services"][bridge]
        assert "BLUESKY_DEVICES_FILE" not in (service.get("environment") or {}), bridge
        assert not any("bluesky_devices" in str(v) for v in service["volumes"]), bridge


def test_no_lane_carries_the_device_mount_when_nothing_was_staged(
    two_lane: dict[str, Any],
) -> None:
    """Browse-only is fail-closed and applies to the whole render, both lanes."""
    for service in two_lane["services"].values():
        assert not any("bluesky_devices" in str(v) for v in service.get("volumes") or [])
        assert "BLUESKY_DEVICES_FILE" not in (service.get("environment") or {})


def test_each_lane_gets_its_own_redis_volume_and_internal_network(
    two_lane: dict[str, Any],
) -> None:
    """Queue state and reachability are both per lane.

    A shared Redis would merge two machines' queues into one keyspace; a
    shared internal network would put either lane's bridge within reach of the
    other lane's manager control socket.
    """
    assert two_lane["services"]["bluesky-redis"]["volumes"] == ["bluesky_queueserver_redis:/data"]
    assert two_lane["services"]["bluesky-live-redis"]["volumes"] == [
        "bluesky_live_queueserver_redis:/data"
    ]
    assert set(two_lane["volumes"]) == {
        "bluesky_queueserver_redis",
        "bluesky_live_queueserver_redis",
        "bluesky_tiled_catalog",
    }
    assert set(two_lane["networks"]) == {
        "osprey-network",
        "bluesky-internal",
        "bluesky-live-internal",
    }
    assert two_lane["services"]["bluesky-redis"]["networks"] == ["bluesky-internal"]
    assert two_lane["services"]["bluesky-live-redis"]["networks"] == ["bluesky-live-internal"]


def test_each_lane_manager_is_reached_only_by_its_own_bridge(two_lane: dict[str, Any]) -> None:
    """Control address, publish address and Redis address all stay inside the lane."""
    assert _env(two_lane, "bluesky-bridge")["QSERVER_ZMQ_CONTROL_ADDRESS"] == (
        "tcp://queueserver:60615"
    )
    assert _env(two_lane, "bluesky-live-bridge")["QSERVER_ZMQ_CONTROL_ADDRESS"] == (
        "tcp://bluesky-live-queueserver:60615"
    )
    assert _env(two_lane, "bluesky-live-queueserver")["BLUESKY_ZMQ_PUBLISH_ADDR"] == (
        "tcp://bluesky-live-bridge:5567"
    )
    manager_argv = " ".join(two_lane["services"]["bluesky-live-queueserver"]["command"])
    assert "--redis-addr bluesky-live-redis:6379" in manager_argv


def test_each_lane_is_told_which_lane_it_is(two_lane: dict[str, Any]) -> None:
    """Both bridges mount ONE config.yml, so identity has to arrive out of band.

    The value is the lane's service key, which is what a reader looks
    ``services.<lane>.target`` up under.
    """
    assert _env(two_lane, "bluesky-bridge")["OSPREY_BLUESKY_LANE"] == "bluesky"
    assert _env(two_lane, "queueserver")["OSPREY_BLUESKY_LANE"] == "bluesky"
    assert _env(two_lane, "bluesky-live-bridge")["OSPREY_BLUESKY_LANE"] == "bluesky_live"
    assert _env(two_lane, "bluesky-live-queueserver")["OSPREY_BLUESKY_LANE"] == "bluesky_live"


def test_a_single_lane_deployment_is_told_nothing_and_defaults() -> None:
    """Omitted rather than emitted as "bluesky": there is one block to read.

    Omission is also what keeps the single-lane render byte-identical, which
    the golden test above is the full statement of.
    """
    rendered = _render(_single_lane_contexts()["minimal"])
    assert "OSPREY_BLUESKY_LANE" not in _env(rendered, "bluesky-bridge")
    assert "OSPREY_BLUESKY_LANE" not in _env(rendered, "queueserver")


def test_only_lane_one_builds_the_shared_image(two_lane: dict[str, Any]) -> None:
    """Two services building one tag race each other — the queueserver's own rule.

    Every lane runs the same image, and compose builds every buildable service
    before creating any container, so the second lane finds the tag already on
    the host.
    """
    assert "build" in two_lane["services"]["bluesky-bridge"]
    assert "build" not in two_lane["services"]["bluesky-live-bridge"]
    assert (
        two_lane["services"]["bluesky-live-bridge"]["image"]
        == (two_lane["services"]["bluesky-bridge"]["image"])
    )


def test_a_va_second_lane_is_named_for_its_target_too() -> None:
    """A live BASELINE puts the VA on lane 2, and the naming follows the target.

    Nothing about the axis is "lane 1 is the VA": the lane keeps the key of the
    machine it serves, whichever way round the deployment is.
    """
    rendered = _render(
        _context(
            lanes={
                "bluesky": _lane_block(
                    8090,
                    target="live",
                    ca_name_servers=f"{CA_NAME_SERVERS_REQUIRED_PREFIX}set it}}",
                ),
                "bluesky_va": _lane_block(8190, target="va"),
            },
            deployed_services=["bluesky", "bluesky_va", "virtual_accelerator"],
        )
    )
    assert {"bluesky-va-bridge", "bluesky-va-queueserver", "bluesky-va-redis"} <= set(
        rendered["services"]
    )
    assert _env(rendered, "bluesky-va-bridge")["EPICS_CA_NAME_SERVERS"] == (
        "virtual-accelerator:5064"
    )
    assert _env(rendered, "bluesky-bridge")["EPICS_CA_NAME_SERVERS"].startswith(
        CA_NAME_SERVERS_REQUIRED_PREFIX
    )
    assert (
        _env(rendered, "bluesky-va-bridge")["BLUESKY_LAUNCH_TOKEN"] == "${BLUESKY_VA_LAUNCH_TOKEN}"
    )


def test_a_lane_block_present_but_undeployed_renders_nothing() -> None:
    """Membership in ``deployed_services`` is what conjures a stack, not a config block."""
    rendered = _render(
        _context(
            lanes={
                "bluesky": _lane_block(8090),
                "bluesky_va": _lane_block(8190, target="va"),
            },
            deployed_services=["bluesky"],
        )
    )
    assert set(rendered["services"]) == {"bluesky-bridge", "queueserver", "bluesky-redis"}


# ---------------------------------------------------------------------------
# The host-port preflight
# ---------------------------------------------------------------------------


def test_the_port_preflight_names_the_key_that_moves_each_lane(tmp_path: Path) -> None:
    """Every published port a two-lane deploy renders is attributed to its own key.

    The preflight's whole value is naming the config key to change, so a
    second lane whose bridge falls back to the generic
    ``services.bluesky-live-bridge.port`` remedy would send an operator to a
    key that does not exist.
    """
    from osprey.deployment.host_ports import _remedy_for_service, parse_host_port_bindings

    compose = tmp_path / "docker-compose.yml"
    compose.write_text(
        _render_text(
            _context(
                lanes=VA_BASELINE_LANES,
                deployed_services=["bluesky", "bluesky_live", "virtual_accelerator"],
            )
        ),
        encoding="utf-8",
    )
    bindings = parse_host_port_bindings([compose])

    assert {(b.service, b.host_port) for b in bindings} == {
        ("bluesky-bridge", 8090),
        ("bluesky-live-bridge", 8190),
        ("tiled", 8091),
    }
    assert {b.service: _remedy_for_service(b.service) for b in bindings} == {
        "bluesky-bridge": "services.bluesky.port",
        "bluesky-live-bridge": "services.bluesky_live.port",
        "tiled": "services.bluesky.tiled_port",
    }


def test_the_other_second_lane_spelling_is_attributed_too() -> None:
    """Which lane key exists depends on the baseline, so both are mapped."""
    from osprey.deployment.host_ports import _remedy_for_service

    assert _remedy_for_service("bluesky-va-bridge") == "services.bluesky_va.port"


# ---------------------------------------------------------------------------
# Per-lane deploy-time resources (`osprey up`)
# ---------------------------------------------------------------------------

TWO_LANE_CONFIG = {"deployed_services": ["bluesky", "bluesky_live", "virtual_accelerator"]}
ONE_LANE_CONFIG = {"deployed_services": ["bluesky", "virtual_accelerator"]}


@pytest.fixture
def env_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project ``.env`` in an isolated project dir, with a clean process env.

    Every per-lane variable these provisioners read is cleared, so a
    developer's own exported values cannot decide the result.
    """
    from osprey.bluesky_bridge_connection import LANE_KEYS, lane_env_prefix

    for lane_key in LANE_KEYS:
        prefix = lane_env_prefix(lane_key)
        for suffix in (
            "_QSERVER_ZMQ_PRIVATE_KEY",
            "_QSERVER_ZMQ_PUBLIC_KEY",
            "_EPICS_SUBSTRATE",
            "_EPICS_SETPOINTS",
            "_EPICS_READBACKS",
            "_LAUNCH_TOKEN",
            "_BRIDGE_URL",
        ):
            monkeypatch.delenv(f"{prefix}{suffix}", raising=False)
    return tmp_path / ".env"


def _dotenv(env_path: Path) -> dict[str, str]:
    from osprey.utils.dotenv import parse_dotenv_file

    return parse_dotenv_file(env_path) if env_path.is_file() else {}


def test_lane_keys_come_from_deployed_services_in_render_order() -> None:
    """A block in config.yml that is not deployed provisions nothing."""
    from osprey.deployment.container_lifecycle import _bluesky_lane_keys

    assert _bluesky_lane_keys(TWO_LANE_CONFIG) == ["bluesky", "bluesky_live"]
    assert _bluesky_lane_keys(ONE_LANE_CONFIG) == ["bluesky"]
    assert _bluesky_lane_keys({"deployed_services": ["postgresql"]}) == []
    assert _bluesky_lane_keys({"deployed_services": ["bluesky", "bluesky_va"]}) == [
        "bluesky",
        "bluesky_va",
    ]


def test_each_lane_gets_its_own_curve_certificate_set(env_path: Path) -> None:
    """Two complete, unrelated sets — a shared pair authenticates either publisher.

    The negative claim is the load-bearing one: lane 2's publisher secret must
    not be the credential lane 1's proxy accepts.
    """
    from osprey.deployment.container_lifecycle import (
        _bluesky_curve_paths,
        _ensure_bluesky_document_plane_certs,
    )

    _ensure_bluesky_document_plane_certs(TWO_LANE_CONFIG, env_path=env_path)

    one = _bluesky_curve_paths(env_path.parent, "bluesky")
    two = _bluesky_curve_paths(env_path.parent, "bluesky_live")
    assert (
        one["bridge"].relative_to(env_path.parent).as_posix()
        == "data/.runtime/bluesky_curve/bridge"
    )
    assert (
        two["bridge"].relative_to(env_path.parent).as_posix()
        == "data/.runtime/bluesky_live_curve/bridge"
    )
    for role in ("proxy_secret", "publisher_secret", "proxy_public", "publisher_public"):
        assert one[role].is_file() and two[role].is_file(), role
        assert one[role].read_bytes() != two[role].read_bytes(), role


def test_a_single_lane_deploy_provisions_only_the_historical_directory(
    env_path: Path,
) -> None:
    """No second lane, no second certificate directory to explain to an operator."""
    from osprey.deployment.container_lifecycle import _ensure_bluesky_document_plane_certs

    _ensure_bluesky_document_plane_certs(ONE_LANE_CONFIG, env_path=env_path)

    assert (env_path.parent / "data" / ".runtime" / "bluesky_curve").is_dir()
    assert not (env_path.parent / "data" / ".runtime" / "bluesky_live_curve").exists()
    assert not (env_path.parent / "data" / ".runtime" / "bluesky_va_curve").exists()


def test_each_lane_gets_its_own_control_socket_keypair(env_path: Path) -> None:
    """Four values, two matched pairs, and the pairs must not be each other's."""
    import zmq

    from osprey.deployment.container_lifecycle import _ensure_bluesky_control_plane_keys

    _ensure_bluesky_control_plane_keys(TWO_LANE_CONFIG, env_path=env_path)
    written = _dotenv(env_path)

    for private_var, public_var in (
        ("BLUESKY_QSERVER_ZMQ_PRIVATE_KEY", "BLUESKY_QSERVER_ZMQ_PUBLIC_KEY"),
        ("BLUESKY_LIVE_QSERVER_ZMQ_PRIVATE_KEY", "BLUESKY_LIVE_QSERVER_ZMQ_PUBLIC_KEY"),
    ):
        assert written[private_var] and written[public_var]
        assert zmq.curve_public(written[private_var].encode()).decode() == written[public_var]
    assert (
        written["BLUESKY_QSERVER_ZMQ_PRIVATE_KEY"]
        != (written["BLUESKY_LIVE_QSERVER_ZMQ_PRIVATE_KEY"])
    )


def test_a_single_lane_deploy_mints_only_the_historical_keypair(env_path: Path) -> None:
    """Nothing new appears in an existing project's ``.env``."""
    from osprey.deployment.container_lifecycle import _ensure_bluesky_control_plane_keys

    _ensure_bluesky_control_plane_keys(ONE_LANE_CONFIG, env_path=env_path)

    assert set(_dotenv(env_path)) == {
        "BLUESKY_QSERVER_ZMQ_PRIVATE_KEY",
        "BLUESKY_QSERVER_ZMQ_PUBLIC_KEY",
    }


def test_each_lane_declares_its_own_launch_token(env_path: Path) -> None:
    """The token map is what ``_ensure_service_tokens`` mints from."""
    from osprey.deployment.container_lifecycle import _SERVICE_TOKEN_VARS

    assert _SERVICE_TOKEN_VARS["bluesky"] == ("BLUESKY_LAUNCH_TOKEN", "BLUESKY_TILED_API_KEY")
    assert _SERVICE_TOKEN_VARS["bluesky_va"] == ("BLUESKY_VA_LAUNCH_TOKEN",)
    assert _SERVICE_TOKEN_VARS["bluesky_live"] == ("BLUESKY_LIVE_LAUNCH_TOKEN",)


def test_one_template_serving_two_lanes_is_passed_to_compose_once() -> None:
    """Both lanes declare the same ``path``, so the lookup reports the file twice.

    Compose would merge a document with itself harmlessly, but passing it twice
    is a claim the deploy does not mean to make and doubles it up in every
    listing that echoes the file list.
    """
    from osprey.deployment.container_lifecycle import _dedupe_compose_files

    assert _dedupe_compose_files(
        [
            "build/services/docker-compose.yml",
            "build/services/bluesky/docker-compose.yml",
            "build/services/bluesky/docker-compose.yml",
            "build/services/virtual_accelerator/docker-compose.yml",
        ]
    ) == [
        "build/services/docker-compose.yml",
        "build/services/bluesky/docker-compose.yml",
        "build/services/virtual_accelerator/docker-compose.yml",
    ]


# ---------------------------------------------------------------------------
# Host-side addressing: which lane's bridge does a caller actually reach
# ---------------------------------------------------------------------------


@pytest.fixture
def rendered_config(monkeypatch: pytest.MonkeyPatch):
    """Patch the config the connection resolvers read, and return the setter."""
    from osprey.utils import workspace

    def _set(config: dict[str, Any]) -> None:
        monkeypatch.setattr(workspace, "load_osprey_config", lambda *a, **kw: config)

    _set({})
    return _set


def test_a_lane_unaware_caller_resolves_exactly_what_it_always_did(
    rendered_config, env_path: Path
) -> None:
    """Lane 1 is the default on every entry point, spelled as it always was."""
    from osprey.bluesky_bridge_connection import (
        DEFAULT_BRIDGE_URL,
        resolve_bridge_url,
        resolve_launch_token,
    )

    rendered_config({"bluesky": {"bridge_url": "http://bridge.example:8090/"}})
    assert resolve_bridge_url() == "http://bridge.example:8090"

    rendered_config({})
    assert resolve_bridge_url() == DEFAULT_BRIDGE_URL

    rendered_config({"bluesky": {"launch_token": "dev-token"}})
    assert resolve_launch_token() == "dev-token"


def test_each_lane_resolves_its_own_bridge_and_token(
    rendered_config, env_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second lane is addressed at its own port under its own token.

    The port comes from the lane's own service block rather than from a second
    ``bluesky.bridge_url`` key, because the build derives it and an operator
    would otherwise have to keep two values in step.
    """
    from osprey.bluesky_bridge_connection import resolve_bridge_url, resolve_launch_token

    rendered_config(
        {
            "bluesky": {"lane_launch_tokens": {"bluesky_live": "live-token"}},
            "services": {"bluesky_live": {"port": 8190}},
        }
    )
    assert resolve_bridge_url("bluesky_live") == "http://127.0.0.1:8190"
    assert resolve_launch_token("bluesky_live") == "live-token"

    monkeypatch.setenv("BLUESKY_LIVE_BRIDGE_URL", "http://elsewhere:9000/")
    monkeypatch.setenv("BLUESKY_LIVE_LAUNCH_TOKEN", "minted")
    assert resolve_bridge_url("bluesky_live") == "http://elsewhere:9000"
    assert resolve_launch_token("bluesky_live") == "minted"


def test_an_unrendered_lane_is_refused_rather_than_resolved_to_lane_one(
    rendered_config, env_path: Path
) -> None:
    """The fallback this refusal replaces is the wrong-machine bug itself.

    Quietly answering with lane 1 would send a plan to a bridge bound to a
    different machine than the caller named — recoverable only by noticing
    afterwards that it ran somewhere else.
    """
    from osprey.bluesky_bridge_connection import UnknownBlueskyLaneError, resolve_bridge_url

    rendered_config({"services": {}})
    with pytest.raises(UnknownBlueskyLaneError):
        resolve_bridge_url("bluesky_live")
    with pytest.raises(UnknownBlueskyLaneError):
        resolve_bridge_url("lane-2")


def test_the_env_prefix_matches_what_the_deploy_mints_and_compose_expands() -> None:
    """Three spellings of one contract: the mint, the template and the resolver."""
    from osprey.bluesky_bridge_connection import lane_env_prefix
    from osprey.deployment.container_lifecycle import (
        _SERVICE_TOKEN_VARS,
        _qserver_zmq_key_vars,
    )

    for lane_key in ("bluesky", "bluesky_va", "bluesky_live"):
        prefix = lane_env_prefix(lane_key)
        assert _SERVICE_TOKEN_VARS[lane_key][0] == f"{prefix}_LAUNCH_TOKEN"
        assert _qserver_zmq_key_vars(lane_key) == (
            f"{prefix}_QSERVER_ZMQ_PRIVATE_KEY",
            f"{prefix}_QSERVER_ZMQ_PUBLIC_KEY",
        )


def _fake_request(bridge_url: str, bridge_urls: dict[str, str] | None = None):
    """A stand-in for the sidecar's request, carrying only the state it reads."""
    from types import SimpleNamespace

    state = SimpleNamespace(bridge_url=bridge_url)
    if bridge_urls is not None:
        state.bridge_urls = bridge_urls
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_the_read_proxy_relays_lane_one_when_no_lane_is_named() -> None:
    """The single-lane sidecar publishes one URL and is asked no lane."""
    from osprey.interfaces.bluesky_web.read_proxy import resolve_lane_bridge_url

    request = _fake_request("http://bridge:8090/")
    assert resolve_lane_bridge_url(request, None) == "http://bridge:8090"
    assert resolve_lane_bridge_url(request, "bluesky") == "http://bridge:8090"


def test_an_empty_lane_parameter_names_no_lane_rather_than_a_missing_one() -> None:
    """``?lane=`` behaves as ``?lane`` omitted, not as a lane called "".

    Refusing the empty form would 404 a request whose bare form is served — a
    distinction the caller cannot see and nothing here means to draw.
    """
    from osprey.interfaces.bluesky_web.read_proxy import resolve_lane_bridge_url

    assert resolve_lane_bridge_url(_fake_request("http://bridge:8090"), "") == "http://bridge:8090"


def test_the_read_proxy_relays_each_lane_to_its_own_bridge() -> None:
    from osprey.interfaces.bluesky_web.read_proxy import resolve_lane_bridge_url

    request = _fake_request("http://bridge:8090", {"bluesky_live": "http://bridge-live:8190/"})
    assert resolve_lane_bridge_url(request, "bluesky_live") == "http://bridge-live:8190"


def test_the_read_proxy_refuses_a_lane_it_does_not_serve() -> None:
    """``None`` here becomes a 404 — never a relay from the other lane.

    A run listing labelled with the wrong machine is worse than no listing.
    """
    from osprey.interfaces.bluesky_web.read_proxy import resolve_lane_bridge_url

    assert resolve_lane_bridge_url(_fake_request("http://bridge:8090"), "bluesky_live") is None
    assert (
        resolve_lane_bridge_url(
            _fake_request("http://bridge:8090", {"bluesky_va": "http://x:1"}), "bluesky_live"
        )
        is None
    )


def _regenerate() -> None:
    """Overwrite the single-lane goldens from today's template.

    Rendered from :func:`_single_lane_contexts`, the same contexts the pinned
    test renders, so a regenerated golden can only differ where the template
    does. See that test's update discipline before running this.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, context in sorted(_single_lane_contexts().items()):
        path = GOLDEN_DIR / f"{name}.yml"
        # The repo's end-of-file-fixer owns the trailing newline the renderer
        # does not emit, and the pinned test normalizes it on both sides.
        path.write_text(_render_text(context).rstrip("\n") + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    _regenerate()
