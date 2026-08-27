"""The Virtual Accelerator instance axis across the compose render.

An *instance* is one PyAT-backed EPICS soft-IOC container. Every project
rendered before this feature has exactly one, and the second is opt-in
(``virtual_accelerator.live_standin``), which makes the build write a
``services.live_standin`` block beside ``services.virtual_accelerator`` and
append ``live_standin`` to ``deployed_services``. Three things are per
instance: the compose service key (and so the in-network CA address), the
published port, and the BPM-offset perturbation that makes the stand-in read
differently from the machine it stands in for.

Two claims are tested here, and the first one is the anchor:

1. **A single-instance deployment renders byte-for-byte what it rendered
   before.** Not "parses to the same YAML" — literally the same bytes. The
   goldens under ``goldens/va_single_instance/`` were produced from the
   template as it stood before the instance axis existed, so a diff against
   them is a diff against history. Every existing project is single-instance,
   so this is the whole compatibility surface of the change.

2. **A two-instance deployment separates the two machines.** Distinct service
   keys, container names, published ports and CA server ports, one build, and
   a stand-in whose readout perturbation comes from its own host variable —
   because an operator, a scenario and the host-port preflight all have to be
   able to say which of the two machines they mean.

The scenario mounts are the deliberate exception: both instances read the same
model and the same active-scenario state, which is what lets one
``osprey sim apply`` be observed on both machines at once.
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
# tests/deployment/test_lane_compose.py, so both suites render the packaged
# template the way the deployment does.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATES_ROOT = _REPO_ROOT / "src" / "osprey" / "templates"
_LOADER_ROOTS = [str(_TEMPLATES_ROOT), str(_TEMPLATES_ROOT / "services")]
VA_TEMPLATE = "virtual_accelerator/docker-compose.yml.j2"

GOLDEN_DIR = Path(__file__).parent / "goldens" / "va_single_instance"

#: The finished scenario-state bind source ``_inject_project_metadata``
#: computes for a project on the default agent-data root. Spelled here as the
#: string the template consumes rather than derived, for the same reason
#: ``test_lane_compose``'s limits mount is: the generator resolves it host-side
#: and the template only ever sees the answer.
STATE_MOUNT_SOURCE = "./var/agent_data/simulation"


def _image_defaults(project_name: str) -> dict[str, str]:
    """The image map ``_inject_project_metadata`` injects, for hand-built ctx.

    Taken from the production helper rather than restated, so these renders
    follow the registry and tag axes instead of pinning a name the generator
    may not produce any more.
    """
    from osprey.deployment.compose_generator import resolve_image_defaults

    return resolve_image_defaults({"project_name": project_name})


def _instance_block(
    port: int,
    *,
    image: str | None = None,
    env: list[str] | None = None,
) -> dict[str, Any]:
    """One ``services.<instance>`` block, as the build writes it.

    Both instances declare the same service ``path``: the stand-in is the same
    IOC image serving a perturbed copy of the same machine, so there is one
    service directory and no second image to build.
    """
    block: dict[str, Any] = {"path": "./services/virtual_accelerator", "port": port}
    if image is not None:
        block["image"] = image
    if env is not None:
        block["env"] = env
    return block


def _context(
    *,
    instances: dict[str, dict[str, Any]],
    deployed_services: list[str],
    dev_mode: bool | None = None,
    standin_bpm_errors_default: str | None = None,
) -> dict[str, Any]:
    """Mirror ``compose_generator.render_template``'s context contract.

    ``osprey_state_mount_source`` is computed by the generator rather than
    configured, and is typed here for the reason every such key is: a
    production render always carries it, so a context that omits one pins a
    render no deploy can reach.

    ``dev_mode`` is omitted unless asked for, and so is
    ``standin_bpm_errors_default`` — the template defaults the latter to the
    empty string precisely so a context assembled before that key existed still
    renders.
    """
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
        "services": dict(instances),
        "osprey_state_mount_source": STATE_MOUNT_SOURCE,
    }
    if dev_mode is not None:
        context["dev_mode"] = dev_mode
    if standin_bpm_errors_default is not None:
        context["standin_bpm_errors_default"] = standin_bpm_errors_default
    return context


def _render_text(context: dict[str, Any]) -> str:
    """Render the packaged VA compose template to raw text."""
    env = Environment(loader=FileSystemLoader(_LOADER_ROOTS), keep_trailing_newline=True)
    return env.get_template(VA_TEMPLATE).render(context)


def _render(context: dict[str, Any]) -> dict[str, Any]:
    """Render and parse the packaged VA compose template."""
    return yaml.safe_load(_render_text(context))


# ---------------------------------------------------------------------------
# The pinned single-instance contexts. Named, because the goldens are named for
# them and a regenerated golden has to come from the same context.
# ---------------------------------------------------------------------------


def _single_instance_contexts() -> dict[str, dict[str, Any]]:
    """Every single-instance shape whose rendered bytes are pinned.

    ``minimal`` is the plainest deploy the build can produce — the historical
    block with nothing but its port. ``overridden`` turns on the axes that open
    branches in this file at once (a non-default port, a pinned image, a
    ``--dev`` build and a host-env passthrough), because those are where a
    template parameterized over instances is most likely to drift.
    """
    return {
        "minimal": _context(
            instances={"virtual_accelerator": _instance_block(5064)},
            deployed_services=["virtual_accelerator"],
        ),
        "overridden": _context(
            instances={
                "virtual_accelerator": _instance_block(
                    5065,
                    image="my-registry/osprey-va:dev",
                    env=["HTTP_PROXY", "NO_PROXY"],
                )
            },
            deployed_services=["virtual_accelerator"],
            dev_mode=True,
        ),
    }


@pytest.mark.parametrize("name", sorted(_single_instance_contexts()))
def test_va_single_instance_render_is_byte_identical_to_the_pinned_shape(name: str) -> None:
    """A single-instance render must reproduce its pinned shape exactly.

    The goldens were first produced from the template BEFORE the instance axis
    was introduced, so what they pin is a before/after equality rather than a
    self-consistency check. Byte equality rather than parsed equality on
    purpose: a rendered compose file is also read by humans and diffed by
    operators, and a reshuffled-but-equivalent document is a change they have
    to review.

    **Update discipline** — a failure here means a template edit moved a
    single-instance render, which is never on its own a reason to hand-edit a
    golden. Regenerate the pair from the same contexts, in the SAME reviewed
    change as the template edit that moved them::

        PYTHONPATH=src ./.venv/bin/python tests/deployment/test_va_compose_instances.py

    then account for every changed byte.
    """
    # Final-newline count is normalized on both sides: the repo's
    # end-of-file-fixer hook owns the goldens' trailing newline, which the
    # renderer does not reproduce. Every other byte still has to match.
    golden = (GOLDEN_DIR / f"{name}.yml").read_text(encoding="utf-8")
    rendered = _render_text(_single_instance_contexts()[name])
    assert rendered.rstrip("\n") + "\n" == golden.rstrip("\n") + "\n"


def test_va_compose_undeployed_standin_block_still_renders_one_instance() -> None:
    """A ``services.live_standin`` block that was never deployed conjures nothing.

    Membership in ``deployed_services`` is the gate, not the presence of a
    config block — the same rule every other service here is gated on. Pinned
    against the ``minimal`` golden rather than by counting services, because
    "renders what it always did" is the whole claim.
    """
    golden = (GOLDEN_DIR / "minimal.yml").read_text(encoding="utf-8")
    rendered = _render_text(
        _context(
            instances={
                "virtual_accelerator": _instance_block(5064),
                "live_standin": _instance_block(5074),
            },
            deployed_services=["virtual_accelerator"],
        )
    )
    assert rendered.rstrip("\n") + "\n" == golden.rstrip("\n") + "\n"


def test_va_compose_null_instance_block_still_defaults_the_port() -> None:
    """A ``virtual_accelerator:`` key with no value renders on 5064.

    The instance axis reads the port off the loop's own ``svc`` binding in
    place of the longer ``(services.virtual_accelerator | default({})).port``
    the single-instance template spelled at each of its three sites. That
    substitution is only safe if it gives up none of the old spelling's
    tolerance, and a null block — the shape a config.yml with a bare
    ``virtual_accelerator:`` key parses to — is the case where the two could
    differ.
    """
    rendered = _render(
        _context(
            instances={"virtual_accelerator": None},
            deployed_services=["virtual_accelerator"],
        )
    )
    service = rendered["services"]["virtual-accelerator"]
    assert service["ports"] == ["127.0.0.1:5064:5064/tcp"]
    assert service["environment"]["EPICS_CA_SERVER_PORT"] == "5064"


# ---------------------------------------------------------------------------
# The two-instance render
# ---------------------------------------------------------------------------

STANDIN_INSTANCES = {
    "virtual_accelerator": _instance_block(5064),
    "live_standin": _instance_block(5074),
}


@pytest.fixture
def two_instances() -> dict[str, Any]:
    """A deployment with the live stand-in alongside the baseline VA."""
    return _render(
        _context(
            instances=STANDIN_INSTANCES,
            deployed_services=["virtual_accelerator", "live_standin"],
        )
    )


def _text_lines(rendered_text: str, needle: str) -> list[str]:
    return [line.strip() for line in rendered_text.splitlines() if needle in line]


@pytest.mark.parametrize(
    ("shape", "block"),
    [
        ("null", None),
        ("portless", {"path": "./services/virtual_accelerator"}),
    ],
)
def test_va_compose_deployed_standin_without_a_port_fails_the_render(
    shape: str, block: dict[str, Any] | None
) -> None:
    """A second instance has no default port, and must not borrow instance 1's.

    5064 is the baseline's port. Defaulting a stand-in to it renders two
    containers publishing the same host port and serving Channel Access on it —
    which compose accepts, and docker rejects only at run time, with a bind
    error that names neither the service nor the missing key. The render is
    refused instead, and the message names ``port``.
    """
    from jinja2 import UndefinedError

    with pytest.raises(UndefinedError, match="port"):
        _render_text(
            _context(
                instances={
                    "virtual_accelerator": _instance_block(5064),
                    "live_standin": block,
                },
                deployed_services=["virtual_accelerator", "live_standin"],
            )
        )


def test_va_compose_renders_one_service_per_deployed_instance(
    two_instances: dict[str, Any],
) -> None:
    assert sorted(two_instances["services"]) == ["live-standin", "virtual-accelerator"]


def test_va_compose_container_names_are_namespaced_per_instance(
    two_instances: dict[str, Any],
) -> None:
    """container_name is host-global, so the two machines cannot share one."""
    assert two_instances["services"]["virtual-accelerator"]["container_name"] == (
        "proj-virtual-accelerator"
    )
    assert two_instances["services"]["live-standin"]["container_name"] == "proj-live-standin"


def test_va_compose_publishes_each_instance_on_its_own_port(
    two_instances: dict[str, Any],
) -> None:
    assert two_instances["services"]["virtual-accelerator"]["ports"] == ["127.0.0.1:5064:5064/tcp"]
    assert two_instances["services"]["live-standin"]["ports"] == ["127.0.0.1:5074:5074/tcp"]


def test_va_compose_serves_channel_access_on_each_instance_port(
    two_instances: dict[str, Any],
) -> None:
    """The IOC binds what its own block configured, not the baseline's port."""
    assert (
        two_instances["services"]["virtual-accelerator"]["environment"]["EPICS_CA_SERVER_PORT"]
        == "5064"
    )
    assert (
        two_instances["services"]["live-standin"]["environment"]["EPICS_CA_SERVER_PORT"] == "5074"
    )


def test_va_compose_probes_each_instance_on_its_own_port(
    two_instances: dict[str, Any],
) -> None:
    """A healthcheck aimed at the other machine's port would never go red."""
    baseline = two_instances["services"]["virtual-accelerator"]["healthcheck"]["test"]
    standin = two_instances["services"]["live-standin"]["healthcheck"]["test"]
    assert "'localhost', 5064" in baseline[-1]
    assert "'localhost', 5074" in standin[-1]


def test_va_compose_builds_the_image_on_the_first_instance_only(
    two_instances: dict[str, Any],
) -> None:
    """Two services building one tag race each other; one build serves both."""
    assert "build" in two_instances["services"]["virtual-accelerator"]
    assert "build" not in two_instances["services"]["live-standin"]


def test_va_compose_gives_every_instance_an_image(two_instances: dict[str, Any]) -> None:
    """The stand-in carries no build, so it can only run a named image."""
    for service in two_instances["services"].values():
        assert service["image"]


def test_va_compose_standin_reads_its_own_bpm_perturbation_variable() -> None:
    """The stand-in's readout offsets are ITS fault axis, not the baseline's.

    Asserted on the raw text because the ``${VAR:-default}`` literal is the
    contract: compose, not the render, is what resolves it, and a stand-in that
    silently answered to ``VA_BPM_ERRORS`` would perturb both machines at once.
    """
    rendered = _render_text(
        _context(
            instances=STANDIN_INSTANCES,
            deployed_services=["virtual_accelerator", "live_standin"],
            standin_bpm_errors_default="SR:BPM:1:X=0.0001",
        )
    )
    assert _text_lines(rendered, "VA_BPM_ERRORS:") == [
        'VA_BPM_ERRORS: "${VA_BPM_ERRORS:-}"',
        'VA_BPM_ERRORS: "${VA_STANDIN_BPM_ERRORS:-SR:BPM:1:X=0.0001}"',
    ]


def test_va_compose_standin_perturbation_default_is_empty_until_supplied() -> None:
    """A context that never sets the key still renders a valid passthrough."""
    rendered = _render_text(
        _context(
            instances=STANDIN_INSTANCES,
            deployed_services=["virtual_accelerator", "live_standin"],
        )
    )
    assert 'VA_BPM_ERRORS: "${VA_STANDIN_BPM_ERRORS:-}"' in rendered


def test_va_compose_instances_share_the_scenario_mounts(
    two_instances: dict[str, Any],
) -> None:
    """One model, one active-scenario state, deliberately read by both.

    A scenario applied on the host is meant to be observable on both machines
    at once — that is what makes the stand-in a stand-in — so the mounts are the
    one thing the instance axis does NOT split.
    """
    baseline = two_instances["services"]["virtual-accelerator"]["volumes"]
    standin = two_instances["services"]["live-standin"]["volumes"]
    assert baseline == standin
    assert standin == [
        "./build/data/simulation:/data/simulation:ro",
        f"{STATE_MOUNT_SOURCE}:/state/simulation:ro",
    ]


def test_va_compose_header_announces_the_second_instance() -> None:
    """The emitted header says a two-instance file is what it is.

    The single-instance goldens pin the other half of this: the paragraph is
    absent there, so the file an existing project renders is unchanged.
    """
    rendered = _render_text(
        _context(
            instances=STANDIN_INSTANCES,
            deployed_services=["virtual_accelerator", "live_standin"],
        )
    )
    header = rendered.split("services:", 1)[0]
    assert "THIS DEPLOYMENT RENDERS TWO INSTANCES" in header
    assert "live-standin" in header


def test_va_compose_hands_each_instance_its_own_env_passthrough() -> None:
    """The env axis is declared per block, so it must arrive per container."""
    rendered = _render(
        _context(
            instances={
                "virtual_accelerator": _instance_block(5064, env=["HTTP_PROXY"]),
                "live_standin": _instance_block(5074, env=["NO_PROXY"]),
            },
            deployed_services=["virtual_accelerator", "live_standin"],
        )
    )
    baseline = rendered["services"]["virtual-accelerator"]["environment"]
    standin = rendered["services"]["live-standin"]["environment"]
    assert baseline["HTTP_PROXY"] == "${HTTP_PROXY}"
    assert "NO_PROXY" not in baseline
    assert standin["NO_PROXY"] == "${NO_PROXY}"
    assert "HTTP_PROXY" not in standin


def _regenerate() -> None:
    """Overwrite the single-instance goldens from today's template.

    Rendered from :func:`_single_instance_contexts`, the same contexts the
    pinned test renders, so a regenerated golden can only differ where the
    template does. See that test's update discipline before running this.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, context in sorted(_single_instance_contexts().items()):
        path = GOLDEN_DIR / f"{name}.yml"
        # The repo's end-of-file-fixer owns the trailing newline the renderer
        # does not emit, and the pinned test normalizes it on both sides.
        path.write_text(_render_text(context).rstrip("\n") + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    _regenerate()
