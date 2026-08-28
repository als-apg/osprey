"""Which machine the archiver recorder records, across the compose render.

The recorder samples one IOC over Channel Access and writes what it reads into
the archiver store. Until the live stand-in existed there was only ever one IOC
it could mean — the co-deployed ``virtual_accelerator`` — so the template named
that service outright at both wiring sites.

A deployment that stands a live stand-in up has two IOCs, and the choice between
them is not arbitrary: **the archive belongs to the machine.** ``live-standin``
is the machine such a deployment calls ``live``, so its readings are the history
the store describes, and the sandbox VA beside it is not what gets recorded.

Two claims are tested here, and the first one is the anchor:

1. **A deployment without a stand-in renders byte-for-byte what it rendered
   before.** Not "parses to the same YAML" — literally the same bytes. The
   goldens under ``goldens/recorder_no_standin/`` were produced from the
   template as it stood before the recorded-machine choice existed, so a diff
   against them is a diff against history. Both branches of the pre-existing
   condition are pinned (a co-deployed VA and an external IOC), because that
   condition is still read at three sites and either branch could drift.

2. **A deployment with a stand-in retargets both wiring sites together.** The
   Channel Access address and the startup dependency have to name the SAME
   instance — a recorder that waits on one machine and reads from another is a
   deploy-time race that looks like an empty archive — so the template derives
   both from one choice made once, and these tests read them back separately.

The image is deliberately not part of the choice: the stand-in is the same IOC
image serving a perturbed copy of the same machine, so the recorder runs the VA
image either way, exactly as it did before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from jinja2 import UndefinedError

# The same helpers the recorder's existing render tests use, rather than a
# second hand-built context: a golden is only worth what the context behind it
# is worth, and these pin the shape those tests already assert against.
from tests.deployment.test_compose_generator import (
    _render_recorder_template,
    _render_service_template,
)

RECORDER_TEMPLATE = "archiver_recorder/docker-compose.yml.j2"

GOLDEN_DIR = Path(__file__).parent / "goldens" / "recorder_no_standin"

#: The deploy every render here starts from. ``mongodb`` is unconditional (the
#: store is the recorder's other half) and the IOC keys are appended per case.
BASE_DEPLOYED = ["mongodb", "archiver_recorder"]


def _render_with_standin(
    *,
    deployed: bool = True,
    has_block: bool = True,
    standin_port: int | None = 5074,
    project_name: str = "proj-a",
) -> str:
    """Render the recorder template for a deployment that has a stand-in block.

    The three knobs move independently on purpose. ``deployed`` and
    ``has_block`` are separate because a config can carry a
    ``services.live_standin`` block that was never deployed, and membership in
    ``deployed_services`` — not the presence of the block — is what the template
    gates on, the same rule every other service here follows. ``standin_port``
    goes to ``None`` for the deployed-but-portless shape.

    Only the two IOC blocks are supplied. The recorder template reads no other
    service's block, so a narrower ``services`` dict than the shared helper's
    default pins exactly the keys this render depends on.
    """
    services: dict[str, Any] = {"virtual_accelerator": {"port": 5064}}
    if has_block:
        services["live_standin"] = {} if standin_port is None else {"port": standin_port}
    deployed_services = [*BASE_DEPLOYED, "virtual_accelerator"]
    if deployed:
        deployed_services.append("live_standin")
    return _render_service_template(
        RECORDER_TEMPLATE,
        project_name,
        services=services,
        deployed_services=deployed_services,
    )


def _standin_service() -> dict[str, Any]:
    """The parsed ``archiver-recorder`` service of a stand-in deployment."""
    return yaml.safe_load(_render_with_standin())["services"]["archiver-recorder"]


# ---------------------------------------------------------------------------
# With a stand-in deployed: the recorder follows the machine, not the model.
# ---------------------------------------------------------------------------


def test_recorder_standin_is_the_channel_access_target() -> None:
    """The recorder reads the stand-in's Channel Access address, not the VA's.

    The port is derived from ``services.live_standin.port`` rather than written
    out, so an operator moving the stand-in's port moves the recorder with it —
    the same rule the single-machine template already followed for the VA.
    """
    env = _standin_service()["environment"]
    assert env["EPICS_CA_NAME_SERVERS"] == "live-standin:5074"
    # Pinned NO, not defaulted: it is what pairs with the name-server TCP
    # transport, which is the only host<->container CA config proven to work.
    assert env["EPICS_CA_AUTO_ADDR_LIST"] == "NO"


def test_recorder_standin_port_override_moves_the_recorder() -> None:
    """A non-default stand-in port reaches the address the recorder connects to.

    Guards the derivation itself: an address that happened to read
    ``live-standin:5074`` from a hardcoded literal would pass the test above.
    """
    rendered = _render_with_standin(standin_port=5084)
    env = yaml.safe_load(rendered)["services"]["archiver-recorder"]["environment"]
    assert env["EPICS_CA_NAME_SERVERS"] == "live-standin:5084"


def test_recorder_standin_is_the_only_ioc_it_waits_for() -> None:
    """Startup ordering names the recorded machine and only the recorded machine.

    Waiting on the VA while reading from the stand-in would let the recorder
    open its CA client before the stand-in's iocInit answers — a race whose
    only symptom is history that starts late, so the two sites are derived from
    one choice and checked here to have landed on the same instance.
    """
    depends = _standin_service()["depends_on"]
    assert depends["live-standin"] == {"condition": "service_healthy"}
    assert "virtual-accelerator" not in depends


def test_recorder_standin_still_waits_for_a_healthy_store() -> None:
    """The store dependency is untouched by the retarget.

    It is unconditional and health-gated for a reason that has nothing to do
    with which IOC is recorded: on a fresh volume mongod creates the admin user
    before it answers commands, and the recorder's first writes fail in that
    window.
    """
    assert _standin_service()["depends_on"]["mongodb"] == {"condition": "service_healthy"}


def test_recorder_standin_still_runs_the_virtual_accelerator_image() -> None:
    """The image is not part of the choice.

    The stand-in is the same IOC image serving a perturbed copy of the same
    machine, so there is one image and the recorder keeps reusing the tag the
    ``virtual_accelerator`` service builds — which is also what stops the
    recorder and the IOC ever running different Channel Access stacks.
    """
    assert _standin_service()["image"] == "${OSPREY_VA_IMAGE:-proj-a-va:local}"


# ---------------------------------------------------------------------------
# Without a stand-in: the render is unchanged, to the byte.
# ---------------------------------------------------------------------------


def _no_standin_contexts() -> dict[str, bool]:
    """The pinned no-stand-in shapes, as ``name -> va_co_deployed``.

    Named, because the goldens are named for them and a regenerated golden has
    to come from the same context. Both branches of the pre-existing condition
    are here: it is still read at three sites, and the external-IOC branch is
    the one no stand-in deployment can reach, so nothing else would catch it
    drifting.
    """
    return {"co_deployed": True, "external_ioc": False}


@pytest.mark.parametrize("name", sorted(_no_standin_contexts()))
def test_recorder_standin_absent_render_is_byte_identical_to_the_pinned_shape(
    name: str,
) -> None:
    """A deployment with no stand-in must reproduce its pinned shape exactly.

    The goldens were first produced from the template BEFORE the recorded-machine
    choice was introduced, so what they pin is a before/after equality rather
    than a self-consistency check. Byte equality rather than parsed equality on
    purpose: the rendered compose file is also read by humans and diffed by
    operators, and the comments explaining WHY the recorder is wired this way
    are part of what a reviewer reads.

    **Update discipline** — a failure here means a template edit moved a
    no-stand-in render, which is never on its own a reason to hand-edit a
    golden. Regenerate the pair from the same contexts, in the SAME reviewed
    change as the template edit that moved them::

        PYTHONPATH=src:. ./.venv/bin/python tests/deployment/test_recorder_standin_compose.py

    then account for every changed byte.
    """
    # Final-newline count is normalized on both sides: the repo's
    # end-of-file-fixer hook owns the goldens' trailing newline, which the
    # renderer does not reproduce. Every other byte still has to match.
    golden = (GOLDEN_DIR / f"{name}.yml").read_text(encoding="utf-8")
    rendered = _render_recorder_template(va_co_deployed=_no_standin_contexts()[name])
    assert rendered.rstrip("\n") + "\n" == golden.rstrip("\n") + "\n"


def test_recorder_standin_block_that_was_never_deployed_changes_nothing() -> None:
    """A ``services.live_standin`` block that was never deployed retargets nothing.

    Membership in ``deployed_services`` is the gate, not the presence of a
    config block — a block alone cannot conjure a stand-in the deploy does not
    run, and pointing the recorder at a container that does not exist would
    leave it reading nothing. Pinned against the co-deployed golden rather than
    by asserting on the address, because "renders what it always did" is the
    whole claim.
    """
    golden = (GOLDEN_DIR / "co_deployed.yml").read_text(encoding="utf-8")
    rendered = _render_with_standin(deployed=False)
    assert rendered.rstrip("\n") + "\n" == golden.rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# The refusal.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("has_block", "standin_port"),
    [
        pytest.param(True, None, id="block-without-a-port"),
        pytest.param(False, None, id="no-block-at-all"),
    ],
)
def test_recorder_standin_without_a_port_refuses_to_render(
    has_block: bool,
    standin_port: int | None,
) -> None:
    """A deployed stand-in with no port is a render-time abort, not a default.

    5064 is the virtual accelerator's port, so defaulting to it would render a
    recorder that silently samples the WRONG machine and files its readings as
    the live machine's history — the exact dishonesty the recorder exists to
    avoid, and invisible in a compose document that comes up green. ``| int``
    refuses instead, because ``int()`` of an undefined raises rather than
    falling back to its default.

    Rendered rather than reasoned about: the assertion is that the PRODUCTION
    Undefined refuses, which is a property of the render, not of the filter.
    """
    with pytest.raises(UndefinedError):
        _render_with_standin(has_block=has_block, standin_port=standin_port)


def _regenerate() -> None:
    """Overwrite the no-stand-in goldens from today's template.

    Rendered from :func:`_no_standin_contexts`, the same contexts the pinned
    test renders, so a regenerated golden can only differ where the template
    does. See that test's update discipline before running this.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, va_co_deployed in sorted(_no_standin_contexts().items()):
        path = GOLDEN_DIR / f"{name}.yml"
        # The repo's end-of-file-fixer owns the trailing newline the renderer
        # does not emit, and the pinned test normalizes it on both sides.
        rendered = _render_recorder_template(va_co_deployed=va_co_deployed)
        path.write_text(rendered.rstrip("\n") + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    _regenerate()
