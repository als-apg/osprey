"""Which machine the archiver recorder records, across the compose render.

The recorder samples one IOC over Channel Access and writes what it reads into
the archiver store. **The archive belongs to the machine it records**, and the
store this recorder writes into holds the virtual accelerator's seeded past —
so the virtual accelerator is what it samples, on every deployment.

A deployment that stands a live stand-in up has two IOCs, and the stand-in is
not a candidate. It rehearses the facility's real machine, so the past it is
shown is the real machine's (the archiver appliance, when the deployment names
one); the deployment's own store stays the virtual accelerator's, and nothing
the stand-in reads is ever written into it. The compose render has two wiring
sites that name the recorded IOC — the Channel Access address and the startup
dependency — and both must name the ``virtual-accelerator`` service whatever
else was deployed. A recorder that waited on one machine and read from another
would be a deploy-time race that looks like an empty archive.

Two claims are tested here:

1. **The render does not depend on the stand-in.** A deployment with a stand-in
   renders byte-for-byte what a deployment without one renders — not "parses to
   the same YAML" but literally the same bytes, against the goldens under
   ``goldens/recorder/``. Both branches of the one condition the template reads
   (a co-deployed VA and an external IOC) are pinned, because that condition is
   read at three sites and either branch could drift.

2. **Both wiring sites name the virtual accelerator**, at the port
   ``services.virtual_accelerator.port`` states, with the stand-in's own port
   and service nowhere in the document.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

# The same helpers the recorder's existing render tests use, rather than a
# second hand-built context: a golden is only worth what the context behind it
# is worth, and these pin the shape those tests already assert against.
from tests.deployment.test_compose_generator import (
    _render_recorder_template,
    _render_service_template,
)

RECORDER_TEMPLATE = "archiver_recorder/docker-compose.yml.j2"

GOLDEN_DIR = Path(__file__).parent / "goldens" / "recorder"

STANDIN_PORT = 5074

#: The deploy every render here starts from. ``mongodb`` is unconditional (the
#: store is the recorder's other half) and the IOC keys are appended per case.
BASE_DEPLOYED = ["mongodb", "archiver_recorder", "virtual_accelerator"]


def _render_with_standin(
    *, standin_port: int | None = STANDIN_PORT, va_port: int = 5064, project_name: str = "proj-a"
) -> str:
    """Render the recorder template for a deployment that also deploys a stand-in.

    ``services.live_standin`` is written and ``live_standin`` is listed in
    ``deployed_services``, the two spellings a build leaves behind when a
    profile asked for a stand-in — so the template is given every chance to
    read the stand-in and the test is that it never does. ``standin_port=None``
    leaves the block portless, the shape a template branching on the stand-in
    once refused at render time.
    """
    services: dict[str, Any] = {"virtual_accelerator": {"port": va_port}}
    services["live_standin"] = {} if standin_port is None else {"port": standin_port}
    return _render_service_template(
        RECORDER_TEMPLATE,
        project_name,
        services=services,
        deployed_services=[*BASE_DEPLOYED, "live_standin"],
    )


def _recorder_service(rendered: str) -> dict[str, Any]:
    return yaml.safe_load(rendered)["services"]["archiver-recorder"]


# ---------------------------------------------------------------------------
# With a stand-in deployed: the virtual accelerator is still what is recorded.
# ---------------------------------------------------------------------------


def test_the_recorder_dials_the_virtual_accelerator_beside_a_standin() -> None:
    """The Channel Access address is the VA's, not the stand-in's."""
    env = _recorder_service(_render_with_standin())["environment"]
    assert env["EPICS_CA_NAME_SERVERS"] == "virtual-accelerator:5064"
    # Pinned NO, not defaulted: it is what pairs with the name-server TCP
    # transport, which is the only host<->container CA config proven to work.
    assert env["EPICS_CA_AUTO_ADDR_LIST"] == "NO"


def test_the_recorder_waits_for_the_virtual_accelerator_only() -> None:
    """Startup ordering names the recorded machine and only the recorded machine."""
    depends = _recorder_service(_render_with_standin())["depends_on"]
    assert depends["virtual-accelerator"] == {"condition": "service_healthy"}
    assert depends["mongodb"] == {"condition": "service_healthy"}
    assert "live-standin" not in depends


def test_the_virtual_accelerators_port_override_moves_the_recorder() -> None:
    """The port is derived from ``services.virtual_accelerator.port``, not written out."""
    env = _recorder_service(_render_with_standin(va_port=5164))["environment"]
    assert env["EPICS_CA_NAME_SERVERS"] == "virtual-accelerator:5164"


def test_the_standins_port_reaches_nothing_in_the_render() -> None:
    """A distinctive stand-in port appears nowhere: no site reads the stand-in."""
    rendered = _render_with_standin(standin_port=5999)
    assert "5999" not in rendered
    assert "live-standin" not in rendered
    assert "live_standin" not in rendered


def test_a_portless_standin_block_is_not_the_recorders_business() -> None:
    """Nothing about the stand-in is read, so nothing about it can refuse the render."""
    env = _recorder_service(_render_with_standin(standin_port=None))["environment"]
    assert env["EPICS_CA_NAME_SERVERS"] == "virtual-accelerator:5064"


def test_a_standin_deployment_renders_byte_for_byte_what_a_plain_one_renders() -> None:
    """The whole claim in one comparison: the stand-in moves not one byte."""
    golden = (GOLDEN_DIR / "co_deployed.yml").read_text(encoding="utf-8")
    rendered = _render_with_standin()
    assert rendered.rstrip("\n") + "\n" == golden.rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Without a stand-in: the render is pinned, to the byte.
# ---------------------------------------------------------------------------


def _pinned_contexts() -> dict[str, bool]:
    """The pinned shapes, as ``name -> va_co_deployed``.

    Named, because the goldens are named for them and a regenerated golden has
    to come from the same context. Both branches of the one condition are here:
    it is read at three sites, and the external-IOC branch is the one no
    stand-in deployment can reach, so nothing else would catch it drifting.
    """
    return {"co_deployed": True, "external_ioc": False}


@pytest.mark.parametrize("name", sorted(_pinned_contexts()))
def test_the_recorder_render_is_byte_identical_to_the_pinned_shape(name: str) -> None:
    """A deployment must reproduce its pinned shape exactly.

    Byte equality rather than parsed equality on purpose: the rendered compose
    file is also read by humans and diffed by operators, and the comments
    explaining WHY the recorder is wired this way are part of what a reviewer
    reads.

    **Update discipline** — a failure here means a template edit moved the
    render, which is never on its own a reason to hand-edit a golden.
    Regenerate the pair from the same contexts, in the SAME reviewed change as
    the template edit that moved them::

        PYTHONPATH=src:. ./.venv/bin/python tests/deployment/test_recorder_compose_target.py

    then account for every changed byte.
    """
    # Final-newline count is normalized on both sides: the repo's
    # end-of-file-fixer hook owns the goldens' trailing newline, which the
    # renderer does not reproduce. Every other byte still has to match.
    golden = (GOLDEN_DIR / f"{name}.yml").read_text(encoding="utf-8")
    rendered = _render_recorder_template(va_co_deployed=_pinned_contexts()[name])
    assert rendered.rstrip("\n") + "\n" == golden.rstrip("\n") + "\n"


def _regenerate() -> None:
    """Overwrite the goldens from today's template.

    Rendered from :func:`_pinned_contexts`, the same contexts the pinned test
    renders, so a regenerated golden can only differ where the template does.
    See that test's update discipline before running this.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, va_co_deployed in sorted(_pinned_contexts().items()):
        path = GOLDEN_DIR / f"{name}.yml"
        # The repo's end-of-file-fixer owns the trailing newline the renderer
        # does not emit, and the pinned test normalizes it on both sides.
        rendered = _render_recorder_template(va_co_deployed=va_co_deployed)
        path.write_text(rendered.rstrip("\n") + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    _regenerate()
