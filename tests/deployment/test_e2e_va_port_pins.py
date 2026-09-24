"""Every deployed e2e stack publishes virtual-accelerator ports of its own.

Two host ports sit outside a deployment's thousand-port block: the Channel
Access port and the pvAccess port of virtual-accelerator instance 1. Moving the
block moves neither, so two stacks deployed on one host at once collide on
whichever one they left at its protocol default. A lane that pins its Channel
Access port has already said it shares the host, and it has to pin the pvAccess
port too.

This reads the SOURCE of the e2e modules rather than importing them, for the
reason ``test_e2e_project_names`` gives: importing one runs its import-time
guards. The shared builders in ``tests/e2e/_orm_stack.py`` are imported, lazily
and only by the test that needs their defaults.
"""

from __future__ import annotations

import inspect
from pathlib import Path

#: The repo's ``tests/`` directory, and the e2e suites whose modules deploy.
TESTS_ROOT = Path(__file__).resolve().parents[1]
E2E_ROOTS = (TESTS_ROOT / "e2e", TESTS_ROOT / "va" / "e2e")

#: The ``--set`` spellings that move instance 1's two protocol ports.
CA_PIN = "virtual_accelerator.port="
PVA_PIN = "virtual_accelerator.pva_port="


def test_every_lane_that_moves_the_channel_access_port_moves_the_pvaccess_port_too() -> None:
    matched: list[str] = []
    offenders: list[str] = []
    for root in E2E_ROOTS:
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if CA_PIN not in text:
                continue
            relative = path.relative_to(TESTS_ROOT.parent).as_posix()
            matched.append(relative)
            if PVA_PIN not in text:
                offenders.append(relative)

    assert "tests/e2e/_orm_stack.py" in matched, (
        f"the scan must reach the shared builders; it matched {matched}"
    )
    assert offenders == [], (
        f"these modules pin {CA_PIN}... but not {PVA_PIN}..., so their stack "
        f"publishes the pvAccess protocol port every other VA on the host does: {offenders}"
    )


def test_the_shared_builders_reserve_a_pvaccess_port_of_their_own() -> None:
    from osprey.port_layout import CA_DEFAULT_PORT, PVA_DEFAULT_PORT
    from tests.e2e._orm_stack import (
        VA_CA_PORT,
        VA_PVA_PORT,
        build_project_subprocess,
        build_via_cli_runner,
        init_args,
    )

    assert VA_PVA_PORT != VA_CA_PORT
    assert not {VA_CA_PORT, VA_PVA_PORT} & {CA_DEFAULT_PORT, PVA_DEFAULT_PORT}
    assert f"{PVA_PIN}{VA_PVA_PORT}" in init_args("p", output_dir=Path("/unused"))
    for fn in (init_args, build_via_cli_runner, build_project_subprocess):
        assert inspect.signature(fn).parameters["va_pva_port"].default == VA_PVA_PORT, fn
