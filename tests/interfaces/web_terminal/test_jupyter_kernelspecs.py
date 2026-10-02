"""The notebook sidecar's kernel-spec manager: what it resolves, and its import.

That a running sidecar refuses a start outside the allow-list, through the
kernels and the sessions API, and starts the ``osprey`` kernel for a request
that names none, is asserted over real HTTP in
``test_proxy_jupyter_integration.py``. What stays here is what needs no
server: the manager's answer for each kind of name, and that importing it does
not pull the web application into the sidecar process.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from jupyter_client.kernelspec import NoSuchKernel

from osprey.interfaces.web_terminal.jupyter_kernelspecs import AllowListKernelSpecManager
from osprey.interfaces.web_terminal.jupyter_sidecar import KERNELSPEC_NAME

#: A second spec on the kernel path, the shape a package-installed kernel takes.
OTHER_KERNEL = "other"


@pytest.fixture
def kernel_dir(tmp_path: Path) -> Path:
    """A kernel path holding the ``osprey`` spec and one other."""
    root = tmp_path / "kernels"
    for name in (KERNELSPEC_NAME, OTHER_KERNEL):
        spec_dir = root / name
        spec_dir.mkdir(parents=True)
        spec = {
            "argv": ["python", "-m", name, "-f", "{connection_file}"],
            "display_name": name,
            "language": "python",
        }
        (spec_dir / "kernel.json").write_text(json.dumps(spec), encoding="utf-8")
    return root


def _manager(kernel_dir: Path, allowed: set[str]) -> AllowListKernelSpecManager:
    """The manager as the sidecar configures it, on *kernel_dir* alone."""
    return AllowListKernelSpecManager(
        kernel_dirs=[str(kernel_dir)],
        ensure_native_kernel=False,
        allowed_kernelspecs=allowed,
    )


def test_the_listed_kernel_resolves(kernel_dir: Path) -> None:
    spec = _manager(kernel_dir, {KERNELSPEC_NAME}).get_kernel_spec(KERNELSPEC_NAME)

    assert spec.argv[:3] == ["python", "-m", KERNELSPEC_NAME]


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("python3", id="the-interpreters-own-spec"),
        pytest.param(OTHER_KERNEL, id="a-spec-on-the-kernel-path"),
        pytest.param(KERNELSPEC_NAME.upper(), id="the-listed-name-in-another-case"),
    ],
)
def test_a_name_outside_the_allow_list_is_no_such_kernel(kernel_dir: Path, name: str) -> None:
    """Each of these resolves on the base class; the allow-list refuses it."""
    with pytest.raises(NoSuchKernel):
        _manager(kernel_dir, {KERNELSPEC_NAME}).get_kernel_spec(name)


def test_the_listing_is_the_allow_list(kernel_dir: Path) -> None:
    assert sorted(_manager(kernel_dir, {KERNELSPEC_NAME}).get_all_specs()) == [KERNELSPEC_NAME]


def test_an_empty_allow_list_starts_nothing_and_lists_nothing(kernel_dir: Path) -> None:
    """The base class reads an empty allow-list as "everything"; this one fails closed."""
    manager = _manager(kernel_dir, set())

    with pytest.raises(NoSuchKernel):
        manager.get_kernel_spec(KERNELSPEC_NAME)
    assert manager.get_all_specs() == {}


def test_importing_the_module_does_not_build_the_web_application() -> None:
    probe = (
        "import sys\n"
        "import osprey.interfaces.web_terminal.jupyter_kernelspecs\n"
        "print('osprey.interfaces.web_terminal.app' in sys.modules)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "False"
