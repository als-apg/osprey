"""The project image installs everything after the framework under a frozen set.

The deps layer primes the framework, lays the dev-staged local delta over it,
freezes what resolved to ``/opt/osprey/constraints.txt`` and only then installs
the profile's own ``dependencies:`` — in an install of their own, under that
file. The wheel layer installs a dropped wheel under the same file, minus the
lines naming the wheels being dropped. Either way a package arriving after the
framework may add distributions and may not move one the framework resolved.

Two kinds of test pin that. The rendered-order tests read the ``RUN`` bodies and
run them against a recording ``pip`` stub. The real-installer tests run the same
bodies against an actual ``uv`` and ``pip`` in a throwaway venv, resolving from a
directory of tiny local wheels with the index switched off — a stub that exits 0
cannot tell a constrained install from an unconstrained one, and the difference
between the two is the whole point of the layer.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import resource
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tests.cli.test_dockerfile_template import _instructions, _render_template
from tests.deployment._pip_probe import deps_pip_calls, wheel_pip_calls

CONSTRAINTS = "/opt/osprey/constraints.txt"
PIP_CONSTRAINTS = "/tmp/deps-ctx/pip-constraints.txt"
PROFILE_DEPS = "numpy 'pydantic>=2'"


def _run_bodies(text: str) -> list[str]:
    """Every RUN body as the shell receives it: BuildKit's ``--mount`` flags dropped."""
    return [
        re.sub(r"^(?:--\S+\s+)+", "", body)
        for keyword, body in _instructions(text)
        if keyword == "RUN"
    ]


def _deps_body(text: str) -> str:
    bodies = [b for b in _run_bodies(text) if "build-essential" in b and "$OSPREY_PIP_SPEC" in b]
    assert len(bodies) == 1, f"expected exactly one deps RUN, got {len(bodies)}"
    return bodies[0]


def _wheel_body(text: str) -> str:
    bodies = [b for b in _run_bodies(text) if "--force-reinstall" in b]
    assert len(bodies) == 1, f"expected exactly one wheel RUN, got {len(bodies)}"
    return bodies[0]


def _uv_bootstrap(deps: str) -> str:
    """The pinned uv requirement the deps layer installs with pip, then removes."""
    match = re.search(r'pip install "(uv==[^"]+)"', deps)
    assert match, f"the deps RUN does not bootstrap a pinned uv:\n{deps}"
    return match.group(1)


# ── Rendered order ───────────────────────────────────────────────────────────


class TestDepsLayerOrder:
    """Primer, delta, freeze, profile dependencies, check — inside one RUN."""

    def test_steps_run_in_order_before_the_purge(self):
        deps = _deps_body(_render_template(pip_dependency_args=PROFILE_DEPS))
        steps = [
            f'pip install "{_uv_bootstrap(deps)}"',
            'uv pip install ${OSPREY_PIP_PRE:+--prerelease=allow} "$OSPREY_PIP_SPEC"',
            "uv pip install -r /tmp/deps-ctx/osprey-local-requirements.txt",
            f"pip freeze --exclude uv --exclude pip --exclude setuptools --exclude wheel > {CONSTRAINTS}",
            f"uv pip install -c {PIP_CONSTRAINTS} -c {CONSTRAINTS} {PROFILE_DEPS}",
            "pip uninstall -y uv",
            'if [ "$OSPREY_DEV" != "1" ] || [ -f /opt/osprey/lock-constraints.txt ]; then pip check; fi',
            "apt-get purge -y build-essential python3-dev",
            "rm -rf /tmp/deps-ctx",
        ]
        # The delta is installed on both paths; the release path's copy, after
        # the primer, is the later of the two.
        positions = [
            deps.rfind(step) if "osprey-local-requirements" in step else deps.find(step)
            for step in steps
        ]
        assert -1 not in positions, [s for s, at in zip(steps, positions, strict=True) if at < 0]
        assert positions == sorted(positions), list(zip(steps, positions, strict=True))

    def test_every_step_is_chained_so_a_failure_fails_the_layer(self):
        deps = _deps_body(_render_template(pip_dependency_args=PROFILE_DEPS))
        for step in (
            f"pip freeze --exclude uv --exclude pip --exclude setuptools --exclude wheel > {CONSTRAINTS}",
            f"uv pip install -c {PIP_CONSTRAINTS} -c {CONSTRAINTS} {PROFILE_DEPS}",
            "pip uninstall -y uv",
            'if [ "$OSPREY_DEV" != "1" ] || [ -f /opt/osprey/lock-constraints.txt ]; then pip check; fi',
        ):
            before = deps[: deps.index(step)].rstrip()
            assert before.endswith("&&"), f"{step!r} is not &&-chained:\n{deps}"

    def test_the_setuptools_dso_constraint_is_unchanged_and_belongs_to_one_run(self):
        text = _render_template(pip_dependency_args=PROFILE_DEPS)
        deps = _deps_body(text)
        assert "printf 'setuptools-dso>=2.12.4\\n' > /tmp/deps-ctx/pip-constraints.txt" in deps
        assert "export PIP_CONSTRAINT=/tmp/deps-ctx/pip-constraints.txt" in deps
        assert [b for b in _run_bodies(text) if "PIP_CONSTRAINT" in b] == [deps]

    def test_the_primer_resolves_the_framework_without_the_profile(self, tmp_path):
        deps = _deps_body(_render_template(pip_dependency_args=PROFILE_DEPS))
        calls = deps_pip_calls(deps, tmp_path, {"OSPREY_PIP_SPEC": "osprey-framework==1.0"})
        assert calls == [
            ["install", _uv_bootstrap(deps)],
            ["install", "osprey-framework==1.0"],
            [
                "freeze",
                "--exclude",
                "uv",
                "--exclude",
                "pip",
                "--exclude",
                "setuptools",
                "--exclude",
                "wheel",
            ],
            ["install", "-c", str(tmp_path / "deps-ctx/pip-constraints.txt")]
            + ["-c", str(tmp_path / "opt-osprey/constraints.txt"), "numpy", "pydantic>=2"],
            ["uninstall", "-y", "uv"],
            ["check"],
        ]

    def test_a_dev_build_skips_the_check(self, tmp_path):
        deps = _deps_body(_render_template(pip_dependency_args=PROFILE_DEPS))
        calls = deps_pip_calls(
            deps, tmp_path, {"OSPREY_PIP_SPEC": "osprey-framework==1.0", "OSPREY_DEV": "1"}
        )
        assert ["check"] not in calls
        assert [
            "freeze",
            "--exclude",
            "uv",
            "--exclude",
            "pip",
            "--exclude",
            "setuptools",
            "--exclude",
            "wheel",
        ] in calls

    def test_a_profile_without_dependencies_still_freezes_and_checks(self, tmp_path):
        deps = _deps_body(_render_template())
        calls = deps_pip_calls(deps, tmp_path, {"OSPREY_PIP_SPEC": "osprey-framework==1.0"})
        assert calls == [
            ["install", _uv_bootstrap(deps)],
            ["install", "osprey-framework==1.0"],
            [
                "freeze",
                "--exclude",
                "uv",
                "--exclude",
                "pip",
                "--exclude",
                "setuptools",
                "--exclude",
                "wheel",
            ],
            ["uninstall", "-y", "uv"],
            ["check"],
        ]


class TestDepsLayerOnTheLockPath:
    """With the lock constraints staged, nothing is primed and the check always runs."""

    LOCK = "fwdep==1.0\n"

    def _calls(self, tmp_path, **env):
        deps = _deps_body(_render_template(pip_dependency_args=PROFILE_DEPS))
        return deps_pip_calls(
            deps,
            tmp_path,
            {"OSPREY_PIP_SPEC": "osprey-framework==1.0", **env},
            staged={
                "osprey-local-requirements.txt": "fwdep\n",
                "osprey-lock-constraints.txt": self.LOCK,
            },
        )

    def test_the_delta_is_installed_alone_and_the_profile_under_the_lock(self, tmp_path):
        calls = self._calls(tmp_path, OSPREY_DEV="1")
        deps_ctx = tmp_path / "deps-ctx"
        assert calls[1] == ["install", "-r", str(deps_ctx / "osprey-local-requirements.txt")]
        assert not [c for c in calls if "osprey-framework==1.0" in c or "osprey-framework" in c]
        assert (tmp_path / "opt-osprey/lock-constraints.txt").read_text() == self.LOCK

    def test_a_dev_build_on_the_lock_path_still_checks(self, tmp_path):
        assert ["check"] in self._calls(tmp_path, OSPREY_DEV="1")


class TestWheelLayerOrder:
    def test_the_wheel_installs_under_the_filtered_set_then_checks(self):
        wheel = _wheel_body(_render_template())
        steps = [
            f"{CONSTRAINTS} /tmp/ctx/*.whl > /tmp/ctx/constraints.txt",
            "pip install -c /tmp/ctx/constraints.txt $lock /tmp/ctx/*.whl",
            "pip install --no-deps --force-reinstall /tmp/ctx/*.whl",
            "pip check;",
            "rm -rf /tmp/ctx",
        ]
        positions = [wheel.find(step) for step in steps]
        assert -1 not in positions, [s for s, at in zip(steps, positions, strict=True) if at < 0]
        assert positions == sorted(positions), list(zip(steps, positions, strict=True))

    def test_the_check_is_not_dev_guarded(self):
        assert "OSPREY_DEV" not in _wheel_body(_render_template())

    def test_the_lock_path_adds_the_lock_to_the_frozen_set(self, tmp_path):
        first = wheel_pip_calls(_wheel_body(_render_template()), tmp_path, lock=True)[0]
        assert first[:5] == [
            "install",
            "-c",
            str(tmp_path / "ctx/constraints.txt"),
            "-c",
            str(tmp_path / "opt-osprey/lock-constraints.txt"),
        ]


# ── Real installers ──────────────────────────────────────────────────────────


#: The framework module the wheel layer runs after its overlay, shipped in the
#: stand-in framework wheel as the real one ships it. It is standard library
#: only, so it runs in these test venvs exactly as in an image. It asserts that
#: one installed distribution owns ``pyaml``; the stand-in owns it itself, in
#: place of the pyAML distribution the real framework requires.
_PYAML_OWNER_SOURCE = (
    Path(__file__).resolve().parents[2] / "src" / "osprey" / "deployment" / "pyaml_owner.py"
)

#: The constraints filter the wheel layer runs before its install, shipped in the
#: stand-in framework wheel as the real one ships it; also standard library only.
_CONSTRAINTS_SOURCE = _PYAML_OWNER_SOURCE.with_name("constraints.py")


def _framework_modules() -> dict[str, str]:
    """The ``osprey`` and ``pyaml`` package files the stand-in framework wheel installs."""
    return {
        "pyaml/__init__.py": "",
        "osprey/__init__.py": "",
        "osprey/deployment/__init__.py": "",
        "osprey/deployment/pyaml_owner.py": _PYAML_OWNER_SOURCE.read_text(encoding="utf-8"),
        "osprey/deployment/constraints.py": _CONSTRAINTS_SOURCE.read_text(encoding="utf-8"),
    }


def _make_wheel(directory: Path, name: str, version: str, requires: tuple[str, ...] = ()) -> Path:
    """Write a minimal pure-Python wheel for *name* into *directory*.

    ``osprey-framework`` also installs the modules the image recipe runs and
    checks (:func:`_framework_modules`).
    """
    dist = name.replace("-", "_")
    info = f"{dist}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n" + "".join(
        f"Requires-Dist: {requirement}\n" for requirement in requires
    )
    files = {
        **(_framework_modules() if name == "osprey-framework" else {}),
        f"{dist}/__init__.py": f'__version__ = "{version}"\n',
        f"{info}/METADATA": metadata,
        f"{info}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    record = []
    for path, content in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content.encode()).digest())
        record.append(f"{path},sha256={digest.rstrip(b'=').decode()},{len(content.encode())}")
    record.append(f"{info}/RECORD,,")
    files[f"{info}/RECORD"] = "\n".join(record) + "\n"
    target = directory / f"{dist}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(target, "w") as archive:
        for path, content in files.items():
            archive.writestr(path, content)
    return target


@pytest.fixture(scope="module")
def wheelhouse(tmp_path_factory) -> Path:
    """The whole package index these tests resolve from.

    The framework resolves ``fwdep`` to 1.0 although 2.0 is on offer, so a later
    install that asks for 2.0 is satisfiable by an unconstrained pip — which
    would upgrade it and exit 0 — and refused only by a constrained one.
    """
    house = tmp_path_factory.mktemp("wheelhouse")
    _make_wheel(house, "osprey-framework", "1.0", ("fwdep<2",))
    _make_wheel(house, "fwdep", "1.0")
    _make_wheel(house, "fwdep", "2.0")
    _make_wheel(house, "site-extra", "1.0", ("fwdep",))
    _make_wheel(house, "pin-mover", "1.0", ("fwdep>=2",))
    # The lock-driven path's own cases: a dependency only a dropped wheel adds,
    # and a build-tool pin the setuptools_dso constraint has to refuse.
    _make_wheel(house, "newdep", "1.0")
    _make_wheel(house, "newdep", "2.0")
    _make_wheel(house, "setuptools-dso", "2.12.3")
    _make_wheel(house, "setuptools-dso", "2.12.4")
    _make_wheel(house, "needs-setuptools-dso", "1.0", ("setuptools-dso",))
    return house


#: The open-file limit a recipe runs under here. A build container's runtime
#: ceiling is far above a login shell's default (256 on macOS), and uv's
#: bytecode compile opens pipes to one interpreter per core, which that default
#: cannot hold on a many-core host.
_RECIPE_OPEN_FILES = 10240


def _recipe_open_files() -> int:
    """:data:`_RECIPE_OPEN_FILES`, held to the hard limit and never below the soft one."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    ceiling = (
        _RECIPE_OPEN_FILES if hard == resource.RLIM_INFINITY else min(_RECIPE_OPEN_FILES, hard)
    )
    return max(soft, ceiling)


#: What ``pip freeze`` leaves out of the image's constraints: the installer's
#: own tooling, which a venv seeds by Python version and the framework never
#: resolved. The template spells the same set.
INSTALLER_TOOLING = frozenset({"pip", "setuptools", "wheel", "uv"})


class _Image:
    """A venv standing in for the image, and the sandbox its RUN bodies run in."""

    def __init__(self, root: Path, wheelhouse: Path) -> None:
        self.root = root
        self.venv = root / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(self.venv)], check=True)
        self.deps_ctx = root / "deps-ctx"
        self.deps_ctx.mkdir()
        (self.deps_ctx / ".dockerignore").write_text("")
        self.wheel_ctx = root / "ctx"
        self.wheel_ctx.mkdir()
        (self.wheel_ctx / ".dockerignore").write_text("")
        self.opt = root / "opt-osprey"
        self.apt_lists = root / "apt-lists"
        self.apt_lists.mkdir()
        stub_bin = root / "bin"
        stub_bin.mkdir()
        # The deps layer bootstraps uv with pip and removes it again; here the
        # host's uv stands in for the pinned one, so those two pip calls are
        # no-ops and every other one reaches the venv's real pip.
        uv = shutil.which("uv")
        assert uv, "the real-installer tests need `uv` on PATH, as the suite runs under it"
        (stub_bin / "uv").symlink_to(uv)
        venv_pip = shlex.quote(str(self.venv / "bin" / "pip"))
        for name, script in (
            ("apt-get", "#!/bin/sh\nexit 0\n"),
            (
                "pip",
                '#!/bin/sh\ncase "$*" in "install uv=="*|"uninstall -y uv") exit 0 ;; esac\n'
                f'exec {venv_pip} "$@"\n',
            ),
        ):
            stub = stub_bin / name
            stub.write_text(script)
            stub.chmod(0o755)
        # Start from the host environment minus anything that would steer the
        # venv's installers or interpreter, then switch the index off: every
        # resolve below is answered by the wheelhouse or not at all.
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("PIP_", "UV_"))
            and key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
        }
        env.update(
            PATH=os.pathsep.join([str(stub_bin), str(self.venv / "bin"), env.get("PATH", "")]),
            PIP_NO_INDEX="1",
            PIP_FIND_LINKS=str(wheelhouse),
            PIP_DISABLE_PIP_VERSION_CHECK="1",
            # The image's uv installs into the system interpreter; the venv is it here.
            UV_PYTHON=str(self.venv / "bin" / "python"),
            # uv has no environment spelling of --no-index; offline it still
            # reads the find-links directory and reaches no index.
            UV_OFFLINE="1",
            UV_FIND_LINKS=str(wheelhouse),
            UV_CACHE_DIR=str(root / "uv-cache"),
            OSPREY_PIP_SPEC="osprey-framework==1.0",
            OSPREY_PIP_PRE="",
            OSPREY_DEV="",
            PIP_NO_PROXY="",
        )
        self.env = env

    def run(self, body: str, **env: str) -> subprocess.CompletedProcess[str]:
        rewritten = (
            body.replace("/tmp/deps-ctx", str(self.deps_ctx))
            .replace("/tmp/ctx", str(self.wheel_ctx))
            .replace("/var/lib/apt/lists", str(self.apt_lists))
            .replace("/opt/osprey", str(self.opt))
        )
        return subprocess.run(
            ["sh", "-c", f"ulimit -n {_recipe_open_files()} && {{\n{rewritten}\n}}"],
            capture_output=True,
            text=True,
            env={**self.env, **env},
        )

    def installed(self) -> dict[str, str]:
        """``{distribution: version}`` for everything in the venv but the installer's
        own tooling (a 3.11 venv seeds setuptools; the image's freeze excludes it)."""
        listed = subprocess.run(
            ["pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            env=self.env,
            check=True,
        )
        return {
            entry["name"].replace("_", "-"): entry["version"]
            for entry in json.loads(listed.stdout)
            if entry["name"] not in INSTALLER_TOOLING
        }

    def constraints(self) -> dict[str, str]:
        text = (self.opt / "constraints.txt").read_text()
        return dict(line.split("==", 1) for line in text.split())

    def drop(self, wheel: Path) -> None:
        (self.wheel_ctx / wheel.name).write_bytes(wheel.read_bytes())


@pytest.fixture
def image(tmp_path, wheelhouse) -> _Image:
    return _Image(tmp_path, wheelhouse)


@pytest.fixture
def primed(image) -> _Image:
    """An image whose deps layer has run for a profile without dependencies."""
    result = image.run(_deps_body(_render_template()))
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    return image


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return f"{result.stdout}\n{result.stderr}"


class TestDepsLayerUnderRealInstallers:
    def test_a_profile_dependency_installs_beside_the_frozen_set(self, image):
        result = image.run(_deps_body(_render_template(pip_dependency_args="site-extra")))
        assert result.returncode == 0, _output(result)
        # Frozen before the profile's packages went in: the file is the
        # framework's resolved set, not the image's.
        assert image.constraints() == {"osprey-framework": "1.0", "fwdep": "1.0"}
        assert image.installed() == {
            "osprey-framework": "1.0",
            "fwdep": "1.0",
            "site-extra": "1.0",
        }
        assert not image.deps_ctx.exists(), "the staged context outlived a successful layer"

    def test_a_pin_moving_profile_dependency_fails_the_layer(self, image):
        result = image.run(_deps_body(_render_template(pip_dependency_args="pin-mover")))
        assert result.returncode != 0, _output(result)
        assert image.installed() == {"osprey-framework": "1.0", "fwdep": "1.0"}
        assert image.deps_ctx.exists(), "cleanup ran although the install failed"

    def test_the_same_dependency_moves_the_pin_when_nothing_constrains_it(self, image):
        """The control for the test above: without ``-c`` uv takes the upgrade
        and reports success, leaving the framework on a version it excluded."""
        assert image.run("uv pip install osprey-framework==1.0").returncode == 0
        result = image.run("uv pip install pin-mover")
        assert result.returncode == 0, _output(result)
        assert image.installed()["fwdep"] == "2.0"

    def test_the_local_delta_is_frozen_and_fails_the_check(self, image):
        """A delta that moves a framework pin is in the set by the time it is
        frozen, and is what the closing check exists to catch."""
        (image.deps_ctx / "osprey-local-requirements.txt").write_text("fwdep==2.0\n")
        result = image.run(_deps_body(_render_template()))
        assert result.returncode != 0, _output(result)
        assert "fwdep" in result.stdout, _output(result)
        assert image.constraints()["fwdep"] == "2.0"

    def test_a_dev_build_tolerates_the_same_delta(self, image):
        (image.deps_ctx / "osprey-local-requirements.txt").write_text("fwdep==2.0\n")
        result = image.run(_deps_body(_render_template()), OSPREY_DEV="1")
        assert result.returncode == 0, _output(result)
        assert image.constraints()["fwdep"] == "2.0"


class TestWheelLayerUnderRealPip:
    def test_a_dropped_framework_wheel_is_excluded_from_the_constraints(self, primed, tmp_path):
        """The frozen set pins the framework at the primed version; the dropped
        wheel is a different one. It installs because its own line is left out."""
        build = tmp_path / "build"
        build.mkdir()
        primed.drop(_make_wheel(build, "osprey-framework", "1.1", ("fwdep<2",)))
        result = primed.run(_wheel_body(_render_template()))
        assert result.returncode == 0, _output(result)
        assert primed.installed() == {"osprey-framework": "1.1", "fwdep": "1.0"}
        assert not primed.wheel_ctx.exists()
        # The image's frozen set is the deps layer's; the filtered copy was scratch.
        assert primed.constraints()["osprey-framework"] == "1.0"

    def test_the_dropped_wheel_would_not_install_under_the_unfiltered_set(self, primed, tmp_path):
        """The control: under the frozen file as written, the wheel conflicts
        with its own pin."""
        build = tmp_path / "build"
        build.mkdir()
        wheel = _make_wheel(build, "osprey-framework", "1.1", ("fwdep<2",))
        result = primed.run(f"pip install -c /opt/osprey/constraints.txt {wheel}")
        assert result.returncode != 0, _output(result)

    def test_a_dropped_wheel_that_moves_a_pin_fails_the_layer(self, primed, tmp_path):
        build = tmp_path / "build"
        build.mkdir()
        primed.drop(_make_wheel(build, "osprey-framework", "1.1", ("fwdep>=2",)))
        result = primed.run(_wheel_body(_render_template()))
        assert result.returncode != 0, _output(result)
        assert primed.installed() == {"osprey-framework": "1.0", "fwdep": "1.0"}
        assert primed.wheel_ctx.exists(), "cleanup ran although the install failed"

    def test_a_second_dropped_wheel_installs_under_the_frozen_set(self, primed, tmp_path):
        """Every dropped wheel's own name is left out, and only those: a payload
        wheel riding beside the framework still cannot move ``fwdep``."""
        build = tmp_path / "build"
        build.mkdir()
        primed.drop(_make_wheel(build, "osprey-framework", "1.1", ("fwdep<2",)))
        primed.drop(_make_wheel(build, "site-payload", "0.1", ("fwdep>=2",)))
        result = primed.run(_wheel_body(_render_template()))
        assert result.returncode != 0, _output(result)
        assert primed.installed()["fwdep"] == "1.0"

    def test_no_wheel_is_a_no_op(self, primed):
        before = primed.installed()
        (primed.opt / "constraints.txt").unlink()
        result = primed.run(_wheel_body(_render_template()))
        assert result.returncode == 0, _output(result)
        assert primed.installed() == before
        assert not primed.wheel_ctx.exists()


class TestLockPathUnderRealInstallers:
    """A source build's deps layer, profile install and wheel layer, under the lock.

    The wheelhouse offers ``fwdep`` 1.0 and 2.0; the manifest asks for it
    unpinned, which an unconstrained installer answers with 2.0.
    """

    @staticmethod
    def _stage(image, lock: str, manifest: str = "fwdep\n") -> None:
        (image.deps_ctx / "osprey-local-requirements.txt").write_text(manifest)
        (image.deps_ctx / "osprey-lock-constraints.txt").write_text(lock)

    def test_the_manifest_installs_at_the_locks_versions(self, image):
        self._stage(image, "fwdep==1.0\n")
        result = image.run(_deps_body(_render_template()), OSPREY_DEV="1")
        assert result.returncode == 0, _output(result)
        assert image.installed() == {"fwdep": "1.0"}
        assert (image.opt / "lock-constraints.txt").read_text() == "fwdep==1.0\n"

    def test_without_the_lock_the_same_manifest_takes_the_newest(self, image):
        """The control: the manifest alone, unconstrained, moves to 2.0."""
        (image.deps_ctx / "osprey-local-requirements.txt").write_text("fwdep\n")
        assert (
            image.run("uv pip install -r /tmp/deps-ctx/osprey-local-requirements.txt").returncode
            == 0
        )
        assert image.installed() == {"fwdep": "2.0"}

    def test_a_profile_dependency_is_held_to_the_lock(self, image):
        """``fwdep`` is the profile's here, not the framework's: only the lock pins it."""
        self._stage(image, "fwdep==1.0\n", manifest="")
        result = image.run(_deps_body(_render_template(pip_dependency_args="fwdep")))
        assert result.returncode == 0, _output(result)
        assert image.installed() == {"fwdep": "1.0"}

    def test_a_lock_that_pins_setuptools_dso_below_the_fix_fails_the_layer(self, image):
        self._stage(image, "setuptools-dso==2.12.3\n", manifest="needs-setuptools-dso\n")
        result = image.run(_deps_body(_render_template()), OSPREY_DEV="1")
        assert result.returncode != 0, _output(result)
        assert "setuptools-dso" in _output(result)

    def test_a_lock_at_the_fix_satisfies_the_setuptools_dso_constraint(self, image):
        self._stage(image, "setuptools-dso==2.12.4\n", manifest="needs-setuptools-dso\n")
        result = image.run(_deps_body(_render_template()), OSPREY_DEV="1")
        assert result.returncode == 0, _output(result)
        assert image.installed()["needs-setuptools-dso"] == "1.0"

    def test_a_dropped_wheels_new_dependency_is_held_to_the_lock(self, image, tmp_path):
        self._stage(image, "fwdep==1.0\nnewdep==1.0\n")
        result = image.run(_deps_body(_render_template()), OSPREY_DEV="1")
        assert result.returncode == 0, _output(result)
        build = tmp_path / "build"
        build.mkdir()
        image.drop(_make_wheel(build, "osprey-framework", "1.1", ("fwdep", "newdep")))
        result = image.run(_wheel_body(_render_template()))
        assert result.returncode == 0, _output(result)
        assert image.installed() == {"osprey-framework": "1.1", "fwdep": "1.0", "newdep": "1.0"}
