"""A deployment's own Python packages reach the build venv and every project image.

``osprey build`` builds a wheel of each package
:func:`osprey.apps.rules.deployment_packages` enumerates — every ``apps/<name>/``
with a ``pyproject.toml`` and every local-path profile ``dependencies`` entry —
into a deployment-packages directory, and installs them on the one existing
path a profile dependency takes, in two spellings of one list:

- the build venv installs each from its wheel, beside the framework, in one
  resolve;
- the recorded ``pyproject.toml`` and the rendered Dockerfile carry the
  distribution names, which the image installs from the directory staged into
  every image context, on the same line as ``$OSPREY_PIP_SPEC``.

Neither the record nor any Dockerfile names a path on the building host.
"""

from __future__ import annotations

import json
import re
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from click.testing import CliRunner
from jinja2 import Template

from osprey.cli.build_environment import (
    _create_project_venv,
    _plan_project_venv,
    deployment_dependencies,
)
from osprey.cli.build_profile import BuildProfile, EnvironmentConfig
from osprey.deployment import wheel_build
from osprey.deployment.wheel_build import (
    DEPLOYMENT_PACKAGES_DIRNAME,
    DEPLOYMENT_PACKAGES_MANIFEST,
    build_deployment_packages,
)
from osprey.utils.workspace import IMAGE_DIR_NAME
from osprey_connectors.errors import BuildProfileError
from osprey_connectors.workspace import BUILD_DIR_NAME

APP_PYPROJECT = """\
[project]
name = "Knob_App"
version = "0.1.0"
dependencies = ["lume-base==0.6"]
"""

KIND_PYPROJECT = """\
[project]
name = "ring-kind"
version = "0.2.0"
dependencies = []
"""


_REAL_BUILD_WHEEL = wheel_build._build_wheel


def _fake_wheel(source: Path, out_dir: Path) -> Path:
    """Stand in for the wheel build: a wheel named from the source's pyproject."""
    import tomllib

    project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    stem = project["name"].replace("-", "_").lower()
    wheel = out_dir / f"{stem}-{project['version']}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{stem}-{project['version']}.dist-info/METADATA", "Name: x\n")
    return wheel


@pytest.fixture(autouse=True)
def _no_real_wheel_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wheel_build, "_build_wheel", _fake_wheel)


def _deployment(root: Path) -> Path:
    (root / "apps" / "knob").mkdir(parents=True)
    (root / "apps" / "knob" / "pyproject.toml").write_text(APP_PYPROJECT, encoding="utf-8")
    (root / "kinds" / "ring").mkdir(parents=True)
    (root / "kinds" / "ring" / "pyproject.toml").write_text(KIND_PYPROJECT, encoding="utf-8")
    return root


DEPENDENCIES = ["numpy>=1.24", "./kinds/ring"]


def _profile(dependencies: list[str]) -> BuildProfile:
    return BuildProfile(
        name="test",
        dependencies=list(dependencies),
        osprey_install="osprey-framework==2026.10.1",
        environment=EnvironmentConfig(),
    )


@pytest.fixture(autouse=True)
def _uv_installer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV", "/opt/bin/uv")


def _built(tmp_path: Path):
    root = _deployment(tmp_path / "repo")
    dest = tmp_path / "stage" / DEPLOYMENT_PACKAGES_DIRNAME
    return root, dest, build_deployment_packages(root, DEPENDENCIES, dest)


# --- the build ---------------------------------------------------------------


def test_each_package_is_built_into_the_directory_with_a_manifest(tmp_path: Path) -> None:
    _root, dest, packages = _built(tmp_path)

    assert [(p.name, p.version) for p in packages] == [
        ("knob-app", "0.1.0"),
        ("ring-kind", "0.2.0"),
    ]
    assert all(p.wheel.parent == dest and p.wheel.is_file() for p in packages)
    manifest = json.loads((dest / DEPLOYMENT_PACKAGES_MANIFEST).read_text(encoding="utf-8"))
    assert manifest == {
        "packages": [
            {"name": "knob-app", "version": "0.1.0", "wheel": packages[0].wheel.name},
            {"name": "ring-kind", "version": "0.2.0", "wheel": packages[1].wheel.name},
        ]
    }


def test_a_deployment_with_no_package_builds_nothing(tmp_path: Path) -> None:
    dest = tmp_path / DEPLOYMENT_PACKAGES_DIRNAME

    assert build_deployment_packages(tmp_path, ["numpy"], dest) == ()
    assert not dest.exists()


def test_a_failed_wheel_build_names_the_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()
    root = _deployment(tmp_path / "repo")
    monkeypatch.setattr(
        wheel_build.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="no backend"),
    )

    with pytest.raises(BuildProfileError, match=r"(?s)apps/knob.*no backend"):
        build_deployment_packages(root, [], tmp_path / DEPLOYMENT_PACKAGES_DIRNAME)


@pytest.mark.parametrize("entry", ["./vendor/foo.whl", "./kinds/mistyped", "/opt/foo"])
def test_a_local_path_with_no_package_to_build_is_refused_naming_it(
    tmp_path: Path, entry: str
) -> None:
    """Passed through, it would break the image build and put a host path in the Dockerfile."""
    root = _deployment(tmp_path / "repo")

    with pytest.raises(BuildProfileError) as refusal:
        build_deployment_packages(root, ["numpy>=1.24", entry], tmp_path / "stage")

    assert repr(entry) in str(refusal.value)
    assert "pyproject.toml" in str(refusal.value)


# --- the two spellings of one list --------------------------------------------


def test_each_local_path_entry_is_substituted_in_place_and_apps_follow(tmp_path: Path) -> None:
    _root, _dest, packages = _built(tmp_path)

    assert deployment_dependencies(DEPENDENCIES, packages) == [
        "numpy>=1.24",
        "ring-kind==0.2.0",
        "knob-app==0.1.0",
    ]
    install = deployment_dependencies(DEPENDENCIES, packages, install=True)
    assert install[0] == "numpy>=1.24"
    assert install[1] == f"ring-kind @ {packages[1].wheel.resolve().as_uri()}"
    assert install[2] == f"knob-app @ {packages[0].wheel.resolve().as_uri()}"


def test_the_plan_carries_both_spellings_and_the_wheels(tmp_path: Path) -> None:
    _root, _dest, packages = _built(tmp_path)

    plan = _plan_project_venv(_profile(DEPENDENCIES), packages)

    assert plan.deployment_packages == packages
    assert plan.extra_deps == ["numpy>=1.24", "ring-kind==0.2.0", "knob-app==0.1.0"]
    assert [d.split(" @ ")[0] for d in plan.install_deps] == [
        "numpy>=1.24",
        "ring-kind",
        "knob-app",
    ]


def test_a_distribution_is_never_named_twice(tmp_path: Path) -> None:
    _root, _dest, packages = _built(tmp_path)

    plan = _plan_project_venv(_profile([*DEPENDENCIES, "ring-kind"]), packages)

    assert plan.extra_deps == ["numpy>=1.24", "ring-kind==0.2.0", "knob-app==0.1.0"]
    assert len(plan.install_deps) == 3


def test_two_spellings_of_one_local_path_are_one_package(tmp_path: Path) -> None:
    root = _deployment(tmp_path / "repo")
    dependencies = ["./kinds/ring", "-e ./kinds/ring", "numpy>=1.24"]

    packages = build_deployment_packages(root, dependencies, tmp_path / "stage")

    assert deployment_dependencies(dependencies, packages) == [
        "ring-kind==0.2.0",
        "numpy>=1.24",
        "knob-app==0.1.0",
    ]


def test_a_symlinked_root_enumerates_an_application_named_in_the_dependencies_once(
    tmp_path: Path,
) -> None:
    from osprey.apps.rules import deployment_packages

    real = _deployment(tmp_path / "repo")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    found = deployment_packages(link, ["./apps/knob"])

    assert found == [(real / "apps" / "knob" / "pyproject.toml").resolve()]


def test_an_entry_naming_an_application_package_is_that_package_in_place(
    tmp_path: Path,
) -> None:
    root = _deployment(tmp_path / "repo")
    dependencies = ["./apps/knob", *DEPENDENCIES]
    packages = build_deployment_packages(root, dependencies, tmp_path / "stage")

    assert [p.name for p in packages] == ["knob-app", "ring-kind"]
    assert deployment_dependencies(dependencies, packages) == [
        "knob-app==0.1.0",
        "numpy>=1.24",
        "ring-kind==0.2.0",
    ]


def test_the_venv_resolves_the_wheels_with_the_framework_and_records_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An app requiring lume-base 0.6 is resolved WITH the framework's pins.

    One install command carries the framework spec and every wheel, so a
    requirement that would move a pinned version fails the resolve instead of
    upgrading it; nothing installs a wheel on its own or with ``--no-deps``.
    """
    _root, _dest, packages = _built(tmp_path)
    plan = _plan_project_venv(_profile(DEPENDENCIES), packages)
    calls: list[list[str]] = []

    def run(cmd: list[str], **_kw: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("osprey.cli.build_environment.subprocess.run", run)
    project = tmp_path / "build"
    project.mkdir()
    _create_project_venv(project, plan)

    installs = [cmd for cmd in calls if "install" in cmd]
    assert len(installs) == 1
    assert plan.osprey_spec in installs[0]
    assert all(dep in installs[0] for dep in plan.install_deps)
    assert "--no-deps" not in installs[0]
    record = (project / "pyproject.toml").read_text(encoding="utf-8")
    assert '"knob-app==0.1.0"' in record and '"ring-kind==0.2.0"' in record
    assert str(tmp_path) not in record


def test_a_registered_kind_with_no_shipped_package_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root, _dest, packages = _built(tmp_path)
    kinds = [
        SimpleNamespace(name="ring", dist=SimpleNamespace(name="ring-kind")),
        SimpleNamespace(name="halo", dist=SimpleNamespace(name="Halo_Kind")),
    ]
    monkeypatch.setattr(wheel_build, "entry_points", lambda group: kinds)

    with pytest.raises(BuildProfileError) as refused:
        _plan_project_venv(_profile(DEPENDENCIES), packages)

    message = str(refused.value)
    assert "'halo'" in message and "Halo_Kind" in message
    assert "the kind's package is not among the packages the project image installs" in message
    assert "'ring'" not in message


def test_a_kind_shipped_as_an_index_dependency_is_not_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kinds = [SimpleNamespace(name="halo", dist=SimpleNamespace(name="halo-kind"))]
    monkeypatch.setattr(wheel_build, "entry_points", lambda group: kinds)

    plan = _plan_project_venv(_profile(["Halo_Kind>=1"]), ())

    assert plan.extra_deps == ["Halo_Kind>=1"]


# --- the Dockerfile ----------------------------------------------------------


def _dockerfile(
    pip_dependency_args: str, packages_dir: str, wheels: tuple[dict[str, str], ...] = ()
) -> str:
    from importlib import resources

    from osprey.deployment.members import WORKSPACE_MEMBERS
    from osprey.port_layout import DEFAULT_PORT_BASE, layout_ports

    source = resources.files("osprey").joinpath("templates/project/Dockerfile.j2")
    return Template(source.read_text(encoding="utf-8")).render(
        project_name="fixture",
        claude_code_cli_version="1.2.3",
        port_base=DEFAULT_PORT_BASE,
        osprey_ports=layout_ports(DEFAULT_PORT_BASE),
        workspace_members=list(WORKSPACE_MEMBERS),
        pip_dependency_args=pip_dependency_args,
        deployment_packages_dir=packages_dir,
        deployment_wheels=list(wheels),
    )


def _install_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if "uv pip install" in line and "osprey" in line]


KNOB_WHEEL = {"name": "knob-app", "wheel": "knob_app-0.1.0-py3-none-any.whl"}


def test_the_image_installs_the_wheels_on_the_framework_line() -> None:
    text = _dockerfile("numpy>=1.24 knob-app==0.1.0", DEPLOYMENT_PACKAGES_DIRNAME, (KNOB_WHEEL,))
    staged = f"/tmp/deps-ctx/{DEPLOYMENT_PACKAGES_DIRNAME}/{KNOB_WHEEL['wheel']}"
    bound = f'"knob-app @ file://{staged}"'

    copy = f"COPY {DEPLOYMENT_PACKAGES_DIRNAME}/ /tmp/deps-ctx/{DEPLOYMENT_PACKAGES_DIRNAME}/"
    assert copy in text
    assert text.index(copy) < text.index("--mount=type=cache,target=/root/.cache/uv")
    primer, fallback = (
        line for line in _install_lines(text) if "$OSPREY_PIP_SPEC" in line or "framework " in line
    )
    for line in (primer, fallback):
        assert "knob-app==0.1.0" in line and bound in line
        # The name is bound to the staged wheel, never looked up: a rebuilt
        # wheel of an unchanged version is not served from the uv cache, and
        # no index distribution of that name competes with it.
        assert "--refresh-package knob-app" in line
    assert "--find-links" not in text
    # A registry build stages no local-requirements file; the wheels are not
    # inside that file's branch, so they install all the same.
    local = next(line for line in text.splitlines() if "osprey-local-requirements.txt;" in line)
    assert staged not in local


def test_a_deployment_without_packages_renders_no_copy_and_no_binding() -> None:
    text = _dockerfile("numpy>=1.24", "")

    assert DEPLOYMENT_PACKAGES_DIRNAME not in text
    assert "--refresh-package" not in text and "file:///tmp/deps-ctx" not in text


# --- end to end: every image context ------------------------------------------


def test_every_image_context_carries_the_wheels_and_installs_them(tmp_path: Path) -> None:
    """The deployment's image and a login's render image, from one build."""
    from osprey.cli.build_cmd import build as build_command
    from osprey.cli.init_cmd import init

    repo = tmp_path / "pkgs"
    runner = CliRunner()
    created = runner.invoke(init, [str(repo), "--preset", "hello-world", "--no-git"])
    assert created.exit_code == 0, created.output
    _deployment(repo)
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    profile["dependencies"] = list(DEPENDENCIES)
    (repo / "profile.yml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    (repo / "tiers").mkdir(exist_ok=True)
    (repo / "tiers" / "viewer.yml").write_text(
        "name: Viewer\nconfig:\n  control_system.writes: never\n", encoding="utf-8"
    )

    result = runner.invoke(build_command, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert result.exit_code == 0, result.output

    contexts = sorted((repo / BUILD_DIR_NAME / IMAGE_DIR_NAME).iterdir())
    assert [c.name for c in contexts] == ["pkgs", "pkgs-viewer"]
    for context in contexts:
        staged = context / DEPLOYMENT_PACKAGES_DIRNAME
        wheels = sorted(p.name for p in staged.glob("*.whl"))
        assert wheels == ["knob_app-0.1.0-py3-none-any.whl", "ring_kind-0.2.0-py3-none-any.whl"]
        assert not list(context.glob("*.whl")), "a root wheel would be reinstalled as osprey"
        dockerfile = (context / BUILD_DIR_NAME / "Dockerfile").read_text(encoding="utf-8")
        primer = next(line for line in _install_lines(dockerfile) if "$OSPREY_PIP_SPEC" in line)
        assert "ring-kind==0.2.0" in primer and "knob-app==0.1.0" in primer
        for wheel in wheels:
            assert f"file:///tmp/deps-ctx/{DEPLOYMENT_PACKAGES_DIRNAME}/{wheel}" in primer
        assert "./kinds/ring" not in dockerfile
        assert not re.search(re.escape(str(tmp_path)), dockerfile)


def _skip_deps_build_with_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **declared: list[str]
) -> Any:
    """``osprey build --skip-deps`` with kind ``halo`` registered by ``halo-kind``."""
    from osprey.cli.build_cmd import build as build_command
    from osprey.cli.init_cmd import init

    kinds = [SimpleNamespace(name="halo", dist=SimpleNamespace(name="halo-kind"))]
    monkeypatch.setattr(wheel_build, "entry_points", lambda group: kinds)
    repo = tmp_path / "pkgs"
    runner = CliRunner()
    created = runner.invoke(init, [str(repo), "--preset", "hello-world", "--no-git"])
    assert created.exit_code == 0, created.output
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    if "packages" in declared:
        profile.setdefault("environment", {})["packages"] = declared["packages"]
    if "dependencies" in declared:
        profile["dependencies"] = declared["dependencies"]
    (repo / "profile.yml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    return runner.invoke(build_command, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])


def test_a_skip_deps_build_judges_the_kinds_over_what_its_image_installs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``--skip-deps`` installs the dependencies only: environment packages do not ship."""
    result = _skip_deps_build_with_kind(tmp_path, monkeypatch, packages=["halo-kind>=1"])

    assert result.exit_code != 0
    # The build reports a refusal through its logger, not on stdout.
    assert "'halo'" in caplog.text
    assert "halo-kind" in caplog.text
    assert "not among the packages the project image installs" in caplog.text


def test_a_skip_deps_build_ships_a_kind_named_in_the_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _skip_deps_build_with_kind(tmp_path, monkeypatch, dependencies=["halo-kind>=1"])

    assert result.exit_code == 0, result.output


def test_a_skip_deps_build_names_a_distribution_once(tmp_path: Path) -> None:
    from osprey.cli.build_cmd import build as build_command
    from osprey.cli.init_cmd import init

    repo = tmp_path / "pkgs"
    runner = CliRunner()
    created = runner.invoke(init, [str(repo), "--preset", "hello-world", "--no-git"])
    assert created.exit_code == 0, created.output
    _deployment(repo)
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    profile["dependencies"] = ["./kinds/ring", "ring-kind"]
    (repo / "profile.yml").write_text(yaml.safe_dump(profile), encoding="utf-8")

    result = runner.invoke(build_command, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert result.exit_code == 0, result.output

    dockerfile = (repo / BUILD_DIR_NAME / "Dockerfile").read_text(encoding="utf-8")
    primer = next(line for line in _install_lines(dockerfile) if "$OSPREY_PIP_SPEC" in line)
    assert '"$OSPREY_PIP_SPEC" ring-kind==0.2.0 knob-app==0.1.0 --refresh-package' in primer


@pytest.mark.parametrize(
    ("operator", "expected"), [(None, "315532800"), ("1700000000", "1700000000")]
)
def test_a_wheel_build_is_reproducible_unless_the_operator_says_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operator: str | None, expected: str
) -> None:
    """``SOURCE_DATE_EPOCH`` defaults to the earliest zip time; the operator's wins."""
    recorded: dict[str, Any] = {}

    def fake(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        flag = "--out-dir" if "--out-dir" in cmd else "--wheel-dir"
        (Path(cmd[cmd.index(flag) + 1]) / "x-1.0-py3-none-any.whl").write_bytes(b"")
        recorded["env"] = kwargs["env"]
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(
        wheel_build,
        "subprocess",
        SimpleNamespace(run=fake, TimeoutExpired=subprocess.TimeoutExpired),
    )
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    if operator is not None:
        monkeypatch.setenv("SOURCE_DATE_EPOCH", operator)
    out = tmp_path / "out"
    out.mkdir()

    _REAL_BUILD_WHEEL(tmp_path, out)

    assert recorded["env"]["SOURCE_DATE_EPOCH"] == expected
