"""pyAML reaches hardware only through the OSPREY control system.

pyAML ships native control-system backends — ``pyaml-cs-oa`` (ophyd-async over
EPICS or Tango) and ``tango-pyaml`` — that would open a second route to the
machine, bypassing the connector's limits, approval, and audit chain. OSPREY
refuses them in two layers:

- **Configuration**: none of those backends is installed, so a pyAML
  configuration naming one fails to build with ``PyAMLConfigException``.
- **Dependency closure**: nothing OSPREY resolves or installs pulls them in —
  not the lock file, not a bundled preset's ``dependencies``, not the pip
  arguments of a rendered Dockerfile. ``accelerator-middle-layer`` may appear,
  but never with its ``cs-oa-epics``, ``cs-oa-tango`` or ``tango-pyaml`` extras.

These tests pin that neither layer regresses. Should a native backend still
reach a sandbox, the runtime refusals are the backstop: the workspace sandbox
refuses importing ``aioca`` (the EPICS transport ophyd-async uses), the Python
executor's write surface classifies ``aioca`` ``caput`` and ``ophyd_async``
``SignalW``/``SignalRW`` ``set`` and ``SignalX`` ``trigger`` as hardware
writes, and the executor wrapper rebinds ``aioca.caput`` to a limits-checked
call.
"""

from __future__ import annotations

import copy
import pathlib
import re
import tomllib
from typing import Any

import pytest
from click.testing import CliRunner
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from pyaml.accelerator import Accelerator
from pyaml.common.exception import PyAMLConfigException

from osprey.cli.build_profile_presets import _load_preset_raw, list_presets
from osprey.cli.main import cli

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
UV_LOCK = REPO_ROOT / "uv.lock"

FORBIDDEN_PACKAGES = frozenset({"pyaml-cs-oa", "tango-pyaml"})
FORBIDDEN_EXTRAS = frozenset({"cs-oa-epics", "cs-oa-tango", "tango-pyaml"})
AML_PACKAGE = "accelerator-middle-layer"

# Spellings of a forbidden name in free text: any of `-`, `_`, `.` between the
# words (pip and PEP 503 treat them alike). The trailing guard keeps a longer
# name such as ``pyaml-cs-osprey`` from matching.
_SEP = r"[-_.]"
_FORBIDDEN_NAME_RE = re.compile(
    rf"(?<![\w-])(?:pyaml{_SEP}cs{_SEP}oa|tango{_SEP}pyaml)(?![\w-])", re.IGNORECASE
)
_AML_EXTRAS_RE = re.compile(
    rf"(?<![\w-])accelerator{_SEP}middle{_SEP}layer\s*\[([^\]]*)\]", re.IGNORECASE
)

_FRAMEWORK_EXTRAS_RE = re.compile(
    r"""(?:osprey[-_.]framework|\$\{whl\}|["'\s]\.)\[([^\]]+)\]""", re.IGNORECASE
)

#: Images whose build installs every staged workspace member beside the
#: framework: the two chat-bridge services, the attached VA service, and the
#: standalone VA image.
IMAGE_FILES = (
    REPO_ROOT / "src/osprey/templates/services/gchat_bridge/Dockerfile",
    REPO_ROOT / "src/osprey/templates/services/teams_bridge/Dockerfile",
    REPO_ROOT / "src/osprey/templates/services/virtual_accelerator/Dockerfile",
    REPO_ROOT / "docker/virtual-accelerator/Containerfile",
)
MEMBER_DIRS = tuple(sorted(p.parent for p in (REPO_ROOT / "packages").glob("*/pyproject.toml")))

NATIVE_BACKEND_CONFIG: dict[str, Any] = {
    "type": "pyaml.accelerator",
    "facility": "Test",
    "machine": "sr",
    "energy": 1.0e9,
    "controls": [{"type": "pyaml_cs_oa.controlsystem", "name": "live"}],
    "devices": [],
}


def _requirement_violations(spec: str) -> list[str]:
    """Name what a pip requirement string pulls in that the closure forbids."""
    try:
        req = Requirement(spec)
    except InvalidRequirement:
        return _text_violations(spec)
    name = canonicalize_name(req.name)
    if name in FORBIDDEN_PACKAGES:
        return [name]
    if name == AML_PACKAGE:
        bad = sorted({canonicalize_name(e) for e in req.extras} & FORBIDDEN_EXTRAS)
        return [f"{AML_PACKAGE}[{e}]" for e in bad]
    return []


def _text_violations(text: str) -> list[str]:
    """Name every forbidden package or extra mentioned anywhere in *text*."""
    found = [m.group(0) for m in _FORBIDDEN_NAME_RE.finditer(text)]
    for match in _AML_EXTRAS_RE.finditer(text):
        extras = {canonicalize_name(e.strip()) for e in match.group(1).split(",") if e.strip()}
        found += [f"{AML_PACKAGE}[{e}]" for e in sorted(extras & FORBIDDEN_EXTRAS)]
    return found


def _lock_violations(lock: dict[str, Any]) -> list[str]:
    """Name forbidden packages and extras in a parsed ``uv.lock``."""
    found: list[str] = []
    for package in lock.get("package", []):
        name = canonicalize_name(package.get("name", ""))
        if name in FORBIDDEN_PACKAGES:
            found.append(f"package {name}")
        dep_lists = [package.get("dependencies", [])]
        dep_lists += list(package.get("optional-dependencies", {}).values())
        dep_lists += list(package.get("dev-dependencies", {}).values())
        for deps in dep_lists:
            for dep in deps:
                dep_name = canonicalize_name(dep.get("name", ""))
                if dep_name in FORBIDDEN_PACKAGES:
                    found.append(f"{name} -> {dep_name}")
                if dep_name == AML_PACKAGE:
                    extras = {canonicalize_name(e) for e in dep.get("extra", [])}
                    found += [
                        f"{name} -> {AML_PACKAGE}[{e}]" for e in sorted(extras & FORBIDDEN_EXTRAS)
                    ]
        for req in package.get("metadata", {}).get("requires-dist", []):
            spec = req.get("name", "")
            extras = req.get("extras", [])
            if extras:
                spec += "[" + ",".join(extras) + "]"
            found += [f"{name} requires {v}" for v in _requirement_violations(spec)]
    return found


def _pip_install_lines(dockerfile: str) -> list[str]:
    """Return the Dockerfile lines that invoke ``pip install``."""
    return [line for line in dockerfile.splitlines() if "pip install" in line]


def _pip_install_commands(dockerfile: str) -> list[str]:
    """Return each backslash-continued logical line that invokes ``pip install``,
    so a requirement on a continuation line is scanned with its command."""
    logical: list[str] = []
    current = ""
    for line in dockerfile.splitlines():
        if line.lstrip().startswith("#"):
            continue
        current += " " + line.rstrip()
        if current.endswith("\\"):
            current = current[:-1]
            continue
        logical.append(current)
        current = ""
    logical.append(current)
    return [command for command in logical if "pip install" in command]


def _framework_extras(command: str) -> set[str]:
    """The framework extras a pip command installs: ``osprey-framework[x]``,
    a staged framework wheel ``${whl}[x]``, or the checkout itself ``.[x]``."""
    found: set[str] = set()
    for match in _FRAMEWORK_EXTRAS_RE.finditer(command):
        found |= {canonicalize_name(e.strip()) for e in match.group(1).split(",") if e.strip()}
    return found


def _extra_violations(pyproject: dict[str, Any], extra: str) -> list[str]:
    """Name what one framework extra pulls in, following self-references."""
    extras = pyproject["project"]["optional-dependencies"]
    by_name = {canonicalize_name(name): specs for name, specs in extras.items()}
    seen: set[str] = set()
    pending = [canonicalize_name(extra)]
    found: list[str] = []
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        assert name in by_name, f"osprey-framework has no extra {name!r}"
        for spec in by_name[name]:
            req = Requirement(spec)
            if canonicalize_name(req.name) == "osprey-framework":
                pending += [canonicalize_name(e) for e in req.extras]
            else:
                found += _requirement_violations(spec)
    return found


def _render_dockerfile(repo: pathlib.Path, preset: str) -> str:
    """Materialize a deployment from *preset* and return its rendered Dockerfile."""
    result = CliRunner().invoke(cli, ["init", str(repo), "--preset", preset, "--no-git"])
    assert result.exit_code == 0, result.output
    result = CliRunner().invoke(
        cli, ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"]
    )
    assert result.exit_code == 0, result.output
    return (repo / "build" / "Dockerfile").read_text()


class TestNativeBackendConfigurationRefused:
    """A configuration naming a native backend does not build."""

    @pytest.mark.parametrize(
        "controls_type",
        ["pyaml_cs_oa.controlsystem", "tango.pyaml.controlsystem"],
        ids=["pyaml-cs-oa", "tango-pyaml"],
    )
    def test_native_controls_block_raises_config_exception(self, controls_type):
        config = copy.deepcopy(NATIVE_BACKEND_CONFIG)
        config["controls"] = [{"type": controls_type, "name": "live"}]
        with pytest.raises(PyAMLConfigException, match=re.escape(controls_type)):
            Accelerator.from_dict(config)

    def test_osprey_controls_block_builds(self):
        """The same configuration on the OSPREY control system builds, so the
        refusal above is about the backend and not a malformed configuration."""
        config = copy.deepcopy(NATIVE_BACKEND_CONFIG)
        config["controls"] = [{"type": "pyaml_cs_osprey.controlsystem", "name": "live"}]
        assert Accelerator.from_dict(config) is not None


class TestViolationDetectors:
    """The closure checks below catch what they claim to catch."""

    @pytest.mark.parametrize(
        "spec",
        [
            "pyaml-cs-oa",
            "pyaml_cs_oa>=0.1",
            "Tango-PyAML",
            "accelerator-middle-layer[cs-oa-epics]",
            "accelerator-middle-layer[cs_oa_tango]==0.3.1",
            "accelerator-middle-layer[tango-pyaml]",
        ],
    )
    def test_forbidden_requirement_detected(self, spec):
        assert _requirement_violations(spec)
        assert _text_violations(f"pip install 'numpy' '{spec}'")

    @pytest.mark.parametrize(
        "spec",
        [
            "accelerator-middle-layer",
            "accelerator-middle-layer==0.3.*",
            "pyaml-cs-osprey",
            "accelerator-middle-layer[docs]",
        ],
    )
    def test_allowed_requirement_passes(self, spec):
        assert _requirement_violations(spec) == []
        assert _text_violations(f"pip install '{spec}'") == []

    def test_lock_detector_sees_extra_on_a_dependency_edge(self):
        lock = {
            "package": [
                {
                    "name": "pyaml-cs-osprey",
                    "dependencies": [{"name": AML_PACKAGE, "extra": ["cs-oa-epics"]}],
                },
            ]
        }
        assert _lock_violations(lock) == [f"pyaml-cs-osprey -> {AML_PACKAGE}[cs-oa-epics]"]

    def test_lock_detector_sees_forbidden_package(self):
        assert _lock_violations({"package": [{"name": "pyaml_cs_oa"}]}) == ["package pyaml-cs-oa"]


class TestDependencyClosure:
    """Nothing OSPREY resolves or installs pulls in a native backend."""

    def test_uv_lock_names_no_native_backend(self):
        text = UV_LOCK.read_text()
        assert _lock_violations(tomllib.loads(text)) == []
        assert _text_violations(text) == []

    @pytest.mark.parametrize("preset", list_presets())
    def test_preset_dependencies_name_no_native_backend(self, preset):
        raw, _ = _load_preset_raw(preset)
        dependencies = raw.get("dependencies") or []
        violations = [v for spec in dependencies for v in _requirement_violations(str(spec))]
        assert violations == [], f"preset {preset!r} depends on {violations}"

    @pytest.mark.parametrize("image", IMAGE_FILES, ids=[p.parent.name for p in IMAGE_FILES])
    def test_image_file_installs_no_native_backend(self, image):
        """The service images and the VA image install every staged workspace
        member beside the framework extra they serve. Each file is copied into
        its build context verbatim, so the source is what the build runs."""
        text = image.read_text()
        assert "{{" not in text and "{%" not in text, f"{image} became a template; render it"
        commands = _pip_install_commands(text)
        assert commands, f"{image} no longer installs anything with pip"
        violations = [v for command in commands for v in _text_violations(command)]
        assert violations == [], f"{image} installs {violations}"

        extras = sorted({e for command in commands for e in _framework_extras(command)})
        assert extras, f"{image} installs no framework extra this test can resolve"
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        for extra in extras:
            found = _extra_violations(pyproject, extra)
            assert found == [], f"{image} installs osprey-framework[{extra}], which pulls {found}"

    @pytest.mark.parametrize("member", MEMBER_DIRS, ids=[p.name for p in MEMBER_DIRS])
    def test_workspace_member_depends_on_no_native_backend(self, member):
        """Every image installs every member, so each member's own requirements
        are part of every image's closure."""
        project = tomllib.loads((member / "pyproject.toml").read_text())["project"]
        specs = list(project.get("dependencies", []))
        for extra_specs in project.get("optional-dependencies", {}).values():
            specs += extra_specs
        violations = [v for spec in specs for v in _requirement_violations(spec)]
        assert violations == [], f"{member.name} depends on {violations}"

    @pytest.mark.parametrize("preset", list_presets())
    def test_rendered_dockerfile_installs_no_native_backend(self, preset, tmp_path):
        dockerfile = _render_dockerfile(tmp_path / "deploy", preset)
        install_lines = _pip_install_lines(dockerfile)
        assert install_lines, "the rendered Dockerfile no longer installs anything with pip"
        violations = [v for line in install_lines for v in _text_violations(line)]
        assert violations == [], f"preset {preset!r} Dockerfile installs {violations}"
