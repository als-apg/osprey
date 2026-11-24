"""The release-stream table resolves exactly the component tags, and agrees with the repository.

Every ref is the full ``refs/tags/...`` string ``GITHUB_REF`` carries.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = REPO_ROOT / "scripts" / "ci" / "release_streams.py"
_spec = importlib.util.spec_from_file_location("release_streams", _MODULE_PATH)
assert _spec and _spec.loader
rs = importlib.util.module_from_spec(_spec)
# import-time required because scripts/ is not a package: register before exec so
# @dataclass resolves annotations, and the stream table feeds module-level cases.
sys.modules[_spec.name] = rs
_spec.loader.exec_module(rs)

_cf_spec = importlib.util.spec_from_file_location(
    "changelog_fragments", REPO_ROOT / "scripts" / "changelog_fragments.py"
)
assert _cf_spec and _cf_spec.loader
cf = sys.modules.get("changelog_fragments") or importlib.util.module_from_spec(_cf_spec)
if "changelog_fragments" not in sys.modules:
    sys.modules[_cf_spec.name] = cf
    _cf_spec.loader.exec_module(cf)

BY_KEY = {stream.key: stream for stream in rs.STREAMS}

#: GitHub's single-line output form: ``name=value``, the name an identifier,
#: the value free of newlines; ``name<<DELIM`` is the multi-line form.
OUTPUT_LINE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*=[^\r\n]*$")
RESOLVE_NAMES = [
    "key",
    "dist",
    "import_name",
    "package_dir",
    "environment",
    "changelog",
    "tag_prefix",
    "lean_extras",
    "smoke",
    "version",
    "prerelease",
]


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(_MODULE_PATH), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _parse_outputs(text: str) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for line in text.splitlines():
        assert OUTPUT_LINE.match(line), f"not a single-line name=value output: {line!r}"
        assert "<<" not in line.split("=", 1)[0], f"delimiter form: {line!r}"
        name, value = line.split("=", 1)
        assert name not in outputs, f"duplicate output {name}"
        outputs[name] = value
    return outputs


def test_the_table_holds_connectors_facility_panels_in_order():
    assert [stream.key for stream in rs.STREAMS] == ["connectors", "facility", "panels"]


EXPECTED = {
    "connectors": {
        "dist": "osprey-connectors",
        "import_name": "osprey_connectors",
        "package_dir": "packages/osprey-connectors",
        "environment": "pypi-connectors",
        "changelog": "packages/osprey-connectors/CHANGELOG.md",
        "tag_prefix": "osprey-connectors-v",
        "lean_extras": "",
        "smoke": "",
    },
    "facility": {
        "dist": "osprey-facility",
        "import_name": "osprey_facility",
        "package_dir": "packages/osprey-facility",
        "environment": "pypi-facility",
        "changelog": "packages/osprey-facility/CHANGELOG.md",
        "tag_prefix": "osprey-facility-v",
        "lean_extras": "mml",
        "smoke": "",
    },
    "panels": {
        "dist": "osprey-panels",
        "import_name": "osprey_panels",
        "package_dir": "packages/osprey-panels",
        "environment": "pypi-panels",
        "changelog": "packages/osprey-panels/CHANGELOG.md",
        "tag_prefix": "osprey-panels-v",
        "lean_extras": "",
        "smoke": "bin/osprey-panels --help",
    },
}


@pytest.mark.parametrize(
    ("version", "prerelease"),
    [
        ("2026.12.0", "false"),
        ("2026.12.10", "false"),
        ("2026.12.0b1", "true"),
        ("2026.12.0rc1", "true"),
        ("2026.12.0a2", "true"),
        ("2026.12.0.dev1", "true"),
        ("2026.12.0rc1.dev3", "true"),
    ],
    ids=["final", "final-two-digit", "beta", "rc", "alpha", "dev", "rc-dev"],
)
@pytest.mark.parametrize("key", list(EXPECTED))
def test_each_tag_glob_resolves_to_its_streams_fields(key, version, prerelease, tmp_path):
    output = tmp_path / "github_output"
    output.write_text("earlier=kept\n")
    ref = f"refs/tags/{EXPECTED[key]['tag_prefix']}{version}"
    result = _run("resolve", ref, env={"GITHUB_OUTPUT": str(output)})
    assert result.returncode == 0, result.stderr

    written = _parse_outputs(output.read_text())
    assert written.pop("earlier") == "kept", "resolve must append, not truncate"
    assert list(written) == RESOLVE_NAMES
    assert written == {"key": key, **EXPECTED[key], "version": version, "prerelease": prerelease}
    assert _parse_outputs(result.stdout) == written


def test_the_table_fields_match_the_contract():
    connectors, facility, panels = (BY_KEY[k] for k in ("connectors", "facility", "panels"))
    for stream in rs.STREAMS:
        assert stream.tag_glob == f"{stream.tag_prefix}2*"
        assert stream.lean_entry_modules[0] == stream.import_name
        assert "osprey" in stream.lean_forbidden
        assert stream.import_name not in stream.lean_forbidden
    assert connectors.typed and not facility.typed and not panels.typed
    assert connectors.compat and facility.compat and not panels.compat
    assert connectors.floor_consumers == ("osprey", "pyaml_cs_osprey")
    assert facility.floor_consumers == ("osprey",)
    assert panels.floor_consumers == ()
    assert connectors.lean_forbidden == frozenset(
        {"litellm", "openai", "anthropic", "fastapi", "playwright", "osprey"}
    )
    assert connectors.lean_entry_modules == (
        "osprey_connectors",
        "osprey_connectors.config",
        "osprey_connectors.control_system",
        "osprey_connectors.control_system._epics_connector",
        "osprey_connectors.errors",
        "osprey_connectors.logger",
        "osprey_connectors.simulation",
    )


def test_records_are_frozen():
    with pytest.raises(AttributeError):
        rs.STREAMS[0].key = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "ref",
    [
        "refs/tags/osprey-facility-v2026.12.0-rc1",
        "refs/tags/osprey-facility-v2026.12.0.RC1",
        "refs/tags/osprey-facility-v2026.01.0",
        "refs/tags/osprey-facility-v2026.12.01",
        "refs/tags/osprey-panels-v2026.12.0rc01",
        "refs/tags/osprey-connectors-v2026.12",
        "refs/tags/osprey-connectors-v2026.12.0.post1",
        "refs/tags/osprey-connectors-v2026.12.0+local",
    ],
    ids=[
        "dash-rc",
        "upper-rc",
        "zero-padded-month",
        "zero-padded-micro",
        "zero-padded-rc",
        "two-part",
        "post",
        "local",
    ],
)
def test_a_non_canonical_version_is_refused_not_normalised(ref):
    for command in ("resolve", "lanes"):
        result = _run(command, ref, env={})
        assert result.returncode != 0
        assert "canonical" in result.stderr
        assert result.stdout == ""


@pytest.mark.parametrize(
    "ref",
    [
        "osprey-facility-v2026.12.0",
        "tags/osprey-facility-v2026.12.0",
        "v2026.12.0",
        "refs/tags/v2026.12.0",
        "refs/heads/main",
        "refs/heads/osprey-facility-v2026.12.0",
        "refs/tags/osprey-connectors-v0.1.0",
        "refs/tags/osprey-optimizer-v2026.12.0",
        "",
    ],
    ids=[
        "bare-tag",
        "partial-ref",
        "bare-framework-tag",
        "framework-tag",
        "branch",
        "branch-named-like-a-tag",
        "retired-connectors",
        "unknown-package",
        "empty",
    ],
)
def test_a_ref_outside_the_tag_globs_is_refused_and_the_globs_named(ref, tmp_path):
    output = tmp_path / "github_output"
    for command in ("resolve", "lanes"):
        result = _run(command, ref, env={"GITHUB_OUTPUT": str(output)})
        assert result.returncode != 0
        assert result.stdout == ""
        for stream in rs.STREAMS:
            assert f"refs/tags/{stream.tag_glob}" in result.stderr
    assert not output.exists()


def test_in_process_refusal_raises_stream_error():
    with pytest.raises(rs.StreamError):
        rs.stream_for("osprey-panels-v2026.12.0")
    assert rs.main(["resolve", "refs/heads/main"]) == 1


EXPECTED_LANES = {
    "connectors": [
        "test=Test Python",
        "lint=Lint and Code Quality",
        "package=Package Build",
        "connectors-standalone=Connectors Standalone Tests",
        "connectors-compat=Connectors Compat",
        "framework-at-connectors-floor=Framework at Connectors Floor Tests",
    ],
    "facility": [
        "test=Test Python",
        "lint=Lint and Code Quality",
        "package=Package Build",
        "osprey-facility-standalone=Facility Standalone Tests",
        "facility-compat=Facility Compat",
        "framework-at-facility-floor=Framework at Facility Floor Tests",
        "framework-at-connectors-floor=Framework at Connectors Floor Tests",
    ],
    "panels": [
        "test=Test Python",
        "lint=Lint and Code Quality",
        "osprey-panels-standalone=Panels Standalone Tests",
    ],
}


@pytest.mark.parametrize("key", list(EXPECTED_LANES))
def test_lanes_prints_the_streams_id_name_lines(key):
    ref = f"refs/tags/{BY_KEY[key].tag_prefix}2026.12.0b1"
    result = _run("lanes", ref, env={})
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == EXPECTED_LANES[key]
    # The form require_green_lanes.parse_lanes reads: unique ids, both sides non-empty.
    ids = [line.split("=", 1)[0] for line in EXPECTED_LANES[key]]
    assert len(ids) == len(set(ids))
    assert all(a and b for a, b in (line.split("=", 1) for line in EXPECTED_LANES[key]))


def test_dists_prints_each_streams_wheel_prefix():
    result = _run("dists", env={})
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "osprey_connectors-",
        "osprey_facility-",
        "osprey_panels-",
    ]


def test_dists_joined_fits_one_name_value_line():
    result = _run("dists", "--joined", env={})
    assert result.returncode == 0, result.stderr
    value = result.stdout.rstrip("\n")
    assert "\n" not in value and "\r" not in value
    assert _parse_outputs(f"dists={value}") == {"dists": value}
    assert shlex.split(value) == ["osprey_connectors-", "osprey_facility-", "osprey_panels-"]


def test_prefixes_and_changelogs_equal_the_changelog_trees():
    trees = [tree for tree in cf.TREES if tree.root]
    assert [(s.tag_prefix, s.changelog) for s in rs.STREAMS] == [
        (tree.tag_prefix, f"{tree.prefix}CHANGELOG.md") for tree in trees
    ]
    assert [s.package_dir for s in rs.STREAMS] == [tree.root for tree in trees]


def _pyproject(stream) -> dict:
    return tomllib.loads((REPO_ROOT / stream.package_dir / "pyproject.toml").read_text())


@pytest.mark.parametrize("key", list(EXPECTED))
def test_every_package_dir_exists_and_names_its_dist(key):
    stream = BY_KEY[key]
    package = REPO_ROOT / stream.package_dir
    assert package.is_dir()
    assert (REPO_ROOT / stream.changelog).is_file()
    assert (package / "src" / stream.import_name / "__init__.py").is_file()
    assert _pyproject(stream)["project"]["name"] == stream.dist
    assert stream.dist_prefix == stream.dist.replace("-", "_") + "-"


@pytest.mark.parametrize("key", list(EXPECTED))
def test_every_tag_prefix_matches_the_hatch_vcs_describe_glob(key):
    stream = BY_KEY[key]
    raw = _pyproject(stream)["tool"]["hatch"]["version"]["raw-options"]
    words = shlex.split(raw["scm"]["git"]["describe_command"])
    glob = words[words.index("--match") + 1]
    assert glob == f"{stream.tag_prefix}2[0-9][0-9][0-9].*"
    # Both globs accept a release and refuse the retired pre-calendar line.
    for tag, accepted in (
        (f"{stream.tag_prefix}2026.12.0", True),
        (f"{stream.tag_prefix}2026.12.0rc1", True),
        (f"{stream.tag_prefix}0.1.0", False),
    ):
        assert fnmatch.fnmatchcase(tag, glob) is accepted
        assert fnmatch.fnmatchcase(tag, stream.tag_glob) is accepted


def test_the_table_is_stdlib_only():
    imported = set()
    for line in _MODULE_PATH.read_text().splitlines():
        match = re.match(r"^(?:from|import) ([A-Za-z_][\w]*)", line)
        if match:
            imported.add(match.group(1))
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported
