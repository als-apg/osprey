"""With no ``pv_directory`` section, a build renders what ``main`` renders.

Two comparisons, both over every file of the rendered repo (see ``_off_render``):

- Against the baseline ref, rendered live from a scratch worktree: the three
  configurations, the branch side with the section stripped from the config block of
  every profile it renders and with the connectors modules newer than the floor blocked
  from import. Only the allow-list below may differ. Once the baseline holds the
  feature this skips, and the permanent guard holds it.
- From this checkout: demo with ``pv_directory: null`` against demo without the section.

Nothing rendered is stored, so a template change on ``main`` never turns this red.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile_merge import _hash_resolved_profile, _load_preset_raw
from tests.cli import _off_render
from tests.cli._off_render import (
    CONFIGURATIONS,
    DEMO,
    MANIFEST_NAME,
    REPO_NAME,
    ROOT,
    Configuration,
    differing,
    is_excluded,
    manifest_view,
    normalise,
    preset_hash,
    render,
    scratch_worktree,
    snapshot,
)

pytestmark = pytest.mark.slow

SECTION = "pv_directory"
#: A baseline holding this has the feature; the permanent guard then holds the Off render.
LANDED = "src/osprey/services/pv_directory/resolve.py"
#: The connectors modules newer than the connectors floor, blocked in every off render.
NEW_CONNECTORS_MODULES = (
    "osprey_connectors.pv_directory",
    "osprey_connectors.alarm_state",
    "osprey_connectors.alarm_log",
    "osprey_connectors.put_log",
)

BLOCKER = f"""
import sys

BLOCKED = {NEW_CONNECTORS_MODULES!r}


class _Blocked:
    def find_spec(self, name, path=None, target=None):
        if any(name == module or name.startswith(module + ".") for module in BLOCKED):
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
        return None


sys.meta_path.insert(0, _Blocked())
"""

SOURCE_ZONE_MARK = "<source zone>"
#: The repo's own profile and the image's copy of it, left out as source zone: the
#: branch side's is the stripped one, and the hash proof covers its content.
SOURCE_ZONE = ("**/profile.yml",)

_OPT_OUT_LINES = re.compile(
    r"^  # Channel records and alarms are the `channel-finder` login's\.\n"
    r"^  claude_code\.servers\.pv-directory\.enabled: false\n",
    re.MULTILINE,
)
_PROVENANCE_DIGEST = re.compile(r"((?:content hash|preset_hash): sha256:)[0-9a-f]{64}")


def _without_ioc_health(text: str) -> str:
    document = json.loads(text)
    document.pop("ioc_health", None)
    return json.dumps(document, sort_keys=True, indent=1)


def _without_server(text: str) -> str:
    document = yaml.safe_load(text)
    document.get("claude_code", {}).get("servers", {}).pop("pv-directory", None)
    return yaml.safe_dump(document, sort_keys=True)


def _without_opt_out(text: str) -> str:
    return _PROVENANCE_DIGEST.sub(r"\g<1><digest>", _OPT_OUT_LINES.sub("", text))


@dataclass(frozen=True)
class Allowed:
    """Files compared after one stated change is taken out of them, on both sides."""

    globs: tuple[str, ...]
    reason: str
    compare: Callable[[str], str]


#: What the feature necessarily moves in an off render, over every render tree.
ALLOW_LIST = (
    Allowed(
        ("**/data/simulator/addresses.json",),
        "the simulator view lists the IOC health PVs the VA serves",
        _without_ioc_health,
    ),
    Allowed(
        (
            "**/demo-logbook/config.yml",
            "**/demo-knowledge/config.yml",
            "**/.image/demo-logbook/build/config.yml",
            "**/.image/demo-knowledge/build/config.yml",
        ),
        "the logbook and knowledge logins switch the pv-directory server off by name",
        _without_server,
    ),
    Allowed(
        ("**/tiers/logbook.yml", "**/tiers/knowledge.yml"),
        "the opt-out lines, and the provenance digests they move",
        _without_opt_out,
    ),
)


def _without_off_spelling(text: str) -> str:
    document = yaml.safe_load(text)
    if isinstance(document, dict) and SECTION in document and document[SECTION] is None:
        del document[SECTION]
    return yaml.safe_dump(document, sort_keys=True)


#: The Off spelling stays in the rendered config and in each copy of it (the image's,
#: every login's and every service's); the comparison takes out that one key and nothing
#: else.
OFF_SPELLING = Allowed(
    ("**/config.yml",),
    "the rendered config carries the section's null as the resolved config",
    _without_off_spelling,
)


def _match_parts(parts: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    if not pattern:
        return not parts
    if pattern[0] == "**":
        return any(_match_parts(parts[start:], pattern[1:]) for start in range(len(parts) + 1))
    return (
        bool(parts) and fnmatchcase(parts[0], pattern[0]) and _match_parts(parts[1:], pattern[1:])
    )


def _matches(relative: str, globs: tuple[str, ...]) -> bool:
    """Whether a whole relative path matches a glob: ``*`` within one segment, ``**`` across any."""
    parts = PurePosixPath(relative).parts
    return any(_match_parts(parts, PurePosixPath(glob).parts) for glob in globs)


def _rule(relative: str, allowed: tuple[Allowed, ...]) -> Allowed | None:
    return next((rule for rule in allowed if _matches(relative, rule.globs)), None)


class Seen:
    """What the comparison sees of one rendered repo.

    :func:`snapshot`, with each allow-listed file compared through its rule, the source
    zone by path and mode only, and every manifest without the checksum rows of files
    compared another way and without its digests of the source zone.
    """

    def __init__(self, repo: Path, allowed: tuple[Allowed, ...]) -> None:
        self.repo = repo
        self.allowed = allowed
        files = [
            path.relative_to(repo).as_posix()
            for path in repo.rglob("*")
            if stat.S_ISREG(path.lstat().st_mode)
        ]
        self.unseen = frozenset(relative for relative in files if self._compared_apart(relative))
        self.digests = snapshot(repo, unseen_rows=self.unseen)
        for relative in files:
            mode = self.digests[relative].split()[0]
            if _matches(relative, SOURCE_ZONE):
                self.digests[relative] = f"{mode} {SOURCE_ZONE_MARK}"
            elif _rule(relative, allowed) is not None or _is_manifest(relative):
                text = self.text(relative) or ""
                self.digests[relative] = f"{mode} {hashlib.sha256(text.encode()).hexdigest()}"

    def _compared_apart(self, relative: str) -> bool:
        if (
            _matches(relative, SOURCE_ZONE)
            or _rule(relative, self.allowed)
            or is_excluded(relative)
        ):
            return True
        if _is_manifest(relative):
            return False
        content = (self.repo / relative).read_bytes()
        return normalise(relative, content) != content

    def text(self, relative: str) -> str | None:
        """The compared text of a file, or None for a binary or missing one."""
        path = self.repo / relative
        if not path.is_file() or path.is_symlink():
            return None
        try:
            text = normalise(relative, path.read_bytes()).decode("utf-8")
        except UnicodeDecodeError:
            return None
        if _is_manifest(relative):
            return manifest_view(
                text, PurePosixPath(relative).parent, self.unseen, source_zone=False
            )
        rule = _rule(relative, self.allowed)
        return rule.compare(text) if rule is not None else text


def _is_manifest(relative: str) -> bool:
    return PurePosixPath(relative).name == MANIFEST_NAME


def compare(left: Path, right: Path, allowed: tuple[Allowed, ...]) -> tuple[list[str], str]:
    """The paths two rendered repos differ on, and a report of each difference."""
    before, after = Seen(left, allowed), Seen(right, allowed)
    changed = differing(before.digests, after.digests)
    lines: list[str] = []
    for relative in changed:
        old, new = before.text(relative), after.text(relative)
        if old is None or new is None or old == new:
            lines.append(
                f"{relative}: {_describe(left / relative)} -> {_describe(right / relative)}"
            )
            continue
        lines.extend(
            difflib.unified_diff(
                old.splitlines(),
                new.splitlines(),
                f"{left.name}/{relative}",
                f"{right.name}/{relative}",
                lineterm="",
                n=2,
            )
        )
    return changed, "\n".join(lines)


def _describe(path: Path) -> str:
    if not path.exists() and not path.is_symlink():
        return "absent"
    return stat.filemode(path.lstat().st_mode)


def stripped(config: Mapping[str, Any]) -> dict[str, Any]:
    """*config* without the section, spelled nested or as dotted keys."""
    return {
        key: item
        for key, item in config.items()
        if key != SECTION and not str(key).startswith(f"{SECTION}.")
    }


def _rewrite_config(path: Path, change: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["config"] = change(dict(document.get("config") or {}))
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def strip_section(repo: Path) -> None:
    """Take the section out of the profile and of every tier layer that spells it."""
    _rewrite_config(repo / "profile.yml", stripped)
    for layer in sorted((repo / "tiers").glob("*.yml")):
        config = (yaml.safe_load(layer.read_text(encoding="utf-8")) or {}).get("config") or {}
        if stripped(config) != config:
            _rewrite_config(layer, stripped)


def switch_section_off(repo: Path) -> None:
    """Spell the section Off in the profile: ``pv_directory: null`` over whatever it held."""
    _rewrite_config(repo / "profile.yml", lambda config: {**stripped(config), SECTION: None})


def blocked_environment(blocker: Path) -> Callable[[Path], dict[str, str]]:
    """``render_environment`` with a ``sitecustomize`` that blocks the new modules."""
    plain = _off_render.render_environment

    def environment(source: Path) -> dict[str, str]:
        env = plain(source)
        env["PYTHONPATH"] = f"{blocker}:{env['PYTHONPATH']}"
        return env

    return environment


@pytest.fixture(scope="module")
def blocker(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("blocker")
    (path / "sitecustomize.py").write_text(BLOCKER, encoding="utf-8")
    return path


@pytest.fixture(scope="module", autouse=True)
def blocked(blocker: Path) -> Iterator[None]:
    """Every render of this module runs with the new connectors modules blocked."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_off_render, "render_environment", blocked_environment(blocker))
        yield


@pytest.mark.parametrize(
    ("relative", "glob", "expected"),
    [
        ("profile.yml", "**/profile.yml", True),
        ("build/.image/demo-admin/profile.yml", "**/profile.yml", True),
        ("build/demo-logbook/config.yml", "**/demo-logbook/config.yml", True),
        (
            "build/.image/demo-logbook/build/config.yml",
            "**/.image/demo-logbook/build/config.yml",
            True,
        ),
        ("build/.image/demo-logbook/build/config.yml", "**/.image/*/build/config.yml", True),
        ("build/.image/a/b/build/config.yml", "**/.image/*/build/config.yml", False),
        ("build/demo-logbook/config.yml.bak", "**/config.yml", False),
        ("build/tiers/logbook.yml", "**/tiers/logbook.yml", True),
        ("build/xtiers/logbook.yml", "**/tiers/logbook.yml", False),
    ],
)
def test_a_glob_matches_the_whole_relative_path(relative: str, glob: str, expected: bool) -> None:
    assert _matches(relative, (glob,)) is expected


def test_the_blocker_refuses_exactly_the_new_modules(blocker: Path) -> None:
    env = blocked_environment(blocker)(ROOT)
    probe = (
        "import importlib, osprey_connectors.workspace\n"
        f"for name in {NEW_CONNECTORS_MODULES!r}:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except ModuleNotFoundError:\n"
        "        continue\n"
        "    raise SystemExit(f'{name} imported')\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("off-render")


def _render_kept(
    source: Path,
    store: Path,
    name: str,
    configuration: Configuration,
    prepare: Callable[[Path], None] | None,
) -> Path:
    """Render at the one repo path, then move the tree aside so it stays on disk."""
    repo = store / "render" / REPO_NAME
    render(source, repo, configuration, prepare=prepare)
    kept = store / name
    shutil.rmtree(kept, ignore_errors=True)
    repo.rename(kept)
    return kept


@pytest.fixture(scope="module")
def baseline(store: Path) -> Iterator[Path]:
    """A scratch worktree of the baseline ref, unless the feature is already there."""
    ref = _off_render.baseline_ref()
    if _off_render._holds(ref, LANDED, ROOT):
        pytest.skip("landed; the permanent guard holds it")
    with scratch_worktree(ref, store / "baseline-tree") as tree:
        yield tree


@pytest.fixture(scope="module")
def renders(store: Path, baseline: Path) -> Callable[[Configuration], tuple]:
    """Baseline and stripped-branch trees of a configuration, rendered once each."""
    done: dict[str, tuple[Path, Path]] = {}

    def pair(configuration: Configuration) -> tuple[Path, Path]:
        if configuration.name not in done:
            done[configuration.name] = (
                _render_kept(baseline, store, f"main-{configuration.name}", configuration, None),
                _render_kept(
                    ROOT, store, f"branch-{configuration.name}", configuration, strip_section
                ),
            )
        return done[configuration.name]

    return pair


@pytest.mark.parametrize("configuration", CONFIGURATIONS, ids=lambda item: item.name)
def test_the_off_render_is_the_baseline_render(
    configuration: Configuration, renders: Callable[[Configuration], tuple]
) -> None:
    main, branch = renders(configuration)
    changed, report = compare(main, branch, ALLOW_LIST)
    assert changed == [], report


def test_every_allow_list_entry_is_still_needed(
    renders: Callable[[Configuration], tuple],
) -> None:
    main, branch = renders(DEMO)
    moved, _ = compare(main, branch, ())
    for rule in ALLOW_LIST:
        assert any(_rule(path, (rule,)) for path in moved), rule.reason


def test_the_stripped_demo_preset_hashes_as_the_baseline_preset(baseline: Path) -> None:
    raw, path = _load_preset_raw("demo")
    document = copy.deepcopy(raw)
    document["config"] = stripped(document.get("config") or {})

    assert _hash_resolved_profile(document, path, conventions=False) == preset_hash(baseline)


@pytest.fixture(scope="module")
def no_section(store: Path) -> Path:
    return _render_kept(ROOT, store, "guard-no-section", DEMO, strip_section)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("pv_directory: null\nx: 1\n", {"x": 1}),
        (
            "pv_directory: null\nsimulation:\n  models: null\n",
            {"simulation": {"models": None}},
        ),
        ("other: null\n", {"other": None}),
        ("pv_directory:\n  type: channelfinder\n", {"pv_directory": {"type": "channelfinder"}}),
        ("a:\n  pv_directory: null\n", {"a": {"pv_directory": None}}),
    ],
)
def test_the_off_rule_takes_out_only_the_sections_own_null(
    text: str, expected: dict[str, Any]
) -> None:
    assert yaml.safe_load(_without_off_spelling(text)) == expected


def test_a_moved_value_null_is_seen_by_the_off_guard(tmp_path: Path) -> None:
    def tree(name: str, config: str) -> Path:
        path = tmp_path / name / "build" / "config.yml"
        path.parent.mkdir(parents=True)
        path.write_text(config, encoding="utf-8")
        return tmp_path / name

    off = tree("off", "pv_directory: null\nsimulation:\n  models: null\n")
    moved = tree("moved", "simulation:\n  models: all\n")
    same = tree("same", "simulation:\n  models: null\n")
    assert compare(off, moved, (OFF_SPELLING,))[0] == ["build/config.yml"]
    assert compare(off, same, (OFF_SPELLING,))[0] == []


def test_the_off_spelling_renders_what_no_section_renders(store: Path, no_section: Path) -> None:
    off = _render_kept(ROOT, store, "guard-off", DEMO, switch_section_off)
    changed, report = compare(no_section, off, (OFF_SPELLING,))
    assert changed == [], report
    rendered = yaml.safe_load((off / "build" / "config.yml").read_text(encoding="utf-8"))
    assert rendered[SECTION] is None
