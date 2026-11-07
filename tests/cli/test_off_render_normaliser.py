"""The off-render comparison hides what names a render, and nothing else.

Most of this is a pure check of :func:`normalise` and the baseline rule over throwaway
repositories. One render of the smallest configuration, done twice at one repo path, is
what shows the whole normalisation holds on a real tree.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.cli._off_render import (
    AGENT_RECORD_SECRETS,
    CONFIGURATIONS,
    DEMO_HIERARCHICAL,
    DEMO_PRESET,
    EXCLUDED,
    EXCLUDED_MARK,
    GRAPH_INDEX,
    HELLO_WORLD,
    REPO_NAME,
    ROOT,
    SECRET_MARK,
    UNSET_ENV,
    VERSION_MARK,
    VOLATILE_CREATION_KEYS,
    VOLATILE_MANIFEST_KEYS,
    baseline_ref,
    differing,
    initialise,
    is_excluded,
    normalise,
    preset_hash,
    render,
    render_environment,
    scratch_worktree,
    snapshot,
)

SCM_VERSION = "2026.9.0b3.post5815+g359ca32b2.d20261107"
CONNECTORS_VERSION = "2026.11.4.post78+g359ca32b2"

VERSION_FORMS = [
    f'        OSPREY_VERSION: "{SCM_VERSION}"\n',
    '        OSPREY_VERSION: "2026.9.0b3"\n',
    '        OSPREY_VERSION: "0.0.0+unknown"\n',
    f"        OSPREY_CONNECTORS_VERSION={CONNECTORS_VERSION}\n",
    '        OSPREY_CONNECTORS_VERSION: "0.0.0+unbuilt"\n',
    f"#   emitted by OSPREY {SCM_VERSION}\n",
]


@pytest.mark.parametrize("line", VERSION_FORMS)
def test_every_version_form_is_replaced(line: str) -> None:
    masked = normalise("services/x/docker-compose.yml", line.encode()).decode()
    assert VERSION_MARK in masked
    assert not any(char.isdigit() for char in masked), masked


@pytest.mark.parametrize(
    "line",
    [
        "requires_osprey_version: '>=2026.10.0'\n",
        '    "canopy_version": "0.0.2.dev291+g166989aad"\n',
        "certifi==2026.1.1\n",
        "image: neo4j:2026.1.1\n",
        "osprey-connectors>=2026.11.3\n",
        "pip install osprey-connectors==2026.11.3\n",
        "pip install osprey-framework==2026.10.1\n",
        '{"osprey_version": "2026.10.1"}\n',
    ],
)
def test_a_version_that_is_content_stays(line: str) -> None:
    assert normalise("Dockerfile", line.encode()).decode() == line


def _manifest() -> bytes:
    return json.dumps(
        {
            "schema_version": "1.3.0",
            "creation": {
                "osprey_version": SCM_VERSION,
                "timestamp": "2026-11-07T02:09:02",
                "preset_hash": "sha256:five",
                "template": "demo",
                "profile_key_digests": {"watchdogs": "sha256:six"},
            },
            "build_args": {"profile_path": "/somewhere/profile.yml"},
            "reproducible_command": "osprey build",
            "timestamp": "2026-11-07T02:09:02",
            "preset_hash": "sha256:one",
            "watchdogs": "sha256:two",
            "artifacts": {"hooks": ["approval"], "nested": {"preset_hash": "sha256:three"}},
            "file_checksums": {"Dockerfile": "sha256:four"},
        }
    ).encode()


def test_the_manifest_loses_each_volatile_key_and_nothing_else() -> None:
    assert set(VOLATILE_MANIFEST_KEYS) == {
        "build_args",
        "reproducible_command",
        "timestamp",
        "preset_hash",
        "watchdogs",
    }
    assert set(VOLATILE_CREATION_KEYS) == {"timestamp", "osprey_version", "preset_hash"}
    assert all(VOLATILE_MANIFEST_KEYS.values()) and all(VOLATILE_CREATION_KEYS.values())
    stable = json.loads(normalise("build/.osprey-manifest.json", _manifest()))
    assert stable == {
        "schema_version": "1.3.0",
        "creation": {"template": "demo", "profile_key_digests": {"watchdogs": "sha256:six"}},
        "artifacts": {"hooks": ["approval"], "nested": {"preset_hash": "sha256:three"}},
        "file_checksums": {"Dockerfile": "sha256:four"},
    }


def test_a_new_key_under_creation_stays_visible() -> None:
    document = json.loads(_manifest())
    document["creation"]["pv_directory"] = "enabled"
    stable = json.loads(normalise("build/.osprey-manifest.json", json.dumps(document).encode()))
    assert stable["creation"]["pv_directory"] == "enabled"


def _agent_record(value: str, *extra: str) -> bytes:
    lines = ["CLICKHOUSE_USER=admin", *(f"{key}={value}" for key in AGENT_RECORD_SECRETS), *extra]
    return ("\n".join(lines) + "\n").encode()


def test_only_the_values_of_measured_secrets_are_blanked() -> None:
    masked = normalise(".env.agent-record.service", _agent_record("one", "NEW_KEY=kept")).decode()
    assert masked.splitlines() == [
        "CLICKHOUSE_USER=admin",
        *(f"{key}={SECRET_MARK}" for key in AGENT_RECORD_SECRETS),
        "NEW_KEY=kept",
    ]


@pytest.mark.parametrize("relative", [".env", "build/.env.agent-record.service", "var/x"])
def test_a_secret_shaped_line_elsewhere_stays(relative: str) -> None:
    assert normalise(relative, _agent_record("one")) == _agent_record("one")


@pytest.mark.parametrize(
    ("relative", "content"),
    [
        *(("services/x/docker-compose.yml", line.encode()) for line in VERSION_FORMS),
        ("build/.osprey-manifest.json", _manifest()),
        ("data/blob.bin", b"\xff\xfe 2026.1.1 \x00"),
        (".env.agent-record.service", _agent_record("one")),
    ],
)
def test_normalising_twice_changes_nothing(relative: str, content: bytes) -> None:
    once = normalise(relative, content)
    assert normalise(relative, once) == once


def test_a_binary_file_passes_through_unchanged() -> None:
    blob = b'\xff\xfe OSPREY_VERSION: "2026.9.0b3" \x00'
    assert normalise("data/blob.bin", blob) == blob


@pytest.mark.parametrize(
    "relative",
    [
        f"build/{GRAPH_INDEX}",
        f"build/demo-logbook/{GRAPH_INDEX}",
        f"build/.image/demo-admin/build/{GRAPH_INDEX}",
    ],
)
def test_every_copy_of_the_graph_index_is_excluded(relative: str) -> None:
    assert is_excluded(relative)


def test_the_graph_index_exclusion_states_its_reason() -> None:
    assert EXCLUDED == {
        GRAPH_INDEX: "binary index, not reproducible; its input facility.json is compared"
    }
    assert not is_excluded("build/data/channel_databases/facility.json")


def _tree(tmp_path: Path, files: dict[str, bytes]) -> Path:
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return tmp_path


def test_a_snapshot_sees_through_what_names_the_render(tmp_path: Path) -> None:
    def tree(name: str, version: str, index: bytes, secret: str) -> dict[str, str]:
        manifest = {
            "timestamp": version,
            "file_checksums": {
                "services/x/docker-compose.yml": f"sha256:{version}",
                GRAPH_INDEX: f"sha256:{index.hex()}",
                "config.yml": "sha256:same",
            },
        }
        return snapshot(
            _tree(
                tmp_path / name,
                {
                    "build/.osprey-manifest.json": json.dumps(manifest).encode(),
                    "build/services/x/docker-compose.yml": (
                        f'OSPREY_VERSION: "{version}"\n'.encode()
                    ),
                    f"build/{GRAPH_INDEX}": index,
                    "build/config.yml": b"key: value\n",
                    ".env": b"KEY=value\n",
                    ".env.agent-record.service": _agent_record(secret),
                    ".env.example": b"KEY=\n",
                    "var/state/one": b"state\n",
                },
            )
        )

    left = tree("left", "2026.9.0b3", b"\x00one", "one")
    right = tree("right", "2026.10.0", b"\x00two", "two")
    assert differing(left, right) == []
    assert left[f"build/{GRAPH_INDEX}"].endswith(EXCLUDED_MARK)
    assert {".env", ".env.agent-record.service", ".env.example", "var/state/one"} <= left.keys()


@pytest.mark.parametrize(
    ("relative", "content"),
    [
        ("var/pv_directory/state.db", b"new state"),
        ("var/agent_data/control_target/.osprey-control-tree", b"changed"),
        (".env", b"KEY=value\nPV_DIRECTORY_TOKEN=new\n"),
        (".env.lock", b"held"),
        (".env.agent-record.login.pv-directory", b"OSPREY_AGENT_RECORD_LOGIN=x\n"),
        (".env.agent-record.service", _agent_record("one", "PV_DIRECTORY_SECRET=x")),
        (f"build/demo-new/{GRAPH_INDEX}", b"\x00a new copy"),
    ],
)
def test_a_stray_or_changed_file_cannot_hide(tmp_path: Path, relative: str, content: bytes) -> None:
    files = {
        ".env": b"KEY=value\n",
        ".env.lock": b"",
        ".env.agent-record.service": _agent_record("one"),
        "var/agent_data/control_target/.osprey-control-tree": b"written by osprey build\n",
        "build/config.yml": b"key: value\n",
    }
    base = snapshot(_tree(tmp_path / "base", files))
    changed = snapshot(_tree(tmp_path / "changed", {**files, relative: content}))
    assert relative in differing(base, changed)


def test_a_snapshot_sees_a_new_empty_directory(tmp_path: Path) -> None:
    repo = _tree(tmp_path / "repo", {"build/config.yml": b"key: value\n"})
    before = snapshot(repo)
    (repo / "build" / "pv_directory").mkdir()
    assert differing(before, snapshot(repo)) == ["build/pv_directory"]


def test_a_snapshot_sees_a_symlink_and_its_target(tmp_path: Path) -> None:
    repo = _tree(tmp_path / "repo", {"build/config.yml": b"key: value\n"})
    before = snapshot(repo)
    link = repo / "build" / "link"
    link.symlink_to("config.yml")
    with_link = snapshot(repo)
    assert differing(before, with_link) == ["build/link"]
    link.unlink()
    link.symlink_to("elsewhere.yml")
    assert differing(with_link, snapshot(repo)) == ["build/link"]


def test_a_snapshot_does_not_walk_through_a_directory_symlink(tmp_path: Path) -> None:
    repo = _tree(tmp_path / "repo", {"build/data/file": b"x"})
    (repo / "build" / "alias").symlink_to("data", target_is_directory=True)
    assert snapshot(repo)["build/alias"] == "link:data"
    assert "build/alias/file" not in snapshot(repo)


def test_a_snapshot_sees_the_exec_bit(tmp_path: Path) -> None:
    repo = _tree(tmp_path / "repo", {"build/entrypoint.sh": b"#!/bin/sh\n"})
    script = repo / "build" / "entrypoint.sh"
    script.chmod(0o755)
    before = snapshot(repo)
    script.chmod(0o644)
    assert differing(before, snapshot(repo)) == ["build/entrypoint.sh"]


def test_a_snapshot_drops_the_rows_a_caller_compares_itself(tmp_path: Path) -> None:
    def tree(name: str, row: str) -> Path:
        manifest = {"file_checksums": {"data/addresses.json": row, "config.yml": "sha256:same"}}
        return _tree(
            tmp_path / name,
            {
                "build/.osprey-manifest.json": json.dumps(manifest).encode(),
                "build/config.yml": b"key: value\n",
            },
        )

    left, right = tree("left", "sha256:one"), tree("right", "sha256:two")
    assert differing(snapshot(left), snapshot(right)) == ["build/.osprey-manifest.json"]
    rows = {"build/data/addresses.json"}
    assert differing(snapshot(left, unseen_rows=rows), snapshot(right, unseen_rows=rows)) == []


def test_a_snapshot_keeps_the_differences_that_matter(tmp_path: Path) -> None:
    def tree(name: str, config: bytes, row: str) -> dict[str, str]:
        manifest = {"file_checksums": {"config.yml": row}}
        return snapshot(
            _tree(
                tmp_path / name,
                {
                    "build/.osprey-manifest.json": json.dumps(manifest).encode(),
                    "build/config.yml": config,
                },
            )
        )

    base = tree("base", b"key: value\n", "sha256:one")
    assert differing(base, tree("content", b"key: other\n", "sha256:one")) == ["build/config.yml"]
    assert differing(base, tree("row", b"key: value\n", "sha256:two")) == [
        "build/.osprey-manifest.json"
    ]
    assert differing(base, {**base, "extra": "x"}) == ["extra"]


def test_the_configurations_add_no_directory_section() -> None:
    assert [item.name for item in CONFIGURATIONS] == ["demo", "demo-hierarchical", "hello-world"]
    assert not any("pv_directory" in pair for item in CONFIGURATIONS for pair in item.overrides)


def test_the_hierarchical_configuration_reaches_the_profile(tmp_path: Path) -> None:
    repo = initialise(ROOT, tmp_path / REPO_NAME, DEMO_HIERARCHICAL)
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    assert profile["channel_finder_mode"] == "hierarchical"


def test_the_render_environment_drops_the_canopy_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in UNSET_ENV:
        monkeypatch.setenv(name, "/some/payload")
    env = render_environment(Path("/tree"))
    assert not set(UNSET_ENV) & set(env)
    assert UNSET_ENV == ("OSPREY_CANOPY_PAYLOAD", "OSPREY_CANOPY_PAYLOAD_CHECK")
    assert env["PYTHONPATH"].split(":") == ["/tree/src", "/tree/packages/osprey-connectors/src"]


def _commit(root: Path, name: str, path: str = "kept.txt") -> str:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(name, encoding="utf-8")
    for command in (
        ["add", "-A"],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", name],
    ):
        subprocess.run(["git", *command], cwd=root, check=True, capture_output=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _set_ref(root: Path, name: str, commit: str) -> None:
    subprocess.run(["git", "update-ref", f"refs/remotes/{name}", commit], cwd=root, check=True)


@pytest.fixture
def history(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def test_the_baseline_needs_origin_main(history: Path) -> None:
    _commit(history, "base")
    with pytest.raises(pytest.fail.Exception, match="needs origin/main"):
        baseline_ref(history)


def test_the_baseline_is_the_main_merge_base_once_it_holds_the_preset(history: Path) -> None:
    base = _commit(history, "base", DEMO_PRESET)
    _set_ref(history, "origin/main", base)
    _commit(history, "feature")
    assert baseline_ref(history) == base


def test_the_baseline_falls_back_to_the_rename_branch(history: Path) -> None:
    main = _commit(history, "base")
    _set_ref(history, "origin/main", main)
    rename = _commit(history, "rename", DEMO_PRESET)
    _set_ref(history, "origin/feature/rename-control-assistant-to-demo", rename)
    _commit(history, "feature")
    assert baseline_ref(history) == rename


def test_a_baseline_without_the_preset_fails_rather_than_compares(history: Path) -> None:
    main = _commit(history, "base")
    _set_ref(history, "origin/main", main)
    _set_ref(history, "origin/feature/rename-control-assistant-to-demo", main)
    _commit(history, "feature", DEMO_PRESET)
    with pytest.raises(pytest.fail.Exception, match="no merge-base holds"):
        baseline_ref(history)


def test_a_scratch_worktree_is_detached_at_the_ref_and_removed(
    history: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    first = _commit(history, "first")
    _commit(history, "second")
    path = tmp_path_factory.mktemp("scratch") / "tree"
    with scratch_worktree(first, path, history) as tree:
        assert (tree / "kept.txt").read_text(encoding="utf-8") == "first"
    assert not path.exists()


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("off-render") / REPO_NAME


@pytest.fixture(scope="module")
def first_render(repo: Path) -> dict[str, str]:
    return snapshot(render(ROOT, repo, HELLO_WORLD))


def test_a_repeat_render_compares_equal(repo: Path, first_render: dict[str, str]) -> None:
    assert ".env.agent-record.service" in first_render
    assert any(path.startswith("var/") for path in first_render)
    assert differing(first_render, snapshot(render(ROOT, repo, HELLO_WORLD))) == []


def test_the_render_rewrites_the_repo_and_prepares_its_source_zone(repo: Path) -> None:
    repo.mkdir(exist_ok=True)
    (repo / "stale.txt").write_text("from an earlier render\n", encoding="utf-8")

    seen: list[tuple[bool, bool]] = []

    def prepare(root: Path) -> None:
        seen.append(((root / "profile.yml").is_file(), (root / "build").exists()))

    render(ROOT, repo, HELLO_WORLD, prepare=prepare)
    assert seen == [(True, False)]
    assert not (repo / "stale.txt").exists()
    assert (repo / "build" / "config.yml").is_file()


def test_the_preset_hash_is_the_one_the_tree_resolves() -> None:
    from osprey.cli.build_profile_merge import compute_preset_hash

    assert preset_hash(ROOT) == compute_preset_hash("demo")
