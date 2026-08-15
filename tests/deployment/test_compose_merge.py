"""The root-merged compose document — what it contains, and what it must not.

``.osprey-compose.yml`` exists because podman-compose decides a project's
working directory from the first ``-f`` file's dirname on versions ≤ 1.3 and
from ``COMPOSE_PROJECT_DIR`` on 1.4.1+. One merged file AT THE REPO ROOT makes
both land on the root, so every relative path in every rendered compose file
resolves the same way on every provider.

Two properties carry that promise, and both are tested here rather than assumed:

* **The merge is compose's.** Mappings merge key by key with later ``-f`` files
  winning; anything else is replaced whole. OSPREY's rendered files are disjoint
  today, so the precedence rule would go unexercised in practice — which is
  exactly why it is pinned.
* **The file is a machine artifact, and holds no secrets.** ``${VAR}`` goes in
  unresolved and comes out unresolved: interpolation belongs to the runtime,
  fed by ``--env-file``. A merge that substituted values would write the
  deployment's tokens to a file at the repo root.

The last section is the classification: an artifact at the root that nothing
ignores, excludes from image contexts, or removes on reset would be a file that
looks committed, ships inside images, and survives a factory reset. Each of
those four is asserted against the code that owns it.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from osprey.cli.build_cmd import _NON_SOURCE_ROOT_ENTRIES
from osprey.cli.init_cmd import _repo_gitignore, _repo_readme
from osprey.deployment import reset as reset_mod
from osprey.deployment.compose_merge import (
    MERGED_COMPOSE_FILENAME,
    ComposeMergeError,
    merge_compose_documents,
    merged_compose_path,
    render_merged_compose,
    write_merged_compose,
)
from osprey.deployment.reset import RuntimeProbe, plan_reset

TOP_LEVEL = """\
services:
  event-dispatcher:
    image: ${OSPREY_IMAGE}
    environment:
      OSPREY_DISPATCH_TOKEN: ${EVENT_DISPATCHER_TOKEN}
    ports:
      - "127.0.0.1:8020:8020"
networks:
  osprey-network:
    driver: bridge
volumes:
  dispatch_workspace: {}
"""

SERVICE_FILE = """\
services:
  dispatch-worker-1:
    image: ${OSPREY_IMAGE}
    env_file:
      - ./.env
    networks:
      - osprey-network
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A deployment repo with a render zone holding two compose files."""
    root = tmp_path / "als-exemplar"
    (root / "build" / "services").mkdir(parents=True)
    (root / "build" / "docker-compose.yml").write_text(TOP_LEVEL, encoding="utf-8")
    (root / "build" / "services" / "dispatch_worker.yml").write_text(SERVICE_FILE, encoding="utf-8")
    return root


def rendered_files(repo: Path) -> list[Path]:
    """The ``-f`` list, in order, as a compose invocation would spell it."""
    return [
        repo / "build" / "docker-compose.yml",
        repo / "build" / "services" / "dispatch_worker.yml",
    ]


def merged_document(repo: Path) -> dict:
    """Write the merged document and read it back as YAML."""
    return yaml.safe_load(write_merged_compose(repo, rendered_files(repo)).read_text())


# ---------------------------------------------------------------------------
# Where the document goes — the reason the whole file exists
# ---------------------------------------------------------------------------


def test_the_document_is_written_at_the_repo_root(repo: Path) -> None:
    """Not in ``build/``: podman-compose ≤ 1.3 chdir to the first -f file's dir."""
    written = write_merged_compose(repo, rendered_files(repo))

    assert written == repo / MERGED_COMPOSE_FILENAME
    assert written.parent == repo
    assert written.is_absolute()


def test_the_path_helper_and_the_writer_agree(repo: Path) -> None:
    assert merged_compose_path(repo) == write_merged_compose(repo, rendered_files(repo))


def test_a_relative_f_entry_resolves_against_the_repo_root(repo: Path) -> None:
    """``compose_base_cmd`` accepts relative -f paths; so does the merge."""
    relative = [Path("build/docker-compose.yml"), Path("build/services/dispatch_worker.yml")]

    written = write_merged_compose(repo, relative)

    assert yaml.safe_load(written.read_text())["services"].keys() == {
        "event-dispatcher",
        "dispatch-worker-1",
    }


# ---------------------------------------------------------------------------
# The merge itself
# ---------------------------------------------------------------------------


def test_disjoint_service_maps_are_unioned(repo: Path) -> None:
    """The shape OSPREY actually renders: one services: map per -f file."""
    document = merged_document(repo)

    assert list(document["services"]) == ["event-dispatcher", "dispatch-worker-1"]
    assert document["networks"] == {"osprey-network": {"driver": "bridge"}}
    assert document["volumes"] == {"dispatch_workspace": {}}


def test_a_later_file_wins_on_a_key_two_files_set() -> None:
    """Compose's own -f precedence, stated because the renders do not exercise it."""
    merged = merge_compose_documents(
        [
            {"services": {"worker": {"image": "first", "restart": "always"}}},
            {"services": {"worker": {"image": "second"}}},
        ]
    )

    assert merged["services"]["worker"]["image"] == "second"
    assert merged["services"]["worker"]["restart"] == "always"


def test_a_sequence_is_replaced_whole_rather_than_concatenated() -> None:
    """A list is a value, not a map. Compose replaces it; a concatenation would
    leave a service listening on a port a later file deliberately dropped."""
    merged = merge_compose_documents(
        [
            {"services": {"worker": {"ports": ["8020:8020", "9190:9190"]}}},
            {"services": {"worker": {"ports": ["9190:9190"]}}},
        ]
    )

    assert merged["services"]["worker"]["ports"] == ["9190:9190"]


def test_the_inputs_are_never_mutated() -> None:
    first = {"services": {"worker": {"image": "first"}}}
    second = {"services": {"worker": {"image": "second"}}}

    merge_compose_documents([first, second])

    assert first == {"services": {"worker": {"image": "first"}}}
    assert second == {"services": {"worker": {"image": "second"}}}


def test_an_empty_compose_file_contributes_nothing(repo: Path) -> None:
    (repo / "build" / "empty.yml").write_text("", encoding="utf-8")

    document = yaml.safe_load(
        write_merged_compose(
            repo, [*rendered_files(repo), repo / "build" / "empty.yml"]
        ).read_text()
    )

    assert set(document["services"]) == {"event-dispatcher", "dispatch-worker-1"}


# ---------------------------------------------------------------------------
# No secrets: ${VAR} goes through unresolved
# ---------------------------------------------------------------------------


def test_variable_references_are_left_for_the_runtime(repo: Path, monkeypatch) -> None:
    """The value is set in the environment, and STILL must not appear."""
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "minted-dispatch-secret")
    monkeypatch.setenv("OSPREY_IMAGE", "als-exemplar:local")

    text = write_merged_compose(repo, rendered_files(repo)).read_text()

    assert "${EVENT_DISPATCHER_TOKEN}" in text
    assert "${OSPREY_IMAGE}" in text
    assert "minted-dispatch-secret" not in text
    assert "als-exemplar:local" not in text


def test_the_env_file_reference_survives_as_a_reference(repo: Path) -> None:
    """``env_file: ./.env`` stays a path — the merge never inlines a secret store."""
    document = merged_document(repo)

    assert document["services"]["dispatch-worker-1"]["env_file"] == ["./.env"]


# ---------------------------------------------------------------------------
# Deterministic bytes
# ---------------------------------------------------------------------------


def test_identical_input_produces_identical_bytes(repo: Path, tmp_path: Path) -> None:
    other = tmp_path / "second-checkout"
    other.mkdir()

    first = render_merged_compose([yaml.safe_load(TOP_LEVEL), yaml.safe_load(SERVICE_FILE)])
    second = render_merged_compose([yaml.safe_load(TOP_LEVEL), yaml.safe_load(SERVICE_FILE)])

    assert first == second
    assert write_merged_compose(repo, rendered_files(repo)).read_text() == first


def test_rewriting_over_an_existing_document_is_a_no_op(repo: Path) -> None:
    """Every deploy rewrites it; an unchanged render must not churn the file."""
    first = write_merged_compose(repo, rendered_files(repo)).read_bytes()

    assert write_merged_compose(repo, rendered_files(repo)).read_bytes() == first


def test_the_header_carries_no_timestamp_or_host_path(repo: Path) -> None:
    """Determinism, said the other way round: nothing environmental in the file."""
    text = write_merged_compose(repo, rendered_files(repo)).read_text()
    header = text.split("\n\n", 1)[0]

    assert str(repo) not in header
    assert "osprey up" in header


def test_the_document_is_not_written_with_a_secret_file_mode(repo: Path) -> None:
    """0600 would claim this holds secrets. It does not, and must not say it does."""
    written = write_merged_compose(repo, rendered_files(repo))

    assert stat.S_IMODE(os.stat(written).st_mode) == 0o644


# ---------------------------------------------------------------------------
# A broken render is refused, not half-written
# ---------------------------------------------------------------------------


def test_an_unparseable_compose_file_refuses(repo: Path) -> None:
    broken = repo / "build" / "broken.yml"
    broken.write_text("services:\n  worker:\n   - [unbalanced\n", encoding="utf-8")

    with pytest.raises(ComposeMergeError) as excinfo:
        write_merged_compose(repo, [*rendered_files(repo), broken])

    assert str(broken) in str(excinfo.value)
    assert not (repo / MERGED_COMPOSE_FILENAME).exists()


def test_a_compose_file_that_is_not_a_mapping_refuses(repo: Path) -> None:
    listy = repo / "build" / "listy.yml"
    listy.write_text("- services\n- networks\n", encoding="utf-8")

    with pytest.raises(ComposeMergeError):
        write_merged_compose(repo, [listy])


def test_a_missing_compose_file_refuses(repo: Path) -> None:
    with pytest.raises(ComposeMergeError):
        write_merged_compose(repo, [repo / "build" / "not-rendered.yml"])


# ---------------------------------------------------------------------------
# Classification: git, image contexts, reset, and the README
# ---------------------------------------------------------------------------


def test_the_emitted_gitignore_carries_the_artifact_anchored() -> None:
    """Anchored, because an unanchored pattern also swallows same-named files
    deeper in the tree — and git then skips them on `git add` silently."""
    assert f"\n/{MERGED_COMPOSE_FILENAME}\n" in _repo_gitignore()
    assert f"\n{MERGED_COMPOSE_FILENAME}\n" not in _repo_gitignore()


def test_the_packaged_project_template_gitignore_carries_it_too() -> None:
    template = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "osprey"
        / "templates"
        / "project"
        / "gitignore"
    ).read_text(encoding="utf-8")

    assert f"\n/{MERGED_COMPOSE_FILENAME}\n" in template


def test_the_artifact_never_enters_a_container_image() -> None:
    """It is deploy output about the host's runtime; an image copy would be a
    stale project description baked into a distributable artifact."""
    assert MERGED_COMPOSE_FILENAME in _NON_SOURCE_ROOT_ENTRIES


def test_reset_removes_the_artifact_and_says_so(repo: Path) -> None:
    (repo / "var" / "agent_data").mkdir(parents=True)
    (repo / "build" / "config.yml").write_text(
        "project_name: als-exemplar\nbuild_dir: ./build\n", encoding="utf-8"
    )
    written = write_merged_compose(repo, rendered_files(repo))

    plan = plan_reset(repo, probe=RuntimeProbe("docker", run=_no_resources))

    assert written in plan.paths
    assert MERGED_COMPOSE_FILENAME in "\n".join(plan.render())


def test_reset_leaves_no_directory_where_the_file_was(repo: Path, monkeypatch) -> None:
    """It is a file. Recreating it empty — as reset does for the state zones —
    would leave a DIRECTORY of that name for the next deploy to trip over."""
    monkeypatch.setattr(reset_mod, "down_deployment", lambda root: None)
    (repo / "var" / "agent_data").mkdir(parents=True)
    (repo / "build" / "config.yml").write_text(
        "project_name: als-exemplar\nbuild_dir: ./build\n", encoding="utf-8"
    )
    write_merged_compose(repo, rendered_files(repo))
    plan = plan_reset(repo, probe=RuntimeProbe("docker", run=_no_resources))

    reset_mod.execute_reset(plan, probe=RuntimeProbe("docker", run=_no_resources))

    assert not (repo / MERGED_COMPOSE_FILENAME).exists()
    assert not (repo / "build").exists()


def test_the_readme_classifies_the_artifact() -> None:
    """An operator meeting a file at the repo root needs to know it is machinery."""
    readme = _repo_readme("als-exemplar")

    assert MERGED_COMPOSE_FILENAME in readme
    assert "osprey reset" in readme


def _no_resources(argv, **kwargs) -> subprocess.CompletedProcess:
    """A runtime with nothing of this project's on it: empty lists, no inspects."""
    if argv[1:3] == ["image", "inspect"] or argv[1:3] == ["container", "inspect"]:
        return subprocess.CompletedProcess(argv, 1, "", "Error: no such object")
    return subprocess.CompletedProcess(argv, 0, "", "")
