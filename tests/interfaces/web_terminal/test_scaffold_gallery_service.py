"""Tests for ScaffoldGalleryService — bridges BuildArtifactCatalog + TemplateManager for web UI."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.init_cmd import init
from osprey.cli.profile_conventions import NOT_PROJECT_RELATIVE_CHANNEL
from osprey.interfaces.web_terminal.ownership import (
    OwnershipMode,
    OwnershipStore,
    reserved_write_channel,
    resolve_ownership,
)
from osprey.interfaces.web_terminal.scaffold_gallery_service import (
    ProtectedArtifactError,
    ScaffoldGalleryService,
    restore_scaffold_bodies,
)
from osprey.services.build_artifacts.catalog import BuildArtifactCatalog

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAFE_ARTIFACT = "rules/safety"  # Always available, real content

#: An artifact the gallery may still WRITE. ``.claude/rules/**`` is in the
#: protected set, so :data:`SAFE_ARTIFACT` stays readable and listable but is
#: no longer savable, claimable or creatable — every test that drives a write,
#: a claim or a create names this one instead. ``.claude/agents/**`` is the
#: ordinary-file stand-in the delete path already settled on, and an agent is a
#: real rendered artifact of the control-assistant preset, so it carries
#: framework content to diff against.
WRITABLE_ARTIFACT = "agents/channel-finder"


@pytest.fixture(scope="session")
def _baked_repo(tmp_path_factory) -> Path:
    """A real render of the control-assistant preset, through the real verbs.

    ``osprey init`` materializes the deployment repo, ``osprey build`` renders
    its build zone, and that RENDER is what the gallery service reads — it is
    the directory holding config.yml, the manifest and the ``.claude/`` tree.

    The control-assistant preset specifically: the gallery tests assert the
    channel-finder and data-visualizer agents and the diagnose skill, none of
    which ship with hello-world.

    Session-scoped: a render is expensive, and under coverage measurement it is
    expensive enough that one render per test blows the CI job budget. Tests
    get an isolated copy via ``project_dir``, never this tree itself.
    """
    runner = CliRunner()
    repo = tmp_path_factory.mktemp("gallery-bake") / "gallery-test"

    created = runner.invoke(init, [str(repo), "--preset", "control-assistant", "--no-git"])
    assert created.exit_code == 0, created.output

    result = runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert result.exit_code == 0, result.output
    return repo


@pytest.fixture()
def project_dir(_baked_repo, tmp_path):
    """A private copy of the baked render, free for the test to mutate.

    A render records absolute paths — ``project_root`` in config.yml,
    ``build_args.profile_path_abs`` in the manifest — and the baked tree still
    exists while tests run, so a copy that kept those bytes would resolve
    every path back to the shared bake and mutate it. Re-anchoring them to the
    copy is what makes each test's repo genuinely its own.
    """
    repo = tmp_path / "gallery-test"
    shutil.copytree(_baked_repo, repo, symlinks=True)

    old, new = str(_baked_repo).encode(), str(repo).encode()
    for path in repo.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        if old in data:
            path.write_bytes(data.replace(old, new))
    return repo / "build"


@pytest.fixture()
def service(project_dir):
    return ScaffoldGalleryService(project_dir)


def _unreachable_profile(project_dir: Path) -> Path:
    """Remove the profile a render names, leaving the manifest pointing at it.

    What a deployed container looks like: the manifest records the path the
    build machine resolved, and that path does not exist here.

    The SOURCE ZONE goes; the render does not. Under the three-zone layout the
    profile's parent IS the repo root, so deleting the parent — which is what a
    sibling-profile layout allowed — would take the render under test with it.
    """
    manifest = json.loads((project_dir / ".osprey-manifest.json").read_text(encoding="utf-8"))
    profile_path = manifest.get("build_args", {}).get("profile_path_abs")
    assert profile_path, "a render is expected to record the profile it came from"
    repo_root = Path(profile_path).parent
    assert project_dir.parent == repo_root, (
        "this helper assumes the render sits inside the repo whose profile it names"
    )
    Path(profile_path).unlink()
    for source in ("data", "personas", "triggers.yml"):
        entry = repo_root / source
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink(missing_ok=True)
    return project_dir


@pytest.fixture()
def detached_project_dir(project_dir, monkeypatch):
    """A project that names no profile at all — a pre-profile manifest.

    The only topology where ownership still belongs in the project's own
    config.yml. Note that this drops ``build_args.profile_path_abs`` as well as
    the profile directory: a project that *names* a profile it cannot reach is
    a different case, and must refuse rather than fall back here (see
    :class:`TestDegradedTopology`).
    """
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    _unreachable_profile(project_dir)

    manifest_path = project_dir / ".osprey-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.get("build_args", {}).pop("profile_path_abs", None)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return project_dir


@pytest.fixture()
def detached_service(detached_project_dir):
    return ScaffoldGalleryService(detached_project_dir)


@pytest.fixture()
def degraded_project_dir(project_dir, monkeypatch):
    """A project naming a profile that cannot be reached, with no volume."""
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return _unreachable_profile(project_dir)


@pytest.fixture()
def audit_zone(tmp_path, monkeypatch):
    """Redirect the durable audit zone, so refusal records land in the test's tree.

    ``writer.audit_dir`` is the one seam every ledger derives from, so patching
    it catches the gallery's and the restore's records alike without standing up
    a project root.
    """
    from osprey.audit import writer

    zone = tmp_path / "audit"
    monkeypatch.setattr(writer, "audit_dir", lambda: zone)
    return zone


@pytest.fixture()
def volume_dir(tmp_path):
    """Stand-in for the per-user claude-config volume mount."""
    return tmp_path / "claude-config"


@pytest.fixture()
def container_project(project_dir, volume_dir, monkeypatch):
    """A deployed web terminal: image-baked project tree plus a durable volume.

    The manifest still names its profile — that is exactly what an image
    carries — and the path is unreachable from inside the container. What
    survives a recreation is the volume, and nothing else.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(volume_dir))
    return _unreachable_profile(project_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store_index(volume_dir: Path) -> dict:
    """The durable ownership index as written on the volume."""
    path = volume_dir / "osprey" / "scaffold" / "user_owned.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _plant_store_record(volume_dir: Path, name: str, output_path: str, body: str) -> None:
    """Write a claimed record and its body straight onto the volume.

    Planted rather than claimed through the gallery, because the gallery
    refuses to claim some of these paths — which is the point. The store is a
    file on a volume the agent can write; a record in it is not evidence that
    any gate ever approved it.
    """
    store_dir = volume_dir / "osprey" / "scaffold"
    store_dir.mkdir(parents=True, exist_ok=True)
    index_path = store_dir / "user_owned.json"
    index = (
        json.loads(index_path.read_text(encoding="utf-8"))
        if index_path.exists()
        else {"version": 1, "artifacts": {}}
    )
    index["artifacts"][name] = {"state": "claimed", "output_path": output_path}
    index_path.write_text(json.dumps(index), encoding="utf-8")

    body_path = store_dir / "files" / output_path
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_path.write_text(body, encoding="utf-8")


def _recreate_container(project_dir: Path, destination: Path) -> Path:
    """Copy a pristine image-baked tree, as a container recreation would.

    The point of the copy is that it is taken *before* anything is claimed: it
    is the image, unchanged. Pairing it with the same volume is the whole test
    — the operator's work has to come back from somewhere, and the tree is not
    that somewhere.
    """
    shutil.copytree(project_dir, destination)
    return destination


def _protected_records(audit_zone: Path) -> list[dict]:
    """Every protected-write record this identity filed, grouped by surface.

    The gallery and the container-start restore file into ledgers of their own
    (``scaffold_gallery.jsonl`` / ``scaffold_restore.jsonl``), so both are read
    here -- the counts in this suite are about how many refusals happened, not
    about which file they happened to land in.
    """
    from osprey.audit.protected import SURFACE_SCAFFOLD_GALLERY, SURFACE_SCAFFOLD_RESTORE
    from osprey.utils.identity import acting_identity

    records: list[dict] = []
    for surface in (SURFACE_SCAFFOLD_GALLERY, SURFACE_SCAFFOLD_RESTORE):
        log = audit_zone / acting_identity() / f"{surface}.jsonl"
        if not log.exists():
            continue
        records += [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return records


def _profile_root(project_dir: Path) -> Path:
    """The profile directory a render was built from."""
    manifest = json.loads((project_dir / ".osprey-manifest.json").read_text(encoding="utf-8"))
    return Path(manifest["build_args"]["profile_path_abs"]).parent


def _read_config(project_dir: Path) -> dict:
    with open(project_dir / "config.yml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_user_owned(project_dir: Path) -> list[str]:
    cfg = _read_config(project_dir)
    return cfg.get("scaffold", {}).get("user_owned", [])


def _add_user_owned(project_dir: Path, name: str) -> None:
    """Register *name* in config.yml the way a build would have.

    Ownership an image carries is a name in config.yml with nothing behind it —
    no store record, no claim. The render's own config-owned artifact is a rule,
    which the write gate refuses, so a test about *editing* what you already own
    has to put a writable one there itself rather than borrow that one.
    """
    cfg = _read_config(project_dir)
    scaffold = cfg.setdefault("scaffold", {})
    owned = scaffold.setdefault("user_owned", [])
    if name not in owned:
        owned.append(name)
    with open(project_dir / "config.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _raising_resolve(real_resolve, filename: str):
    """A ``Path.resolve`` that fails for one file and behaves for every other.

    Scoped to a single name rather than patched wholesale, so the test drives
    the guard's own ``resolve`` call and nothing else: a blanket failure would
    break the fixture machinery around it and prove nothing about the branch.
    """

    def resolve(self, *args, **kwargs):
        if self.name == filename:
            raise OSError(62, "Too many levels of symbolic links")
        return real_resolve(self, *args, **kwargs)

    return resolve


# ===========================================================================
# List artifacts
# ===========================================================================


class TestListArtifacts:
    """Tests for ScaffoldGalleryService.list_artifacts()."""

    def test_list_artifacts_only_existing_files(self, service, project_dir):
        """Every returned artifact corresponds to a file that exists on disk."""
        result = service.list_artifacts()
        assert len(result) > 0, "Expected at least some artifacts from osprey build"
        for art in result:
            fpath = project_dir / art["output_path"]
            assert fpath.exists(), f"{art['output_path']} listed but not on disk"

    def test_list_artifacts_bounded_by_registry(self, service):
        """Returned count is at most the registry size (no phantom artifacts)."""
        registry = BuildArtifactCatalog.default()
        result = service.list_artifacts()
        assert len(result) <= len(registry.all_artifacts())

    def test_list_artifacts_status_framework(self, service):
        """An artifact without user-ownership has status 'framework'."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        assert SAFE_ARTIFACT in by_name
        assert by_name[SAFE_ARTIFACT]["status"] == "framework"

    def test_list_artifacts_status_user_owned(self, service, project_dir):
        """After claiming, the artifact shows status 'user-owned'."""
        service.scaffold_override(WRITABLE_ARTIFACT)
        # Re-create service to pick up refreshed config
        svc = ScaffoldGalleryService(project_dir)
        result = svc.list_artifacts()
        by_name = {a["name"]: a for a in result}
        assert by_name[WRITABLE_ARTIFACT]["status"] == "user-owned"

    def test_list_artifacts_marks_a_reserved_artifact_read_only(self, service, audit_zone):
        """The badge has to mean what the gate does, so both are asserted here.

        A card that offers an edit the save then refuses is worse than one that
        says plainly the artifact is not this operator's to write — and the two
        can only stay in step while they are the same policy call.
        """
        by_name = {a["name"]: a for a in service.list_artifacts()}
        assert by_name[SAFE_ARTIFACT]["read_only"] is True
        assert by_name["rules/facility"]["read_only"] is True

        with pytest.raises(ProtectedArtifactError):
            service.save_override("rules/facility", "# Rewritten by the agent\n")

    def test_list_artifacts_leaves_an_ordinary_artifact_writable(self, service):
        """An artifact outside the protected set carries the flag as false."""
        by_name = {a["name"]: a for a in service.list_artifacts()}
        assert by_name[WRITABLE_ARTIFACT]["read_only"] is False

    def test_list_artifacts_categories(self, service):
        """Category is derived from the canonical name prefix."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}

        # "agents/channel-finder" -> category "agents"
        assert by_name["agents/channel-finder"]["category"] == "agents"
        # "rules/safety" -> category "rules"
        assert by_name["rules/safety"]["category"] == "rules"
        # "claude-md" (no slash) -> category "config"
        assert by_name["claude-md"]["category"] == "config"

    def test_list_artifacts_summary_counts(self, service):
        """Sum of framework + overridden equals total."""
        result = service.list_artifacts()
        framework = sum(1 for a in result if a["status"] == "framework")
        owned = sum(1 for a in result if a["status"] == "user-owned")
        assert framework + owned == len(result)

    def test_list_artifacts_reads_disk_metadata(self, service, project_dir):
        """Modifying front-matter on disk is reflected in list_artifacts()."""
        # Claim and overwrite with custom front-matter
        service.scaffold_override(WRITABLE_ARTIFACT)
        disk_path = project_dir / ".claude" / "agents" / "channel-finder.md"
        disk_path.write_text(
            "---\nsummary: Custom safety summary\n"
            "description: Custom safety description\n---\n# Custom\n",
            encoding="utf-8",
        )

        svc = ScaffoldGalleryService(project_dir)
        result = svc.list_artifacts()
        by_name = {a["name"]: a for a in result}
        assert by_name[WRITABLE_ARTIFACT]["summary"] == "Custom safety summary"
        assert by_name[WRITABLE_ARTIFACT]["description"] == "Custom safety description"

    def test_list_artifacts_excludes_missing_files(self, project_dir):
        """Artifacts not on disk are not returned (filesystem-first)."""
        # Delete a known artifact file from the initialized project
        safety_file = project_dir / ".claude" / "rules" / "safety.md"
        if safety_file.exists():
            safety_file.unlink()

        svc = ScaffoldGalleryService(project_dir)
        result = svc.list_artifacts()
        names = {a["name"] for a in result}
        assert SAFE_ARTIFACT not in names


# ===========================================================================
# Content retrieval
# ===========================================================================


class TestGetContent:
    """Tests for get_content, get_framework_content, get_override_content."""

    def test_get_content_framework(self, service):
        """Framework artifact returns non-empty content with source='framework'."""
        result = service.get_content(SAFE_ARTIFACT)
        assert result["source"] == "framework"
        assert isinstance(result["content"], str)
        assert len(result["content"]) > 0

    def test_get_content_user_owned(self, service, project_dir):
        """After claim + modify, get_content returns the user's version."""
        service.scaffold_override(WRITABLE_ARTIFACT)
        custom = "# Custom safety rules\nDo not touch anything.\n"
        service.save_override(WRITABLE_ARTIFACT, custom)

        svc = ScaffoldGalleryService(project_dir)
        result = svc.get_content(WRITABLE_ARTIFACT)
        assert result["source"] == "user-owned"
        assert result["content"] == custom

    def test_get_framework_content_renders(self, service):
        """get_framework_content returns non-empty rendered content."""
        content = service.get_framework_content(SAFE_ARTIFACT)
        assert isinstance(content, str)
        assert len(content) > 0

    def test_get_framework_content_unknown(self, service):
        """Unknown artifact name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown artifact"):
            service.get_framework_content("nonexistent/artifact")

    def test_get_override_content_not_owned(self, service):
        """Non-owned artifact returns None."""
        result = service.get_override_content(SAFE_ARTIFACT)
        assert result is None

    def test_get_override_content_exists(self, service, project_dir):
        """After claiming, get_override_content returns file content."""
        scaffold_result = service.scaffold_override(WRITABLE_ARTIFACT)
        expected_content = scaffold_result["content"]

        svc = ScaffoldGalleryService(project_dir)
        content = svc.get_override_content(WRITABLE_ARTIFACT)
        assert content is not None
        assert content == expected_content


# ===========================================================================
# Diff
# ===========================================================================


class TestComputeDiff:
    """Tests for compute_diff."""

    def test_compute_diff_identical(self, service):
        """Claim without modification yields has_diff=False."""
        service.scaffold_override(WRITABLE_ARTIFACT)
        result = service.compute_diff(WRITABLE_ARTIFACT)
        assert result["has_diff"] is False
        assert result["additions"] == 0
        assert result["deletions"] == 0

    def test_compute_diff_with_changes(self, service, project_dir):
        """Claim, modify, diff shows additions and deletions."""
        service.scaffold_override(WRITABLE_ARTIFACT)
        service.save_override(WRITABLE_ARTIFACT, "# Completely replaced content\n")

        svc = ScaffoldGalleryService(project_dir)
        result = svc.compute_diff(WRITABLE_ARTIFACT)
        assert result["has_diff"] is True
        assert result["additions"] > 0
        assert result["deletions"] > 0
        assert isinstance(result["unified_diff"], str)
        assert len(result["unified_diff"]) > 0

    def test_compute_diff_not_owned(self, service):
        """Diff on a non-owned artifact raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not user-owned"):
            service.compute_diff(SAFE_ARTIFACT)


# ===========================================================================
# Scaffold (Claim)
# ===========================================================================


class TestScaffold:
    """Tests for scaffold_override (claim)."""

    def test_scaffold_claims_and_updates_config(self, detached_service, detached_project_dir):
        """With no profile to claim into, ownership is recorded in config.yml."""
        result = detached_service.scaffold_override(WRITABLE_ARTIFACT)

        assert result["status"] == "claimed"
        assert "output_path" in result
        assert len(result["content"]) > 0

        # config.yml has the user_owned entry
        user_owned = _get_user_owned(detached_project_dir)
        assert WRITABLE_ARTIFACT in user_owned

    def test_scaffold_already_owned(self, service):
        """Claiming the same artifact twice raises FileExistsError."""
        service.scaffold_override(WRITABLE_ARTIFACT)
        with pytest.raises(FileExistsError, match="already user-owned"):
            service.scaffold_override(WRITABLE_ARTIFACT)


# ===========================================================================
# Save
# ===========================================================================


class TestSaveOverride:
    """Tests for save_override."""

    def test_save_override_writes_file(self, detached_service, detached_project_dir):
        """Save writes new content to the user-owned file."""
        detached_service.scaffold_override(WRITABLE_ARTIFACT)
        new_content = "# Channel finder\nAsk before writing any setpoint.\n"
        result = detached_service.save_override(WRITABLE_ARTIFACT, new_content)

        assert result["status"] == "saved"

        # Verify file content on disk
        output_file = detached_project_dir / result["path"]
        assert output_file.read_text(encoding="utf-8") == new_content

    def test_save_override_not_owned(self, service):
        """Saving to a non-owned artifact raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not user-owned"):
            service.save_override(WRITABLE_ARTIFACT, "some content")


class TestSaveOverrideProtectedSet:
    """The gallery's write path consults the protected set on both surfaces.

    ``save_override`` reaches disk through two different branches of
    ``_write_body`` — the profile's copy where a profile supplies the artifact,
    the project tree (plus the volume) where none does. A guard on one of them
    would leave the other open, so each is driven separately here: a reserved
    artifact the PROFILE holds, and a reserved artifact the PROJECT holds.
    """

    #: Reserved, and already owned by the render — ``osprey init`` auto-claims
    #: it into ``scaffold.user_owned``. Owned is what makes it reach the write
    #: path at all: the ownership check runs first, and a refusal that only
    #: ever fired on unowned names would prove nothing.
    RESERVED_OWNED = "rules/facility"

    def _plant_in_profile(self, project_dir: Path, rel: str, body: str) -> Path:
        """Put a file in the profile's convention tree, as an operator would.

        Planted rather than claimed: the claim path has its own gate, and this
        test is about the save that follows — it must not be able to pass
        because the claim happened to be refused first.
        """
        target = _profile_root(project_dir) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        return target

    # ── The profile branch ───────────────────────────────────────────

    def test_save_override_refuses_a_reserved_rule_the_profile_supplies(
        self, project_dir, audit_zone
    ):
        """A rule is instruction text; the profile's copy is not the way in."""
        planted = self._plant_in_profile(project_dir, "rules/planted.md", "# Planted\n")

        svc = ScaffoldGalleryService(project_dir)
        assert "rules/planted" in svc._user_owned, (
            "a profile-supplied artifact is owned, which is what makes the name reach save"
        )
        assert svc._profile_file("rules/planted") == planted, (
            "this test is only meaningful while the profile branch is the one selected"
        )

        with pytest.raises(ProtectedArtifactError) as exc:
            svc.save_override("rules/planted", "# Rewritten by the agent\n")

        message = str(exc.value)
        assert "`rules/` convention directory" in message, message
        assert "NOTHING WAS WRITTEN" in message
        assert planted.read_text(encoding="utf-8") == "# Planted\n"

        records = _protected_records(audit_zone)
        assert len(records) == 1
        assert records[0]["surface"] == "scaffold_gallery"
        assert records[0]["subject"] == ".claude/rules/planted.md"

    def test_save_override_still_writes_the_profile_copy_when_unreserved(self, project_dir):
        """The same branch, an ordinary artifact: the save goes through."""
        planted = self._plant_in_profile(project_dir, "agents/planted.md", "# Planted\n")

        svc = ScaffoldGalleryService(project_dir)
        edited = "# Planted\nEdited through the gallery.\n"
        assert svc.save_override("agents/planted", edited)["status"] == "saved"
        assert planted.read_text(encoding="utf-8") == edited

    # ── The project-tree branch ──────────────────────────────────────

    def test_save_override_refuses_a_reserved_rule_in_the_project_tree(
        self, detached_project_dir, detached_service, audit_zone
    ):
        """No profile to hold it, so the write would have landed in the tree."""
        on_disk = detached_project_dir / ".claude" / "rules" / "facility.md"
        before = on_disk.read_text(encoding="utf-8")
        assert self.RESERVED_OWNED in detached_service._user_owned
        assert detached_service._profile_file(self.RESERVED_OWNED) is None, (
            "this test is only meaningful while the project-tree branch is the one selected"
        )

        with pytest.raises(ProtectedArtifactError) as exc:
            detached_service.save_override(self.RESERVED_OWNED, "# Rewritten by the agent\n")

        message = str(exc.value)
        assert "`rules/` convention directory" in message, message
        assert "NOTHING WAS WRITTEN" in message
        assert on_disk.read_text(encoding="utf-8") == before, "the tree copy must be untouched"

        records = _protected_records(audit_zone)
        assert len(records) == 1
        assert "target=.claude/rules/facility.md" in records[0]["detail"]
        assert records[0]["reason"] == "reserved path"

    def test_save_override_still_writes_the_project_tree_when_unreserved(
        self, detached_project_dir, detached_service
    ):
        """The same branch, an ordinary artifact: the save goes through."""
        orphan = detached_project_dir / ".claude" / "agents" / "tree-edit.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Original\n", encoding="utf-8")
        detached_service.register_untracked("agents/tree-edit")

        svc = ScaffoldGalleryService(detached_project_dir)
        edited = "# Edited in the tree\n"
        assert svc.save_override("agents/tree-edit", edited)["status"] == "saved"
        assert orphan.read_text(encoding="utf-8") == edited

    def test_save_override_refuses_before_the_volume_copy_is_written(
        self, container_project, volume_dir, audit_zone
    ):
        """The container's second surface must not be written either.

        The volume is the copy that outlives the container, so a refusal that
        landed there first would put the agent's rewrite back over the
        framework's rule at the next recreation — the refusal would have
        deferred the write rather than prevented it.
        """
        svc = ScaffoldGalleryService(container_project)

        with pytest.raises(ProtectedArtifactError, match="NOTHING WAS WRITTEN"):
            svc.save_override(self.RESERVED_OWNED, "# Rewritten by the agent\n")

        assert _store_index(volume_dir) == {}
        assert not (volume_dir / "osprey" / "scaffold" / "files").exists()
        assert len(_protected_records(audit_zone)) == 1

    # ── The route ────────────────────────────────────────────────────

    def test_save_override_route_refuses_with_403_and_records_activity(
        self, project_dir, audit_zone
    ):
        """The PUT route maps the refusal to 403 and publishes it to the ring."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from osprey.interfaces.web_terminal.routes import scaffold as scaffold_routes

        on_disk = project_dir / ".claude" / "rules" / "facility.md"
        before = on_disk.read_text(encoding="utf-8")

        app = FastAPI()
        app.include_router(scaffold_routes.router)
        app.state.project_cwd = str(project_dir)
        app.state.agent_activity_ring = []

        response = TestClient(app).put(
            f"/api/scaffold/{self.RESERVED_OWNED}/override",
            json={"content": "# Rewritten by the agent\n"},
        )

        assert response.status_code == 403, response.text
        detail = response.json()["detail"]
        assert "`rules/` convention directory" in detail
        assert "NOTHING WAS WRITTEN" in detail
        assert on_disk.read_text(encoding="utf-8") == before

        assert [event["tool"] for event in app.state.agent_activity_ring] == ["save_override"]
        recorded = app.state.agent_activity_ring[0]["target"]
        assert recorded["kind"] == "artifact"
        assert self.RESERVED_OWNED in recorded["detail"]


# ===========================================================================
# Unclaim
# ===========================================================================


class TestUnoverride:
    """Tests for unoverride (unclaim)."""

    def test_unclaim_removes_config(self, detached_service, detached_project_dir):
        """Unclaim removes the config entry.

        Config mode, because that is the only topology where config.yml holds
        ownership and removing the entry is what releasing means. Asserting
        this against a profile-built project used to pass for the wrong reason:
        the claim there never wrote config.yml in the first place, so "the
        entry is gone" was true before the unclaim ran. What profile mode does
        instead is pinned in :class:`TestUnclaimIsHonestAboutTheProfile`.
        """
        detached_service.scaffold_override(WRITABLE_ARTIFACT)
        assert WRITABLE_ARTIFACT in _get_user_owned(detached_project_dir)

        result = ScaffoldGalleryService(detached_project_dir).unoverride(WRITABLE_ARTIFACT)

        assert result["status"] == "removed"
        assert WRITABLE_ARTIFACT not in _get_user_owned(detached_project_dir)

    def test_unclaim_not_owned(self, service):
        """Unclaiming a non-owned artifact raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not user-owned"):
            service.unoverride(SAFE_ARTIFACT)


# ===========================================================================
# Description extraction
# ===========================================================================


class TestDescriptionExtraction:
    """Tests for two-tier summary/description extraction from front matter."""

    def test_agent_has_summary_from_front_matter(self, service):
        """Agent artifact summary comes from template front matter, not registry."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["agents/data-visualizer"]
        assert art["summary"] == ("Creates plots, charts, dashboards, and compiles LaTeX reports")
        # Summary should differ from the full description
        assert art["summary"] != art["description"]

    def test_hook_has_summary_from_front_matter(self, service):
        """Hook artifact summary comes from docstring YAML front matter."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["hooks/limits"]
        assert art["summary"] == ("Validates channel write values against the limits database")

    def test_config_falls_back_to_registry(self, service):
        """Config artifacts (JSON templates) fall back to registry description."""
        registry = BuildArtifactCatalog.default()
        reg_art = registry.get("claude-md")
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["claude-md"]
        # No front matter in JSON templates -> falls back to registry
        assert art["summary"] == reg_art.description

    def test_description_is_full_from_front_matter(self, service):
        """Agent description field comes from full front matter description."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["agents/data-visualizer"]
        assert "Creates data visualizations" in art["description"]

    def test_rule_has_summary_and_description(self, service):
        """Rule artifacts get both summary and description from front matter."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["rules/safety"]
        assert art["summary"] == "Safety boundaries, channel write safety, and data integrity"
        assert "tool confinement" in art["description"]

    def test_skill_diagnose_has_summary_from_front_matter(self, service):
        """Diagnose skill gets summary from skill front matter."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["skills/diagnose"]
        assert art["summary"] == "Investigate OSPREY infrastructure and agent failures"


# ===========================================================================
# Untracked file detection
# ===========================================================================


class TestScanUntracked:
    """Tests for scan_untracked — detecting files active in Claude Code but not managed."""

    def test_scan_untracked_finds_orphaned_files(self, service, project_dir):
        """A .md file in .claude/agents/ not in the registry is reported as untracked.

        The stand-in for "an ordinary hand-written file" is an agent rather
        than a rule: ``.claude/rules/**`` is in the protected set, and a
        reserved artifact is deliberately never offered for adoption.
        """
        orphan = project_dir / ".claude" / "agents" / "my-custom-agent.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# My Custom Agent\nDo something special.\n", encoding="utf-8")

        result = service.scan_untracked()
        names = [u["canonical_name"] for u in result]
        assert "agents/my-custom-agent" in names

    def test_scan_untracked_excludes_registered(self, service, project_dir):
        """Registry artifacts that exist on disk are NOT reported as untracked."""
        safety_file = project_dir / ".claude" / "rules" / "safety.md"
        safety_file.parent.mkdir(parents=True, exist_ok=True)
        safety_file.write_text("# Safety\nExisting framework content.\n", encoding="utf-8")

        result = service.scan_untracked()
        names = [u["canonical_name"] for u in result]
        assert "rules/safety" not in names

    def test_scan_untracked_excludes_user_owned(self, service, project_dir):
        """Custom files already in user_owned are NOT reported as untracked."""
        orphan = project_dir / ".claude" / "agents" / "already-claimed.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Already Claimed\n", encoding="utf-8")

        service.register_untracked("agents/already-claimed")

        svc = ScaffoldGalleryService(project_dir)
        result = svc.scan_untracked()
        names = [u["canonical_name"] for u in result]
        assert "agents/already-claimed" not in names

    def test_scan_untracked_returns_correct_category(self, service, project_dir):
        """Category is derived from the first path component."""
        orphan = project_dir / ".claude" / "agents" / "rogue-agent.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Rogue Agent\n", encoding="utf-8")

        result = service.scan_untracked()
        by_name = {u["canonical_name"]: u for u in result}
        assert by_name["agents/rogue-agent"]["category"] == "agents"

    def test_scan_untracked_returns_preview(self, service, project_dir):
        """Untracked files include a text preview."""
        orphan = project_dir / ".claude" / "agents" / "preview-test.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        content = "# Preview Test\nSome content here.\n"
        orphan.write_text(content, encoding="utf-8")

        result = service.scan_untracked()
        by_name = {u["canonical_name"]: u for u in result}
        assert by_name["agents/preview-test"]["preview"] == content

    def test_scan_untracked_empty_when_no_orphans(self, service):
        """Returns empty list when all files are tracked."""
        result = service.scan_untracked()
        assert result == []

    def test_scan_untracked_skips_reserved_paths(self, service, project_dir):
        """A file in a reserved subtree is never offered; a plain one still is.

        A reserved artifact can be neither registered nor deleted, so listing
        it as untracked would advertise two actions that both refuse.
        """
        reserved = project_dir / ".claude" / "rules" / "hand-written.md"
        reserved.parent.mkdir(parents=True, exist_ok=True)
        reserved.write_text("# Hand written\n", encoding="utf-8")
        plain = project_dir / ".claude" / "agents" / "plain-untracked.md"
        plain.parent.mkdir(parents=True, exist_ok=True)
        plain.write_text("# Plain\n", encoding="utf-8")

        names = [u["canonical_name"] for u in service.scan_untracked()]
        assert "rules/hand-written" not in names
        assert "agents/plain-untracked" in names

    def test_scan_untracked_skips_seeded_skills(self, service, project_dir):
        """Seeded per-user skills land inside the scan zone and must stay invisible.

        Since the seeding retarget they are written to ``build/.claude/skills``,
        which ``scan_untracked`` walks — without the protected-set skip every
        seeded skill would surface as an orphan the operator is invited to delete.
        """
        seeded = project_dir / ".claude" / "skills" / "seeded-skill" / "SKILL.md"
        seeded.parent.mkdir(parents=True, exist_ok=True)
        seeded.write_text("# Seeded skill\n", encoding="utf-8")

        names = [u["canonical_name"] for u in service.scan_untracked()]
        assert [n for n in names if n.startswith("skills/")] == []


class TestRegisterUntracked:
    """Tests for register_untracked — adding custom files to config."""

    def test_register_untracked_adds_to_config(self, detached_service, detached_project_dir):
        """Registering adds the canonical name to scaffold.user_owned in config."""
        orphan = detached_project_dir / ".claude" / "agents" / "new-rule.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# New Rule\n", encoding="utf-8")

        result = detached_service.register_untracked("agents/new-rule")
        assert result["status"] == "registered"

        user_owned = _get_user_owned(detached_project_dir)
        assert "agents/new-rule" in user_owned

    def test_register_untracked_file_must_exist(self, service):
        """Registering a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found on disk"):
            service.register_untracked("agents/nonexistent")

    def test_register_untracked_already_registered(self, service, project_dir):
        """Registering an already-registered file raises FileExistsError."""
        orphan = project_dir / ".claude" / "agents" / "dupe.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Dupe\n", encoding="utf-8")

        service.register_untracked("agents/dupe")
        svc = ScaffoldGalleryService(project_dir)
        with pytest.raises(FileExistsError, match="already registered"):
            svc.register_untracked("agents/dupe")

    def test_register_untracked_refuses_a_climbing_name(self, service, project_dir, audit_zone):
        """A name is not a menu choice — the listing filters, the API does not.

        ``scan_untracked`` never offers a name like this, but the route reads
        one straight from the request body. Ungated, registering it would read
        an arbitrary file off the host and carry its bytes into the ownership
        store as an artifact body.
        """
        outside = project_dir.parent / "outside.md"
        outside.write_text("# Outside the render\n", encoding="utf-8")

        with pytest.raises(ProtectedArtifactError) as exc:
            service.register_untracked("../../outside")

        assert "not project-relative" in str(exc.value)
        assert "NOTHING WAS REGISTERED" in str(exc.value)
        assert outside.read_text(encoding="utf-8") == "# Outside the render\n"
        assert ScaffoldGalleryService(project_dir)._user_owned == service._user_owned
        assert len(_protected_records(audit_zone)) == 1

    def test_register_untracked_refuses_a_reserved_name(self, service, project_dir, audit_zone):
        """Registering a rule would claim a file the profile's channel writes."""
        target = project_dir / ".claude" / "rules" / "hand-written.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Hand written\n", encoding="utf-8")

        with pytest.raises(ProtectedArtifactError) as exc:
            service.register_untracked("rules/hand-written")

        message = str(exc.value)
        assert "`rules/` convention directory" in message, message
        assert "NOTHING WAS REGISTERED" in message
        assert "rules/hand-written" not in _get_user_owned(project_dir)

        records = _protected_records(audit_zone)
        assert len(records) == 1
        assert records[0]["surface"] == "scaffold_gallery"
        assert records[0]["subject"] == ".claude/rules/hand-written.md"


class TestDeleteUntracked:
    """Tests for delete_untracked — removing orphaned files from disk."""

    def test_delete_untracked_removes_file(self, service, project_dir):
        """Deleting removes the file from disk.

        Targets an agent: the deletable case has to live outside the protected
        set, and ``.claude/agents/`` is a subtree an operator does own.
        """
        orphan = project_dir / ".claude" / "agents" / "to-delete.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Delete Me\n", encoding="utf-8")

        result = service.delete_untracked("agents/to-delete")
        assert result["status"] == "deleted"
        assert not orphan.exists()

    def test_delete_untracked_file_must_exist(self, service):
        """Deleting a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found on disk"):
            service.delete_untracked("agents/ghost")

    def test_delete_untracked_rejects_framework_artifact(self, service, project_dir):
        """Cannot delete a framework-registered artifact via delete_untracked."""
        safety_file = project_dir / ".claude" / "rules" / "safety.md"
        safety_file.parent.mkdir(parents=True, exist_ok=True)
        safety_file.write_text("# Safety\n", encoding="utf-8")

        with pytest.raises(ValueError, match="framework artifact"):
            service.delete_untracked("rules/safety")


class TestDeleteUntrackedProtectedSet:
    """The delete path consults the protected set before it unlinks anything."""

    def test_delete_untracked_refuses_a_climbing_path(self, service, project_dir, audit_zone):
        """A name that climbs out of the project is refused, not resolved and deleted."""
        outside = project_dir.parent / "outside.md"
        outside.write_text("# Outside the render\n", encoding="utf-8")

        with pytest.raises(ProtectedArtifactError) as exc:
            service.delete_untracked("../../outside")

        assert "not project-relative" in str(exc.value)
        assert "NOTHING WAS DELETED" in str(exc.value)
        assert outside.exists()
        assert len(_protected_records(audit_zone)) == 1

    def test_delete_untracked_refuses_an_absolute_path(self, service, tmp_path, audit_zone):
        """An absolute name is refused rather than joined onto the project dir."""
        victim = tmp_path / "victim.md"
        victim.write_text("# Victim\n", encoding="utf-8")

        with pytest.raises(ProtectedArtifactError) as exc:
            service.delete_untracked(str(victim))

        assert "not project-relative" in str(exc.value)
        assert "NOTHING WAS DELETED" in str(exc.value)
        assert victim.exists()
        assert len(_protected_records(audit_zone)) == 1

    def test_delete_untracked_refuses_a_reserved_rule(self, service, project_dir, audit_zone):
        """A rule is instruction text the agent may not delete out from under itself."""
        target = project_dir / ".claude" / "rules" / "hand-written.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Hand written\n", encoding="utf-8")

        with pytest.raises(ProtectedArtifactError) as exc:
            service.delete_untracked("rules/hand-written")

        message = str(exc.value)
        assert "`rules/` convention directory" in message, message
        assert "NOTHING WAS DELETED" in message
        assert target.exists()

        records = _protected_records(audit_zone)
        assert len(records) == 1
        assert records[0]["surface"] == "scaffold_gallery"
        assert records[0]["subject"] == ".claude/rules/hand-written.md"
        assert "`rules/` convention directory" in records[0]["detail"]

    def test_delete_untracked_refuses_a_reserved_skill(self, service, project_dir, audit_zone):
        """A seeded skill body is reserved too, nested path and all."""
        target = project_dir / ".claude" / "skills" / "seeded-skill" / "SKILL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Seeded skill\n", encoding="utf-8")

        with pytest.raises(ProtectedArtifactError) as exc:
            service.delete_untracked("skills/seeded-skill/SKILL.md")

        assert "`skills/` convention directory" in str(exc.value)
        assert "NOTHING WAS DELETED" in str(exc.value)
        assert target.exists()
        assert len(_protected_records(audit_zone)) == 1

    def test_delete_untracked_route_refuses_with_403_and_records_activity(
        self, project_dir, audit_zone
    ):
        """The DELETE route maps the refusal to 403 and publishes it to the ring.

        The service holds no ``Request``, so naming the refusal in the agent's
        activity history is the route's job.
        """
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from osprey.interfaces.web_terminal.routes import scaffold as scaffold_routes

        target = project_dir / ".claude" / "rules" / "route-guard.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Route guard\n", encoding="utf-8")

        app = FastAPI()
        app.include_router(scaffold_routes.router)
        app.state.project_cwd = str(project_dir)
        app.state.agent_activity_ring = []

        response = TestClient(app).delete("/api/scaffold/untracked/rules/route-guard")

        assert response.status_code == 403
        detail = response.json()["detail"]
        assert "`rules/` convention directory" in detail
        assert "NOTHING WAS DELETED" in detail
        assert target.exists()

        assert [event["tool"] for event in app.state.agent_activity_ring] == ["delete_untracked"]
        recorded = app.state.agent_activity_ring[0]["target"]
        assert recorded["kind"] == "artifact"
        assert "rules/route-guard" in recorded["detail"]


class TestCustomArtifacts:
    """Tests for custom user artifacts appearing in list_artifacts and get_content."""

    def test_list_artifacts_includes_custom(self, service, project_dir):
        """After registering a custom file, it appears in list_artifacts."""
        orphan = project_dir / ".claude" / "agents" / "custom.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("---\nsummary: My custom rule\n---\n# Custom\n", encoding="utf-8")

        service.register_untracked("agents/custom")

        svc = ScaffoldGalleryService(project_dir)
        result = svc.list_artifacts()
        by_name = {a["name"]: a for a in result}
        assert "agents/custom" in by_name
        art = by_name["agents/custom"]
        assert art["status"] == "user-owned"
        assert art["custom"] is True
        assert art["category"] == "agents"
        assert art["summary"] == "My custom rule"

    def test_get_content_custom_artifact(self, service, project_dir):
        """get_content works for registered custom files."""
        orphan = project_dir / ".claude" / "agents" / "readable.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        content = "# Readable Custom Rule\nContent here.\n"
        orphan.write_text(content, encoding="utf-8")

        service.register_untracked("agents/readable")

        svc = ScaffoldGalleryService(project_dir)
        result = svc.get_content("agents/readable")
        assert result["source"] == "user-owned"
        assert result["content"] == content

    def test_get_content_unknown_artifact_raises(self, service):
        """get_content for a completely unknown name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown artifact"):
            service.get_content("rules/totally-unknown")

    def test_save_override_custom_artifact(self, detached_service, detached_project_dir):
        """save_override works for registered custom files."""
        orphan = detached_project_dir / ".claude" / "agents" / "editable.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Original\n", encoding="utf-8")

        detached_service.register_untracked("agents/editable")

        svc = ScaffoldGalleryService(detached_project_dir)
        new_content = "# Edited Custom Rule\nUpdated content.\n"
        result = svc.save_override("agents/editable", new_content)
        assert result["status"] == "saved"
        assert orphan.read_text(encoding="utf-8") == new_content

    def test_unoverride_custom_with_delete(self, detached_service, detached_project_dir):
        """Unclaiming a custom artifact with delete_file=True removes the file."""
        orphan = detached_project_dir / ".claude" / "agents" / "removable.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Removable\n", encoding="utf-8")

        detached_service.register_untracked("agents/removable")

        svc = ScaffoldGalleryService(detached_project_dir)
        result = svc.unoverride("agents/removable", delete_file=True)
        assert result["status"] == "removed"
        assert result["deleted_file"] is True
        assert not orphan.exists()

        user_owned = _get_user_owned(detached_project_dir)
        assert "agents/removable" not in user_owned


# ===========================================================================
# Unoverride framework restore
# ===========================================================================


# ===========================================================================
# Create artifact
# ===========================================================================


class TestCreateArtifact:
    """Creating a custom artifact, on whichever surface will keep it.

    In profile mode that surface is the profile, not the project: the project's
    ``config.yml`` is derived output the next build regenerates from the
    profile, so a registration written there would be dropped and the file
    beside it pruned as unowned. The operator would watch the artifact appear
    and then quietly vanish at the next build.
    """

    def _profile_root(self, project_dir: Path) -> Path:
        manifest = json.loads((project_dir / ".osprey-manifest.json").read_text(encoding="utf-8"))
        return Path(manifest["build_args"]["profile_path_abs"]).parent

    def test_create_artifact_writes_the_profile_copy(self, service, project_dir):
        """The new artifact lands in the profile's convention directory."""
        result = service.create_artifact("agents", "my-agent")
        assert result["status"] == "created"
        assert result["created_in_profile"] is True
        assert result["output_path"] == ".claude/agents/my-agent.md"

        slot = self._profile_root(project_dir) / "agents" / "my-agent.md"
        assert slot.is_file()
        assert len(slot.read_text(encoding="utf-8")) > 0
        assert not (project_dir / ".claude" / "agents" / "my-agent.md").exists(), (
            "the project copy is the build's to make, from the profile"
        )

    def test_create_artifact_writes_no_project_ownership(self, service, project_dir):
        """Registration is the next build's convention scan, not the gallery's."""
        service.create_artifact("agents", "unregistered")

        assert "agents/unregistered" not in _get_user_owned(project_dir)
        manifest = json.loads((project_dir / ".osprey-manifest.json").read_text(encoding="utf-8"))
        assert "agents/unregistered" not in manifest.get("user_owned", {})

    def test_a_created_artifact_is_owned_and_visible_immediately(self, service, project_dir):
        """It must not disappear from the gallery until someone rebuilds."""
        service.create_artifact("agents", "shift-handover", "# Shift handover\nCheck the orbit.\n")

        svc = ScaffoldGalleryService(project_dir)
        assert "agents/shift-handover" in svc._user_owned
        listed = {a["name"]: a for a in svc.list_artifacts()}
        assert listed["agents/shift-handover"]["status"] == "user-owned"
        assert svc.get_content("agents/shift-handover")["content"].startswith("# Shift handover")

    def test_a_created_artifact_can_be_edited_where_it_now_lives(self, service, project_dir):
        """Create → edit → save has to land on the profile copy, like a claim."""
        service.create_artifact("agents", "editable", "# First\n")

        svc = ScaffoldGalleryService(project_dir)
        svc.save_override("agents/editable", "# Second\n")

        slot = self._profile_root(project_dir) / "agents" / "editable.md"
        assert slot.read_text(encoding="utf-8") == "# Second\n"

    def test_create_refuses_when_the_project_already_carries_the_file(self, service, project_dir):
        """That is a claim — which moves the body — not a creation."""
        existing = project_dir / ".claude" / "agents" / "already-here.md"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("# Written by hand\n", encoding="utf-8")

        with pytest.raises(FileExistsError, match="claim it instead"):
            service.create_artifact("agents", "already-here")
        assert existing.read_text(encoding="utf-8") == "# Written by hand\n"

    def test_create_refuses_in_a_degraded_project(self, degraded_project_dir):
        """No reachable profile means nowhere durable to author into."""
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        svc = ScaffoldGalleryService(degraded_project_dir)
        with pytest.raises(ScaffoldClaimError):
            svc.create_artifact("agents", "nowhere-to-put-this")
        assert not (degraded_project_dir / ".claude" / "agents" / "nowhere-to-put-this.md").exists()

    def test_create_still_registers_in_config_mode(self, detached_service, detached_project_dir):
        """A pre-profile project has no profile, so config.yml is still the place."""
        detached_service.create_artifact("agents", "legacy-agent")

        assert "agents/legacy-agent" in _get_user_owned(detached_project_dir)
        assert (detached_project_dir / ".claude" / "agents" / "legacy-agent.md").is_file()

    def test_create_artifact_invalid_category_raises(self, service):
        """Invalid category raises ValueError."""
        with pytest.raises(ValueError, match="Invalid category"):
            service.create_artifact("widgets", "bad-widget")

    def test_create_artifact_conflict_with_framework_raises(self, service):
        """Creating an artifact with a framework name raises ValueError."""
        with pytest.raises(ValueError, match="framework artifact"):
            service.create_artifact("agents", "channel-finder")

    def test_create_artifact_duplicate_raises(self, service, project_dir):
        """Creating the same artifact twice raises FileExistsError."""
        service.create_artifact("agents", "dupe-test")
        svc = ScaffoldGalleryService(project_dir)
        with pytest.raises(FileExistsError, match="already exists"):
            svc.create_artifact("agents", "dupe-test")

    def test_create_artifact_hook_gets_py_extension(self, service, project_dir):
        """Hook artifacts are created with .py extension, not .md."""
        result = service.create_artifact("hooks", "my-hook")
        assert result["output_path"] == ".claude/hooks/my-hook.py"

        slot = self._profile_root(project_dir) / "hooks" / "my-hook.py"
        assert slot.is_file()
        assert not slot.with_suffix(".md").exists()

    def test_a_created_hook_is_owned_under_its_filename(self, service, project_dir):
        """A hook is known by its filename — that is what settings.json wires.

        Assembling the name as ``<category>/<name>`` instead would drop the
        ``.py``, and every later lookup would resolve it back to a ``.md`` path
        that was never written: the body would be stored against a file nothing
        reads, and releasing it would delete nothing.
        """
        result = service.create_artifact("hooks", "shift-check")

        assert result["canonical_name"] == "hooks/shift-check.py"
        svc = ScaffoldGalleryService(project_dir)
        assert svc._canonical_to_path("hooks/shift-check.py") == ".claude/hooks/shift-check.py"
        assert svc.get_content("hooks/shift-check.py")["content"]

    def test_create_artifact_hook_executable(self, service, project_dir):
        """Hook .py files get the execute permission bit."""
        import stat

        service.create_artifact("hooks", "exec-hook")
        slot = self._profile_root(project_dir) / "hooks" / "exec-hook.py"
        assert slot.stat().st_mode & stat.S_IXUSR

    def test_create_artifact_starter_content(self, service, project_dir):
        """Default content is category-appropriate starter content."""
        profile_root = self._profile_root(project_dir)

        service.create_artifact("agents", "starter-test")
        content = (profile_root / "agents" / "starter-test.md").read_text(encoding="utf-8")
        assert content.startswith("# Starter Test")

        # The other shape ``_starter_content`` knows. The skill branch is no
        # longer reachable from here — ``.claude/skills/**`` is reserved, and
        # the refusal is pinned in :class:`TestCreateArtifactProtectedSet`.
        svc = ScaffoldGalleryService(project_dir)
        svc.create_artifact("hooks", "starter-hook")
        hook_content = (profile_root / "hooks" / "starter-hook.py").read_text(encoding="utf-8")
        assert hook_content.startswith('"""Hook: Starter Hook."""')

    @pytest.mark.parametrize("bad", ["../escape", "../../etc/passwd"])
    def test_create_refuses_a_name_that_climbs_out(self, service, project_dir, bad):
        """The name is resolved, not concatenated, so it cannot escape.

        Assembling ``.claude/rules/<name>.md`` by hand let a name containing
        ``..`` address a file outside the project entirely.
        """
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        with pytest.raises((ScaffoldClaimError, ValueError)):
            service.create_artifact("rules", bad)
        assert not (project_dir.parent / "escape.md").exists()

    def test_a_nested_name_is_legitimate(self, service, project_dir):
        """Markdown categories nest — ``commands/osprey/scan`` is a real name.

        Refusing every name with a slash would have been the easy way to stop
        the traversal above, and it would have broken nesting the convention
        table explicitly allows.
        """
        result = service.create_artifact("commands", "osprey/handover")

        assert result["canonical_name"] == "commands/osprey/handover"
        slot = self._profile_root(project_dir) / "commands" / "osprey" / "handover.md"
        assert slot.is_file()

    def test_register_untracked_writes_manifest(self, detached_service, detached_project_dir):
        """Regression: config-mode register_untracked also writes a manifest entry."""
        project_dir = detached_project_dir
        orphan = project_dir / ".claude" / "agents" / "manifest-reg.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Manifest Reg\n", encoding="utf-8")

        detached_service.register_untracked("agents/manifest-reg")

        manifest_path = project_dir / ".osprey-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest.get("user_owned", {}).get("agents/manifest-reg")
        assert entry is not None
        assert "claimed_at" in entry


class TestUnoverrideFrameworkRestore:
    """Tests for unoverride restoring framework file content on disk.

    Config mode throughout: releasing is only the last word on ownership where
    nothing else supplies the artifact. Where a profile does, the next build
    registers it again — see :class:`TestProfileMode`.
    """

    def test_unoverride_restores_framework_file(self, detached_service, detached_project_dir):
        """Claim, customize, release → get_content returns framework content."""
        # Claim the artifact
        detached_service.scaffold_override(WRITABLE_ARTIFACT)

        # Customize it with user content
        custom_text = "# My custom safety rules\nUser wrote this.\n"
        detached_service.save_override(WRITABLE_ARTIFACT, custom_text)

        # Release to framework with delete_file=True (what the web UI sends)
        svc = ScaffoldGalleryService(detached_project_dir)
        svc.unoverride(WRITABLE_ARTIFACT, delete_file=True)

        # After release, get_content should return framework content, not custom
        svc2 = ScaffoldGalleryService(detached_project_dir)
        result = svc2.get_content(WRITABLE_ARTIFACT)
        assert result["source"] == "framework"
        assert result["content"] != custom_text

    def test_unoverride_restored_content_matches_render(
        self, detached_service, detached_project_dir
    ):
        """After release, disk file matches _render_framework() output exactly."""
        # Get expected framework content before any changes
        expected = detached_service.get_framework_content(WRITABLE_ARTIFACT)

        # Claim and customize
        detached_service.scaffold_override(WRITABLE_ARTIFACT)
        detached_service.save_override(WRITABLE_ARTIFACT, "# Totally different content\n")

        # Release to framework
        svc = ScaffoldGalleryService(detached_project_dir)
        result = svc.unoverride(WRITABLE_ARTIFACT, delete_file=True)
        assert result["restored_file"] is True

        # Verify disk matches rendered template exactly
        art = svc._registry.get(WRITABLE_ARTIFACT)
        disk_path = detached_project_dir / art.output_path
        assert disk_path.read_text(encoding="utf-8") == expected

    def test_unoverride_without_delete_file_preserves_disk(
        self, detached_service, detached_project_dir
    ):
        """delete_file=False does NOT touch the file (CLI deferred-regen contract)."""
        # Claim and customize
        detached_service.scaffold_override(WRITABLE_ARTIFACT)
        custom_text = "# My custom rules — should persist\n"
        detached_service.save_override(WRITABLE_ARTIFACT, custom_text)

        # Release WITHOUT delete_file (CLI semantics)
        svc = ScaffoldGalleryService(detached_project_dir)
        svc.unoverride(WRITABLE_ARTIFACT, delete_file=False)

        # File on disk should still be the user's customized version
        art = svc._registry.get(WRITABLE_ARTIFACT)
        disk_path = detached_project_dir / art.output_path
        assert disk_path.read_text(encoding="utf-8") == custom_text


# ===========================================================================
# Profile mode — the project names a profile this process can reach
# ===========================================================================


class TestProfileMode:
    """The gallery delegates to the CLI claim path and follows the artifact.

    A claim MOVES the artifact into the profile's convention directory, which
    is the source of truth the next build registers ownership from. That makes
    the profile copy — not the project copy the claim removed — the thing the
    operator is looking at from then on.
    """

    def _profile_root(self, project_dir: Path) -> Path:
        manifest = json.loads((project_dir / ".osprey-manifest.json").read_text(encoding="utf-8"))
        return Path(manifest["build_args"]["profile_path_abs"]).parent

    def test_claim_moves_artifact_into_the_profile(self, service, project_dir):
        """The artifact lands in the profile, and leaves the project tree."""
        result = service.scaffold_override(WRITABLE_ARTIFACT)

        assert result["moved_to_profile"] is True
        profile_copy = self._profile_root(project_dir) / "agents" / "channel-finder.md"
        assert profile_copy.is_file()
        assert not (project_dir / result["output_path"]).exists()

    def test_claim_writes_no_project_ownership(self, service, project_dir):
        """Registration is the next build's job, not the gallery's.

        Writing config.yml here would be the second source of truth the profile
        model removes — and the next build would overwrite it anyway.
        """
        service.scaffold_override(WRITABLE_ARTIFACT)
        assert WRITABLE_ARTIFACT not in _get_user_owned(project_dir)

    def test_claimed_artifact_stays_visible_and_owned(self, service, project_dir):
        """A claim must not make the artifact disappear from the gallery.

        The project copy is gone until the next build, so a gallery that only
        looked at the project tree would show the operator's own artifact
        vanishing the moment they claimed it.
        """
        service.scaffold_override(WRITABLE_ARTIFACT)

        svc = ScaffoldGalleryService(project_dir)
        assert WRITABLE_ARTIFACT in svc._user_owned
        listed = {a["name"]: a for a in svc.list_artifacts()}
        assert listed[WRITABLE_ARTIFACT]["status"] == "user-owned"

    def test_save_writes_the_profile_copy(self, service, project_dir):
        """An edit has to land where the next build reads from.

        Writing the project copy would look identical in the UI and be gone at
        the next `build` — the same silent loss the claim path exists
        to prevent, one layer up.
        """
        service.scaffold_override(WRITABLE_ARTIFACT)

        svc = ScaffoldGalleryService(project_dir)
        edited = "# Facility safety rules\nTwo-person rule for magnet writes.\n"
        svc.save_override(WRITABLE_ARTIFACT, edited)

        profile_copy = self._profile_root(project_dir) / "agents" / "channel-finder.md"
        assert profile_copy.read_text(encoding="utf-8") == edited
        assert (
            ScaffoldGalleryService(project_dir).get_content(WRITABLE_ARTIFACT)["content"] == edited
        )

    def test_register_untracked_moves_the_file_too(self, service, project_dir):
        """A hand-written file's durable home is the profile, same as a claim."""
        orphan = project_dir / ".claude" / "agents" / "hand-written.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Hand written\n", encoding="utf-8")

        result = service.register_untracked("agents/hand-written")

        assert result["status"] == "registered"
        assert (self._profile_root(project_dir) / "agents" / "hand-written.md").is_file()
        assert not orphan.exists()

    def test_refusal_reaches_the_caller_with_its_message(self, service, monkeypatch):
        """A CLI refusal is surfaced verbatim, not paraphrased or swallowed.

        Every refusal names what was refused and what to do instead; that text
        is the whole value of the refusal to an operator. The route family
        turns it into a 409 with the message intact — re-wrapping it here would
        throw away the only part worth reading.
        """
        from osprey.cli import scaffold_cmd

        message = "Nothing was moved.\n  Rebuild it from a profile, then claim again."

        def _refuse(project_dir, name):
            raise scaffold_cmd.ScaffoldClaimError(message)

        monkeypatch.setattr(scaffold_cmd, "claim_into_profile", _refuse)

        with pytest.raises(scaffold_cmd.ScaffoldClaimError) as raised:
            service.scaffold_override(WRITABLE_ARTIFACT)
        assert str(raised.value) == message


# ===========================================================================
# Volume mode — deployed container, per-user claude-config volume
# ===========================================================================


class TestVolumeMode:
    """Claims in a deployed container land on the volume, and survive it."""

    def test_claim_writes_the_store_not_config(self, container_project, volume_dir):
        """config.yml is image-baked: writing it would vanish with the container."""
        svc = ScaffoldGalleryService(container_project)
        before = _get_user_owned(container_project)

        svc.scaffold_override(WRITABLE_ARTIFACT)

        assert _get_user_owned(container_project) == before
        assert _store_index(volume_dir)["artifacts"][WRITABLE_ARTIFACT]["state"] == "claimed"

    def test_claim_survives_a_container_recreation(self, container_project, volume_dir, tmp_path):
        """The test this feature exists for.

        A recreated container gets a pristine tree from the image and the same
        volume. The operator's claim — and the content it was a claim of — has
        to still be there, or the claim silently never happened.
        """
        pristine = _recreate_container(container_project, tmp_path / "image-rebuild")

        svc = ScaffoldGalleryService(container_project)
        svc.scaffold_override(WRITABLE_ARTIFACT)
        edited = "# Safety rules\nOperator wrote this in the container.\n"
        svc.save_override(WRITABLE_ARTIFACT, edited)

        # The recreated container: fresh tree, same volume, app start.
        assert restore_scaffold_bodies(pristine) == [WRITABLE_ARTIFACT]
        recreated = ScaffoldGalleryService(pristine)

        assert WRITABLE_ARTIFACT in recreated._user_owned
        assert recreated.get_content(WRITABLE_ARTIFACT)["content"] == edited
        art = recreated._registry.get(WRITABLE_ARTIFACT)
        assert (pristine / art.output_path).read_text(encoding="utf-8") == edited, (
            "the running agent reads the project tree, so the body must be restored there too"
        )

    def test_created_artifact_survives_a_container_recreation(
        self, container_project, volume_dir, tmp_path
    ):
        """A file that exists only in the container is the easiest thing to lose."""
        pristine = _recreate_container(container_project, tmp_path / "image-rebuild")

        svc = ScaffoldGalleryService(container_project)
        svc.create_artifact("agents", "shift-handover", "# Shift handover\nCheck the orbit.\n")

        restore_scaffold_bodies(pristine)
        recreated = ScaffoldGalleryService(pristine)
        assert "agents/shift-handover" in recreated._user_owned
        assert (pristine / ".claude" / "agents" / "shift-handover.md").is_file()

    def test_release_outranks_build_derived_ownership(self, container_project, volume_dir):
        """Merge precedence: the store's record wins over config.yml.

        config.yml is what the profile owned when the image was built. The
        store is what this operator did afterwards — later, and by hand. If the
        build-derived entry won instead, unclaim would have nowhere to be
        recorded and the button would report success and change nothing.
        """
        baked = _get_user_owned(container_project)
        assert baked, "the preset build is expected to derive some ownership"
        name = baked[0]

        ScaffoldGalleryService(container_project).unoverride(name)

        svc = ScaffoldGalleryService(container_project)
        assert name not in svc._user_owned
        assert _store_index(volume_dir)["artifacts"][name]["state"] == "released"

    def test_release_survives_a_container_recreation(self, container_project, volume_dir, tmp_path):
        """A release is durable for the same reason a claim is."""
        pristine = _recreate_container(container_project, tmp_path / "image-rebuild")
        name = _get_user_owned(container_project)[0]

        ScaffoldGalleryService(container_project).unoverride(name)

        assert name not in ScaffoldGalleryService(pristine)._user_owned

    def test_claim_adds_to_build_derived_ownership(self, container_project):
        """The gallery reads the union, not one surface or the other."""
        baked = set(_get_user_owned(container_project))

        ScaffoldGalleryService(container_project).scaffold_override(WRITABLE_ARTIFACT)

        owned = set(ScaffoldGalleryService(container_project)._user_owned)
        assert baked <= owned
        assert WRITABLE_ARTIFACT in owned

    def test_corrupt_store_does_not_take_the_gallery_down(self, container_project, volume_dir):
        """The store is on a volume — truncation is a thing that happens.

        Build-derived ownership is still readable, so the gallery degrades to
        that rather than failing every request.
        """
        ScaffoldGalleryService(container_project).scaffold_override(WRITABLE_ARTIFACT)
        index = volume_dir / "osprey" / "scaffold" / "user_owned.json"
        index.write_text('{"artifacts": {"agents/channel-finder": {"stat', encoding="utf-8")

        svc = ScaffoldGalleryService(container_project)
        assert svc.list_artifacts()
        assert set(_get_user_owned(container_project)) <= set(svc._user_owned)

    def test_corrupt_store_is_set_aside_not_overwritten(self, container_project, volume_dir):
        """A later claim recovers the store, keeping the damaged file as evidence."""
        store_dir = volume_dir / "osprey" / "scaffold"
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / "user_owned.json").write_text("}{ not json", encoding="utf-8")

        ScaffoldGalleryService(container_project).scaffold_override(WRITABLE_ARTIFACT)

        assert (store_dir / "user_owned.json.corrupt").is_file()
        assert _store_index(volume_dir)["artifacts"][WRITABLE_ARTIFACT]["state"] == "claimed"

    def test_concurrent_claims_do_not_lose_records(self, container_project, volume_dir):
        """Requests are served from a thread pool; a claim must not be lost."""
        import threading

        names = [f"agents/concurrent-{i}" for i in range(8)]
        for name in names:
            path = container_project / ".claude" / "agents" / f"{name.split('/')[1]}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {name}\n", encoding="utf-8")

        def register(name: str) -> None:
            ScaffoldGalleryService(container_project).register_untracked(name)

        threads = [threading.Thread(target=register, args=(name,)) for name in names]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        recorded = _store_index(volume_dir)["artifacts"]
        assert set(names) <= set(recorded)

    def test_record_for_an_artifact_the_image_dropped_is_harmless(
        self, container_project, volume_dir
    ):
        """A claim outlives the image that carried it; the gallery must not care."""
        store_dir = volume_dir / "osprey" / "scaffold"
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / "user_owned.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "artifacts": {
                        "rules/retired": {
                            "state": "claimed",
                            "output_path": ".claude/rules/retired.md",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        svc = ScaffoldGalleryService(container_project)
        listed = {a["name"] for a in svc.list_artifacts()}

        assert "rules/retired" in svc._user_owned
        assert "rules/retired" not in listed
        assert not (container_project / ".claude" / "rules" / "retired.md").exists()

    def test_local_edit_is_not_clobbered_by_the_durable_copy(self, container_project, volume_dir):
        """Rehydration restores an image-fresh file, never a newer local edit.

        Someone editing through the terminal rather than the gallery is doing
        the same work by another route; overwriting it on the next request
        would destroy it.
        """
        svc = ScaffoldGalleryService(container_project)
        svc.scaffold_override(WRITABLE_ARTIFACT)
        svc.save_override(WRITABLE_ARTIFACT, "# Saved through the gallery\n")

        art = svc._registry.get(WRITABLE_ARTIFACT)
        edited_in_terminal = "# Edited in the terminal, not the gallery\n"
        (container_project / art.output_path).write_text(edited_in_terminal, encoding="utf-8")

        assert restore_scaffold_bodies(container_project) == []

        assert (container_project / art.output_path).read_text(
            encoding="utf-8"
        ) == edited_in_terminal


# ===========================================================================
# Degraded topology — a profile is named but cannot be reached
# ===========================================================================


class TestDegradedTopology:
    """No durable surface, so every write refuses — it does not fall back.

    Recording ownership in the project's own config.yml here would report
    success and then be erased by the next build, which regenerates config.yml
    from the profile. Refusing is the honest answer, and it is the same answer
    ``osprey scaffold claim`` gives.
    """

    def test_claim_refuses_with_the_cli_message(self, degraded_project_dir):
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        svc = ScaffoldGalleryService(degraded_project_dir)
        with pytest.raises(ScaffoldClaimError, match="profile"):
            svc.scaffold_override(WRITABLE_ARTIFACT)

    def test_claim_writes_nothing(self, degraded_project_dir):
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        before = _get_user_owned(degraded_project_dir)
        svc = ScaffoldGalleryService(degraded_project_dir)
        with pytest.raises(ScaffoldClaimError):
            svc.scaffold_override(WRITABLE_ARTIFACT)

        assert _get_user_owned(degraded_project_dir) == before

    def test_create_artifact_refuses(self, degraded_project_dir):
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        svc = ScaffoldGalleryService(degraded_project_dir)
        with pytest.raises(ScaffoldClaimError, match="cannot be\n *reached|cannot be reached"):
            svc.create_artifact("agents", "nowhere-to-put-this")

    def test_release_refuses(self, degraded_project_dir):
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        owned = _get_user_owned(degraded_project_dir)
        assert owned, "the preset build is expected to derive some ownership"

        svc = ScaffoldGalleryService(degraded_project_dir)
        with pytest.raises(ScaffoldClaimError):
            svc.unoverride(owned[0])

    def test_the_gallery_still_opens(self, degraded_project_dir):
        """Reads must keep working — a refused write is not a broken page."""
        svc = ScaffoldGalleryService(degraded_project_dir)
        assert svc.list_artifacts()
        assert svc.get_content(SAFE_ARTIFACT)["content"]


# ===========================================================================
# Restore-path containment
# ===========================================================================


class TestRestoreContainment:
    """The store is data on a mounted volume, so its paths are not trusted.

    An entry's ``output_path`` decides where a restore writes. Left
    unconstrained it could name ``.env``, ``config.yml``, or a path climbing out
    of the project altogether, and the app would obligingly overwrite it at
    startup.
    """

    def _seed(self, volume_dir: Path, output_path: str) -> None:
        store_dir = volume_dir / "osprey" / "scaffold"
        (store_dir / "files").mkdir(parents=True, exist_ok=True)
        (store_dir / "user_owned.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "artifacts": {
                        "rules/planted": {"state": "claimed", "output_path": output_path}
                    },
                }
            ),
            encoding="utf-8",
        )

    @pytest.mark.parametrize(
        "output_path",
        [
            "config.yml",
            ".env",
            "../escaped.md",
            "/etc/passwd",
            ".claude/../../escaped.md",
            # Inside the ownable tree, and still refused: the build generates
            # it, so a stored copy would shadow the regenerated one forever.
            ".claude/hooks/hook_config.json",
            ".claude/settings.json",
        ],
    )
    def test_restore_refuses_paths_outside_the_ownable_tree(
        self, container_project, volume_dir, output_path
    ):
        self._seed(volume_dir, output_path)

        assert restore_scaffold_bodies(container_project) == []

    def test_a_planted_path_does_not_overwrite_config(self, container_project, volume_dir):
        self._seed(volume_dir, "config.yml")
        store_dir = volume_dir / "osprey" / "scaffold"
        (store_dir / "files").mkdir(parents=True, exist_ok=True)
        before = (container_project / "config.yml").read_text(encoding="utf-8")

        restore_scaffold_bodies(container_project)

        assert (container_project / "config.yml").read_text(encoding="utf-8") == before


# ===========================================================================
# Artifact shapes the naming rules trip over
# ===========================================================================


class TestArtifactShapes:
    """Two shapes whose paths do not follow the ``<name>.md`` assumption."""

    def test_hook_paths_keep_their_py_suffix(self, container_project, volume_dir):
        """A hook is a ``.py`` script and is owned under a name that says so.

        Appending ``.md`` to it would record the claimed body against a path
        that does not exist, so the body would be stored and never read back.
        """
        svc = ScaffoldGalleryService(container_project)
        assert svc._canonical_to_path("hooks/osprey_cf_feedback_capture.py") == (
            ".claude/hooks/osprey_cf_feedback_capture.py"
        )
        assert svc._canonical_to_path("rules/safety") == ".claude/rules/safety.md"

    def test_a_claimed_hook_body_round_trips(self, container_project, volume_dir):
        """The body of a hook claim is retrievable, not written into the void."""
        hook = container_project / ".claude" / "hooks" / "shift-check.py"
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text("#!/usr/bin/env python\nprint('ok')\n", encoding="utf-8")

        svc = ScaffoldGalleryService(container_project)
        svc._record_claim("hooks/shift-check.py", hook.read_text(encoding="utf-8"))

        stored = _store_index(volume_dir)["artifacts"]["hooks/shift-check.py"]
        assert stored["output_path"] == ".claude/hooks/shift-check.py"
        body = volume_dir / "osprey" / "scaffold" / "files" / ".claude" / "hooks" / "shift-check.py"
        assert body.read_text(encoding="utf-8") == hook.read_text(encoding="utf-8")

    def test_directory_artifacts_are_not_read_as_text(self, service, project_dir):
        """A skill is a directory. Reading its profile slot as text would raise.

        The gallery edits single files; a directory-shaped artifact has no body
        to open, and asking for one must not crash the page.
        """
        skills = [a["name"] for a in service.list_artifacts() if a["name"].startswith("skills/")]
        assert skills, "the control-assistant preset is expected to ship a skill"

        for name in skills:
            assert service._profile_file(name) is None


class TestVolumeSaveDurability:
    """Editing an artifact you already own is as losable as claiming one."""

    def test_editing_a_build_derived_artifact_survives_recreation(
        self, container_project, volume_dir, tmp_path
    ):
        """Ownership from the image's config.yml carries no store record.

        A save that wrote only the body would leave it on the volume with
        nothing pointing at it, and the restore — which walks the records —
        would skip it. The operator's edit would sit intact on a disk nobody
        reads and come back as the framework's original.
        """
        name = WRITABLE_ARTIFACT
        _add_user_owned(container_project, name)
        pristine = _recreate_container(container_project, tmp_path / "image-rebuild")

        svc = ScaffoldGalleryService(container_project)
        edited = f"# {name}\nEdited in the container, never claimed separately.\n"
        result = svc.save_override(name, edited)

        assert restore_scaffold_bodies(pristine) == [name]
        assert (pristine / result["path"]).read_text(encoding="utf-8") == edited


# ===========================================================================
# What may be owned at all
# ===========================================================================


class TestGeneratedPathsAreNeverOwnable:
    """A generated file must not be frozen by a claim, in any mode.

    ``osprey scaffold claim`` refuses these already, so the two modes that hand
    it the claim inherit the rule. The two that record ownership themselves —
    a container writing the volume, a pre-profile project writing config.yml —
    have to apply it themselves or the gallery is a way around it.
    """

    #: The catalog name and the project path of a generated artifact that lives
    #: inside an otherwise claimable channel. This one is the write-safety
    #: layer's runtime configuration: ``osprey_writes_check.py`` reads its
    #: ``write_tools``, and the render derives that from the resolved config.
    HOOK_CONFIG = ("hooks/hook-config", ".claude/hooks/hook_config.json")

    def test_claim_of_a_generated_hook_config_is_refused_on_the_volume(
        self, container_project, volume_dir
    ):
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        name, path = self.HOOK_CONFIG
        svc = ScaffoldGalleryService(container_project)
        with pytest.raises(ScaffoldClaimError, match="generated, not authored"):
            svc.scaffold_override(name)

        assert _store_index(volume_dir) == {}, "a refused claim records nothing"
        assert not (volume_dir / "osprey" / "scaffold" / "files" / path).exists()

    def test_claim_of_settings_json_is_refused_on_the_volume(self, container_project, volume_dir):
        """settings.json is the permissions channel — the profile's config keys own it."""
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        svc = ScaffoldGalleryService(container_project)
        with pytest.raises(ScaffoldClaimError, match="generated, not authored"):
            svc.scaffold_override("settings-json")
        assert _store_index(volume_dir) == {}

    def test_claim_of_a_generated_path_is_refused_in_config_mode(self, detached_service):
        """The legacy surface is durable, which makes freezing it worse, not better."""
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        name, _ = self.HOOK_CONFIG
        before = _get_user_owned(detached_service.project_dir)
        with pytest.raises(ScaffoldClaimError, match="generated, not authored"):
            detached_service.scaffold_override(name)
        assert _get_user_owned(detached_service.project_dir) == before

    def test_the_other_spelling_is_refused_too(self, container_project, volume_dir, audit_zone):
        """Ownership follows the path, so naming it by filename changes nothing.

        ``register_untracked`` takes the name the file has on disk rather than
        the catalog canonical, and it reaches the store by a different route —
        a guard on one spelling only would be no guard at all.

        Which of the two guards answers changed with the protected set: the
        registration path now consults it before it touches the filesystem, and
        ``hook_config.json`` is reserved as well as generated, so the refusal
        arrives as the protected one. Both name the channel that writes the
        file, and the assertion that matters — nothing recorded — is unchanged.
        """
        svc = ScaffoldGalleryService(container_project)
        with pytest.raises(ProtectedArtifactError, match="write-safety layer"):
            svc.register_untracked("hooks/hook_config.json")
        assert _store_index(volume_dir) == {}

    def test_a_regenerated_hook_config_is_never_restored_over(
        self, container_project, volume_dir, tmp_path
    ):
        """The failure the refusal exists to prevent, asserted end to end.

        A frozen copy on the volume would be written back over the freshly
        generated file at every container start, so a rebuilt image would ship
        a new write-tool set and the container would go on running the stale
        one — silently, and for as long as the volume lived.
        """
        _name, path = self.HOOK_CONFIG
        pristine = _recreate_container(container_project, tmp_path / "image-rebuild")
        generated = (pristine / path).read_text(encoding="utf-8")

        # Plant the claim directly: the gallery refuses to create it, and the
        # point here is that the restore refuses to act on one either way.
        store_dir = volume_dir / "osprey" / "scaffold"
        (store_dir / "files" / ".claude" / "hooks").mkdir(parents=True, exist_ok=True)
        (store_dir / "files" / path).write_text('{"write_tools": []}', encoding="utf-8")
        (store_dir / "user_owned.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "artifacts": {"hooks/hook-config": {"state": "claimed", "output_path": path}},
                }
            ),
            encoding="utf-8",
        )

        assert restore_scaffold_bodies(pristine) == []
        assert (pristine / path).read_text(encoding="utf-8") == generated

    def test_an_artifact_the_store_cannot_hold_is_refused(self, container_project, volume_dir):
        """A record with no body would claim the framework's own text.

        CLAUDE.md lives outside the ownable tree, so the store has nowhere to
        keep it. Recording ownership anyway would report success, keep no copy,
        and hand the operator the image's original back after the next
        recreation while still calling it theirs.
        """
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        svc = ScaffoldGalleryService(container_project)
        with pytest.raises(ScaffoldClaimError):
            svc.scaffold_override("claude-md")

        assert _store_index(volume_dir) == {}
        assert "claude-md" not in ScaffoldGalleryService(container_project)._user_owned

    def test_a_refused_claim_leaves_no_file_behind(self, container_project):
        """The claim path renders missing artifacts to disk — a refusal must not."""
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        _name, path = self.HOOK_CONFIG
        (container_project / path).unlink()

        svc = ScaffoldGalleryService(container_project)
        with pytest.raises(ScaffoldClaimError):
            svc.scaffold_override("hooks/hook-config")
        assert not (container_project / path).exists()

    def _poison_index(self, volume_dir: Path, name: str, output_path: str | None) -> None:
        """Plant an ownership record the way an older OSPREY would have."""
        store_dir = volume_dir / "osprey" / "scaffold"
        store_dir.mkdir(parents=True, exist_ok=True)
        entry: dict = {"state": "claimed"}
        if output_path is not None:
            entry["output_path"] = output_path
        (store_dir / "user_owned.json").write_text(
            json.dumps({"version": 1, "artifacts": {name: entry}}), encoding="utf-8"
        )

    @pytest.mark.parametrize(
        ("name", "output_path"),
        [
            ("hooks/hook-config", ".claude/hooks/hook_config.json"),
            ("hooks/hook_config.json", ".claude/hooks/hook_config.json"),
            # Written before the guard existed, so it carries no path at all —
            # the name is the only thing left to recognize it by.
            ("hooks/hook-config", None),
            ("settings-json", ".claude/settings.json"),
        ],
    )
    def test_a_planted_record_is_not_ownership(
        self, container_project, volume_dir, name, output_path
    ):
        """Containment covers bodies; a record is not a body.

        The store outlives the image, so a record can predate the guard that
        would refuse it today. Left to flow through, it tells the operator they
        own the write-safety layer's configuration — and the gallery would show
        them the framework's own text as theirs.
        """
        self._poison_index(volume_dir, name, output_path)

        svc = ScaffoldGalleryService(container_project)

        assert name not in svc._user_owned
        assert all(a["name"] != name for a in svc.list_artifacts() if a["status"] == "user-owned")

    def test_a_planted_record_does_not_survive_a_recreation(
        self, container_project, volume_dir, tmp_path
    ):
        """Not listed, not restored, and the generated file is untouched."""
        path = ".claude/hooks/hook_config.json"
        pristine = _recreate_container(container_project, tmp_path / "image-rebuild")
        generated = (pristine / path).read_text(encoding="utf-8")

        self._poison_index(volume_dir, "hooks/hook-config", path)
        files = volume_dir / "osprey" / "scaffold" / "files" / ".claude" / "hooks"
        files.mkdir(parents=True, exist_ok=True)
        (files / "hook_config.json").write_text('{"write_tools": []}', encoding="utf-8")

        assert restore_scaffold_bodies(pristine) == []
        assert (pristine / path).read_text(encoding="utf-8") == generated

    def test_saving_over_a_planted_record_is_refused_in_words(self, container_project, volume_dir):
        """The refusal names the file and its channel, like every other one.

        Without this the save reaches the store, fails as a write it cannot
        make durable, and the operator gets a bare server error where every
        comparable refusal explains itself.
        """
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        self._poison_index(volume_dir, "hooks/hook-config", ".claude/hooks/hook_config.json")
        svc = ScaffoldGalleryService(container_project)

        with pytest.raises(ScaffoldClaimError, match="generated, not authored"):
            svc.save_override("hooks/hook-config", '{"write_tools": []}')

    def test_a_generated_path_reaching_save_is_refused_by_name(self, container_project):
        """The guard on save itself, independent of ownership filtering."""
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        svc = ScaffoldGalleryService(container_project)
        svc._user_owned = [*svc._user_owned, "hooks/hook-config"]

        with pytest.raises(ScaffoldClaimError, match="generated, not authored"):
            svc.save_override("hooks/hook-config", '{"write_tools": []}')

    def test_ordinary_artifacts_are_unaffected(self, container_project, volume_dir):
        """The guard names generated files, not whole channels.

        ``hooks/`` holds authored hooks as well as the generated config, and
        refusing the channel to protect one file in it would be the wrong
        trade — the operator would lose every hook to save one.
        """
        hook = container_project / ".claude" / "hooks" / "shift-check.py"
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text("#!/usr/bin/env python\nprint('ok')\n", encoding="utf-8")

        svc = ScaffoldGalleryService(container_project)
        svc.register_untracked("hooks/shift-check.py")
        svc.scaffold_override(WRITABLE_ARTIFACT)

        recorded = _store_index(volume_dir)["artifacts"]
        assert recorded["hooks/shift-check.py"]["state"] == "claimed"
        assert recorded[WRITABLE_ARTIFACT]["state"] == "claimed"


class TestBodilessClaimsAreRefused:
    """A record is only written once the body it is a claim of is safely down."""

    def test_a_failed_body_write_records_nothing(self, container_project, volume_dir, monkeypatch):
        """A full volume must not leave a claim on the framework's own text."""
        from osprey.interfaces.web_terminal import ownership as ownership_mod

        monkeypatch.setattr(
            ownership_mod.OwnershipStore, "write_content", lambda self, path, content: False
        )

        svc = ScaffoldGalleryService(container_project)
        with pytest.raises(ownership_mod.OwnershipStoreError):
            svc.scaffold_override(WRITABLE_ARTIFACT)

        assert WRITABLE_ARTIFACT not in _store_index(volume_dir).get("artifacts", {})

    def test_a_body_is_replaced_atomically(self, container_project, volume_dir):
        """No reader ever sees half a body, so a restore cannot install one."""
        svc = ScaffoldGalleryService(container_project)
        svc.scaffold_override(WRITABLE_ARTIFACT)
        store = svc._store

        store.write_content(".claude/agents/channel-finder.md", "# One\n")
        store.write_content(".claude/agents/channel-finder.md", "# Two, longer than the first\n")

        assert (
            store.read_content(".claude/agents/channel-finder.md")
            == "# Two, longer than the first\n"
        )
        leftovers = list(
            (volume_dir / "osprey" / "scaffold" / "files" / ".claude" / "agents").glob(".*tmp")
        )
        assert leftovers == [], f"atomic write left staging files behind: {leftovers}"


class TestTreeAndStoreDisagree:
    """Which copy the gallery shows when the two surfaces differ.

    ``rehydrate`` already decides this for writes — it leaves a tree copy alone
    unless it is provably still the image's. The read path has to answer the
    same way, or an edit it declined to overwrite is one the operator cannot
    see and will destroy with their next save.
    """

    def test_an_edit_made_outside_the_gallery_is_what_the_gallery_shows(
        self, container_project, volume_dir
    ):
        svc = ScaffoldGalleryService(container_project)
        svc.scaffold_override(WRITABLE_ARTIFACT)
        svc.save_override(WRITABLE_ARTIFACT, "# Saved through the gallery\n")

        art = svc._registry.get(WRITABLE_ARTIFACT)
        edited_in_terminal = "# Edited in the terminal, not the gallery\n"
        (container_project / art.output_path).write_text(edited_in_terminal, encoding="utf-8")

        assert restore_scaffold_bodies(container_project) == []
        shown = ScaffoldGalleryService(container_project).get_content(WRITABLE_ARTIFACT)
        assert shown["content"] == edited_in_terminal, (
            "the gallery must show the copy the agent is reading, not the older durable one"
        )

    def test_saving_what_the_gallery_showed_does_not_revert_that_edit(
        self, container_project, volume_dir
    ):
        """The consequence of getting the read wrong, asserted directly."""
        svc = ScaffoldGalleryService(container_project)
        svc.scaffold_override(WRITABLE_ARTIFACT)
        svc.save_override(WRITABLE_ARTIFACT, "# Saved through the gallery\n")

        art = svc._registry.get(WRITABLE_ARTIFACT)
        edited_in_terminal = "# Edited in the terminal, not the gallery\n"
        (container_project / art.output_path).write_text(edited_in_terminal, encoding="utf-8")

        # Open the editor, change nothing, save — the round trip a UI makes easy.
        reopened = ScaffoldGalleryService(container_project)
        reopened.save_override(
            WRITABLE_ARTIFACT, reopened.get_content(WRITABLE_ARTIFACT)["content"]
        )

        assert (container_project / art.output_path).read_text(
            encoding="utf-8"
        ) == edited_in_terminal

    def test_an_untouched_image_copy_still_loses_to_the_durable_one(
        self, container_project, volume_dir, tmp_path
    ):
        """The recreation case must keep working: image-fresh tree, older claim."""
        pristine = _recreate_container(container_project, tmp_path / "image-rebuild")

        svc = ScaffoldGalleryService(container_project)
        svc.scaffold_override(WRITABLE_ARTIFACT)
        edited = "# The operator's version\n"
        svc.save_override(WRITABLE_ARTIFACT, edited)

        # Read before the restore runs: the tree is still the image's.
        assert ScaffoldGalleryService(pristine).get_content(WRITABLE_ARTIFACT)["content"] == edited


class TestDirectoryArtifactsHaveNoBody:
    """A skill or a service is a directory; saving text to one is a bad request."""

    def test_saving_a_profile_owned_directory_is_refused(self, project_dir):
        """Reachable by name even though the gallery does not list it.

        A profile-held skill is in ``_user_owned`` — that is what makes it show
        as owned — so ``save_override`` accepts the name and used to try to
        write text onto the directory itself, which is an unhandled OSError
        rather than an answer.
        """
        manifest = json.loads((project_dir / ".osprey-manifest.json").read_text(encoding="utf-8"))
        profile_root = Path(manifest["build_args"]["profile_path_abs"]).parent
        skill = profile_root / "skills" / "orbit-check"
        skill.mkdir(parents=True, exist_ok=True)
        (skill / "SKILL.md").write_text(
            "---\nname: orbit-check\ndescription: Check the orbit\n---\n", encoding="utf-8"
        )

        svc = ScaffoldGalleryService(project_dir)
        assert "skills/orbit-check" in svc._user_owned, (
            "a profile-held skill is owned, which is exactly why the name reaches save"
        )
        with pytest.raises(ValueError, match="whole directory"):
            svc.save_override("skills/orbit-check", "# not a directory\n")


class TestUnclaimIsHonestAboutTheProfile:
    """Releasing an artifact the profile supplies does not release it.

    Ownership of such an artifact is derived: the build's convention scan
    re-registers it from the profile every time. Dropping the project's record
    changes nothing that survives the next build, so reporting "removed" is
    simply false — and it is false in the direction that matters, because the
    operator walks away believing they gave something up.
    """

    def _profile_root(self, project_dir: Path) -> Path:
        manifest = json.loads((project_dir / ".osprey-manifest.json").read_text(encoding="utf-8"))
        return Path(manifest["build_args"]["profile_path_abs"]).parent

    def _claimed(self, project_dir: Path) -> str:
        """Claim WRITABLE_ARTIFACT into the profile and return its name."""
        ScaffoldGalleryService(project_dir).scaffold_override(WRITABLE_ARTIFACT)
        return WRITABLE_ARTIFACT

    def test_unclaim_says_the_profile_still_supplies_it(self, project_dir):
        name = self._claimed(project_dir)

        outcome = ScaffoldGalleryService(project_dir).unoverride(name)

        assert outcome["status"] == "still-supplied-by-profile"
        assert "still supplies it" in outcome["message"]
        assert "Remove it from the profile" in outcome["message"]

    def test_unclaim_changes_nothing(self, project_dir):
        """Not the profile copy, not the tree, not config.yml."""
        name = self._claimed(project_dir)
        slot = self._profile_root(project_dir) / "agents" / "channel-finder.md"
        before = slot.read_text(encoding="utf-8")
        config_before = _get_user_owned(project_dir)

        ScaffoldGalleryService(project_dir).unoverride(name, delete_file=True)

        assert slot.read_text(encoding="utf-8") == before
        assert _get_user_owned(project_dir) == config_before

    def test_the_artifact_is_still_owned_afterwards(self, project_dir):
        """Honesty in the other direction: the gallery must not hide it either."""
        name = self._claimed(project_dir)

        ScaffoldGalleryService(project_dir).unoverride(name)

        svc = ScaffoldGalleryService(project_dir)
        assert name in svc._user_owned
        listed = {a["name"]: a for a in svc.list_artifacts()}
        assert listed[name]["status"] == "user-owned"

    def test_the_message_is_the_cli_s_own_words(self, project_dir):
        """One sentence, two surfaces — so they cannot drift apart."""
        from osprey.cli.scaffold_cmd import still_supplied_by_profile_message

        name = self._claimed(project_dir)
        slot = self._profile_root(project_dir) / "agents" / "channel-finder.md"

        outcome = ScaffoldGalleryService(project_dir).unoverride(name)

        assert outcome["message"] == still_supplied_by_profile_message(str(slot))

    def test_an_artifact_the_profile_does_not_supply_still_releases(
        self, container_project, volume_dir
    ):
        """The honest path must not swallow the ordinary one."""
        svc = ScaffoldGalleryService(container_project)
        svc.scaffold_override(WRITABLE_ARTIFACT)

        outcome = ScaffoldGalleryService(container_project).unoverride(WRITABLE_ARTIFACT)

        assert outcome["status"] == "removed"
        assert WRITABLE_ARTIFACT not in ScaffoldGalleryService(container_project)._user_owned


# ===========================================================================
# Route-level refusals
# ===========================================================================


class TestRouteRefusals:
    """A refused write has to reach the operator as a reason, not a 500.

    The service raises in its own vocabulary; the route family — its own
    ``except`` clauses plus the app-level conflict handlers — decides what the
    browser sees. An exception nobody names becomes a bare 500 with the message
    stripped, which is the one outcome that helps nobody.
    """

    def _client(self, project_dir: Path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from osprey.interfaces.web_terminal.app import register_scaffold_conflict_handlers
        from osprey.interfaces.web_terminal.routes import scaffold as scaffold_routes

        app = FastAPI()
        app.include_router(scaffold_routes.router)
        register_scaffold_conflict_handlers(app)
        app.state.project_cwd = str(project_dir)
        return TestClient(app)

    def test_a_store_that_will_not_take_the_write_is_a_409(
        self, container_project, volume_dir, monkeypatch
    ):
        """A full or read-only volume, reported rather than swallowed."""
        from osprey.interfaces.web_terminal import ownership as ownership_mod

        ScaffoldGalleryService(container_project).scaffold_override(WRITABLE_ARTIFACT)
        monkeypatch.setattr(
            ownership_mod.OwnershipStore, "write_content", lambda self, path, content: False
        )

        response = self._client(container_project).put(
            f"/api/scaffold/{WRITABLE_ARTIFACT}/override", json={"content": "# edited\n"}
        )

        assert response.status_code == 409, response.text
        assert "ownership store" in response.json()["detail"]

    def test_claiming_a_generated_file_is_a_409_with_the_reason(self, container_project):
        response = self._client(container_project).post("/api/scaffold/hooks/hook-config/claim")

        assert response.status_code == 409, response.text
        assert "generated, not authored" in response.json()["detail"]

    def test_unclaiming_a_profile_held_artifact_is_a_409_not_a_success(self, project_dir):
        """The gallery's error banner is where the operator would see 'done'."""
        ScaffoldGalleryService(project_dir).scaffold_override(WRITABLE_ARTIFACT)

        response = self._client(project_dir).delete(
            f"/api/scaffold/{WRITABLE_ARTIFACT}/override?delete_file=true"
        )

        assert response.status_code == 409, response.text
        assert "still supplies it" in response.json()["detail"]

    def test_creating_a_traversing_name_is_refused(self, container_project):
        response = self._client(container_project).post(
            "/api/scaffold/create", json={"category": "rules", "name": "../../evil"}
        )

        assert response.status_code in (400, 409), response.text
        assert not (container_project.parent / "evil.md").exists()


# ===========================================================================
# Create / claim / unclaim — the protected set
# ===========================================================================


class TestCreateClaimUnoverrideProtectedSet:
    """The gallery's remaining three doors onto disk, closed on the protected set.

    ``save_override`` and ``delete_untracked`` were gated first, which left the
    three ways to reach a reserved path that do not go through either: creating
    a new artifact in a reserved subtree, claiming an existing one, and
    unclaiming one with ``delete_file=true``. All three write or delete, so all
    three ask the same question — and the answer has to be the same phrase
    naming the same channel, or an operator who is refused twice learns two
    different stories about who owns their rules directory.

    ``.claude/rules/**`` and ``.claude/skills/**`` are reserved by SHAPE rather
    than by name, which is what made the claim path worth closing separately:
    the CLI refusal the profile modes inherit covers the exactly-reserved paths
    only, so a whole reserved subtree was claimable on every mode.
    """

    #: A reserved subtree entry that does not exist yet, so a refusal cannot be
    #: mistaken for "something was already there".
    NEW_RULE = ("rules", "shift-handover")

    # ── create ───────────────────────────────────────────────────────

    def test_create_artifact_refuses_a_rule_and_names_the_channel(
        self, service, project_dir, audit_zone
    ):
        """A rule is instruction text — creating one is authoring it."""
        category, name = self.NEW_RULE

        with pytest.raises(ProtectedArtifactError) as exc:
            service.create_artifact(category, name, "# Written by the agent\n")

        message = str(exc.value)
        assert "`rules/` convention directory" in message, message
        assert "NOTHING WAS CREATED" in message
        assert not (project_dir / ".claude" / "rules" / f"{name}.md").exists()
        assert not (_profile_root(project_dir) / "rules" / f"{name}.md").exists()

        records = _protected_records(audit_zone)
        assert len(records) == 1
        assert records[0]["surface"] == "scaffold_gallery"
        assert records[0]["subject"] == f".claude/rules/{name}.md"
        assert records[0]["reason"] == "reserved path"

    def test_create_artifact_refuses_a_skill_before_the_directory_is_made(
        self, service, project_dir, audit_zone
    ):
        """The directory-shaped category, refused on its resolved entry file.

        A skill is the one category the gallery creates as a directory, so the
        path the guard judges is the ``SKILL.md`` inside it — and a refusal has
        to leave neither the directory nor a flat file behind.
        """
        with pytest.raises(ProtectedArtifactError) as exc:
            service.create_artifact("skills", "orbit-check")

        message = str(exc.value)
        assert "`skills/` convention directory" in message, message
        assert "NOTHING WAS CREATED" in message

        slot = _profile_root(project_dir) / "skills" / "orbit-check"
        assert not slot.exists()
        assert not slot.with_suffix(".md").exists()
        assert not (project_dir / ".claude" / "skills" / "orbit-check").exists()

        records = _protected_records(audit_zone)
        assert len(records) == 1
        assert records[0]["subject"] == ".claude/skills/orbit-check/SKILL.md"

    def test_create_artifact_refuses_an_osprey_hook_but_not_its_neighbours(
        self, service, project_dir, audit_zone
    ):
        """The guard reads the resolved path, not the category.

        ``hooks/`` is a claimable channel that happens to contain the
        write-safety layer, so the refusal has to land on the ``osprey_``
        prefix alone — refusing the channel would cost the operator every hook
        they are entitled to write.
        """
        # A name the render does NOT already carry, so the guard is the only
        # thing that can refuse it: an existing ``osprey_`` hook would be
        # refused as a file that is already there, and the test would pass
        # without the guard ever running.
        with pytest.raises(ProtectedArtifactError) as exc:
            service.create_artifact("hooks", "osprey_shift_check")

        assert "write-safety layer" in str(exc.value)
        assert "NOTHING WAS CREATED" in str(exc.value)
        assert not (_profile_root(project_dir) / "hooks" / "osprey_shift_check.py").exists()
        assert len(_protected_records(audit_zone)) == 1

        ScaffoldGalleryService(project_dir).create_artifact("hooks", "shift-check")
        assert (_profile_root(project_dir) / "hooks" / "shift-check.py").is_file()
        assert len(_protected_records(audit_zone)) == 1, "an allowed create records nothing"

    def test_create_artifact_of_an_agent_still_goes_through(self, service, project_dir, audit_zone):
        """The ordinary case, unchanged: an agent is the operator's to author."""
        result = service.create_artifact("agents", "shift-handover", "# Shift handover\n")

        assert result["status"] == "created"
        assert result["output_path"] == ".claude/agents/shift-handover.md"
        assert (_profile_root(project_dir) / "agents" / "shift-handover.md").is_file()
        assert _protected_records(audit_zone) == []

    def test_create_artifact_refuses_before_the_volume_is_written(
        self, container_project, volume_dir, audit_zone
    ):
        """The container's durable surface must not take the create either.

        In a deployed container the volume is what outlives the container, so a
        creation recorded there would put an agent-authored rule back into the
        project tree at every restart — the refusal would have delayed the
        write rather than prevented it.
        """
        category, name = self.NEW_RULE
        svc = ScaffoldGalleryService(container_project)

        with pytest.raises(ProtectedArtifactError, match="NOTHING WAS CREATED"):
            svc.create_artifact(category, name, "# Written by the agent\n")

        assert _store_index(volume_dir) == {}
        assert not (container_project / ".claude" / "rules" / f"{name}.md").exists()
        assert len(_protected_records(audit_zone)) == 1

    def test_create_artifact_route_refuses_with_403_and_records_activity(
        self, project_dir, audit_zone
    ):
        """The POST route maps the refusal to 403 and publishes it to the ring."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from osprey.interfaces.web_terminal.routes import scaffold as scaffold_routes

        app = FastAPI()
        app.include_router(scaffold_routes.router)
        app.state.project_cwd = str(project_dir)
        app.state.agent_activity_ring = []

        category, name = self.NEW_RULE
        response = TestClient(app).post(
            "/api/scaffold/create",
            json={"category": category, "name": name, "content": "# Written by the agent\n"},
        )

        assert response.status_code == 403, response.text
        detail = response.json()["detail"]
        assert "`rules/` convention directory" in detail
        assert "NOTHING WAS CREATED" in detail
        assert not (_profile_root(project_dir) / "rules" / f"{name}.md").exists()

        assert [event["tool"] for event in app.state.agent_activity_ring] == ["create_artifact"]
        recorded = app.state.agent_activity_ring[0]["target"]
        assert recorded["kind"] == "artifact"
        assert f"rules/{name}" in recorded["detail"]

    # ── claim ────────────────────────────────────────────────────────

    def test_claim_of_a_reserved_rule_is_refused_in_profile_mode(
        self, service, project_dir, audit_zone
    ):
        """A claim moves the artifact into the profile — that is a write."""
        on_disk = project_dir / ".claude" / "rules" / "safety.md"
        before = on_disk.read_text(encoding="utf-8")

        with pytest.raises(ProtectedArtifactError) as exc:
            service.scaffold_override(SAFE_ARTIFACT)

        message = str(exc.value)
        assert "`rules/` convention directory" in message, message
        assert "NOTHING WAS CLAIMED" in message
        assert on_disk.read_text(encoding="utf-8") == before, "the claim moves the file — or would"
        assert not (_profile_root(project_dir) / "rules" / "safety.md").exists()
        assert SAFE_ARTIFACT not in ScaffoldGalleryService(project_dir)._user_owned

        records = _protected_records(audit_zone)
        assert len(records) == 1
        assert "target=.claude/rules/safety.md" in records[0]["detail"]

    def test_claim_of_a_pattern_reserved_skill_is_refused(self, service, project_dir, audit_zone):
        """The subtree the exact table never covered, and the review finding.

        ``.claude/skills/**`` is reserved by shape, and the CLI refusal the
        profile modes inherit reads the exact table only — so before this gate
        a skill was claimable on every mode, and the gallery said so.
        """
        with pytest.raises(ProtectedArtifactError) as exc:
            service.scaffold_override("skills/diagnose")

        assert "`skills/` convention directory" in str(exc.value)
        assert "NOTHING WAS CLAIMED" in str(exc.value)
        assert not (_profile_root(project_dir) / "skills" / "diagnose").exists()
        assert len(_protected_records(audit_zone)) == 1

    def test_claim_of_a_reserved_rule_is_refused_on_the_volume(
        self, container_project, volume_dir, audit_zone
    ):
        """The mode that records ownership itself, where no CLI is in the way."""
        svc = ScaffoldGalleryService(container_project)

        with pytest.raises(ProtectedArtifactError, match="NOTHING WAS CLAIMED"):
            svc.scaffold_override(SAFE_ARTIFACT)

        assert _store_index(volume_dir) == {}
        assert not (volume_dir / "osprey" / "scaffold" / "files").exists()
        assert len(_protected_records(audit_zone)) == 1

    def test_claim_of_a_generated_path_keeps_the_refusal_it_already_had(
        self, container_project, audit_zone
    ):
        """One path, one answer — and for these paths it is the older one.

        ``hook_config.json`` is generated as well as reserved, and "it is
        generated, change what generates it" is the more useful of the two
        answers: it names the way in. The route renders it as a 409, which is
        the contract the gallery's error banner was written against, so the
        protected-set gate deliberately does not preempt it.
        """
        from osprey.cli.scaffold_cmd import ScaffoldClaimError

        svc = ScaffoldGalleryService(container_project)
        with pytest.raises(ScaffoldClaimError, match="generated, not authored"):
            svc.scaffold_override("hooks/hook-config")

        assert _protected_records(audit_zone) == [], "the older refusal audits nothing"

    def test_claim_of_an_ordinary_agent_still_goes_through(self, service, project_dir, audit_zone):
        """The ordinary case, unchanged."""
        result = service.scaffold_override(WRITABLE_ARTIFACT)

        assert result["status"] == "claimed"
        assert (_profile_root(project_dir) / "agents" / "channel-finder.md").is_file()
        assert _protected_records(audit_zone) == []

    def test_claim_route_refuses_a_reserved_skill_with_403_and_records_activity(
        self, project_dir, audit_zone
    ):
        """A pattern-reserved claim is a 403 naming the channel, not a 500.

        The app-level handler turns the CLI's refusal into a 409, but it only
        ever sees the exactly-reserved paths; a refusal on a reserved SUBTREE
        would have reached the browser as a bare 500 with the channel stripped
        out of it.
        """
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from osprey.interfaces.web_terminal.app import register_scaffold_conflict_handlers
        from osprey.interfaces.web_terminal.routes import scaffold as scaffold_routes

        app = FastAPI()
        app.include_router(scaffold_routes.router)
        register_scaffold_conflict_handlers(app)
        app.state.project_cwd = str(project_dir)
        app.state.agent_activity_ring = []

        response = TestClient(app).post("/api/scaffold/skills/diagnose/claim")

        assert response.status_code == 403, response.text
        detail = response.json()["detail"]
        assert "`skills/` convention directory" in detail
        assert "NOTHING WAS CLAIMED" in detail
        assert not (_profile_root(project_dir) / "skills" / "diagnose").exists()

        assert [event["tool"] for event in app.state.agent_activity_ring] == ["claim"]
        assert "skills/diagnose" in app.state.agent_activity_ring[0]["target"]["detail"]

    # ── unclaim with delete ──────────────────────────────────────────

    def _owned_reserved_file(self, project_dir: Path) -> Path:
        """A reserved artifact this project owns, with a body on disk.

        Ownership an image carries is exactly this: a name in ``config.yml``
        with a file beside it. It is planted rather than claimed or registered,
        because both of those doors are shut now — and an artifact that came
        through one of them before the gate existed is precisely the one an
        operator still has in front of them.
        """
        planted = project_dir / ".claude" / "rules" / "planted.md"
        planted.parent.mkdir(parents=True, exist_ok=True)
        planted.write_text("# Planted\n", encoding="utf-8")
        _add_user_owned(project_dir, "rules/planted")
        return planted

    def test_unoverride_with_delete_refuses_a_reserved_artifact(
        self, detached_project_dir, audit_zone
    ):
        """``delete_file=true`` is a delete, and this one is not the gallery's."""
        planted = self._owned_reserved_file(detached_project_dir)
        svc = ScaffoldGalleryService(detached_project_dir)

        with pytest.raises(ProtectedArtifactError) as exc:
            svc.unoverride("rules/planted", delete_file=True)

        message = str(exc.value)
        assert "`rules/` convention directory" in message, message
        assert "NOTHING WAS DELETED" in message
        assert planted.read_text(encoding="utf-8") == "# Planted\n"
        assert "rules/planted" in _get_user_owned(detached_project_dir), (
            "the release must not have happened either — the refusal says nothing did"
        )

        records = _protected_records(audit_zone)
        assert len(records) == 1
        assert "target=.claude/rules/planted.md" in records[0]["detail"]

    def test_unoverride_without_delete_still_releases_a_reserved_artifact(
        self, detached_project_dir, audit_zone
    ):
        """Giving up ownership touches no file, so it stays open.

        The protected set is about who writes the bytes. Releasing without
        deleting leaves the file exactly as the owning channel would want it
        and hands management back to the build — refusing that would trap the
        operator in an ownership they cannot act on.
        """
        planted = self._owned_reserved_file(detached_project_dir)
        svc = ScaffoldGalleryService(detached_project_dir)

        outcome = svc.unoverride("rules/planted", delete_file=False)

        assert outcome["status"] == "removed"
        assert outcome["deleted_file"] is False
        assert planted.exists()
        assert "rules/planted" not in _get_user_owned(detached_project_dir)
        assert _protected_records(audit_zone) == []

    def test_unoverride_with_delete_still_removes_an_ordinary_artifact(
        self, detached_project_dir, detached_service, audit_zone
    ):
        """The ordinary case, unchanged."""
        orphan = detached_project_dir / ".claude" / "agents" / "removable.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Removable\n", encoding="utf-8")
        detached_service.register_untracked("agents/removable")

        svc = ScaffoldGalleryService(detached_project_dir)
        outcome = svc.unoverride("agents/removable", delete_file=True)

        assert outcome["deleted_file"] is True
        assert not orphan.exists()
        assert _protected_records(audit_zone) == []

    def test_unoverride_route_refuses_with_403_and_records_activity(
        self, detached_project_dir, audit_zone
    ):
        """The DELETE route had no clause for this: the refusal was a 500."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from osprey.interfaces.web_terminal.routes import scaffold as scaffold_routes

        planted = self._owned_reserved_file(detached_project_dir)

        app = FastAPI()
        app.include_router(scaffold_routes.router)
        app.state.project_cwd = str(detached_project_dir)
        app.state.agent_activity_ring = []

        response = TestClient(app).delete("/api/scaffold/rules/planted/override?delete_file=true")

        assert response.status_code == 403, response.text
        detail = response.json()["detail"]
        assert "`rules/` convention directory" in detail
        assert "NOTHING WAS DELETED" in detail
        assert planted.exists()

        assert [event["tool"] for event in app.state.agent_activity_ring] == ["unoverride"]
        assert "rules/planted" in app.state.agent_activity_ring[0]["target"]["detail"]


class TestLinkedWritesAreJudgedOnTheResolvedFile:
    """A name is not a file: the guard follows the link before it answers.

    Every writer in this service reaches disk through the filesystem, which
    follows symlinks — a write opens the target, a delete removes what the link
    points at. So a link planted at an unprotected name (``.claude/agents/x.md``
    -> ``../rules/safety.md``) is lexically an agent, physically a rule, and
    passes both of the questions the guard used to ask: the name carries no
    ``..``, and the resolved path is still inside the project.

    The gallery has no way to create a symlink, which is exactly why this is
    worth pinning: the link comes from somewhere else — a mounted volume, an
    operator, an agent with a shell — and the guard is the layer that has to
    hold when it is already there. All six call sites inherit the answer,
    because they all ask this one method.
    """

    RESERVED_TARGET = Path(".claude") / "rules" / "safety.md"

    def _link(self, project_dir: Path, name: str, target: str) -> Path:
        """Plant ``.claude/agents/<name>.md`` as a link to *target*, and own it."""
        link = project_dir / ".claude" / "agents" / f"{name}.md"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(Path(target))
        _add_user_owned(project_dir, f"agents/{name}")
        return link

    def test_create_artifact_refuses_a_link_that_lands_on_a_reserved_file(
        self, service, project_dir, audit_zone
    ):
        """Creating "an agent" that is really the safety rule is still a rule."""
        reserved = project_dir / self.RESERVED_TARGET
        before = reserved.read_text(encoding="utf-8")
        link = project_dir / ".claude" / "agents" / "linked.md"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(Path("..") / "rules" / "safety.md")

        with pytest.raises(ProtectedArtifactError) as exc:
            service.create_artifact("agents", "linked", "# Written by the agent\n")

        message = str(exc.value)
        assert "`rules/` convention directory" in message, message
        assert "NOTHING WAS CREATED" in message
        assert reserved.read_text(encoding="utf-8") == before

        records = _protected_records(audit_zone)
        assert len(records) == 1
        assert records[0]["subject"] == ".claude/agents/linked.md", (
            "the audit records the path the caller named — that is the one an "
            "operator is looking for when they go asking what happened"
        )

    def test_save_override_refuses_a_link_that_lands_on_a_reserved_file(
        self, detached_project_dir, audit_zone
    ):
        """The write path opens the target, so the target is what is judged."""
        reserved = detached_project_dir / self.RESERVED_TARGET
        before = reserved.read_text(encoding="utf-8")
        self._link(detached_project_dir, "planted-link", "../rules/safety.md")

        svc = ScaffoldGalleryService(detached_project_dir)
        with pytest.raises(ProtectedArtifactError) as exc:
            svc.save_override("agents/planted-link", "# Rewritten by the agent\n")

        assert "`rules/` convention directory" in str(exc.value)
        assert "NOTHING WAS WRITTEN" in str(exc.value)
        assert reserved.read_text(encoding="utf-8") == before
        assert len(_protected_records(audit_zone)) == 1

    def test_save_override_still_writes_through_a_link_to_an_ordinary_file(
        self, detached_project_dir, audit_zone
    ):
        """A link is not itself suspicious — only where it lands.

        Refusing every symlinked artifact would be the easy guard and the wrong
        one: an operator who keeps their agents in a directory of their own and
        links them into the render is doing nothing the protected set is about.
        """
        ordinary = detached_project_dir / ".claude" / "agents" / "ordinary.md"
        ordinary.parent.mkdir(parents=True, exist_ok=True)
        ordinary.write_text("# Original\n", encoding="utf-8")
        self._link(detached_project_dir, "linked-ok", "ordinary.md")

        svc = ScaffoldGalleryService(detached_project_dir)
        assert svc.save_override("agents/linked-ok", "# Edited\n")["status"] == "saved"

        assert ordinary.read_text(encoding="utf-8") == "# Edited\n"
        assert _protected_records(audit_zone) == []

    def test_unoverride_with_delete_refuses_a_link_onto_a_reserved_file(
        self, detached_project_dir, audit_zone
    ):
        """``unlink`` on a link removes the link — but the rule is the target.

        The gallery's delete is the one operation where following the link and
        not following it differ, and neither answer is safe to guess at: the
        artifact the operator is releasing IS the reserved file as far as every
        reader of the render is concerned.
        """
        reserved = detached_project_dir / self.RESERVED_TARGET
        link = self._link(detached_project_dir, "linked-del", "../rules/safety.md")

        svc = ScaffoldGalleryService(detached_project_dir)
        with pytest.raises(ProtectedArtifactError) as exc:
            svc.unoverride("agents/linked-del", delete_file=True)

        assert "`rules/` convention directory" in str(exc.value)
        assert "NOTHING WAS DELETED" in str(exc.value)
        assert link.is_symlink()
        assert reserved.is_file()
        assert "agents/linked-del" in _get_user_owned(detached_project_dir)
        assert len(_protected_records(audit_zone)) == 1

    def test_a_link_onto_a_reserved_file_is_shown_read_only(self, detached_project_dir):
        """The badge inherits the same answer, because it asks the same method.

        Not a separate rule — ``_is_read_only`` is a rename of the write gate's
        question — but worth pinning: a card that offered an edit the save then
        refused would be the gallery lying about the one thing it is showing.
        """
        self._link(detached_project_dir, "planted-link", "../rules/safety.md")
        self._link(detached_project_dir, "linked-ok", "channel-finder.md")

        listed = {
            a["name"]: a for a in ScaffoldGalleryService(detached_project_dir).list_artifacts()
        }

        assert listed["agents/planted-link"]["read_only"] is True
        assert listed["agents/linked-ok"]["read_only"] is False


class TestWritesThatLeaveTheProjectAreRefused:
    """Following the link is only half the answer — it can land nowhere judgeable.

    :func:`~...ownership.reserved_write_channel` resolves the path before it
    answers, and the resolved path has two ways of being unusable. It can land
    OUTSIDE the project, where no protected pattern applies because the pattern
    table is written in project-relative terms and there is nothing left to be
    relative to: the operator's ``~/.ssh/authorized_keys`` is not
    ``.claude/rules/**`` by any reading, and a guard that only asked "is the
    resolved path protected?" would wave it through. Or it can fail to resolve
    at all — a link cycle, a path segment the process cannot traverse — and a
    path this function cannot judge must not read as writable.

    Both answer with :data:`NOT_PROJECT_RELATIVE_CHANNEL`, and both are the
    LAST gate: ``_write_body`` opens ``project_dir / output_path`` and
    ``delete_untracked`` unlinks it, each with no resolve check of its own. The
    filesystem follows the link on the way in, so whatever this function
    declines is exactly what those two would otherwise have written or removed.

    The restore path has its own escape tests, but they never reach this
    branch: ``_safe_relative`` rejects a store record before the resolve
    happens. The gallery has no such pre-filter — the link is planted at an
    ordinary, ownable, unreserved name — so these are the tests that execute
    the guard.
    """

    def _outside(self, tmp_path: Path) -> Path:
        """A file that is emphatically not in the project."""
        outside = tmp_path / "outside-the-project" / "authorized_keys"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_text("ssh-ed25519 AAAA-the-operators-own-key\n", encoding="utf-8")
        return outside

    def _escaping_link(self, project_dir: Path, name: str, target: Path) -> Path:
        """Plant ``.claude/agents/<name>.md`` as a link out of the project, and own it."""
        link = project_dir / ".claude" / "agents" / f"{name}.md"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
        _add_user_owned(project_dir, f"agents/{name}")
        return link

    def test_save_override_refuses_a_link_that_lands_outside_the_project(
        self, detached_project_dir, tmp_path, audit_zone
    ):
        """The save opens the target, and the target is not ours to open."""
        outside = self._outside(tmp_path)
        before = outside.read_bytes()
        self._escaping_link(detached_project_dir, "escaping-save", outside)

        svc = ScaffoldGalleryService(detached_project_dir)
        with pytest.raises(ProtectedArtifactError) as exc:
            svc.save_override("agents/escaping-save", "# Written by the agent\n")

        assert exc.value.channel == NOT_PROJECT_RELATIVE_CHANNEL
        assert "NOTHING WAS WRITTEN" in str(exc.value)
        assert outside.read_bytes() == before, "the write would have gone through the link"

        records = _protected_records(audit_zone)
        assert len(records) == 1
        assert records[0]["subject"] == ".claude/agents/escaping-save.md"
        assert NOT_PROJECT_RELATIVE_CHANNEL in records[0]["detail"]

    def test_unoverride_with_delete_refuses_a_link_that_lands_outside_the_project(
        self, detached_project_dir, tmp_path, audit_zone
    ):
        """``unlink`` through a link removes the file at the far end of it."""
        outside = self._outside(tmp_path)
        link = self._escaping_link(detached_project_dir, "escaping-del", outside)

        svc = ScaffoldGalleryService(detached_project_dir)
        with pytest.raises(ProtectedArtifactError) as exc:
            svc.unoverride("agents/escaping-del", delete_file=True)

        assert exc.value.channel == NOT_PROJECT_RELATIVE_CHANNEL
        assert "NOTHING WAS DELETED" in str(exc.value)
        assert outside.exists(), "the delete would have removed a file outside the project"
        assert link.is_symlink()
        assert "agents/escaping-del" in _get_user_owned(detached_project_dir), (
            "the refusal is raised before the release, so ownership is untouched"
        )

        records = _protected_records(audit_zone)
        assert len(records) == 1
        assert NOT_PROJECT_RELATIVE_CHANNEL in records[0]["detail"]

    def test_delete_untracked_refuses_a_link_that_lands_outside_the_project(
        self, detached_project_dir, tmp_path, audit_zone
    ):
        """The orphan sweep's delete reaches disk through the same one question."""
        outside = self._outside(tmp_path)
        link = detached_project_dir / ".claude" / "agents" / "escaping-orphan.md"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(outside)

        svc = ScaffoldGalleryService(detached_project_dir)
        with pytest.raises(ProtectedArtifactError) as exc:
            svc.delete_untracked("agents/escaping-orphan")

        assert exc.value.channel == NOT_PROJECT_RELATIVE_CHANNEL
        assert outside.exists()
        assert link.is_symlink()
        assert len(_protected_records(audit_zone)) == 1

    def test_a_link_out_of_the_project_is_shown_read_only(self, detached_project_dir, tmp_path):
        """The badge inherits the refusal, so the card cannot offer a dead edit."""
        self._escaping_link(detached_project_dir, "escaping-badge", self._outside(tmp_path))

        listed = {
            a["name"]: a for a in ScaffoldGalleryService(detached_project_dir).list_artifacts()
        }

        assert listed["agents/escaping-badge"]["read_only"] is True

    def test_an_unresolvable_path_answers_with_a_refusal_not_none(self, project_dir, monkeypatch):
        """ "Cannot judge" is not "writable" — the fail-closed branch, on its own.

        Driven by making ``resolve`` raise, because the real ways to provoke it
        (a link cycle, an untraversable segment) are host- and
        filesystem-dependent and one of them is not even an error on every
        platform. What is under test is the branch, and the branch is reached
        by exactly one thing: ``resolve`` raising ``OSError``.
        """
        monkeypatch.setattr(Path, "resolve", _raising_resolve(Path.resolve, "unresolvable.md"))

        channel = reserved_write_channel(project_dir, ".claude/agents/unresolvable.md")

        assert channel == NOT_PROJECT_RELATIVE_CHANNEL

    def test_save_override_refuses_a_path_it_cannot_resolve(
        self, detached_project_dir, monkeypatch, audit_zone
    ):
        """And the caller that would have written the bytes gets that refusal."""
        target = detached_project_dir / ".claude" / "agents" / "unresolvable.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Framework's own\n", encoding="utf-8")
        _add_user_owned(detached_project_dir, "agents/unresolvable")

        svc = ScaffoldGalleryService(detached_project_dir)
        monkeypatch.setattr(Path, "resolve", _raising_resolve(Path.resolve, "unresolvable.md"))

        with pytest.raises(ProtectedArtifactError) as exc:
            svc.save_override("agents/unresolvable", "# Written by the agent\n")

        assert exc.value.channel == NOT_PROJECT_RELATIVE_CHANNEL
        assert "NOTHING WAS WRITTEN" in str(exc.value)
        assert target.read_text(encoding="utf-8") == "# Framework's own\n"

        records = _protected_records(audit_zone)
        assert len(records) == 1
        assert NOT_PROJECT_RELATIVE_CHANNEL in records[0]["detail"]


class TestRestoreRefusesReservedRecords:
    """The container-start restore is a writer too, and the least supervised one.

    Its input is the ownership store on the claude-config volume — which is the
    agent's own ``$HOME`` and writable by anything running in the container —
    and its output is a file in the render zone. ``_safe_relative`` keeps it
    inside the ownable tree, but that tree deliberately contains
    ``.claude/rules/**`` and ``.claude/skills/**``: they are what the gallery
    exists to claim. So a planted record naming a reserved path was a way to
    have the restore install agent-authored instruction text — and under the
    container entrypoint it runs as ROOT into a tree the agent cannot otherwise
    write, which turns a store record into a privilege escalation.

    The gate lives in ``rehydrate``, the single walk that turns a record into a
    write, rather than in ``restore_scaffold_bodies`` around it: the entrypoint
    and the in-process web-terminal startup both reach disk through that walk,
    and a pre-filter one layer up would leave the walk itself ungated for
    whatever calls it next.

    It SKIPS rather than raises. One poisoned record must not cost the operator
    every other body they claimed, and a container that refuses to start is a
    worse outcome than one that starts with a single file as the image shipped
    it — the refusal is durable on the ``scaffold_restore`` ledger either way.
    """

    EVIL = ("rules/evil", ".claude/rules/evil.md")

    _plant_record = staticmethod(_plant_store_record)

    def test_restore_reserved_record_is_not_installed(
        self, container_project, volume_dir, audit_zone
    ):
        """The bare case: a rule on the volume stays on the volume."""
        name, output_path = self.EVIL
        self._plant_record(volume_dir, name, output_path, "# Written by the agent\n")

        assert restore_scaffold_bodies(container_project) == []
        assert not (container_project / output_path).exists()

        records = _protected_records(audit_zone)
        assert len(records) == 1
        assert records[0]["surface"] == "scaffold_restore"
        assert f"target={output_path}" in records[0]["detail"]
        assert records[0]["subject"] == output_path
        assert "`rules/` convention directory" in records[0]["detail"]

    def test_restore_reserved_skips_only_the_poisoned_record(
        self, container_project, volume_dir, audit_zone, tmp_path
    ):
        """A mixed store: the operator's own work still comes back.

        Skipping is only the right answer if it is surgical. A restore that
        gave up on the whole store when one record was bad would hand the
        poisoned record a second, larger prize — every body the operator
        actually claimed, silently not restored.
        """
        pristine = _recreate_container(container_project, tmp_path / "image-rebuild")

        svc = ScaffoldGalleryService(container_project)
        svc.scaffold_override(WRITABLE_ARTIFACT)
        edited = "# Channel finder\nEdited by the operator.\n"
        svc.save_override(WRITABLE_ARTIFACT, edited)

        name, output_path = self.EVIL
        self._plant_record(volume_dir, name, output_path, "# Written by the agent\n")

        assert restore_scaffold_bodies(pristine) == [WRITABLE_ARTIFACT]

        good = pristine / ".claude" / "agents" / "channel-finder.md"
        assert good.read_text(encoding="utf-8") == edited
        assert not (pristine / output_path).exists()
        assert len(_protected_records(audit_zone)) == 1

    def test_restore_reserved_skips_a_record_resolving_onto_a_reserved_file(
        self, container_project, volume_dir, audit_zone
    ):
        """A link in the tree makes an innocent record name a reserved file.

        The record's path is an agent by every lexical test there is. What
        decides it is where the write would land, which is the same question
        the gallery's own gates ask — one helper, so a body cannot be refused
        at the save and installed at the restart.
        """
        # The link points at a rule that does not exist yet, which is the sharp
        # version of this: with a file already there the restore's "leave a
        # non-pristine copy alone" rule would decline the write for an
        # unrelated reason. A dangling link has nothing to decline — the write
        # goes through it and CREATES the reserved file.
        link = container_project / ".claude" / "agents" / "linked.md"
        link.unlink(missing_ok=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(Path("..") / "rules" / "planted.md")

        reserved = container_project / ".claude" / "rules" / "planted.md"
        assert not reserved.exists()
        self._plant_record(
            volume_dir, "agents/linked", ".claude/agents/linked.md", "# Written by the agent\n"
        )

        assert restore_scaffold_bodies(container_project) == []
        assert not reserved.exists(), "the write would have created the rule through the link"
        assert link.is_symlink(), "the link itself is the operator's business, not ours to remove"

        records = _protected_records(audit_zone)
        assert len(records) == 1
        assert "target=.claude/agents/linked.md" in records[0]["detail"]
        assert "`rules/` convention directory" in records[0]["detail"]

    def test_restore_reserved_gate_is_on_the_entrypoint_s_own_call_path(self):
        """The root-privileged caller reaches the same gate, with no second path.

        Asserted by identity rather than by running a container: the entrypoint
        imports ``restore_scaffold_bodies`` and calls it, that function's only
        route to a write is ``rehydrate``, and ``rehydrate``'s gate is the
        shared ``reserved_write_channel``. Every link in that chain is an
        object identity here, so a future "just inline it for the entrypoint"
        breaks this test rather than the container.
        """
        from osprey.interfaces.web_terminal import ownership as ownership_mod
        from osprey.interfaces.web_terminal import scaffold_gallery_service as service_mod

        entrypoint = (
            Path(service_mod.__file__).parents[3]
            / "osprey"
            / "templates"
            / "project"
            / "entrypoint.sh"
        ).read_text(encoding="utf-8")
        assert "from osprey.interfaces.web_terminal.scaffold_gallery_service import" in entrypoint
        assert "restore_scaffold_bodies," in entrypoint
        assert "restore_scaffold_bodies(render_dir)" in entrypoint

        assert service_mod.restore_scaffold_bodies.__globals__["rehydrate"] is (
            ownership_mod.rehydrate
        ), "the entrypoint's call target must reach the gated walk, not a private copy"
        assert ownership_mod.rehydrate.__globals__["reserved_write_channel"] is (
            ownership_mod.reserved_write_channel
        ), "and that walk must ask the same question the gallery's gates ask"


class TestReservedSubtreeRootsAreClosed:
    """The directory itself is the subtree, and ``**`` does not say so.

    ``fnmatch`` needs the separator to be present: ``.claude/rules/**`` answers
    for a file inside the directory and for the trailing-slash spelling, and
    says nothing about the bare ``.claude/rules``. The bare name is the one a
    writer uses to replace, move or remove the whole convention at once — the
    largest version of the write the pattern exists to refuse — and it is the
    spelling every path that has been through ``normpath`` carries.
    """

    @pytest.mark.parametrize(
        "path",
        [".claude/rules", ".claude/rules/", ".claude/skills", ".claude/skills/", "./.claude/rules"],
    )
    def test_a_bare_reserved_directory_is_refused(self, project_dir, path):
        channel = reserved_write_channel(project_dir, path)
        assert channel is not None, f"{path!r} names a reserved subtree"
        assert "convention directory" in channel

    def test_the_bare_name_answers_with_the_subtree_s_own_channel(self, project_dir):
        """One refusal wording, whether the writer named the directory or a file."""
        assert reserved_write_channel(project_dir, ".claude/rules") == reserved_write_channel(
            project_dir, ".claude/rules/anything.md"
        )

    @pytest.mark.parametrize("path", [".claude/rulesfoo", ".claude/agents", ".claude/commands"])
    def test_a_directory_that_merely_starts_the_same_stays_open(self, project_dir, path):
        """Prefix matching on strings is how a guard swallows its neighbours."""
        assert reserved_write_channel(project_dir, path) is None


class TestScanUntrackedJudgesTheResolvedFile:
    """The list offers two actions, and both ask the resolved path.

    A lexical filter and a resolved-path gate disagree on exactly one entry: a
    link at an unreserved name onto a reserved file. That entry is listed as an
    orphan the operator can register or delete, and both buttons 403 — which is
    the outcome the filter's own comment says it exists to prevent.
    """

    def test_an_entry_linked_onto_a_reserved_file_is_not_listed(self, service, project_dir):
        link = project_dir / ".claude" / "agents" / "linked.md"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(Path("..") / "rules" / "safety.md")

        listed = {entry["canonical_name"] for entry in service.scan_untracked()}

        assert "agents/linked" not in listed, (
            "both actions this list offers would refuse it — listing it advertises a dead end"
        )

    def test_an_ordinary_orphan_is_still_listed(self, service, project_dir):
        """The filter must stay narrow: an unlinked orphan is the point of the list."""
        orphan = project_dir / ".claude" / "agents" / "orphan.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Orphan\n", encoding="utf-8")

        listed = {entry["canonical_name"] for entry in service.scan_untracked()}

        assert "agents/orphan" in listed

    def test_a_link_onto_an_ordinary_file_is_still_listed(self, service, project_dir):
        """A link is not itself the problem — only where it lands."""
        target = project_dir / ".claude" / "agents" / "target.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Target\n", encoding="utf-8")
        (project_dir / ".claude" / "agents" / "linked-ok.md").symlink_to(Path("target.md"))

        listed = {entry["canonical_name"] for entry in service.scan_untracked()}

        assert "agents/linked-ok" in listed


class TestSaveOverrideReportsAppliesOnRestart:
    """A save that only lands on the volume has to say so.

    The degrade itself is old and deliberate: in a deployed container the image
    tree can be read-only, and refusing the save outright would cost the
    operator an edit the volume could perfectly well have kept. What was
    missing is the sentence. Without it the operator gets "saved", a gallery
    showing their text, and an agent still reading the framework's — and
    nothing anywhere says the two disagree until the container restarts. A gap
    that is both survivable and invisible is the wrong combination; this makes
    it merely survivable.
    """

    RESERVED_OWNED = "rules/facility"

    @staticmethod
    def _read_only_tree(monkeypatch, target: Path) -> None:
        """Make exactly one path refuse writes, as a read-only image tree does.

        Only that path: the volume copy has to stay writable, because the whole
        claim of this response is that the body IS somewhere durable.
        """
        original = Path.write_text

        def refusing(self, *args, **kwargs):
            if self == target:
                raise PermissionError(f"Read-only file system: {self}")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", refusing)

    def test_save_override_reports_applies_on_restart_when_the_tree_refuses(
        self, container_project, volume_dir, monkeypatch
    ):
        """True, and the body really is on the volume — the claim has to be earned."""
        svc = ScaffoldGalleryService(container_project)
        svc.scaffold_override(WRITABLE_ARTIFACT)

        tree_copy = container_project / ".claude" / "agents" / "channel-finder.md"
        before = tree_copy.read_text(encoding="utf-8")
        self._read_only_tree(monkeypatch, tree_copy)

        edited = "# Channel finder\nEdited while the tree was read-only.\n"
        result = ScaffoldGalleryService(container_project).save_override(WRITABLE_ARTIFACT, edited)

        assert result["status"] == "saved"
        assert result["applies_on_restart"] is True
        assert tree_copy.read_text(encoding="utf-8") == before, "the tree did not take it"

        store = ScaffoldGalleryService(container_project)._store
        assert store is not None
        assert store.read_content(".claude/agents/channel-finder.md") == edited, (
            "'applies on restart' is a promise about the volume — it has to be true"
        )
        assert _store_index(volume_dir)["artifacts"][WRITABLE_ARTIFACT]["state"] == "claimed"

    def test_save_override_reports_applies_on_restart_false_when_the_tree_takes_it(
        self, container_project
    ):
        """The ordinary container save: both surfaces written, nothing to warn about."""
        svc = ScaffoldGalleryService(container_project)
        svc.scaffold_override(WRITABLE_ARTIFACT)

        edited = "# Channel finder\nEdited normally.\n"
        result = ScaffoldGalleryService(container_project).save_override(WRITABLE_ARTIFACT, edited)

        assert result["applies_on_restart"] is False
        tree_copy = container_project / ".claude" / "agents" / "channel-finder.md"
        assert tree_copy.read_text(encoding="utf-8") == edited

    def test_save_override_reports_applies_on_restart_false_on_the_profile_branch(
        self, service, project_dir
    ):
        """The other branch of the same method: a profile copy is live immediately."""
        service.scaffold_override(WRITABLE_ARTIFACT)

        result = ScaffoldGalleryService(project_dir).save_override(WRITABLE_ARTIFACT, "# Edited\n")

        assert result["applies_on_restart"] is False

    def test_the_reserved_gate_still_wins_before_applies_on_restart_is_reported(
        self, project_dir, audit_zone
    ):
        """Ordering: a protected artifact is a 403, never a degraded 200.

        Both answers are "your edit is not in the tree", and they must not be
        confused for one another — one says come back after a restart, the
        other says this is not yours to write at all. The refusal is raised
        before either surface is touched, so it cannot arrive as a save that
        merely applies later.
        """
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from osprey.interfaces.web_terminal.routes import scaffold as scaffold_routes

        app = FastAPI()
        app.include_router(scaffold_routes.router)
        app.state.project_cwd = str(project_dir)
        app.state.agent_activity_ring = []

        response = TestClient(app).put(
            f"/api/scaffold/{self.RESERVED_OWNED}/override",
            json={"content": "# Rewritten by the agent\n"},
        )

        assert response.status_code == 403, response.text
        assert "applies_on_restart" not in response.text
        assert "NOTHING WAS WRITTEN" in response.json()["detail"]
        assert len(_protected_records(audit_zone)) == 1


# ===========================================================================
# The container render: a profile that exists but is not a source of truth
# ===========================================================================


class TestContainerRenderIgnoresTheBakedProfile:
    """The topology every other fixture in this file could not express.

    ``container_project`` above simulates a container by DELETING the profile,
    and that is not what an image contains. ``osprey build`` renders a container
    project whose manifest names ``/app/<project>/profile.yml`` — a file that is
    really there, root-owned, and copied in from the build. So "can I write the
    profile root?" stopped being a test of whether the profile is the source of
    truth and became a test of WHO IS ASKING:

    * the web app runs as uid 1000, cannot write the root-owned tree, and falls
      through to the volume;
    * the entrypoint runs as ROOT before dropping privileges, can write it, and
      resolved ``PROFILE`` — so it never read the durable store, restored
      nothing, and left the restore path's reserved-path gate unexercised on the
      one caller that runs as root. Since the app skips the restore entirely
      under ``OSPREY_RENDER_ZONE_READONLY=1``, no user-owned body was EVER
      restored in a container.

    One project, two answers, decided by privilege. These pin the rule that
    replaces it: on a container render the baked profile is skipped outright.
    """

    MARKER = "OSPREY_RENDER_ZONE_READONLY"

    @pytest.fixture()
    def container_render(self, project_dir, volume_dir, monkeypatch):
        """A real container render: profile PRESENT and writable, plus a volume.

        The profile is left exactly as the build wrote it — this fixture's whole
        point is that it exists and that the test process can write it, which is
        what root sees inside the image.
        """
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(volume_dir))
        monkeypatch.setenv(self.MARKER, "1")
        assert _profile_root(project_dir).is_dir(), "the render is expected to name a real profile"
        assert os.access(_profile_root(project_dir), os.W_OK), (
            "this fixture is only meaningful while the profile root is writable — "
            "that is the condition that used to decide the answer"
        )
        return project_dir

    def test_a_writable_baked_profile_does_not_win(self, container_render, volume_dir):
        """The defect, stated directly: root must not resolve PROFILE here."""
        ownership = resolve_ownership(container_render)

        assert ownership.mode is OwnershipMode.VOLUME
        assert ownership.profile_root is None
        assert ownership.store is not None
        assert ownership.store.root == volume_dir / "osprey" / "scaffold"

    def test_without_the_marker_a_writable_profile_still_wins(self, project_dir, monkeypatch):
        """Bare-host behaviour is unchanged, which is the constraint on the fix.

        Same tree, same writable profile, same mounted store — only the
        container marker is absent. A rule that keyed off the volume alone would
        flip this one too.
        """
        monkeypatch.delenv(self.MARKER, raising=False)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(project_dir.parent / "claude-config"))

        ownership = resolve_ownership(project_dir)

        assert ownership.mode is OwnershipMode.PROFILE
        assert ownership.profile_root == _profile_root(project_dir)

    def test_a_container_render_without_a_volume_refuses(self, project_dir, monkeypatch):
        """No durable surface, so writes refuse rather than edit the image.

        A claim written into the baked profile lives in the container's writable
        layer and is gone on the next recreation — reported as success and then
        erased, which is exactly what DEGRADED exists to refuse. This is also
        the answer the app (as uid 1000) already gave here, so root and the
        server now agree.
        """
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        monkeypatch.setenv(self.MARKER, "1")

        assert resolve_ownership(project_dir).mode is OwnershipMode.DEGRADED

    def test_only_an_exact_1_is_the_marker(self, container_render, monkeypatch):
        """The same reading the web app applies, so the two cannot come apart."""
        monkeypatch.setenv(self.MARKER, "true")

        assert resolve_ownership(container_render).mode is OwnershipMode.PROFILE

    def test_the_root_restore_installs_a_user_owned_body(self, container_render, tmp_path):
        """The no-op, end to end: the entrypoint's restore now finds the store.

        Claim and edit an artifact, then hand the restore a pristine tree — the
        image, unchanged — and the operator's body has to come back. Before the
        fix this returned ``[]`` against a store that held the body all along.
        """
        pristine = _recreate_container(container_render, tmp_path / "image-rebuild")

        svc = ScaffoldGalleryService(container_render)
        svc.scaffold_override(WRITABLE_ARTIFACT)
        edited = "# Channel finder\nEdited by the operator.\n"
        svc.save_override(WRITABLE_ARTIFACT, edited)

        assert restore_scaffold_bodies(pristine) == [WRITABLE_ARTIFACT]
        restored = pristine / ".claude" / "agents" / "channel-finder.md"
        assert restored.read_text(encoding="utf-8") == edited

    def test_the_root_restore_still_refuses_a_reserved_record(
        self, container_render, volume_dir, audit_zone
    ):
        """The gate 4.5 built for the root path finally runs on the root path.

        While the entrypoint resolved PROFILE it never read the store, so the
        reserved-path gate in ``rehydrate`` was only ever exercised by the
        unprivileged in-process startup — the caller that could not have written
        the render zone anyway. This is the privileged one.
        """
        name, output_path = "rules/evil", ".claude/rules/evil.md"
        _plant_store_record(volume_dir, name, output_path, "# Written by the agent\n")

        assert restore_scaffold_bodies(container_render) == []
        assert not (container_render / output_path).exists()

        records = _protected_records(audit_zone)
        assert len(records) == 1
        assert records[0]["surface"] == "scaffold_restore"
        assert f"target={output_path}" in records[0]["detail"]


class TestRestoreRefusesBodiesThatEscapeTheStore:
    """The SOURCE side of a restore, which was open while the destination was shut.

    ``rehydrate`` already refuses a record whose write would land on a reserved
    file. That guards where the bytes GO. Where they come FROM was a plain
    ``read_text`` on ``<content_dir>/<output_path>``, which follows symlinks —
    and the content directory is the per-user claude-config volume, the agent's
    own ``$HOME``, writable by uid 1000.

    That mattered the moment the container entrypoint's restore started
    resolving VOLUME: it runs as ROOT. The agent could plant a body as a link
    onto any root-readable file, or make an intermediate directory the link, and
    have root copy that file's bytes into the render zone — which the agent can
    read. A file the agent is allowed to write, turned into an arbitrary-read
    primitive.

    Refused like a reserved destination: skipped, audited, never raised.
    """

    def _store(self, volume_dir: Path) -> OwnershipStore:
        return OwnershipStore(root=volume_dir / "osprey" / "scaffold")

    def _index(self, volume_dir: Path, name: str, output_path: str) -> None:
        store = self._store(volume_dir)
        store.index_path.parent.mkdir(parents=True, exist_ok=True)
        store.index_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "artifacts": {name: {"state": "claimed", "output_path": output_path}},
                }
            ),
            encoding="utf-8",
        )

    def test_a_symlinked_body_is_not_restored(self, container_project, volume_dir, audit_zone):
        """The final component is the link: the classic shape."""
        secret = volume_dir.parent / "root-only.txt"
        secret.write_text("SUPER-SECRET-ROOT-BYTES\n", encoding="utf-8")
        body = self._store(volume_dir).content_dir / ".claude" / "agents" / "pwn.md"
        body.parent.mkdir(parents=True)
        body.symlink_to(secret)
        self._index(volume_dir, "agents/pwn", ".claude/agents/pwn.md")

        assert restore_scaffold_bodies(container_project) == []

        installed = container_project / ".claude" / "agents" / "pwn.md"
        assert not installed.exists(), (
            "root read a file outside the store and wrote it into the render"
        )
        records = _protected_records(audit_zone)
        assert [r["surface"] for r in records] == ["scaffold_restore"]
        assert records[0]["subject"] == ".claude/agents/pwn.md"
        assert records[0]["reason"] == "ownership store body escapes the store"

    def test_a_body_reached_through_a_symlinked_directory_is_not_restored(
        self, container_project, volume_dir, audit_zone
    ):
        """The variant ``O_NOFOLLOW`` alone would miss.

        ``O_NOFOLLOW`` guards the final component only, and every intermediate
        directory under the content root is just as plantable by the agent. The
        containment check on the RESOLVED path is what covers this one, which is
        why both are kept rather than either alone.
        """
        outside = volume_dir.parent / "outside"
        outside.mkdir()
        (outside / "pwn.md").write_text("VIA-A-SYMLINKED-DIRECTORY\n", encoding="utf-8")
        content = self._store(volume_dir).content_dir / ".claude"
        content.mkdir(parents=True)
        (content / "agents").symlink_to(outside, target_is_directory=True)
        self._index(volume_dir, "agents/pwn", ".claude/agents/pwn.md")

        assert restore_scaffold_bodies(container_project) == []
        assert not (container_project / ".claude" / "agents" / "pwn.md").exists()
        assert len(_protected_records(audit_zone)) == 1

    def test_a_fifo_body_is_refused_under_its_own_reason(
        self, container_project, volume_dir, audit_zone
    ):
        """Inside the store, so not an escape — and a stall, not a leak.

        A FIFO exfiltrates nothing, but a blocking open on one waits for a
        writer that never comes, and this walk runs in the container entrypoint
        BEFORE the server exists: the container would hang at start forever.
        Every other failure here is fail-open; an unbounded wait is the one that
        is neither open nor closed. The reason is its own so an operator reading
        the log can tell a stall attempt from a read attempt.
        """
        body = self._store(volume_dir).content_dir / ".claude" / "agents" / "pwn.md"
        body.parent.mkdir(parents=True)
        os.mkfifo(body)
        self._index(volume_dir, "agents/pwn", ".claude/agents/pwn.md")

        assert restore_scaffold_bodies(container_project) == []
        assert not (container_project / ".claude" / "agents" / "pwn.md").exists()

        records = _protected_records(audit_zone)
        assert [r["reason"] for r in records] == ["ownership store body is not a regular file"]

    def test_a_directory_body_is_refused_under_the_same_reason(
        self, container_project, volume_dir, audit_zone
    ):
        """The other non-regular shape, and the one a naive ``is_file()``
        already declined — pinned so the reason stays attached to it."""
        body = self._store(volume_dir).content_dir / ".claude" / "agents" / "pwn.md"
        body.mkdir(parents=True)
        self._index(volume_dir, "agents/pwn", ".claude/agents/pwn.md")

        assert restore_scaffold_bodies(container_project) == []
        records = _protected_records(audit_zone)
        assert [r["reason"] for r in records] == ["ownership store body is not a regular file"]

    def test_a_hard_linked_body_is_refused(self, container_project, volume_dir, audit_zone):
        """The escape again, with the symlink removed.

        Inside the store, a regular file, and still not one the store wrote:
        bodies are always created fresh through ``NamedTemporaryFile`` +
        ``os.replace``, so a legitimate body has exactly one link. A second link
        means the same inode is reachable under a name the store does not
        control — and on a shared volume that is the agent's name for it. The
        containment check cannot see this one, because the path really is
        inside the store.
        """
        target = volume_dir.parent / "root-only.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("SUPER-SECRET-ROOT-BYTES\n", encoding="utf-8")
        body = self._store(volume_dir).content_dir / ".claude" / "agents" / "pwn.md"
        body.parent.mkdir(parents=True)
        os.link(target, body)
        self._index(volume_dir, "agents/pwn", ".claude/agents/pwn.md")

        assert restore_scaffold_bodies(container_project) == []
        assert not (container_project / ".claude" / "agents" / "pwn.md").exists()

        records = _protected_records(audit_zone)
        assert [r["reason"] for r in records] == ["ownership store body is a hard link"]

    def test_a_missing_body_is_not_audited_as_an_escape(
        self, container_project, volume_dir, audit_zone
    ):
        """A record with no body kept is ordinary, not an attack.

        The two must stay distinguishable, or the audit log fills with records
        an operator cannot act on and the real one stops standing out.
        """
        self._index(volume_dir, "agents/absent", ".claude/agents/absent.md")

        assert restore_scaffold_bodies(container_project) == []
        assert _protected_records(audit_zone) == []

    def test_an_ordinary_body_is_still_restored(self, container_project, volume_dir, tmp_path):
        """The guard must not cost the feature: a real body still comes back."""
        pristine = _recreate_container(container_project, tmp_path / "image-rebuild")
        svc = ScaffoldGalleryService(container_project)
        svc.scaffold_override(WRITABLE_ARTIFACT)
        svc.save_override(WRITABLE_ARTIFACT, "# Channel finder\nMine.\n")

        assert restore_scaffold_bodies(pristine) == [WRITABLE_ARTIFACT]
