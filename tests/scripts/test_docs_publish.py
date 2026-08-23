"""Tests for the documentation publish helper.

The site root belongs to the newest release, so "which tag is newest" is the
one answer in this module that can quietly destroy the published site if it
is wrong. These tests pin it against the two ways the old shell one-liner got
it wrong: tags that are not releases at all (backup and connector tags) being
considered, and a text sort ranking `v2026.6.2` above `v2026.10.0`.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "docs_publish.py"
_spec = importlib.util.spec_from_file_location("docs_publish", _MODULE_PATH)
assert _spec and _spec.loader
docs_publish = importlib.util.module_from_spec(_spec)
# scripts/ is not a package, so the module is loaded by path. It must be
# registered before exec so `@dataclass` can resolve its annotations through
# `cls.__module__`.
sys.modules[_spec.name] = docs_publish
_spec.loader.exec_module(docs_publish)


class TestReleaseTags:
    def test_stray_tags_are_dropped(self):
        """Backup and per-component tags share the repository with releases."""
        assert docs_publish.release_tags(
            [
                "checkpoint-final",
                "osprey-connectors-v0.1.0",
                "safety-p1-precheck-backup",
                "v2026.6.2",
            ]
        ) == ["v2026.6.2"]

    def test_non_release_v_tags_are_dropped(self):
        """`v2026.7` and `v1.0.0rc1` look like releases but are not `vX.Y.Z`."""
        assert docs_publish.release_tags(["v2026.7", "v1.0.0rc1", "v2026.6.2"]) == ["v2026.6.2"]

    def test_ordering_is_numeric_not_lexical(self):
        """A text sort puts `v2026.6.2` above `v2026.10.0`; the site would regress."""
        assert docs_publish.release_tags(
            ["v2026.6.2", "v2026.10.0", "v2026.6.10", "v2027.1.0"]
        ) == ["v2027.1.0", "v2026.10.0", "v2026.6.10", "v2026.6.2"]

    def test_release_wins_over_a_stray_that_sorts_first_as_text(self):
        """The exact failure mode of `git tag --sort=... | head -n1`."""
        candidates = ["zzz-tag", "v2026.6.2", "osprey-connectors-v9.9.9", ""]
        # Guard the premise: plain string order really does put a stray first.
        assert sorted(candidates, reverse=True)[0] == "zzz-tag"
        assert docs_publish.release_tags(candidates)[0] == "v2026.6.2"

    def test_empty_input_yields_empty_list(self):
        assert docs_publish.release_tags([]) == []

    def test_no_release_tags_yields_empty_list(self):
        assert docs_publish.release_tags(["checkpoint-final", "v2026.7"]) == []

    def test_git_tag_output_split_on_newlines_is_tolerated(self):
        """`git tag` output ends in a newline, so splitting leaves a blank entry."""
        raw = "checkpoint-final\nv2026.6.2\n  v2026.10.0  \n\n"
        assert docs_publish.release_tags(raw.split("\n")) == ["v2026.10.0", "v2026.6.2"]

    def test_accepts_any_iterable(self):
        """The CLI passes a list; callers may pass a generator or a tuple."""
        assert docs_publish.release_tags(t for t in ("v1.2.3", "nope")) == ["v1.2.3"]


class TestPlan:
    """`plan()` decides the deploy; the main-is-never-latest rule lives here."""

    # -- rule 4: the invariant ------------------------------------------------

    @pytest.mark.parametrize(
        "tags",
        [
            pytest.param([], id="no-tags-at-all"),
            pytest.param(["v2026.6.2"], id="one-release"),
            pytest.param(["v2026.6.2", "v2026.8.0"], id="several-releases"),
            pytest.param(["checkpoint-final", "osprey-connectors-v0.1.0"], id="only-strays"),
        ],
    )
    @pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
    def test_main_is_never_latest(self, event, tags):
        """The bug this module exists for: every merge to main overwrote the root."""
        result = docs_publish.plan("refs/heads/main", event, "", tags)
        assert result.is_latest is False
        assert result.deploy_dir == "latest"
        assert result.version == "dev"
        assert result.deploy is True

    def test_main_is_not_latest_even_when_its_describe_tag_is_newest(self):
        """Right after a release, main sits on the newest tag -- still not the root."""
        result = docs_publish.plan("refs/heads/main", "push", "", ["v2026.6.2", "v2026.8.0"])
        assert result.is_latest is False
        assert result.deploy_dir == "latest"

    # -- rule 2: release tag pushes -------------------------------------------

    def test_newest_release_tag_takes_the_root(self):
        """Exactly one build may rewrite the site root: the newest release."""
        result = docs_publish.plan("refs/tags/v2026.8.0", "push", "", ["v2026.6.2", "v2026.8.0"])
        assert result == docs_publish.Plan(
            version="2026.8.0", deploy_dir="v2026.8.0", is_latest=True, deploy=True
        )

    def test_older_release_tag_deploys_to_its_own_directory_only(self):
        """Re-running an old release must archive it, not regress the root."""
        result = docs_publish.plan("refs/tags/v2026.6.2", "push", "", ["v2026.6.2", "v2026.8.0"])
        assert result.deploy_dir == "v2026.6.2"
        assert result.version == "2026.6.2"
        assert result.is_latest is False
        assert result.deploy is True

    def test_release_tag_with_no_known_tags_is_not_latest(self):
        """An empty tag list must yield "not latest", never an IndexError."""
        result = docs_publish.plan("refs/tags/v2026.8.0", "push", "", [])
        assert result.is_latest is False
        assert result.deploy is True
        assert result.deploy_dir == "v2026.8.0"

    # -- rule 3: tag refs that are not releases -------------------------------

    @pytest.mark.parametrize(
        "ref",
        [
            "refs/tags/v2026.7",
            "refs/tags/v1.0.0rc1",
            "refs/tags/v1.0.0-rc1",
            "refs/tags/osprey-connectors-v0.1.0",
            "refs/tags/checkpoint-final",
        ],
    )
    def test_non_release_tag_refs_do_not_deploy(self, ref):
        """These share the workflow's loose `v*` trigger but are not the site."""
        result = docs_publish.plan(ref, "push", "", ["v2026.6.2"])
        assert result == docs_publish.Plan(
            version="dev", deploy_dir="pr-preview", is_latest=False, deploy=False
        )

    # -- rule 5: everything else ----------------------------------------------

    @pytest.mark.parametrize(
        ("ref", "event"),
        [
            pytest.param("refs/pull/42/merge", "pull_request", id="pull-request"),
            pytest.param("refs/heads/feature/x", "workflow_dispatch", id="dispatch-topic"),
            pytest.param("refs/heads/docs/versioned-publishing", "push", id="push-topic"),
        ],
    )
    def test_other_refs_build_but_do_not_deploy(self, ref, event):
        """A preview build must never touch the published site."""
        result = docs_publish.plan(ref, event, "", ["v2026.6.2"])
        assert result.deploy is False
        assert result.is_latest is False
        assert result.deploy_dir == "pr-preview"
        assert result.version == "dev"

    # -- rule 1: the dispatch tag input ---------------------------------------

    def test_input_tag_overrides_the_ref_it_was_dispatched_from(self):
        """Re-publishing a release is dispatched from main, but builds the tag."""
        result = docs_publish.plan(
            "refs/heads/main", "workflow_dispatch", "v2026.8.0", ["v2026.8.0"]
        )
        assert result == docs_publish.Plan(
            version="2026.8.0", deploy_dir="v2026.8.0", is_latest=True, deploy=True
        )

    def test_input_tag_is_stripped(self):
        """Dispatch inputs are hand-typed, so stray whitespace is expected."""
        result = docs_publish.plan("refs/heads/main", "workflow_dispatch", "  v2026.8.0  ", [])
        assert result.deploy_dir == "v2026.8.0"

    @pytest.mark.parametrize("blank", ["", "   ", "\n"])
    def test_blank_input_tag_falls_through_to_the_ref(self, blank):
        """An unset optional workflow input arrives as an empty string."""
        result = docs_publish.plan("refs/heads/main", "workflow_dispatch", blank, [])
        assert result.deploy_dir == "latest"

    @pytest.mark.parametrize(
        "bad",
        ["2026.6.2", "v2026.6", "v2026.6.2-rc1", "main", "v2026.6.2.1", "latest"],
    )
    def test_malformed_input_tag_raises(self, bad):
        """A typo must fail loudly, not deploy somewhere unintended."""
        with pytest.raises(docs_publish.PlanError) as excinfo:
            docs_publish.plan("refs/heads/main", "workflow_dispatch", bad, ["v2026.6.2"])
        message = str(excinfo.value)
        assert bad in message
        assert "vX.Y.Z" in message

    def test_plan_error_is_a_value_error(self):
        """Callers that catch `ValueError` still see a malformed dispatch."""
        assert issubclass(docs_publish.PlanError, ValueError)


def _seed(root: Path, files: dict[str, str]) -> Path:
    """Create `root` and write each `"relative/path": text` entry beneath it."""
    root.mkdir(parents=True, exist_ok=True)
    for relative, text in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return root


def _snapshot(root: Path) -> dict[str, str]:
    """Map every file under `root` to its contents, keyed by relative path."""
    return {
        str(path.relative_to(root)): path.read_text()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _release_plan(tag: str, *, is_latest: bool) -> object:
    return docs_publish.Plan(version=tag[1:], deploy_dir=tag, is_latest=is_latest, deploy=True)


_MAIN_PLAN = docs_publish.Plan(version="dev", deploy_dir="latest", is_latest=False, deploy=True)


class TestStage:
    """A staged tree replaces `gh-pages` wholesale, so its mistakes are losses."""

    def _build(self, tmp_path: Path) -> Path:
        return _seed(
            tmp_path / "build",
            {
                "index.html": "new root",
                "_static/new.css": "new css",
                "guide/page.html": "new guide",
            },
        )

    def test_missing_existing_tree_refuses_to_publish(self, tmp_path):
        """A failed `gh-pages` clone must not be mistaken for an empty site."""
        with pytest.raises(docs_publish.PlanError) as excinfo:
            docs_publish.stage(
                self._build(tmp_path),
                tmp_path / "absent",
                tmp_path / "deployment",
                _MAIN_PLAN,
            )
        assert "allow_empty_site" in str(excinfo.value)
        assert not (tmp_path / "deployment").exists()

    def test_none_existing_tree_refuses_to_publish(self, tmp_path):
        """The workflow passes `None` when it never even attempted a clone."""
        with pytest.raises(docs_publish.PlanError):
            docs_publish.stage(self._build(tmp_path), None, tmp_path / "deployment", _MAIN_PLAN)

    def test_empty_existing_tree_refuses_to_publish(self, tmp_path):
        """A clone that produced a directory but no files is the same failure."""
        existing = _seed(tmp_path / "existing", {})
        with pytest.raises(docs_publish.PlanError):
            docs_publish.stage(self._build(tmp_path), existing, tmp_path / "deployment", _MAIN_PLAN)

    def test_empty_existing_tree_is_allowed_when_asked_for(self, tmp_path):
        """The genuine first deploy has to be spelled out, and then works."""
        deployment = tmp_path / "deployment"
        docs_publish.stage(
            self._build(tmp_path),
            None,
            deployment,
            _MAIN_PLAN,
            allow_empty_site=True,
        )
        assert (deployment / "latest" / "index.html").read_text() == "new root"
        assert (deployment / ".nojekyll").exists()

    def test_root_refresh_drops_pages_removed_upstream(self, tmp_path):
        """Renamed or deleted pages must stop being served from the root."""
        existing = _seed(
            tmp_path / "existing",
            {
                "index.html": "old root",
                "how-to/deploy-project.html": "renamed away",
                "_static/old.css": "old css",
                "latest/index.html": "development build",
                "v2026.6.1/index.html": "archived release",
            },
        )
        deployment = tmp_path / "deployment"
        docs_publish.stage(
            self._build(tmp_path),
            existing,
            deployment,
            _release_plan("v2026.6.2", is_latest=True),
        )

        assert not (deployment / "how-to").exists()
        assert not (deployment / "_static" / "old.css").exists()
        assert (deployment / "index.html").read_text() == "new root"
        assert (deployment / "_static" / "new.css").read_text() == "new css"
        assert (deployment / "guide" / "page.html").read_text() == "new guide"

    def test_root_refresh_preserves_the_cname(self, tmp_path):
        """`CNAME` is GitHub Pages' custom-domain file, not stale root content.

        No Sphinx build emits one, so a root refresh that treated it like any
        other leftover page would take the custom domain offline on the next
        release. Pinned alongside a page that *must* be swept, so the test
        cannot pass by the refresh having stopped deleting anything.
        """
        existing = _seed(
            tmp_path / "existing",
            {
                "index.html": "old root",
                "CNAME": "docs.example.org\n",
                "how-to/x.html": "renamed away",
            },
        )
        deployment = tmp_path / "deployment"
        docs_publish.stage(
            self._build(tmp_path),
            existing,
            deployment,
            _release_plan("v2026.6.2", is_latest=True),
        )

        assert (deployment / "CNAME").read_text() == "docs.example.org\n"
        assert not (deployment / "how-to").exists()

    def test_root_refresh_preserves_latest_and_release_directories(self, tmp_path):
        """Other builds' output is not stale root content; wiping it is an outage."""
        existing = _seed(
            tmp_path / "existing",
            {
                "index.html": "old root",
                "latest/index.html": "development build",
                "v2026.6.1/index.html": "archived release",
            },
        )
        deployment = tmp_path / "deployment"
        docs_publish.stage(
            self._build(tmp_path),
            existing,
            deployment,
            _release_plan("v2026.6.2", is_latest=True),
        )

        assert (deployment / "latest" / "index.html").read_text() == "development build"
        assert (deployment / "v2026.6.1" / "index.html").read_text() == "archived release"
        # The release's own directory matches `RELEASE_TAG_RE`, so the same
        # rule that spares the archive spares the build just written.
        assert (deployment / "v2026.6.2" / "index.html").read_text() == "new root"

    def test_main_push_leaves_the_root_untouched(self, tmp_path):
        """The invariant the module exists for: `main` never rewrites the root."""
        existing = _seed(
            tmp_path / "existing",
            {
                "index.html": "stable root",
                "_static/stable.css": "stable css",
                "v2026.6.2/index.html": "stable release",
            },
        )
        deployment = tmp_path / "deployment"
        docs_publish.stage(self._build(tmp_path), existing, deployment, _MAIN_PLAN)

        assert (deployment / "index.html").read_text() == "stable root"
        assert (deployment / "_static" / "stable.css").read_text() == "stable css"
        assert (deployment / "v2026.6.2" / "index.html").read_text() == "stable release"
        assert (deployment / "latest" / "index.html").read_text() == "new root"
        assert (deployment / "latest" / "_static" / "new.css").read_text() == "new css"

    def test_republishing_a_tag_replaces_its_directory(self, tmp_path):
        """A re-run must not leave pages the current build no longer produces."""
        existing = _seed(
            tmp_path / "existing",
            {
                "index.html": "stable root",
                "v2026.6.1/index.html": "first attempt",
                "v2026.6.1/stale.html": "page dropped since",
            },
        )
        deployment = tmp_path / "deployment"
        docs_publish.stage(
            self._build(tmp_path),
            existing,
            deployment,
            _release_plan("v2026.6.1", is_latest=False),
        )

        assert not (deployment / "v2026.6.1" / "stale.html").exists()
        assert (deployment / "v2026.6.1" / "index.html").read_text() == "new root"
        assert (deployment / "v2026.6.1" / "guide" / "page.html").read_text() == "new guide"

    def test_nojekyll_is_written(self, tmp_path):
        """Without it, Pages runs Jekyll and drops every `_static/` asset."""
        existing = _seed(tmp_path / "existing", {"index.html": "stable root"})
        deployment = tmp_path / "deployment"
        docs_publish.stage(
            self._build(tmp_path),
            existing,
            deployment,
            _release_plan("v2026.6.2", is_latest=True),
        )
        assert (deployment / ".nojekyll").is_file()

    def test_existing_tree_is_never_modified(self, tmp_path):
        """`existing` is the downloaded live site; staging only ever reads it."""
        existing = _seed(
            tmp_path / "existing",
            {
                "index.html": "old root",
                "how-to/deploy-project.html": "renamed away",
                "latest/index.html": "development build",
                "v2026.6.1/index.html": "archived release",
            },
        )
        before = _snapshot(existing)
        docs_publish.stage(
            self._build(tmp_path),
            existing,
            tmp_path / "deployment",
            _release_plan("v2026.6.2", is_latest=True),
        )
        assert _snapshot(existing) == before


class TestVersions:
    """The switcher is the only navigation between versions; a wrong entry strands readers."""

    def test_newest_release_is_the_single_preferred_entry(self, tmp_path):
        """pydata-sphinx-theme requires exactly one preferred version."""
        _seed(tmp_path, {"v2026.6.1/index.html": "", "latest/index.html": ""})
        entries = docs_publish.versions(["v2026.6.1", "v2026.6.2"], tmp_path)
        preferred = [entry for entry in entries if entry.get("preferred")]
        assert len(preferred) == 1
        assert preferred[0]["name"] == "v2026.6.2 (stable)"
        assert entries[0] is preferred[0]

    def test_development_entry_appears_whenever_latest_exists(self, tmp_path):
        """A tag push used to drop `/latest/` from the switcher entirely."""
        _seed(tmp_path, {"latest/index.html": ""})
        entries = docs_publish.versions(["v2026.6.2"], tmp_path)
        assert {
            "name": "latest (development)",
            "version": "dev",
            "url": docs_publish.SITE_URL + "latest/",
        } in entries

    def test_development_entry_is_absent_without_a_latest_directory(self, tmp_path):
        """Before the first `main` build there is nothing to link to."""
        entries = docs_publish.versions(["v2026.6.2"], tmp_path)
        assert all(entry["version"] != "dev" for entry in entries)

    def test_snapshot_without_a_directory_is_skipped(self, tmp_path):
        """Tags cut before this workflow existed have no published tree."""
        _seed(tmp_path, {"v2026.6.1/index.html": ""})
        entries = docs_publish.versions(["v2026.5.0", "v2026.6.1", "v2026.6.2"], tmp_path)
        assert [entry["name"] for entry in entries] == ["v2026.6.2 (stable)", "v2026.6.1"]

    def test_newest_release_is_not_repeated_as_a_snapshot(self, tmp_path):
        """Its directory exists too, but the root already represents it."""
        _seed(tmp_path, {"v2026.6.2/index.html": "", "v2026.6.1/index.html": ""})
        entries = docs_publish.versions(["v2026.6.1", "v2026.6.2"], tmp_path)
        assert [entry["name"] for entry in entries] == ["v2026.6.2 (stable)", "v2026.6.1"]

    def test_stray_and_non_release_tags_are_ignored(self, tmp_path):
        """The switcher must not offer a backup tag as a documentation version."""
        _seed(
            tmp_path,
            {
                "checkpoint-final/index.html": "",
                "v2026.7/index.html": "",
                "v2026.6.1/index.html": "",
            },
        )
        entries = docs_publish.versions(
            ["checkpoint-final", "osprey-connectors-v0.1.0", "v2026.7", "v2026.6.1", "v2026.6.2"],
            tmp_path,
        )
        assert [entry["name"] for entry in entries] == ["v2026.6.2 (stable)", "v2026.6.1"]

    def test_no_tags_yields_only_the_development_entry(self, tmp_path):
        """A repository publishing docs before its first release."""
        _seed(tmp_path, {"latest/index.html": ""})
        assert docs_publish.versions([], tmp_path) == [
            {
                "name": "latest (development)",
                "version": "dev",
                "url": docs_publish.SITE_URL + "latest/",
            }
        ]

    def test_no_tags_and_no_latest_yields_nothing(self, tmp_path):
        assert docs_publish.versions([], tmp_path) == []

    def test_versions_match_what_conf_py_emits_as_release(self, tmp_path):
        """`version_match` comes from `release`: bare `X.Y.Z`, or literal `dev`."""
        _seed(tmp_path, {"v2026.6.1/index.html": "", "latest/index.html": ""})
        entries = docs_publish.versions(["v2026.6.1", "v2026.6.2"], tmp_path)
        assert [entry["version"] for entry in entries] == ["2026.6.2", "2026.6.1", "dev"]

    def test_urls_are_rooted_at_the_published_site(self, tmp_path):
        """A relative or stale host here sends every switcher click to a 404."""
        _seed(tmp_path, {"v2026.6.1/index.html": "", "latest/index.html": ""})
        entries = docs_publish.versions(["v2026.6.1", "v2026.6.2"], tmp_path)
        assert [entry["url"] for entry in entries] == [
            docs_publish.SITE_URL,
            docs_publish.SITE_URL + "v2026.6.1/",
            docs_publish.SITE_URL + "latest/",
        ]

    def test_payload_round_trips_through_json(self, tmp_path):
        """The result is written verbatim as `_static/versions.json`."""
        _seed(tmp_path, {"v2026.6.1/index.html": "", "latest/index.html": ""})
        entries = docs_publish.versions(["v2026.6.1", "v2026.6.2"], tmp_path)
        assert json.loads(json.dumps(entries)) == entries


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the script the way the workflow does: a fresh `python3` process.

    Every call passes `--tags` explicitly. The CLI falls back to `git tag` in
    this repository otherwise, which would make the assertions depend on
    whichever tags the checkout running the suite happens to carry.
    """
    return subprocess.run(
        [sys.executable, str(_MODULE_PATH), *args],
        capture_output=True,
        text=True,
    )


_TAGS = ("--tags", "v2026.6.2", "--tags", "v2026.8.0")


def _run_main_stage(
    build: Path, existing: Path, deployment: Path, *extra: str
) -> subprocess.CompletedProcess[str]:
    """`stage` for a push to `main`, spelled the way the workflow spells it."""
    return _run(
        "stage",
        "--ref",
        "refs/heads/main",
        "--event",
        "push",
        *_TAGS,
        "--build",
        str(build),
        "--existing",
        str(existing),
        "--deployment",
        str(deployment),
        *extra,
    )


class TestCli:
    """The workflow reads stdout; anything unexpected there corrupts a deploy."""

    def test_plan_for_a_main_push_emits_the_four_output_lines(self):
        """`main` publishes `/latest/` and must never claim to be the stable build."""
        result = _run("plan", "--ref", "refs/heads/main", "--event", "push", *_TAGS)
        assert result.returncode == 0
        assert result.stdout == ("version=dev\ndeploy_dir=latest\nis_latest=false\ndeploy=true\n")

    def test_plan_for_the_newest_release_tag_claims_the_root(self):
        result = _run("plan", "--ref", "refs/tags/v2026.8.0", "--event", "push", *_TAGS)
        assert result.returncode == 0
        assert result.stdout == (
            "version=2026.8.0\ndeploy_dir=v2026.8.0\nis_latest=true\ndeploy=true\n"
        )

    def test_plan_rejects_a_malformed_tag_input_without_writing_stdout(self):
        """A bad dispatch must fail, not append a partial line to $GITHUB_OUTPUT."""
        result = _run(
            "plan",
            "--ref",
            "refs/heads/main",
            "--event",
            "workflow_dispatch",
            "--input-tag",
            "2026.6.2",
            *_TAGS,
        )
        assert result.returncode == 2
        assert result.stdout == ""
        assert "2026.6.2" in result.stderr

    def test_stage_writes_latest_and_leaves_the_existing_root_alone(self, tmp_path):
        """The regression this module exists for, exercised through the CLI."""
        build = _seed(tmp_path / "build", {"index.html": "dev build"})
        existing = _seed(
            tmp_path / "existing",
            {"index.html": "released root", "v2026.6.2/index.html": "released"},
        )
        deployment = tmp_path / "deployment"

        result = _run_main_stage(build, existing, deployment)

        assert result.returncode == 0
        assert (deployment / "latest" / "index.html").read_text() == "dev build"
        assert (deployment / "index.html").read_text() == "released root"
        assert (deployment / "v2026.6.2" / "index.html").read_text() == "released"
        assert (deployment / ".nojekyll").is_file()

    def test_stage_refuses_an_empty_existing_tree(self, tmp_path):
        build = _seed(tmp_path / "build", {"index.html": "dev build"})
        existing = _seed(tmp_path / "existing", {})
        result = _run_main_stage(build, existing, tmp_path / "deployment")
        assert result.returncode == 2
        assert result.stdout == ""

    def test_stage_publishes_an_empty_site_only_when_told_to(self, tmp_path):
        """The first deploy, where there is no `gh-pages` branch to clone."""
        build = _seed(tmp_path / "build", {"index.html": "dev build"})
        existing = _seed(tmp_path / "existing", {})
        deployment = tmp_path / "deployment"
        result = _run_main_stage(build, existing, deployment, "--allow-empty-site")
        assert result.returncode == 0
        assert (deployment / "latest" / "index.html").read_text() == "dev build"

    def test_versions_writes_the_switcher_payload_and_echoes_it(self, tmp_path):
        deployment = _seed(
            tmp_path / "deployment",
            {"latest/index.html": "", "v2026.6.2/index.html": ""},
        )
        result = _run(
            "versions",
            "--deployment",
            str(deployment),
            "--tags",
            "v2026.8.0",
            "--tags",
            "v2026.6.2",
        )
        assert result.returncode == 0

        written = json.loads((deployment / "_static" / "versions.json").read_text())
        assert json.loads(result.stdout) == written
        # pydata-sphinx-theme requires exactly one; more than one and the
        # switcher has no single stable version to fall back to.
        assert [entry for entry in written if entry.get("preferred")] == [written[0]]
        assert written[0]["version"] == "2026.8.0"
