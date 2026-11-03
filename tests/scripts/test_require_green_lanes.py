"""Tests for the release tag gate shared by both release workflows.

git and the GitHub API are injected as tables, so no test builds a repository
or reaches the network. Every rule the gate applies has a negative control: a
lane that is missing, skipped or failed in an otherwise green run; a green run
on a commit with a different tree; a later green-but-empty re-run that must not
hide the run that did the work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._by_path import load_module

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "require_green_lanes.py"
gate = load_module("require_green_lanes", _MODULE_PATH)

REPO = "als-apg/osprey"
TAG = "osprey-connectors-v2026.10.0"
TAG_SHA = "a" * 40
MERGE_SHA = "b" * 40
OTHER_SHA = "c" * 40
LANES = "test=Test Python\nconnectors-standalone=Connectors suite without the framework\n"


class FakeGit:
    """``git rev-parse`` / ``git log`` over a small history: tag -> merge -> other.

    The tag and the merge commit carry the same tree; the commit below them does not.
    """

    TREES = {TAG_SHA: "tree1", MERGE_SHA: "tree1", OTHER_SHA: "tree2"}

    def __call__(self, *args: str) -> str:
        if args[0] == "log":
            return f"{TAG_SHA}\n{MERGE_SHA}\n{OTHER_SHA}"
        assert args[0] == "rev-parse", args
        ref = args[1]
        if ref.endswith("^{tree}"):
            return self.TREES[ref.removesuffix("^{tree}")]
        if ref.endswith("^{commit}"):
            return ref.removesuffix("^{commit}")
        if ref.endswith("^@"):
            return MERGE_SHA
        raise AssertionError(ref)


def job(name: str, conclusion: str = "success") -> dict:
    return {"name": name, "conclusion": conclusion}


GREEN_JOBS = [
    job("Test Python 3.11 on ubuntu-latest (rest-a)"),
    job("Test Python 3.11 on macos-latest (rest-a)"),
    job("Connectors suite without the framework"),
]


class FakeApi:
    """``{sha: [(run id, conclusion, jobs)]}`` answered as the two endpoints the gate reads."""

    def __init__(self, runs: dict[str, list[tuple[int, str, list[dict]]]]):
        self.runs = runs
        self.paths: list[str] = []

    def __call__(self, path: str, paginate: bool):
        self.paths.append(path)
        if "/workflows/ci.yml/runs?head_sha=" in path:
            sha = path.split("head_sha=")[1].split("&")[0]
            return {
                "workflow_runs": [
                    {"id": rid, "conclusion": c, "html_url": f"https://ci/{rid}"}
                    for rid, c, _ in self.runs.get(sha, [])
                ]
            }
        rid = int(path.split("/runs/")[1].split("/")[0])
        assert paginate
        jobs = next(j for runs in self.runs.values() for r, _, j in runs if r == rid)
        return [{"jobs": jobs}]


def run_gate(api: FakeApi, *extra: str) -> int:
    argv = ["--lanes", LANES, "--repo", REPO, "--sha", TAG_SHA, "--tag", TAG, *extra]
    return gate.main(argv, git=FakeGit(), api=api)


class TestParseLanes:
    def test_one_lane_per_line(self):
        assert gate.parse_lanes(LANES) == {
            "test": "Test Python",
            "connectors-standalone": "Connectors suite without the framework",
        }

    def test_a_prefix_may_hold_an_equals_sign_and_blank_lines_are_skipped(self):
        assert gate.parse_lanes("\n  a=b=c  \n\n") == {"a": "b=c"}

    @pytest.mark.parametrize("text", ["", "\n\n", "no-separator", "=prefix", "lane=", "a=x\na=y"])
    def test_a_lane_list_that_gates_on_something_else_is_refused(self, text):
        with pytest.raises(gate.GateError):
            gate.parse_lanes(text)


class TestSameTreeCandidates:
    def test_only_commits_carrying_the_tagged_tree(self):
        assert gate.same_tree_candidates(TAG_SHA, FakeGit()) == [TAG_SHA, MERGE_SHA]


class TestMissingLanes:
    def test_every_matrix_cell_must_succeed(self):
        jobs = [*GREEN_JOBS[:1], job("Test Python 3.12 on ubuntu-latest (rest-b)", "failure")]
        lanes = {"test": "Test Python"}
        assert gate.missing_lanes(jobs, lanes) == {"test": "failure, success"}

    def test_a_skipped_lane_did_not_run(self):
        jobs = [job("Connectors suite without the framework", "skipped")]
        lanes = {"connectors-standalone": "Connectors suite"}
        assert gate.missing_lanes(jobs, lanes) == {"connectors-standalone": "skipped"}

    def test_an_absent_lane_is_named(self):
        assert gate.missing_lanes([], {"lint": "Lint"}) == {"lint": "no such job in the run"}


class TestMain:
    def test_a_green_run_on_the_tag_passes(self, capsys):
        assert run_gate(FakeApi({TAG_SHA: [(1, "success", GREEN_JOBS)]})) == 0
        assert "https://ci/1" in capsys.readouterr().out

    def test_a_green_run_on_a_same_tree_merge_commit_passes(self):
        assert run_gate(FakeApi({MERGE_SHA: [(2, "success", GREEN_JOBS)]})) == 0

    def test_a_green_run_on_a_different_tree_does_not_count(self, capsys):
        api = FakeApi({OTHER_SHA: [(3, "success", GREEN_JOBS)]})
        assert run_gate(api) == 1
        assert not any(OTHER_SHA in path for path in api.paths)
        assert "no successful ci.yml run on any of them" in capsys.readouterr().out

    def test_a_later_empty_rerun_does_not_hide_the_run_that_did_the_work(self):
        skipped = [job(j["name"], "skipped") for j in GREEN_JOBS]
        api = FakeApi({TAG_SHA: [(4, "success", skipped), (5, "success", GREEN_JOBS)]})
        assert run_gate(api) == 0

    def test_a_failed_run_is_not_read(self):
        api = FakeApi({TAG_SHA: [(6, "failure", GREEN_JOBS)]})
        assert run_gate(api) == 1
        assert not any("/runs/6/jobs" in path for path in api.paths)

    def test_a_refusal_names_the_lanes_and_the_remedy(self, capsys):
        api = FakeApi({TAG_SHA: [(7, "success", GREEN_JOBS[:2])]})
        assert run_gate(api) == 1
        out = capsys.readouterr().out
        assert f"::error::no CI run proves the required lanes on the tree tagged {TAG}" in out
        assert "connectors-standalone: no such job in the run" in out
        assert f"  gh workflow run ci.yml --ref {TAG}" in out
        assert f"git push --delete origin {TAG} && git push origin {TAG}" in out

    def test_hints_replace_the_plain_dispatch(self, capsys):
        assert run_gate(FakeApi({}), "--hint", "add the full-ci label") == 1
        out = capsys.readouterr().out
        assert "  add the full-ci label" in out
        assert f"gh workflow run ci.yml --ref {TAG}\n" not in out

    def test_a_malformed_lane_list_is_an_environment_error(self):
        argv = ["--lanes", "nonsense", "--repo", REPO, "--sha", TAG_SHA, "--tag", TAG]
        assert gate.main(argv, git=FakeGit(), api=FakeApi({})) == 2

    def test_missing_context_is_an_environment_error(self, monkeypatch):
        for name in ("REPO", "TAG_SHA", "TAG_NAME"):
            monkeypatch.delenv(name, raising=False)
        assert gate.main(["--lanes", LANES], git=FakeGit(), api=FakeApi({})) == 2

    def test_the_context_defaults_to_the_workflow_env(self, monkeypatch):
        monkeypatch.setenv("REPO", REPO)
        monkeypatch.setenv("TAG_SHA", TAG_SHA)
        monkeypatch.setenv("TAG_NAME", TAG)
        api = FakeApi({TAG_SHA: [(8, "success", GREEN_JOBS)]})
        assert gate.main(["--lanes", LANES], git=FakeGit(), api=api) == 0
