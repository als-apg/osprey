"""Assertions against the parsed `.github/workflows/docs.yml`, pinning the
versioned-publishing wiring the workflow is now built around.

The rules being enforced here, and why each one exists:

1. **No invocation hardcodes which copy of the helper it runs.** All three
   spell the path resolved once by the ``Resolve publishing helper`` step,
   which prefers ``.docs-ci/scripts/ci/docs_publish.py`` and falls back to the
   in-tree ``scripts/ci/docs_publish.py`` only when that is absent. Hardcoding
   the in-tree path would break a tag build, whose tree predates the helper;
   hardcoding the ``.docs-ci`` path would break the one pull request that adds
   the helper, where the default branch does not carry it yet. The preference
   order is what keeps a fix to the helper applying to a rebuild of an old tag.
2. **That second checkout happens before the plan step**, is sparse, and lands
   in ``.docs-ci``: the resolve step is the first thing that needs it.
3. **No hand-rolled copying.** The old workflow assembled the site with
   ``cp -r docs/build/html``; all of that now lives in the helper's ``stage``
   subcommand, which is unit-tested. A stray ``cp -r`` would silently bypass it.
4. **Fail-closed.** No step carries ``continue-on-error`` and no ``run:``
   contains ``|| true``. Publishing is destructive — the deployed tree replaces
   gh-pages wholesale — so a swallowed failure can delete every published
   version. It must abort the job instead.
5. **Everything that touches the deployment tree is gated** on
   ``steps.plan.outputs.deploy == 'true'``. The helper decides once whether this
   ref deploys at all; no step may act on its own opinion.
6. **``--allow-empty-site`` is only ever passed conditionally**, from the
   first-deploy probe. Passing it unconditionally would let a failed fetch of
   the existing site publish an empty tree.
7. **The plan step runs before uv is installed** — the helper is stdlib-only, so
   the deploy decision is available even when the docs build fails.
8. **Deploying runs are serialised, pull requests are not**, via the
   ``docs-${{ ... }}`` concurrency group.
9. **The triggers are unchanged** — push to main and to ``v*`` tags, pull
   requests, and a ``workflow_dispatch`` carrying an optional ``tag`` input.
10. **The three helper invocations cannot drift.** Their shared inputs are
    hoisted into a job-level ``env:`` block and referenced as shell variables,
    so the plan and stage calls describe the same build by construction.
11. **The tag list the helper reads is the real one.** ``plan`` and ``versions``
    shell out to ``git tag`` in their working directory, so the build checkout
    keeps ``fetch-depth: 0`` and no helper step sets ``working-directory``.
    Either slip leaves the helper reading a truncated tag list — from which the
    newest release simply looks absent, so no build claims the root and the
    switcher loses its archive, with nothing failing.
12. **The first-deploy probe fails closed.** ``git ls-remote --exit-code``
    distinguishes "no gh-pages branch" (status 2) from any other failure, and
    only that one arm may set ``first_deploy=true``. Any other status exits with
    it, because ``--allow-empty-site`` on a merely-unreachable remote publishes
    a tree containing this build alone and deletes every other version.
13. **The switcher is written after the tree is staged.** A release build's root
    refresh clears the root before copying, so a ``versions.json`` written first
    would be deleted on exactly the builds that need it.

Everything loads docs.yml with ``yaml.safe_load`` and asserts against the parsed
structure, so re-flowing YAML style cannot fool a check. YAML 1.1 parses the
bare ``on:`` workflow-trigger key to the Python boolean ``True``; this module
never indexes ``workflow["on"]`` and uses ``workflow[True]`` instead.

Every positive assertion is paired with a mutation test: a fresh, in-memory
copy of the parsed workflow reintroduces exactly the bug the assertion exists to
catch, and the same assertion must then fail. docs.yml itself is never edited.

One check leaves YAML behind entirely: the workflow reads
``steps.plan.outputs.*``, but the keys that actually exist are decided by the
helper's ``plan`` subcommand in Python. That check runs the helper in a
subprocess and cross-references the two. Note which copy of the helper it runs:
the workflow prefers the *default-branch* copy out of ``.docs-ci`` wherever it
exists, while this check runs the in-tree ``scripts/ci/docs_publish.py``.
That is the point — it pins the copy a pull request proposes against the
workflow that pull request also proposes, so a rename of a plan output is caught
in review. It is not a check of what production runs today: a helper change is
exercised by the real docs job only once it has landed on the default branch.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_YML = REPO_ROOT / ".github" / "workflows" / "docs.yml"
#: The in-tree helper, used only to run it directly in the output-key test. No
#: *invocation* may spell this path; only the resolve step, as its fallback.
HELPER_SOURCE = REPO_ROOT / "scripts" / "ci" / "docs_publish.py"

BUILD_JOB = "build"

CHECKOUT_STEP = "Checkout repository"
PLAN_STEP = "Plan the deploy"
BUILD_DOCS_STEP = "Build documentation"
FETCH_STEP = "Fetch published site"
STAGE_STEP = "Stage deployment tree"
VERSIONS_STEP = "Write version switcher index"
DEPLOY_STEP = "Deploy to GitHub Pages"

#: The only spelling of the helper an *invocation* may use: the path the
#: resolve step chose. A literal in its place would pin one of the two
#: checkouts and break the builds the other one exists to serve.
HELPER_REF = "${{ steps.helper.outputs.path }}"
#: The resolve step's two candidates, in the order it must try them.
HELPER_PATH = ".docs-ci/scripts/ci/docs_publish.py"
HELPER_IN_TREE_PATH = "scripts/ci/docs_publish.py"
#: The step that picks between them, and the id the invocations read it from.
HELPER_RESOLVE_STEP = "Resolve publishing helper"
HELPER_STEP_ID = "helper"
#: Matches any path ending in the helper's filename, so a bare or otherwise
#: re-rooted invocation is found rather than skipped. A plain ``in`` check for
#: HELPER_REF cannot do this: it reports "absent", not "spelled wrong".
_HELPER_INVOCATION_RE = re.compile(r"[\w./-]*docs_publish\.py")
#: Where the sparse checkout of the default branch lands, and the one directory
#: it needs to bring with it.
HELPER_CHECKOUT_PATH = ".docs-ci"
HELPER_SPARSE_DIR = "scripts/ci"

CHECKOUT_ACTION = "actions/checkout"
SETUP_UV_ACTION = "astral-sh/setup-uv"
DEPLOY_ACTION = "peaceiris/actions-gh-pages"

#: ``fetch-depth: 0`` on the build checkout — the helper shells out to
#: ``git tag`` there, and a shallow checkout carries no tags at all.
FETCH_DEPTH_KEY = "fetch-depth"
FULL_HISTORY = 0
#: A helper step must run in the build checkout, so it may never set this.
WORKING_DIRECTORY_KEY = "working-directory"

#: The probe that distinguishes "no gh-pages branch" from any other failure.
LS_REMOTE_PROBE = "git ls-remote --exit-code"
#: `git ls-remote --exit-code` exits 2 when the ref is simply absent.
MISSING_BRANCH_ARM = "2)"
CATCH_ALL_ARM = "*)"
FIRST_DEPLOY_TRUE = 'echo "first_deploy=true" >> "$GITHUB_OUTPUT"'
RETHROW = 'exit "$rc"'

#: The single gate every deployment-touching step must carry.
DEPLOY_GATE = "steps.plan.outputs.deploy == 'true'"
#: Substrings in a ``run:`` that mean the step handles the publishable tree.
STAGING_TOKENS = ("deployment", "gh-pages-existing")

#: Exact strings — the concurrency behaviour is the assertion, not an
#: approximation of it. Pull requests key on ``github.run_id`` so they get a
#: group per run; everything else collapses onto ``docs-deploy``.
CONCURRENCY_GROUP = "docs-${{ github.event_name == 'pull_request' && github.run_id || 'deploy' }}"

ALLOW_EMPTY_FLAG = "--allow-empty-site"
ALLOW_EMPTY_EXPR = "${{ steps.pages.outputs.first_deploy == 'true' && '--allow-empty-site' || '' }}"

#: Job-level env → the shell spelling each helper flag must use.
HELPER_INPUT_FLAGS = {
    "--ref": '"$DOCS_REF"',
    "--event": '"$DOCS_EVENT"',
    "--input-tag": '"$DOCS_INPUT_TAG"',
}
JOB_ENV_EXPRESSIONS = {
    "DOCS_REF": "${{ github.ref }}",
    "DOCS_EVENT": "${{ github.event_name }}",
    "DOCS_INPUT_TAG": "${{ github.event.inputs.tag || '' }}",
}

FORBIDDEN_COPY = "cp -r docs/build/html"
FORBIDDEN_SWALLOW = "|| true"
CONTINUE_ON_ERROR = "continue-on-error"

#: ``steps.plan.outputs.<key>`` references anywhere in the workflow.
_PLAN_OUTPUT_REF_RE = re.compile(r"steps\.plan\.outputs\.([A-Za-z_][A-Za-z0-9_]*)")


def _load_workflow() -> dict[str, Any]:
    with DOCS_YML.open() as f:
        loaded = yaml.safe_load(f)
    assert loaded is not None, f"{DOCS_YML} parsed to None"
    return loaded


@pytest.fixture()
def workflow() -> dict[str, Any]:
    return _load_workflow()


def _job(wf: dict[str, Any]) -> dict[str, Any]:
    return wf["jobs"][BUILD_JOB]


def _steps(wf: dict[str, Any]) -> list[dict[str, Any]]:
    return _job(wf)["steps"]


def _index_of_named_step(wf: dict[str, Any], step_name: str) -> int:
    for index, step in enumerate(_steps(wf)):
        if step.get("name") == step_name:
            return index
    raise AssertionError(f"job '{BUILD_JOB}' has no step named '{step_name}'")


def _find_named_step(wf: dict[str, Any], step_name: str) -> dict[str, Any]:
    return _steps(wf)[_index_of_named_step(wf, step_name)]


def _index_of_action(wf: dict[str, Any], action_prefix: str) -> int:
    for index, step in enumerate(_steps(wf)):
        if str(step.get("uses", "")).startswith(action_prefix):
            return index
    raise AssertionError(f"job '{BUILD_JOB}' has no step using '{action_prefix}'")


def _index_of_helper_checkout(wf: dict[str, Any]) -> int:
    """Locate the sparse helper checkout by what it *does* (`path: .docs-ci`)
    rather than by its name, so renaming the step cannot hide its removal."""
    for index, step in enumerate(_steps(wf)):
        if (step.get("with") or {}).get("path") == HELPER_CHECKOUT_PATH:
            return index
    raise AssertionError(f"no checkout step lands in '{HELPER_CHECKOUT_PATH}'")


def _run_texts(wf: dict[str, Any]) -> list[str]:
    return [str(step["run"]) for step in _steps(wf) if "run" in step]


def _helper_invocation_steps(wf: dict[str, Any]) -> list[dict[str, Any]]:
    """The steps that *run* the helper — located by the resolved path they
    spell, since after resolution no invocation names the file itself."""
    return [step for step in _steps(wf) if HELPER_REF in str(step.get("run", ""))]


def _resolve_step(wf: dict[str, Any]) -> dict[str, Any]:
    """Locate the resolve step by the id the invocations read, not by name:
    renaming it is harmless, but changing the id silently empties every
    invocation's path into the string ``python3 ""``."""
    for step in _steps(wf):
        if step.get("id") == HELPER_STEP_ID:
            return step
    raise AssertionError(f"no step carries id '{HELPER_STEP_ID}'")


def _helper_steps(wf: dict[str, Any]) -> list[dict[str, Any]]:
    """Every step that touches the helper: the one that resolves it plus the
    three that run it. All of them read paths relative to the build checkout."""
    return [_resolve_step(wf), *_helper_invocation_steps(wf)]


def _helper_invocation_runs(wf: dict[str, Any]) -> list[str]:
    return [str(step["run"]) for step in _helper_invocation_steps(wf)]


def _serialized(wf: dict[str, Any]) -> str:
    """The whole job as JSON, so a search also reaches step-level ``if:``,
    ``env:`` and ``with:`` values, not just top-level keys."""
    return json.dumps(wf)


# ---------------------------------------------------------------------------
# (1) no invocation hardcodes which copy of the helper it runs
# ---------------------------------------------------------------------------


def _assert_helper_paths(wf: dict[str, Any]) -> None:
    """Only the resolve step may name a copy of the helper; the three
    invocations must defer to whichever one it picked."""
    runs = _helper_invocation_runs(wf)
    assert runs, f"no step invokes the helper via '{HELPER_REF}'"
    resolve_run = str(_resolve_step(wf).get("run", ""))
    for step in _steps(wf):
        run = str(step.get("run", ""))
        if not run or run == resolve_run:
            continue
        hardcoded = _HELPER_INVOCATION_RE.findall(run)
        assert not hardcoded, (
            f"step {step.get('name')!r} hardcodes the helper as {hardcoded}; "
            f"invocations must spell '{HELPER_REF}' so the resolve step decides"
        )


def test_invocations_defer_to_the_resolved_helper_path(workflow: dict[str, Any]) -> None:
    _assert_helper_paths(workflow)


def test_invocations_defer_to_the_resolved_helper_path__mutation_uses_in_tree_path() -> None:
    """Pinning an invocation to the in-tree path must fail — that path is
    absent from any tag tree cut before the helper landed."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, STAGE_STEP)
    step["run"] = step["run"].replace(HELPER_REF, HELPER_IN_TREE_PATH)
    with pytest.raises(AssertionError):
        _assert_helper_paths(mutated)


def test_invocations_defer_to_the_resolved_helper_path__mutation_uses_docs_ci_path() -> None:
    """Pinning it to the ``.docs-ci`` path must fail too. That copy is missing
    on the one pull request that adds the helper, which is the entire reason
    the resolve step exists."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, STAGE_STEP)
    step["run"] = step["run"].replace(HELPER_REF, HELPER_PATH)
    with pytest.raises(AssertionError):
        _assert_helper_paths(mutated)


def _assert_resolve_step_prefers_the_default_branch_copy(wf: dict[str, Any]) -> None:
    """The ``.docs-ci`` copy must be tested first and used when present, so a
    fix to the helper reaches a rebuild of an old tag rather than being
    shadowed by that tag's own contemporaneous copy."""
    run = str(_resolve_step(wf).get("run", ""))
    # Matched as the assignment rather than the bare path: the in-tree path is
    # a suffix of the .docs-ci one, so a plain `in` check for it is satisfied by
    # a step that only ever mentions .docs-ci.
    picks_default_branch = f"path={HELPER_PATH}"
    picks_in_tree = f"path={HELPER_IN_TREE_PATH}"
    assert picks_default_branch in run, f"resolve step never selects '{HELPER_PATH}'"
    assert picks_in_tree in run, (
        f"resolve step has no fallback to '{HELPER_IN_TREE_PATH}'; the pull "
        "request that adds the helper could never find it"
    )
    assert run.index(HELPER_PATH) < run.index(picks_in_tree), (
        "the in-tree copy is selected before the .docs-ci copy is tested; the "
        "default-branch copy must win wherever it exists"
    )


def test_resolve_step_prefers_the_default_branch_copy(workflow: dict[str, Any]) -> None:
    _assert_resolve_step_prefers_the_default_branch_copy(workflow)


def test_resolve_step_prefers_the_default_branch_copy__mutation_drops_fallback() -> None:
    """Without the fallback arm the helper can never bootstrap onto main."""
    mutated = copy.deepcopy(_load_workflow())
    step = _resolve_step(mutated)
    step["run"] = f'echo "path={HELPER_PATH}" >> "$GITHUB_OUTPUT"'
    with pytest.raises(AssertionError):
        _assert_resolve_step_prefers_the_default_branch_copy(mutated)


def test_resolve_step_prefers_the_default_branch_copy__mutation_inverts_preference() -> None:
    """Preferring the in-tree copy would stop a helper fix from reaching a
    rebuild of an old tag — that tag would run its own stale copy."""
    mutated = copy.deepcopy(_load_workflow())
    step = _resolve_step(mutated)
    step["run"] = (
        f"if [ -f {HELPER_IN_TREE_PATH} ]; then\n"
        f'  echo "path={HELPER_IN_TREE_PATH}" >> "$GITHUB_OUTPUT"\n'
        f"else\n"
        f'  echo "path={HELPER_PATH}" >> "$GITHUB_OUTPUT"\n'
        f"fi\n"
    )
    with pytest.raises(AssertionError):
        _assert_resolve_step_prefers_the_default_branch_copy(mutated)


def _assert_resolve_step_precedes_the_plan_step(wf: dict[str, Any]) -> None:
    resolve_index = _steps(wf).index(_resolve_step(wf))
    assert resolve_index > _index_of_helper_checkout(wf), (
        "the helper path is resolved before the .docs-ci checkout that supplies it"
    )
    assert resolve_index < _index_of_named_step(wf, PLAN_STEP), (
        "the plan step runs before the path it invokes has been resolved"
    )


def test_resolve_step_sits_between_the_helper_checkout_and_the_plan(
    workflow: dict[str, Any],
) -> None:
    _assert_resolve_step_precedes_the_plan_step(workflow)


def test_resolve_step_sits_between_the_helper_checkout_and_the_plan__mutation_reorders() -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _steps(mutated)
    resolve = steps.pop(steps.index(_resolve_step(mutated)))
    steps.insert(_index_of_named_step(mutated, PLAN_STEP) + 1, resolve)
    with pytest.raises(AssertionError):
        _assert_resolve_step_precedes_the_plan_step(mutated)


def test_helper_invocation_count_matches_the_three_subcommands(workflow: dict[str, Any]) -> None:
    """plan, stage, versions — exactly three calls, no more."""
    assert len(_helper_invocation_runs(workflow)) == 3


def test_helper_invocation_count_matches_the_three_subcommands__mutation_adds_call() -> None:
    mutated = copy.deepcopy(_load_workflow())
    mutated["jobs"][BUILD_JOB]["steps"].append(
        {"name": "extra", "run": f'python3 "{HELPER_REF}" plan'}
    )
    with pytest.raises(AssertionError):
        assert len(_helper_invocation_runs(mutated)) == 3


# ---------------------------------------------------------------------------
# (2) the sparse helper checkout exists and precedes the plan step
# ---------------------------------------------------------------------------


def _assert_helper_checkout_precedes_plan(wf: dict[str, Any]) -> None:
    checkout_index = _index_of_helper_checkout(wf)
    step = _steps(wf)[checkout_index]
    sparse = str((step.get("with") or {}).get("sparse-checkout", ""))
    assert HELPER_SPARSE_DIR in sparse, (
        f"helper checkout must sparse-fetch '{HELPER_SPARSE_DIR}', got {sparse!r}"
    )
    assert checkout_index < _index_of_named_step(wf, PLAN_STEP), (
        "the helper must be checked out before the plan step that runs it"
    )


def test_helper_checkout_is_sparse_and_precedes_the_plan_step(workflow: dict[str, Any]) -> None:
    _assert_helper_checkout_precedes_plan(workflow)


def test_helper_checkout_is_sparse_and_precedes_the_plan_step__mutation_reorders() -> None:
    """Moving the helper checkout after the plan step must fail: the plan step
    would run a path that does not exist yet."""
    mutated = copy.deepcopy(_load_workflow())
    steps = _steps(mutated)
    checkout = steps.pop(_index_of_helper_checkout(mutated))
    steps.insert(_index_of_named_step(mutated, PLAN_STEP) + 1, checkout)
    with pytest.raises(AssertionError):
        _assert_helper_checkout_precedes_plan(mutated)


def test_helper_checkout_is_sparse_and_precedes_the_plan_step__mutation_drops_step() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _steps(mutated)[_index_of_helper_checkout(mutated)]
    with pytest.raises(AssertionError):
        _assert_helper_checkout_precedes_plan(mutated)


def test_helper_checkout_is_sparse_and_precedes_the_plan_step__mutation_drops_sparse() -> None:
    """A full checkout of the default branch would still work, but silently
    stops being the narrow, obviously-safe fetch the design calls for."""
    mutated = copy.deepcopy(_load_workflow())
    del _steps(mutated)[_index_of_helper_checkout(mutated)]["with"]["sparse-checkout"]
    with pytest.raises(AssertionError):
        _assert_helper_checkout_precedes_plan(mutated)


def test_helper_checkout_targets_the_default_branch(workflow: dict[str, Any]) -> None:
    """It must NOT inherit the build's own ref — that is the whole point."""
    step = _steps(workflow)[_index_of_helper_checkout(workflow)]
    assert "github.event.repository.default_branch" in str(step["with"]["ref"])


def test_helper_checkout_targets_the_default_branch__mutation_uses_build_ref() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _steps(mutated)[_index_of_helper_checkout(mutated)]
    step["with"]["ref"] = "${{ github.ref }}"
    with pytest.raises(AssertionError):
        assert "github.event.repository.default_branch" in str(step["with"]["ref"])


# ---------------------------------------------------------------------------
# (3) no hand-rolled copying of the built site
# ---------------------------------------------------------------------------


def _assert_no_manual_copy(wf: dict[str, Any]) -> None:
    for run in _run_texts(wf):
        assert FORBIDDEN_COPY not in run, f"'{FORBIDDEN_COPY}' bypasses the tested stage subcommand"


def test_no_step_copies_the_built_site_by_hand(workflow: dict[str, Any]) -> None:
    _assert_no_manual_copy(workflow)


def test_no_step_copies_the_built_site_by_hand__mutation_injects_copy() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, STAGE_STEP)
    step["run"] = step["run"] + "\ncp -r docs/build/html deployment/\n"
    with pytest.raises(AssertionError):
        _assert_no_manual_copy(mutated)


# ---------------------------------------------------------------------------
# (4) fail-closed: nothing swallows a failure
# ---------------------------------------------------------------------------


def _assert_fails_closed(wf: dict[str, Any]) -> None:
    assert CONTINUE_ON_ERROR not in _job(wf), "the job itself must not continue on error"
    for step in _steps(wf):
        assert CONTINUE_ON_ERROR not in step, f"step {step.get('name')!r} swallows its own failure"
    for run in _run_texts(wf):
        assert FORBIDDEN_SWALLOW not in run, "'|| true' hides a failure that must abort the job"


def test_no_step_swallows_failures(workflow: dict[str, Any]) -> None:
    _assert_fails_closed(workflow)


def test_no_step_swallows_failures__mutation_adds_continue_on_error() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _find_named_step(mutated, FETCH_STEP)[CONTINUE_ON_ERROR] = True
    with pytest.raises(AssertionError):
        _assert_fails_closed(mutated)


def test_no_step_swallows_failures__mutation_adds_or_true() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, FETCH_STEP)
    step["run"] = step["run"].replace("set -e", "set -e || true")
    with pytest.raises(AssertionError):
        _assert_fails_closed(mutated)


# ---------------------------------------------------------------------------
# (5) every deployment-touching step is gated on the planned decision
# ---------------------------------------------------------------------------


def _deployment_touching_steps(wf: dict[str, Any]) -> list[dict[str, Any]]:
    touching = []
    for step in _steps(wf):
        run = str(step.get("run", ""))
        uses = str(step.get("uses", ""))
        if any(token in run for token in STAGING_TOKENS) or uses.startswith(DEPLOY_ACTION):
            touching.append(step)
    return touching


def _assert_deployment_steps_are_gated(wf: dict[str, Any]) -> None:
    touching = _deployment_touching_steps(wf)
    # Under-discovery would make this vacuously true: fetch, stage, versions and
    # the deploy action are the four that must be found.
    assert len(touching) == 4, (
        f"expected 4 deployment-touching steps, found {[s.get('name') for s in touching]}"
    )
    for step in touching:
        assert DEPLOY_GATE in str(step.get("if", "")), (
            f"step {step.get('name')!r} touches the deployment tree without the deploy gate"
        )


def test_deployment_steps_are_gated_on_the_plan(workflow: dict[str, Any]) -> None:
    _assert_deployment_steps_are_gated(workflow)


def test_deployment_steps_are_gated_on_the_plan__mutation_drops_one_gate() -> None:
    """Dropping a single ``if:`` must fail — a pull request would then publish."""
    mutated = copy.deepcopy(_load_workflow())
    del _find_named_step(mutated, STAGE_STEP)["if"]
    with pytest.raises(AssertionError):
        _assert_deployment_steps_are_gated(mutated)


def test_deployment_steps_are_gated_on_the_plan__mutation_weakens_one_gate() -> None:
    """Replacing the gate with the old hand-rolled ref test must fail too: the
    helper is the only thing allowed to decide whether a ref deploys."""
    mutated = copy.deepcopy(_load_workflow())
    _find_named_step(mutated, DEPLOY_STEP)["if"] = "github.ref == 'refs/heads/main'"
    with pytest.raises(AssertionError):
        _assert_deployment_steps_are_gated(mutated)


def test_deploy_step_publishes_the_staged_tree(workflow: dict[str, Any]) -> None:
    step = _find_named_step(workflow, DEPLOY_STEP)
    assert str(step["uses"]).startswith(DEPLOY_ACTION)
    assert step["with"]["publish_dir"] == "./deployment"


def test_deploy_step_publishes_the_staged_tree__mutation_publishes_build_dir() -> None:
    """Publishing the raw Sphinx output would flatten the whole version tree."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, DEPLOY_STEP)
    step["with"]["publish_dir"] = "./docs/build/html"
    with pytest.raises(AssertionError):
        assert step["with"]["publish_dir"] == "./deployment"


# ---------------------------------------------------------------------------
# (6) --allow-empty-site is only ever passed conditionally
# ---------------------------------------------------------------------------


def _assert_allow_empty_is_conditional(wf: dict[str, Any]) -> None:
    joined = "\n".join(_run_texts(wf))
    flag_uses = joined.count(ALLOW_EMPTY_FLAG)
    guarded_uses = joined.count(ALLOW_EMPTY_EXPR)
    assert guarded_uses == 1, f"expected exactly one guarded use, found {guarded_uses}"
    # The expression itself contains the flag once, so the counts must agree.
    assert flag_uses == guarded_uses, (
        f"{ALLOW_EMPTY_FLAG} appears {flag_uses} times but only {guarded_uses} are guarded"
    )


def test_allow_empty_site_is_only_passed_on_a_first_deploy(workflow: dict[str, Any]) -> None:
    _assert_allow_empty_is_conditional(workflow)


def test_allow_empty_site_is_only_passed_on_a_first_deploy__mutation_adds_bare_flag() -> None:
    """An unconditional flag would let a failed site fetch publish an empty
    tree, wiping every version already online."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, STAGE_STEP)
    step["run"] = step["run"] + f" {ALLOW_EMPTY_FLAG}"
    with pytest.raises(AssertionError):
        _assert_allow_empty_is_conditional(mutated)


def test_first_deploy_probe_is_produced_by_the_fetch_step(workflow: dict[str, Any]) -> None:
    """The guard expression reads ``steps.pages.outputs.first_deploy``; that id
    and that output have to be the fetch step's."""
    step = _find_named_step(workflow, FETCH_STEP)
    assert step["id"] == "pages"
    assert "first_deploy=true" in step["run"]
    assert "first_deploy=false" in step["run"]


def test_first_deploy_probe_is_produced_by_the_fetch_step__mutation_renames_id() -> None:
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, FETCH_STEP)
    step["id"] = "gh_pages"
    with pytest.raises(AssertionError):
        assert step["id"] == "pages"


# ---------------------------------------------------------------------------
# (7) the stdlib-only plan step runs before uv is installed
# ---------------------------------------------------------------------------


def _assert_plan_precedes_uv(wf: dict[str, Any]) -> None:
    assert _index_of_named_step(wf, PLAN_STEP) < _index_of_action(wf, SETUP_UV_ACTION), (
        "the plan step is stdlib-only and must not wait on the toolchain install"
    )


def test_plan_step_runs_before_uv_is_installed(workflow: dict[str, Any]) -> None:
    _assert_plan_precedes_uv(workflow)


def test_plan_step_runs_before_uv_is_installed__mutation_reorders() -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _steps(mutated)
    plan = steps.pop(_index_of_named_step(mutated, PLAN_STEP))
    steps.insert(_index_of_named_step(mutated, BUILD_DOCS_STEP) + 1, plan)
    with pytest.raises(AssertionError):
        _assert_plan_precedes_uv(mutated)


def test_plan_step_exposes_its_outputs_under_the_expected_id(workflow: dict[str, Any]) -> None:
    step = _find_named_step(workflow, PLAN_STEP)
    assert step["id"] == "plan"
    assert '>> "$GITHUB_OUTPUT"' in step["run"]


def test_plan_step_exposes_its_outputs_under_the_expected_id__mutation_drops_redirect() -> None:
    """Without the redirect the helper prints to the log and every gated step
    silently evaluates to false."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, PLAN_STEP)
    step["run"] = step["run"].replace('>> "$GITHUB_OUTPUT"', "")
    with pytest.raises(AssertionError):
        assert '>> "$GITHUB_OUTPUT"' in step["run"]


# ---------------------------------------------------------------------------
# (8) deploying runs serialise; pull requests get a group per run
# ---------------------------------------------------------------------------


def _assert_concurrency(wf: dict[str, Any]) -> None:
    concurrency = wf["concurrency"]
    assert concurrency["group"] == CONCURRENCY_GROUP, (
        "pull requests must key on github.run_id so they never queue behind a deploy"
    )
    assert concurrency["cancel-in-progress"] is False, (
        "a deploy that has started must always finish; cancelling it mid-push "
        "can leave gh-pages half-written"
    )


def test_concurrency_separates_pull_requests_from_deploys(workflow: dict[str, Any]) -> None:
    _assert_concurrency(workflow)


def test_concurrency_separates_pull_requests_from_deploys__mutation_shares_one_group() -> None:
    """The old ``"pages"`` group put PR builds in the deploy queue."""
    mutated = copy.deepcopy(_load_workflow())
    mutated["concurrency"]["group"] = "pages"
    with pytest.raises(AssertionError):
        _assert_concurrency(mutated)


def test_concurrency_separates_pull_requests_from_deploys__mutation_flips_cancel() -> None:
    mutated = copy.deepcopy(_load_workflow())
    mutated["concurrency"]["cancel-in-progress"] = True
    with pytest.raises(AssertionError):
        _assert_concurrency(mutated)


# ---------------------------------------------------------------------------
# (9) triggers are unchanged
# ---------------------------------------------------------------------------


def _triggers(wf: dict[str, Any]) -> dict[str, Any]:
    # YAML 1.1 parses the bare `on:` key to the boolean True, so it is read as
    # wf[True] rather than wf["on"].
    return wf[True]


def _assert_triggers(wf: dict[str, Any]) -> None:
    triggers = _triggers(wf)
    assert "main" in triggers["push"]["branches"]
    assert "v*" in triggers["push"]["tags"]
    assert "pull_request" in triggers
    assert "tag" in triggers["workflow_dispatch"]["inputs"]
    assert triggers["workflow_dispatch"]["inputs"]["tag"]["required"] is False


def test_triggers_cover_main_tags_pull_requests_and_dispatch(workflow: dict[str, Any]) -> None:
    _assert_triggers(workflow)


def test_triggers_cover_main_tags_pull_requests_and_dispatch__mutation_drops_tag_input() -> None:
    """Without the ``tag`` input a superseded tag deploy cannot be re-run."""
    mutated = copy.deepcopy(_load_workflow())
    del _triggers(mutated)["workflow_dispatch"]["inputs"]["tag"]
    with pytest.raises(AssertionError):
        _assert_triggers(mutated)


def test_triggers_cover_main_tags_pull_requests_and_dispatch__mutation_drops_tag_trigger() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _triggers(mutated)["push"]["tags"] = ["never-matches-*"]
    with pytest.raises(AssertionError):
        _assert_triggers(mutated)


# ---------------------------------------------------------------------------
# (10) the helper's shared inputs are hoisted so no call can drift
# ---------------------------------------------------------------------------


def _assert_job_env_hoists_inputs(wf: dict[str, Any]) -> None:
    env = _job(wf).get("env") or {}
    for key, expression in JOB_ENV_EXPRESSIONS.items():
        assert env.get(key) == expression, f"job env {key} is {env.get(key)!r}, want {expression!r}"


def test_job_env_hoists_the_shared_helper_inputs(workflow: dict[str, Any]) -> None:
    _assert_job_env_hoists_inputs(workflow)


def test_job_env_hoists_the_shared_helper_inputs__mutation_drops_one_key() -> None:
    mutated = copy.deepcopy(_load_workflow())
    del _job(mutated)["env"]["DOCS_INPUT_TAG"]
    with pytest.raises(AssertionError):
        _assert_job_env_hoists_inputs(mutated)


def _assert_helper_calls_share_their_inputs(wf: dict[str, Any]) -> None:
    """Any invocation that describes the build must describe all of it.

    ``versions`` needs none of the trio — it reads the staged tree — so the rule
    is conditional: a call that passes one of the three flags must pass all
    three, and always through the hoisted env vars rather than inline
    expressions that could drift apart.
    """
    describing = [
        run
        for run in _helper_invocation_runs(wf)
        if any(flag in run for flag in HELPER_INPUT_FLAGS)
    ]
    assert len(describing) == 2, (
        f"expected plan and stage to describe the build, found {len(describing)} calls"
    )
    for run in describing:
        for flag, value in HELPER_INPUT_FLAGS.items():
            assert f"{flag} {value}" in run, f"invocation is missing `{flag} {value}`"


def test_plan_and_stage_pass_identical_build_inputs(workflow: dict[str, Any]) -> None:
    _assert_helper_calls_share_their_inputs(workflow)


def test_plan_and_stage_pass_identical_build_inputs__mutation_drops_input_tag() -> None:
    """Dropping ``--input-tag`` from stage makes it stage a different version
    than plan decided on, for exactly the dispatch case tags are re-run with."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, STAGE_STEP)
    step["run"] = step["run"].replace('--input-tag "$DOCS_INPUT_TAG" \\\n', "")
    assert "--input-tag" not in step["run"]
    with pytest.raises(AssertionError):
        _assert_helper_calls_share_their_inputs(mutated)


def test_plan_and_stage_pass_identical_build_inputs__mutation_inlines_expression() -> None:
    """Spelling one input as a raw expression re-opens the drift the env block
    closes, even though the value is the same today."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, STAGE_STEP)
    step["run"] = step["run"].replace('--ref "$DOCS_REF"', '--ref "${{ github.ref }}"')
    with pytest.raises(AssertionError):
        _assert_helper_calls_share_their_inputs(mutated)


# ---------------------------------------------------------------------------
# (11) the helper reads the repository's real tag list
# ---------------------------------------------------------------------------


def _assert_build_checkout_fetches_all_tags(wf: dict[str, Any]) -> None:
    """The build checkout is where `git tag` runs, so it must be unshallow."""
    step = _find_named_step(wf, CHECKOUT_STEP)
    assert _index_of_action(wf, CHECKOUT_ACTION) == _index_of_named_step(wf, CHECKOUT_STEP), (
        f"'{CHECKOUT_STEP}' must be the first {CHECKOUT_ACTION} step"
    )
    depth = (step.get("with") or {}).get(FETCH_DEPTH_KEY)
    assert depth == FULL_HISTORY, (
        f"build checkout has {FETCH_DEPTH_KEY}={depth!r}; the helper reads "
        "`git tag` here and a shallow checkout carries no tags"
    )


def test_build_checkout_fetches_the_full_tag_list(workflow: dict[str, Any]) -> None:
    _assert_build_checkout_fetches_all_tags(workflow)


def test_build_checkout_fetches_the_full_tag_list__mutation_drops_the_key() -> None:
    """Without it the checkout is shallow by default, and no tag is fetched."""
    mutated = copy.deepcopy(_load_workflow())
    del _find_named_step(mutated, CHECKOUT_STEP)["with"][FETCH_DEPTH_KEY]
    with pytest.raises(AssertionError):
        _assert_build_checkout_fetches_all_tags(mutated)


def test_build_checkout_fetches_the_full_tag_list__mutation_shallow_depth() -> None:
    """``fetch-depth: 1`` looks deliberate and is just as tag-less."""
    mutated = copy.deepcopy(_load_workflow())
    _find_named_step(mutated, CHECKOUT_STEP)["with"][FETCH_DEPTH_KEY] = 1
    with pytest.raises(AssertionError):
        _assert_build_checkout_fetches_all_tags(mutated)


def _assert_helper_steps_run_in_the_build_checkout(wf: dict[str, Any]) -> None:
    """No helper step may relocate its cwd away from the build checkout.

    ``.docs-ci`` is a sparse checkout of ``scripts/ci`` alone; running the
    helper from there would put `git tag` in a tree with a different tag list.
    The resolve step is held to the same rule: it probes for the ``.docs-ci``
    copy by a relative path, which answers differently from another cwd.
    """
    steps = _helper_steps(wf)
    assert steps, f"no step invokes the helper via '{HELPER_REF}'"
    for step in steps:
        assert WORKING_DIRECTORY_KEY not in step, (
            f"step {step.get('name')!r} sets {WORKING_DIRECTORY_KEY}; the helper "
            "must read `git tag` from the build checkout"
        )


def test_helper_steps_never_change_their_working_directory(workflow: dict[str, Any]) -> None:
    _assert_helper_steps_run_in_the_build_checkout(workflow)


def test_helper_steps_never_change_their_working_directory__mutation_sets_one() -> None:
    mutated = copy.deepcopy(_load_workflow())
    _find_named_step(mutated, PLAN_STEP)[WORKING_DIRECTORY_KEY] = HELPER_CHECKOUT_PATH
    with pytest.raises(AssertionError):
        _assert_helper_steps_run_in_the_build_checkout(mutated)


# ---------------------------------------------------------------------------
# (12) the first-deploy probe fails closed
# ---------------------------------------------------------------------------


def _assert_fetch_fails_closed(wf: dict[str, Any]) -> None:
    """Only a genuinely absent gh-pages branch may claim a first deploy.

    The arms are located by slicing the ``run:`` text on the ``2)`` and ``*)``
    case labels rather than by parsing shell: the check is about which arm one
    literal line sits in, and a two-index slice says that far more legibly than
    any tokenizer would.
    """
    run = str(_find_named_step(wf, FETCH_STEP)["run"])
    assert LS_REMOTE_PROBE in run, (
        f"the probe must be `{LS_REMOTE_PROBE}`; without --exit-code a missing "
        "branch is indistinguishable from a reachable empty one"
    )

    missing_at = run.find(MISSING_BRANCH_ARM)
    assert missing_at != -1, f"no '{MISSING_BRANCH_ARM}' case arm for a missing branch"
    catch_all_at = run.find(CATCH_ALL_ARM, missing_at)
    assert catch_all_at != -1, f"no '{CATCH_ALL_ARM}' case arm after '{MISSING_BRANCH_ARM}'"
    missing_arm = run[missing_at:catch_all_at]
    catch_all = run[catch_all_at:]

    assert run.count(FIRST_DEPLOY_TRUE) == 1, (
        "exactly one line may announce a first deploy; a second one elsewhere "
        "would let some other failure publish an empty site"
    )
    assert FIRST_DEPLOY_TRUE in missing_arm, (
        f"the first-deploy announcement must sit in the '{MISSING_BRANCH_ARM}' arm"
    )
    assert RETHROW in catch_all, (
        f"the '{CATCH_ALL_ARM}' arm must `{RETHROW}` rather than guess whether the site exists"
    )


def test_fetch_step_treats_only_a_missing_branch_as_a_first_deploy(
    workflow: dict[str, Any],
) -> None:
    _assert_fetch_fails_closed(workflow)


def test_fetch_step_treats_only_a_missing_branch_as_a_first_deploy__mutation_opens_catch_all() -> (
    None
):
    """Turning the catch-all into a first deploy is the destructive bug: a
    network or auth blip would publish a site holding this build alone."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, FETCH_STEP)
    run = str(step["run"])
    step["run"] = run[: run.find(CATCH_ALL_ARM)] + f"*)\n    {FIRST_DEPLOY_TRUE}\n    ;;\nesac\n"
    with pytest.raises(AssertionError):
        _assert_fetch_fails_closed(mutated)


def test_fetch_step_treats_only_a_missing_branch_as_a_first_deploy__mutation_drops_exit_code() -> (
    None
):
    """Without ``--exit-code`` the probe reports success for a branch that is
    not there, and the run proceeds to clone something that does not exist."""
    mutated = copy.deepcopy(_load_workflow())
    step = _find_named_step(mutated, FETCH_STEP)
    step["run"] = str(step["run"]).replace(LS_REMOTE_PROBE, "git ls-remote")
    with pytest.raises(AssertionError):
        _assert_fetch_fails_closed(mutated)


# ---------------------------------------------------------------------------
# (13) the version switcher is written after the tree is staged
# ---------------------------------------------------------------------------


def _assert_versions_follows_stage(wf: dict[str, Any]) -> None:
    assert _index_of_named_step(wf, STAGE_STEP) < _index_of_named_step(wf, VERSIONS_STEP), (
        "a release build's root refresh clears the root before copying, so a "
        "versions.json written before staging would be deleted again"
    )


def test_version_switcher_is_written_after_staging(workflow: dict[str, Any]) -> None:
    _assert_versions_follows_stage(workflow)


def test_version_switcher_is_written_after_staging__mutation_swaps_them() -> None:
    mutated = copy.deepcopy(_load_workflow())
    steps = _steps(mutated)
    versions = steps.pop(_index_of_named_step(mutated, VERSIONS_STEP))
    steps.insert(_index_of_named_step(mutated, STAGE_STEP), versions)
    with pytest.raises(AssertionError):
        _assert_versions_follows_stage(mutated)


# ---------------------------------------------------------------------------
# The workflow reads steps.plan.outputs.*; the helper decides what those are
# ---------------------------------------------------------------------------


def _plan_output_keys() -> set[str]:
    """Run the helper's ``plan`` subcommand and collect the keys it emits.

    ``--tags`` is passed explicitly so the result does not depend on which tags
    happen to be fetched in the checkout the tests run from.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(HELPER_SOURCE),
            "plan",
            "--ref",
            "refs/heads/main",
            "--event",
            "push",
            "--input-tag",
            "",
            "--tags",
            "v2026.5.0",
            "--tags",
            "v2026.4.0",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.split("=", 1)[0] for line in result.stdout.splitlines() if "=" in line}


def _assert_referenced_outputs_exist(wf: dict[str, Any], produced: set[str]) -> None:
    referenced = set(_PLAN_OUTPUT_REF_RE.findall(_serialized(wf)))
    assert referenced, "the workflow reads no plan outputs at all"
    missing = referenced - produced
    assert not missing, f"docs.yml reads plan outputs the helper never writes: {sorted(missing)}"


def test_workflow_only_reads_plan_outputs_the_helper_writes(workflow: dict[str, Any]) -> None:
    produced = _plan_output_keys()
    assert "deploy" in produced, "the gate every step keys on must be one of them"
    _assert_referenced_outputs_exist(workflow, produced)


def test_workflow_only_reads_plan_outputs_the_helper_writes__mutation_reads_unknown_key() -> None:
    """A typo'd output silently evaluates to the empty string in an ``if:``, so
    nothing fails at runtime — it just never deploys. Catch it here instead."""
    mutated = copy.deepcopy(_load_workflow())
    _find_named_step(mutated, DEPLOY_STEP)["if"] = "steps.plan.outputs.should_deploy == 'true'"
    with pytest.raises(AssertionError):
        _assert_referenced_outputs_exist(mutated, _plan_output_keys())
