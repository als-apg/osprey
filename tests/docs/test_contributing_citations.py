"""What the contributor pages cite by name still exists.

``contributing/development-setup.rst`` and ``contributing/workflow.rst`` are
read by someone about to type what they say: a ``scripts/check.py`` verb and
its flags, an ``npm run`` script, a test file to copy, a workflow input to
dispatch with. Sphinx renders a stale one as a well-formed literal, so it
rots without a warning. This guard resolves each citation against its owner:

* ``scripts/check.py <verb> --flag`` against the command's own parser, and
  ``check.py run <ID>`` against the declared checks;
* ``scripts/ci/full_ci_on_main.py <subcommand> [lane]`` against that script's parser
  and its ``SPENDING_LANES``;
* the execution flags after ``check.py … --`` against ``validate_exec_flags``;
* ``npm run <script>`` against ``package.json``;
* a repository path (``tests/``, ``scripts/``, ``src/``, ``packages/``,
  ``plugins/``, ``changelog.d/``, ``.github/``) against the tree;
* ``gh workflow run <file> -f <input>=…`` against that workflow's
  ``workflow_dispatch`` inputs.

The negative controls at the end run each resolver on a planted stale
citation, so a resolver that stopped matching cannot pass vacuously.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import re
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _REPO_ROOT / "docs" / "source" / "contributing"
_PAGES = ("development-setup.rst", "workflow.rst")

#: Paths a page shows as a pattern for the reader to fill in, not as a file.
_ILLUSTRATIVE_PATHS = {
    "tests/path/to/test_file.py": "the single-file recipe's stand-in for the reader's own file",
}

_CHECK_CALL = re.compile(r"scripts/check\.py[ \t]+([^\n`]*)")
_FULL_CI_CALL = re.compile(r"scripts/ci/full_ci_on_main\.py[ \t]+([a-z-]+)(?:[ \t]+([a-z0-9-]+))?")
_NPM_RUN = re.compile(r"\bnpm run ([\w:-]+)")
_REPO_PATH = re.compile(
    r"(?<![\w/.<-])((?:tests|scripts|src|packages|plugins|changelog\.d|\.github)/[\w./*<>{}-]*)"
)
_WORKFLOW_RUN = re.compile(r"gh workflow run ([\w.-]+\.ya?ml)([^\n`]*)")
_DISPATCH_INPUT = re.compile(r"-f[ \t]+(\w+)=")


def _page_text(name: str) -> str:
    return (_DOCS / name).read_text(encoding="utf-8")


@cache
def _check_module():
    # scripts/ is not a package: load check.py by path, under a name no import
    # uses, registered before exec so @dataclass resolves its annotations.
    name = "_check_by_path_for_citations"
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / "check.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _check_verbs() -> dict[str, tuple[set[str], set[str]]]:
    """Each ``check.py`` verb: the options it accepts, and those that take a value."""
    parser = _check_module()._parser()
    verbs: dict[str, tuple[set[str], set[str]]] = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for verb, sub in action.choices.items():
                options = {opt for a in sub._actions for opt in a.option_strings}
                valued = {opt for a in sub._actions if a.nargs != 0 for opt in a.option_strings}
                verbs[verb] = (options, valued)
    return verbs


def _stale_check_calls(text: str) -> list[str]:
    verbs = _check_verbs()
    ids = {check.id for check in _check_module().CHECKS}
    stale = []
    for match in _CHECK_CALL.finditer(text):
        tokens = match.group(1).split()
        if not tokens or not re.fullmatch(r"[a-z]+", tokens[0]):
            continue
        verb, rest = tokens[0], iter(tokens[1:])
        if verb not in verbs:
            stale.append(f"check.py {verb}: no such verb")
            continue
        options, valued = verbs[verb]
        for token in rest:
            if token.startswith("#"):
                break
            if token == "--":
                flags = list(itertools.takewhile(lambda t: not t.startswith("#"), rest))
                try:
                    _check_module().validate_exec_flags(flags)
                except _check_module().Refused as refusal:
                    stale.append(f"check.py {verb} -- {' '.join(flags)}: {refusal}")
                break
            if token.startswith("--"):
                name = token.split("=", 1)[0]
                if name not in options:
                    stale.append(f"check.py {verb} {token}: no such option")
                elif name in valued and "=" not in token:
                    next(rest, None)
            elif verb == "run" and re.fullmatch(r"[a-z][a-z0-9-]*", token) and token not in ids:
                stale.append(f"check.py run {token}: no such check")
    return stale


@cache
def _spending_lanes() -> frozenset[str]:
    name = "_full_ci_on_main_by_path_for_citations"
    path = _REPO_ROOT / "scripts" / "ci" / "full_ci_on_main.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return frozenset(module.SPENDING_LANES)


def _stale_full_ci_calls(text: str) -> list[str]:
    calls = _FULL_CI_CALL.findall(text)
    if not calls:
        return []
    cited = {command for command, _lane in calls}
    stale = [
        f"full_ci_on_main.py last-green {lane}: no such spending lane"
        for command, lane in sorted(set(calls))
        if command == "last-green" and lane and lane not in _spending_lanes()
    ]
    out = subprocess.run(
        [sys.executable, "scripts/ci/full_ci_on_main.py", "--help"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    ).stdout
    choices = re.search(r"\{([a-z,-]+)\}", out)
    known = set(choices.group(1).split(",")) if choices else set()
    return [
        *(f"full_ci_on_main.py {name}: no such subcommand" for name in sorted(cited - known)),
        *stale,
    ]


def _stale_npm_scripts(text: str, scripts: set[str]) -> list[str]:
    return [f"npm run {name}" for name in sorted(set(_NPM_RUN.findall(text)) - scripts)]


def _stale_paths(text: str, root: Path) -> list[str]:
    stale = []
    for raw in sorted(set(_REPO_PATH.findall(text))):
        path = raw.rstrip(".,:;)")
        if path in _ILLUSTRATIVE_PATHS or any(c in path for c in "<>{}"):
            continue
        if any(c in path for c in "*?["):
            found = any(root.glob(path))
        else:
            found = (root / path).exists()
        if not found:
            stale.append(path)
    return stale


def _dispatch_inputs(workflow: Path) -> set[str]:
    data = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    triggers = data.get("on", data.get(True)) or {}
    dispatch = triggers.get("workflow_dispatch") or {}
    return set((dispatch.get("inputs") or {}).keys())


def _stale_workflow_inputs(text: str, root: Path) -> list[str]:
    stale = []
    for match in _WORKFLOW_RUN.finditer(text):
        workflow = root / ".github" / "workflows" / match.group(1)
        if not workflow.exists():
            stale.append(f"{match.group(1)}: no such workflow")
            continue
        inputs = _dispatch_inputs(workflow)
        stale += [
            f"{match.group(1)} -f {name}: no such input"
            for name in _DISPATCH_INPUT.findall(match.group(2))
            if name not in inputs
        ]
    return stale


@cache
def _npm_scripts() -> set[str]:
    package = json.loads((_REPO_ROOT / "package.json").read_text(encoding="utf-8"))
    return set(package["scripts"])


@pytest.mark.parametrize("page", _PAGES)
def test_check_py_citations_resolve(page):
    assert not _stale_check_calls(_page_text(page))


@pytest.mark.parametrize("page", _PAGES)
def test_full_ci_on_main_citations_resolve(page):
    assert not _stale_full_ci_calls(_page_text(page))


@pytest.mark.parametrize("page", _PAGES)
def test_npm_script_citations_resolve(page):
    assert not _stale_npm_scripts(_page_text(page), _npm_scripts())


@pytest.mark.parametrize("page", _PAGES)
def test_repository_paths_exist(page):
    assert not _stale_paths(_page_text(page), _REPO_ROOT)


@pytest.mark.parametrize("page", _PAGES)
def test_workflow_dispatch_inputs_exist(page):
    assert not _stale_workflow_inputs(_page_text(page), _REPO_ROOT)


def test_illustrative_paths_are_still_cited():
    """An exemption the pages no longer need leaves the list."""
    text = "".join(_page_text(page) for page in _PAGES)
    assert all(path in text for path in _ILLUSTRATIVE_PATHS)


def test_the_pages_cite_something_of_each_kind():
    """The resolvers are fed: each kind of citation occurs on the pages."""
    text = "".join(_page_text(page) for page in _PAGES)
    assert _CHECK_CALL.search(text)
    assert _FULL_CI_CALL.search(text)
    assert _NPM_RUN.search(text)
    assert _REPO_PATH.search(text)
    assert _WORKFLOW_RUN.search(text)


def test_planted_stale_citations_are_reported():
    planted = (
        "uv run python scripts/check.py quick --no-such-flag\n"
        "uv run python scripts/check.py lint\n"
        "uv run python scripts/check.py run no-such-check\n"
        "python3 scripts/ci/full_ci_on_main.py last-red\n"
        "python3 scripts/ci/full_ci_on_main.py last-green no-such-lane\n"
        "uv run python scripts/check.py run pytest -- -k selection\n"
        "npm run no-such-script\n"
        "``tests/docs/test_no_such_file.py``\n"
        "gh workflow run ci.yml --ref main -f no_such_input=true\n"
        "gh workflow run no-such.yml\n"
    )
    assert len(_stale_check_calls(planted)) == 4
    assert _stale_full_ci_calls(planted) == [
        "full_ci_on_main.py last-red: no such subcommand",
        "full_ci_on_main.py last-green no-such-lane: no such spending lane",
    ]
    assert _stale_npm_scripts(planted, _npm_scripts()) == ["npm run no-such-script"]
    assert _stale_paths(planted, _REPO_ROOT) == ["tests/docs/test_no_such_file.py"]
    assert _stale_workflow_inputs(planted, _REPO_ROOT) == [
        "ci.yml -f no_such_input: no such input",
        "no-such.yml: no such workflow",
    ]
