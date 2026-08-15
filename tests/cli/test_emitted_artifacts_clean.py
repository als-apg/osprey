"""Nothing a deployment ships may name a command that no longer exists.

``test_init_verb`` asks this of the repo ``osprey init`` writes. That repo is
only half of what a deployment carries: a profile with deployment coordinates
also ships a CI pipeline and a health check, and ``init`` emits neither unless
the coordinates are there, so its sweep can never see them. This module covers
that half, and pins the shape the two emitted files were rewritten onto.

Why a comment counts. The pipeline is the file an operator opens when a deploy
fails, and every command in it — in a ``script:`` or in the header above it — is
an instruction. A pipeline whose comment names a verb the CLI no longer has is
worse than one that says nothing, because it costs the reader the time to try
it.

Two shapes are checked separately, because they fail differently:

* The COMMAND SURFACE — which verbs run, with which flags. Wrong here and the
  deploy job fails loudly, on a real deploy.
* The LAYOUT — the ``profile/`` prefix and the ``build/<name>/`` sibling the
  retired shape keyed every path off. Wrong here and the commands still parse;
  they just act on a directory that is not there.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile_deploy import DeployConfig, parse_deploy_block
from osprey.cli.deploy_scaffold_templates import (
    CI_TEMPLATES,
    VERIFY_TEMPLATE,
    build_ci_context,
    build_verify_context,
    render,
)
from tests.cli.test_init_verb import RETIRED_VERB_STRINGS
from tests.fixtures.lifecycle_repo import (
    GITLAB_CI_YML,
    VERIFY_SH,
    build_exemplar_repo,
    expand_sentinels,
)

#: The sentinel the fixture leaves in place of the installed version. Passing it
#: as the render's version keeps both sides of a byte comparison symbolic, so
#: the comparison stays meaningful across releases.
VERSION_SENTINEL = "@OSPREY_VERSION@"

#: Spellings that only make sense in the retired sibling-project layout, where a
#: repo held its profile under ``profile/`` and its build under
#: ``build/<project-name>/``. The three-zone repo has neither, so any of these
#: in an emitted file is a path that will not resolve on the deploy host.
RETIRED_LAYOUT_STRINGS = (
    "OSPREY_PROJECT_NAME",
    "OSPREY_BUILD_DIR",
    "OSPREY_PROFILE",
    "--project",
    "profile/profile.yml",
    "profile/services/",
)

#: Every ``osprey`` command the emitted pipeline runs, in the order it runs
#: them. Pinned as a literal rather than read back off the render: this list IS
#: the deployment story the templates exist to tell, and a render that quietly
#: stopped running one of these steps — the secrets render above all — would
#: otherwise satisfy every other assertion in this file.
EXPECTED_INVOCATIONS = [
    "osprey validate",
    "osprey build --skip-lifecycle --skip-deps",
    "osprey build",
    "osprey users env --output .env.users",
    "osprey up -d",
]

_INVOCATION_RE = re.compile(r"^\s*(?:- )?(osprey .+)$", re.MULTILINE)


@pytest.fixture(scope="module")
def exemplar_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The exemplar deployment repo, with deployment coordinates filled in."""
    return build_exemplar_repo(tmp_path_factory.mktemp("ci") / "als-exemplar", with_ci=True)


@pytest.fixture(scope="module")
def profile(exemplar_repo: Path) -> dict[str, Any]:
    """The exemplar profile, as the scaffolder reads it."""
    loaded = yaml.safe_load((exemplar_repo / "profile.yml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture(scope="module")
def deploy(profile: dict[str, Any]) -> DeployConfig:
    """The exemplar's parsed ``deploy:`` block."""
    parsed = parse_deploy_block(profile)
    assert parsed is not None, "the exemplar must declare deployment coordinates"
    return parsed


@pytest.fixture(scope="module")
def rendered(profile: dict[str, Any], deploy: DeployConfig, exemplar_repo: Path) -> dict[str, str]:
    """Both emitted files, keyed by the repo-relative path they land at."""
    return {
        ".gitlab-ci.yml": render(
            CI_TEMPLATES[deploy.ci],
            build_ci_context(profile, deploy, exemplar_repo, exemplar_repo.name, VERSION_SENTINEL),
        ),
        "scripts/verify.sh": render(
            VERIFY_TEMPLATE, build_verify_context(profile, VERSION_SENTINEL)
        ),
    }


# ── SC-8: no emitted artifact names a retired verb ───────────────────────────


def test_the_retired_verb_list_is_not_empty() -> None:
    """The sweeps below are only worth as much as the list they sweep for.

    ``RETIRED_VERB_STRINGS`` is imported from ``test_init_verb`` so both halves
    of a deployment are judged against one list rather than two that can drift.
    An empty import would turn every sweep here green while checking nothing.
    """
    assert RETIRED_VERB_STRINGS


@pytest.mark.parametrize("name", (".gitlab-ci.yml", "scripts/verify.sh"))
def test_no_emitted_deploy_file_names_a_retired_verb(rendered: dict[str, str], name: str) -> None:
    """SC-8, for the two files ``init`` does not write and its sweep cannot see."""
    assert [verb for verb in RETIRED_VERB_STRINGS if verb in rendered[name]] == []


@pytest.mark.parametrize("name", (".gitlab-ci.yml", "scripts/verify.sh"))
def test_no_emitted_deploy_file_assumes_the_sibling_project_layout(
    rendered: dict[str, str], name: str
) -> None:
    """Every path is repo-relative: no ``profile/`` prefix, no ``build/<name>/``.

    Checked apart from the verb sweep because it fails silently. A command
    naming a directory that is not there parses fine and only goes wrong on a
    host, where it reads as a broken deployment rather than a broken pipeline.
    """
    assert [token for token in RETIRED_LAYOUT_STRINGS if token in rendered[name]] == []


# ── The command surface the pipeline runs ────────────────────────────────────


def test_the_pipeline_runs_exactly_the_new_surface(rendered: dict[str, str]) -> None:
    """The osprey invocations are the new verbs, in order, with their flags."""
    found = _INVOCATION_RE.findall(rendered[".gitlab-ci.yml"])
    assert found, "extracted no osprey invocations — every sweep here would be vacuous"
    assert found == EXPECTED_INVOCATIONS


def test_the_secrets_render_still_writes_to_a_file(rendered: dict[str, str]) -> None:
    """``--output`` is not optional, and it is the file's only mention.

    Without the flag ``osprey users env`` writes the assembled
    secrets to stdout, which in this job is the retained CI log. The flag also
    creates the file at mode 0600 from its first byte, which a shell redirect
    would not. Asserted positively AND as the file's only mention, so neither
    dropping the flag nor reintroducing a CI-side assembly can pass.
    """
    pipeline = rendered[".gitlab-ci.yml"]
    mentions = [line.strip() for line in pipeline.splitlines() if ".env.users" in line]
    assert mentions == ["osprey users env --output .env.users"]
    assert ">" not in mentions[0]
    assert "ENVEOF" not in pipeline


def test_no_declared_secret_crosses_the_ssh_boundary(
    rendered: dict[str, str], profile: dict[str, Any]
) -> None:
    """The deploy heredoc is unquoted, so a ``$NAME`` in it is expanded by CI.

    An expanded secret lands in the deploy host's process arguments, visible to
    every account on the box. The host reads its own ``.env`` instead.
    """
    heredoc = re.search(r"<<REMOTE\n(.*?)\n\s*REMOTE\n", rendered[".gitlab-ci.yml"], re.DOTALL)
    assert heredoc, "could not locate the deploy job's remote heredoc"

    declared = profile["env"]["required"]
    assert declared, "the exemplar declares no secrets, so this proves nothing"
    for name in declared:
        assert f"${name}" not in heredoc.group(1)


def test_the_pipeline_is_valid_yaml_gitlab_would_read(rendered: dict[str, str]) -> None:
    """A comment rewrite that broke the document would otherwise fail only in CI."""
    pipeline = yaml.safe_load(rendered[".gitlab-ci.yml"])
    assert pipeline["stages"] == ["validate", "deploy"]
    assert pipeline["deploy"]["needs"] == ["render-build"]
    assert pipeline["deploy"]["rules"] == [
        {"if": "$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH", "when": "manual"}
    ]


@pytest.mark.parametrize(
    ("name", "specification"),
    [(".gitlab-ci.yml", GITLAB_CI_YML), ("scripts/verify.sh", VERIFY_SH)],
)
def test_the_emitted_pair_reproduces_the_exemplar_byte_for_byte(
    rendered: dict[str, str], name: str, specification: str
) -> None:
    """The templates render the hand-authored specification exactly.

    The exemplar in ``tests/fixtures/lifecycle_repo.py`` was written by hand and
    is the specification; the templates are made to reproduce it, never the
    reverse. Compared against the fixture's own text, whose version sentinel is
    the value the render was given, so the comparison survives a release bump.
    """
    assert rendered[name] == specification


# ── The health check ─────────────────────────────────────────────────────────


def test_health_check_always_exits_zero(rendered: dict[str, str]) -> None:
    """Verification is advisory: a failed probe must never fail a deploy."""
    verify = rendered["scripts/verify.sh"]
    assert verify.rstrip().endswith("exit 0")
    assert not re.search(r"^\s*exit [1-9]", verify, re.MULTILINE)
    assert not re.search(r"^set -e", verify, re.MULTILINE)


def test_probe_group_filter_is_not_named_groups(rendered: dict[str, str]) -> None:
    """The filter variable must not be ``GROUPS``.

    Bash owns ``GROUPS`` — assigning to it is silently ignored, so the rename
    would leave every ``wants`` call false. The script exits 0 either way, and a
    health check that runs no probes reports perfect health.
    """
    verify = rendered["scripts/verify.sh"]
    assert re.search(r'^PROBE_GROUPS="\$\{\*:-[a-z ]+\}"$', verify, re.MULTILINE)
    assert not re.search(r"^\s*GROUPS=", verify, re.MULTILINE)
    assert not re.search(r"\$\{?GROUPS\b", verify)


def test_the_health_check_probes_the_dispatcher(rendered: dict[str, str]) -> None:
    """A deployment that runs the event dispatcher gets a probe for it.

    The dispatcher is the one service whose failure is invisible from the web
    tier: the terminals answer, the stack looks up, and every webhook is
    silently dropped.
    """
    assert (
        "probe_http 'dispatcher health' http://localhost:8020/health"
        in (rendered["scripts/verify.sh"])
    )


def test_the_tcp_helper_is_emitted_only_where_a_probe_uses_it(
    profile: dict[str, Any], rendered: dict[str, str]
) -> None:
    """A helper shelling out to ``python3`` must not appear unused.

    The deploy host needs ``python3`` only because of this helper, so a script
    carrying it with nothing to probe documents a dependency the deployment does
    not have.
    """
    assert ("probe_tcp" in rendered["scripts/verify.sh"]) is True
    assert build_verify_context(profile).has_tcp_probe

    http_only = {key: value for key, value in profile.items() if key != "virtual_accelerator"}
    http_only["services"] = {}
    assert "probe_tcp" not in render(VERIFY_TEMPLATE, build_verify_context(http_only))


def test_the_usage_example_names_a_group_the_script_has(rendered: dict[str, str]) -> None:
    """``./scripts/verify.sh <group>`` in the header must select something.

    The header is the only documentation the script has, and an example naming
    a group the ``wants`` filter never matches runs no probes and still exits 0
    — a health check reporting perfect health by doing nothing.
    """
    verify = rendered["scripts/verify.sh"]
    example = re.search(r"^#   \./scripts/verify\.sh (\S+)\s+# one group$", verify, re.MULTILINE)
    assert example, "the header must show a one-group example"

    default = re.search(r'^PROBE_GROUPS="\$\{\*:-([a-z ]+)\}"$', verify, re.MULTILINE)
    assert default and example.group(1) in default.group(1).split()


def test_the_automatic_run_is_claimed_only_where_it_happens(profile: dict[str, Any]) -> None:
    """Only a web-tier deployment may be told ``osprey up`` runs the check.

    ``run_verify_script`` is called from ``deploy_up_web_terminals``, which
    ``_start_stack`` reaches only when ``modules.web_terminals.enabled`` is
    set. Both emitted files used to state the automatic run unconditionally, so
    a backend-only deployment read that its health check was covered, and the
    pipeline cited that same sentence as the reason it ships no verify job —
    leaving nothing to probe the deploy at all.

    The promise is the thing under test, not the wording: a deployment without
    the web tier must not be told the run is automatic.

    BOTH ways of switching the module off are checked. A ``config:`` block is a
    flat bag of dotted keys, and the bundled persona presets inherit the whole
    ``modules.web_terminals:`` subtree — ``enabled: true`` and all — while
    switching it off with a separate, deeper ``modules.web_terminals.enabled:
    false``. Reading the subtree key alone answers "enabled" for a profile the
    build treats as disabled, which puts the false promise back exactly where
    this test exists to keep it out.
    """
    assert profile["config"]["modules.web_terminals"]["enabled"] is True, (
        "the exemplar must enable the web tier, or the enabled half proves nothing"
    )
    deploy = parse_deploy_block(profile)
    assert deploy is not None

    subtree_key = copy.deepcopy(profile)
    subtree_key["config"]["modules.web_terminals"]["enabled"] = False

    # The persona-preset spelling: the inherited subtree above still says
    # `enabled: true`, and only this deeper key turns the module off.
    deeper_key = copy.deepcopy(profile)
    deeper_key["config"]["modules.web_terminals.enabled"] = False
    assert deeper_key["config"]["modules.web_terminals"]["enabled"] is True, (
        "the deeper-key case must leave the subtree saying enabled, or it is the other case"
    )

    def _render_pair(source: dict[str, Any]) -> tuple[str, str]:
        return (
            render(
                CI_TEMPLATES[deploy.ci],
                build_ci_context(source, deploy, Path("/repo"), "als-exemplar", VERSION_SENTINEL),
            ),
            render(VERIFY_TEMPLATE, build_verify_context(source, VERSION_SENTINEL)),
        )

    for off in (subtree_key, deeper_key):
        for enabled, disabled in zip(_render_pair(profile), _render_pair(off), strict=True):
            assert "on its own" in enabled or "runs it automatically" in enabled
            assert "on its own" not in disabled
            assert "runs it automatically" not in disabled
            assert "does NOT run" in disabled or "Nothing" in disabled


# ── The scaffolded pair travels with the coordinates ─────────────────────────


def test_the_exemplar_ships_both_files_and_nothing_else_new(exemplar_repo: Path) -> None:
    """Deployment coordinates add exactly the pipeline pair to the source zone.

    ``expand_sentinels`` is exercised here too: the emitted files carry the
    provenance stamp, and a fixture that stopped expanding it would leave a
    literal ``@OSPREY_VERSION@`` in a file an operator reads.
    """
    for relative in (".gitlab-ci.yml", "scripts/verify.sh"):
        text = exemplar_repo.joinpath(relative).read_text(encoding="utf-8")
        assert VERSION_SENTINEL not in text
        assert text.startswith(expand_sentinels(text[: text.index("\n")]))
