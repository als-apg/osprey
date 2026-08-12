"""The emitted pipeline's ``osprey`` commands, run through the real CLI parser.

``tests/cli/test_emitted_artifacts_clean.py`` proves the templates render the
hand-authored exemplar byte for byte. That says nothing about whether the
commands inside them exist. A template can reproduce its specification perfectly
and still name a flag that was renamed three releases ago — the specification
was hand-written against the CLI as it was, and nothing since then has
re-checked it. The pipeline only finds out in CI, on a facility's deploy day.

This module welds the two halves together: it renders the pipeline through the
real emission engine, pulls every ``osprey`` invocation out of the rendered
YAML, and drives each one through the real ``osprey`` command tree. A flag that
no longer exists, a verb that was moved under a different group, an argument
that changed arity — each one fails here, in this repo, at the commit that
caused it.

Parse, not execute
------------------

Nothing here runs a command body. ``Command.make_context`` does the whole
parsing layer — option lookup, type conversion, arity and required-argument
checks — and then stops; ``Command.invoke`` is what would call the function
underneath, and it is never reached. So the harness walks the command tree by
hand: make a context for the group, take the tokens it did not consume, ask the
group to resolve the next name, repeat until a leaf command holds them all.

That also keeps the root group's callback out of it. ``osprey``'s own callback
loads ``.env`` into the process environment, configures logging, and launches
the interactive menu when no subcommand follows — all things a test must not
trigger, and all things ``make_context`` skips.

What a failure looks like is therefore unambiguous. Every way the CLI can reject
an argv shape raises :class:`click.UsageError` — ``NoSuchOption`` and
``NoSuchCommand`` are both subclasses, and all of them carry ``exit_code == 2``.
Nothing else can raise from a parse. Working at the exception layer rather than
on rendered output also sidesteps Click's ``CliRunner``, whose Rich-wrapped
stdout breaks any assertion on error text.

Concrete values
---------------

The pipeline's invocations carry no arguments at all now: the repo IS the
deployment, so every verb walks up to the profile from wherever it runs. The
harness still expands any ``$NAME`` against the pipeline's own ``variables:``
block — so the substitution is the pipeline's and not the test's opinion of it —
and still parses from inside a real repo, because part of what a parse checks is
the filesystem.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import click
import pytest
import yaml

from osprey.cli.deploy_scaffold import scaffold_deploy_files
from osprey.cli.main import cli

GOLDENS = Path(__file__).parent / "goldens"
EXEMPLAR_DIR = GOLDENS / "exemplar-profile"

#: The deployment repo's directory name — its deployment name, and nothing a
#: path in the emitted pipeline keys off.
REPO_NAME = "demo-facility"

#: Passed in place of the installed version so the render does not depend on
#: the release the test happens to run under.
FROZEN_VERSION = "OSPREY_VERSION"

#: Job keys whose value is shell to be run. Everything the pipeline asks
#: ``osprey`` to do lives under one of these.
SCRIPT_KEYS = ("before_script", "script", "after_script")

#: An ``osprey`` invocation on a line of shell. The optional ``- `` covers a
#: YAML list item that survived into a block scalar's text; a plain
#: ``^\s*osprey`` would miss those.
INVOCATION_RE = re.compile(r"^\s*(?:- )?(osprey .+)$", re.MULTILINE)

#: A ``$NAME`` reference, expanded from the pipeline's ``variables:`` block.
VARIABLE_RE = re.compile(r"\$(\w+)")

#: What the pipeline must ask the CLI to do, in pipeline order. Pinned as a
#: literal rather than derived: this list is the deployment story the templates
#: exist to tell, and a render that quietly stops running one of these steps is
#: the failure this file is here to catch.
EXPECTED_INVOCATIONS = [
    "osprey validate",
    "osprey build --skip-lifecycle --skip-deps",
    "osprey build",
    "osprey users env-production --output .env.production",
    "osprey up -d",
]


# ── Rendering the pipeline through the real engine ───────────────────────────


@pytest.fixture(scope="module")
def facility_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A deployment repo with the exemplar profile, scaffolded by the real engine.

    Three-zone: ``profile.yml`` at the repo root and the health check under
    ``scripts/``, which is where the emitted pipeline says both of them are —
    and where the engine puts them, since those destinations are the layout's
    and not a caller's to choose. Emitted by the same function the scaffolding
    verb calls, so what is parsed below is what a facility would get.
    """
    repo_root = tmp_path_factory.mktemp("facility") / REPO_NAME
    shutil.copytree(EXEMPLAR_DIR, repo_root)

    emitted = scaffold_deploy_files(repo_root, osprey_version=FROZEN_VERSION)
    assert [file.action for file in emitted] == ["created", "created"], emitted
    return repo_root


@pytest.fixture(scope="module")
def pipeline(facility_repo: Path) -> dict[str, Any]:
    """The emitted pipeline, as GitLab would read it."""
    loaded = yaml.safe_load((facility_repo / ".gitlab-ci.yml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), "the emitted pipeline must be a YAML mapping"
    return loaded


@pytest.fixture(scope="module")
def variables(pipeline: dict[str, Any]) -> dict[str, str]:
    """The pipeline's own ``variables:`` block — the substitution CI performs."""
    declared = pipeline.get("variables")
    assert isinstance(declared, dict) and declared, "the pipeline must declare variables"
    return {name: str(value) for name, value in declared.items()}


@pytest.fixture(autouse=True)
def in_facility_repo(facility_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse from the repo root, with the build directory a deploy would have.

    ``build/`` is ``osprey build``'s output; on the deploy host it exists by the
    time the later verbs run, because the build ran first in the same heredoc.
    Creating it here reproduces that, and is what lets any existence check in a
    parse mean something.
    """
    (facility_repo / "build").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(facility_repo)


# ── Pulling the invocations out ──────────────────────────────────────────────


def script_texts(node: Any) -> list[str]:
    """Every shell fragment in the pipeline, in document order.

    Walks the loaded YAML rather than the raw file. A list item and a block
    scalar arrive here as the same thing — text — which is the point: reading
    them post-parse proves they are shell GitLab would actually run, not just
    characters that happen to sit in the file.
    """
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in SCRIPT_KEYS:
                found.extend(_flatten(value))
            else:
                found.extend(script_texts(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(script_texts(item))
    return found


def _flatten(value: Any) -> list[str]:
    """One script block's entries as flat text."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _flatten(item)]
    return []


def extract_invocations(pipeline: dict[str, Any]) -> list[str]:
    """Every ``osprey`` command the pipeline runs, in pipeline order."""
    return INVOCATION_RE.findall("\n".join(script_texts(pipeline)))


@pytest.fixture(scope="module")
def invocations(pipeline: dict[str, Any]) -> list[str]:
    """The extracted invocations, guarded against extracting nothing.

    An empty extraction is the one result this file cannot report as success:
    every assertion below is over this list, so a regex that stops matching
    would turn the whole module green while checking nothing at all.
    """
    found = extract_invocations(pipeline)
    assert found, (
        "extracted no 'osprey' invocations from the emitted pipeline. Either the "
        "pipeline stopped running any, or the extraction here stopped seeing "
        "them — both make every test in this module vacuous."
    )
    return found


# ── Driving them through the parser ──────────────────────────────────────────


def expand(invocation: str, variables: dict[str, str]) -> list[str]:
    """An invocation as argv, with the pipeline's variables expanded.

    Split before substituting, so a value containing a space cannot silently
    become two arguments — CI would not split it either.
    """

    def substitute(match: re.Match[str]) -> str:
        return variables.get(match.group(1), match.group(0))

    argv = invocation.split()[1:]  # drop the 'osprey' program name
    expanded = [VARIABLE_RE.sub(substitute, arg) for arg in argv]

    unresolved = [arg for arg in expanded if "$" in arg]
    assert not unresolved, (
        f"{invocation!r} references variables the pipeline does not declare: "
        f"{unresolved}. An osprey invocation may only use its own variables block."
    )
    return expanded


def unconsumed(context: click.Context) -> list[str]:
    """The tokens a group parsed but left for its subcommand.

    Click splits them across two attributes — the subcommand's name lands in
    ``_protected_args`` and its arguments in ``args``. The private name is read
    through ``getattr`` so that the layout Click is moving to, everything in
    ``args``, keeps working; the public alias for it emits a
    ``DeprecationWarning`` and is not worth the noise in a test run.
    """
    return [*getattr(context, "_protected_args", []), *context.args]


def parse(argv: list[str]) -> click.Context:
    """Resolve *argv* against the real command tree, without running anything.

    Returns the leaf command's context — proof that the whole argv was consumed
    by real parameters. Raises :class:`click.UsageError` for anything the CLI
    would reject.
    """
    command: click.Command = cli
    context: click.Context | None = None
    name = "osprey"

    while True:
        context = command.make_context(name, list(argv), parent=context)
        if not isinstance(command, click.Group):
            return context
        remaining = unconsumed(context)
        if not remaining:
            return context
        resolved_name, resolved, argv = command.resolve_command(context, remaining)
        assert resolved is not None, f"osprey has no command {resolved_name!r}"
        name, command = resolved_name or "", resolved


def parse_or_fail(invocation: str, variables: dict[str, str]) -> click.Context:
    """Parse an invocation, reporting a rejection as the pipeline bug it is."""
    argv = expand(invocation, variables)
    try:
        return parse(argv)
    except click.UsageError as error:
        pytest.fail(
            f"the emitted pipeline runs {invocation!r}, which the osprey CLI "
            f"rejects: {type(error).__name__}: {error.format_message()}"
        )
    except SystemExit as error:  # pragma: no cover - an eager option would exit
        pytest.fail(f"parsing {invocation!r} exited the interpreter (code {error.code})")


# ── The weld ─────────────────────────────────────────────────────────────────


def test_pipeline_runs_the_invocations_the_deployment_story_promises(
    invocations: list[str],
) -> None:
    """The extracted set is the expected one, in order.

    Both halves matter. Extracting *something* is what keeps the parse tests
    below from being vacuous; extracting exactly *this* is what keeps a render
    from dropping a step — the secrets render in particular, whose absence a
    parse check could never notice.
    """
    assert invocations == EXPECTED_INVOCATIONS


def test_every_key_step_is_present(invocations: list[str]) -> None:
    """The three commands the deploy cannot happen without are all run."""
    joined = "\n".join(invocations)
    for step in ("users env-production", "build", "up -d"):
        assert f"osprey {step}" in joined, f"the pipeline never runs 'osprey {step}'"


#: Identified by the invocation itself, so a rejection names the command that
#: was rejected rather than the position it sat at.
_INVOCATION_CASES = [
    pytest.param(index, id=invocation) for index, invocation in enumerate(EXPECTED_INVOCATIONS)
]


@pytest.mark.parametrize("index", _INVOCATION_CASES)
def test_the_cli_accepts_every_invocation(
    index: int, invocations: list[str], variables: dict[str, str]
) -> None:
    """Each invocation resolves to a real command with real options.

    Parametrized over positions rather than looped so a rename shows up as one
    named failing case, and so the count itself is pinned: a pipeline that
    grew or lost an invocation fails here before any parse runs.
    """
    assert index < len(invocations), (
        f"the pipeline runs {len(invocations)} osprey invocations, fewer than the "
        f"{len(EXPECTED_INVOCATIONS)} expected"
    )
    context = parse_or_fail(invocations[index], variables)
    assert not isinstance(context.command, click.Group), (
        f"{invocations[index]!r} stops at a group, not a command"
    )


def test_the_secrets_render_takes_output_as_the_pipeline_writes_it(
    invocations: list[str], variables: dict[str, str]
) -> None:
    """``--output`` is accepted, with the repo-relative path the pipeline writes.

    ``--output`` is what keeps the assembled secrets out of the CI job log. If
    the option is ever renamed, or stops accepting a relative path, the pipeline
    breaks on a real deploy with real credentials in it — and the failure mode
    without the flag is not a crash but a job log full of them.
    """
    (secrets,) = [line for line in invocations if "env-production" in line]
    context = parse_or_fail(secrets, variables)

    output = context.params["output"]
    assert Path(output).name == ".env.production"
    assert not Path(output).is_absolute()


# ── The instrument itself ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["build", "--no-such-flag"], "No such option"),
        (["users", "no-such-verb"], "No such command"),
        (["validate", "--repo", "nowhere"], "does not exist"),
    ],
)
def test_the_parser_rejects_what_it_should(argv: list[str], expected: str) -> None:
    """The harness can fail — a renamed flag or verb does not slip through.

    Without this the tests above would pass just as happily against a parser
    that accepted anything, which is the failure mode a weld test is most prone
    to. The path case is here too: existence checks are part of the parse, so a
    pipeline naming a path that will not be there is caught the same way.
    """
    with pytest.raises(click.UsageError) as caught:
        parse(argv)
    assert expected in caught.value.format_message()
    assert caught.value.exit_code == 2


def test_extraction_reads_every_shape_a_script_block_takes() -> None:
    """The shapes the pipeline uses, and the one it could grow into.

    The validate stage writes its commands as list items and the deploy stage
    buries three inside a heredoc in a block scalar; an extraction handling only
    one of those would silently check half the pipeline. The third case is why
    the pattern tolerates a leading ``- ``: a block scalar's text is shell, not
    YAML, so a dash-prefixed line inside one is a command like any other.

    The last two entries are the boundary. Text outside a script key is not run,
    and a line only mentioning ``osprey`` is not an invocation — counting either
    would put commands through the parser that CI never executes.
    """
    document = yaml.safe_load(
        "list-shaped:\n"
        "  script:\n"
        "    - osprey build one\n"
        "block-shaped:\n"
        "  script:\n"
        "    - |\n"
        "      ssh host <<REMOTE\n"
        "      osprey up -d\n"
        "      - osprey status\n"
        "      REMOTE\n"
        "not-a-script:\n"
        "  variables:\n"
        "    NOTE: osprey build should-not-be-found\n"
        "mentioned-only:\n"
        "  script:\n"
        "    - echo 'run osprey build yourself'\n"
    )
    assert extract_invocations(document) == [
        "osprey build one",
        "osprey up -d",
        "osprey status",
    ]
