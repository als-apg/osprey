"""The lifecycle success criteria, as checks that run instead of checks that are read.

SC-5, SC-7 and SC-8 were written as grep criteria over ``src/``. A grep in a
plan document is a thing somebody remembers to run once. The same grep as a
test is a thing the next change has to get past, which is the only form in
which "zero occurrences" survives contact with a codebase that keeps moving.
That is what this module is: the source-tree-wide criteria, scripted.

It deliberately does NOT re-test behavior. Every per-command contract already
has its own suite — ``test_repo_resolver`` for the walk, ``test_discovery_rewire``
for the rewired verbs, ``test_init_verb`` and ``test_emitted_artifacts_clean``
for what a fresh deployment ships. Those check that the right thing happens.
This checks that the wrong thing is nowhere left to be found, which is a
different question and needs a different instrument: a sweep over the source
tree rather than an invocation of any part of it.

Three habits keep a sweep like this honest.

**Word-exact, and proven word-exact.** ``OSPREY_PROJECT`` is retired;
``OSPREY_PROJECT_DIR``, ``OSPREY_PROJECT_NAME`` and ``OSPREY_PROJECT_LABEL``
are load-bearing. A substring grep condemns all three, so the criterion is a
word match — and the test that asserts the retired name is absent also asserts
the three compounds are PRESENT. A word-boundary bug that silently matched
nothing would otherwise read as a pass.

**Every exemption is named, with its reason.** An allowlist entry says which
file, which string, and why that occurrence has to stay. An allowlist that
grew by accident is a criterion that has quietly stopped meaning anything, so
each entry carries the argument for its own existence and the test fails if an
entry stops matching — a stale exemption is as much a defect as a new
violation.

**Structure, not enumeration.** The resolver criterion does not list the
commands that must use the resolver. It asks the CLI for its own command list
and requires every entry to either reach the resolver or appear in the
resolver's own exemption sets. A command added tomorrow is covered the day it
is registered, which a hand-written list would not be.
"""

from __future__ import annotations

import bisect
import re
import sys
from pathlib import Path

import pytest

from osprey.cli.main import LazyGroup
from osprey.cli.repo_resolver import (
    EXPLICIT_TARGET_COMMANDS,
    MODE_EXEMPT_FLAGS,
    REPO_FREE_COMMANDS,
    SELF_DISCOVERING_COMMANDS,
)

#: Every way a command is allowed to not reach the resolver. Assembled from the
#: resolver's own sets rather than restated, so a set added there is covered
#: here without an edit — and a command quietly dropped from one is not.
_EXEMPT = REPO_FREE_COMMANDS | SELF_DISCOVERING_COMMANDS | EXPLICIT_TARGET_COMMANDS

#: The shipped source tree, located from this file rather than from an import
#: of ``osprey``. The criteria are about what the REPOSITORY ships; resolving
#: the package through its import would follow whichever copy happens to be
#: installed, which in a worktree is not reliably this one.
SRC = Path(__file__).resolve().parents[2] / "src" / "osprey"

#: Suffixes that are not text, plus caches. Everything else in ``src/`` is
#: read: the criteria cover Jinja templates, Dockerfiles and dotfiles just as
#: much as Python, because a comment in a rendered ``config.yml`` is an
#: instruction to an operator exactly as much as a ``click.echo`` is.
_BINARY_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".otf", ".pyc", ".so"}
)


def _shipped_files() -> list[Path]:
    """Every text file under ``src/osprey``, sorted for stable failure output."""
    return sorted(
        path
        for path in SRC.rglob("*")
        if path.is_file()
        and path.suffix.lower() not in _BINARY_SUFFIXES
        and "__pycache__" not in path.parts
    )


def _rel(path: Path) -> str:
    return str(path.relative_to(SRC.parents[1]))


#: Leading comment decoration stripped before lines are joined, so a comment
#: whose ``#`` markers land mid-sentence does not break the pattern the way the
#: line break itself would.
_COMMENT_LEAD = re.compile(r"^(?:#+|//+|\*)\s*")


def _scan(pattern: re.Pattern[str]) -> list[tuple[str, int, str]]:
    """Every ``(relative path, line number, line)`` in ``src/`` matching *pattern*.

    Matched twice per file: once line by line, and once against the whole file
    re-joined into single-spaced text. The second pass is what makes the
    criterion hold across a LINE BREAK — ``osprey deploy`` at the end of one
    docstring line and its verb at the start of the next reaches an operator as
    one sentence, and a per-line scan alone reads it as two harmless halves.
    That is not hypothetical: it is how four ``osprey deploy
    render-env-production`` occurrences sat in shipped text, one of them a
    raised error message, while this criterion reported clean.

    A hit is always reported against the line the match STARTS on, so the
    exact-line allowlist keys the same way whichever pass found it.
    """
    hits: list[tuple[str, int, str]] = []
    for path in _shipped_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # pragma: no cover - defensive, no such file today
            continue
        lines = text.splitlines()
        stripped = [_COMMENT_LEAD.sub("", line.strip()) for line in lines]

        # Offset of each stripped line within the joined text, so a match
        # position maps back to the line it began on.
        starts: list[int] = []
        offset = 0
        for piece in stripped:
            starts.append(offset)
            offset += len(piece) + 1
        joined = " ".join(stripped)

        found: set[int] = set()
        for lineno, line in enumerate(lines, start=1):
            if pattern.search(line):
                found.add(lineno)
        for match in pattern.finditer(joined):
            found.add(bisect.bisect_right(starts, match.start()))

        for lineno in sorted(found):
            hits.append((_rel(path), lineno, lines[lineno - 1].strip()))
    return hits


def _unexplained(
    hits: list[tuple[str, int, str]], allowlist: dict[str, str]
) -> list[tuple[str, int, str]]:
    """Hits whose file is not named in *allowlist*."""
    return [hit for hit in hits if hit[0] not in allowlist]


def _describe(hits: list[tuple[str, int, str]]) -> str:
    return "\n".join(f"  {path}:{lineno}: {line}" for path, lineno, line in hits)


# --------------------------------------------------------------------------
# SC-5 — one discovery rule, and no location flags to remember
# --------------------------------------------------------------------------


def test_every_registered_command_either_uses_the_resolver_or_is_a_recorded_exemption() -> None:
    """SC-5, structurally: no command decides for itself where the deployment is.

    The list comes from the CLI, not from this file. That is the whole point:
    a command registered tomorrow is subject to the criterion the moment it is
    registered, and the only two ways to satisfy it are to reach the shared
    resolver or to be written down in the resolver's exemption sets as a
    decision somebody made on purpose.
    """
    group = LazyGroup()

    offenders: list[str] = []
    for name in group.list_commands(ctx=None):
        if name in _EXEMPT:
            continue
        command = group.get_command(None, name)
        assert command is not None, f"{name} is listed but does not resolve"
        # ``__module__`` rather than ``__globals__``: click's own decorators
        # wrap a callback in a function DEFINED IN click, so the globals belong
        # to click.decorators while ``functools.update_wrapper`` carries the
        # original module across. Reading the globals scores every decorated
        # command against click's source and reports a uniform false failure.
        module_file = Path(sys.modules[command.callback.__module__].__file__)
        source = module_file.read_text(encoding="utf-8")
        if "find_repo_root" not in source and "repo_option" not in source:
            offenders.append(f"{name} ({module_file.name})")

    assert offenders == [], (
        "These commands neither obtain their repo through repo_resolver nor appear "
        "in any of the resolver's exemption sets:\n  "
        + "\n  ".join(offenders)
        + "\n\nEither route the command through find_repo_root, or record the "
        "exemption in repo_resolver with the reason — the point of those sets is "
        "that 'no --repo here' reads as a decision rather than an oversight."
    )


def test_the_exemption_sets_name_only_commands_that_exist() -> None:
    """An exemption for a command that is gone is a claim nothing supports.

    Checked in the same breath as the criterion above because the two fail in
    opposite directions: that one catches a command missing from the sets,
    this one catches a set entry missing from the CLI. Without it the
    exemption lists could drift into fiction and still make every other test
    pass.
    """
    registered = set(LazyGroup().list_commands(ctx=None))
    named = _EXEMPT | set(MODE_EXEMPT_FLAGS)
    assert named - registered == set(), (
        f"repo_resolver exempts commands the CLI does not register: {sorted(named - registered)}"
    )


#: Why each file may still spell ``--project``. Nothing here is a location flag
#: on a repo-scoped verb; that is what the criterion forbids.
_PROJECT_FLAG_ALLOWLIST = {
    # The flag itself, on a command the resolver names in SELF_DISCOVERING_COMMANDS:
    # channel-finder resolves its own inputs and was deliberately not rewired.
    "src/osprey/cli/channel_finder_cmd.py": "recorded exemption: SELF_DISCOVERING_COMMANDS",
    # Same set, same reason: `health` reports on a deployment, but a rendered
    # project directory is also a valid subject and holds no profile.yml, so
    # `--project` is what accepts both. It resolves through project_utils, which
    # applies find_repo_root's walk when the directory is not itself a project.
    "src/osprey/cli/health_cmd.py": "recorded exemption: SELF_DISCOVERING_COMMANDS",
    # health's own not-found diagnostic, telling the operator about health's own
    # exempt flag. Same fact as the entry above, spelled where the message is
    # written rather than where the option is declared — the command resolves the
    # enclosing repository, so "run this from the project directory" would be
    # wrong advice and --project is the accurate escape hatch.
    "src/osprey/health/core/configuration.py": "health's own message names health's own flag",
    # The shared helper those two commands resolve their --project through. Its
    # docstring names them both, so it is documentation of the exemption rather
    # than a third discovery rule.
    "src/osprey/cli/project_utils.py": "helper owned by the two exempt commands",
    # History, deliberately: the module's opening paragraph names the four rules
    # it replaced, and --project was one of them.
    "src/osprey/cli/repo_resolver.py": "narrates the retired rule it replaced",
}

#: ``--project`` not followed by ``-`` or a word character, so the docker-compose
#: flag ``--project-directory`` — which is load-bearing for the pinned compose
#: invocation and appears in a dozen templates — is not swept up with it.
_PROJECT_FLAG = re.compile(r"--project(?![-\w])")


def test_no_repo_scoped_verb_carries_a_project_flag() -> None:
    """SC-5: ``--project`` survives only on the commands the resolver exempts."""
    hits = _scan(_PROJECT_FLAG)
    assert hits, "the --project scan matched nothing at all — the pattern is broken"
    offenders = _unexplained(hits, _PROJECT_FLAG_ALLOWLIST)
    assert offenders == [], "`--project` appears outside the recorded exemptions:\n" + _describe(
        offenders
    )


def test_every_project_flag_exemption_still_has_something_to_explain() -> None:
    """A file that no longer spells ``--project`` should leave the allowlist."""
    seen = {path for path, _, _ in _scan(_PROJECT_FLAG)}
    stale = sorted(set(_PROJECT_FLAG_ALLOWLIST) - seen)
    assert stale == [], f"allowlist entries with no remaining occurrence: {stale}"


#: Compounds that are NOT the retired variable and must not be swept up with
#: it. Listed so the word-boundary semantics is asserted rather than assumed.
_PROJECT_ENV_SURVIVORS = (
    # The dispatch/compose contract: names the project directory inside a container.
    "OSPREY_PROJECT_DIR",
    # A build arg feeding the com.osprey.project image label.
    "OSPREY_PROJECT_NAME",
    # The label key reset filters containers by.
    "OSPREY_PROJECT_LABEL",
)


def test_the_retired_project_env_var_is_gone_and_its_compounds_are_not() -> None:
    """SC-5: ``OSPREY_PROJECT`` as a whole word, nowhere; its three compounds, intact.

    Both halves in one test on purpose. The criterion is only meaningful if
    the match is word-exact, and the way to show a word-exact match is working
    is to point it at three strings that CONTAIN the retired name and require
    them to survive. Split across two tests, a pattern that matched nothing
    would pass the half that matters.
    """
    hits = _scan(re.compile(r"\bOSPREY_PROJECT\b"))
    assert hits == [], "OSPREY_PROJECT is retired as a discovery input:\n" + _describe(hits)

    for survivor in _PROJECT_ENV_SURVIVORS:
        assert _scan(re.compile(rf"\b{survivor}\b")), (
            f"{survivor} has vanished. Either it was removed on purpose — in which "
            "case drop it from _PROJECT_ENV_SURVIVORS — or the word-boundary check "
            "above is no longer testing anything."
        )


#: Reading one of these from the environment is discovery; writing one is
#: publication. The redesign kept the second and deleted the first, so the
#: criterion is about reads.
_CONFIG_ENV_READ = re.compile(
    r"""environ\.get\(\s*["'](?:OSPREY_CONFIG|CONFIG_FILE)["']"""
    r"""|getenv\(\s*["'](?:OSPREY_CONFIG|CONFIG_FILE)["']"""
    # The subscript form is a read only when it is NOT the target of an
    # assignment. The lookahead sits immediately after the closing bracket and
    # swallows its own whitespace: written as `\s*(?!=...)` instead, the `\s*`
    # backtracks to zero width and the lookahead then inspects the space rather
    # than the `=`, so every publication reads as a discovery.
    r"""|environ\[\s*["'](?:OSPREY_CONFIG|CONFIG_FILE)["']\s*\](?!\s*=[^=])"""
)


def test_no_cli_module_discovers_a_deployment_from_the_environment() -> None:
    """SC-5: the config env vars are something the CLI PUBLISHES, never something it reads.

    An ambient ``OSPREY_CONFIG`` naming some other deployment must not change
    which deployment a command acts on — that is the FR-10 carve-out, and the
    difference between a publication and a discovery rule is exactly the
    difference between a read and a write.
    """
    cli = SRC / "cli"
    offenders = [
        (_rel(path), lineno, line.strip())
        for path in sorted(cli.rglob("*.py"))
        if "__pycache__" not in path.parts
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if _CONFIG_ENV_READ.search(line)
    ]
    assert offenders == [], (
        "A CLI module reads OSPREY_CONFIG/CONFIG_FILE to decide what to act on:\n"
        + _describe(offenders)
    )


#: The runtime injection sites SC-5 requires to be UNCHANGED. Each one is a
#: process that is HANDED a config path by the CLI rather than finding one, so
#: reading the variable is the whole mechanism there.
_INJECTION_SITES = (
    "registry/mcp.py",
    "mcp_env.py",
    "mcp_server/startup.py",
    "services/python_executor/execution/wrapper.py",
)


@pytest.mark.parametrize("relative", _INJECTION_SITES)
def test_the_runtime_injection_sites_still_read_the_config_env_vars(relative: str) -> None:
    """SC-5's other half: removing discovery must not have removed the handoff.

    The criterion is two-sided, and a sweep that only checked the deletions
    would score a subsystem that had lost its config path entirely as a
    perfect pass.
    """
    path = SRC / relative
    assert path.is_file(), f"{relative} has moved; SC-5's unchanged-sites list needs updating"
    text = path.read_text(encoding="utf-8")
    assert "OSPREY_CONFIG" in text or "CONFIG_FILE" in text, (
        f"{relative} no longer references either config env var, but it is one of "
        "the sites that receives its config path that way."
    )


# --------------------------------------------------------------------------
# SC-7 — nothing shipped names a command or a layout that is gone
# --------------------------------------------------------------------------

#: Spellings from the retired surface. ``osprey deploy `` catches the whole
#: deleted group at once; the bare forms catch prose that drops the program
#: name. The layout patterns catch the sibling-project shape, where a repo kept
#: its profile in ``<name>-profile/`` and its build in ``build/<name>/``.
_RETIRED_SPELLINGS = {
    "the deleted `deploy` group": re.compile(r"\bosprey deploy [a-z]"),
    # The group name ALONE, with no verb after it. `osprey deploy` reads as a
    # thing an operator types, so prose that says "``osprey deploy`` chdirs
    # first" looks like a live command while the verb-pair patterns above
    # cannot see it — the exact gap a docstring can slip through.
    # `(?![- \w])` keeps it off the verb-pair form, which is already reported by
    # the entry above and would otherwise be counted twice.
    "the deleted `deploy` group (no verb)": re.compile(r"\bosprey deploy(?![-\s\w])"),
    "the deleted `deploy` group (bare)": re.compile(
        r"\bdeploy (?:up|down|restart|seed|status|logs|passwd|clean|rebuild|nuke|prune"
        r"|scaffold|decommission|render-env-production)\b"
    ),
    # The retired `osprey build` options. `--force` has its own entry below
    # because it was written both with and without the program name; these two
    # were absorbed by `osprey set`, which writes them into the profile rather
    # than layering them onto one invocation.
    "the retired `build` options": re.compile(r"\bbuild --(?:tier|set)\b"),
    "the deleted `profile try` verb": re.compile(r"\bprofile try\b"),
    # SC-7 names three spellings, but the redesign retired more than three
    # verbs, and every one of them fails the criterion for the same reason.
    # Successors are CHANGELOG's Removed table, which is where an operator is
    # sent to look them up and therefore the only place they belong.
    "the deleted `profile new` verb": re.compile(r"\bprofile new\b"),
    "the deleted `claude` group": re.compile(r"\bosprey claude [a-z]"),
    "the deleted `config` subcommands": re.compile(
        r"\bconfig (?:set-control-system|set-epics-gateway|show|export)\b"
    ),
    # Both spellings: a click docstring IS the terminal help text, and the one
    # that named this flag wrote it without the program name.
    "the retired `build --force` flag": re.compile(r"\bbuild --force\b"),
    "the retired sibling-profile layout": re.compile(r"-profile/profile\.yml"),
    "the retired positional build invocation": re.compile(r"\bosprey build \S+ \S*profile\.ya?ml"),
}

#: One line each pattern MUST match, so every pattern proves it still works.
#:
#: A retirement criterion succeeds by finding nothing, which is also exactly what
#: a broken pattern does. Most of these now match nothing in ``src/`` — that is
#: the point of the criterion — so without this, a typo'd regex, a stray escape,
#: or an alternation that lost a branch would report a clean sweep forever.
#:
#: Per-pattern, not collective. The existing exemption-liveness check unions
#: every pattern's hits, so one broken pattern is masked by any other pattern
#: matching the same allowlisted line. Each entry below is a canned example
#: written for its own pattern and nothing else.
_RETIRED_SPELLING_EXAMPLES = {
    "the deleted `deploy` group": "run `osprey deploy up` to start it",
    "the deleted `deploy` group (no verb)": "the label ``osprey deploy`` stamps",
    "the deleted `deploy` group (bare)": "then deploy down to tear it back down",
    "the retired `build` options": "override with build --tier 3",
    "the deleted `profile try` verb": "profile try is how you preview one",
    "the deleted `profile new` verb": "osprey profile new my-facility",
    "the deleted `claude` group": "osprey claude regen rewrites the artifacts",
    "the deleted `config` subcommands": "config set-control-system epics",
    "the retired `build --force` flag": "re-run osprey build --force to rebuild",
    "the retired sibling-profile layout": "reads my-facility-profile/profile.yml",
    "the retired positional build invocation": "osprey build my-app ./my-profile/profile.yml",
}


def test_every_retirement_pattern_has_a_liveness_example() -> None:
    """No pattern may sit in the criterion without a proof that it can fire."""
    missing = sorted(set(_RETIRED_SPELLINGS) - set(_RETIRED_SPELLING_EXAMPLES))
    extra = sorted(set(_RETIRED_SPELLING_EXAMPLES) - set(_RETIRED_SPELLINGS))
    assert missing == [], f"retirement patterns with no liveness example: {missing}"
    assert extra == [], f"liveness examples for patterns that no longer exist: {extra}"


@pytest.mark.parametrize("what", sorted(_RETIRED_SPELLING_EXAMPLES))
def test_each_retirement_pattern_still_matches_its_own_example(what: str) -> None:
    """SC-7's patterns find nothing in ``src/`` — prove that is a result, not a bug.

    The failure this prevents is silent by construction: a pattern that matches
    nothing anywhere reports the same clean sweep as a pattern that is working
    and has nothing to find.
    """
    example = _RETIRED_SPELLING_EXAMPLES[what]
    assert _RETIRED_SPELLINGS[what].search(example), (
        f"pattern for {what} no longer matches its own example: {example!r}"
    )


#: The only occurrences allowed to remain, keyed by ``(file, exact line)`` and
#: not by file alone. Both files below also hold lines that WERE rewritten, so a
#: file-level exemption would stop the criterion applying to them at all — and
#: `container_lifecycle` in particular is where a new remediation string naming
#: a dead verb is most likely to be written next.
_RETIRED_SPELLING_ALLOWLIST: dict[tuple[str, str], str] = {
    # The three `.env` banner headers. `reset` removes a minted section by
    # matching its header string EXACTLY against files already on operators'
    # disks, so this text is a data format, not a message: re-wording it would
    # orphan every block written by an earlier release, leaving stale tokens in
    # place while reset reported them stripped. Written in container_lifecycle,
    # matched in reset — hence the same three strings under both files.
    **{
        (path, line): "`.env` banner header matched against files already on disk"
        for path in (
            "src/osprey/deployment/reset.py",
            "src/osprey/deployment/container_lifecycle.py",
        )
        for line in (
            '"Auto-generated service auth tokens (osprey deploy up)",',
            '"Auto-configured bluesky bridge scan devices (osprey deploy up)",',
            '"Auto-generated bluesky RE manager control-socket keypair (osprey deploy up)",',
        )
    },
    # The web-terminal auth banners, for the same reason as the three above and
    # found by the same criterion once it learned to see the group name with no
    # verb after it. `auth_credentials` locates an existing block by matching
    # these strings against the `.env.auth` on the operator's disk
    # (`if _HASH_HEADER in lines`, `lines.index(_HASH_HEADER)`), so the text is
    # a lookup key, not a message: re-word it and the next mint appends a
    # SECOND block instead of extending the one already there.
    **{
        ("src/osprey/deployment/web_terminals/auth_credentials.py", line): (
            "`.env.auth` block header matched against files already on disk"
        )
        for line in (
            '_HASH_HEADER = "# Auto-generated web-terminal auth password hashes (osprey deploy)"',
            '_SECRET_HEADER = "# Auto-generated web-terminal auth signing secrets (osprey deploy)"',
        )
    },
    # Deliberate history: the sentence exists to contrast today's read-only
    # `osprey up` with what the retired verb did, so naming it is the point.
    (
        "src/osprey/deployment/container_lifecycle.py",
        "the legacy ``deploy up``. The locations come from the rendered config's own",
    ): "narrates the retired verb it replaced",
    # A history sentence, kept deliberately: it names a dead verb in order to
    # explain where a surviving behaviour came from, and says so in the same
    # breath. Listed individually rather than matched by a keyword, because a
    # rule that excused any line containing the word "retired" would be an
    # escape hatch rather than an exemption.
    (
        "src/osprey/cli/set_cmd.py",
        "#: CLI-only shorthand absorbed from ``osprey config set-epics-gateway",
    ): "names the retired verb this shorthand replaced",
}


@pytest.mark.parametrize("what,pattern", sorted(_RETIRED_SPELLINGS.items()))
def test_no_shipped_text_names_a_command_or_layout_that_is_gone(
    what: str, pattern: re.Pattern[str]
) -> None:
    """SC-7: a string telling an operator to run a dead verb is worse than silence.

    Worse, because it costs the reader the time to try it and then the time to
    work out whether they typed it wrong. This covers templates and comments as
    well as code: a comment in a rendered ``config.yml`` reaches an operator by
    exactly the same route as an error message, and neither is documentation
    that somebody chose to open.
    """
    offenders = [
        hit for hit in _scan(pattern) if (hit[0], hit[2]) not in _RETIRED_SPELLING_ALLOWLIST
    ]
    assert offenders == [], f"Shipped text still names {what}:\n" + _describe(offenders)


def test_every_retired_spelling_exemption_still_has_something_to_explain() -> None:
    """A banner line that stops carrying the historical text should leave the list.

    Exact-line keys make this check sharp: an allowlisted line that was edited
    — even to fix a typo — stops matching and is reported, so the exemption is
    re-argued rather than inherited.
    """
    seen = {
        (path, line) for pattern in _RETIRED_SPELLINGS.values() for path, _, line in _scan(pattern)
    }
    stale = sorted(set(_RETIRED_SPELLING_ALLOWLIST) - seen)
    assert stale == [], f"allowlist entries with no remaining occurrence: {stale}"


# --------------------------------------------------------------------------
# SC-8 — the retired operations skill
# --------------------------------------------------------------------------


def test_the_deploy_ops_skill_is_gone() -> None:
    """SC-8: a skill teaching the deleted verbs cannot be allowed to reappear.

    The emitted-artifact half of SC-8 lives in ``test_init_verb`` and
    ``test_emitted_artifacts_clean``, which walk a real init'ed repo. This is
    the half those two cannot see, because a skill directory that no command
    renders leaves no trace in the repo they inspect.
    """
    skill = SRC / "templates" / "skills" / "osprey-deploy-ops"
    assert not skill.exists(), (
        f"{skill} is back. It documents the deleted `deploy` group; the lifecycle "
        "verbs it would teach are `up`, `down`, `restart`, `status` and `logs`."
    )
