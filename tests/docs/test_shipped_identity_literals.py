"""Guard: nothing OSPREY ships names a real person, site or institution.

An operator reads the framework's own prose — an agent description, a log line,
a `--help` example, a comment in a rendered compose file — as a statement about
*their* deployment. When that prose carries the name of whoever happened to
write it, it is wrong everywhere else, and it is wrong in the one place a
deployer cannot edit: files that `osprey init --force` and `profile expand`
regenerate.

The rule this pins is therefore not "fewer names" but "no identities": shipped
text under ``src/osprey/`` and ``docs/source/`` names no real person, mailbox,
institution or site. Magnitudes stay — "~135,000 documents", "~41 minutes" —
because they are facts about the software's behaviour rather than about whose
machine produced them. Examples use the placeholder cast the roster
documentation already uses: ``alice`` and ``carol`` at ``example.org``, a
facility called *Example Research Facility*, a gateway written as its own
dotted keys (``gw.example.org``), a project at ``~/my-assistant``.

Removing a literal is a one-line edit; keeping it removed is what needs a
guard, because every new pull request is an opportunity to add one back and
nothing else would notice.

**The table grows, and so does each entry's reach.** An entry starts at what
has actually been swept, because a pattern that fires on the current tree is
not a guard, it is a failing test: a literal joins :data:`DENIED` in the
change that removes it from the shipped tree, and an entry's ``roots`` widen
to :data:`REPO_ROOTS` in the change that clears it from the repository's own
tests and scripts as well.

Five kinds of surface legitimately name an institution and carry an ``allow``
entry rather than an edit: the shipped provider adapters for named LLM
gateways, the named ingestion adapter, a case that asserts the literal's
absence and so has to spell it, the project's own packaging metadata, and the
packaging and escalation metadata that has to spell the upstream project's own
``owner/repo`` — the last of these is out of scope here, because a project's
own address is not a facility's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_FILE = Path(__file__).resolve()

#: Everything OSPREY ships that a deployer or operator can read. Code and
#: comments as much as documentation: a rendered template's comment reaches a
#: deployment, and a docstring reaches the API reference.
SHIPPED_ROOTS = ("src/osprey", "docs/source")

#: The trees that describe what ships without shipping themselves. An identity
#: reaches them once the repo is clean of it there, so a sweep stays swept
#: rather than merely performed.
REPO_ROOTS = SHIPPED_ROOTS + ("tests", "scripts")

#: Every extension under the roots that holds text a person reads. A comment
#: in a test module states as much about a deployment as a paragraph of
#: documentation does, so an extension belongs here whenever the repository
#: writes prose in it — not only when the file's purpose is prose.
SCAN_SUFFIXES = (
    ".py",
    ".md",
    ".rst",
    ".txt",
    ".j2",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".sh",
    ".js",
    ".mjs",
    ".html",
    ".css",
)

#: The filenames that carry a reader's text under no extension at all. A
#: container recipe's comments reach whoever builds the image exactly as a
#: template's reach whoever renders it, and its name is the only extension
#: it has.
SCAN_NAMES = ("Dockerfile", "Containerfile")


@dataclass(frozen=True)
class Denied:
    """One identity the shipped tree may not name."""

    name: str
    """How the guard's failure message calls it."""

    pattern: re.Pattern[str]
    """What a reintroduction looks like."""

    why: str
    """What is wrong with shipping it, in one clause."""

    sample: str
    """A line this pattern must match, so a broken pattern cannot read as a
    clean tree."""

    roots: tuple[str, ...] = SHIPPED_ROOTS
    """The trees this identity may not appear in."""

    allow: frozenset[str] = frozenset()
    """Repo-relative paths that may keep it, each for a stated reason."""


#: Each identity the guard holds out of the trees its ``roots`` name. An entry
#: is here because those trees are clean of it, and its ``why`` is what is
#: wrong with putting it back.
DENIED: tuple[Denied, ...] = (
    Denied(
        name="maintainer account",
        pattern=re.compile(r"thellert", re.IGNORECASE),
        why="a maintainer's own login and mailbox is not an example anyone can copy",
        sample='  # bare username, e.g. "thellert"',
        roots=REPO_ROOTS,
        # A case that asserts the literal's absence has to spell it.
        allow=frozenset({"tests/integration/test_preset_static.py"}),
    ),
    Denied(
        name="site EPICS gateway host",
        pattern=re.compile(r"cagw-alsdmz|pvgatemain1", re.IGNORECASE),
        why=(
            "a gateway address is one site's own infrastructure — unreachable "
            "everywhere else, and read as this deployment's own machine when it "
            "ships in an example"
        ),
        sample="  epics_gateway: cagw-alsdmz.example-site.org:5064",
        roots=REPO_ROOTS,
        # A case that asserts the literal's absence has to spell it.
        allow=frozenset({"tests/mcp_server/test_phoebus_plt_generator.py"}),
    ),
    Denied(
        name="maintainers' gateway host",
        pattern=re.compile(r"gianluca[-.]?martino", re.IGNORECASE),
        why="a maintainer's own gateway endpoint is not one another deployment can call",
        sample="  base_url: https://llm.gianluca-martino.com/v1",
        roots=REPO_ROOTS,
    ),
    Denied(
        name="institutional domain",
        pattern=re.compile(r"lbl\.gov", re.IGNORECASE),
        why="an institution's own domain is not an example anyone else can copy",
        sample='"""Prefix of a principal naming an identity domain: ``domain:lbl.gov``."""',
        roots=REPO_ROOTS,
        # Each exemption names the gateway provider OSPREY ships an adapter
        # for, or the reference ingestion adapter — surfaces where the
        # institution is the subject rather than the example.
        allow=frozenset(
            {
                "docs/source/getting-started/installation.rst",
                "docs/source/how-to/llm-providers/configure-providers.rst",
                "docs/source/how-to/llm-providers/run-open-models.rst",
                "src/osprey/build/claude_code_resolver.py",
                "src/osprey/models/providers/als_apg.py",
                "src/osprey/models/providers/cborg.py",
                "src/osprey/profiles/providers.yml",
                "src/osprey/services/ariel_search/ingestion/adapters/als.py",
                "src/osprey/services/channel_finder/benchmarks/evaluation.py",
                # A case that asserts the literal's absence has to spell it.
                "tests/integration/test_preset_static.py",
                # A case that asserts the literal is ignored has to spell it.
                "tests/docs/test_linkcheck_ignore.py",
                # The named gateway the shipped adapter fronts. Each of these
                # asserts a value the packaged provider catalog supplies, so a
                # rewrite here would pin an address no deployment renders.
                "tests/cli/test_base_url_override.py",
                "tests/cli/test_chat_verb.py",
                "tests/cli/test_claude_code_resolver.py",
                "tests/cli/test_init_providers.py",
                "tests/cli/test_provider_isolation.py",
                "tests/cli/test_resolver_cborg_oss.py",
                "tests/deployment/goldens/exemplar-profile/providers.yml",
                "tests/models/test_completion.py",
                "tests/models/test_providers_litellm_delegating.py",
                # The harness and the live lanes that call that gateway, where
                # the address is the endpoint under test rather than an example.
                "scripts/benchmark/README.md",
                "scripts/benchmark/matrix.yaml",
                "scripts/benchmark/matrix_curate_models.py",
                "scripts/benchmark/matrix_run.py",
                "tests/benchmark/test_matrix.py",
                "tests/e2e/claude_code/test_proxy_live_roundtrip_e2e.py",
                "tests/e2e/claude_code/test_proxy_open_model_harness_e2e.py",
                "tests/manual/test_sdk_image_block.py",
            }
        ),
    ),
    Denied(
        name="ring name",
        pattern=re.compile(r"ALS-U", re.IGNORECASE),
        why=(
            "one laboratory's ring is the bundled demo lattice, not a machine "
            "another deployment's prose should describe as its own"
        ),
        sample="# \u2500\u2500 The ALS-U Accumulator Ring instance \u2500\u2500",
        roots=REPO_ROOTS,
        # The demo ring ships as the simulation and virtual-accelerator
        # packages' own subject.
        allow=frozenset(
            {
                "src/osprey/services/channel_finder/naming.py",
                "src/osprey/services/virtual_accelerator/model/bindings.py",
                "src/osprey/simulation/channel_schema.py",
                "src/osprey/simulation/facility_spec.py",
                "src/osprey/simulation/lattice/__init__.py",
                "src/osprey/simulation/lattice/artifact.py",
                "src/osprey/simulation/lattice/ring.py",
                # The suites that exercise those packages: the ring is what they
                # are a test of.
                "tests/simulation/matlab_reference.py",
                "tests/simulation/test_artifact.py",
                "tests/simulation/test_facility_spec.py",
                "tests/simulation/test_fidelity.py",
                "tests/simulation/test_lattice.py",
                "tests/simulation/test_orbit_closure.py",
                "tests/va/e2e/test_orbit_response.py",
            }
        ),
    ),
    Denied(
        # Case-sensitive on purpose: the acronym is always capitalised, while
        # a case-insensitive read also matches the project's own ``als-apg``
        # slug and the ``als-assistant`` example paths across the shipped tree.
        name="facility abbreviation",
        pattern=re.compile(r"\bALS\b"),
        why=(
            "one laboratory's abbreviation reads as this deployment's own "
            "facility wherever the prose ships"
        ),
        sample="#   facility_name: ALS",
        roots=REPO_ROOTS,
        # The bundled demo ring, whose own name this is, in the simulation and
        # virtual-accelerator packages plus the two files that name the lattice
        # they load.
        allow=frozenset(
            {
                "src/osprey/services/channel_finder/naming.py",
                "src/osprey/services/virtual_accelerator/model/bindings.py",
                "src/osprey/simulation/channel_schema.py",
                "src/osprey/simulation/facility_spec.py",
                "src/osprey/simulation/lattice/__init__.py",
                "src/osprey/simulation/lattice/artifact.py",
                "src/osprey/simulation/lattice/ring.py",
                # The shipped reference ingestion format, and the places that
                # quote the ``source_system`` values its adapter returns —
                # rewriting those would name a value no adapter produces.
                "docs/source/how-to/ariel/data-ingestion.rst",
                "docs/source/reference/contracts/ariel.rst",
                "src/osprey/registry/builtins.py",
                "src/osprey/services/ariel_search/ingestion/adapters/als.py",
                "src/osprey/services/ariel_search/ingestion/base.py",
                "src/osprey/services/ariel_search/models.py",
                # The shipped named-gateway provider adapter, and the two
                # surfaces that name the gateway it fronts.
                "docs/source/how-to/llm-providers/configure-providers.rst",
                "src/osprey/models/providers/als_apg.py",
                "src/osprey/services/channel_finder/benchmarks/evaluation.py",
                # The suites that exercise the bundled demo ring, whose name carries
                # the abbreviation.
                "tests/simulation/matlab_reference.py",
                "tests/simulation/test_artifact.py",
                "tests/simulation/test_facility_spec.py",
                "tests/simulation/test_fidelity.py",
                "tests/simulation/test_lattice.py",
                "tests/simulation/test_orbit_closure.py",
                "tests/va/e2e/test_orbit_response.py",
                # The suites for the shipped reference ingestion adapter, which
                # returns "ALS eLog": an expectation spelled any other way would
                # assert a value no adapter produces.
                "tests/services/ariel_search/conftest.py",
                "tests/services/ariel_search/integration/test_cli.py",
                "tests/services/ariel_search/integration/test_ingestion.py",
                "tests/services/ariel_search/test_ingestion.py",
                "tests/services/ariel_search/test_ingestion_branches.py",
                # The suite that asserts on the shipped named-gateway adapter's
                # description.
                "tests/models/test_providers_litellm_delegating.py",
                # Cases that assert the literal's absence, and so have to spell it.
                # The third holds the roster of real facility names an invented
                # export may not carry, which is that assertion written as a
                # pattern.
                "tests/dispatch/test_dashboard_config_injection.py",
                "tests/registry/test_pyat_specialist_agent.py",
                "tests/services/mml/test_fixtures_wellformed.py",
                # A byte-faithful copy of the shipped plugin manifest, whose author
                # field is the project's own.
                "tests/scripts/test_plugin_version.py",
            }
        ),
    ),
    Denied(
        # Case-sensitive on purpose: the acronym is always written in capitals,
        # and a case-insensitive read matches a word inside the vendored
        # plotly bundles, which would tie an exemption to a pinned version.
        name="named external installation",
        pattern=re.compile(r"BELLA"),
        why="another site's installation is that site's own facility, not a shipped example",
        sample="sends. Mirrors BELLA's ``runs.require_armed`` / ``launch_intent``",
        roots=REPO_ROOTS,
        # Both name the upstream contract these modules were generalized from.
        allow=frozenset(
            {
                "src/osprey/services/bluesky_bridge/live_rows.py",
                "src/osprey/services/bluesky_bridge/security.py",
            }
        ),
    ),
    Denied(
        name="named external control system",
        pattern=re.compile(r"GEECS", re.IGNORECASE),
        why="another site's control system is that site's own stack, not a shipped example",
        sample="parameter, so a document-shaped parameter (a GEECS ``ScanRequest``, say)",
        roots=REPO_ROOTS,
        # Each names the document-shaped scan parameter this code accepts, by
        # the upstream system the shape comes from.
        allow=frozenset(
            {
                "src/osprey/cli/build_profile_schema.py",
                "src/osprey/services/bluesky_bridge/live_rows.py",
                "src/osprey/services/bluesky_bridge/queue_backend.py",
            }
        ),
    ),
)


@cache
def _sources(repo_root: Path, roots: tuple[str, ...]) -> tuple[Path, ...]:
    """Every scannable file under *roots*, minus this module.

    This module is the table itself — it spells every pattern and a sample
    line for each — so scanning it would report the rule as its own
    violation. The cache is keyed on the root as well as the tuple, so a
    test that repoints :data:`_REPO_ROOT` at a fixture tree gets its own
    listing rather than the repository's.
    """
    files: list[Path] = []
    for root in roots:
        base = repo_root / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in SCAN_SUFFIXES and path.name not in SCAN_NAMES:
                continue
            if "__pycache__" in path.parts or path == _THIS_FILE:
                continue
            files.append(path)
    return tuple(files)


def _hits(denied: Denied) -> list[tuple[str, int, str]]:
    """Every ``(repo-relative path, line number, stripped line)`` naming it."""
    found: list[tuple[str, int, str]] = []
    for path in _sources(_REPO_ROOT, denied.roots):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - defensive
            continue
        if not denied.pattern.search(content):
            continue
        relative = str(path.relative_to(_REPO_ROOT))
        for number, line in enumerate(content.splitlines(), start=1):
            if denied.pattern.search(line):
                found.append((relative, number, line.strip()))
    return found


@pytest.mark.parametrize("denied", DENIED, ids=lambda d: d.name)
def test_no_shipped_text_names_a_real_identity(denied: Denied) -> None:
    offenders = [hit for hit in _hits(denied) if hit[0] not in denied.allow]
    assert offenders == [], (
        f"shipped text names {denied.name} — {denied.why}. "
        f"Use the placeholder cast (alice/carol@example.org, Example Research "
        f"Facility, ~/my-assistant, simulation) instead:\n"
        + "\n".join(f"{path}:{number}: {line}" for path, number, line in offenders)
    )


@pytest.mark.parametrize("denied", DENIED, ids=lambda d: d.name)
def test_every_exemption_still_has_something_to_explain(denied: Denied) -> None:
    """An exemption whose file stopped carrying the literal should be deleted.

    Otherwise the allowlist quietly becomes a list of files the guard does not
    cover, which is the failure mode a guard is written to prevent.
    """
    seen = {path for path, _, _ in _hits(denied)}
    stale = sorted(denied.allow - seen)
    assert stale == [], f"{denied.name}: exemptions with no remaining occurrence: {stale}"


@pytest.mark.parametrize("denied", DENIED, ids=lambda d: d.name)
def test_the_sweep_would_catch_a_regression(
    denied: Denied, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep that finds nothing reads exactly like one whose pattern is dead.

    Each entry carries the line it must trip on, so a pattern that is edited
    into uselessness fails here rather than reporting a clean tree forever.
    """
    fake_root = tmp_path / "repo"
    (fake_root / "src" / "osprey").mkdir(parents=True)
    (fake_root / "src" / "osprey" / "regress.py").write_text(
        f"# {denied.sample}\n", encoding="utf-8"
    )
    monkeypatch.setattr(f"{__name__}._REPO_ROOT", fake_root, raising=True)

    with pytest.raises(AssertionError, match="shipped text names"):
        test_no_shipped_text_names_a_real_identity(denied)


def test_the_placeholder_cast_is_not_swept() -> None:
    """The names the rule tells an author to use must survive the rule.

    A guard that condemned its own replacement text would push the next author
    back to a real one.
    """
    placeholders = (
        "alice@example.org",
        "carol@example.org",
        "Example Research Facility",
        "~/my-assistant",
        "config.control_system.connector.epics.gateways.read_only.address=gw.example.org",
    )
    for denied in DENIED:
        for placeholder in placeholders:
            assert not denied.pattern.search(placeholder), (
                f"{denied.name} condemns the placeholder {placeholder!r}"
            )


def test_the_sweep_reaches_shipped_templates_and_docs() -> None:
    """The guard has to cover the surfaces the literals actually lived in.

    A rendered template's comment and a how-to page are exactly where an
    identity survives review, and both would be missed by a rule written for
    ``.py`` alone. A container recipe is the same surface under no extension
    at all.
    """
    scanned = {str(path.relative_to(_REPO_ROOT)) for path in _sources(_REPO_ROOT, SHIPPED_ROOTS)}
    for required in (
        "src/osprey/templates/modules/web_terminals/docker-compose.web.yml.j2",
        "src/osprey/profiles/presets/control-assistant.yml",
        "docs/source/how-to/web-terminal/multi-user/login.rst",
        "src/osprey/templates/services/qmd/Dockerfile",
    ):
        assert required in scanned, f"{required} is shipped but the sweep cannot see it"


def test_the_table_is_not_scanned_as_prose() -> None:
    """This module spells every pattern and a sample line for each.

    Scanned alongside the trees it guards, it would report the rule as its
    own violation, and each entry would need an exemption for the table
    that defines it.
    """
    assert _THIS_FILE not in _sources(_REPO_ROOT, REPO_ROOTS)


def test_an_entry_on_the_repo_roots_reaches_them() -> None:
    """A widened entry has to scan the trees it widened onto.

    Dropping ``tests`` or ``scripts`` from the tuple would read exactly like
    a clean repository rather than like a guard that stopped looking, and
    dropping a suffix reads the same way: a tree that carries no denied
    identity is indistinguishable from one the sweep never opened.
    """
    assert any(denied.roots == REPO_ROOTS for denied in DENIED)

    scanned = {str(path.relative_to(_REPO_ROOT)) for path in _sources(_REPO_ROOT, REPO_ROOTS)}
    for required in (
        "tests/services/bluesky_bridge/test_live_rows.py",
        "tests/mcp_server/test_phoebus_plt_generator.py",
        "scripts/qmd_probe/export_corpus.py",
        "tests/vitest.setup.mjs",
    ):
        assert required in scanned, f"{required} is in the repo but the sweep cannot see it"
