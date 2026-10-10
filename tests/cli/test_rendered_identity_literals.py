"""Guard: a deployment rendered from a shipped preset names no real identity.

A deployer does not read OSPREY's source tree; they read what ``osprey init``
and ``osprey build`` write into their repository. That output names no real
person, site or institution, for the same reason the shipped source names none:
whatever it says, a deployer reads as a statement about their own facility.

The source guard (:mod:`tests.docs.test_shipped_identity_literals`) cannot
cover this on its own. Its exemptions are keyed on the source path where an
institution is the subject, and nothing follows that text into the files the
render copies it to. Text composed at render time — the emitter's commented
templates, preset comments copied into ``profile.yml``, persona renders, data
bundles, the ``build/.image`` copies — never exists as a source file at all.
So this module materialises every root preset the way a deployer does and
walks every text file of the repository that results, matching each line
against the source guard's own table (imported, never copied) plus the entries
that are denied in renders only. A literal that joins the source table is
refused in renders from then on, with no edit here.

Two things in a render are allowed rather than fixed:

* **The named gateways' endpoints.** The render copies the provider catalog
  into ``providers.yml`` and into the ``api.providers`` block of every
  ``config.yml``, so the ``base_url`` of each named gateway OSPREY ships an
  adapter for travels with it. The allowance is that exact value, read from
  the packaged catalog at test time, and only inside a file that carries a
  copy of the catalog; the rest of the line, and every other file, is scanned
  in full. An allowance no render uses any more fails its own test.
* **Host paths.** A ``--skip-deps`` build writes the interpreter that ran it
  (``sys.executable``, from the ``elif skip_deps:`` branch of the build
  command) and the repository's absolute path into its settings, compose,
  manifest and environment files. Those strings are facts about the host that
  ran the build, not text OSPREY ships, and on a developer machine they carry
  the login. The scan replaces exactly those strings with a neutral token
  before matching; a self-test proves the redaction neither hides an identity
  elsewhere on the line nor reports one inside the path.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile_presets import _read_preset_document, list_presets
from osprey.cli.init_cmd import init
from osprey.profiles.providers import PROVIDERS_FILENAME, load_provider_catalog
from tests.docs.test_shipped_identity_literals import DENIED, Denied

#: Identities denied in a render but not in source. The framework's
#: documentation cites the work it came from on purpose; a facility's
#: deployment has no reason to. ``roots`` and ``allow`` are the source guard's
#: fields and are not read here.
RENDER_ONLY_DENIED: tuple[Denied, ...] = (
    Denied(
        name="provenance citation",
        pattern=re.compile(r"\barxiv\b", re.IGNORECASE),
        why=(
            "where the framework came from belongs in its documentation, "
            "not in a facility's deployment"
        ),
        sample="# Method after arXiv:2501.00001, section 3",
    ),
)

RENDERED_DENIED: tuple[Denied, ...] = DENIED + RENDER_ONLY_DENIED


def _root_presets() -> tuple[str, ...]:
    """Every shipped preset that extends no other; personas render inside these."""
    return tuple(name for name in list_presets() if "extends" not in _read_preset_document(name)[0])


def _persona_presets() -> dict[str, str]:
    """Each preset that extends a root, mapped to the root it extends."""
    personas: dict[str, str] = {}
    for name in list_presets():
        parent = _read_preset_document(name)[0].get("extends")
        if parent:
            personas[name] = str(parent)
    return personas


ROOT_PRESETS = _root_presets()

#: Rendered basenames that carry a copy of the provider catalog.
CATALOG_COPIES = frozenset({PROVIDERS_FILENAME, "config.yml"})

#: Denied entry name -> catalog providers whose ``base_url`` may carry it.
#: These are the named gateways OSPREY ships adapters for; the render copies
#: their endpoints wherever it copies the catalog, just as the source guard
#: allows the packaged catalog itself to spell them.
CATALOG_ENDPOINTS: dict[str, frozenset[str]] = {
    "institutional domain": frozenset({"cborg", "als-apg"}),
}

HOST_TOKEN = "<host path>"


def _catalog_allowances() -> dict[str, frozenset[str]]:
    """Each denied entry name -> the catalog ``base_url`` values allowed to carry it."""
    entries = load_provider_catalog(None).entries
    return {
        name: frozenset(str(entries[provider]["base_url"]) for provider in providers)
        for name, providers in CATALOG_ENDPOINTS.items()
    }


@dataclass(frozen=True)
class Hit:
    """One rendered line naming a denied identity."""

    denied: str
    path: str
    line: int
    text: str


@dataclass(frozen=True)
class Render:
    """A materialised preset and what the scan found in it."""

    repo: Path
    hits: list[Hit]
    used: set[tuple[str, str]]


def _text_files(repo: Path) -> Iterator[tuple[Path, str]]:
    """Every text file under ``repo`` with its content.

    Text is decided by content, not by name: a render carries text under names
    the source guard's suffix list does not cover (``.example`` and ``.env``
    files, ``.gitignore``, ``.dockerignore``, ontology, MATLAB and CSV data).
    A file whose first 8192 bytes hold a NUL is binary and skipped.
    """
    for path in sorted(repo.rglob("*")):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if b"\0" in raw[:8192]:
            continue
        yield path, raw.decode("utf-8", errors="ignore")


def _scan(repo: Path, host_values: Sequence[str]) -> tuple[list[Hit], set[tuple[str, str]]]:
    """Every rendered line naming a denied identity, and the allowances used."""
    hosts = sorted({value for value in host_values if value}, key=len, reverse=True)
    allowances = _catalog_allowances()
    hits: list[Hit] = []
    used: set[tuple[str, str]] = set()
    for path, text in _text_files(repo):
        for host in hosts:
            text = text.replace(host, HOST_TOKEN)
        catalog_copy = path.name in CATALOG_COPIES
        relative = str(path.relative_to(repo))
        for number, line in enumerate(text.splitlines(), start=1):
            for denied in RENDERED_DENIED:
                if not denied.pattern.search(line):
                    continue
                probe = line
                if catalog_copy:
                    for value in allowances.get(denied.name, frozenset()):
                        if value in probe:
                            used.add((denied.name, value))
                            probe = probe.replace(value, "")
                if denied.pattern.search(probe):
                    hits.append(Hit(denied.name, relative, number, line.strip()))
    return hits, used


def _render(repo: Path, preset: str) -> None:
    """Materialise ``preset`` at ``repo`` as a deployer's ``--skip-deps`` build does."""
    runner = CliRunner()
    result = runner.invoke(init, [str(repo), "--preset", preset, "--no-git"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(build, ["--repo", str(repo), "--skip-deps"])
    assert result.exit_code == 0, result.output


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, Render]:
    """Every root preset rendered once, with its scan."""
    renders: dict[str, Render] = {}
    for preset in ROOT_PRESETS:
        # Named after the preset so persona renders land at build/<persona>.
        repo = tmp_path_factory.mktemp("rendered") / preset
        _render(repo, preset)
        hits, used = _scan(repo, (str(repo), str(repo.resolve()), sys.executable))
        renders[preset] = Render(repo=repo, hits=hits, used=used)
    return renders


def _report(hits: Sequence[Hit]) -> str:
    by_entry = {denied.name: denied for denied in RENDERED_DENIED}
    blocks = []
    for name in sorted({hit.denied for hit in hits}):
        lines = [f"{h.path}:{h.line}: {h.text}" for h in hits if h.denied == name]
        blocks.append(
            f"rendered deployment names {name} — {by_entry[name].why}. "
            f"Use the placeholder cast (alice/carol@example.org, Example Research "
            f"Facility, ~/my-assistant, simulation) instead:\n" + "\n".join(lines)
        )
    return "\n\n".join(blocks)


# The render-backed checks are marked slow; the unit lane deselects only `pty`,
# so they still run on every pull request.
@pytest.mark.slow
@pytest.mark.parametrize("preset", ROOT_PRESETS, ids=list(ROOT_PRESETS))
def test_no_rendered_file_names_a_real_identity(
    rendered: Mapping[str, Render], preset: str
) -> None:
    hits = rendered[preset].hits
    assert hits == [], _report(hits)


@pytest.mark.slow
def test_every_catalog_allowance_is_still_rendered(rendered: Mapping[str, Render]) -> None:
    """An allowance naming a value no render carries should be deleted."""
    used = set().union(*(render.used for render in rendered.values()))
    expected = {(name, value) for name, values in _catalog_allowances().items() for value in values}
    stale = sorted(expected - used)
    assert stale == [], f"catalog allowances no render uses: {stale}"


@pytest.mark.slow
def test_every_persona_renders_inside_its_root_preset(rendered: Mapping[str, Render]) -> None:
    personas = _persona_presets()
    assert personas, "no shipped preset extends a root preset"
    missing = sorted(
        persona
        for persona, root in personas.items()
        if not (rendered[root].repo / "build" / persona).is_dir()
    )
    assert missing == [], f"persona renders not under their root's build/: {missing}"


@pytest.mark.slow
def test_the_walk_reaches_every_surface_a_deployer_reads(
    rendered: Mapping[str, Render],
) -> None:
    repo = rendered["control-assistant"].repo
    walked = {str(path.relative_to(repo)) for path, _ in _text_files(repo)}
    for surface in (
        "profile.yml",
        "providers.yml",
        ".env.example",
        "build/data/graph/facility.ttl",
        "build/config.yml",
        "build/.claude/settings.json",
        "build/.mcp.json",
        "build/Dockerfile",
        "build/.image/control-assistant/providers.yml",
        "build/.image/control-assistant/profile.yml",
        "build/control-assistant-readonly/config.yml",
    ):
        assert surface in walked, f"the walk misses {surface}"
    binary = "build/data/channel_databases/graph.duckdb"
    assert (repo / binary).is_file(), f"{binary} was not rendered"
    assert binary not in walked, f"the walk reads binary {binary}"


def test_every_catalog_allowance_names_a_catalog_endpoint_its_entry_denies() -> None:
    by_entry = {denied.name: denied for denied in RENDERED_DENIED}
    entries = load_provider_catalog(None).entries
    for name, providers in CATALOG_ENDPOINTS.items():
        assert name in by_entry, f"allowance keyed on unknown entry {name!r}"
        for provider in providers:
            assert provider in entries, f"allowance names unknown provider {provider!r}"
            base_url = str(entries[provider]["base_url"])
            assert by_entry[name].pattern.search(base_url), (
                f"{provider} base_url {base_url!r} does not carry {name}"
            )


@pytest.mark.parametrize("denied", RENDERED_DENIED, ids=lambda d: d.name)
def test_the_scan_would_catch_a_regression(denied: Denied, tmp_path: Path) -> None:
    line = f"{sys.executable} {denied.sample}\n"
    for name in ("CLAUDE.md", "config.yml"):
        (tmp_path / name).write_text(line, encoding="utf-8")
    hits, _ = _scan(tmp_path, (sys.executable,))
    found = {hit.path for hit in hits if hit.denied == denied.name}
    assert found == {"CLAUDE.md", "config.yml"}


@pytest.mark.parametrize("denied", RENDERED_DENIED, ids=lambda d: d.name)
def test_host_paths_neither_hide_nor_invent_an_identity(denied: Denied, tmp_path: Path) -> None:
    match = denied.pattern.search(denied.sample)
    assert match is not None
    host = f"/opt/{match.group(0)}/venv/bin/python"
    target = tmp_path / "settings.json"

    target.write_text(f'"command": "{host}"\n', encoding="utf-8")
    hits, _ = _scan(tmp_path, (host,))
    assert hits == []

    target.write_text(f'"command": "{host}" {denied.sample}\n', encoding="utf-8")
    hits, _ = _scan(tmp_path, (host,))
    assert {hit.denied for hit in hits} >= {denied.name}
