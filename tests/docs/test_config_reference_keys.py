"""Every config key the reference section documents must really exist.

``docs/source/reference/configuration/`` is the operator-facing catalogue of
``config.yml`` settings, and the CLI reference, the ports page and the contract
pages name keys the same way. A key that appears there and nowhere in the shipped
code is worse than an undocumented one: the reader copies it into their own
config, nothing happens, and there is no error to tell them why. The reverse
drift is just as quiet — a key gets renamed or retired in the code, the guard
manifest records that, and the reference page keeps advertising the old
spelling forever.

This sweep closes that gap by cross-checking the two artefacts that already
exist for their own reasons:

* the reference pages, where a documented key is written as an RST inline
  literal — ``web.theme``, ``services.postgresql.port`` — and
* ``src/osprey/profiles/config_key_manifest.yml``, the config-key resurrection guard's
  ledger, whose top-level ``keys:`` mapping is the authoritative list of every
  dotted path the shipped templates render and the code reads. That manifest is
  independently enforced against the source tree by
  ``scripts/check_config_keys.py``, so "declared there" is a claim with teeth
  rather than a second list to keep in sync by hand.

A key listed under the manifest's ``deleted:`` section counts as undeclared —
that section exists precisely to record spellings that must not come back, and
documenting one is the resurrection the guard is named after.

Scope is deliberately narrow. Only inline literals whose whole content is a
dotted path rooted at one of the known top-level config sections are checked,
so ordinary prose, file paths, CLI verbs, environment variables and YAML
snippets in code blocks pass by untouched (``test_config_yaml_blocks`` reads
those). The directory is allowed not to
exist yet: the reference section is being built out, and a sweep that hard
-failed on an absent tree would block its own scaffolding.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from osprey.cli.build_profile_load import _KNOWN_DISPATCH_KEYS

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The reference pages this sweep reads, relative to the repo root. Absent
#: today; created by the docs restructure. See the module docstring on why an
#: absent directory is a vacuous pass rather than a failure.
_REFERENCE_DIR = "docs/source/reference/configuration"

#: The guard manifest, relative to the repo root. It ships inside the package,
#: beside providers.yml: `osprey config --defaults` renders its `default:`
#: column at run time and a wheel carries only src/osprey.
_MANIFEST_PATH = "src/osprey/profiles/config_key_manifest.yml"

#: The pages this sweep reads beside the configuration reference, relative to
#: the repo root. The CLI reference, the ports page and the contract pages
#: name config keys as inline literals too (``deployment.port_base``,
#: for one), and a reader copies them just the same.
_EXTRA_PAGES = ("docs/source/reference/cli.rst", "docs/source/reference/ports.rst")
_EXTRA_GLOBS = ("docs/source/reference/contracts/*.rst",)


def _manifest_sections() -> tuple[str, ...]:
    """The first segment of every key the manifest declares."""
    path = _REPO_ROOT / _MANIFEST_PATH
    keys = yaml.safe_load(path.read_text(encoding="utf-8"))["keys"]
    return tuple(sorted({str(key).split(".")[0] for key in keys}))


#: Top-level sections of ``config.yml``. A dotted literal has to start with one
#: of these to be treated as a config key at all — the bar that keeps prose
#: like ``osprey.version`` or ``docs.source`` out of the sweep.
#:
#: Read off the manifest rather than listed by hand: a hand list missed every
#: section added after it was written, and a section missing here turns its
#: keys into prose the sweep never looks at (``modules.web_terminals.*`` was
#: one). ``facility`` is the one addition: the manifest declares no
#: ``facility.*`` key, so a ``facility.*`` literal is a key the code does not
#: read and must be flagged, not passed as prose. The pattern below anchors a
#: literal dot after the section name, so the entry cannot swallow
#: ``facility_knowledge.*`` or the retired ``facility_name``.
_SECTIONS = tuple(sorted({*_manifest_sections(), "facility"}))

#: An RST inline literal: ``like this``. Content may not span lines or contain
#: a backtick, which is exactly what Sphinx itself accepts.
_LITERAL_PATTERN = re.compile(r"``([^`\n]+)``")

#: A literal that is entirely a config key — a known section followed by at
#: least one dotted segment. Anchored at both ends so ``web.theme:`` (a YAML
#: fragment) or ``web.theme is set`` (prose that happens to be in a literal)
#: are not mistaken for a bare key reference.
_KEY_PATTERN = re.compile(r"^(?:" + "|".join(_SECTIONS) + r")\.[a-z0-9_.-]+$")

#: Documented keys that are deliberately absent from the manifest, each with
#: the reason it is allowed to be. Keyed by the dotted key so that an edit to
#: the surrounding prose does not silently inherit the exemption. Keep it
#: short: an exemption here is a promise to a reader that the key works, made
#: without the guard's evidence behind it, so every entry needs an argument.
_EXEMPTIONS: dict[str, str] = {
    "api.js": "a file of the ARIEL frontend, named in the contract's module table",
    "facility.json": "a file name on the channel-finder contract page, not a key",
    "ariel.database.uri": (
        "recorded as deleted so no preset renders it live, but an explicit uri "
        "is still read and wins (services/ariel_search/config.py); the "
        "manifest's deleted_commented_examples names the same exception"
    ),
}


#: Build-profile blocks that share their name with a config section. The
#: profile page documents their keys (``dispatch.worker_port_stride``) as the
#: profile loader reads them, so a literal naming one of the loader's keys is
#: not a config.yml key at all; any other spelling is still checked.
_PROFILE_BLOCK_KEYS: dict[str, frozenset[str]] = {"dispatch": _KNOWN_DISPATCH_KEYS}


def _is_profile_block_key(key: str) -> bool:
    """Whether *key* is ``<block>.<key>`` of a profile block the loader closes."""
    block, _, rest = key.partition(".")
    return rest in _PROFILE_BLOCK_KEYS.get(block, frozenset())


def _reference_pages(root_dir: Path | None = None) -> list[Path]:
    """Every ``.rst`` page in the configuration reference, and the extra pages."""
    root = root_dir if root_dir is not None else _REPO_ROOT
    base = root / _REFERENCE_DIR
    pages = sorted(base.glob("*.rst")) if base.is_dir() else []
    pages += [root / page for page in _EXTRA_PAGES if (root / page).is_file()]
    for pattern in _EXTRA_GLOBS:
        pages += sorted(root.glob(pattern))
    return pages


def _documented_keys(root_dir: Path | None = None) -> list[tuple[str, int, str]]:
    """Every ``(repo-relative path, line number, key)`` documented as a literal."""
    base_root = root_dir if root_dir is not None else _REPO_ROOT
    found: list[tuple[str, int, str]] = []
    for path in _reference_pages(root_dir):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - defensive, no such file today
            continue
        relative = str(path.relative_to(base_root))
        for number, line in enumerate(content.splitlines(), start=1):
            for match in _LITERAL_PATTERN.finditer(line):
                literal = match.group(1).strip()
                if _KEY_PATTERN.match(literal):
                    found.append((relative, number, literal))
    return found


def _manifest(root_dir: Path | None = None) -> dict[str, object]:
    """The guard manifest, parsed."""
    path = (root_dir if root_dir is not None else _REPO_ROOT) / _MANIFEST_PATH
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{_MANIFEST_PATH} did not parse as a mapping"
    return loaded


def _declared_keys(root_dir: Path | None = None) -> set[str]:
    """Every dotted path declared in the manifest's top-level ``keys:`` mapping."""
    keys = _manifest(root_dir).get("keys")
    assert isinstance(keys, dict), f"{_MANIFEST_PATH} has no top-level `keys:` mapping"
    return set(keys)


def _deleted_keys(root_dir: Path | None = None) -> set[str]:
    """Every dotted path the manifest records as retired and not to come back."""
    deleted = _manifest(root_dir).get("deleted") or []
    return {str(entry["path"] if isinstance(entry, dict) else entry) for entry in deleted}


def _undeclared_hits(
    docs_root: Path | None = None,
    manifest_root: Path | None = None,
) -> list[tuple[str, int, str]]:
    """Documented keys with no declaration in the manifest.

    The two roots are separate on purpose. A negative control needs a fake docs
    tree checked against the *real* manifest — a fake manifest would only prove
    the two fakes disagree, which is not the property under test.
    """
    declared = _declared_keys(manifest_root)
    return [hit for hit in _documented_keys(docs_root) if hit[2] not in declared]


def test_every_documented_config_key_is_declared_in_the_manifest() -> None:
    """The rule: the reference section may only name keys the code really reads."""
    deleted = _deleted_keys()
    offenders = [
        hit
        for hit in _undeclared_hits()
        if hit[2] not in _EXEMPTIONS and not _is_profile_block_key(hit[2])
    ]
    detail = []
    for path, number, key in offenders:
        why = "recorded as DELETED in the manifest" if key in deleted else "not in the manifest"
        detail.append(f"{path}:{number}: {key} ({why})")
    assert offenders == [], (
        "Every config key documented in the configuration reference must be a "
        "declared key in osprey/profiles/config_key_manifest.yml — a documented key the "
        "code does not read is a setting the reader will copy in and watch do "
        "nothing. Undeclared keys remain:\n" + "\n".join(detail)
    )


def test_every_exemption_still_has_something_to_explain() -> None:
    """An exemption whose key is no longer documented should leave the list.

    Exemptions here weaken a promise made to a reader, so they are not allowed
    to outlive the page that needed them.
    """
    documented = {key for _, _, key in _documented_keys()}
    stale = sorted(set(_EXEMPTIONS) - documented)
    assert stale == [], f"exemption entries with no remaining occurrence: {stale}"


def test_the_manifest_declares_keys() -> None:
    """Guard against the manifest silently moving or emptying out.

    An empty ``keys:`` mapping would red the sweep rather than hide a problem,
    but it would red it with a confusing message; pinning it here says plainly
    which artefact broke.
    """
    declared = _declared_keys()
    assert declared, f"{_MANIFEST_PATH} declared no keys at all"
    assert any(_KEY_PATTERN.match(key) for key in declared), (
        "no declared key matches the documented-key shape this sweep looks for — "
        "the section list or the manifest's key spelling has drifted"
    )


def test_the_phoebus_rows_are_swept() -> None:
    """The ``phoebus`` section entry is only worth something if its rows reach the check.

    A sweep that silently stopped recognising the reference page's Phoebus rows
    would look identical to one that passes, so the rows are pinned as found.
    """
    expected = {
        "phoebus.host",
        "phoebus.port",
        "phoebus.panels",
        "phoebus.require_handle",
        "phoebus.agent_access",
        "phoebus.archiver_url",
        "phoebus.plot_dir",
        "phoebus.snapshot_dir",
    }
    swept = {key for _, _, key in _documented_keys()}
    assert expected <= swept, f"phoebus rows the sweep does not see: {sorted(expected - swept)}"


def _write_page(root: Path, name: str, body: str) -> None:
    target = root / _REFERENCE_DIR
    target.mkdir(parents=True, exist_ok=True)
    (target / name).write_text(body, encoding="utf-8")


def test_the_sweep_would_catch_a_bogus_key(tmp_path: Path) -> None:
    """A sweep that finds nothing looks identical to one whose pattern is broken.

    The configuration reference does not exist yet, so the real sweep passes
    vacuously today. This proves the machinery works against a fake docs tree
    checked, as in production, against the real manifest.
    """
    fake_root = tmp_path / "repo"
    _write_page(
        fake_root,
        "web.rst",
        "Web UI\n======\n\nSet ``web.not_a_real_key`` to enable the thing.\n",
    )

    offenders = _undeclared_hits(docs_root=fake_root)
    assert [hit[2] for hit in offenders] == ["web.not_a_real_key"], (
        "the sweep should have flagged a documented key that is not in the manifest"
    )
    assert offenders[0][1] == 4, "the offender should be reported at its real line number"


def test_the_sweep_reads_ariel_keys(tmp_path: Path) -> None:
    """An ``ariel.*`` literal is a checked key, not prose.

    Pins the ``ariel`` entry in :data:`_SECTIONS`: without it both literals below
    are skipped and the sweep reports nothing either way.
    """
    fake_root = tmp_path / "repo"
    _write_page(
        fake_root,
        "ariel.rst",
        "ARIEL\n=====\n\n"
        "Set ``ariel.entry_url_template`` to link entries.\n"
        "Set ``ariel.not_a_real_key`` to do nothing.\n",
    )

    assert [hit[2] for hit in _undeclared_hits(docs_root=fake_root)] == ["ariel.not_a_real_key"], (
        "the sweep should flag the bogus ariel key and pass the declared one"
    )


def test_a_declared_key_is_not_flagged(tmp_path: Path) -> None:
    """The other half of the control: a real key must survive the sweep.

    The key is drawn from the manifest rather than hard-coded, so a rename in
    the config schema cannot turn this test into a false alarm.
    """
    real_key = sorted(key for key in _declared_keys() if _KEY_PATTERN.match(key))[0]
    fake_root = tmp_path / "repo"
    _write_page(
        fake_root,
        "web.rst",
        f"Web UI\n======\n\nSet ``{real_key}`` to change the behaviour.\n",
    )

    assert _undeclared_hits(docs_root=fake_root) == [], (
        f"the declared key {real_key} was flagged as undeclared"
    )


@pytest.mark.parametrize(
    "literal",
    (
        "web.theme: main",
        "osprey.version",
        "docs/source/reference",
        "OSPREY_WEB_APP_NAME",
        "uv run osprey health",
        "web",
    ),
)
def test_non_key_literals_are_left_alone(literal: str, tmp_path: Path) -> None:
    """Reference prose is full of literals that are not config keys.

    YAML fragments, module paths, file paths, environment variables, commands
    and bare section names all appear in inline literals on these pages. None
    of them is a dotted key reference and none may be checked against the
    manifest, or the sweep becomes noise the next author learns to ignore.
    """
    fake_root = tmp_path / "repo"
    _write_page(fake_root, "web.rst", f"Web UI\n======\n\nSee ``{literal}`` here.\n")

    assert _documented_keys(fake_root) == [], f"{literal!r} was mistaken for a config key"


def test_an_absent_reference_directory_is_not_an_error(tmp_path: Path) -> None:
    """The reference section is still being built; a missing tree is a pass.

    Pinned rather than left implicit, because the day the directory does exist
    this behaviour must not be what makes the sweep look green.
    """
    assert _documented_keys(tmp_path / "empty-repo") == []


@pytest.mark.parametrize(
    ("key", "allowed"),
    [
        ("dispatch.worker_port_stride", True),
        ("dispatch.triggers", True),
        ("dispatch.worker_port_strde", False),
        ("web.worker_port_stride", False),
    ],
)
def test_only_the_profile_loaders_own_keys_pass_as_profile_keys(key: str, allowed: bool) -> None:
    """A profile block's real key passes; a misspelling of it is still checked."""
    assert _is_profile_block_key(key) is allowed
