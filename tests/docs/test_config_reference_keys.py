"""Every config key the reference section documents must really exist.

``docs/source/reference/configuration/`` is the operator-facing catalogue of
``config.yml`` settings. A key that appears there and nowhere in the shipped
code is worse than an undocumented one: the reader copies it into their own
config, nothing happens, and there is no error to tell them why. The reverse
drift is just as quiet — a key gets renamed or retired in the code, the guard
manifest records that, and the reference page keeps advertising the old
spelling forever.

This sweep closes that gap by cross-checking the two artefacts that already
exist for their own reasons:

* the reference pages, where a documented key is written as an RST inline
  literal — ``web.theme``, ``services.postgresql.port`` — and
* ``scripts/config_key_manifest.yml``, the config-key resurrection guard's
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
snippets in code blocks pass by untouched. The directory is allowed not to
exist yet: the reference section is being built out, and a sweep that hard
-failed on an absent tree would block its own scaffolding.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The reference pages this sweep reads, relative to the repo root. Absent
#: today; created by the docs restructure. See the module docstring on why an
#: absent directory is a vacuous pass rather than a failure.
_REFERENCE_DIR = "docs/source/reference/configuration"

#: The guard manifest, relative to the repo root.
_MANIFEST_PATH = "scripts/config_key_manifest.yml"

#: Top-level sections of ``config.yml``. A dotted literal has to start with one
#: of these to be treated as a config key at all — the bar that keeps prose
#: like ``osprey.version`` or ``docs.source`` out of the sweep.
_SECTIONS = (
    "health",
    "web",
    "control_system",
    "archiver",
    "bluesky",
    "services",
    "deployed_services",
    "approval",
    "hooks",
    "claude_code",
    "artifacts",
    "agent_data",
    "file_paths",
)

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
#: the surrounding prose does not silently inherit the exemption. Empty by
#: design: an exemption here is a promise to a reader that the key works, made
#: without the guard's evidence behind it, so every entry needs an argument.
_EXEMPTIONS: dict[str, str] = {}


def _reference_pages(root_dir: Path | None = None) -> list[Path]:
    """Every ``.rst`` page in the configuration reference directory."""
    base = (root_dir if root_dir is not None else _REPO_ROOT) / _REFERENCE_DIR
    if not base.is_dir():
        return []
    return sorted(base.glob("*.rst"))


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
    return {str(entry) for entry in deleted}


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
    offenders = [hit for hit in _undeclared_hits() if hit[2] not in _EXEMPTIONS]
    detail = []
    for path, number, key in offenders:
        why = "recorded as DELETED in the manifest" if key in deleted else "not in the manifest"
        detail.append(f"{path}:{number}: {key} ({why})")
    assert offenders == [], (
        "Every config key documented in the configuration reference must be a "
        "declared key in scripts/config_key_manifest.yml — a documented key the "
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
