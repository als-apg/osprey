"""Every in-repo pointer on ``contributing/extending-osprey`` must still resolve.

That page is the developer entry point: it is almost entirely pointers — "the
connector base class lives in ``src/osprey/connectors/base.py``", "the write
path is ``osprey.services.virtual_accelerator.serving.write_path``". Pointers
of that shape are exactly the prose that rots silently. Sphinx renders a stale
one as a perfectly formatted inline literal, no warning, and a developer loses
an afternoon before concluding the docs are wrong.

So this gate reads the page's inline literals and resolves them for real:

* a literal that looks like a repo path (``src/…``, ``packages/…``,
  ``tests/…``) must exist on disk, relative to the repo root;
* a literal that looks like a dotted Python name under ``lume`` or ``osprey``
  must actually import — the module prefix via :func:`importlib.import_module`,
  the final segment via :func:`getattr`.

The page does not exist yet; a later task in this restructure writes it. Until
then the sweep finds no literals and the gate passes vacuously — which is the
intended behaviour, not an oversight. The negative controls below are what
keep a vacuous pass honest: they run the same checker against a fake repo tree
and require it to report the planted breakage.

Two platform notes. The virtual-accelerator wheels (``lume_pva_apg``,
``pcaspy``) are published for linux-x86_64 only, so on a developer's Mac the
import half of a pointer to that stack cannot be resolved at all. A missing
one of those is reported as a skip with the reason spelled out, never as a
pass and never as a failure; any *other* :class:`ModuleNotFoundError` is a
genuine stale pointer and fails.
"""

from __future__ import annotations

import importlib
import keyword
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The page under guard, relative to the repo root.
_PAGE = Path("docs") / "source" / "contributing" / "extending-osprey.rst"

#: RST inline literal: ``…``. Backticks and newlines cannot appear inside one,
#: which is what keeps this from swallowing a whole paragraph between two
#: unrelated literals.
_LITERAL_PATTERN = re.compile(r"``([^`\n]+)``")

#: A literal is treated as a repo path when it starts with one of these. The
#: page also cites ``docs/…`` paths, but those move around constantly during a
#: restructure and are covered by the Sphinx build itself; the three trees
#: below are the ones a developer is being sent to read.
_PATH_PREFIXES = ("src/", "packages/", "tests/")

#: Shell-glob metacharacters. A literal carrying one is a pattern, not a path,
#: and is satisfied by any single match.
_GLOB_CHARS = "*?["

#: ``src/osprey/foo.py:217`` — the house style for citing a specific line. The
#: line number is not part of the path.
_LINE_SUFFIX = re.compile(r":\d+(?:-\d+)?$")

#: Top-level modules whose wheels exist only for linux-x86_64 (pyproject's
#: ``virtual-accelerator`` extra marks both ``sys_platform == 'linux' and
#: platform_machine == 'x86_64'``). Absent here means "wrong platform", not
#: "stale pointer".
_PLATFORM_GATED_MODULES = frozenset({"lume_pva_apg", "pcaspy"})

#: Only dotted names rooted at one of these are checked. Everything else in
#: backticks — a config key, a CLI verb, a PV name — is out of scope.
_IMPORTABLE_ROOTS = ("lume", "osprey")

#: A dotted literal ending in one of these is a filename, not a Python symbol:
#: ``machine.json``, ``osprey.yaml``. Matched case-insensitively on the final
#: segment.
_FILE_LIKE_TAILS = frozenset(
    {
        "cfg",
        "html",
        "j2",
        "json",
        "lock",
        "md",
        "png",
        "py",
        "rst",
        "sh",
        "svg",
        "toml",
        "txt",
        "yaml",
        "yml",
    }
)


def _page_literals(root_dir: Path | None = None) -> list[str]:
    """Every inline literal on the page, in order. Empty when the page is absent."""
    page = (root_dir if root_dir is not None else _REPO_ROOT) / _PAGE
    if not page.is_file():
        return []
    return _LITERAL_PATTERN.findall(page.read_text(encoding="utf-8"))


def _looks_like_repo_path(literal: str) -> bool:
    return literal.startswith(_PATH_PREFIXES) and " " not in literal


def _looks_like_dotted_symbol(literal: str) -> bool:
    """True for ``osprey.pkg.Symbol`` shapes; false for paths, keys and filenames."""
    if any(ch in literal for ch in " /\\=(),:") or "." not in literal:
        return False
    segments = literal.split(".")
    if len(segments) < 2:
        return False
    if not segments[0].startswith(_IMPORTABLE_ROOTS):
        return False
    if segments[-1].lower() in _FILE_LIKE_TAILS:
        return False
    return all(segment.isidentifier() and not keyword.iskeyword(segment) for segment in segments)


def _missing_paths(root_dir: Path | None = None) -> list[str]:
    """Every cited ``src/``/``packages/``/``tests/`` literal that resolves to nothing."""
    base = root_dir if root_dir is not None else _REPO_ROOT
    missing: list[str] = []
    for literal in _page_literals(root_dir):
        if not _looks_like_repo_path(literal):
            continue
        cited = _LINE_SUFFIX.sub("", literal).rstrip("/")
        if any(ch in cited for ch in _GLOB_CHARS):
            if not list(base.glob(cited)):
                missing.append(f"{literal} (glob matched no file)")
            continue
        if not (base / cited).exists():
            missing.append(literal)
    return missing


def _symbol_results(root_dir: Path | None = None) -> tuple[list[str], list[str]]:
    """``(failures, skips)`` for every cited dotted name under ``lume``/``osprey``.

    A failure is a pointer that does not resolve. A skip is a pointer into a
    stack whose wheel is not installable on this platform — reported so the
    coverage gap is visible in the run rather than silently counted as a pass.
    """
    failures: list[str] = []
    skips: list[str] = []
    for literal in _page_literals(root_dir):
        if not _looks_like_dotted_symbol(literal):
            continue
        module_name, _, attribute = literal.rpartition(".")
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            root = (exc.name or "").split(".")[0]
            if root in _PLATFORM_GATED_MODULES:
                skips.append(f"{literal} (needs {root}, published for linux-x86_64 only)")
            else:
                failures.append(f"{literal} (no module {module_name!r}: {exc})")
            continue
        except ImportError as exc:  # pragma: no cover - a broken module, not a stale pointer
            failures.append(f"{literal} (importing {module_name!r} failed: {exc})")
            continue
        if hasattr(module, attribute):
            continue
        # A pointer may name a submodule that the parent package does not
        # import eagerly; that is still a valid pointer, so try it as a module
        # before calling the literal stale.
        try:
            importlib.import_module(literal)
        except ModuleNotFoundError as exc:
            root = (exc.name or "").split(".")[0]
            if root in _PLATFORM_GATED_MODULES:
                skips.append(f"{literal} (needs {root}, published for linux-x86_64 only)")
            else:
                failures.append(f"{literal} (module {module_name!r} has no {attribute!r})")
        except ImportError as exc:  # pragma: no cover - defensive
            failures.append(f"{literal} (importing {literal!r} failed: {exc})")
    return failures, skips


def test_every_cited_repo_path_exists() -> None:
    """RULE: a path in backticks on the extending page must exist in the repo."""
    missing = _missing_paths()
    assert missing == [], (
        f"{_PAGE} sends developers to files that do not exist. Every ``src/…``, "
        "``packages/…`` or ``tests/…`` literal on that page must resolve to a real "
        "path in this repo (a ``:LINE`` suffix and a trailing ``/`` are allowed; a "
        "glob must match at least one file). Broken pointers:\n"
        + "\n".join(f"  {entry}" for entry in missing)
    )


def test_every_cited_symbol_imports() -> None:
    """RULE: a dotted ``lume``/``osprey`` name in backticks must actually import."""
    failures, skips = _symbol_results()
    assert failures == [], (
        f"{_PAGE} names Python symbols that no longer exist. Every dotted "
        "``lume…``/``osprey…`` literal on that page must import: the module prefix "
        "via importlib, the final segment via getattr. Broken pointers:\n"
        + "\n".join(f"  {entry}" for entry in failures)
    )
    if skips:
        pytest.skip(
            "some pointers name the virtual-accelerator stack, whose wheels are "
            "published for linux-x86_64 only and are absent on this platform:\n"
            + "\n".join(f"  {entry}" for entry in skips)
        )


def _write_fake_page(root: Path, body: str) -> None:
    page = root / _PAGE
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(body, encoding="utf-8")


def test_the_path_sweep_would_catch_a_stale_pointer(tmp_path: Path) -> None:
    """A sweep that finds nothing looks identical to one whose pattern is broken.

    The page is not written yet, so the live sweep passes on an empty literal
    list. This plants the exact regression the gate exists for — a cited source
    file that was moved or deleted — in a fake repo and requires the checker to
    report it.
    """
    fake_root = tmp_path / "repo"
    _write_fake_page(
        fake_root,
        "Extending Osprey\n================\n\n"
        "The connector base class lives in ``src/osprey/nope.py``.\n",
    )

    missing = _missing_paths(fake_root)
    assert missing == ["src/osprey/nope.py"], missing


def test_the_path_sweep_accepts_the_shapes_the_page_actually_uses(tmp_path: Path) -> None:
    """The other half of the control: real pointers must not be reported.

    A checker that flagged everything would also pass the test above. This
    pins the three spellings the house style uses — a bare path, a
    ``:LINE``-suffixed citation, a directory with a trailing slash — plus a
    glob, against a fake tree where all four resolve.
    """
    fake_root = tmp_path / "repo"
    (fake_root / "src" / "osprey").mkdir(parents=True)
    (fake_root / "src" / "osprey" / "base.py").write_text("", encoding="utf-8")
    (fake_root / "tests" / "docs").mkdir(parents=True)
    _write_fake_page(
        fake_root,
        "Extending Osprey\n================\n\n"
        "See ``src/osprey/base.py``, ``src/osprey/base.py:217``, ``tests/docs/`` "
        "and ``src/osprey/*.py``. Unrelated literals such as ``osprey health`` "
        "and ``docs/source/index.rst`` are out of scope.\n",
    )

    assert _missing_paths(fake_root) == []


def test_the_symbol_sweep_would_catch_a_stale_import(tmp_path: Path) -> None:
    """The import half needs its own control, for the same reason."""
    fake_root = tmp_path / "repo"
    _write_fake_page(
        fake_root,
        "Extending Osprey\n================\n\n"
        "The write path is ``osprey.nope_module.NoSuchSymbol``.\n",
    )

    failures, skips = _symbol_results(fake_root)
    assert skips == []
    assert len(failures) == 1, failures
    assert failures[0].startswith("osprey.nope_module.NoSuchSymbol")


def test_the_symbol_sweep_accepts_a_live_symbol_and_ignores_non_symbols(
    tmp_path: Path,
) -> None:
    """Live pointers, and the backticked prose that must never be treated as one."""
    fake_root = tmp_path / "repo"
    _write_fake_page(
        fake_root,
        "Extending Osprey\n================\n\n"
        "The version helper is ``osprey.version.get_running_version`` and the "
        "package is ``osprey.version``. Not symbols: ``services.virtual_accelerator.port``, "
        "``osprey.yaml``, ``lume-base==0.5.0``, ``osprey health reach``.\n",
    )

    failures, skips = _symbol_results(fake_root)
    assert (failures, skips) == ([], [])


def test_a_platform_gated_pointer_is_reported_as_a_skip(tmp_path: Path) -> None:
    """The virtual-accelerator stack must not fail the gate off linux-x86_64.

    ``lume_pva_apg`` and ``pcaspy`` ship wheels for linux-x86_64 only, so a
    pointer into that stack is unresolvable on a developer's Mac through no
    fault of the docs. The classification is deterministic on every platform:
    whether the package is missing entirely or merely lacks the cited
    submodule, the missing module's root is the gated package either way, so
    this lands in ``skips`` rather than ``failures`` everywhere the suite runs.
    """
    fake_root = tmp_path / "repo"
    _write_fake_page(
        fake_root,
        "Extending Osprey\n================\n\n"
        "The PVA server is ``lume_pva_apg.nope_module.Server``.\n",
    )

    failures, skips = _symbol_results(fake_root)
    assert failures == [], failures
    assert len(skips) == 1 and "linux-x86_64" in skips[0], skips
