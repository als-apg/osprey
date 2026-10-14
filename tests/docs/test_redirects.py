"""Gate: every ``redirects`` entry in ``conf.py`` points a dead page at a live one.

``sphinx_reredirects`` emits one stub HTML file per key at the key's old URL.
That gives the map two ways to go quietly wrong, and neither shows up as a
build warning:

* a key whose page still exists — the stub overwrites the real page's HTML, so
  the live page silently becomes a redirect to somewhere else;
* a value that does not resolve — the stub is still written, so the old URL
  keeps working right up until a reader follows it into a 404.

The map is read out of ``conf.py`` with :mod:`runpy`, patching the version
source the same way ``test_publishing_config.py`` does: ``conf.py`` binds
``get_running_version``/``is_release`` at exec time, so the *module attributes*
on :mod:`osprey.version` are what it actually picks up, and patching them
sidesteps the ``functools.cache`` on the real implementation.

Values are PAGE-relative, exactly as ``sphinx_reredirects`` interprets them: a
value is resolved against the directory of the key, not against the docs root.
"""

from __future__ import annotations

import os
import re
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

import osprey.version

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONF_PY = _REPO_ROOT / "docs" / "source" / "conf.py"
_DOCS_SOURCE = _REPO_ROOT / "docs" / "source"

#: ``scheme:`` prefix per RFC 3986. A redirect target that names a scheme (or
#: starts at the server root) escapes the versioned docs tree, so ``/latest/``
#: and every published snapshot would all land on the same absolute URL.
_URL_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")


def _run_conf(monkeypatch) -> dict[str, Any]:
    """Execute ``conf.py`` with a pinned version source and return its namespace."""
    monkeypatch.setattr(osprey.version, "is_release", lambda: True)
    monkeypatch.setattr(osprey.version, "get_running_version", lambda: "9.9.9")
    monkeypatch.chdir(_CONF_PY.parent)

    saved_path = list(sys.path)
    try:
        return runpy.run_path(str(_CONF_PY))
    finally:
        sys.path[:] = saved_path


@pytest.fixture
def redirects(monkeypatch) -> dict[str, str]:
    """The live ``redirects`` map as ``conf.py`` defines it."""
    ns = _run_conf(monkeypatch)
    value = ns["redirects"]
    assert isinstance(value, dict), "conf.py must define `redirects` as a dict of docname -> target"
    return value


def _shadowed_keys(redirects: dict[str, str], docs_source: Path) -> list[str]:
    """Keys whose source page still exists — the stub would overwrite a live page."""
    return sorted(key for key in redirects if (docs_source / f"{key}.rst").exists())


def _target_problems(redirects: dict[str, str], docs_source: Path) -> list[tuple[str, str, str]]:
    """``(key, value, reason)`` for every target that is absolute, unsuffixed or dangling.

    At most one reason is reported per entry: an absolute target has no
    page-relative resolution to check, and a target without the ``.html``
    suffix has no source page to look for.
    """
    problems: list[tuple[str, str, str]] = []
    for key, value in sorted(redirects.items()):
        if value.startswith("/") or _URL_SCHEME.match(value):
            problems.append((key, value, "absolute — targets must be relative to the old page"))
            continue
        if not value.endswith(".html"):
            problems.append((key, value, "missing the .html suffix"))
            continue
        resolved = os.path.normpath((docs_source / key).parent / value)
        source_page = Path(resolved).with_suffix(".rst")
        if not source_page.exists():
            try:
                shown = str(source_page.relative_to(docs_source))
            except ValueError:
                shown = str(source_page)
            problems.append((key, value, f"no such page: docs/source/{shown}"))
    return problems


def test_no_redirect_shadows_a_page_that_still_exists(redirects: dict[str, str]) -> None:
    offenders = _shadowed_keys(redirects, _DOCS_SOURCE)
    assert offenders == [], (
        "A redirect key must name a page that NO LONGER exists: the emitted stub is "
        "written at the key's URL and overwrites the real page. Still present:\n"
        + "\n".join(f"docs/source/{key}.rst" for key in offenders)
    )


def test_every_redirect_target_is_a_relative_html_page(redirects: dict[str, str]) -> None:
    offenders = [
        (key, value, reason)
        for key, value, reason in _target_problems(redirects, _DOCS_SOURCE)
        if "no such page" not in reason
    ]
    assert offenders == [], (
        "A redirect target must be RELATIVE to the old page and end in .html "
        "(e.g. '../reference/cli.html') — never a server-root path or a URL:\n"
        + "\n".join(f"{key} -> {value}: {reason}" for key, value, reason in offenders)
    )


def test_every_redirect_target_resolves_to_a_live_page(redirects: dict[str, str]) -> None:
    offenders = [
        (key, value, reason)
        for key, value, reason in _target_problems(redirects, _DOCS_SOURCE)
        if "no such page" in reason
    ]
    assert offenders == [], (
        "A redirect target is resolved relative to the old page's directory and must "
        "name a page that exists — otherwise the old URL redirects into a 404:\n"
        + "\n".join(f"{key} -> {value}: {reason}" for key, value, reason in offenders)
    )


def test_the_checks_would_catch_a_broken_map(tmp_path: Path) -> None:
    """Negative control: a deliberately broken map must trip every rule.

    The real map is empty today, so all three gates above pass vacuously. This
    feeds a fake map and a fake docs root through the same two checkers, so the
    rules are proven to fire — and the two well-formed entries prove they do
    not fire indiscriminately.
    """
    docs_source = tmp_path / "docs" / "source"
    (docs_source / "guides").mkdir(parents=True)
    (docs_source / "shadowed.rst").write_text("still here\n", encoding="utf-8")
    (docs_source / "guides" / "target.rst").write_text("the new home\n", encoding="utf-8")

    fake_map = {
        "shadowed": "guides/target.html",
        "gone-absolute": "/guides/target.html",
        "gone-scheme": "https://example.com/guides/target.html",
        "gone-suffix": "guides/target",
        "gone-dangling": "guides/missing.html",
        "gone-ok": "guides/target.html",
        "old/nested": "../guides/target.html",
    }

    assert _shadowed_keys(fake_map, docs_source) == ["shadowed"]

    reasons = {key: reason for key, _value, reason in _target_problems(fake_map, docs_source)}
    assert set(reasons) == {"gone-absolute", "gone-scheme", "gone-suffix", "gone-dangling"}
    assert "absolute" in reasons["gone-absolute"]
    assert "absolute" in reasons["gone-scheme"]
    assert ".html suffix" in reasons["gone-suffix"]
    assert "no such page" in reasons["gone-dangling"]


# The html build itself refuses a missing stub: conf.py connects a build-finished
# handler that looks for one stub page per key in the output directory. The
# handler's refusal raises Sphinx's own error class, which only the docs extra
# installs, so these cases drive the check and the handler's guards without it.


class _FakeBuilder:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeConfig:
    def __init__(self, redirects: dict[str, str]) -> None:
        self.redirects = redirects


class _FakeApp:
    def __init__(self, builder: str, outdir: Path, redirects: dict[str, str]) -> None:
        self.builder = _FakeBuilder(builder)
        self.outdir = outdir
        self.config = _FakeConfig(redirects)
        self.connected: list[tuple[str, Any]] = []

    def connect(self, event: str, handler: Any) -> None:
        self.connected.append((event, handler))


def _write_stubs(outdir: Path, keys: list[str]) -> None:
    for key in keys:
        stub = outdir / f"{key}.html"
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text("<html></html>\n", encoding="utf-8")


def test_the_build_connects_the_stub_check_to_build_finished(monkeypatch, tmp_path: Path) -> None:
    ns = _run_conf(monkeypatch)
    app = _FakeApp("html", tmp_path, {})
    ns["setup"](app)
    assert app.connected == [("build-finished", ns["_check_redirect_stubs"])]


def test_the_stub_check_names_each_missing_stub(
    monkeypatch, tmp_path: Path, redirects: dict[str, str]
) -> None:
    ns = _run_conf(monkeypatch)
    keys = sorted(redirects)
    assert keys, "conf.py defines no redirects; this check is vacuous"
    _write_stubs(tmp_path, keys)
    assert ns["_missing_redirect_stubs"](tmp_path, redirects) == []
    (tmp_path / f"{keys[0]}.html").unlink()
    assert ns["_missing_redirect_stubs"](tmp_path, redirects) == [keys[0]]


def test_the_stub_check_passes_a_complete_html_build(
    monkeypatch, tmp_path: Path, redirects: dict[str, str]
) -> None:
    ns = _run_conf(monkeypatch)
    _write_stubs(tmp_path, sorted(redirects))
    assert ns["_check_redirect_stubs"](_FakeApp("html", tmp_path, redirects), None) is None


@pytest.mark.parametrize("builder", ["linkcheck", "dirhtml"])
def test_the_stub_check_ignores_builders_that_write_no_stubs(
    monkeypatch, tmp_path: Path, redirects: dict[str, str], builder: str
) -> None:
    ns = _run_conf(monkeypatch)
    assert ns["_check_redirect_stubs"](_FakeApp(builder, tmp_path, redirects), None) is None


def test_the_stub_check_leaves_a_failed_build_its_own_error(
    monkeypatch, tmp_path: Path, redirects: dict[str, str]
) -> None:
    ns = _run_conf(monkeypatch)
    app = _FakeApp("html", tmp_path, redirects)
    assert ns["_check_redirect_stubs"](app, RuntimeError("the build broke")) is None
