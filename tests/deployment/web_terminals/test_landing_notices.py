"""Landing-page notices and footer: `modules.web_terminals.landing.notices`.

Notices are markdown files a facility lists in config; each becomes one
collapsible section on the landing page. These tests pin the three cases that
are deliberately distinct (key absent / empty list / explicit list), plus the
escaping asymmetry: a notice body is injected as HTML, the footer never is.
"""

from __future__ import annotations

import copy
from pathlib import Path

from osprey.deployment.web_terminals.render import render_web_terminals

from .test_golden_render import EXAMPLE_CONFIG


def _config(tmp_path: Path, **landing: object) -> dict:
    """EXAMPLE_CONFIG rooted at *tmp_path*, with `landing` keys merged in."""
    cfg = copy.deepcopy(EXAMPLE_CONFIG)
    cfg["project_root"] = str(tmp_path)
    cfg["modules"]["web_terminals"]["landing"].update(landing)
    return cfg


def _write(tmp_path: Path, rel: str, text: str) -> str:
    doc = tmp_path / rel
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(text, encoding="utf-8")
    return rel


def _landing(cfg: dict) -> str:
    return render_web_terminals(cfg)["nginx/landing.html"]


def test_absent_notices_key_renders_the_packaged_default(tmp_path: Path) -> None:
    """A config that says nothing about notices still ships the safety copy."""
    cfg = copy.deepcopy(EXAMPLE_CONFIG)
    cfg["project_root"] = str(tmp_path)
    html = _landing(cfg)
    assert 'class="landing-notice"' in html
    assert "Before you start: working safely with the agent" in html
    assert "Be mindful of what you approve" in html


def test_empty_notices_list_renders_none(tmp_path: Path) -> None:
    """`notices: []` is how a deployment explicitly asks for no notices."""
    html = _landing(_config(tmp_path, notices=[]))
    assert 'class="landing-notice"' not in html
    assert "Before you start" not in html


def test_h1_becomes_the_summary_and_the_rest_becomes_the_body(tmp_path: Path) -> None:
    rel = _write(
        tmp_path,
        "data/landing/local-procedures.md",
        "# Local procedures\n\nCall the CR first.\n",
    )
    html = _landing(_config(tmp_path, notices=[rel]))
    assert "<summary>Local procedures</summary>" in html
    assert "Call the CR first." in html
    # The H1 is consumed as the label, not repeated inside the panel.
    assert "<h1>" not in html


def test_markdown_body_is_rendered_as_html_not_escaped(tmp_path: Path) -> None:
    """The body is the one value on the page injected with `|safe`."""
    rel = _write(
        tmp_path,
        "data/landing/n.md",
        "# N\n\n- first point\n- second point\n\n**bold**\n",
    )
    html = _landing(_config(tmp_path, notices=[rel]))
    assert "<li>first point</li>" in html
    assert "<strong>bold</strong>" in html
    assert "&lt;li&gt;" not in html


def test_missing_path_is_skipped_without_falling_back(tmp_path: Path) -> None:
    """A listed-but-missing file leaves a gap; it is never silently replaced by
    the packaged default, which would show OSPREY's safety text in place of a
    facility's own missing document."""
    rel = _write(tmp_path, "data/landing/present.md", "# Present\n\nHere.\n")
    html = _landing(_config(tmp_path, notices=["data/landing/gone.md", rel]))
    assert "<summary>Present</summary>" in html
    assert "Before you start: working safely with the agent" not in html


def test_notices_render_in_config_order(tmp_path: Path) -> None:
    a = _write(tmp_path, "data/landing/a.md", "# Alpha\n\nA.\n")
    b = _write(tmp_path, "data/landing/b.md", "# Beta\n\nB.\n")
    html = _landing(_config(tmp_path, notices=[b, a]))
    assert html.index("<summary>Beta</summary>") < html.index("<summary>Alpha</summary>")


def test_notice_id_slugs_from_the_filename(tmp_path: Path) -> None:
    """Each notice gets a stable deep-link target."""
    rel = _write(tmp_path, "data/landing/Who Is On Call.md", "# On call\n\nX.\n")
    html = _landing(_config(tmp_path, notices=[rel]))
    assert 'id="who-is-on-call"' in html


def test_document_without_h1_keeps_its_text(tmp_path: Path) -> None:
    rel = _write(tmp_path, "data/landing/no-heading.md", "Just a paragraph.\n")
    html = _landing(_config(tmp_path, notices=[rel]))
    assert "Just a paragraph." in html
    assert "<summary>no heading</summary>" in html


def test_footer_defaults_when_unset(tmp_path: Path) -> None:
    cfg = copy.deepcopy(EXAMPLE_CONFIG)
    cfg["project_root"] = str(tmp_path)
    assert "OSPREY multi-user web terminal stack." in _landing(cfg)


def test_footer_is_configurable(tmp_path: Path) -> None:
    html = _landing(_config(tmp_path, footer="ALS control room. Ring ext. 5555."))
    assert "ALS control room. Ring ext. 5555." in html
    assert "Experimental system" not in html


def test_empty_footer_renders_no_footer_element(tmp_path: Path) -> None:
    html = _landing(_config(tmp_path, footer=""))
    assert 'class="landing-footer"' not in html


def test_footer_is_escaped(tmp_path: Path) -> None:
    """Unlike a notice body, the footer is a plain string and stays escaped."""
    html = _landing(_config(tmp_path, footer="<script>alert(1)</script>"))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def _lint(cfg: dict) -> list:
    from osprey.deployment.web_terminals.lint import lint_web_terminals

    return lint_web_terminals(cfg)


def test_lint_warns_on_a_missing_notice(tmp_path: Path) -> None:
    cfg = _config(tmp_path, notices=["data/landing/gone.md"])
    codes = [(f.severity, f.code) for f in _lint(cfg)]
    assert ("warn", "notice-missing") in codes


def test_lint_is_quiet_when_every_notice_exists(tmp_path: Path) -> None:
    rel = _write(tmp_path, "data/landing/here.md", "# Here\n\nX.\n")
    cfg = _config(tmp_path, notices=[rel])
    assert [f for f in _lint(cfg) if f.code.startswith("notice-")] == []


def test_lint_warns_on_a_non_path_notice_entry(tmp_path: Path) -> None:
    cfg = _config(tmp_path, notices=[{"doc": "x.md"}])
    codes = [f.code for f in _lint(cfg)]
    assert "notice-invalid" in codes
