"""Guard against the web_terminals module regaining a hand-rendered compose fragment.

The scaffolding skill used to hand-render the ``modules.web_terminals`` compose
overlay from prose (a ``# FOR each in ...users`` DSL block emitting per-user
``services:`` YAML). That is now generated deterministically by
``osprey scaffold web-terminals render`` (see
``osprey.deployment.web_terminals.render``), and
``references/modules/web-terminals.md`` documents that verb as the single render
path instead of showing the skill how to hand-expand the fragment itself. These
tests fail if the doc ever regains a hand-rendered fragment, which would let the
skill drift back to rendering ``web_terminals`` artifacts on its own.
"""

from __future__ import annotations

import re
from importlib.resources import as_file, files

import pytest

_DOC_PACKAGE_PATH = (
    "templates/skills/osprey-build-deploy/references/modules/web-terminals.md"
)

# Markers that identify a hand-rendered per-user compose/nginx fragment for
# web_terminals: a prose "FOR each ... users" loop that emits a `services:`-shaped
# per-user block (`web-${each}:` or similar `web-<user>` service stanza).
_FOR_EACH_USERS_LOOP = re.compile(r"#\s*FOR\s+each\s+in\s+.*\busers\b", re.IGNORECASE)
_PER_USER_SERVICE_STANZA = re.compile(r"web-\$\{each\}:|web-<user>:")

_RENDER_VERB = "osprey scaffold web-terminals render"


@pytest.fixture()
def doc_text() -> str:
    """Read the current text of web-terminals.md via package-data resolution."""
    traversable = files("osprey").joinpath(_DOC_PACKAGE_PATH)
    with as_file(traversable) as doc_path:
        return doc_path.read_text(encoding="utf-8")


def test_doc_has_no_hand_rendered_users_loop(doc_text: str) -> None:
    """web-terminals.md must not contain a prose `# FOR each in ...users` loop."""
    match = _FOR_EACH_USERS_LOOP.search(doc_text)
    assert match is None, (
        "web-terminals.md contains a hand-rendered '# FOR each in ...users' loop "
        f"({match.group(0)!r}); web_terminals compose/nginx artifacts must be "
        "generated only by `osprey scaffold web-terminals render`, not hand-"
        "expanded by the scaffolding skill."
    )


def test_doc_has_no_per_user_service_stanza(doc_text: str) -> None:
    """web-terminals.md must not emit a per-user `web-<user>:` compose service stanza."""
    match = _PER_USER_SERVICE_STANZA.search(doc_text)
    assert match is None, (
        "web-terminals.md contains a hand-rendered per-user compose service "
        f"stanza ({match.group(0)!r}); this belongs only in the deterministic "
        "renderer, not in the doc's prose."
    )


def test_doc_references_the_render_verb(doc_text: str) -> None:
    """web-terminals.md must document `osprey scaffold web-terminals render` as the generation path."""
    assert _RENDER_VERB in doc_text, (
        f"web-terminals.md no longer references '{_RENDER_VERB}'; the doc must "
        "direct the skill to invoke the CLI verb instead of hand-rendering "
        "web_terminals artifacts."
    )


def test_render_verb_appears_in_compose_section(doc_text: str) -> None:
    """The render-verb instruction must live in the '### compose' section, not elsewhere.

    This keeps the assertion meaningful: the verb must specifically replace the
    old hand-rendered compose block under "## What scaffolding adds when this
    module is enabled" / "### compose", not just be mentioned in passing (e.g.
    only in an unrelated troubleshooting table).
    """
    compose_section_match = re.search(
        r"### compose\n(.*?)\n### ", doc_text, re.DOTALL
    )
    assert compose_section_match is not None, "could not locate the '### compose' section"
    assert _RENDER_VERB in compose_section_match.group(1), (
        "the '### compose' section of web-terminals.md must instruct the skill "
        f"to run '{_RENDER_VERB}' rather than hand-rendering the overlay itself."
    )
