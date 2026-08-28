"""The four auth postures, as a truth table over the booleans everything reads.

``modules.web_terminals.auth.method`` has four values, and every consumer that
used to ask ``auth_method != "none"`` now asks one of the derived booleans that
`_auth_tls_context` computes once:

======== ============== ============= ====== ============== ==============
method   sidecar_active inject_secret walled token_exchange open_perimeter
======== ============== ============= ====== ============== ==============
none      no             YES           no     no             YES
token     no             no            no     YES            no
password  YES            YES           YES    no             no
oidc      YES            YES           YES    no             no
======== ============== ============= ====== ============== ==============

Read down the columns and the reason for the refactor is visible: not one of
them is the old ``auth_method != "none"``. ``inject_secret`` is true for three
postures, ``token_exchange`` and ``open_perimeter`` for one each (a different
one each), ``sidecar_active`` and ``walled`` for two.
That last pair coincides by construction — render.py derives ``walled`` FROM
``sidecar_active`` — and is spelled separately anyway because its readers ask a
different question: the sidecar's callers ask whether a service is rendered, the
wall's ask whether any entry has a login. ``none`` (open) injects the secret
without a wall; ``token`` walls nothing and injects nothing, leaving each
terminal's own magic link as the only gate. So a scheme with two values could be
spelled as one comparison and this one cannot, which is why the table is pinned
here rather than left implicit in the render.

This module owns that table and nothing else. It does not render artifacts —
`test_auth_off_baseline_pin.py` pins the ``token`` bytes and
`test_nginx_auth_surface.py` pins what each posture does to nginx. What fails
here is the semantics one layer up: a method whose booleans came out wrong, a
default that stopped being ``token``, or a ``== "none"`` comparison that grew
back somewhere the four-value scheme cannot reach.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest

from osprey.deployment.web_terminals.personas import auth_is_enforced
from osprey.deployment.web_terminals.render import (
    SUPPORTED_AUTH_METHODS,
    _auth_tls_context,
)

pytestmark = pytest.mark.unit

#: The derived booleans, in the order the table above reads them.
_BOOLEANS = (
    "sidecar_active",
    "inject_secret",
    "walled",
    "token_exchange",
    "open_perimeter",
)

#: ``method -> that method's row of :data:`_BOOLEANS`.``
_TRUTH_TABLE = {
    "none": (False, True, False, False, True),
    "token": (False, False, False, True, False),
    "password": (True, True, True, False, False),
    "oidc": (True, True, True, False, False),
}


def _context(web_terminals: dict[str, Any]) -> dict[str, Any]:
    return _auth_tls_context(web_terminals)


# ---------------------------------------------------------------------------
# The table itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", sorted(_TRUTH_TABLE))
def test_each_method_derives_exactly_its_row_of_the_truth_table(method: str) -> None:
    """Every posture's four booleans at once, as one tuple comparison.

    Asserted as a tuple rather than as four assertions so a failure prints the
    whole row: when a posture is wrong it is almost never wrong in one column,
    and the shape of the wrongness (``none`` that walls, ``token`` that injects)
    is what names the mistake.
    """
    context = _context({"auth": {"method": method}})

    assert tuple(context[name] for name in _BOOLEANS) == _TRUTH_TABLE[method]


@pytest.mark.parametrize("method", sorted(_TRUTH_TABLE))
def test_every_derived_boolean_is_a_real_bool(method: str) -> None:
    """The booleans are consumed by Jinja ``{% if %}`` and by Python ``if``
    alike, and both are happy with a truthy string. Every method name is a
    non-empty string, so a "boolean" derived as the name itself — or as
    ``auth_method if ... else ""`` — would read True in every one of those
    conditionals. That is accidentally correct for ``password`` and ``oidc``,
    whose row is true in three of four columns, and silently wrong for ``none``
    and ``token``, which is exactly the pair the four-value scheme exists to
    keep apart. Pinning the type catches it at the source rather than at the
    seam."""
    context = _context({"auth": {"method": method}})

    for name in _BOOLEANS:
        assert type(context[name]) is bool, f"{name} is {type(context[name])!r}, not bool"


@pytest.mark.parametrize(
    "web_terminals,label",
    [
        ({}, "no web_terminals keys at all"),
        ({"auth": {}}, "an auth stanza with no method"),
        ({"auth": {"method": None}}, "an explicit null method"),
        ({"auth": {"method": ""}}, "an empty-string method"),
        ({"auth": {"method": 7}}, "a non-string method"),
        ({"auth": None}, "an auth key with no mapping under it"),
    ],
)
def test_an_unwritten_method_is_token(web_terminals: dict[str, Any], label: str) -> None:
    """``token`` is the default, and every way of not-writing a method reaches
    it — including the ways a hand-edited YAML file produces by accident.

    This is the safe-by-default claim: a facility that writes no ``auth`` block
    gets the magic-link posture, never the open one. If this ever fails for one
    of the rows, some spelling of "unset" is falling through to a different
    posture, and the failure mode is a deployment that is open without anybody
    having asked for it.
    """
    context = _context(web_terminals)

    assert context["auth_method"] == "token", label
    assert tuple(context[name] for name in _BOOLEANS) == _TRUTH_TABLE["token"], label


def test_the_supported_methods_are_the_four_this_module_tables() -> None:
    """The table above is only complete while these are the only four methods.

    A fifth method added to `SUPPORTED_AUTH_METHODS` without a row here would
    otherwise ship with its four booleans untested — which is exactly the
    forked-branch failure the derived-boolean refactor exists to prevent."""
    assert SUPPORTED_AUTH_METHODS == ("none", "token", "password", "oidc")
    assert set(SUPPORTED_AUTH_METHODS) == set(_TRUTH_TABLE)


def test_an_unknown_method_is_refused_by_name_with_the_supported_set() -> None:
    """The one input `_auth_tls_context` raises on. The message must carry both
    the rejected spelling and the full supported set: an operator who typed
    ``basic`` needs to be told ``token`` exists, and a message that named only
    the mistake would send them to the source to find the alternatives."""
    with pytest.raises(ValueError) as excinfo:
        _context({"auth": {"method": "basic"}})

    message = str(excinfo.value)
    assert "basic" in message
    for method in SUPPORTED_AUTH_METHODS:
        assert method in message, f"the refusal does not name {method!r}: {message}"


# ---------------------------------------------------------------------------
# The one boolean read from outside the render
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,enforced",
    [
        ("none", False),
        ("token", False),
        ("password", True),
        ("oidc", True),
    ],
)
def test_auth_is_enforced_reports_the_walled_boolean(method: str, enforced: bool) -> None:
    """`auth_is_enforced` is the capability guards' name for ``walled``, and the
    two postures without a login wall must both report False.

    ``token`` is the trap: it gates each terminal behind a magic link, so it
    feels enforced, but no entry has a login and the privileged-terminal guards
    have to treat its terminals the way they treat ``login: false`` ones. A
    ``token`` that reported True here would silence exactly the warning an
    operator running a privileged unauthenticated terminal needs to see.
    """
    assert auth_is_enforced({"auth": {"method": method}}) is enforced


def test_auth_is_enforced_reads_an_unparseable_stanza_as_walled() -> None:
    """Fail-walled on the input `_auth_tls_context` raises for. The config is
    already being rejected by lint for the unknown method; answering False here
    would add a second, more alarming finding about a deployment that will never
    be built."""
    assert auth_is_enforced({"auth": {"method": "basic"}}) is True


# ---------------------------------------------------------------------------
# The comparison that must not grow back
# ---------------------------------------------------------------------------

_SRC = Path(__file__).resolve().parents[3] / "src" / "osprey"

#: Where `_auth_tls_context` lives, and the only file allowed an exemption.
_PARSER_MODULE = _SRC / "deployment" / "web_terminals" / "render.py"

#: The templates that used to gate on the method name directly.
_WEB_TERMINAL_TEMPLATES = _SRC / "templates" / "modules" / "web_terminals"

#: ``auth_method == "none"`` in either dialect and either operand order, with or
#: without the ``context["auth_method"]`` subscript around the name. Deliberately
#: narrow: it matches the exact idiom the refactor removed, so a hit is a real
#: regression rather than something that merely mentions the word.
_NONE_COMPARISONS = (
    re.compile(r"""auth_method["']?\]?\s*[!=]=\s*["']none["']"""),
    re.compile(r"""["']none["']\s*[!=]=\s*[\w.]*\[?["']?auth_method"""),
)


def _compares_to_none(line: str) -> bool:
    return any(pattern.search(line) for pattern in _NONE_COMPARISONS)


def _parser_line_span() -> range:
    """The 1-based line numbers `_auth_tls_context`'s body occupies."""
    tree = ast.parse(_PARSER_MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_auth_tls_context":
            assert node.end_lineno is not None
            return range(node.lineno, node.end_lineno + 1)
    raise AssertionError(f"_auth_tls_context is no longer defined in {_PARSER_MODULE}")


def _guarded_sources() -> list[Path]:
    """Every shipped Python module, plus the web-terminal Jinja templates."""
    return sorted(_SRC.rglob("*.py")) + sorted(_WEB_TERMINAL_TEMPLATES.rglob("*.j2"))


def test_the_exemption_below_is_load_bearing() -> None:
    """The guard's own guard: `_auth_tls_context` really does hold a
    ``== "none"`` comparison, so the regex demonstrably matches the idiom it
    hunts for. Without this, a typo in the pattern would make the guard pass
    over a codebase full of the comparison it was written to forbid."""
    lines = _PARSER_MODULE.read_text(encoding="utf-8").splitlines()
    span = _parser_line_span()

    assert any(_compares_to_none(lines[number - 1]) for number in span)


def test_only_the_parser_compares_the_auth_method_to_none() -> None:
    """`_auth_tls_context` is the one place the method NAME is interpreted;
    everywhere else reads a derived boolean.

    A comparison that reappears outside the parser is how the four-value scheme
    forks a branch somebody forgot: ``!= "none"`` used to mean "walled", and
    under the new scheme it silently means "walled OR magic-linked OR open",
    which is true of three postures at once and correct for none of the callers.
    The fix for a failure here is to read the boolean that names what the site
    actually asks — ``sidecar_active``, ``inject_secret``, ``walled``,
    ``token_exchange`` or ``open_perimeter`` — not to widen this test.
    """
    sources = _guarded_sources()
    # A moved templates directory would shrink this walk to Python only, and the
    # two files that carried the most `!= "none"` gates are Jinja.
    assert {"nginx.conf.j2", "docker-compose.web.yml.j2"} <= {path.name for path in sources}, (
        f"the web-terminal templates are no longer under {_WEB_TERMINAL_TEMPLATES}"
    )

    exempt = _parser_line_span()
    offenders: list[str] = []

    for path in sources:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not _compares_to_none(line):
                continue
            if path == _PARSER_MODULE and number in exempt:
                continue
            offenders.append(f"{path.relative_to(_SRC)}:{number}: {line.strip()}")

    assert not offenders, (
        "the auth method is compared to 'none' outside `_auth_tls_context`; read a "
        "derived boolean instead:\n" + "\n".join(offenders)
    )
