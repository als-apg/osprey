"""The one place the shipped Dockerfiles' proxy-delivery idiom is spelled.

Every OSPREY image installs Debian packages, and on a site network that apt
traffic has to go through a proxy. Operators configure the proxy with the
uppercase spellings (``HTTP_PROXY``/``HTTPS_PROXY``/``NO_PROXY``, the names
``.env.shared`` documents and the ones a container runtime forwards into the
build), while apt itself reads the lowercase ones — so every layer that runs
``apt-get`` bridges the two before it does anything else:

``export http_proxy="${http_proxy:-${HTTP_PROXY:-}}" ...``

which fills each lowercase name from its uppercase counterpart while leaving an
explicitly-set lowercase value alone, and expands to the empty string (not an
unbound-variable error) when neither is set.

:func:`assert_apt_runs_carry_proxy_idiom` is that rule as an assertion, applied
to every shipped recipe from :mod:`tests.deployment.test_service_dockerfiles`
(the on-disk service and module Dockerfiles) and
:mod:`tests.cli.test_dockerfile_template` (the rendered project template). It is
deliberately the *only* place the export string appears in the test suite: a
future move to a different single idiom edits this module, not an assertion in
each of the nine recipes.
"""

from __future__ import annotations

import re

#: The canonical bridge, keyed on its first assignment. An apt-using RUN opens
#: with this and may follow it with further exports (see :data:`_ARG_EXPORT`).
PROXY_EXPORT_BRIDGE = 'export http_proxy="${http_proxy:-${HTTP_PROXY:-}}"'

#: The alternative accepted idiom: an opening export that delivers the proxy
#: settings from a declared build ARG rather than from the ambient environment
#: — ``export NO_PROXY="$PIP_NO_PROXY" no_proxy="$PIP_NO_PROXY"`` and kin.
_ARG_EXPORT = re.compile(
    r"^export\s+(?:[A-Za-z_]+=\S+\s+)*"
    r"(?:http_proxy|https_proxy|no_proxy|HTTP_PROXY|HTTPS_PROXY|NO_PROXY)="
    r'"?\$\{?[A-Za-z_][A-Za-z0-9_]*'
)

#: Statement separators. The bridge is one simple command, so the text before
#: the first of these is the RUN's opening statement.
_STATEMENT_BREAK = re.compile(r";|&&|\|\|")


def run_instructions(text: str) -> list[str]:
    """Every ``RUN`` instruction in *text*, reconstructed as Docker runs it.

    Line continuations are joined, and a comment line *inside* a continuation
    is dropped rather than folded into the body — which is how Docker treats
    it, and the difference matters: a naive join turns everything after such a
    comment into inert shell comment text, hiding the commands that actually
    run.
    """
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if current is None and not line.startswith("RUN "):
            continue
        if current is not None and line.strip().startswith("#"):
            continue
        current = [line] if current is None else [*current, line]
        if not line.rstrip().endswith("\\"):
            blocks.append(current)
            current = None
    if current:
        blocks.append(current)
    return [re.sub(r"\\\n\s*", " ", "\n".join(block)) for block in blocks]


def apt_run_instructions(text: str) -> list[str]:
    """The apt-using ``RUN`` instructions in *text*.

    An **apt-using RUN** is a RUN whose reconstructed instruction (see
    :func:`run_instructions`) invokes ``apt-get`` anywhere in its body — the
    install itself, but equally a ``purge``/``autoremove`` cleanup, since those
    reach the network too. A RUN that only edits apt's *configuration* (mirror
    rewrites, retry settings) does not qualify; it needs no proxy to succeed.
    """
    return [instr for instr in run_instructions(text) if "apt-get" in instr]


def _opening_statement(instruction: str) -> str:
    """The first shell statement of a reconstructed ``RUN`` instruction."""
    body = instruction[len("RUN ") :] if instruction.startswith("RUN ") else instruction
    return _STATEMENT_BREAK.split(body, maxsplit=1)[0].strip()


def carries_proxy_idiom(instruction: str) -> bool:
    """Whether a reconstructed RUN opens with a proxy-delivery idiom."""
    opening = _opening_statement(instruction)
    return opening.startswith(PROXY_EXPORT_BRIDGE) or bool(_ARG_EXPORT.match(opening))


def assert_apt_runs_carry_proxy_idiom(text: str, label: str) -> None:
    """Assert every apt-using RUN in *text* delivers the proxy settings.

    *label* names the recipe in failure messages (e.g. ``"qmd"``). The recipe
    must contain at least one apt-using RUN: a file this is called on with
    nothing to check is a parse failure, not a pass.
    """
    apt_runs = apt_run_instructions(text)
    assert apt_runs, (
        f"{label}: no apt-using RUN found — either the recipe stopped installing "
        f"packages, or the RUN parser no longer matches how it is written"
    )
    for instruction in apt_runs:
        assert carries_proxy_idiom(instruction), (
            f"{label}: an apt-using RUN does not open with the proxy-delivery "
            f"idiom, so an apt fetch behind a site proxy hangs at build time:\n"
            f"{instruction}"
        )
