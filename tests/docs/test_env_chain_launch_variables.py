"""The env-chain page's launch-variable paragraph matches the refusal it describes.

``docs/source/how-to/deploy-project/env-chain.rst`` tells a deployer which
variables the ``.env`` chain never applies. The refusal itself lives in
:data:`osprey_connectors.config.SHELL_ONLY_ENV_VARS` and
:data:`osprey_connectors.config.SHELL_ONLY_ENV_PREFIXES`. The paragraph names
most entries literally and covers the rest with a phrase for a whole group, so
each direction is checked here: every refused name or prefix is on the page,
and every name the page lists is refused.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pytest

from osprey_connectors.config import (
    SHELL_ONLY_ENV_PREFIXES,
    SHELL_ONLY_ENV_VARS,
    is_shell_only_env_var,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The page this test reads, relative to the repo root.
_PAGE = "docs/source/how-to/deploy-project/env-chain.rst"

#: The bold lead that opens the paragraph stating the rule.
_LEAD = "**Launch variables come from the shell only.**"

#: An RST inline literal: ``like this``.
_LITERAL_PATTERN = re.compile(r"``([^`\n]+)``")

#: A literal that is entirely an upper-case variable name or prefix.
_UPPER_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")

#: Names the paragraph covers with a phrase instead of naming each one. The key
#: is the phrase as it reads on the page, whitespace collapsed; the value is
#: every refused name that phrase stands for.
_GROUPS: dict[str, frozenset[str]] = {
    "start-up files or run a command at its prompt": frozenset(
        {"ENV", "BASH_ENV", "ZDOTDIR", "PROMPT_COMMAND"}
    ),
    "the Python, Node, Perl, Ruby and Java start-up and load-path variables": frozenset(
        {
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONSTARTUP",
            "PYTHONUSERBASE",
            "PYTHONWARNINGS",
            "PYTHONINSPECT",
            "PYTHONBREAKPOINT",
            "PYTHONEXECUTABLE",
            "PYTHONPLATLIBDIR",
            "PYTHONPYCACHEPREFIX",
            "NODE_OPTIONS",
            "NODE_PATH",
            "NODE_REPL_EXTERNAL_MODULE",
            "PERL5LIB",
            "PERLLIB",
            "PERL5OPT",
            "PERL5DB",
            "RUBYOPT",
            "RUBYLIB",
            "JAVA_TOOL_OPTIONS",
            "_JAVA_OPTIONS",
            "JDK_JAVA_OPTIONS",
            "CLASSPATH",
        }
    ),
    "askpass helpers": frozenset({"SSH_ASKPASS", "SUDO_ASKPASS"}),
    "their flag and include-path variables": frozenset(
        {
            "CFLAGS",
            "CPPFLAGS",
            "CXXFLAGS",
            "LDFLAGS",
            "ARCHFLAGS",
            "CPATH",
            "C_INCLUDE_PATH",
            "CPLUS_INCLUDE_PATH",
            "LIBRARY_PATH",
            "COMPILER_PATH",
            "GCC_EXEC_PREFIX",
        }
    ),
}


@lru_cache(maxsize=1)
def _paragraph() -> str:
    """The paragraph from the bold lead to the next blank line."""
    text = (_REPO_ROOT / _PAGE).read_text(encoding="utf-8")
    start = text.find(_LEAD)
    assert start != -1, f"{_PAGE} has no paragraph opening with {_LEAD}"
    end = text.find("\n\n", start)
    return text[start : end if end != -1 else len(text)]


def _literals() -> set[str]:
    return set(_LITERAL_PATTERN.findall(_paragraph()))


def _collapsed() -> str:
    return " ".join(_paragraph().split())


@pytest.mark.parametrize("prefix", sorted(SHELL_ONLY_ENV_PREFIXES))
def test_every_shell_only_prefix_is_on_the_page(prefix: str) -> None:
    assert prefix in _literals(), f"{_PAGE} does not name the refused prefix ``{prefix}``"


@pytest.mark.parametrize("name", sorted(SHELL_ONLY_ENV_VARS))
def test_every_shell_only_name_is_on_the_page(name: str) -> None:
    if name in _literals():
        return
    paragraph = _collapsed()
    phrases = [phrase for phrase, names in _GROUPS.items() if name in names]
    assert any(phrase in paragraph for phrase in phrases), (
        f"{_PAGE} neither names ``{name}`` nor carries a group phrase covering it"
    )


def test_every_name_the_paragraph_lists_is_refused() -> None:
    listed = sorted(n for n in _literals() if _UPPER_NAME_PATTERN.match(n))
    assert listed, f"{_PAGE} launch-variable paragraph lists no names"
    not_refused = [n for n in listed if not is_shell_only_env_var(n)]
    assert not not_refused, f"{_PAGE} lists names the env chain does not refuse: {not_refused}"
