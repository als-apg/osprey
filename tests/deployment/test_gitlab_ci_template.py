"""Security regression tests for the shipped ``.gitlab-ci.yml`` template.

The ``osprey-build`` job assembles ``.env.production`` from masked CI variables
via a heredoc. That file is COPYed into the runtime web-terminal image, so it
must contain only the runtime secrets the running containers need — never the
build/push/registry tokens that gate CI-registry access.

These tests lock in that heredoc's contents against the same allowlist the
local-mode deploy path enforces in
``osprey.deployment.web_terminals.env_production._build_env_production_subset``
(its docstring is the security spec). They mirror the local-path token-absence
tests in ``tests/deployment/web_terminals/test_env_production.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import osprey

TEMPLATE_PATH = (
    Path(osprey.__file__).parent
    / "templates"
    / "skills"
    / "osprey-build-deploy"
    / "templates"
    / "core"
    / ".gitlab-ci.yml"
)


@pytest.fixture(scope="module")
def template_text() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def env_production_heredoc(template_text: str) -> str:
    """Return the combined bodies of every ``.env.production`` heredoc.

    The template documents ``ci.token_env_var`` elsewhere (the header comment
    listing required CI/CD variables — legitimately, that token still gates
    build/push), so the token-absence assertions must be scoped to the heredoc
    bodies, not the whole file.

    There are two heredocs by design: an unquoted one for the ``$VAR``
    references the runner shell must expand from masked CI variables, and a
    quoted one for config-literal values (``ARIEL_DSN``, ``TZ``) the shell must
    NOT touch — an unquoted heredoc would expand ``$`` sequences and execute
    backticks inside a facility's DSN. Both bodies land in ``.env.production``,
    so both are in scope here.
    """
    matches = re.findall(
        r"cat >>? \.env\.production << '?(?:ENVEOF|LITERALEOF)'?\n(.*?)\n\s*'?(?:ENVEOF|LITERALEOF)",
        template_text,
        re.DOTALL,
    )
    assert len(matches) == 2, "expected the two .env.production heredocs in the template"
    return "\n".join(matches)


def test_ci_token_absent_from_env_production(env_production_heredoc: str) -> None:
    """The CI/registry token must never be written into ``.env.production``."""
    assert "ci.token_env_var" not in env_production_heredoc
    assert "token_env_var}=${env.${config.ci." not in env_production_heredoc


def test_sidecar_token_absent_from_env_production(env_production_heredoc: str) -> None:
    """The dispatcher sidecar token gates build/push, not runtime — excluded."""
    assert "sidecar_token_env_var" not in env_production_heredoc


def test_allowlisted_vars_present_in_env_production(env_production_heredoc: str) -> None:
    """The heredoc still writes the runtime secrets the allowlist permits.

    Guards against an over-broad edit that strips the whole heredoc: these are
    the entries ``_build_env_production_subset`` enumerates.
    """
    assert "${config.llm.api_key_env_var}" in env_production_heredoc
    assert "${config.modules.event_dispatcher.token_env_var}" in env_production_heredoc
    assert "ARIEL_DSN=" in env_production_heredoc
    assert "TZ=" in env_production_heredoc
