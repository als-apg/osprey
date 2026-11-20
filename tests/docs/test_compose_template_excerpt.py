"""The compose-templates guide quotes the shipped templates, not a drifted copy.

``docs/source/how-to/deploy-project/compose-templates.rst`` shows an excerpt of
the PostgreSQL service template. Every line of that excerpt must be a line of
the shipped template, so the excerpt cannot teach a key or a name the template
no longer uses. The guide also names the ``osprey.env.digest`` label a template
writes itself; that label line must be one the shipped templates carry. The
run-once label block names the label and the values the build reads.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import yaml

from osprey.deployment import run_once

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PAGE = _REPO_ROOT / "docs" / "source" / "how-to" / "deploy-project" / "compose-templates.rst"
_SERVICES = _REPO_ROOT / "src" / "osprey" / "templates" / "services"
_POSTGRES = _SERVICES / "postgresql" / "docker-compose.yml.j2"


def _excerpt() -> str:
    text = _PAGE.read_text(encoding="utf-8")
    match = re.search(
        r"^\.\. code-block:: yaml\n   :caption: services/postgresql/docker-compose\.yml\.j2 \(excerpt\)\n\n"
        r"((?:   .*\n|\n)+?)(?=\S)",
        text,
        re.M,
    )
    assert match is not None, "the PostgreSQL template excerpt is missing"
    return textwrap.dedent(match.group(1))


def test_every_excerpt_line_is_a_template_line() -> None:
    template_lines = set(_POSTGRES.read_text(encoding="utf-8").splitlines())
    excerpt_lines = [line for line in _excerpt().splitlines() if line.strip()]
    assert excerpt_lines
    missing = [line for line in excerpt_lines if line not in template_lines]
    assert not missing, f"not in {_POSTGRES.name}: {missing}"


def test_env_digest_label_is_the_shipped_one() -> None:
    text = _PAGE.read_text(encoding="utf-8")
    match = re.search(r"^      (osprey\.env\.digest: .+)$", text, re.M)
    assert match is not None
    label = match.group(1)
    carriers = [
        path
        for path in sorted(_SERVICES.glob("*/docker-compose.yml.j2"))
        if any(line.strip() == label for line in path.read_text(encoding="utf-8").splitlines())
    ]
    assert carriers, f"no shipped service template carries {label!r}"


def test_run_once_label_block_is_the_declaration_the_build_reads() -> None:
    text = _PAGE.read_text(encoding="utf-8")
    match = re.search(r"^      labels:\n((?:        .*\n)+)", text, re.M)
    assert match is not None, "the run-once label block is missing"
    block = match.group(1)
    labels = yaml.safe_load(textwrap.dedent(block))
    assert labels == {run_once.RUN_ONCE_LABEL: run_once.MUST_SUCCEED}
    named = set(re.findall(r"[a-z]+-[a-z]+", block.split("#", 1)[1]))
    assert named == {run_once.MAY_FAIL}
    assert {run_once.MUST_SUCCEED, run_once.MAY_FAIL} == set(run_once.VALUES)
