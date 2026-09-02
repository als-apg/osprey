"""The ``.env`` block banners the deploy writes name a command that exists.

A banner is matched against files already on operators' disks, so its text is
a data format: the reader tuples (:data:`MINTED_ENV_BANNERS` for ``.env``, the
header tuples in ``auth_credentials`` for ``.env.auth``) only ever grow. What
this file pins on top of that append-only contract is the WRITER side — the
banner a deploy writes today must be one its reader knows, and must not name a
retired verb, because a banner is also the one line of prose an operator reads
when they open the file to see where a value came from.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from osprey.deployment import container_lifecycle
from osprey.deployment import reset as reset_mod
from osprey.deployment.web_terminals import auth_credentials

#: The verbs that no longer exist. ``osprey deploy`` alone covers both the
#: ``osprey deploy up`` banners and the verb-less ``(osprey deploy)`` headers.
_RETIRED_VERB = re.compile(r"\bosprey deploy\b")


def _written_env_banners() -> set[str]:
    """Every literal banner ``container_lifecycle`` passes to ``_append_env_block``."""
    source = Path(container_lifecycle.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    return {
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_append_env_block"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    }


def test_every_written_env_banner_is_one_reset_reads() -> None:
    written = _written_env_banners()
    assert written, "found no _append_env_block call sites — the gate has stopped working"
    assert written <= set(reset_mod.MINTED_ENV_BANNERS), sorted(
        written - set(reset_mod.MINTED_ENV_BANNERS)
    )


def test_no_written_env_banner_names_a_retired_verb() -> None:
    offenders = sorted(banner for banner in _written_env_banners() if _RETIRED_VERB.search(banner))
    assert offenders == [], f"deploy writes a .env banner naming a dead verb: {offenders}"


def test_env_banner_reader_still_carries_the_retired_spellings() -> None:
    """The tuple is append-only: the banners earlier releases wrote stay matchable."""
    retired = [b for b in reset_mod.MINTED_ENV_BANNERS if _RETIRED_VERB.search(b)]
    assert len(retired) >= 3, "reset dropped a retired banner; .env files on disk still carry it"


def test_auth_headers_are_members_of_their_reader_tuples() -> None:
    assert auth_credentials._HASH_HEADER in auth_credentials._HASH_HEADERS
    assert auth_credentials._SECRET_HEADER in auth_credentials._SECRET_HEADERS


def test_written_auth_headers_name_no_retired_verb() -> None:
    for header in (auth_credentials._HASH_HEADER, auth_credentials._SECRET_HEADER):
        assert not _RETIRED_VERB.search(header), header


def test_auth_header_readers_still_carry_the_retired_spellings() -> None:
    """``.env.auth`` files written by earlier releases carry the old headers."""
    for headers in (auth_credentials._HASH_HEADERS, auth_credentials._SECRET_HEADERS):
        assert any(_RETIRED_VERB.search(h) for h in headers), headers
        assert not _RETIRED_VERB.search(headers[-1]), headers
