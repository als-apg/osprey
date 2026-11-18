"""Shared fixtures for the web-terminal render tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from jinja2 import BytecodeCache, Environment
from jinja2.bccache import Bucket

from osprey.deployment.web_terminals import render as render_module


class _InMemoryBytecodeCache(BytecodeCache):
    """Jinja's own compiled-template cache, held in memory for this process."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def load_bytecode(self, bucket: Bucket) -> None:
        data = self._store.get(bucket.key)
        if data is not None:
            bucket.bytecode_from_string(data)

    def dump_bytecode(self, bucket: Bucket) -> None:
        self._store[bucket.key] = bucket.bytecode_to_string()


# Held at module level, not in the fixture, so the package fixture below can be
# set up again (xdist may hand a worker this directory's tests in several runs)
# without compiling anything twice.
_BYTECODE_CACHES: dict[object, _InMemoryBytecodeCache] = {}
_PACKAGED_TOKEN_TREE: list[Any] = []


@pytest.fixture(autouse=True, scope="package")
def _render_reuses_parsed_templates_and_tokens():
    """Parse each template and load the token tree once per worker, not per render.

    Every render builds two fresh Jinja Environments and re-reads the packaged
    design tokens three times, and those reparses are most of a render's time.
    The Environments stay fresh -- a new one per render, with fresh Template
    objects, so a test that wraps a template's ``render`` touches only its own --
    they just share a bytecode cache, one per ``autoescape`` setting since that
    changes the compiled code. Jinja keys each entry by template and checks it
    against the source's checksum, so an edited template recompiles. The token
    tree is memoized for the packaged ``tokens/`` directory only.

    Package scope, so the patch is lifted when this directory's tests are done
    and never reaches a test elsewhere that loads the token tree itself. One
    thing does run under it: the run's shared ``built_demo``, when a test here is
    the first to ask for it. That is harmless: a demo build never calls the
    patched ``Environment`` (counted: none), and the token tree's one consumer,
    ``theme_config``, only reads the memoized tree.
    """
    from osprey.interfaces.design_system.generator import model
    from osprey.interfaces.design_system.generator.build import DEFAULT_TOKENS_DIR

    def environment(*args: Any, **kwargs: Any) -> Environment:
        autoescape = kwargs.get("autoescape", False)
        kwargs["bytecode_cache"] = _BYTECODE_CACHES.setdefault(autoescape, _InMemoryBytecodeCache())
        return Environment(*args, **kwargs)

    real_load_token_tree = model.load_token_tree

    def load_token_tree(tokens_dir: Path) -> Any:
        if Path(tokens_dir) != Path(DEFAULT_TOKENS_DIR):
            return real_load_token_tree(tokens_dir)
        if not _PACKAGED_TOKEN_TREE:
            _PACKAGED_TOKEN_TREE.append(real_load_token_tree(tokens_dir))
        return _PACKAGED_TOKEN_TREE[0]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(render_module, "Environment", environment)
        mp.setattr(model, "load_token_tree", load_token_tree)
        yield
