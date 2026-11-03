"""``patch_restore`` drops the guard finders and leaves every other finder in place."""

from __future__ import annotations

import sys

from osprey.runtime import raw_put_block
from tests.runtime import _patch_restore
from tests.runtime._patch_restore import GUARD_ATTR, guard_finders, patch_restore


class _ForeignFinder:
    """A finder some other library inserts; it carries no guard attribute."""

    def find_spec(self, *_args, **_kwargs):
        return None


class _GuardFinder:
    """A finder tagged the way the block tags the finders it installs."""

    def find_spec(self, *_args, **_kwargs):
        return None


setattr(_GuardFinder, GUARD_ATTR, True)


def test_the_guard_marker_is_the_products_own() -> None:
    assert _patch_restore.GUARD_ATTR is raw_put_block._GUARD_ATTR


def test_a_foreign_finder_inserted_inside_the_block_survives_the_exit() -> None:
    before = list(sys.meta_path)
    foreign = _ForeignFinder()
    guard = _GuardFinder()
    try:
        with patch_restore():
            sys.meta_path.insert(0, foreign)
            sys.meta_path.insert(0, guard)
        assert foreign in sys.meta_path
        assert all(sys.meta_path.index(foreign) < sys.meta_path.index(f) for f in before), (
            "the foreign finder stays ahead of the finders that were there before"
        )
        assert guard not in sys.meta_path
    finally:
        if foreign in sys.meta_path:
            sys.meta_path.remove(foreign)
        if guard in sys.meta_path:
            sys.meta_path.remove(guard)


def test_every_finder_present_before_is_kept_in_order() -> None:
    before = list(sys.meta_path)
    foreign = _ForeignFinder()
    try:
        with patch_restore():
            sys.meta_path.insert(0, foreign)
            sys.meta_path.append(_GuardFinder())
        assert [f for f in before if f in sys.meta_path] == before
        assert not guard_finders()
    finally:
        if foreign in sys.meta_path:
            sys.meta_path.remove(foreign)
