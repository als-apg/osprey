"""Pytest fixtures for an application's own test suite, registered by installing the framework.

``osprey-framework`` declares this module under the ``pytest11`` entry-point
group, so a deployment repository's tests get both fixtures with no
``conftest.py`` line:

* ``synthetic_deployment`` makes a deployment under the test's temporary
  directory (:func:`osprey.apps.sdk.testing.synthetic_deployment`, the same
  keywords);
* ``rendered_tree`` makes one and builds it, answering the
  :class:`~osprey.apps.sdk.testing.RenderedTree`.

Both are factories, so a test states its applications, facility and logins at
the call. Nothing is imported until a test asks for one.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from osprey.apps.sdk.testing import Deployment, RenderedTree


@pytest.fixture
def synthetic_deployment(tmp_path: Path) -> Callable[..., Deployment]:
    """Make a deployment under the test's temporary directory; each call makes another.

    Takes the keywords of :func:`osprey.apps.sdk.testing.synthetic_deployment`.
    """
    from osprey.apps.sdk import testing

    serial = itertools.count(1)

    def make(**kwargs: Any) -> Deployment:
        return testing.synthetic_deployment(tmp_path / f"deployment-{next(serial)}", **kwargs)

    return make


@pytest.fixture
def rendered_tree(synthetic_deployment: Callable[..., Deployment]) -> Callable[..., RenderedTree]:
    """Make a deployment and build it; each call makes another.

    Takes the keywords of :func:`osprey.apps.sdk.testing.synthetic_deployment`.
    """

    def build(**kwargs: Any) -> RenderedTree:
        return synthetic_deployment(**kwargs).build()

    return build
