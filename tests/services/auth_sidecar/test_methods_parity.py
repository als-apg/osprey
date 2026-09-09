"""The servable auth-method set has one producer, and everything keyed by it agrees.

``methods.py`` is a leaf: ``app`` refuses to serve a method outside the set,
``routes.recheck`` refuses to mint a session for one, and ``audit`` keys the
category each success is recorded under by the same constants. Those three
cannot reach each other — ``app`` builds the routes — so before the leaf module
the set was written out three times, twice with different container types.

The last test is the seam to the *other* method vocabulary: ``render.py``
enumerates deployment postures, which is a different fact and deliberately
larger. What must hold is containment — a sidecar method a render cannot even
declare would be unreachable.
"""

from __future__ import annotations

import pytest

from osprey.deployment.web_terminals import render
from osprey.services.auth_sidecar import app, audit
from osprey.services.auth_sidecar.methods import METHOD_OIDC, METHOD_PASSWORD, SUPPORTED_METHODS
from osprey.services.auth_sidecar.routes import recheck

pytestmark = pytest.mark.unit


def test_the_set_is_the_two_named_constants() -> None:
    assert SUPPORTED_METHODS == frozenset({METHOD_PASSWORD, METHOD_OIDC})


def test_both_consumers_read_the_one_producer() -> None:
    """A second definition anywhere makes this fail rather than drift."""
    assert app.SUPPORTED_METHODS is SUPPORTED_METHODS
    assert recheck.SUPPORTED_METHODS is SUPPORTED_METHODS


def test_every_servable_method_has_an_audit_category() -> None:
    """A method with no success reason would be recorded as a generic login."""
    assert set(audit._SUCCESS_REASONS) == set(SUPPORTED_METHODS)


def test_the_servable_set_is_within_the_declarable_postures() -> None:
    """``none`` and ``token`` are postures under which no sidecar runs at all."""
    assert SUPPORTED_METHODS <= set(render.SUPPORTED_AUTH_METHODS) - {"none", "token"}
