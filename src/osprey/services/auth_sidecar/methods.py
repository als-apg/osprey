"""The auth methods this sidecar can actually serve.

A leaf module with no sidecar imports of its own, so both layers that need the
set can have it: :mod:`~osprey.services.auth_sidecar.app`, which refuses to
serve a method outside it, and
:mod:`~osprey.services.auth_sidecar.routes.recheck`, which refuses to mint a
session for one. Those two cannot import each other — ``app`` builds the routes
— so before this module the set was spelled twice, once as a tuple and once as
a frozenset, and :mod:`~osprey.services.auth_sidecar.audit` keyed its success
categories off a third copy written as bare strings.

Distinct from :data:`~osprey.deployment.web_terminals.render.SUPPORTED_AUTH_METHODS`,
which is a different fact: that one enumerates the deployment *postures* a
render may declare, and includes ``none`` and ``token`` — postures under which
no sidecar runs at all.
"""

from __future__ import annotations

METHOD_PASSWORD = "password"
"""The roster-credential posture: this service verifies the proof itself."""

METHOD_OIDC = "oidc"
"""The federated posture: an IdP proves the identity and may decide the role."""

SUPPORTED_METHODS: frozenset[str] = frozenset({METHOD_PASSWORD, METHOD_OIDC})
"""The methods a session may be minted for.

Matched exactly, never case-folded: the value arrives from the environment
through :class:`~osprey.services.auth_sidecar.app.AuthSettings`, which already
lowercases it once. Folding again here would mean two components disagreeing
about what counts as a match, and the one that is stricter should be the one
handing out sessions.
"""
