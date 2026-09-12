"""Which provider an end-to-end run builds with.

Two environment variables decide it and this module is where they meet, so the
precedence between them is stated once rather than rediscovered at each call
site. It carries no pytest fixtures on purpose: the root ``tests/conftest.py``
needs the same answer to gate the lanes on a credential, and one conftest
importing another is not a seam worth opening.
"""

import os

#: Environment variable naming the provider the build-and-run e2e lanes drive.
#:
#: These lanes init a real deployment repo and run an agent against it, so they
#: need a provider whose credential the runner actually holds. That is one fact
#: about a CI environment, not about OSPREY: a facility running this suite on
#: its own gateway sets this variable and the lanes follow it, instead of
#: patching four modules that each spelled one gateway's name inline.
E2E_PROVIDER_ENV = "OSPREY_E2E_PROVIDER"

#: Environment variable overriding the provider of *every* project a run builds,
#: whatever each call site pinned. The benchmark matrix sets it per cell.
FORCE_PROVIDER_ENV = "OSPREY_E2E_FORCE_PROVIDER"


def provider_refusal() -> str:
    """What to tell a run that has not said which provider it builds with.

    The providers are read from the registry's own key table rather than listed
    here, so the refusal cannot name a set OSPREY does not have.
    """
    from osprey.models.provider_registry import PROVIDER_API_KEYS

    known = ", ".join(sorted(PROVIDER_API_KEYS))
    return (
        f"This end-to-end run names no provider. Set {E2E_PROVIDER_ENV} to the provider "
        f"whose credential this environment holds, or {FORCE_PROVIDER_ENV} to point the "
        f"whole suite at one provider. Known providers: {known}."
    )


def e2e_provider() -> str:
    """The provider these lanes build their deployment repo with.

    No constant stands behind the two variables. A gateway is whoever runs it,
    so a run that named none is refused by name instead of being sent somewhere
    a default happened to point. The override is honored first: the benchmark
    runner requires it and sets nothing else.

    Raises:
        RuntimeError: when neither variable names a provider.
    """
    forced = os.environ.get(FORCE_PROVIDER_ENV, "").strip()
    if forced:
        return forced
    selected = os.environ.get(E2E_PROVIDER_ENV, "").strip()
    if selected:
        return selected
    raise RuntimeError(provider_refusal())


def build_provider(pinned: str) -> str:
    """The provider to build a project with, given what the call site pinned.

    Suite-wide override (CBORG model-matrix, issue #259): when
    :data:`FORCE_PROVIDER_ENV` names a provider it replaces the per-call-site
    ``pinned`` value so the *entire* ``tests/e2e/`` suite can be pointed at one
    provider without editing each fixture. Paired with ``OSPREY_E2E_FORCE_MODEL``
    (honored in ``sdk_helpers._resolve_project_spec``), which collapses all
    tiers onto a single model id.

    The override is read the way :func:`e2e_provider` reads it — stripped, and
    empty means unset — so the two functions of this module agree on what the
    same shell said.
    """
    forced = os.environ.get(FORCE_PROVIDER_ENV, "").strip()
    return forced or pinned
