"""Whether a connector-host child came up where its derivation said it would.

Every supervisor of :mod:`osprey_connectors.ipc._host` children asks this once
per child start — the controls MCP server's target switch and the library's
multi-target :class:`~osprey_connectors.ipc.ConnectorHostPool` alike.
**Verification** (:func:`verify_host_report`, and :func:`verify_child_report`
for the gateway case) compares a child's post-connect report against the
derivation :func:`~osprey_connectors.gateways.derive_endpoints` answers from
config: not "nothing looked wrong" but "the child configured exactly this host,
this port, this mode, for exactly the role the derivation selected". A mismatch
names the field, the expected value and the value received.

Nothing here imports a control-system client library.
"""

from __future__ import annotations

from collections.abc import Mapping

from osprey_connectors.gateways import TargetDerivation, _ports_equal
from osprey_connectors.model import FrozenModel

__all__ = ["Verification", "verify_host_report"]


class Verification(FrozenModel):
    """Whether the child came up where the derivation said it would."""

    ok: bool
    field: str | None = None
    expected: object = None
    got: object = None
    detail: str = ""


def verify_child_report(derivation: TargetDerivation, report: Mapping[str, object]) -> Verification:
    """Assert a child came up exactly where the derivation said it would.

    Positive and role-aware: every field of the selected role's endpoint is
    compared, and the child must also say it configured EPICS at all — a child
    that connected without a gateway leaves ``_epics_configured`` false and has
    silently inherited whatever CA environment the process already carried.

    Args:
        derivation: The derivation for the target being switched to, from
            :func:`derive_endpoints`.
        report: The child's post-connect report: ``selected_role``, ``mode``,
            ``host``, ``port`` and ``_epics_configured``.

    Returns:
        A passing :class:`Verification`, or a failing one naming the field, the
        expected value and the value the child reported. The switch aborts on a
        failure and leaves the previous target active.
    """

    if not report.get("_epics_configured"):
        return Verification(
            ok=False,
            field="_epics_configured",
            expected=True,
            got=report.get("_epics_configured"),
            detail="The child reports it never configured an EPICS gateway, so its "
            "environment is whatever the process already carried rather than this "
            "target's.",
        )

    expected_role = derivation.selected_role
    got_role = report.get("selected_role")
    if got_role != expected_role:
        return Verification(
            ok=False,
            field="selected_role",
            expected=expected_role,
            got=got_role,
            detail=f"The child selected the {got_role!r} gateway where target "
            f"{derivation.target!r} derives {expected_role!r}.",
        )

    endpoint = derivation.selected_endpoint()
    if endpoint is None:
        return Verification(
            ok=False,
            field="endpoints",
            expected=f"a derived {expected_role!r} endpoint",
            got=None,
            detail=f"Target {derivation.target!r} has no derived {expected_role!r} endpoint "
            "to compare the child against; it should never have been switched to.",
        )

    for field_name, expected in (("mode", endpoint.mode), ("host", endpoint.host)):
        got = report.get(field_name)
        if got != expected:
            return Verification(
                ok=False,
                field=field_name,
                expected=expected,
                got=got,
                detail=f"The child's {field_name} is {got!r} where target "
                f"{derivation.target!r} derives {expected!r} for its "
                f"{expected_role!r} gateway.",
            )

    if not _ports_equal(endpoint.port, report.get("port")):
        return Verification(
            ok=False,
            field="port",
            expected=endpoint.port,
            got=report.get("port"),
            detail=f"The child's port is {report.get('port')!r} where target "
            f"{derivation.target!r} derives {endpoint.port!r} for its "
            f"{expected_role!r} gateway.",
        )

    return Verification(
        ok=True,
        detail=(
            f"The child is on target {derivation.target!r} via its {expected_role!r} "
            f"gateway at {endpoint.host}:{endpoint.port} ({endpoint.mode})."
        ),
    )


def _verify_posture(
    derivation: TargetDerivation,
    report: Mapping[str, object],
    *,
    readonly_run: bool,
    writes_enabled: bool | None,
) -> Verification | None:
    """The identity and posture failure in *report*, or ``None`` when there is none.

    The child echoes the target it was sent, so only the type it resolved that
    target to can differ — a child from another build of the package. Posture
    is checked for every connector type: the endpoint check sees it only
    through the gateway role, which a connector with no gateways, or with a
    ``read_only`` row alone, never varies.
    """
    got_type = report.get("connector_type")
    if got_type != derivation.connector_type:
        return Verification(
            ok=False,
            field="connector_type",
            expected=derivation.connector_type,
            got=got_type,
            detail=f"The child reports connector type {got_type!r} where target "
            f"{derivation.target!r} derives {derivation.connector_type!r}.",
        )
    got_transport = report.get("transport")
    if got_transport != derivation.transport:
        return Verification(
            ok=False,
            field="transport",
            expected=derivation.transport,
            got=got_transport,
            detail=f"The child for target {derivation.target!r} reports transport "
            f"{got_transport!r} where {derivation.transport!r} was derived.",
        )
    if readonly_run:
        got_readonly = report.get("readonly_run")
        if got_readonly is not True:
            return Verification(
                ok=False,
                field="readonly_run",
                expected=True,
                got=got_readonly,
                detail=f"The child for target {derivation.target!r} was asked to run readonly "
                "but reports it is not in a readonly run.",
            )
        return None
    if writes_enabled is None:
        return None
    got_writes = report.get("writes_enabled")
    if (got_writes is True) != writes_enabled:
        return Verification(
            ok=False,
            field="writes_enabled",
            expected=writes_enabled,
            got=got_writes,
            detail=f"The child for target {derivation.target!r} reports writes "
            f"{'armed' if got_writes is True else 'off'} where this process derives them "
            f"{'armed' if writes_enabled else 'off'}.",
        )
    return None


def verify_host_report(
    derivation: TargetDerivation,
    report: Mapping[str, object],
    *,
    readonly_run: bool | None = None,
    writes_enabled: bool | None = None,
) -> Verification:
    """Assert the child came up where the derivation said it would.

    Given the posture the derivation was taken under (*readonly_run*,
    *writes_enabled*), the report is first held to it: its ``connector_type``
    and its ``transport`` must be the derived ones, and then a readonly run requires the report's
    ``readonly_run`` to be ``True`` while any other run requires its
    ``writes_enabled`` to equal *writes_enabled*. With neither given, only the
    endpoint check below runs.

    :func:`verify_child_report` answers this for every target whose config
    names a gateway. A deployment can also select a connector that talks to no
    gateway at all — the simulator served in process is one, and it is the
    fail-safe default — and for that one the derivation has no endpoint and the child reports
    none. Nothing is verified there because there is no endpoint to get wrong,
    but the *symmetry* is: a child that configured Channel Access where the
    config derived nothing has inherited an environment from somewhere, and
    that is a mismatch as serious as any other.

    The unverified branch is entered only when the derivation has **no endpoint
    rows at all**. A target with rows whose *selected* role is missing — a
    write-only gateway table on a deployment that selects ``read_only``, say —
    is a gateway deployment with a hole in it, not a gatewayless one: its child
    connects to a real control system over whatever default broadcast address
    it finds, and that is precisely the unpinned-CA case verification exists to
    catch. It goes to :func:`verify_child_report`, which refuses it either for
    reporting no gateway or for having no derived endpoint to compare against.

    Args:
        derivation: The derivation for the target, from :func:`derive_endpoints`.
        report: The child's post-connect report.
        readonly_run: Whether the derivation was taken for a readonly run.
        writes_enabled: Whether the derivation was taken with writes armed.

    Returns:
        A passing :class:`Verification`, or a failing one naming the field
        (``connector_type``, ``transport``, ``readonly_run``, ``writes_enabled`` or an
        endpoint field), the expected value and the value the child reported.
    """
    if readonly_run is not None or writes_enabled is not None:
        refusal = _verify_posture(
            derivation, report, readonly_run=bool(readonly_run), writes_enabled=writes_enabled
        )
        if refusal is not None:
            return refusal

    if derivation.endpoints:
        return verify_child_report(derivation, report)

    if report.get("_epics_configured") or report.get("mode") or report.get("host"):
        return Verification(
            ok=False,
            field="_epics_configured",
            expected=False,
            got=report.get("_epics_configured"),
            detail=f"Target {derivation.target!r} derives no gateway on this deployment, but the "
            f"child reports it configured {report.get('mode')!r} routing to "
            f"{report.get('host')!r} — an endpoint this config never described.",
        )
    if derivation.declares_no_endpoints:
        return Verification(
            ok=True,
            detail=(
                f"Target {derivation.target!r} declares no endpoints (`gateways: []`), "
                "so there is no gateway to check the child against, and it configured none."
            ),
        )
    return Verification(
        ok=True,
        detail=(
            f"Target {derivation.target!r} derives no gateway (connector type "
            f"{derivation.connector_type!r}) and the child configured none."
        ),
    )
