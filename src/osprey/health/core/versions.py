"""Core ``versions`` health category.

States which code this deployment runs, read from the running interpreter:

* ``versions.osprey`` — the running osprey, from
  :func:`osprey.version.get_running_version`: the commit (``g<hex>`` label) for
  a development build, the version for a release. Always ``ok``; it is a fact,
  not a verdict.
* ``versions.canopy`` — emitted only when the config declares the CANOPY panel
  (``web.panels.canopy``): the installed CANOPY commit against the commit this
  osprey is proven against (:mod:`osprey.integrations.canopy.proven`, the one
  reader of that pin). ``ok`` at the pin, ``warning`` at any other commit or
  install without a recorded commit, ``skip`` when CANOPY is not installed in
  this interpreter (it may be installed in the image that serves the panel).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from collections.abc import Mapping

    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime
    from osprey.integrations.canopy.proven import Judgement

CATEGORY = "versions"

#: Digits of a pinned commit the row shows.
_SHOWN_HEX = 12


def versions(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,  # noqa: ARG001 - health category factory signature; categories that probe a runtime read the context
) -> CategoryCallable:
    """Build the ``versions`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable). Read
            for ``web.panels.canopy``.
        context: Health runtime. Unused.

    Returns:
        A no-argument callable returning the category's check results.
    """
    cfg: Mapping[str, Any] = config or {}

    def _run() -> list[CheckResult]:
        rows = [_osprey_row(_running_version())]
        from osprey.deployment.web_terminals.roster import config_declares_panel
        from osprey.profiles.sidecar_panels import CANOPY_PANEL_ID

        if config_declares_panel(cfg, CANOPY_PANEL_ID):
            rows.append(_canopy_row(_canopy_judgement()))
        return rows

    return _run


def _running_version() -> str:
    from osprey.version import get_running_version

    return get_running_version()


def _canopy_judgement() -> Judgement:
    from osprey.integrations.canopy.proven import judge

    return judge()


def _osprey_row(running: str) -> CheckResult:
    from osprey.integrations.canopy.proven import version_commit

    # A dirty tree carries a ``d<date>`` label that version_commit refuses; the
    # commit it descends from is still the useful fact, so read the ``g`` label
    # on its own.
    _, _, local = running.partition("+")
    commit = version_commit("+" + local.split(".")[0]) if local else None
    return CheckResult(
        "versions.osprey",
        CATEGORY,
        Status.OK,
        f"osprey {running}",
        value=commit or running,
    )


def _canopy_row(judgement: Judgement) -> CheckResult:
    from osprey.integrations.canopy.proven import NOT_INSTALLED

    reading = next(r for r in judgement.readings if r.distribution == "canopy")
    pin = reading.proven[:_SHOWN_HEX]
    if reading.at_pin:
        return CheckResult(
            "versions.canopy",
            CATEGORY,
            Status.OK,
            "CANOPY at the proven commit",
            value=f"{pin} = pin",
        )
    if reading.held.reason == NOT_INSTALLED:
        return CheckResult(
            "versions.canopy",
            CATEGORY,
            Status.SKIP,
            "CANOPY is not installed in this interpreter",
            value="not installed here",
        )
    return CheckResult(
        "versions.canopy",
        CATEGORY,
        Status.WARNING,
        f"CANOPY is not at the commit this osprey is proven against ({reading.proven})",
        value=f"{reading.held.shown()} ≠ pin {pin}",
    )
