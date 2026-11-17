"""Post-connect reports for the tests that hand one to a verifier.

A :class:`~osprey_connectors.ipc.HostReport` holds every field a child writes,
and a verification test names only the ones its case is about. The rest are a
child that configured nothing, on ``live``, with writes off.
"""

from osprey_connectors.ipc._host import HostReport

#: A child on ``live`` that configured no gateway, with writes off.
_NOTHING_CONFIGURED: dict[str, object] = {
    "selected_role": None,
    "mode": None,
    "host": None,
    "port": None,
    "_epics_configured": False,
    "connector_type": "epics",
    "transport": None,
    "target": "live",
    "writes_enabled": False,
    "readonly_run": False,
    "authorization": None,
    "epics_env": {},
    "pid": 4242,
}


def host_report(**fields: object) -> HostReport:
    """A report with *fields*, keyed as the wire keys them; the rest configure nothing."""
    return HostReport.model_validate({**_NOTHING_CONFIGURED, **fields})
