"""Native OSPREY ``okf`` builtin web panel — read-only OKF knowledge browser.

Serves the "KNOWLEDGE" tab: a two-pane SPA (concept tree + markdown reader +
search) over a facility-knowledge bundle, backed directly by core
:class:`osprey.services.facility_knowledge.okf.bundle.OKFBundle`.

Launched in-process by ``ServerLauncher`` and reverse-proxied at ``/panel/okf/``
(the ``channel_finder`` builtin pattern). Reads ``facility_knowledge.bundle_path``
from resolved osprey config via the registry's ``factory_config_kwargs``. No
env-var indirection, no vendored ``okf/`` copy.
"""
