"""Published documentation addresses that runtime messages link.

Runtime refusals and warnings that enforce a documented limit link the page
that states it. The addresses are the published site's, not a deployment's
``web.docs_url``, because these lines reach a terminal or a container log.

The module imports nothing, so :mod:`osprey.port_layout` stays a stdlib-only
leaf.
"""

#: Where the unit's ``Documentation=`` points. The deployment how-to is the page
#: that covers what a host needs before the unit can bring a stack up.
DEPLOY_DOCS_URL: str = "https://als-apg.github.io/osprey/how-to/deploy-a-facility.html"

#: The section of the deployment how-to that states what the multi-user
#: perimeter needs: its own hostname or host:port, one origin, the host network,
#: and a user ceiling.
PERIMETER_LIMITS_URL: str = f"{DEPLOY_DOCS_URL}#perimeter-limits"
