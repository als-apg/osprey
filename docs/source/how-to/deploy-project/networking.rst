============================
Network Binding and Security
============================

Services bind to ``127.0.0.1`` by default. Reaching them from off-host is a
property of the build, not of a start-time flag: the bind address is rendered
into every published port. Change it with ``osprey set
config.deployment.bind_address=0.0.0.0`` and rebuild, and only when you have
authentication and firewalling in place.

Container networking uses service names as hostnames (e.g.,
``postgresql:5432``). For host access from inside containers, use
``host.docker.internal`` (Docker) or ``host.containers.internal`` (Podman).

.. _deployment-network-attachment:

Network attachment: ``bridge`` or ``host``
------------------------------------------

By default every service joins the compose-managed project network
(``osprey-network``) and publishes the ports it wants reachable. That is
``bridge``, and it is what a deployment gets when it says nothing.

Some services cannot work that way. A service that has to see broadcast traffic
— control-system protocols, device discovery — or that has to reach ports other
software already publishes on the machine needs the host's own network
namespace instead. That is ``host``:

.. code-block:: yaml

   # in profile.yml — the event dispatcher and its workers
   dispatch:
     network: host

   # a facility-owned service
   services:
     my-service:
       template: services/my-service
       config:
         network: host

``dispatch.network`` is deliberately **one knob for two services**. The event
dispatcher and its workers talk to each other over addresses the build writes,
so a dispatcher on the compose network and workers on the host's could not
reach each other at all. Writing ``network:`` on ``services.event_dispatcher``
or ``services.dispatch_worker`` individually is rejected by ``osprey build``,
which tells you to set ``dispatch.network`` instead.

.. raw:: html
   :file: ../../_diagrams/network-attachment.html

Under ``network: host`` the render changes in four ways:

* No ``ports:`` block. There is nothing to publish — the container's listening
  socket *is* a host socket, on the port the service was configured with.
* ``network_mode: host`` replaces the service's ``osprey-network`` membership.
* Services bind **loopback**, ``127.0.0.1``, rather than every interface. On
  the compose network, binding every interface is what makes a service
  reachable by name and the network itself is the boundary; on the host network
  there is no such boundary, so the default is the private one. Reaching the
  event dispatcher from off-host is then a deliberate act:
  ``services.event_dispatcher.bind``.
* Addresses OSPREY writes between services become ``localhost:<port>`` instead
  of compose service names — the dispatcher's target for its workers, and the
  Google Chat and Nextcloud bridges' URLs for the dispatch pair.

Services that talk to each other have to be on the same side of that boundary,
and ``osprey build`` refuses to render a deployment where they are not: a
co-deployed bridge on the compose network with a host-mode dispatch pair, the
reverse of that, or any address naming a service across the boundary. The build
also refuses a service that declares ``network: host`` whose rendered compose
file does not carry it, since the setting would otherwise be quietly inert.
Every one of those failures names the service and the key to change.

Running more than one project on one host
-----------------------------------------

Two OSPREY projects on the same machine compete for the same host ports, and
compose's own report of that is a bare "address already in use" partway through
starting. ``osprey up`` therefore checks every host port this deployment needs
before it touches a container — ports two of its own services would both
publish, and ports something else is already listening on — and stops if any is
taken. Every conflict is listed with the config key that moves it
(``services.postgresql.port_host``, ``dispatch.worker_port_base``, and so on).
A listener that belongs to this project's own containers is not a conflict, so
restarting a running stack stays quiet.

Host-mode services are part of that check even though they publish no ports:
their host bindings are worked out from the rendered configuration instead of
read out of a ``ports:`` block. That covers the case most likely to catch you
out — two projects whose dispatch pairs are both on the host network take the
same default ports (``10010`` for the dispatcher, ``10011`` upward for the
workers), and no ``ports:`` line anywhere would have shown it. Give the second
project its own ``deployment.port_base`` (:ref:`reference-ports`), which moves
the pair along with everything else.
A facility service you place on the host network is covered the same way,
read from its ``services.<name>.port`` key; one without that key cannot be
checked, and ``osprey up`` says so rather than skipping it silently.
