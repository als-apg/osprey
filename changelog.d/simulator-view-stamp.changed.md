The build's simulator view, ``variables.json``, is now schema ``osprey.facility.simulator/3``: each
wiring record carries the ``engine_role``, ``plane`` and ``refresh`` its engine's ``describe()`` states, and
each channel carries the node it is ``on``. ``osprey_connectors.simulation.view`` is the reader of
the view; it refuses a render from an older OSPREY and asks for a rebuild with ``osprey build``.
