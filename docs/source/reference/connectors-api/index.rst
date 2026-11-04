.. _connectors-api:

=====================
osprey-connectors API
=====================

The Python API of ``osprey-connectors``, the package that carries OSPREY's
control-system and archiver connectors and installs without the rest of the
framework (``pip install osprey-connectors``). How to choose and configure a
connector is in :doc:`/how-to/control-systems/use-connectors`; what every
connector must do is in :doc:`/reference/contracts/connectors`.

Every public module lists its public names in ``__all__``, and these pages show
exactly those names. A module whose name starts with an underscore is internal
and can change in any release; the concrete connectors are among them, and a
caller builds one through :class:`~osprey_connectors.factory.ConnectorFactory`
from its registered type name. Three subpackages --
:mod:`osprey_connectors.control_system`, :mod:`osprey_connectors.archiver` and
:mod:`osprey_connectors.ipc` -- gather their implementation behind one
namespace, and their names are imported from the subpackage itself.

How the package versions, and what each release promises to keep compatible,
is in the
`Stability <https://github.com/als-apg/osprey/blob/main/packages/osprey-connectors/README.md#stability>`_
section of its README. These pages are built from the same tree as the rest of
this site, so they can describe a connectors release newer or older than the
one you have installed; the package's
`changelog <https://github.com/als-apg/osprey/blob/main/packages/osprey-connectors/CHANGELOG.md>`_
says which release added what.

.. toctree::
   :maxdepth: 1

   contracts
   building
   errors-logging
   configuration
   write-safety
   deployment
   supervision
   simulation
   utilities
