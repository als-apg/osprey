.. _how-to-use-connectors-standalone:

Use the Connectors Without the Framework
========================================

``osprey-connectors`` installs on its own, without the agent, the web terminal
or a built deployment. A script, a notebook or a service of your own can read a
control system and its archiver through the same connectors OSPREY uses, and
gets their error taxonomy and their write safety with them.

.. code-block:: bash

   pip install osprey-connectors

This page covers reading. A connector built in a script refuses every write
before anything is sent; `When a write is refused`_ says why and where writing
is decided.

Every script and output on this page is a file the test suite runs, so what is
shown is what the package does.

Read channels
-------------

A connector is built from a configuration dict by its type name. The dict has
the shape of a deployment's ``control_system:`` section:

.. literalinclude:: examples/read_channels.py
   :language: python
   :caption: read_channels.py

Against an IOC serving those three records, it prints:

.. literalinclude:: examples/read_channels.out
   :language: text

``create_control_system_connector`` connects before it returns. A state channel
reads as its integer index, with the state's name in ``metadata.enum_label``.
``read_multiple_channels`` leaves out a channel that failed to read rather than
raising, so check the keys of what it returns.

Every connector answers the same methods. To read a DOOCS or Tango control
system, change ``type`` and its ``connector`` block;
:doc:`use-connectors` has the block for each, and the client library the DOOCS
and Tango connectors need, which this package does not install.

Read history
------------

An archiver connector is built the same way, from the shape of a deployment's
``archiver:`` section, and returns history as a pandas ``DataFrame``:

.. literalinclude:: examples/read_history.py
   :language: python
   :caption: read_history.py

The frame is long: one row per sample, with ``timestamp`` (UTC), ``channel``
and ``value`` columns:

.. literalinclude:: examples/read_history.out
   :language: text

``get_data`` also takes ``precision_ms``, a bin width, and ``processing``, how
each bin is reduced; the archiver section of
:doc:`/reference/contracts/connectors` says what each mode returns.

Read the channel directory
--------------------------

A ChannelFinder directory is built from a ``BackendSpec`` by
``create_service``, which reads the kind from the spec; ``settings`` is the
connection block. Building it sends nothing:

.. literalinclude:: examples/read_directory.py
   :language: python
   :caption: read_directory.py

Against a directory listing those three BPM channels, it prints:

.. literalinclude:: examples/read_directory.out
   :language: text

Every answer is a value, never an exception: a search returns a page whose
``outcome`` says how it ended, and ``describe`` returns the record,
``Outcome.NOT_LISTED``, or ``Unavailable`` with the reason.

What needs a deployment
-----------------------

Three things on the connector side are a deployment's, and a script without one
does without them:

The simulator
   ``virtual_accelerator`` and ``mock_archiver`` serve the simulated machine of
   a built facility, and refuse to connect without the simulator view
   ``osprey build`` writes. Every other connector talks to a real system.

``config.yml``
   A few facts are deployment-wide and read from ``config.yml`` rather than
   from the dict a connector is built from: the facility's timezone, the write
   tier and the channel limits. A script that has none reads them at their
   defaults and logs a warning that it fell back to UTC. To set the timezone,
   name a ``config.yml`` in ``CONFIG_FILE`` that states ``system.timezone``:

   .. literalinclude:: examples/system-timezone.yml
      :language: yaml
      :caption: config.yml

   With ``CONFIG_FILE`` unset, the file is the one ``OSPREY_CONFIG`` names,
   else ``build/config.yml`` under the working directory, else ``config.yml``
   in it. A process started in a built project directory therefore reads that
   project's config with nothing set. The same file answers
   ``create_control_system_connector()`` and ``create_archiver_connector()``
   called with no dict, which build the connector its ``control_system:`` or
   ``archiver:`` section selects. Without a config they raise
   ``FileNotFoundError``, naming where they looked.

Writes
   Every write passes the verdict in :ref:`write-safety-config`: the write tier,
   the machine the connector writes to, the login it writes for, and that
   login's switch for that machine, the first refusal winning. A connector built
   in a script names no machine and has no switch to read, so every write is
   refused before anything is sent.

When a write is refused
-----------------------

The refusal is the package's ordinary one.
:meth:`~osprey_connectors.control_system.ControlSystemConnector.write_channel`
returns a result whose ``outcome`` is ``refused``;
``write_channel_checked`` raises
:class:`~osprey_connectors.errors.ChannelWriteBlockedError`, whose message ends
with the reason. Built from the same ``CONFIG`` as above:

.. literalinclude:: examples/write_refused.py
   :language: python
   :caption: write_refused.py
   :pyobject: main

prints

.. literalinclude:: examples/write_refused.out
   :language: text

``tier_never`` is the answer with no ``config.yml``; with the tier armed in one,
the next term answers ``target_unknown``. Writing is decided by a deployment, so
a program that has to write runs in one and is let write by a person: from the
switch, or, for a program with nobody at a prompt, through a bounded
:ref:`write authorization <how-to-write-from-a-program>`.

.. seealso::

   :doc:`/reference/connectors-api/index`
       Every public name of the package.

   :doc:`use-connectors`
       The configuration block of every connector.

   :doc:`/reference/contracts/connectors`
       What every connector must do: large values, write confirmation, who may
       write, and the archiver contract.
