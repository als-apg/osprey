.. _how-to-run-scenarios:

=======================
Run and write scenarios
=======================

A scenario is one way the simulated machine departs from its baseline: values
written into its models, faults in a physics model, the history its archive
shows, and the logbook entries that narrate it. Scenarios are authored under
``data/facility/scenarios/``, one YAML file each, and ``osprey build`` serves
them from the simulator view under ``build/data/simulator/``. The
``osprey sim`` commands list, apply and report on them; their options are in
:ref:`cli-osprey-sim`.

The scenarios the demo ships
============================

The control-assistant preset ships six scenarios in
``data/facility/scenarios/``:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Scenario
     - What it shows
   * - ``nominal``
     - All systems nominal. It carries the background logbook narrative of
       the weeks before the demo and is always in the active set.
   * - ``rf-thermal``
     - Three thermal excursions on cavity 1 in the week before an
       investigation, each a temperature rise, a reflected-power spike and a
       forward-power trip, the last one a beam dump. The archive shows the
       excursions, the logbook narrates the dump, the investigation and the
       cooling repair, and cavity 1 reads 26.5 degC live. Cavity 2 shows one
       minor excursion for contrast.
   * - ``rf-thermal-live``
     - Live thermal wander on cavity 1: one slow shared driver moves its
       temperature, tuner, reflected and forward power together, each with
       its own noise, for correlation plots. It adds no archived history and
       does not compose with ``rf-thermal``, which writes the same channels.
   * - ``vacuum-burst``
     - A vacuum burst in one sector with a correlated beam-current drop at
       14:32:08 local time, recurring daily in the archive.
   * - ``bpm-polarity``
     - BPM 17 reports an inverted reading in both planes. No single channel
       is out of range; the fault shows only as an orbit correction that does
       not converge. The physics fault applies at the virtual accelerator's
       next boot.
   * - ``orm-dual-fault``
     - The BPM 17 inversion and corrector HCM01 at half its nominal gain at
       once, both visible only through an orbit-response measurement. The
       physics faults apply at the virtual accelerator's next boot.

The preset starts a fresh deployment in ``rf-thermal``, as the profile key
``simulation.default_scenarios`` names (see
:doc:`/reference/configuration/config`).

Apply a scenario
================

.. code-block:: bash

   osprey sim list
   osprey sim apply vacuum-burst
   osprey sim status

``osprey sim list`` marks the active set with ``*`` and says which scenarios
carry a logbook narrative. ``osprey sim apply`` replaces the active set with
the names it is given, plus ``nominal``, then purges and reseeds the ARIEL
logbook and rewrites the affected windows of the stored archive, so live
values, history and narrative tell one story. It asks before the purge and
the rewrite; ``--yes`` skips the prompts and ``--no-seed`` leaves both stores
untouched.

Several scenarios apply together only when no two of them write one target,
a channel or a model's fault. A set that breaks the rule is refused before
anything is written, and the refusal names the target.

Write a scenario
================

Add ``data/facility/scenarios/<name>.yaml``; the file name is the scenario
name. Every key is optional:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Key
     - Meaning
   * - ``description``
     - What the scenario shows. ``osprey sim list`` prints it.
   * - ``overrides``
     - ``{<address>: <value>}``, the value each channel reads live while the
       scenario is active.
   * - ``faults``
     - ``{<model>: {<address or engine variable>: <value>}}``, faults written
       into a physics model; a value is a scalar, the word ``stuck``, or a map
       ``{<fault field>: <value>}`` such as ``{polarity: -1}``.
   * - ``archiver``
     - ``[{channel, events}]``, the history each channel's archive shows.
       Event shapes and positions are those of
       :ref:`simulation-bundle-events`; ``at_when`` (``{days_ago, time}``)
       places an event at the instant a logbook entry's ``when`` names.
   * - ``logbook``
     - The entries the scenario narrates, each with ``entry_id``, ``when``
       (``{days_ago, time}``), ``author``, ``title`` and ``text``, and
       optionally ``tags``, ``categories``, ``loto_tag``, ``extra`` and
       ``attachments``.
   * - ``drivers``, ``couple``, ``noise``
     - Slow shared signals, the channels that follow them, and per-channel
       noise while the scenario is active.

An entry's ``attachments`` lists its pictures, each ``{path: <picture file>}``
or ``{plot: <plot spec .json>}``, relative to the folder
``data/facility/scenarios/<name>/`` beside the file. Run ``osprey build``, or
``osprey facility validate`` to check the sources without writing anything;
an address, model or key the facility does not have stops the build and names
the file. The new scenario then appears in ``osprey sim list``.
