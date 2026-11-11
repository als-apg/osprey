.. _how-to-register-a-measurement-kind:

============================
Register a measurement kind
============================

A measurement kind is a word a model's measurement file lists under
``kinds:`` (:doc:`/how-to/describe-your-facility`). OSPREY ships five:
``orm``, ``dispersion``, ``trm``, ``crm`` and ``chromaticity_monitor``
(:doc:`/how-to/measure-with-pyaml`). A deployment adds its own from an
installable package, through the same registry the five go through, with no
OSPREY change. The contract, with every declaration field, the member roles
and what the build writes, is :doc:`/reference/contracts/facility-kinds`.

The worked example is the package ``fixture-kind-deployment`` under
``tests/facility/fixtures/fixture_kind/``, which
``tests/facility/test_fixture_kind_deployment.py`` builds end to end: the kind
``fixture_kind`` and the engine ``fixture_ring`` it runs on.

1. Declare the kind
===================

A kind is one ``KindDeclaration`` from ``osprey.facility.kinds``, the only
OSPREY module the package imports:

.. literalinclude:: ../../../tests/facility/fixtures/fixture_kind/fixture_kind_deployment/__init__.py
   :language: python
   :start-at: FIXTURE_KIND = KindDeclaration(

- ``members`` names each group or instrument role the kind uses, with its
  requirement: ``wired`` (named by the measurement file and wired by the
  model), ``exists`` (named and a channel of the facility) or ``optional``.
  ``member_sorts`` says what a member the pyAML view does not ship is:
  ``monitors`` or ``instrument`` here; it is optional.
- ``stepped_roles`` are the roles whose setpoints a measurement steps;
  ``corrector_roles`` the roles ``correctors`` may name; ``tool_keys`` the
  measurement-file keys the kind's tool takes (``corrector_delta``).
- ``single_pass``, ``matrix`` and ``takes_by`` set where the kind runs and the
  measure tool's arguments; ``measure_plan`` is what the measure script does,
  as data; ``description`` is the line the agent is told.
- ``view_section`` and ``view_files`` write the kind's part of each model's
  pyAML view.
- ``measurement_schema``, ``facts_writer``, ``build_stops`` and
  ``settings_from_measurement`` are the kind's own block, its facts, its build
  stops and the engine settings it needs (below). The last three are optional.

A field not on the declaration is refused, so the package names no hook OSPREY
does not offer. The full field table is *The declaration* in
:doc:`/reference/contracts/facility-kinds`.

2. Register it
==============

The package names the declaration under the entry-point group
``osprey.facility.kinds``, and an engine of its own, when it has one, under
``osprey.simulation.engines``:

.. literalinclude:: ../../../tests/facility/fixtures/fixture_kind/fixture_kind_deployment-0.1.dist-info/entry_points.txt
   :language: ini

In the package's ``pyproject.toml`` these are the
``[project.entry-points."osprey.facility.kinds"]`` and
``[project.entry-points."osprey.simulation.engines"]`` tables.

3. Install it in both places
============================

Only ``osprey build`` reads the declaration; the runtime reads what the build
wrote. The build still refuses a kind whose package the profile's
``dependencies`` do not carry into the project image, so the package is
installed twice:

- **Where** ``osprey build`` **runs.** The build reads every declaration from
  the installed entry points; a package missing there leaves the kind
  unregistered.
- **In the project image.** List the package under the profile's
  ``dependencies``, as an index requirement or a local path (the example keeps
  its source at ``./kinds/fixture_kind`` in the deployment repository). The
  build turns a local path into a wheel in the same deployment-package install
  an application's package gets (:doc:`/how-to/apps/build-your-own-app`), and
  the image installs it. Never list it in the dev-only local-requirements
  manifest: the build refuses a kind whose package the image does not install.

4. Give the kind its block
==========================

A kind takes settings of its own from the measurement file, under
``kind_blocks.<kind>``, validated by its ``measurement_schema``, a pydantic
model:

.. literalinclude:: ../../../tests/facility/fixtures/fixture_kind/fixture_kind_deployment/__init__.py
   :language: python
   :pyobject: FixtureBlock

The example's ``measurement/SR.yaml`` lists ``kinds: [trm, fixture_kind]``,
names ``groups.series_monitors`` and the instruments, states
``corrector_delta``, and holds ``kind_blocks.fixture_kind`` with a
``trigger_sample``, two ``settings`` and three ``steps``. A kind with no schema
takes no block.

Only a block the schema admits reaches the kind's ``build_stops``,
``facts_writer`` and ``settings_from_measurement``. The example's transform
asks its engine for enough turns past the trigger sample; the patch lands in
the model's ``settings`` in the facility file and the simulator's view.

5. Command steps
================

A step that fires a command must name a channel the facility marks
``command``. The facility marks it, or the model's engine does when it
describes the record, never the kind: the example's engine marks the record it
fires, and the kind's ``build_stops`` refuses any other address:

.. literalinclude:: ../../../tests/facility/fixtures/fixture_kind/fixture_kind_deployment/__init__.py
   :language: python
   :pyobject: build_stops

The measurement's approval prompt lists only what ``pyaml_measure`` steps: the
addresses of each stepped role. A chain step's command is fired by the
application's own write tool through ``osprey.apps.sdk.command_write``, so the
channels it fires are shown on that tool's approval card, which the
application's ``tools.approval`` entry describes
(:ref:`how-to-apps-writes`). A command write checks no approval itself; the
call that leads to it is its caller's to check
(:ref:`reference-guarded-run`, *Command write*).

6. Where the kind's facts live
==============================

The build writes the kind's ``facts_writer`` output as ``facts`` in each
model's kind block, ``measurement_models.<model>.kinds.<kind>``, beside the
``block`` it was written from. An application reads it with
``osprey.apps.sdk.read_facts``. Two models may give one kind different blocks,
so each has its own facts. The rest of the kind block is *What the build
writes* in :doc:`/reference/contracts/facility-kinds`.

What a refusal means
====================

Each stops ``osprey build``; the build prints the line and its ``fix:``.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - The build says
     - Meaning
   * - ``kind fixture_kind is not registered under osprey.facility.kinds by any
       installed package``
     - A measurement file lists the kind, and no package installed where
       ``osprey build`` runs declares it.
   * - ``measurement kind 'fixture_kind' (package fixture-kind-deployment): the
       kind's package is not among the packages the project image installs``
     - The package is installed where the build runs but not in the profile's
       ``dependencies``, so the image would not carry it.
   * - ``measurement kind orm — built-in declares a kind of the same name``
     - The package's kind takes a name OSPREY ships.
   * - ``measurement kind fixture_kind — package … declares a kind of the same
       name``
     - Two installed packages declare one name.
   * - ``measurement kind hooked_kind — it names hook 'on_trigger', which OSPREY
       does not offer``
     - The declaration states a field ``KindDeclaration`` does not have.
   * - ``kind <kind> needs `<slot>`, which the file does not name``
     - A required member is missing from the measurement file.
   * - ``kind_blocks.<kind>.<field>: <message>``
     - The ``measurement_schema`` refuses the block.
   * - ``… is the block of a kind the file does not list``
     - The file holds a block for a kind missing from its ``kinds``.
   * - ``kind fixture_kind: step clean fires CLEAN:EN, which the facility does
       not mark command``
     - The kind's ``build_stops`` refused a command step on an unmarked
       channel.
   * - ``kind <kind>'s facts writer fails on this file's block``
     - The ``facts_writer`` raised on a block the schema admits.
