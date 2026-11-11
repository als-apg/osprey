.. _reference-facility-kinds:

========================
Facility Kinds Contract
========================

A *measurement kind* is a measurement a model's measurement file lists under
``kinds:``. OSPREY ships some; a deployment registers its own from an
installable package, with no OSPREY change. This page is what such a package
declares, what the build judges, what it writes for the runtime, and the
facility words a kind leans on: signal roles, engine record words and
scenario stops. Writing the measurement file itself is in
:doc:`/how-to/describe-your-facility`; running a measurement is in
:doc:`/how-to/measure-with-pyaml`; registering a kind step by step is in
:doc:`/how-to/register-a-measurement-kind`.

The worked example throughout is the deployment package under
``tests/facility/fixtures/fixture_kind/``: the kind ``fixture_kind`` and the
engine ``fixture_ring``, proven by ``tests/facility/test_fixture_kind_deployment.py``.

Registering a kind
==================

A kind is one :class:`osprey.facility.kinds.KindDeclaration`, named by an entry
point of the group ``osprey.facility.kinds`` in its package's
``pyproject.toml``. The entry point's name is the kind's name and loads to the
declaration (or a mapping of its fields). The kinds the ``osprey-framework``
distribution declares are the *built-in* kinds: ``orm``, ``dispersion``,
``trm``, ``crm`` and ``chromaticity_monitor``.

The package must be installed in two places: the venv ``osprey build`` runs
in, and the project image. The deployment names the package's source in its
profile's ``dependencies`` (:doc:`/reference/configuration/profile`); the build
turns a local path into a wheel the project image installs. A kind is read by
``osprey build`` alone; the approval hook, the MCP servers and the measure
script read what the build wrote and import neither the registry nor the
kind's package. A kind's package imports OSPREY only through
``osprey.facility.kinds``.

The build stops on:

- a measurement file listing a kind no installed package registers;
- a kind whose package the project image does not install;
- two distributions declaring one name (a built-in name included);
- an entry point that loads anything but a declaration or a mapping of its
  fields, a mapping naming a field the declaration does not have, or a
  declaration whose name differs from its entry point's;
- a declaration value the build cannot use (below).

The declaration
===============

Every field is required except ``member_sorts``, ``facts_writer``,
``build_stops`` and ``settings_from_measurement``.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - What it states
   * - ``name``
     - The word the measurement file lists.
   * - ``tool``
     - The pyAML tool the kind's view section configures, as
       ``KindTool(type, name)``, or ``None`` for a kind that writes no tool
       section.
   * - ``members``
     - Each group or instrument role the kind uses, with its requirement:
       ``wired`` (named by the file and wired by the model), ``exists`` (named
       and present in the facility) or ``optional``.
   * - ``member_sorts``
     - For each member the pyAML view does not ship, one of ``monitors``,
       ``setpoints`` or ``instrument``; it decides whether the file names the
       role under ``groups`` or ``instruments``.
   * - ``stepped_roles``
     - The roles whose setpoints a measurement steps; none, and the kind is
       never measured.
   * - ``tool_keys``
     - The measurement-file keys the kind's tool takes; the file must state
       each.
   * - ``single_pass``
     - Whether the kind runs on a model solved in a single pass.
   * - ``matrix``
     - Whether the result is a response matrix.
   * - ``takes_by``
     - Whether the measure tool takes ``by``.
   * - ``corrector_roles``
     - The roles whose setpoints ``correctors`` may name; empty, no
       ``correctors``.
   * - ``view_files``
     - Each file the kind adds to the model's pyAML view, by name, with its
       writer ``(ViewContext) -> str | None``; ``None`` writes no file.
   * - ``measure_plan``
     - What the measure script does for the kind, as data;
       ``osprey.facility.kinds.plan_problems`` states the vocabulary.
   * - ``description``
     - The one line the agent is told about the kind.
   * - ``measurement_schema``
     - The pydantic model the kind's sub-block ``kind_blocks.<kind>`` is
       validated by, or ``None``.
   * - ``view_section``
     - The writer of the kind's pyAML tool section:
       ``(ViewContext) -> ViewSection(fields, devices)``.
   * - ``facts_writer``
     - ``(facility file, sub-block) -> mapping``: the facts the kind keeps
       for one model.
   * - ``build_stops``
     - ``(facility file, sub-block) -> [KindStop(detail, remedy)]``: what the
       kind refuses beyond its schema.
   * - ``settings_from_measurement``
     - ``(model entry, measurement record) -> {<engine>: {<setting>: value}}``:
       the engine settings the kind's measurement needs.

Registration refuses a requirement outside the three words, a stepped,
corrector or sorted role that is not a member, an unshipped member left
unsorted, a sort
outside the three words, and a measure plan ``plan_problems`` finds fault
with.

Member roles
============

The pyAML view ships these roles, each with its sort:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Role
     - Sort
     - Named under
   * - ``bpm``
     - ``monitors``
     - ``groups``
   * - ``hcor``
     - ``setpoints``
     - ``groups``
   * - ``vcor``
     - ``setpoints``
     - ``groups``
   * - ``quad``
     - ``setpoints``
     - ``groups``
   * - ``sext``
     - ``setpoints``
     - ``groups``
   * - ``tune``
     - ``instrument``
     - ``instruments``
   * - ``rf``
     - ``instrument``
     - ``instruments``

A kind sorts every other member it declares through ``member_sorts``; an
``instrument`` is named under ``instruments`` and the other two under
``groups``. ``groups.<role>`` and ``instruments.<role>`` accept the shipped
roles, whether or not a listed kind uses them, plus every role a listed kind
declares; any other role stops the build. A declaration that re-sorts a
shipped role, or sorts a role differently from another registered kind, is
refused at registration.

The kind's sub-block
====================

What is a kind's own (a trigger, settings, a chain of steps) goes in the
measurement file's ``kind_blocks.<kind>``. A kind with a
``measurement_schema`` takes any block the schema accepts, an absent block
read as ``{}``; a kind without one takes no block. Only an admitted block is
handed, with the facility file, to ``build_stops``; each ``KindStop`` stops
the build with its detail and remedy.

A step that fires a command names a channel the facility marks ``command``
(below); a kind whose block names commands refuses any other address in
``build_stops``, as ``fixture_kind`` does.

View files and the settings transform
=====================================

Each ``view_files`` writer is handed the model's ``ViewContext`` (the kinds
listed, each group role's array name, the measurement file, the device names
and the design optics) and returns the file's text; the files it wrote are
named in the model's kind block.

``settings_from_measurement`` is handed copies of the model's entry and its
measurement record, and runs only on a block the kind admits. It may patch
only the model's own engine. The patched settings reach the facility file and
the simulator view; the engine judges them as it judges hand-written ones.

What the build writes
=====================

The facts document (``data/facility_facts.json``) carries:

- ``measurement_kinds``: one row per registered kind (``members``,
  ``stepped_roles``, ``single_pass``, ``matrix``, ``takes_by``,
  ``corrector_roles``, ``description``);
- ``measurement_models.<model>``: for each pyAML view, ``kinds``, ``names``
  (role to the view's name: a group role's array, the RF plant, the tune
  monitor), ``instruments`` (role to address) and ``rf`` (the setpoints the
  RF plants step).

Everything about a kind on a model is in that model's kind block,
``measurement_models.<model>.kinds.<kind>``: ``members`` (role to
``{requirement, addresses}``), ``stepped_roles``, ``tool_keys``, ``tool``,
``single_pass``, ``matrix``, ``takes_by``, ``corrector_roles``,
``view_files``, ``measure_plan`` and ``description``, plus:

- ``block``: the kind's ``kind_blocks.<kind>`` as the measurement file states
  it (absent: ``{}``), present only for a kind that declares a
  ``measurement_schema``;
- ``facts``: what the kind's ``facts_writer`` wrote for that model, present
  only for a kind that declares one.

Two models may give one kind different blocks, and so different facts. An
application reads both with ``osprey.apps.sdk.read_facts``.

Signal roles, engine words and scenarios
========================================

**Signal roles.** A channel's ``signal`` is a role the vocabulary lists or one
the facility adds in ``signals.yaml`` (``{name, aliases?, description?}``); a
role the vocabulary already has is refused.

**Engine record words.** An engine's ``describe(record)`` states
``engine_role``, ``plane``, ``refresh`` and optional ``marks`` for each wiring
record, in the facility schema's closed enums: ``engine_role`` is
``setpoint``, ``readback``, ``monitor`` or ``output``; ``refresh`` is
``pass``, ``periodic``, ``fast`` or ``event``; the one mark is ``command``.
An engine declares no words of its own; a word the schema does not list stops
the build. A channel is a command (``command: true``) when it is a setpoint
whose signal is a command word, or a channel some model's engine marks
``command``.

**Scenario stops.** An engine's optional ``scenario_stops(wiring, faults)``
refuses a scenario fault it cannot honour, as ``(fault, detail, remedy)``;
``fixture_ring`` refuses a fault on a record it fires, since a scenario holds
no command. The engine contract is in :doc:`/contributing/extending-osprey`.

Stability
=========

Every name in ``osprey.facility.kinds.__all__`` keeps its name, its import
path and its meaning across releases, as do the measurement-file and facts
keys this page names; a change to one is a breaking change of the kind
contract, announced in the changelog.
