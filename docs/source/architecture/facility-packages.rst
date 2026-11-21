.. _architecture-facility-packages:

Facility Packages
=================

The facility code leaves the framework as two distributions. **osprey-facility**
holds the description and its build: it reads the sources under
``data/facility/`` and writes ``facility.json``. **osprey-panels** is the panel
engine: it reads ``facility.json`` and writes Canopy screens. The framework
keeps the views, the settings and the command line, and reaches the
description through its public modules only.

Today all of this is one tree, ``src/osprey/facility``, with an import cycle
through the views, the scenarios and the renderer. This page states the end
state, why each part of it is the way it is, which phase of the split realises
it, and how a check holds the code to it while the split is under way. What the
build writes is in :doc:`facility-file`; the version rules of the document are
in :ref:`reference-facility-json`.

.. raw:: html
   :file: ../_diagrams/facility-packages.html

The decisions
=============

Each decision states the end state, the reason, the phase that realises it and
what holds today.

1. The panel engine reads the document only
-------------------------------------------

The engine reads ``facility.json`` at a declared, checked version and imports no
facility code. The build also writes into the document the two things the engine
derives from the vocabulary today: the effective signal-role table (nameplate
and title) and the relation types.

*Why:* a fork of the engine tracks one versioned document, not the description's
Python API, and importing the engine no longer loads click, the build stack and
``osprey.simulation``.

*Phase:* 2, done. *Today:* the engine reads ``facility.json`` at ``/4`` and
imports no description module; the build writes the ``signals`` and
``relation_types`` tables.

2. ``facility.json`` is a versioned document with a published schema
--------------------------------------------------------------------

The build is the only producer. The header is ``osprey.facility.facility/N``; N
goes up when a key is removed, renamed, retyped or changes meaning, and stays
when a key is added. Readers ignore keys they do not know, check N against the
versions they declare, and refuse any other with "rebuild with osprey build". A
test pins the document's generated JSON Schema, so a shape change fails until
its author bumps N or marks the change additive.

*Why:* the document is the description's product. A fork or a non-Python reader
needs a contract it can check by machine, and one number is the only thing a
reader has to compare.

*Phase:* the panel engine checks the version in phase 2, the framework readers
in phase 3 with the write guard first, and the schema ships in the package in
phase 4. *Today:* the build writes ``/4``, a test pins the schema against the
floor of each ``N``, and the panel engine refuses any other version; the
framework readers do not compare it yet. The exact rules and the reader table
are in :ref:`reference-facility-json`.

3. The description and the framework split as the code reads
------------------------------------------------------------

osprey-facility gets the records, relations, combine, sources, compute, build,
wiring and validate (without its command-line reporting), the serialisation half
of the renderer, the scenario rosters and checks, the scenario session model
(today ``osprey.simulation.sessions``), the layers, the schema and vocabulary,
the kinds with the five built-in kinds, the pure helpers now in the simulator
and pyAML views, and its errors as plain exceptions without click.

The framework keeps the views, the settings, the bundled facility, the step that
renders the facility outputs, the check of scenario sessions against MCP server
names, the click adapter for the errors and the command line. osprey-panels gets
the panel engine.

osprey-facility depends on osprey-connectors, pydantic, PyYAML, packaging and
numpy, with scipy and accelerator-toolbox in an ``mml`` extra. It does not
depend on pyaml-cs-osprey: the one kind name it borrows from there becomes its
own. osprey-panels needs the standard library and jsonschema; it writes its
files atomically itself.

*Why:* this is where the imports already fall once the cycle is cut.
pyaml-cs-osprey imports the framework's runtime, so a dependency on it would
bring the framework back into the description.

*Phase:* 3 untangles the description from the framework, 4 makes it a package,
5 makes the panel engine a package. *Today:* one package; the description
imports the framework in the records, the scenarios, the engine contract and
the relations layer, and imports click in nine modules.

4. The simulation-engine contract lives in osprey-connectors
------------------------------------------------------------

The engine Protocol, its optional names, the entry-point group name and the
calibration math move to ``osprey_connectors.simulation.engines``.

*Why:* the description calls the engines, the framework implements them and the
virtual accelerator loads them. osprey-connectors is the one layer all three
depend on, so the contract is spelled once. The cost is a larger public API in
osprey-connectors and its compatibility table.

*Phase:* 3. *Today:* the contract is framework code, and the group name has
three spellings.

5. Names, public modules and the command line
---------------------------------------------

The distributions are ``osprey-facility`` (import ``osprey_facility``) and
``osprey-panels`` (import ``osprey_panels``). Each declares its public modules;
the rest is private, and a contract enforces it the way the connectors' public
API is enforced. osprey-panels ships its own command,
``osprey-panels write|check|classes``, and the framework's
``osprey facility panels`` goes away. Entry-point groups take the new roots, such
as ``osprey_facility.kinds``.

*Why:* top-level names keep the import contract enforceable. Namespace packages
under ``osprey`` would share the root, import-linter skips modules that overlap,
and which distribution owns a module would blur. A framework command over the
engine would mean the framework imports it, and a fork would have to be wired
into the framework's command line too.

*Phase:* 4 for osprey-facility, 5 for osprey-panels. *Today:* the framework
reaches no description module outside the declared public set, and the
framework's command line imports the panel engine.

6. Both packages have their own tag streams
-------------------------------------------

Releases are tagged ``osprey-facility-v*`` and ``osprey-panels-v*``, built by
one release workflow that takes the package as a parameter. The framework
depends on osprey-facility with a floor. osprey-panels declares the
``facility.json`` versions and the Screen-IR version it supports, not an
osprey-facility floor.

*Why:* each package gets its own version and cadence. One parameterised workflow
generalises the connectors' release stream instead of copying it twice.

*Phase:* 6. *Today:* both are released with the framework.

7. A fork is a separate distribution that replaces osprey-panels
----------------------------------------------------------------

A fork reads ``facility.json`` at a declared version, the facility's ``panels/``
data and a theme file the caller passes in, and emits Screen-IR and
``index.json`` at declared versions. Layout, symbols and the describe step are
the fork's own code. The engine carries fallback state colours and has no path
into the framework. There are no plug-in points inside the engine.

*Why:* the inputs and outputs are files at declared versions, which is all a
replacement needs. Plug-in points would be an abstraction with no second user.

*Phase:* 5. *Today:* the engine finds the framework's theme by climbing from its
own ``__file__``.

8. osprey-panels owns its Screen-IR pin
---------------------------------------

The engine vendors the Screen-IR schema of the version it emits, and a sync test
compares that file with the pinned Canopy commit. ``write`` and ``check``
validate every screen against it at run time. The version is one named
constant. The framework's Canopy pin keeps pinning the runtime stack and names
the engine's Screen-IR version.

*Why:* an engine release, a fork or a facility edit cannot emit a screen that
fails only once it reaches Canopy.

*Phase:* 5. *Today:* the version is an inline literal, and screens are
validated only by a framework test.

9. Tests live inside each package
---------------------------------

Each package keeps its tests in ``packages/<package>/tests/<package>_tests/``
with its own pytest configuration, and a standalone CI job installs it without
the framework and refuses the framework if it appears. The panel tests read a
checked-in built ``facility.json``, which an osprey-facility test rebuilds from
the fixture tree and compares byte for byte. The description helpers the
framework tests use move into ``osprey_facility_tests``; the framework's build
and preset fixtures stay in the framework.

*Why:* only a test run without the framework proves a package works without it.
Test helpers in the wheel would become public API.

*Phase:* the byte-for-byte rebuild test lives in the root suite until phase 4;
the package suites start in phases 4 and 5. *Today:* every test is in the root
``tests/``; the panel tests read ``tests/facility/panels/fixture/facility.json``,
and ``tests/facility/test_panel_fixture_rebuild.py`` rebuilds it.

The boundary check
==================

Two tools hold the code to the end state, and both read one classification.

The classification is the module lists of six import-linter contracts in
``pyproject.toml``. Every module under ``src/osprey`` sits on one side:
description, panel engine, engine contract or framework. The side follows the
end state, not today's directory, so ``osprey.simulation.sessions`` is
description while the views, settings, bundled facility, ``osprey.utils.facility``
and the engine implementations are framework. The renderer, scenarios, validate
and compute are description modules whose framework imports are listed edge by
edge.

``lint-imports``, run by the ``import-contracts`` check, follows every static
import between ``osprey`` and ``osprey_connectors``. Its contracts forbid:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Contract
     - Forbids
   * - ``facility-description-standalone``
     - the description importing the framework
   * - ``facility-description-without-panels``
     - the description importing the panel engine
   * - ``panels-document-only``
     - the panel engine importing the description or the engine contract
   * - ``panels-standalone``
     - the panel engine importing the framework or ``osprey_connectors``
   * - ``framework-without-panels``
     - the framework importing the panel engine
   * - ``facility-public-api``
     - the framework importing a description module outside the public set

The boundary test, ``tests/facility/test_package_boundary.py``, reads the sides
from those lists and checks what import-linter cannot see: imports of
third-party distributions outside each package's dependency set, and dynamic
reaches (``importlib.import_module`` strings, ``importlib.resources.files`` and
paths climbed from ``__file__``). It also keeps the classification whole: a
module on no side, or on two, fails, so a new module fails until a list names
it. It runs with the invariant tests on every check; negative tests prove that
each edge class it guards fails when introduced.

The ratchet
-----------

The edges that existed when the check landed are its baseline. Each is one
``importer -> imported`` entry, in the contract's ``ignore_imports`` for a
static import or in the boundary test's third-party and dynamic ratchets, under
a ``# phase N`` header naming the phase that removes it. Entries are only
removed. A new crossing edge fails, and so does an entry whose edge is gone.

.. raw:: html
   :file: ../_diagrams/facility-packages-today.html

The headers name these phases:

- **Phase 2** removed the panel engine's imports of facility code and its read
  of the vocabulary files; its header is empty.
- **Phase 3** removes the description's imports of the framework (views helpers,
  ``osprey.utils.facility``, the MCP registry, the command line, deployment and
  config paths, the engine implementations), its click imports in nine modules,
  its import of pyaml-cs-osprey and the ``osprey.`` spelling of the engine
  contract.
- **Phase 4** removes framework reaches into private description modules: none
  by static import today, one resource read of the MML layer by the command
  line.
- **Phase 5** removes the framework command line's imports of the panel engine,
  the engine's climb to the framework theme and its atomic writes through
  ``osprey_connectors.safety_file``.
- **Phase 6** is the release wiring and removes no edge.

Refreshing the baseline
-----------------------

Main keeps moving while the split is under way, and nothing outside the split
knows the check. The baseline is therefore refreshed at one point only: when
main is merged into the split's branch, at each merge back to main. After that
merge:

1. Run ``lint-imports`` and the boundary test.
2. Classify every module main added into a side's module list.
3. Enter every new crossing edge main brought under the header of the phase that
   removes it, in the matching contract's ``ignore_imports`` for a static import
   and in the boundary test's third-party or dynamic ratchet otherwise, with the
   reason and phase.
4. Delete every entry whose edge main removed.
5. Run both tools again; they pass with no other change.

At any other time the baseline is never extended: a new edge is fixed, not
entered.

.. seealso::

   :ref:`reference-facility-json`
      The document's header, bump rule, reader rule and published schema.

   :doc:`facility-file`
      What the build writes into ``facility.json`` and the views read from it.
