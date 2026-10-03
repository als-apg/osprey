.. _how-to-import-mml-export:

==============================
Import a Middle Layer export
==============================

``osprey facility import mml`` reads a MATLAB Middle Layer (MML) export and
writes it as sources of your facility description, under
``data/facility/imported/mml/``. ``osprey build`` merges those sources with
everything else under ``data/facility/``; the import itself builds nothing.

The ``osprey mml`` verbs described in :doc:`/how-to/use-channel-finder` still
work and still write ``data/mml/``. Run this import first wherever a recipe
uses both.

Export from MATLAB
==================

The exporter ships with OSPREY. Print it, copy it to the MATLAB host and run it
once per sub-machine:

.. code-block:: bash

   osprey facility import mml --print-exporter > mml_export.m

Each run writes six files that share a stem. The import is given the
``<stem>.ao.json`` and reads the other five from beside it.

Run the import
==============

.. code-block:: bash

   osprey facility import mml mymachine.storagering.ao.json mymachine.booster.ao.json

The first run finds no mapping, writes a draft to
``data/facility/imported/mml/mapping.yaml`` and stops with
``import mml: mapping-draft``. Nothing else is written. Review the draft, then
run the same command again.

The import takes every export the mapping names: a run given a subset is
refused by the mapping check, which names each system no given export carries.

Review the mapping
==================

The mapping holds every decision the export does not state. The draft fills in
what the export does say and leaves ``null`` in each slot only you can decide.
Every ``null`` must be replaced: the import stops with
``import mml: mapping-undecided`` while one remains.

``families.<raw>.devices`` says which device each slot of a family is. It takes
one of four forms:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value
     - Meaning
   * - ``names``
     - The export's ``CommonNames``, by position. This is the default when the
       key is absent.
   * - ``address``
     - The device segment of each slot's address.
   * - a list of local names
     - One name per device, in the family's order.
   * - ``{same_as: <family>}``
     - The same devices as the named family, slot by slot.

.. code-block:: yaml

   families:
     HCM:
       devices: names
     VCM:
       devices: {same_as: HCM}
     BPMx:
       devices: address
     RF:
       devices: [cavity1, cavity2]

The import stops with ``import mml: mapping-invalid`` for a list of the wrong
length, for ``names`` on a family the export leaves nameless, and for a
``same_as`` target that does not itself resolve to names or a list.

``models.<name>.wiring`` lists the families the model drives or reads, each
with the field, engine attribute and calibration that bind it to the deck.

The ``facility:`` block seeds ``data/facility/identity.yaml`` on the first
import and is then removed from the mapping.

A mapping that disagrees with the exports prints one ``<key>: <message>`` line
per problem and writes nothing. Fix each line and run the import again.

What the import writes
======================

A clean run prints one ``wrote <path>`` line per file. The files under
``data/facility/imported/mml/`` are rewritten by every import. The authored
files beside them --- ``limits.yaml``, ``seeds.yaml``, ``identity.yaml``,
``classes.yaml`` and ``measurement/<model>.yaml`` --- are created only when
absent and never rewritten, so your edits to them survive a re-import.

The import refuses to run while ``data/facility/`` holds a record source of
your own that would merge against the imported records. It prints
``import mml: authored-present: <n> files`` and one ``rm <path>`` line per
file. Corrections belong in ``fixes.yaml``, which is never in the way.

Check the result
================

.. code-block:: bash

   osprey facility validate

``validate`` runs every check ``osprey build`` makes. For each model whose
export carried a response matrix it also compares that matrix with the one the
imported model computes, and prints one ``response check <model>: …`` line. A
failing check exits 1.

.. seealso::

   :ref:`cli-osprey-facility`
      Every exit and message of ``osprey facility validate`` and
      ``osprey facility import mml``.
