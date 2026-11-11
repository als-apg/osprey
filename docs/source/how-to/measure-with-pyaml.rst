.. _measure-with-pyaml:

============================
Measure with the pyAML tools
============================

This page shows how to take a measurement on the control target through the
agent, and what happens to the machine while it runs. pyAML is the
accelerator library the measurement tools drive. The page also says when the
tools appear and what a measurement asks for. Every parameter and every result
field is in :ref:`executor-measurement-tools`.

A deployment whose build writes a pyAML view gives the agent two tools on the
``python`` server:

- ``pyaml_measure`` measures a response matrix or the chromaticity on the
  control target.
- ``model_response_matrix`` solves a served model's orbit response on its deck.

When the tools are there
========================

The build writes ``data/pyaml/<model>/configuration.yaml`` for each served
model that names a deck and has a measurement file under
``data/facility/measurement/`` (see :doc:`describe-your-facility`). This file
is the model's *pyAML view*.

A render with at least one such view registers three things: both tools, the
``pyaml-tools`` skill, and the prompt section that teaches them. A render with
no view registers none of the three.

The measurement file decides which kinds a model takes. OSPREY ships these,
and a deployment registers its own (:doc:`/reference/contracts/facility-kinds`):

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Kind
     - Measures
     - Steps
   * - ``orm``
     - the orbit response of the corrector setpoints
     - ``corrector_delta``
   * - ``dispersion``
     - the orbit response to the RF frequency
     - ``frequency_delta``
   * - ``trm``
     - the tune response of the quadrupoles
     - ``quad_delta``
   * - ``crm``
     - the chromaticity response of the sextupoles
     - ``sextu_delta``
   * - ``chromaticity_monitor``
     - the chromaticity
     - the RF, over ``n_step`` points

The tool refuses a kind the file does not list, and never puts such a call to
an approver:

- If the model lacks a group or instrument the kind needs, the refusal names
  it.
- If the model has every group and instrument but the file does not list the
  kind, the refusal names the kinds the file does list.

A model solved in a single pass takes only the kinds that run single pass;
of the shipped ones, ``orm`` alone.

Ask for a measurement
=====================

To measure through the agent:

#. Ask the agent for the measurement in plain words, for example "Measure the
   orbit response matrix." If more than one model has a pyAML view, name the
   model. The agent calls ``pyaml_measure``, and an approval prompt appears.
#. Check the approval prompt. It shows what the call steps; see `What a
   measurement asks`_.
#. Approve the call. The measurement steps the setpoints under the control
   target's run lock; see `What happens to the machine`_.
#. Read the result. The agent answers with the measured values and where it
   saved the data; see `The result`_.

What a measurement asks
=======================

Every ``pyaml_measure`` call for a kind the model's file lists asks for
approval. The prompt shows these:

- the model and the kind;
- the modulation;
- the pyAML view's path and digest;
- every setpoint the call steps.

A prompt lists at most 500 setpoints. A call that would step more is refused
without asking.

The call runs only on the view the build recorded and the prompt showed. A
view edited after the build, or one that differs from what the prompt showed,
is refused before any setpoint moves.

What happens to the machine
===========================

Each step starts as the measurement file's value. The tool shrinks it until no
setpoint's largest single write exceeds half its ``max_step`` in
``data/facility/limits.yaml``. A strength, ``max_step`` or write that is not a
finite number refuses the measurement.

Every write of one measurement runs under the control target's run lock:

- A second read-write run on the target is refused while the measurement runs.
- A measurement that stops part-way, fails or is interrupted leaves each
  setpoint where it stopped.
- After such a stop, the result's ``moved`` lists each setpoint whose write may
  have gone out, an unknown outcome included, with its value before the call.
  The ``pyaml-tools`` skill tells the agent to report these setpoints to the
  operator.
- A measurement that reaches the execution timeout is killed like any other
  run, with no early stop. Its failure lists only the channels it attempted.
- A pyAML write from ``execute`` is refused; measurements go through
  ``pyaml_measure``.

:ref:`reference-guarded-run` has the lock and what a stopped run leaves.

The result
==========

A completed measurement answers with three things:

- its rows and columns, named by address and device id;
- the steps it used;
- the measurement, saved to the artifact store.

The saved file is the one the run left in its own execution folder. A run that
names any other file saves nothing and says so in ``data_file_refused``.

``trm`` and ``crm`` can step *by group*: each column steps every magnet of the
role's array (every group the role names) together, by the same integrated
strength. ``crm`` by group is sized from the view's ``crm.json``. The build
writes ``crm.json`` only when every member of the ``sext`` role's array has a
modelled column. If a member has no modelled column, the view has no
``crm.json`` and refuses ``crm`` by group.

The model's answer
==================

``model_response_matrix`` solves the orbit response of a served model on its
own deck, without touching the machine and without asking for approval. Its
rows and columns carry the same addresses and device ids, so a measured matrix
and the model's line up entry by entry.
