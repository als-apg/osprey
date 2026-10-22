.. _reference-partial-answers:

===============
Partial Answers
===============

A tool asked for several things -- channels to read, entries to fetch -- may
answer some of them and not others. Every such tool says how much it answered
in one shape, so the agent never has to work it out from prose or by comparing
lists.

The rule
========

A tool that answers part of a request returns the answered part plus a map of
what is missing, each entry carrying the source's own text. When nothing
answered, the tool refuses with the same map. A request the tool refuses as a
whole is a different error, with a different remedy.

The map
=======

``missing`` maps each thing asked for and not answered to the source's own
text, in request order:

.. code-block:: json

   {"missing": {"SR:BPM1:X": "TimeoutError: no reply in 5 s", "e2": null}}

- The key is what the request named: the channel address, the entry id.
- The value is the text the source gave for that item -- for a channel read,
  ``"<ExceptionType>: <message>"``. It is ``null`` when the source gave no
  text. A tool never makes a reason up.

Three arms
==========

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - What happened
     - What the tool returns
   * - Some or all answered
     - The result, with ``missing`` beside the answered part. ``missing`` is
       always present; ``{}`` means everything answered.
   * - Every item tried, none answered
     - An error whose ``details.missing`` holds the same map. When every item
       failed with the same error the controls server answers on its own --
       ``not_found``, ``channel_unreachable``, ``timeout_error``,
       ``connection_error`` -- that error is returned, with its own message
       and remedy and ``details.missing`` added. Otherwise the error type is
       ``nothing_answered``. When the supervisor that owns the connector
       dropped it, replaced its connector-host child, or kept a child that
       still answers, because of one of those failures, the error lists each
       action once under ``details.recovery``, as an ``action`` word and its
       sentence, and repeats the sentence in the message. A
       ``channel_unreachable`` failure is that channel's alone and carries no
       ``details.recovery``. A write batch in which no value
       reached any channel is refused the same way, with its rows under
       ``details.results``: as ``write_refused`` when every write was
       refused, and as ``nothing_answered`` when any was sent and not taken.
   * - A request refused before any item was tried
     - The refusal's own error type (``validation_error``, ``write_refused``,
       ...), with no map: nothing was tried, so nothing is missing.

``nothing_answered`` and a whole-request refusal ask different things of the
agent. A refusal names a request to change; ``nothing_answered`` names items
that were tried and did not answer, which the agent reports to the operator
entry by entry.

The tools
=========

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Tool
     - A key in ``missing`` names
     - Its text
   * - ``channel_read``
     - A channel address
     - The connector's error, as ``"<ExceptionType>: <message>"``
   * - ``entries_by_ids``
     - An entry id
     - ``null``: the logbook does not hold the id and gives no text
   * - ``model_response_matrix``
     - A corrector setpoint address
     - The engine's text
   * - ``pyaml_measure``
     - A stepped setpoint address, or the measurement kind when none is left
     - The first line of what stopped the run; ``null`` when nothing raised

Two exceptions
==============

``channel_write``
   A write answers with one row per channel, and each row's ``outcome`` word
   says what happened to that write (see :doc:`connectors`). A write's map is
   its rows, and a refusal carries them under ``details.results``. It has no
   ``missing`` key: a written but unconfirmed channel was answered, and a
   second map would state the refused and failed rows twice.

``archiver_read``
   A channel with no points in the requested window is answered, not missing:
   the data file has its column, and the ``coverage`` block says from the
   archive's own facts why the window is empty.
