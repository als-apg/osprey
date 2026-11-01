.. _reference-watchdog:

=================
Watchdog Contract
=================

:ref:`Safety watchdogs <architecture-safety-watchdogs>` places watchdogs among the write gates
and says what they do not cover, and :doc:`/how-to/safety-watchdogs` sets one up. This page is
the exact behaviour: the hold record and where it lives, the routes and frames of the web
terminal, and the fields of a verdict's audit entry.

The hold record
===============

One file per session, ``var/state/session_holds/<session_key>/record.json`` in the deployment
repository. ``osprey_connectors.session_hold`` owns its format, its reader and its writer. A
session key is one path segment: it starts with a letter or digit, continues with letters,
digits, ``.``, ``_`` and ``-``, and runs to 128 characters.

.. code-block:: json

   {
     "schema": 1,
     "session_key": "<key>",
     "flags": [
       {
         "flag_id": "<id>",
         "source": "guard-circumvention",
         "reason": "Asked the agent to edit hooks, settings or configuration to loosen a gate.",
         "prompt_ref": "<record_session_id>:<submission_id>",
         "prompt_excerpt": "<at most 200 characters>",
         "opened_at": "2026-10-31T00:00:00Z",
         "state": "open",
         "stop_unconfirmed": false,
         "started": []
       }
     ]
   }

A session is held while any flag's ``state`` is ``open``. A flag closes only by an answer that
names it, and stays in the record, closed, with ``answer``, ``operator`` and ``answered_at``.

The web terminal gives every session it launches the store through
``OSPREY_SESSION_HOLD_DIR``, and only when the build has an enabled watchdog. A process without
that variable or a session key is never held. **Under a store, the hold fails closed:** a
missing or unreadable record, a record of another schema or another session, a symbolic link at
the directory or the record, and an unsafe key all hold. Such a record is reported as the one
flag ``hold-record-unreadable``, with source ``session_hold``, and answering it writes a fresh
record. ``GET /api/watchdog/flags`` reports no flags only for a key with no running session and
nothing on disk: a session that has not started.

Writes hold an exclusive lock on ``record.json.lock`` and land atomically, so a reader sees the
old record or the new one and two writers never lose each other's flag.

Where the hold refuses
======================

A held session's write is refused with the reason ``session_held``, naming the watchdogs that
raised the hold where the record can be read::

   Writes are paused: this session has an unanswered hold the operator must answer. Raised by: guard-circumvention.
   [osprey-refusal: session_held]

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Site
     - What it refuses
   * - ``osprey_writes_check`` hook
     - Every write tool it judges, Bluesky lane tools included.
   * - The MCP servers' write-tool clamp
     - Every tool the render classifies as a write, except read/write tools such as
       ``execute``, which gate themselves.
   * - ``channel_write``, ``execute``, ``queue_add``, ``queue_start``, ``phoebus_drive``
     - Each asks again as it executes, so a write approved while the flag was open is refused.
       ``execute`` refuses read-write runs only.
   * - The connector
     - Every put, as a term of the write verdict
       (:ref:`the verdict's order <write-safety-config>`).
   * - The terminal's panel proxy
     - A call that would fire a dispatch job.

Stopping a plan or a run, and ``control_target_set``, are never held.

The routes
==========

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Route
     - Tier
     - Purpose
   * - ``POST /api/watchdog/prompt``
     - panel token
     - The trigger hook hands over one prompt.
   * - ``GET /api/watchdog/flags``
     - operator
     - The session's open flags.
   * - ``POST /api/watchdog/flags/answer``
     - operator
     - The operator's answer.

The last two are outside the panel tier, so an agent holding the panel token can neither read
its flags nor release its own hold.

``POST /api/watchdog/prompt``
-----------------------------

The body holds ``session_key``, ``record_session_id`` and ``submission_id``, each a UUID, and
``prompt``, up to 1,000,000 characters. The answer comes before anything is judged: 202 with
``{"accepted": true, "duplicate": false}``, with ``duplicate: true`` for a prompt already seen,
which is not judged again.

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Status
     - ``detail``
     - When
   * - 400
     - ``invalid_session_id``
     - An id is not a UUID.
   * - 404
     - ``watchdog_disabled``
     - No watchdog runs in this build.
   * - 404
     - ``unknown_session``
     - The key names no session this process runs.

On any failure the hook falls back to judging the prompt in a process of its own, with the
session key where it has one, so a refused or unreachable intake cannot leave a web session
unheld. A session with no key is judged and recorded, and never held.

``GET /api/watchdog/flags``
---------------------------

``?session_key=<key>`` answers ``{"enabled": true, "flags": [...]}``. Each flag carries
``flag_id``, ``source``, ``reason``, ``prompt_ref``, ``prompt_excerpt``, ``opened_at``,
``stop_unconfirmed`` and ``started``, plus ``watchdog``, which repeats ``source``. In a build
with no watchdog it answers ``{"enabled": false, "flags": []}``. A key that is not a safe path
segment is 400 ``invalid_session_id``.

``GET /api/terminal/posture?session_id=<key>`` reports the same state as one field, ``held``.

``POST /api/watchdog/flags/answer``
-----------------------------------

The body is ``{"session_key": "<key>", "flag_ids": ["<id>"], "answer": "stop" | "continue"}``,
with up to 256 ids. It answers ``{"released": [...], "stop_unconfirmed": [...]}``. Only the
named flags are touched, so a flag that joined after the dialog was drawn stays open and the
session stays held. A flag already closed is ignored. A record that cannot be written is 503
``hold_record_unavailable``, and an unsafe key is 400 ``invalid_session_id``.

``continue`` closes the named flags.

``stop`` ends what is running in that session first, in expert mode by an interrupt and in
simple mode by cancelling the turn, which also denies pending approvals and stops background
agents. It never terminates the session. Then it reads what the agent started. A flag closes
only if the turn has ended and nothing the agent started is still running, and this holds when
no turn was running too. Otherwise the flag stays open with ``stop_unconfirmed: true`` and
``started`` naming the commands, and a process table that could not be read counts as a command
still running. A hookless terminal cannot confirm its turn ended, so it stays unconfirmed.
Answering again repeats the check; Continue releases it, audited as such.

Frames
======

Frames are change notices on the terminal's event stream (``/api/files/events``) and the chat
feed. They carry ids and never prompt text, and the hold record stays the only truth: a client
re-reads ``GET /api/watchdog/flags`` on any frame, and a low-rate poll recovers one it missed.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - ``type``
     - Fields
     - Sent when
   * - ``watchdog_flag``
     - ``session_key``, ``flag_id``, ``watchdog``
     - A flag opened, or a Stop marked it unconfirmed.
   * - ``watchdog_resolved``
     - ``session_key``, ``flag_ids``, ``answer``
     - An answer released flags.
   * - ``watchdog_status``
     - ``session_key``, ``watchdog``, ``status``
     - ``unavailable``, or ``flag_write_failed`` when a verdict flagged and its flag could not
       be written. No record describes this state, so the frame alone paints it.

The audit entries
=================

Every entry is on the ``safety_watchdog`` surface of the
:ref:`audit trail <reference-audit-trail>`, with the watchdog as ``subject``, and joined on the
record session the prompt belongs to, whichever process wrote it. ``session`` is the session
key, or the record session where there is none. ``detail`` holds ``key=value`` identifiers
separated by spaces, and never the prompt, the judge's prose or an agent's message.

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - ``decision``
     - ``reason``
     - ``detail``
   * - ``flagged``, ``clear``
     - the option id where the judge has one; otherwise the decision
     - the verdict fields below
   * - ``unavailable``
     - the ``error_type``
     - the verdict fields below
   * - ``stop``
     - ``stop``
     - the answer fields below
   * - ``continue``
     - ``continue``, or ``after_stop_unconfirmed`` on a flag an earlier Stop could not confirm
     - the answer fields below; ``started`` on ``after_stop_unconfirmed``
   * - ``stop_unconfirmed``
     - ``stop_unconfirmed``
     - the answer fields below, with ``started``

**Verdict fields:** ``prompt_ref``, ``engine``, ``status`` (``ok`` or ``unavailable``),
``score`` and ``threshold`` where the engine has them, ``uncalibrated``, ``error_type`` when
unavailable, ``latency_s``, ``entries_kept``, ``entries_cut``, ``tokens_estimate``,
``drain_timed_out``, ``record_available``, ``input_sha256`` (the hash of the session view the
judge read), ``cost_usd`` where known, and ``refused_tools`` where an agent judge was refused
any.

**Answer fields:** ``prompt_ref``, ``flag_id``, ``operator``, and where they apply ``denied``
(pending approvals), ``stopped`` (background agents), ``after_stop_unconfirmed`` and ``started``
(commands still running). ``on_behalf_of`` carries the operator too, and an answering surface
that can name nobody records ``operator=<unattributed>``.

**``error_type``** is one of ``timeout``, ``budget``, ``in_flight_bound``, ``intake_lost``,
``no_probabilities``, ``decision_error``, ``missing_key``, ``parse_error``, ``run_error`` or
``intake_failed``.

No decision word or field says "safe"; a value that happens to contain the word is masked.

What each engine reads
======================

**Decision.** The instructions and the session's prompts, answers, tool calls and tool results,
newest kept first, bounded by the route's ``max_state_plus_question_tokens`` (30,000 by default)
and cut to fit rather than refused. Facility option ids never reach the model. The score is the
summed probability of the flag options; the reason is the ``text`` of the likeliest flag option.
The route, model and interactive budget (three seconds in all) are under ``api.jev``, and the
model id is pinned.

**Agent.** The session view, the watchdog's ``references/``, and the read tools of the
``controls`` and ``osprey_facility_knowledge`` servers plus its ``tools``. Its servers run in
read-only mode, its environment ties it to no session, and a refused call is reported in
``refused_tools``. A run that reaches ``timeout_s`` or ``max_cost_usd``, or that finds no room
under the in-flight bound, is ``unavailable``.
