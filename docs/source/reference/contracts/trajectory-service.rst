.. _reference-trajectory-service:

===========================
Trajectory Service Contract
===========================

The trajectory service claims one control target for first-turn threading
through a session, behind two doors. :ref:`architecture-trajectory-service`
says how the doors fit and whose action an action is;
:doc:`/how-to/thread-the-first-turn` says how the loop is driven. This page is
the exact behaviour: the routes, what each takes and returns, the headers,
and every refusal a caller can be given.

Routes
======

The **panel door** is the service path ``/panel/api/``, reached through the
terminal's panel proxy at ``/panel/trajectory/panel/api/``. The **agent
door** is ``/api/``, reached at ``/panel/trajectory/api/``. Each route in this table
is relative to its door.

.. list-table::
   :header-rows: 1
   :widths: 12 34 18 18 18

   * - Method
     - Route
     - Panel door
     - Agent door
     - Records
   * - ``POST``
     - ``session``
     - Yes
     - Yes
     - ``session_open``
   * - ``POST``
     - ``takeover``
     - Yes
     - Yes
     - ``session_takeover``
   * - ``DELETE``
     - ``session``
     - Yes
     - Yes
     - ``session_release``
   * - ``GET``
     - ``state``
     - Yes
     - Yes
     - ---
   * - ``POST``
     - ``stop``
     - Yes
     - Yes
     - ``stop``
   * - ``POST``
     - ``shot``
     - Yes
     - Yes
     - ``shot``
   * - ``GET``
     - ``step``
     - Yes
     - Yes
     - ---
   * - ``POST``
     - ``apply``
     - Yes
     - Yes
     - ``apply`` or ``put_back``
   * - ``POST``
     - ``setup``
     - Yes
     - ---
     - ``setup``
   * - ``POST``
     - ``continuous``
     - Yes
     - ---
     - ``continuous_start``
   * - ``POST``
     - ``continuous/heartbeat``
     - Yes
     - ---
     - ---
   * - ``GET``
     - ``references``
     - Yes
     - ---
     - ---
   * - ``POST``
     - ``references``
     - Yes
     - ---
     - ``reference_save``
   * - ``GET``
     - ``references/{name}``
     - Yes
     - ---
     - ---
   * - ``POST``
     - ``references/select``
     - Yes
     - ---
     - ``reference_select``
   * - ``GET``
     - ``shots``
     - Yes
     - ---
     - ---
   * - ``GET``
     - ``events``
     - Yes
     - ---
     - ---

``GET /health`` sits outside both doors and the gate, and is not proxied. The
panel bundle is served under ``/panel/`` and, through the terminal, under
``/panel/trajectory/panel/``.

Who may call a route through the terminal:

- **The panel door and the bundle** take an operator credential only. The
  panel door is the operator's own browser; the panel token cannot reach it.
- **The agent door** takes an operator credential or the terminal's panel
  token. With the panel token alone a caller can claim, take over, release
  and read the session, Stop, fire a shot's commands and write a step's or a
  put-back's corrector settings on the terminal's target. That grant has three
  bounds. The proxy adds the service's launch token and the terminal's secret
  on that hop and never hands them back, so the caller holds neither, and it
  drops any owner the caller sends: every action is the terminal owner's and
  is recorded as theirs. The service admits every write itself --- the
  owner's control context, narrowing and the writes posture --- and runs it
  under the target's guarded-run lock and its limits validator. The human
  approval of a shot or an apply is checked by the agent's approval hook and
  its trajectory server (see :ref:`trajectory-mcp-server`), not by the proxy
  or the service.

A route under ``/panel/api/`` records its actions with ``kind`` ``panel``; a
route under ``/api/`` records ``kind`` ``agent``.

Headers
=======

``X-Launch-Token``
   Required on every agent-door route, ``GET /api/state`` included. The proxy
   adds it on the hop from the terminal; a caller never sends it. The panel
   door never reads it.

``X-Trajectory-Client``
   The ``client_id`` an open or a take-over returned. Shot, apply, setup,
   continuous start, heartbeat, reference select and release name the session
   they act in with it. Stop, state, step and the reading routes take none.

``X-Osprey-Owner``
   Never read from the request. The service's gate drops it and names the
   account the admitting secret maps to (see
   :ref:`architecture-safety-chain-owner`).

Requests and replies
====================

A body is a JSON object. Any key a route does not take is refused
``bad_request``, a key named ``owner`` ``owner_in_body``; a body that is not
JSON is refused ``bad_request``.

Sessions
--------

Open, take over, release and Stop carry no body.

``POST {door}/session`` and ``POST {door}/takeover`` reply with the session
the caller now holds:

.. code-block:: json

   {"client_id": "…", "owner": "alice", "target": "va"}

``DELETE {door}/session`` replies with the session it ended, the same three
fields.

State
-----

``GET {door}/state`` answers whoever calls it, with or without an owner, with
the service's **snapshot**. Its main fields:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - Meaning
   * - ``target``
     - The service's control target.
   * - ``caller``
     - The owner the gate named for this request, ``null`` when it named
       nobody.
   * - ``writes_enabled``, ``posture``
     - Whether the deployment arms writes on the target, and the posture the
       caller acts under: ``writes``, ``read_only``, ``sandbox``, or the code
       admission would refuse the caller with.
   * - ``limits``
     - The limits posture actions run under: ``off``, ``optional``,
       ``exclusive`` or ``unavailable``.
   * - ``writing_available``, ``model``, ``model_kind``
     - Whether the build gave the service a trajectory model, its name, and
       ``single_pass`` or ``none``.
   * - ``holder``, ``taken_over``
     - The holder's ``owner`` and door ``kind``, and the latest take-over;
       ``null`` when there is none.
   * - ``mode``, ``action``
     - ``idle``, ``action`` or ``continuous``, and the action in flight with
       whether it is stopping.
   * - ``queue_wait_s``, ``action_timeout_s``, ``action_budget_s``,
       ``stop_budget_s``
     - How long a call submitted now waits to start, the configured action
       timeout, the longest an admitted action may take, and the longest an
       action takes to end after a Stop.
   * - ``last_shot``, ``step``, ``reference``, ``last_action``
     - The last shot in brief, the pending step, the current reference and
       its match, and the last finished writing action with what it moved.
   * - ``continuous``
     - The continuous run in progress: ``shots``, ``max_shots`` and
       ``lease_expires_in_s``; ``null`` when none runs.

The snapshot never carries a ``client_id``. Its full schema is the service's
``panel/state.schema.json``.

Stop
----

``POST {door}/stop`` answers ``202`` at once:

.. code-block:: json

   {"owner": "alice", "kind": "panel", "target": "va", "action": "shot",
    "stop_budget_s": 2.6}

``action`` names the action it reached, ``null`` when none was in flight. The
action ends at its next checkpoint, within ``stop_budget_s``: the sequence
module's ``max_pause_s``, the connector's call timeout and the checkpoint
poll. A continuous run ends ``stopped``.

Shot
----

``POST {door}/shot`` takes ``n_shots`` (a whole number, at least 1, default
1) and ``average`` (``mean`` or ``nanmean``, default ``mean``). It answers
when the shot is done, with the shot:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - Meaning
   * - ``shot_id``, ``model``, ``timestamp``
     - The shot's id, the model and when it was taken.
   * - ``x``, ``y``, ``sum``
     - The averaged first-turn position and sum, keyed by the response
       matrix's row names; ``x`` and ``y`` are ``null`` at every lost BPM.
   * - ``bpm_order``, ``lost``, ``last_bpm``
     - The BPMs in the order the beam passes them, those that lost it, and
       the last that saw it.
   * - ``window``, ``per_shot_lost_at``
     - The raw window ``[start, stop)`` read, in samples, and the first lost
       BPM of each included injection (``null`` for a full turn).
   * - ``shots_included``, ``shots_failed``
     - How many injections were averaged, and each failed one with its
       ``index``, its ``reason`` and the BPMs that never armed.
   * - ``commands_fired``
     - Every command written, in order.

A shot that included at least one injection is stored as a shot file and
leaves a pending step; one whose every injection failed leaves none. A shot
that was stopped is refused ``action_stopped`` with ``commands_fired`` in the
body, and stores nothing.

A shot is refused ``over_budget`` before it starts when
``n_shots × (3·armed_timeout_s + max_pause_s + call_timeout_s)`` exceeds
``services.trajectory.action_timeout_s``.

Step
----

``GET {door}/step`` answers the pending step to any owner, without a
session, and never waits behind the worker:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - Meaning
   * - ``digest``
     - What an apply of this step must carry. It binds the target, the
       model, the holder's ``client_id``, the shot, the reference's name and
       content, and every increment.
   * - ``deltas``
     - The increment of each corrector, ``δ = −factor·step``.
   * - ``factor``
     - The uniform scale, at most 1, that keeps every ``|δ|`` inside its
       ``max_step``.
   * - ``reason``, ``status``, ``all_zero``
     - The tool's reason (``OK``, ``ON_REFERENCE``, ``NO_BEAM``, …), its
       sentence for it, and whether every increment is zero.
   * - ``zeroed_h``, ``zeroed_v``, ``no_response_names``, ``unsolved``
     - The correctors zeroed in each plane, those with no response, and
       those the response matrix could not solve.
   * - ``singular_values``, ``inverse_weights``, ``spectrum_kind``,
       ``rank_discarded``, ``n_discard``
     - The solve's spectrum.
   * - ``shot_id``, ``reference``
     - The shot it was computed from, and the reference it was solved
       against (``null`` for a zero trajectory).

It is refused ``no_step`` when none is pending. A new shot or a reference
selection replaces the pending step; a take-over, a release, a setup or a
put-back voids it.

Apply
-----

``POST {door}/apply`` takes exactly one of two bodies:

- ``{"step_digest": "…"}`` writes the pending step: each corrector with
  ``δ ≠ 0`` is set to the value read just before plus ``δ``. It is recorded
  as ``apply``.
- ``{"model": "…", "values": [{"name": "…", "before": …}, …]}`` is a
  **put-back**: each named setting is set back to its ``before``. A name is a
  corrector of the response matrix or an address of a slot the sequence
  module declares. The service stores nothing about a put-back. It is
  recorded as ``put_back``.

Before the first write the whole apply is refused, with nothing written, on
any check the refusal table lists for it. The settings are then written one
at a time, the stop event and ``action_timeout_s`` checked between writes. The
reply names every write that was not refused:

.. code-block:: json

   {"name": "apply", "stopped": false,
    "moved": [{"name": "HCM1", "before": 0.12, "outcome": "confirmed"}]}

``before`` is the value read just before the write and ``outcome`` what the
connector reported (``confirmed``, ``mismatch``, ``unconfirmed``, ``failed``,
…), so ``moved`` is a valid put-back body's ``values``. A write that raised is listed too, since it may
have landed. A stopped apply answers ``stopped: true`` with what moved before
the Stop. A write that fails or is refused, or a deadline passed, answers
``502`` ``apply_failed`` with ``moved`` in the body.

Setup
-----

``POST /panel/api/setup`` takes ``bunch_clean`` (``true`` or ``false``,
default ``false``). It runs the sequence module's ``bpm_mode``, then, when
asked, ``bunch_clean``, under the run lock and outside any shot (see
:ref:`reference-trajectory-sequence`). It answers like an apply, ``name``
``setup``; an entry point that raises answers ``502`` ``setup_failed`` with
``moved`` in the body.

Continuous
----------

``POST /panel/api/continuous`` takes ``n_shots``, ``average`` and
``max_shots`` (default ``services.trajectory.continuous_max_shots``). It
answers at once with the run's ``shots``, ``max_shots`` and
``lease_expires_in_s``, and the run takes one shot action after another, the
run lock released between them. It never applies a step.

``POST /panel/api/continuous/heartbeat`` moves the run's lease and nothing
else; it answers ``{"continuous": …}``, the run's status or ``null`` when none
runs. A lease no heartbeat renews within ten seconds ends the run.

A run ends, first cause wins, as ``stopped`` (a Stop), ``taken_over``,
``released``, ``max_shots``, ``zero_step`` (the latest step is all zero),
``lease_expired``, ``failed`` (a shot raised) or ``shutdown``. The end is
published once on the state stream.

References and shots
--------------------

All under the panel door, and all refused ``store_unavailable`` when the
service has no state directory.

``GET references``
   ``{"model", "names", "current"}``: the model's saved references and the
   current one.

``POST references``
   Takes ``name`` and, optionally, ``shot_id``; saves that shot (else the
   pending step's, else the newest) as the reference ``name``. Answers
   ``{"model", "name", "sha256", "shot_id"}``.

``GET references/{name}``
   The reference matched over the current BPMs: ``matched``,
   ``missing_in_reference``, ``unused_in_reference`` and ``nan_count``.

``POST references/select``
   Takes ``name``, or ``null`` for none, makes it current and recomputes the
   pending step against it. Answers ``{"reference", "step"}``.

``GET shots``
   ``{"model", "shots"}``: every stored shot of the model, newest first.

State stream
------------

``GET /panel/api/events`` is a server-sent event stream. Its first event is
a ``snapshot``; every change sends its events and then a fresh
``snapshot``. The event types are ``snapshot``, ``shot``, ``step``,
``apply``, ``stopped``, ``taken_over`` and ``continuous_ended``. An idle
stream sends a keep-alive comment every 15 seconds. A stream ends only when
the service shuts down.

Health
------

``GET /health`` reads the target live on every call:

.. code-block:: json

   {"status": "ok", "target": "reachable", "control_target": "va"}

``target`` is ``reachable``, ``target_unreachable`` or
``trajectory_unavailable`` (the build has no trajectory model).

Sessions
========

- One holder per target. A second open is refused ``session_held``, naming
  the holder and its door.
- Opening is the only way to acquire a session. A take-over with no holder is
  refused ``no_session``.
- Any owner the target admits may take over, through either door, except
  while an apply or a put-back is writing (``apply_in_progress``). The old
  holder's ``client_id`` is void at once: its action in flight is stopped,
  its queued calls and later ones are refused ``session_taken_over``, and the
  pending step is voided.
- A Stop needs no session: any admitted owner may send one, through either
  door, and the holder keeps the session.
- A call or release from anyone but the holder, or with an unknown
  ``client_id``, is refused ``not_session_holder``.
- Take-over, release, Stop and heartbeat act at admission: they take no lock,
  probe nothing and write nothing, so they are answered while an action is in
  flight. A release checks only the holder, so a narrowed owner or an
  unreachable target never blocks it.
- Sessions live in memory: a restart leaves no holder.

Refusals
========

A refusal is FastAPI's ``HTTPException`` with a coded body:

.. code-block:: json

   {"detail": {"code": "run_busy", "detail": "…"}}

``503`` is configuration, ``403`` posture, ``409`` state, ``404`` a missing
thing, ``400`` the request, and ``502`` an action that failed part-way.

Admission and the worker
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 8 62

   * - Code
     - Status
     - When
   * - ``launch_token_required``
     - 503
     - The service has no launch token configured (agent door)
   * - ``launch_token_invalid``
     - 403
     - The launch token is missing or wrong (agent door)
   * - ``no_operator_identity``
     - 403
     - A writing action or a step read with no owner, or with ``unknown`` or
       the service's own account as the owner
   * - ``writes_disabled``
     - 403
     - The deployment does not arm writes on the service's target, or the
       process runs read-only
   * - ``control_context_unavailable``
     - 403
     - The owner's control context cannot be read
   * - ``narrowing``
     - 403
     - The owner's control context is narrowed
   * - ``control_target_mismatch``
     - 409
     - The owner's control context names another target than the service's
   * - ``run_busy``
     - 409
     - Another run holds the target's guarded-run lock
   * - ``action_stopped``
     - 409
     - The action's session was taken over or stopped before or while it ran
   * - ``guarded_run_unavailable``
     - 503
     - The guarded-run directory cannot be used
   * - ``limits_unavailable``
     - 503
     - The target's limits validator is the fail-safe one, or none was built
       and the config does not state limits checking off
   * - ``target_unreachable``
     - 503
     - The target did not answer the live read
   * - ``trajectory_unavailable``
     - 503
     - The build has no trajectory model for the service
   * - ``service_stopping``
     - 503
     - The service is shutting down

Sessions and bodies
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 8 62

   * - Code
     - Status
     - When
   * - ``session_held``
     - 409
     - An open while another session holds the target
   * - ``no_session``
     - 409
     - A take-over with no session to take over
   * - ``not_session_holder``
     - 409
     - A release or call from anyone but the holder
   * - ``session_taken_over``
     - 409
     - A call carrying a ``client_id`` a take-over voided
   * - ``apply_in_progress``
     - 409
     - A take-over while an apply or a put-back is writing
   * - ``continuous_running``
     - 409
     - A setup, an apply or a second continuous start while a continuous run
       is in progress
   * - ``owner_in_body``
     - 400
     - The body names an ``owner``
   * - ``bad_request``
     - 400
     - The body carries another unknown key, is not JSON, or holds a value of
       the wrong kind; a call names no ``client_id``; an apply carries
       neither or both shapes; a put-back names a setting that is neither a
       corrector nor a module slot

Shots and setup
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 8 62

   * - Code
     - Status
     - When
   * - ``no_sequence_module``
     - 503
     - A shot, setup or continuous start with no sequence module loaded
   * - ``no_bunch_clean``
     - 503
     - A setup asking for ``bunch_clean`` of a module that defines none
   * - ``over_budget``
     - 400
     - A shot whose worst case does not fit ``action_timeout_s``
   * - ``store_unavailable``
     - 503
     - The service has no state directory, so it keeps no shot files and no
       references
   * - ``setup_failed``
     - 502
     - A setup entry point raised; the body lists ``moved``

Steps and applies
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 8 62

   * - Code
     - Status
     - When
   * - ``tool_unavailable``
     - 503
     - The service has no correction tool (no response matrix for the model)
   * - ``shot_mismatch``
     - 409
     - The shot's rows are not the tool's
   * - ``reference_unusable``
     - 409
     - The current reference cannot be matched to the shot
   * - ``step_failed``
     - 422
     - The tool refused to solve the shot
   * - ``no_step``
     - 404
     - ``GET step`` with no step pending
   * - ``digest_mismatch``
     - 409
     - The apply's digest is not the pending step's, or no step is pending
   * - ``stale_shot``
     - 409
     - The pending step's shot is not the latest shot
   * - ``step_consumed``
     - 409
     - The pending step was already applied
   * - ``zero_step``
     - 409
     - Every increment of the step is zero
   * - ``model_mismatch``
     - 409
     - A put-back names another model than the service's
   * - ``limits_violation``
     - 409
     - A value to write breaks the target's limits validator (its bounds, or
       its ``max_step`` against the value just read)
   * - ``plan_running``
     - 409
     - A Bluesky plan runs on a bridge serving the target; ``503`` when a
       bridge deployed for the target does not answer
   * - ``apply_failed``
     - 502
     - A write failed or was refused, or the deadline passed, part-way; the
       body lists ``moved``

References
----------

.. list-table::
   :header-rows: 1
   :widths: 30 8 62

   * - Code
     - Status
     - When
   * - ``invalid_name``
     - 400
     - A reference name the store does not take
   * - ``reference_not_found``
     - 404
     - No reference of that name
   * - ``shot_not_found``
     - 404
     - No stored shot of that id
   * - ``no_shot``
     - 409
     - A reference save with no shot to save from
   * - ``no_matching_bpms``
     - 409
     - The reference names none of the BPMs it is matched over

A request the web gate refuses before it reaches a door --- no credential,
the panel token on an operator-only route --- is answered by the gate, not
by these tables, with ``401`` or ``403``.

Each check runs in the order :ref:`architecture-trajectory-service` lists, and
the first that refuses decides. Take-over, release and Stop are decided at
admission and never meet ``run_busy``, ``target_unreachable`` or
``limits_unavailable``.

Audit
=====

Each writing action is one record in ``trajectory_service.jsonl`` under
``var/audit/trajectory/``, allowed or refused, with the owner as ``actor``.
Its ``subject`` is the name in the route table: ``session_open``,
``session_takeover``, ``session_release``, ``stop``, ``shot``, ``apply``,
``put_back``, ``setup``, ``continuous_start``, ``reference_save`` or
``reference_select``. ``detail`` names the owner, the door (``kind=``) and the
target, and what the action took and did:

- a shot, ``n_shots``, ``average``, the ``shot_id`` and ``commands_fired``;
  a stopped shot, the stopper too;
- an apply, the ``step_digest``; a put-back, the ``model``; both, and a
  setup, ``stopped`` and what ``moved``;
- a Stop, the action it reached;
- a release, the owner, the target and the ``client_id`` only.

A control-path action (take-over, release, Stop, continuous start) is
recorded as it is decided; a worker action (open, shot, setup, apply,
put-back, reference select) after the worker answered. ``GET`` routes and
heartbeats are not recorded. A launch-token refusal is recorded by the
request layer, in ``http_mutation.jsonl``, and a request the web gate refused
in ``web_auth.jsonl``, both in the same directory. See
:ref:`reference-audit-trail`.

.. seealso::

   :ref:`reference-trajectory-sequence`
      The facility module a shot and a setup run.

   :ref:`reference-trajectory-model`
      The model the service drives: its states, command slots and buffers.

   :ref:`reference-guarded-run`
      The run lock every worker action takes, and its holder record.
