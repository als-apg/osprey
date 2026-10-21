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

What the service drives --- the BPM devices, the operator settings, the
chain's steps and the trigger sample --- is the selected model's
``trajectory`` declaration (see :doc:`/how-to/describe-your-facility`). The
service reads it at start; a declaration it cannot read stops the start.

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
   * - ``GET``
     - ``setup``
     - Yes
     - ---
     - ---
   * - ``POST``
     - ``solve``
     - Yes
     - ---
     - ``solve``
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
   * - ``POST``
     - ``references/import``
     - Yes
     - ---
     - ``reference_import``
   * - ``GET``
     - ``references/{name}``
     - Yes
     - ---
     - ---
   * - ``GET``
     - ``references/{name}/readings``
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
     - ``shots/{shot_id}/raw``
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
  and read the session, Stop, run the declared default chain and write a
  step's or a put-back's settings on the terminal's target. That grant has three
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
   solve, continuous start, heartbeat, reference select, an import that
   selects, and release name the session they act in with it. Stop and the
   reading routes take none.

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
the service's **snapshot**. Its fields:

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
   * - ``n_shots_max``
     - The largest ``n_shots`` of the declared default chain that fits the
       action timeout.
   * - ``chain``
     - The declared chain, in order: each step's ``id``, ``label``,
       ``description``, ``default``, ``position``, and for a command step its
       ``command``, ``timeout_s`` and ``done``.
   * - ``catalog``
     - The devices the response matrix covers, as the facility names them:
       ``bpms``, ``hcms`` and ``vcms`` in ring order, and the facility
       ``groups`` that list them (see `Device catalog`_).
   * - ``machine``
     - ``rf_on`` and ``stored_current`` as last read; ``null`` where the
       declaration names no such reading.
   * - ``setup``
     - The current setup: ``applied_at``, ``by``, ``settings_sha``,
       ``turns_tracked`` and ``void``; ``null`` when none was applied.
   * - ``solve``
     - The published solve: its ``sha``, the spectrum (``singular_values``,
       ``inverse_weights``, ``kind``, ``L``, ``live_rank``), the response
       matrix summary ``rm`` (``source``, ``rows``, ``h_cols``, ``v_cols``,
       ``computed_at``) and the SolveConfig in force, ``config``.
   * - ``last_shot``, ``step``, ``reference``, ``last_action``
     - The last shot in brief, the pending step, the current reference and
       its match, and the last finished writing action with what it moved.
   * - ``continuous``
     - The continuous run in progress, ``shots`` and ``lease_expires_in_s``;
       ``null`` when none runs.
   * - ``retention``
     - How many shots' raw buffers the service keeps.
   * - ``notices``
     - What the service voided or dropped since, in words for the operator.

The snapshot never carries a ``client_id``. Its full schema is the service's
``panel/state.schema.json``.

Device catalog
~~~~~~~~~~~~~~

The facility build writes the catalog beside the response matrix, from the same
build, so the two always list the same devices. Each entry:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - Meaning
   * - ``name``
     - The name the service addresses the device by: a BPM's x-row address, a
       corrector's setpoint address. Exclusions use it.
   * - ``device``, ``label``
     - The facility device id and the name a person reads for it.
   * - ``places``
     - The places holding the device, root to leaf, each an ``id`` and the
       facility's own ``level`` word.
   * - ``s``
     - Its position along the model in metres, ``null`` when unknown.
   * - ``plane``
     - ``x`` or ``y`` for a corrector, ``null`` for a BPM.
   * - ``groups``
     - The ids of the facility groups that list the device.

Ring order is ``s``, then matrix order; a device without ``s`` follows the
placed ones. Each of ``groups`` is an ``id``, a ``label`` and its ``members`` by
name, and only groups listing matrix devices appear. An excluded device keeps
its entry.

Every device is in use unless the SolveConfig's ``excluded_bpms`` or
``excluded_cms`` names it. Every shot, through either door, is taken with the
SolveConfig's ``excluded_bpms`` and ``loss_cutoff`` in force, so the shot, the
step solved from it, the published solve and Apply see one BPM set.

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

``POST {door}/shot`` runs the chain in one holder action. The two doors take
different bodies.

**The panel door** takes the chain to run:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Key
     - Meaning
   * - ``steps``
     - The ticked step ids, at least one. They run in chain order, once per
       injection, whatever order the body lists them in.
   * - ``readout``
     - ``tbt`` (default) or ``pseudo_orbit``.
   * - ``n_turns``
     - The turns a pseudo-orbit readout averages over, at least 1, default
       1; above 1 only with ``pseudo_orbit``.
   * - ``n_shots``
     - The injections, at least 1, default 1; above 1 only with ``inject``
       ticked.
   * - ``average``
     - ``mean`` (default) or ``nanmean``.

It needs a current setup, and answers with the chain's ``steps_run``,
``commands_fired`` and ``shot``, ``null`` for a chain without ``readout``.

**The agent door** takes only ``n_shots`` and ``average``, runs the declared
default chain read as ``tbt``, is never refused for setup, and answers with the
shot. A default chain without ``arm``, ``inject`` and ``readout`` is refused
``invalid_chain``.

A shot:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - Meaning
   * - ``shot_id``, ``model``, ``timestamp``
     - The shot's id, the model and when it was taken.
   * - ``x``, ``y``, ``sum``
     - The averaged position and sum, keyed by the response matrix's row
       names; ``x`` and ``y`` are ``null`` at every lost or excluded BPM.
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
   * - ``readout``, ``average``, ``n_turns``
     - How the injections were read and combined.
   * - ``loss_cutoff``, ``excluded_bpms``
     - The cutoff the shot was extracted with and the BPMs it left out.
   * - ``setpoints``, ``setup_sha``, ``beam_current``
     - The response matrix's corrector setpoints read after readout, the
       setup identity the action ran under, and the stored current read
       before inject.
   * - ``raw_samples``, ``parent_id``, ``fresh``
     - The samples per BPM in the raw buffer, the shot it was re-extracted
       from, and whether it may make a step.

A readout is **fresh** when ``arm`` and ``inject`` ran in the same action and
the correctors held still across it. Only a fresh ``tbt`` shot with an
included injection makes a step; a pseudo-orbit shot is plotted and stored
and makes none. A failed injection is listed, left out of the average and
never retried.

A chain with ``readout`` stores its shot and its raw buffer; a chain without
it stores nothing. A stopped chain is refused ``action_stopped``, and a step
that failed part-way ``chain_failed``, each with ``commands_fired`` in the
body; neither stores anything.

A chain is refused ``over_budget`` before it starts when ``n_shots`` times its
per-injection worst case does not fit ``services.trajectory.action_timeout_s``.
The worst case is summed from declared bounds only: each ticked step's call
timeouts, the armed waits, the module's ``max_pause_s`` and each command
step's ``timeout_s``.

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
       content, the solve's ``sha`` and every increment.
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
   * - ``shot_id``, ``reference``
     - The shot it was computed from, and the reference it was solved
       against (``null`` for a zero trajectory).

The step solves on the model response matrix restricted to the shot's
excluded BPMs and the SolveConfig's excluded correctors, with the SolveConfig's
``alpha``, ``gain`` and ``keep_live``. Its spectrum is the published solve's.

It is refused ``no_step`` when none is pending. A new shot, a reference
selection or a SolveConfig knob replaces the pending step. A take-over, a
release, a lapsed continuous lease, an exclusion change, a setup that moved a
setting, a void setup and a put-back void it.

SolveConfig
-----------

``POST /panel/api/solve`` is the holder's SolveConfig update, one per target
and model. Its body carries only the keys the caller changes:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Key
     - Meaning
   * - ``alpha``
     - The damping.
   * - ``keep_live``
     - How many live singular values the solve keeps. The discard count it
       stands for over the current exclusions is published as ``n_discard``.
   * - ``gain``
     - The fraction of the step applied.
   * - ``excluded_bpms``, ``excluded_cms``
     - The excluded BPMs and correctors, by name.
   * - ``loss_cutoff``
     - The loss cutoff, a fraction of the injected-charge scale.

An empty body is **Calculate RM**: it restricts the model response matrix to
the current selection and republishes the solve. Every update is validated
--- the names are the build's, each plane keeps at least one BPM and one live
corrector, the numbers are in range --- and the solve is recomputed and
published in the same turn, or nothing changes. Only operator overrides are
persisted; an override whose file default changed, or which names a device the
build no longer has, is dropped at start with a notice. It answers
``{"solve", "step"}``.

A ``loss_cutoff`` change re-extracts the latest shot from its raw buffer as a
derived shot: same lineage (``parent_id``), freshness and readout, nothing
fired. During a continuous run only ``loss_cutoff`` may change, and the next
iteration reads it.

Apply
-----

``POST {door}/apply`` takes exactly one of two bodies:

- ``{"step_digest": "…"}`` writes the pending step: each corrector with
  ``δ ≠ 0`` is set to the value read just before plus ``δ``. It is recorded
  as ``apply``.
- ``{"model": "…", "values": [{"name": "…", "before": …}, …]}`` is a
  **put-back**: each named setting is set back to its ``before``. A name is a
  corrector of the response matrix or the written address of a declared
  setting. A put-back that names a setting voids the current setup. The
  service stores nothing about a put-back. It is recorded as ``put_back``.

Before the first write the whole apply is refused, with nothing written, on
any check the refusal table lists for it. A step is refused
``machine_moved`` when a response-matrix corrector setpoint or the setup
identity differs from the one its shot recorded, so a write from outside the
service since the shot voids it. A recompute on the same shot stays
appliable while the machine is unmoved and the step unconsumed.

The settings are then written one at a time, the stop event and
``action_timeout_s`` checked between writes. The reply names every write that
was not refused:

.. code-block:: json

   {"step_digest": "…", "shot_id": "…", "stopped": false,
    "moved": [{"name": "HCM1", "before": 0.12, "after": 0.13,
               "outcome": "confirmed"}]}

``before`` is the value read just before the write, ``after`` the value
written, and ``outcome`` what the connector reported (``confirmed``,
``mismatch``, ``unconfirmed``, ``failed``, …), so ``moved`` is a valid
put-back body's ``values``. A write that raised is listed too, since it may
have landed. A stopped apply answers ``stopped: true`` with what moved before
the Stop. A write that fails or is refused, or a deadline passed, answers
``502`` ``apply_failed`` with ``moved`` in the body.

Setup
-----

``POST /panel/api/setup`` is one holder action and a partial update. Its body
carries ``settings``, the value per declared setting id to apply, and
optionally ``excluded_bpms``, which it hands to the SolveConfig. Every setting
it does not name is kept as it reads. It writes only where a readback differs
from the value asked, waits for each written readback to settle within its
tolerance and ``timeout_s``, and answers:

.. code-block:: json

   {"moved": [{"name": "…", "before": 0, "after": 3, "outcome": "confirmed"}],
    "unsettled": [],
    "setup": {"applied_at": 1760000000.0, "by": "alice", "settings_sha": "…",
              "turns_tracked": 250, "void": false}}

When every write landed and settled, the setup **takes**: it is persisted
per target and model and becomes the current setup. It is re-checked against
the readbacks at start and before every chain turn on both doors; a
divergence voids it and the step, with a notice. Panel chains need a current
setup; the agent door never does. A setup is refused while a continuous run
is in progress. A write that fails part-way answers ``502`` ``setup_failed``
with ``moved`` in the body.

``GET /panel/api/setup`` answers every declared setting as read back,
``{"settings", "read_at", "setup"}``: a ``device`` setting's ``values`` per
BPM, a ``global`` one's ``value``.

Continuous
----------

``POST /panel/api/continuous`` takes the panel door's chain body, needs a
current setup, and answers at once with the run's ``shots`` and
``lease_expires_in_s``. The run takes one chain action after another, the run
lock released between them, two starts at least
``services.trajectory.continuous_min_period_s`` apart. It never applies a step
and is the only occupant of the run slot.

``POST /panel/api/continuous/heartbeat`` moves the run's lease and nothing
else; it answers ``{"continuous": …}``, the run's status or ``null`` when none
runs. A lease no heartbeat renews within ten seconds ends the run.

A run ends, first cause wins, as ``stopped`` (a Stop), ``taken_over``,
``released``, ``lease_expired``, ``failed`` (an iteration raised, or every
injection of one failed) or ``shutdown``. The end is published once on the
state stream, with the last step's reason.

Raw buffers
-----------

``GET /panel/api/shots/{shot_id}/raw`` answers the samples a shot was made
from, as ``application/octet-stream``:

.. code-block:: text

   bytes 0..3      header length n, uint32 little-endian, a multiple of 4
   bytes 4..4+n    the header: UTF-8 JSON, padded with spaces
   bytes 4+n..     the samples: float32 little-endian, C order

The header is ``{"shape": [n_included, 3, n_bpm, raw_samples],
"trigger_sample": …, "units": {"x": …, "y": …, "sum": …}}``: per included
injection the planes ``x``, ``y`` and ``sum``, per plane one row per included
BPM in ``bpm_order``, each from the trigger sample through the turns tracked.
A derived shot answers its parent's buffer.

The service keeps the ``retention`` latest shots that a chain made, each with
its buffer and the shots derived from it, and prunes the oldest beyond it; the
latest is always kept. A pruned shot's buffer answers ``410``
``raw_evicted``.

References and shots
--------------------

All under the panel door, and all refused ``store_unavailable`` when the
service has no state directory. Shots and references are kept per target and
model; a reference belonging to another target is refused
``reference_foreign``. A reference carries the readout mode it was taken in
and is never used across modes.

``GET references``
   ``{"model", "names", "current"}``: the model's saved references and the
   current one.

``POST references``
   Takes ``name`` and, optionally, ``shot_id``; saves that shot (else the
   pending step's, else the newest) as the reference ``name``. Only a fresh
   ``tbt`` shot is saved. Answers ``{"model", "name", "sha256", "shot_id"}``.

``POST references/import``
   Takes a reference file's content: ``name``, ``readings`` (``{"x", "y"}``
   per BPM, each a finite number or ``null``), ``readout``, and optionally
   ``select``, ``source_target`` and ``source_model``. Every BPM must be one
   the tool reads, and the body is at most 1 MiB. The reference belongs to
   this target and keeps its source. With ``select`` it is made current in
   the holder's session and the step recomputed. Answers ``{"model", "name",
   "sha256", "reference", "step"}``.

``GET references/{name}``
   The reference matched over the current BPMs: ``matched``,
   ``missing_in_reference``, ``unused_in_reference``, ``nan_count`` and
   ``readout``.

``GET references/{name}/readings``
   The stored file, in the import body's shape: a reference exported this
   way imports elsewhere.

``POST references/select``
   Takes ``name``, or ``null`` for none, makes it current and recomputes the
   pending step against it. Answers ``{"reference", "step"}``. dx and dy are
   always against the current reference.

``GET shots``
   ``{"model", "shots"}``: every stored shot of the model, newest first.

State stream
------------

``GET /panel/api/events`` is a server-sent event stream
(``text/event-stream``). Each event is one frame: an ``event:`` line naming
its type, a ``data:`` line carrying its JSON, and a blank line. The JSON
carries the same name in its ``type`` field:

.. code-block:: text

   event: snapshot
   data: {"type": "snapshot", "snapshot": {"target": "va", …}}

The event types are ``snapshot``, ``shot``, ``step``, ``apply``,
``stopped``, ``taken_over`` and ``continuous_ended``. A browser client
listens for each type by name; a frame with an ``event:`` line never reaches
a generic message handler.

The first frame of every stream is a ``snapshot``, so a client that connects
or reconnects starts from the whole state. Every change sends its events and
then a fresh ``snapshot``. An idle stream sends a keep-alive comment
(``: keep-alive``) every 15 seconds, which a client ignores. A stream ends
only when the service shuts down.

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
     - A setup, an apply, a SolveConfig key other than ``loss_cutoff`` or a
       second continuous start while a continuous run is in progress
   * - ``owner_in_body``
     - 400
     - The body names an ``owner``
   * - ``bad_request``
     - 400
     - The body carries another unknown key, is not JSON, or holds a value of
       the wrong kind; a call names no ``client_id``; an apply carries
       neither or both shapes; a put-back names a setting that is neither a
       corrector nor a declared setting

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
     - A chain that injects with no sequence module loaded
   * - ``unknown_step``
     - 400
     - A chain body names a step the declaration does not
   * - ``invalid_chain``
     - 400
     - ``pseudo_orbit`` without ``readout``, ``n_turns`` above 1 without
       ``pseudo_orbit``, ``n_shots`` above 1 without ``inject``, every BPM
       excluded, or an agent-door default chain without ``arm``, ``inject``
       and ``readout``
   * - ``over_budget``
     - 400
     - A chain whose worst case does not fit ``action_timeout_s``
   * - ``setup_required``
     - 409
     - A panel chain or continuous start with no current setup, or a void one
   * - ``action_stopped``
     - 409
     - The chain was stopped; the body lists ``commands_fired``
   * - ``chain_failed``
     - 502
     - A step failed part-way; the body lists ``steps_run`` and
       ``commands_fired``
   * - ``unknown_setting``
     - 400
     - A setup names a setting id the model does not declare
   * - ``setup_failed``
     - 502
     - A setup write failed part-way; the body lists ``moved``
   * - ``setup_unreadable``
     - 503
     - The declared settings cannot be read
   * - ``setup_unsaved``
     - 503
     - The setup cannot be persisted
   * - ``store_unavailable``
     - 503
     - The service has no state directory, so it keeps no shots, buffers or
       references
   * - ``raw_evicted``
     - 410
     - The shot's raw buffer was pruned; a re-extraction of it is refused the
       same way
   * - ``exclusions_changed``
     - 409
     - A re-extraction of a shot read under other BPM exclusions

SolveConfig
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 8 62

   * - Code
     - Status
     - When
   * - ``invalid_solve_config``
     - 422
     - A value out of range or of the wrong kind
   * - ``selection_unknown``
     - 409
     - An exclusion names no BPM or corrector of the build
   * - ``selection_too_small``
     - 409
     - The selection leaves a plane without a BPM or a live corrector
   * - ``solve_failed``
     - 422
     - The tool cannot solve with the selection and settings
   * - ``solve_config_unsaved``
     - 503
     - The overrides cannot be persisted

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
   * - ``machine_moved``
     - 409
     - A corrector setpoint or the setup identity changed since the step's
       shot, or the shot recorded no setpoints
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
   * - ``reference_foreign``
     - 409
     - The stored reference belongs to another target
   * - ``shot_not_found``
     - 404
     - No stored shot of that id
   * - ``no_shot``
     - 409
     - A reference save with no shot to save from
   * - ``shot_not_fresh``
     - 409
     - A reference save from a shot that is not fresh
   * - ``readout_mismatch``
     - 409
     - A reference save from a pseudo-orbit shot, or a selection or an
       import that selects whose readout is not the shown shot's
   * - ``no_matching_bpms``
     - 409
     - The reference names none of the BPMs it is matched over
   * - ``unknown_bpm``
     - 400
     - An import names a BPM the tool does not read
   * - ``reading_not_finite``
     - 400
     - An import carries a reading that is neither finite nor ``null``
   * - ``reference_too_large``
     - 413
     - An import body above 1 MiB

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
``put_back``, ``setup``, ``solve``, ``continuous_start``,
``reference_save``, ``reference_import`` or ``reference_select``.
``detail`` names the owner, the door (``kind=``) and the target, and what the
action took and did:

- a shot or a continuous start, the ``steps``, ``readout``, ``n_shots`` and
  ``average``; a shot also the ``shot_id`` and ``commands_fired``, and a
  stopped one the stopper;
- an apply, the ``step_digest``; a put-back, the ``model``; both, and a
  setup, ``stopped`` and what ``moved``;
- a setup, the ``settings`` it named, the ``excluded_bpms`` it carried, what
  stayed ``unsettled`` and the ``settings_sha``;
- a SolveConfig update, the ``keys`` sent, those ``changed`` and the
  ``solve_sha``;
- a reference save or import, the ``name`` and ``sha256``;
- a Stop, the action it reached;
- a release, the owner, the target and the ``client_id`` only.

A control-path action (take-over, release, Stop, continuous start) is
recorded as it is decided; a worker action (open, shot, setup, SolveConfig
update, apply, put-back, reference select) after the worker answered. ``GET`` routes and
heartbeats are not recorded. A launch-token refusal is recorded by the
request layer, in ``http_mutation.jsonl``, and a request the web gate refused
in ``web_auth.jsonl``, both in the same directory. See
:ref:`reference-audit-trail`.

.. seealso::

   :ref:`reference-trajectory-sequence`
      The facility module a chain's ``inject`` step runs.

   :ref:`reference-trajectory-model`
      The model the service drives: its states, command records and buffers.

   :ref:`reference-guarded-run`
      The run lock every worker action takes, and its holder record.
