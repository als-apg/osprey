.. _reference-trajectory-service:

===========================
Trajectory Service Contract
===========================

The trajectory service claims one control target for first-turn threading
through a session, behind two doors. :ref:`architecture-trajectory-service`
says how the doors fit and whose action an action is. This page is the exact
behaviour: the routes, what each takes and returns, the headers, and every
refusal a caller can be given.

Routes
======

Each door serves the same four routes. The service's own path is what the
service sees; a caller reaches it through the terminal's panel proxy, under
``/panel/trajectory/``.

.. list-table::
   :header-rows: 1
   :widths: 14 26 34 26

   * - Method
     - Service path
     - Through the terminal
     - Who may call it there
   * - ``POST``
     - ``/panel/api/session``
     - ``/panel/trajectory/panel/api/session``
     - Operator only
   * - ``POST``
     - ``/panel/api/takeover``
     - ``/panel/trajectory/panel/api/takeover``
     - Operator only
   * - ``DELETE``
     - ``/panel/api/session``
     - ``/panel/trajectory/panel/api/session``
     - Operator only
   * - ``GET``
     - ``/panel/api/state``
     - ``/panel/trajectory/panel/api/state``
     - Operator only
   * - ``POST``
     - ``/api/session``
     - ``/panel/trajectory/api/session``
     - Operator or panel token
   * - ``POST``
     - ``/api/takeover``
     - ``/panel/trajectory/api/takeover``
     - Operator or panel token
   * - ``DELETE``
     - ``/api/session``
     - ``/panel/trajectory/api/session``
     - Operator or panel token
   * - ``GET``
     - ``/api/state``
     - ``/panel/trajectory/api/state``
     - Operator or panel token
   * - ``GET``
     - ``/health``
     - Not proxied
     - Anyone; outside the gate

The panel bundle is served under ``/panel/`` and, through the terminal, under
``/panel/trajectory/panel/``; the panel token cannot reach it. A route under
``/panel/api/`` belongs to the **panel door** and records its actions with
``kind`` ``panel``; a route under ``/api/`` belongs to the **agent door** and
records ``kind`` ``agent``.

Headers
=======

``X-Launch-Token``
   Required on every agent-door route, ``GET /api/state`` included. The proxy
   adds it on the hop from the terminal; a caller never sends it. The panel
   door never reads it.

``X-Trajectory-Client``
   The ``client_id`` an open or a take-over returned. A release names the
   session it ends with it.

``X-Osprey-Owner``
   Never read from the request. The service's gate drops it and names the
   account the admitting secret maps to (see
   :ref:`architecture-safety-chain-owner`).

Requests and replies
====================

Open, take over and release carry no body. Any key in a body is refused, a
key named ``owner`` with its own code; a body that is not JSON is refused too.

``POST {door}/session`` and ``POST {door}/takeover`` reply with the session
the caller now holds:

.. code-block:: json

   {"client_id": "…", "owner": "alice", "target": "va"}

``DELETE {door}/session`` replies with the session it ended, the same three
fields.

``GET {door}/state`` answers whoever calls it, with or without an owner:

.. code-block:: json

   {"target": "va", "owner": "alice", "kind": "panel",
    "caller": "alice", "writing_available": true}

``owner`` and ``kind`` name the holder and the door it opened through, both
``null`` when no session is held. ``caller`` is the owner the gate named for
this request, ``null`` when it named nobody. ``writing_available`` says
whether the service has a trajectory model to drive. The reply never carries
a ``client_id``.

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
- Any owner the target admits may take over, through either door. The old
  holder's ``client_id`` is void at once: its queued calls and later ones are
  refused ``session_taken_over``.
- A release, or a call, from anyone but the holder, or with an unknown
  ``client_id``, is refused ``not_session_holder``.
- A release checks only the holder, so a narrowed owner or an unreachable
  target never blocks it.
- Sessions live in memory: a restart leaves no holder.

Refusals
========

A refusal is FastAPI's ``HTTPException`` with a coded body:

.. code-block:: json

   {"detail": {"code": "run_busy", "detail": "…"}}

``503`` is configuration, ``403`` posture, ``409`` state and ``400`` the
request body.

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
     - A writing action with no owner, or with ``unknown`` or the service's
       own account as the owner
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
     - The action's session was taken over or stopped before it ran
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
   * - ``guarded_run_unavailable``
     - 503
     - The guarded-run directory cannot be used
   * - ``limits_unavailable``
     - 503
     - No enforcing limits validator could be built for the target
   * - ``target_unreachable``
     - 503
     - The target did not answer the live read
   * - ``trajectory_unavailable``
     - 503
     - The build has no trajectory model for the service
   * - ``owner_in_body``
     - 400
     - The body names an ``owner``
   * - ``bad_request``
     - 400
     - The body carries any other key or is not JSON, or a release names no
       ``client_id``

A request the web gate refuses before it reaches a door --- no credential,
the panel token on an operator-only route --- is answered by the gate, not
by this table, with ``401`` or ``403``.

Each check runs in the order :ref:`architecture-trajectory-service` lists, and
the first that refuses decides. Take-over and release are decided at
admission and never meet ``run_busy``, ``target_unreachable`` or
``limits_unavailable``.

Audit
=====

Each open, take-over and release is one record in
``trajectory_service.jsonl`` under ``var/audit/trajectory/``, allowed or
refused, with the owner as ``actor``; ``detail`` names the owner, the door and
the target. A release's record names the owner, the target and the
``client_id`` only. ``GET`` routes are not recorded. A launch-token refusal of
an open, take-over or release is recorded by the request layer, in
``http_mutation.jsonl``, and a request the web gate refused in
``web_auth.jsonl``, both in the same directory. See
:ref:`reference-audit-trail`.

.. seealso::

   :ref:`reference-trajectory-model`
      The model the service drives: its states, command slots and buffers.

   :ref:`reference-guarded-run`
      The run lock every worker action takes, and its holder record.
