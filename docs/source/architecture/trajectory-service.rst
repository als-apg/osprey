.. _architecture-trajectory-service:

==================
Trajectory Service
==================

The trajectory service is the one process that may claim a control target for
first-turn threading. It runs as its own container, holds one connector to one
target, and serves one gated app with two doors: a **panel door** the
operator's browser uses and an **agent door** the agent's tools use. Both doors
reach it through the web terminal's panel proxy, both name the person acting
the same way, and both put every writing action through the same admission
check and the same guarded-run lock as every other write path on that target.

.. raw:: html
   :file: ../_diagrams/trajectory-service.html

Its writing actions are the first-turn threading loop --- a chain of arm,
inject, declared command steps and readout, the apply of the step computed
from its shot, a put-back of earlier values --- with a setup, a SolveConfig
update, a continuous run of chains, a reference's save, import and selection,
a Stop, and a session's open, take-over and release. A session says who
drives the target; the service fires nothing physical outside one. The exact
routes, bodies and refusal codes are in :ref:`reference-trajectory-service`;
how the loop is driven is in :doc:`/how-to/thread-the-first-turn`; the model
the service drives is in :ref:`reference-trajectory-model`.


One app, two doors
==================

The service is one FastAPI app behind the same web gate every OSPREY panel
service uses. It serves:

- ``/panel/`` --- the panel bundle;
- ``/panel/api/*`` --- the **panel door**;
- ``/api/*`` --- the **agent door**, which also refuses a request without the
  service's launch token;
- ``GET /health`` --- outside the gate; it reads the target live.

Both doors serve the loop itself: open a session, take one over, release
it, read the state, Stop, take a shot, read the pending step, and apply it or
put values back. The agent door runs only the declared default chain; the
panel door runs any ticked part of it. The panel door alone serves the rest:
setup, the SolveConfig, the continuous run and its heartbeat, the references,
the shots and their raw buffers, and the state stream.
The door a request came through is its route, never something the request
says, and an action records which door it took as its ``kind``, ``panel`` or
``agent``.

The two kinds are not equally strong evidence. ``panel`` proves an operator
credential: the panel token the agent holds cannot reach the panel door
through the terminal's proxy. ``agent`` says only that the request took the
agent door.

Every request arrives through the calling terminal's panel proxy, on one of
two hops. The operator's browser loads the panel from the terminal and calls
the panel door at ``…/trajectory/panel/api``. The agent calls the
``trajectory`` MCP server's tools, and that server calls the agent door at
``…/trajectory/api`` with the panel token. On either hop the proxy adds the
terminal's own secret, on the agent's hop the service's launch token as well,
and drops any owner the request arrives with. It adds them only for a backend
the deployment's config declares that answers on the loopback address. The
agent never holds the launch token or the secret: the terminal takes the token
out of its own environment before it starts the agent, and keeps it only for
the proxy.

The proxy bounds how long it waits to connect to either door, and sets no
bound on how long it waits for the answer. A shot, an apply or a setup answers
only when the action ends, which can take minutes; a read bound on the hop
would cut the answer off while the service still works on the action, and the
caller would see a failure for an action that went on to finish. Each caller
bounds the wait itself instead. The MCP server bounds a tool's call by the
action's own timeout, the time a Stop of it may take and the wait behind the
actions already queued, all read from the service's state. The operator has Stop, which the service answers at once on
either door. The panel's page and its static files keep the proxy's ordinary
bounds.

One declaration, one state
==========================

What the service drives is declared once, in the selected model's
measurement file: its ``trajectory`` block names the BPM devices, the
operator settings and the chain's steps (see
:doc:`/how-to/describe-your-facility`). The service, its panel and the
virtual accelerator read that declaration and nothing else; the facility's
sequence module only fires the injection
(:ref:`reference-trajectory-sequence`).

Everything the service decides is one state per target and model: the
session, the current setup, the SolveConfig and its published solve, the
pending step, the shots and the references. Both doors read the same state,
the snapshot; neither keeps its own. A take-over hands the whole of it to the
new holder, and the setup, the SolveConfig and the shots survive a restart.
An action that finds the machine moved under the state --- a setup readback
diverged, a corrector written from outside --- voids what depended on it
rather than acting on it.

Whose action it is
==================

The service names the person acting exactly as
:ref:`Whose write is it <architecture-safety-chain-owner>` describes for every
OSPREY door; that section is the rule, and this page does not restate it. For
this service the rule reads:

    *The owner is minted by the door from the credential that admitted the
    request --- a matched roster secret on the panel door, the calling
    terminal's identity on the agent door --- never from what the request
    claimed.*

Both doors read that one owner, from the name the shared web gate stamps for
the secret that admitted the request. In a multi-user deployment the two
coincide: the terminal's secret names that terminal's account. In a
single-user deployment the deployment secret names the account the project
was built under; the session's state reply shows that name as ``caller``.

A request whose credential names nobody has no owner, and every writing
action without an owner is refused. A caller that holds only the launch token
is therefore refused: the token says the request may use the agent door, not
who sent it. A name in a request body is refused as such
(``owner_in_body``); any other body key is refused too.

The approval prompt's stamp carries no owner. Its ``approver`` is a kind,
``human`` or ``dispatch_policy``, never a person, so the agent door does not
read a name from it.

What an action passes
=====================

One worker runs every writing action, one at a time, in the order the doors
submitted them. Before an action runs, the worker checks, in this order, and
the first check that refuses decides:

1. **Admission.** The owner is a real person, not ``unknown`` and not this
   service's own account. The deployment arms writes on the service's target.
   The owner's control context is readable, names the service's target, and is
   not narrowed. Each is read as it stands when the action reaches the worker,
   so an action queued behind a narrowing is refused rather than run.
2. **The session.** The caller holds the session it acts in.
3. **Reachability.** A live, short read of the model's injection status; the
   service latches nothing at startup.
4. **Limits.** A limits validator rebuilt for the target. The fail-safe one
   that stands in for a limits file that could not be read is refused. When
   none is built, the action runs with limits checking off only where the
   deployment's config states it off for the target; a posture it does not
   state, or a config that cannot be read, is refused, because a command
   fires unchecked only where the deployment says so.
5. **The run lock.** The target's guarded-run lock, taken with a record of
   whose action it is (:ref:`reference-guarded-run`). Another run on the target
   --- an agent's Python run, a ``pyaml_measure`` run --- refuses the action,
   and the lock refuses theirs while the action runs.

Inside the action, the service writes two kinds of channel. A **command**
--- a BPM's arm, the injection request, a declared command step such as
cleaning --- acts and holds no setting. It goes
out with no confirmation read-back, checked against the limits validator in
the validator's own mode: under ``optional`` a command with no limits record
is written with no limits, under ``exclusive`` it is refused, and a record's
bounds always apply. A **setting** --- a corrector an apply or a put-back
writes, an operator setting a setup writes --- is written through
the connector's checked path, held to its limits record and confirmed as that
record asks. An apply is refused whole, before its first write, when any
value it would write breaks the validator.

Take-over, release, Stop and a continuous run's heartbeat do not wait their
turn. They are decided at admission, take no lock, probe nothing and write
nothing, so each is answered while an action is in flight and an unreachable
target never blocks one. A Stop or a take-over sets the running action's stop
event, and the action ends at its next checkpoint, listing what it moved; the
service never undoes a write. A take-over also refuses the old holder's
queued and later calls. A release checks only that the caller is the holder,
so a narrowed holder or an unreachable target never keeps a session open.

Sessions live in memory. A restarted service holds no session.

Where it runs
=============

``osprey up`` deploys the service as its own container on the project image
when a profile carries a ``trajectory:`` block. A persona shows its tab, and
its agent gets the ``trajectory`` tools, when it lists ``- trajectory`` in its
web panels, as the ``control-assistant-readwrite`` and
``control-assistant-admin`` presets do. The container:

- holds one connector for its target, the deployment's baseline unless the
  profile names one;
- writes as ``osprey-trajectory`` on the control system
  (:ref:`multi-user-control-identity` covers the per-card names);
- files its audit records under ``var/audit/trajectory/``
  (:ref:`reference-audit-trail`);
- shares the guarded-run directory and the control-context tree with every
  other writer of the deployment;
- loads the facility's sequence module, read-only, from
  ``services.trajectory.sequence_dir``
  (:ref:`reference-trajectory-sequence`).

A missing limits file on a target that arms writes and checks limits, an unset
control-context tree or an unset guarded-run directory stops the service at
startup with the missing piece named. A build with no trajectory model still
starts the service: ``/health`` says so and every writing action is refused
``trajectory_unavailable``.

What it does not cover
======================

The gaps of this service's doors are listed with every other door's in
:ref:`architecture-safety-chain-owner`: the agent's panel token reaches the
agent door from a shell the persona allows, a single-user terminal running
under another account than the build account acts under the build account's
name, and an agent running on the host can read the launch token from the
deployment's ``.env``.

.. seealso::

   :ref:`reference-trajectory-service`
      The routes, the request and reply bodies, and every refusal code.

   :doc:`/how-to/thread-the-first-turn`
      Driving the loop from the agent and the panel, and what a stopped
      action leaves behind.

   :doc:`safety-chain`
      The checks every write passes, and whose write it is.

   :ref:`reference-guarded-run`
      The run lock the worker takes, and what its holder record names.
