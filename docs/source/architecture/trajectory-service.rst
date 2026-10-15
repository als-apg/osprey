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

Its writing actions are a session's open, take-over and release. A session
says who drives the target; the service fires nothing physical outside one.
The exact routes, bodies and refusal codes are in
:ref:`reference-trajectory-service`; the model the service drives is in
:ref:`reference-trajectory-model`.

.. raw:: html

   <div class="osprey-diagram">
   <svg viewBox="0 0 1000 320" role="img" aria-labelledby="trajectory-service-title trajectory-service-desc">
     <title id="trajectory-service-title">The trajectory service and its two doors</title>
     <desc id="trajectory-service-desc">The operator's browser and the agent both reach the trajectory service through the web terminal's panel proxy. The proxy drops any owner a request claims and adds the terminal's secret, and on the agent's hop the service's launch token as well. The service's shared web gate admits on the secret and names its account as the owner. The panel door and the agent door, which also checks the launch token, both hand the owner to one admission check and one worker, which takes the target's guarded-run lock before it acts.</desc>
     <defs>
       <marker id="trajectory-service-arr" viewBox="0 0 8 6" refX="7" refY="3" markerWidth="8" markerHeight="6" orient="auto-start-reverse">
         <polygon points="0,0 8,3 0,6" fill="context-stroke"/>
       </marker>
     </defs>

     <line class="e" x1="180" y1="68" x2="246" y2="140" marker-end="url(#trajectory-service-arr)"/>
     <line class="e" x1="180" y1="252" x2="246" y2="180" marker-end="url(#trajectory-service-arr)"/>
     <line class="e" x1="430" y1="160" x2="486" y2="160" marker-end="url(#trajectory-service-arr)"/>
     <line class="e" x1="650" y1="140" x2="696" y2="78" marker-end="url(#trajectory-service-arr)"/>
     <line class="e" x1="650" y1="180" x2="696" y2="242" marker-end="url(#trajectory-service-arr)"/>
     <line class="e" x1="830" y1="78" x2="866" y2="140" marker-end="url(#trajectory-service-arr)"/>
     <line class="e" x1="830" y1="242" x2="866" y2="180" marker-end="url(#trajectory-service-arr)"/>

     <rect class="mask" x="150" y="94" width="120" height="12" rx="2"/>
     <text class="ta" x="210" y="103" text-anchor="middle">…/trajectory/panel/api</text>
     <rect class="mask" x="150" y="214" width="120" height="12" rx="2"/>
     <text class="ta" x="210" y="223" text-anchor="middle">…/trajectory/api</text>

     <rect class="nb-in" x="20" y="40" width="160" height="56" rx="6"/>
     <text class="tn" x="100" y="64" text-anchor="middle">operator's browser</text>
     <text class="ts" x="100" y="80" text-anchor="middle">operator credential</text>

     <rect class="nb-in" x="20" y="224" width="160" height="56" rx="6"/>
     <text class="tn" x="100" y="248" text-anchor="middle">the agent</text>
     <text class="ts" x="100" y="264" text-anchor="middle">panel token only</text>

     <rect class="nb" x="250" y="120" width="180" height="80" rx="6"/>
     <text class="tn" x="340" y="146" text-anchor="middle">terminal panel proxy</text>
     <text class="ts" x="340" y="164" text-anchor="middle">drops a claimed owner</text>
     <text class="ts" x="340" y="180" text-anchor="middle">adds secret · launch token</text>

     <rect class="nb-foc" x="490" y="120" width="160" height="80" rx="6"/>
     <text class="tn" x="570" y="146" text-anchor="middle">shared web gate</text>
     <text class="ts" x="570" y="164" text-anchor="middle">admits on the secret</text>
     <text class="ts" x="570" y="180" text-anchor="middle">owner = its account</text>

     <rect class="nb-foc" x="700" y="50" width="130" height="56" rx="6"/>
     <text class="tn" x="765" y="74" text-anchor="middle">panel door</text>
     <text class="ts" x="765" y="90" text-anchor="middle">/panel/api/*</text>

     <rect class="nb-foc" x="700" y="214" width="130" height="56" rx="6"/>
     <text class="tn" x="765" y="238" text-anchor="middle">agent door</text>
     <text class="ts" x="765" y="254" text-anchor="middle">/api/* · launch token</text>

     <rect class="nb-st" x="870" y="120" width="110" height="80" rx="6"/>
     <text class="tn" x="925" y="146" text-anchor="middle">admit</text>
     <text class="ts" x="925" y="164" text-anchor="middle">one worker</text>
     <text class="ts" x="925" y="180" text-anchor="middle">run lock</text>
   </svg>
   </div>

One app, two doors
==================

The service is one FastAPI app behind the same web gate every OSPREY panel
service uses. It serves:

- ``/panel/`` --- the panel bundle;
- ``/panel/api/*`` --- the **panel door**;
- ``/api/*`` --- the **agent door**, which also refuses a request without the
  service's launch token;
- ``GET /health`` --- outside the gate; it reads the target live.

Each door serves the same four routes: open a session, take one over, release
it, and read the session's state. The door a request came through is its
route, never something the request says, and an action records which door it
took as its ``kind``, ``panel`` or ``agent``.

The two kinds are not equally strong evidence. ``panel`` proves an operator
credential: the panel token the agent holds cannot reach the panel door
through the terminal's proxy. ``agent`` says only that the request took the
agent door.

Every request arrives through the calling terminal's panel proxy. On that hop
the proxy adds the terminal's own secret and, for this service, its launch
token, and drops any owner the request arrives with. It adds them only for a
backend the deployment's config declares that answers on the loopback
address. The agent never holds the launch token or the secret: the terminal
takes the token out of its own environment before it starts the agent, and
keeps it only for the proxy.

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
4. **Limits.** A limits validator rebuilt for the target. None, or the
   fail-safe one that stands in for a limits file that could not be read, is
   refused, because a command fires without a confirmation read-back only when
   a limits record says so.
5. **The run lock.** The target's guarded-run lock, taken with a record of
   whose action it is (:ref:`reference-guarded-run`). Another run on the target
   --- an agent's Python run, a ``pyaml_measure`` run --- refuses the action,
   and the lock refuses theirs while the action runs.

Take-over and release do not wait their turn. They are decided at admission,
take no lock, probe nothing and write nothing, so a take-over is answered
while an action is in flight and an unreachable target never blocks one. A
take-over stops the old holder: its queued and later calls are refused. A
release checks only that the caller is the holder, so a narrowed holder or an
unreachable target never keeps a session open.

Sessions live in memory. A restarted service holds no session.

Where it runs
=============

``osprey up`` deploys the service as its own container on the project image
when a profile carries a ``trajectory:`` block, and the build selects its panel
for the personas that select the Bluesky panel. The container:

- holds one connector for its target, the deployment's baseline unless the
  profile names one;
- writes as ``osprey-trajectory`` on the control system
  (:ref:`multi-user-control-identity` covers the per-card names);
- files its audit records under ``var/audit/trajectory/``
  (:ref:`reference-audit-trail`);
- shares the guarded-run directory and the control-context tree with every
  other writer of the deployment.

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

   :doc:`safety-chain`
      The checks every write passes, and whose write it is.

   :ref:`reference-guarded-run`
      The run lock the worker takes, and what its holder record names.
