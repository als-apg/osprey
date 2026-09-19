.. _how-to-event-dispatch:

==============
Event Dispatch
==============

How to turn external events (webhooks, cron ticks) into headless OSPREY agent runs.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What the event dispatcher and dispatch worker do
   - How to bring the pipeline up and fire your first trigger
   - How to fire a trigger from a web-terminal session, and who the job runs as
   - How to author your own triggers in ``triggers.yml``
   - How the two bearer tokens guard inbound and internal traffic

   **Prerequisites:** A project built from the ``control-assistant`` preset
   (or any profile with a ``dispatch:`` block). Docker/Podman only for the
   container path.

Overview
========

Event dispatch lets an external event start an agent run with no human at a
keyboard. It is built from two services:

- **Event dispatcher** (``python -m osprey.dispatch``, port ``10010``) — accepts
  authenticated webhook ``POST``\s (and cron ticks), matches them to a trigger,
  applies the trigger's tool allowlist and error policy, and forwards the run to
  a worker. It also serves the monitoring **dashboard**.
- **Dispatch worker** (``python -m osprey.mcp_server.dispatch_worker``, port
  ``10011``) — runs the headless agent session and streams progress back.

Both numbers are the deployment's port layout at its default base; move
``deployment.port_base`` and they move with it. See :ref:`reference-ports`.

.. raw:: html
   :file: ../../_diagrams/event-dispatch.html

The ``control-assistant`` preset ships this enabled, wired to four
control-system-free **tutorial triggers** so you can exercise the full pipeline
with a single ``curl``. ``osprey build`` writes ``triggers.yml``, both service
compose templates, and the ``services.{event_dispatcher,dispatch_worker}``
config into your project, and appends both to ``deployed_services``.

Bring It Up
===========

Both services are registered in ``deployed_services``, so they come up with the
rest of the stack. ``osprey up`` auto-generates both bearer tokens into
the profile's ``.env`` when they are unset, then derives the project's ``.env``
from it:

.. code-block:: bash

   osprey up        # add --dev to bake in a local osprey checkout

The first build is slow: both images install Node and the agent CLI the
worker runs on.

.. dropdown:: Image build & overrides
   :icon: package

   The two services use two different images, both built locally on first
   ``osprey up``:

   - the **dispatcher** gets its own small image
     (``<project>-dispatch:local``, from
     ``services/event_dispatcher/Dockerfile``);
   - the **worker** runs the full *project image* (``<project>:local`` — the
     same image :doc:`../deploy-project/project-image` describes, with your profile's
     artifacts and ``data/`` baked in), so the agent it launches sees the real
     project.

   Pass ``--dev`` to install your local osprey checkout (incl. unreleased
   code) via a wheel; otherwise the images install ``osprey-framework`` from
   PyPI. To use prebuilt/published images instead of building, set the
   override env vars — note they take *different kinds* of image: the worker
   override must be a project-style image containing ``/app/<project>``, not
   a dispatch image:

   .. code-block:: bash

      OSPREY_DISPATCH_IMAGE=my-registry/osprey-dispatch:dev \
      OSPREY_WORKER_IMAGE=my-registry/my-project:dev \
        osprey up

   Inside the compose network the worker is reachable as
   ``dispatch-worker-1:10011`` — the default ``dispatch_target`` in
   ``triggers.yml``. See :doc:`../deploy-project/index` for the deploy mechanics.

.. dropdown:: Run without containers (dev)
   :icon: terminal

   Both services are plain Python entrypoints, so you can run them straight from
   your venv — handy for development. First repoint the worker URL in
   ``triggers.yml`` (the Docker hostname does not resolve on the host):

   .. code-block:: yaml

      dispatcher:
        dispatch_target: http://localhost:10011

   Generate the two bearer tokens once (the containerized path does this for you;
   here you set them by hand) and export them so both shells share them:

   .. code-block:: bash

      export EVENT_DISPATCHER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
      export DISPATCH_WORKER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

   Start the **worker** (it reads ``config.yml`` to inject the same provider auth
   the web server uses):

   .. code-block:: bash

      OSPREY_PROJECT_DIR="$PWD" \
      DISPATCH_WORKER_TOKEN="$DISPATCH_WORKER_TOKEN" DISPATCH_WORKER_PORT=10011 \
        uv run python -m osprey.mcp_server.dispatch_worker

   Start the **dispatcher** in a second shell (re-export the same two tokens
   there first):

   .. code-block:: bash

      TRIGGERS_YML="$PWD/triggers.yml" \
      EVENT_DISPATCHER_TOKEN="$EVENT_DISPATCHER_TOKEN" DISPATCH_WORKER_TOKEN="$DISPATCH_WORKER_TOKEN" \
      FASTMCP_TRANSPORT=http FASTMCP_HOST=127.0.0.1 FASTMCP_PORT=10010 \
        uv run python -m osprey.dispatch

.. _event-dispatch-fire:

Fire a Trigger
==============

The bundled ``tutorial_triggers.yml`` defines four demos, each isolating one
concept:

- ``hello-dispatch`` — anatomy of a trigger and a first successful round-trip
  (zero tools, empty payload).
- ``triage-event`` — the webhook JSON body becomes the agent's context; it
  reasons about the event with no tools.
- ``save-report`` — tool use across a short multi-turn loop, persisting a status
  report as an artifact in the worker workspace.
- ``denied-tool-demo`` — requests ``WebFetch`` to prove the worker's server-side
  denylist rejects it regardless of the trigger's allowlist.

First read the generated token back from ``.env`` so the
``$EVENT_DISPATCHER_TOKEN`` reference resolves:

.. code-block:: bash

   export $(grep -E '^EVENT_DISPATCHER_TOKEN=' .env | xargs)

Then ``POST`` to a trigger's webhook (the JSON body is passed to the agent as
untrusted payload):

.. code-block:: bash

   curl -X POST http://localhost:10010/webhook/hello-dispatch \
     -H "Authorization: Bearer $EVENT_DISPATCHER_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{}'

To see a payload reach the agent, fire ``triage-event`` with a realistic body:

.. code-block:: bash

   curl -X POST http://localhost:10010/webhook/triage-event \
     -H "Authorization: Bearer $EVENT_DISPATCHER_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"signal":"demo:vacuum:pressure","value":4.2,"threshold":3.0,"severity":"warning"}'

Watch runs stream live on the dashboard at http://localhost:10010/dashboard, or
in the **EVENTS** tab of ``osprey web``.

The EVENTS Panel
================

Projects built from the ``control-assistant`` preset surface this dashboard as
an **EVENTS** tab inside the web terminal, so ``osprey web`` exposes it without a
separate browser window. The tab health-gates itself: while the dispatcher is
down the tab is disabled with an offline indicator rather than showing a broken
frame, and it turns live once the dispatcher answers ``/health``.

The panel points at ``http://localhost:<dispatcher_port>``, which works out of
the box for the host-run flow in *Bring It Up* and for every persona built
beside the deployment: per-user containers share the host's network, so the
dispatcher answers on their loopback too, and the build copies this entry from
the deployment's render into each persona (:doc:`/how-to/build-profiles`).

Auto-derived URL
----------------

You do not have to set ``web.panels.events.url`` by hand. Whenever a profile
lists the ``events`` panel **and** declares a ``dispatch:`` block, the build
derives the whole panel entry from ``dispatch.dispatcher_port``:

.. code-block:: yaml

   web.panels.events.url: http://localhost:<dispatcher_port>   # bare host
   web.panels.events.path: /dashboard                          # route
   web.panels.events.label: EVENTS                             # tab title
   web.panels.events.health_endpoint: /health                  # what the tab health-gates on

The ``url`` is the bare dispatcher host and ``path`` carries the ``/dashboard``
route — the web terminal composes the backend target as ``url`` + ``path``, so
baking ``/dashboard`` into ``url`` would double-prefix sub-routes. Keep them
split. Any of these keys a profile already pins the build leaves untouched, and
an explicit ``web.panels.events.url`` always wins — pin it on the **hosting**
profile when the web terminals cannot reach the dispatcher on ``localhost``;
every persona follows it, and a persona that pins a different one is refused.

.. _event-dispatch-from-a-session:

Firing a Trigger from a Session
===============================

A deployment that declares the EVENTS panel also hands the agent in each web
terminal a small ``event_dispatcher`` MCP server, so a job can be started from
the session an operator is already working in. Three of its tools read and
answer straight away — ``list_triggers``, ``trigger_status`` and
``trigger_history``. The fourth, ``manual_fire``, starts a job and asks first.

Ask for it in plain language — *"fire the save-report trigger"* — and an
approval prompt appears. It names the trigger the fire would start, and the
payload where the call carries one, so the name to check is on the prompt
rather than in the agent's message above it.

The job runs as you
-------------------

Approve, and the run is attributed to your account: the terminal's proxy stamps
your name onto the call, and the worker starts the agent under it. Every
control-system write that job makes then meets your :ref:`control-target chip
<web-terminal-session-posture>` — narrow yourself to read-only and the job you
fired refuses its first write, with the refusal you would have seen in the
terminal. The name comes from the session the call left, never from a tool
argument or a header the agent can set, so there is no way to fire on someone
else's behalf.

A job that cron or an inbound webhook started carries no name, and neither does
one whose account name the dispatcher cannot read — a name that is not a plain
account spelling is refused, and the only trace is a warning in the dispatcher's
own log. No chip narrows a job with no name: it runs with whatever writes the
deployment gives it.

Where the tools work
--------------------

The entry is rendered wherever the profile declares the EVENTS panel — which
includes a bare ``osprey chat`` session and the dispatch worker's own runs.
Neither of those has a terminal serving the panel route, so the server cannot
connect and reports itself ``failed`` or ``needs-auth`` while the session
starts. That is inert: the session comes up as usual, without the dispatcher
tools. **The dispatcher tools are reachable from a web-terminal session only.**

A job also cannot fire jobs. A trigger whose ``allowed_tools`` names any
``mcp__event_dispatcher__`` tool is refused when the triggers file loads, with
an error naming both the trigger and the tool, and the worker's server-side
denylist blocks ``manual_fire`` whatever a dispatch request asks for.

How the call is authorized
--------------------------

The agent never holds ``EVENT_DISPATCHER_TOKEN`` — it is stripped from every
agent child environment. The MCP entry points at the terminal's own panel proxy
instead, and presents the panel token the agent already carries. The proxy is
the one place in the container holding the dispatcher's bearer: it drops
whatever the caller sent, injects the bearer, and adds your account name. It
does that only for the events backend the configuration declares, and only when
that backend answers on loopback — an EVENTS panel pointed at a dashboard
elsewhere receives neither the bearer nor your name.

A shell reaches the same route
------------------------------

The panel token sits in the agent process's environment, so a persona whose
profile lifts the ``Bash`` deny — a build failure unless that profile also
declares its own ``PreToolUse`` gate covering the shell — can post to the proxy
route from a shell command and fire a job with no approval prompt. What that
skips is the prompt, not the rest: such a job still carries your name from the
proxy, still meets your chip at every write, and still runs under the worker's
denylist. Code running in a sandbox holds neither the panel token nor the
terminal's port, so it cannot make this call.

Authoring Triggers
==================

.. dropdown:: Trigger schema & error policy
   :icon: code

   Triggers live in ``triggers.yml`` at the project root. The file has a
   ``dispatcher:`` block (where the dispatcher finds its worker) and a list of
   ``triggers:``. A minimal webhook trigger:

   .. code-block:: yaml

      dispatcher:
        dispatch_target: http://dispatch-worker-1:10011  # worker URL
        max_concurrent_runs: 2
        max_queue_depth: 50

      triggers:
        - name: hello-dispatch
          source: webhook                # or "cron"
          action:
            prompt: >-
              Reply with a single sentence confirming the pipeline works.
            allowed_tools: []            # tools this run may use
            max_turns: 25                # optional; defaults to dispatch.max_turns
          on_error:                      # optional: retry if the worker is unreachable
            action: retry
            max_retries: 2
            backoff_sec: 1.0

   Each webhook trigger is reachable at ``POST /webhook/<name>``.

   ``max_concurrent_runs`` and ``max_queue_depth`` are shown with their
   defaults: leave them out and the dispatcher carries two runs at once and
   holds fifty events waiting for a slot. A build writes both keys from the
   profile's ``dispatch:`` block, which starts at the same pair.

   **Turn ceiling.** How many agentic turns one dispatched run may take is
   ``dispatch.max_turns`` in the build profile (default 25) — the third budget
   beside ``dispatch.timeout_sec`` and ``dispatch.inactivity_sec``, and the one
   about the work rather than the clock. A trigger that needs more, or less,
   than the deployment's number states its own ``max_turns:`` in ``action:``;
   a trigger that names none gets the deployment's. A trigger's own value must
   be a whole number of turns of at least one, and the dispatcher refuses the
   triggers file at load if it is not.

   **Tool denylist (defence in depth).** The worker enforces a server-side tool
   denylist regardless of what a trigger requests: ``WebFetch``, ``WebSearch``,
   the Playwright browser tools, and all shell tools (``Bash``, ``BashOutput``,
   ``KillShell``). This sits on top of the per-trigger allowlist, so a trigger
   can never widen its way to a shell or the open network.

   **Retry policy.** ``on_error: retry`` fires only when the *dispatch itself*
   fails — the worker being unreachable (connection error or timeout), returning
   an HTTP error, or rejecting the dispatcher's token — and it re-dispatches up to
   ``max_retries`` with ``backoff_sec`` between attempts. It does *not* retry a
   run that the agent itself ends in error, so firing a trigger against a healthy
   stack never exercises it; the behaviour is covered by
   ``tests/dispatch/test_server_routes.py``.

Reaching the Machine
====================

A trigger's ``source:`` decides what wakes it. ``webhook`` and ``cron`` need
nothing from the network; ``epics_ca`` monitors channels and does, and on a
site where those channels live behind a gateway the dispatcher has to be told
where the gateway is. Both halves of the pair take that from the deployment's
env chain, through one profile key:

.. code-block:: yaml

   dispatch:
     triggers: my_triggers.yml
     env: [EPICS_CA_ADDR_LIST, EPICS_CA_NAME_SERVERS]

Each name is passed through to both containers as ``NAME: ${NAME}``, so the
values live in ``.env`` / ``.env.shared`` and rotate with an edit and a restart.
The worker gets the same list as the dispatcher: a run acting on what a trigger
saw has to be able to see it too.

.. _event-dispatch-auth:

Authentication
==============

Two bearer tokens guard the stack. ``osprey up`` auto-generates a strong
random value for each when it is unset (and logs where it wrote it), so a
containerized deploy needs no token editing. A generated value is written into
the **profile's** ``.env`` and derived from there into the project's, so a
rebuild comes up on the same token. To pick your own values, set them in the
profile's ``.env`` and rebuild:

- ``EVENT_DISPATCHER_TOKEN`` — guards **inbound** webhook and write endpoints,
  and the MCP transport at ``/mcp`` that ``manual_fire`` arrives on. Send it as
  ``Authorization: Bearer <token>``.
- ``DISPATCH_WORKER_TOKEN`` — guards the **dispatcher → worker** calls.

Anything that drives the pipeline from outside holds both. A :doc:`chat bridge
<chat-bridges/index>`, for instance, fires its trigger at the dispatcher with
``EVENT_DISPATCHER_TOKEN`` and then collects the finished answer and its files
from the worker with ``DISPATCH_WORKER_TOKEN``, so a bridge that has only one of
them stalls partway through every question.

.. dropdown:: How the tokens work
   :icon: shield-lock

   The dispatcher **fails closed** if ``EVENT_DISPATCHER_TOKEN`` is unset — it
   never accepts an empty token. The webhook and dashboard routes answer HTTP
   503 while it is unset, which is how an unconfigured dispatcher announces
   itself, and 401 for anything else, a missing header included. The MCP
   transport at
   ``/mcp`` answers 401 in both cases: a bearer check there has one way to
   refuse, so it cannot draw the distinction the routes draw. The dashboard
   *read* endpoints (run
   feed, trigger list, state, SSE stream) are gated by the same token: the
   in-terminal EVENTS tab injects it server-side so the browser never holds it,
   while the standalone dashboard receives it via a one-time URL-fragment
   handoff.

   To call the API by hand, read the generated token back from ``.env`` (as in
   :ref:`Fire a Trigger <event-dispatch-fire>` above):

   .. code-block:: bash

      export $(grep -E '^EVENT_DISPATCHER_TOKEN=' .env | xargs)

.. seealso::

   :doc:`../deploy-project/index`
       Container deployment mechanics for all Osprey services.

   :doc:`/reference/cli`
       Full ``osprey build`` and lifecycle-verb reference.
