==============
Event Dispatch
==============

How to turn external events (webhooks, cron ticks) into headless Osprey agent runs.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What the event dispatcher and dispatch worker do
   - How to bring the pipeline up and fire your first trigger
   - How to author your own triggers in ``triggers.yml``
   - How the two bearer tokens guard inbound and internal traffic

   **Prerequisites:** A project built from the ``control-assistant`` preset
   (or any profile with a ``dispatch:`` block). Docker/Podman only for the
   container path.

Overview
========

Event dispatch lets an external event start an agent run with no human at a
keyboard. It is built from two services:

- **Event dispatcher** (``python -m osprey.dispatch``, port ``8020``) — accepts
  authenticated webhook ``POST``\s (and cron ticks), matches them to a trigger,
  applies the trigger's tool allowlist and error policy, and forwards the run to
  a worker. It also serves the monitoring **dashboard**.
- **Dispatch worker** (``python -m osprey.mcp_server.dispatch_worker``, port
  ``9190``) — runs the headless agent session and streams progress back.

.. mermaid::

   flowchart LR
       E[External event] -->|POST /webhook/name| D[Event dispatcher :8020]
       D -->|allowlist + policy| W[Dispatch worker :9190]
       W -->|headless agent run| R[Result + SSE stream]
       D --- Dash[Dashboard /dashboard]

The ``control-assistant`` preset ships this enabled, wired to four
control-system-free **tutorial triggers** so you can exercise the full pipeline
with a single ``curl``. ``osprey build`` writes ``triggers.yml``, both service
compose templates, and the ``services.{event_dispatcher,dispatch_worker}``
config into your project, and appends both to ``deployed_services``.

Bring It Up
===========

Both services are registered in ``deployed_services``, so they come up with the
rest of the stack. ``osprey deploy up`` auto-generates both bearer tokens into
``.env`` when they are unset:

.. code-block:: bash

   osprey deploy up        # add --dev to bake in a local osprey checkout

The first build is slow: both images install Node and the agent CLI the
worker runs on.

.. dropdown:: Image build & overrides
   :icon: package

   The two services use two different images, both built locally on first
   ``osprey deploy up``:

   - the **dispatcher** gets its own small image
     (``<project>-dispatch:local``, from
     ``services/event_dispatcher/Dockerfile``);
   - the **worker** runs the full *project image* (``<project>:local`` — the
     same image :doc:`containerize-project` describes, with your overlays and
     ``data/`` baked in), so the agent it launches sees the real project.

   Pass ``--dev`` to install your local osprey checkout (incl. unreleased
   code) via a wheel; otherwise the images install ``osprey-framework`` from
   PyPI. To use prebuilt/published images instead of building, set the
   override env vars — note they take *different kinds* of image: the worker
   override must be a project-style image containing ``/app/<project>``, not
   a dispatch image:

   .. code-block:: bash

      OSPREY_DISPATCH_IMAGE=my-registry/osprey-dispatch:dev \
      OSPREY_WORKER_IMAGE=my-registry/my-project:dev \
        osprey deploy up

   Inside the compose network the worker is reachable as
   ``dispatch-worker-1:9190`` — the default ``dispatch_target`` in
   ``triggers.yml``. See :doc:`deploy-project` for the deploy mechanics.

.. dropdown:: Run without containers (dev)
   :icon: terminal

   Both services are plain Python entrypoints, so you can run them straight from
   your venv — handy for development. First repoint the worker URL in
   ``triggers.yml`` (the Docker hostname does not resolve on the host):

   .. code-block:: yaml

      dispatcher:
        dispatch_target: http://localhost:9190

   Generate the two bearer tokens once (the containerized path does this for you;
   here you set them by hand) and export them so both shells share them:

   .. code-block:: bash

      export EVENT_DISPATCHER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
      export DISPATCH_WORKER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

   Start the **worker** (it reads ``config.yml`` to inject the same provider auth
   the web server uses):

   .. code-block:: bash

      OSPREY_PROJECT_DIR="$PWD" \
      DISPATCH_WORKER_TOKEN="$DISPATCH_WORKER_TOKEN" DISPATCH_WORKER_PORT=9190 \
        uv run python -m osprey.mcp_server.dispatch_worker

   Start the **dispatcher** in a second shell (re-export the same two tokens
   there first):

   .. code-block:: bash

      TRIGGERS_YML="$PWD/triggers.yml" \
      EVENT_DISPATCHER_TOKEN="$EVENT_DISPATCHER_TOKEN" DISPATCH_WORKER_TOKEN="$DISPATCH_WORKER_TOKEN" \
      FASTMCP_TRANSPORT=http FASTMCP_HOST=127.0.0.1 FASTMCP_PORT=8020 \
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

   curl -X POST http://localhost:8020/webhook/hello-dispatch \
     -H "Authorization: Bearer $EVENT_DISPATCHER_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{}'

To see a payload reach the agent, fire ``triage-event`` with a realistic body:

.. code-block:: bash

   curl -X POST http://localhost:8020/webhook/triage-event \
     -H "Authorization: Bearer $EVENT_DISPATCHER_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"signal":"demo:vacuum:pressure","value":4.2,"threshold":3.0,"severity":"warning"}'

Watch runs stream live on the dashboard at http://localhost:8020/dashboard, or
in the **EVENTS** tab of ``osprey web``.

The EVENTS Panel
================

Projects built from the ``control-assistant`` preset surface this dashboard as
an **EVENTS** tab inside the web terminal, so ``osprey web`` exposes it without a
separate browser window. The tab health-gates itself: while the dispatcher is
down the tab is disabled with an offline indicator rather than showing a broken
frame, and it turns live once the dispatcher answers ``/health``.

The panel points at ``${EVENT_DISPATCHER_URL:-http://localhost:8020}``, which
works out of the box for the host-run flow above. When the web terminal runs in
a container, repoint it at the dispatcher service:

.. code-block:: bash

   EVENT_DISPATCHER_URL=http://event-dispatcher:8020

Auto-derived URL
----------------

You do not have to set ``web.panels.events.url`` by hand. Whenever a profile
lists the ``events`` panel **and** declares a ``dispatch:`` block, the build
derives the panel's route from ``dispatch.dispatcher_port``:

.. code-block:: yaml

   web.panels.events.url: http://localhost:<dispatcher_port>   # bare host
   web.panels.events.path: /dashboard                          # route

The ``url`` is the bare dispatcher host and ``path`` carries the ``/dashboard``
route — the web terminal composes the backend target as ``url`` + ``path``, so
baking ``/dashboard`` into ``url`` would double-prefix sub-routes. Keep them
split. If a profile already pins ``web.panels.events.path`` the build leaves it
untouched, and an explicit ``web.panels.events.url`` override always wins (use it
for remote/containerized terminals that cannot reach ``localhost``).

Authoring Triggers
==================

.. dropdown:: Trigger schema & error policy
   :icon: code

   Triggers live in ``triggers.yml`` at the project root. The file has a
   ``dispatcher:`` block (where the dispatcher finds its worker) and a list of
   ``triggers:``. A minimal webhook trigger:

   .. code-block:: yaml

      dispatcher:
        dispatch_target: http://dispatch-worker-1:9190   # worker URL
        max_concurrent_runs: 2
        max_queue_depth: 50

      triggers:
        - name: hello-dispatch
          source: webhook                # or "cron"
          action:
            prompt: >-
              Reply with a single sentence confirming the pipeline works.
            allowed_tools: []            # tools this run may use
          on_error:                      # optional: retry if the worker is unreachable
            action: retry
            max_retries: 2
            backoff_sec: 1.0

   Each webhook trigger is reachable at ``POST /webhook/<name>``.

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
   ``tests/unit/dispatch/test_server_routes.py``.

Authentication
==============

Two bearer tokens live in the project ``.env``. ``osprey deploy up``
auto-generates a strong random value for each when it is unset (and logs where
it wrote it), so a containerized deploy needs no token editing. Set your own
values in ``.env`` to override:

- ``EVENT_DISPATCHER_TOKEN`` — guards **inbound** webhook and write endpoints.
  Send it as ``Authorization: Bearer <token>``.
- ``DISPATCH_WORKER_TOKEN`` — guards the **dispatcher → worker** calls.

.. dropdown:: How the tokens work
   :icon: shield-lock

   The dispatcher **fails closed** (HTTP 503) if ``EVENT_DISPATCHER_TOKEN`` is
   unset — it never accepts an empty token. The dashboard *read* endpoints (run
   feed, trigger list, state, SSE stream) are gated by the same token: the
   in-terminal EVENTS tab injects it server-side so the browser never holds it,
   while the standalone dashboard receives it via a one-time URL-fragment
   handoff.

   To call the API by hand, read the generated token back from ``.env`` (as in
   :ref:`Fire a Trigger <event-dispatch-fire>` above):

   .. code-block:: bash

      export $(grep -E '^EVENT_DISPATCHER_TOKEN=' .env | xargs)

.. seealso::

   :doc:`deploy-project`
       Container deployment mechanics for all Osprey services.

   :doc:`../cli-reference/index`
       Full ``osprey build`` and ``osprey deploy`` reference.
