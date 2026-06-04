==============
Event Dispatch
==============

How to turn external events (webhooks, cron ticks) into headless Osprey agent runs.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What the event dispatcher and dispatch worker do
   - How to author triggers in ``triggers.yml``
   - The two bearer tokens that guard inbound and internal traffic
   - How to run the pipeline as containers or directly from your venv
   - How to fire a trigger and watch it on the dashboard

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

The ``control-assistant`` preset ships this enabled, wired to a set of
control-system-free **tutorial triggers** so you can exercise the full pipeline
with a single ``curl``. ``osprey build`` writes ``triggers.yml``, both service
compose templates, and the ``services.{event_dispatcher,dispatch_worker}``
config into your project, and appends both to ``deployed_services``.

.. mermaid::

   flowchart LR
       E[External event] -->|POST /webhook/name| D[Event dispatcher :8020]
       D -->|allowlist + policy| W[Dispatch worker :9190]
       W -->|headless agent run| R[Result + SSE stream]
       D --- Dash[Dashboard /dashboard]

Triggers
========

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

Each webhook trigger is reachable at ``POST /webhook/<name>``. The bundled
``tutorial_triggers.yml`` defines four demos, each isolating one concept:

- ``hello-dispatch`` — anatomy of a trigger and a first successful round-trip
  (zero tools, empty payload).
- ``triage-event`` — the webhook JSON body becomes the agent's context; it
  reasons about the event with no tools.
- ``save-report`` — tool use across a short multi-turn loop, persisting a
  status report as an artifact in the worker workspace volume (via the
  workspace MCP artifact tool — the preset's memory guard blocks arbitrary
  file writes, so persistence goes through the sanctioned channel).
- ``denied-tool-demo`` — requests ``WebFetch`` to prove the worker's
  server-side denylist rejects it regardless of the trigger's allowlist.

.. note::

   The worker enforces a server-side tool **denylist** (``WebFetch``,
   ``WebSearch``, Playwright) regardless of what a trigger requests — defence in
   depth on top of the per-trigger allowlist.

.. note::

   The ``on_error: retry`` policy fires when the dispatcher **cannot reach the
   worker** (connection error or timeout) — it re-dispatches up to
   ``max_retries`` with ``backoff_sec`` between attempts. It does *not* retry a
   run that the agent itself ends in error, so firing a trigger against a
   healthy stack never exercises it; the behaviour is covered by
   ``tests/unit/dispatch/test_server_routes.py``.

Authentication
==============

Two bearer tokens live in the project ``.env``. ``osprey deploy up``
auto-generates a strong random value for each one when it is unset (and logs
where it wrote it), so a containerized deploy is secure by default — no editing
required. Set your own values in ``.env`` to override:

- ``EVENT_DISPATCHER_TOKEN`` — guards **inbound** webhook and write endpoints.
  Send it as ``Authorization: Bearer <token>``. The dispatcher fails closed
  (HTTP 503) if it is unset.
- ``DISPATCH_WORKER_TOKEN`` — guards the **dispatcher → worker** calls.

To send a request, read the generated token back from ``.env`` (e.g.
``export $(grep -E '^EVENT_DISPATCHER_TOKEN=' .env | xargs)``) so the
``$EVENT_DISPATCHER_TOKEN`` references below resolve.

Running the Pipeline
====================

.. tab-set::

   .. tab-item:: Containers

      Both services are registered in ``deployed_services``, so they come up
      with the rest of the stack:

      .. code-block:: bash

         osprey deploy up        # add --dev to bake in a local osprey checkout

      The dispatcher and worker share one image, built locally from
      ``services/event_dispatcher/Dockerfile`` on first ``osprey deploy up``
      (the first build is slow — it installs Node and the Claude Code CLI the
      worker needs). Pass ``--dev`` to install your local osprey checkout (incl.
      unreleased code) via a wheel; otherwise the image installs
      ``osprey-framework`` from PyPI. To use a prebuilt/published image instead
      of building, set the override env vars:

      .. code-block:: bash

         OSPREY_DISPATCH_IMAGE=my-registry/osprey-dispatch:dev \
         OSPREY_WORKER_IMAGE=my-registry/osprey-dispatch:dev \
           osprey deploy up

      Inside the compose network the worker is reachable as
      ``dispatch-worker-1:9190`` — the default ``dispatch_target`` in
      ``triggers.yml``. See :doc:`deploy-project` for the deploy mechanics.

   .. tab-item:: Local (no containers)

      Both services are plain Python entrypoints, so you can run them straight
      from your venv — handy for development. First repoint the worker URL in
      ``triggers.yml`` (the Docker hostname does not resolve on the host):

      .. code-block:: yaml

         dispatcher:
           dispatch_target: http://localhost:9190

      Generate the two bearer tokens once (the containerized path does this for
      you; here you set them by hand) and export them so both shells share them:

      .. code-block:: bash

         export EVENT_DISPATCHER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
         export DISPATCH_WORKER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

      Start the **worker** (it reads ``config.yml`` to inject the same provider
      auth the web server uses):

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

Firing a Trigger
================

With the dispatcher running, ``POST`` to a trigger's webhook (the JSON body is
passed to the agent as untrusted payload):

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

Watch runs stream live on the dashboard at http://localhost:8020/dashboard.

The EVENTS Panel
================

Projects built from the ``control-assistant`` preset also surface this dashboard
as an **EVENTS** tab inside the web terminal, so ``osprey web`` exposes it
without a separate browser window. The tab health-gates itself: while the
dispatcher is down it shows *unavailable* rather than a broken frame, and turns
live once the dispatcher answers ``/health``.

The panel points at ``${EVENT_DISPATCHER_URL:-http://localhost:8020}``, which
works out of the box for the host-run flow above. When the web terminal runs in
a container, repoint it at the dispatcher service:

.. code-block:: bash

   EVENT_DISPATCHER_URL=http://event-dispatcher:8020

.. seealso::

   :doc:`deploy-project`
       Container deployment mechanics for all Osprey services.

   :doc:`../cli-reference/index`
       Full ``osprey build`` and ``osprey deploy`` reference.
