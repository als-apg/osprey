=========================
Monitor Your OSPREY Agent
=========================

How to emit the OSPREY agent's operational telemetry — logs and metrics — over
OpenTelemetry (OTLP), and optionally view it in a self-hosted store deployed
alongside your project.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What telemetry the agent emits and how it is transported (OTLP)
   - **Phase 1:** enabling emit against any OTLP-compatible backend
     (backend-agnostic)
   - **Phase 2:** the opt-in local OpenObserve add-on for an all-in-one,
     air-gapped store
   - The full-content-capture posture and how to suppress content categories
   - Single-admin and data-volume caveats

   **Prerequisites (Phase 2 only):** Docker (or Podman) installed locally.

Overview
========

The OSPREY agent can emit its operational telemetry — structured event logs and
runtime metrics — over the OpenTelemetry Protocol (OTLP). Emit is **off by
default**; you turn it on with a ``telemetry:`` block under ``claude_code:`` in
``config.yml``. There are two ways to consume it:

- **Phase 1 — any OTLP endpoint (backend-agnostic).** Point the agent at an
  OTLP collector or observability platform you already run. OSPREY only produces
  OTLP; it does not care what receives it.
- **Phase 2 — the local OpenObserve add-on (opt-in).** Deploy a single-binary
  OpenObserve store next to your project with ``osprey deploy up`` and keep all
  telemetry on the same host. This is the turn-key option when you have no
  existing observability stack.

.. note::

   Only logs and metrics are wired. Distributed **tracing** (and the
   ``OTEL_LOG_TOOL_CONTENT`` toggle) is intentionally left out of this
   configuration surface.

Phase 1 — Emit to any OTLP endpoint
===================================

Add a ``telemetry:`` block under ``claude_code:`` in your project's
``config.yml``. The minimal backend-agnostic form only needs to be enabled and
pointed at an endpoint:

.. code-block:: yaml

   claude_code:
     telemetry:
       enabled: true
       endpoint: ${OTEL_EXPORTER_OTLP_ENDPOINT}   # your OTLP collector
       protocol: http/protobuf                    # default; or grpc
       resource_attributes:                       # attached to every record
         service.name: osprey-agent
         deployment.environment: dev

Keys:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Key
     - Meaning
   * - ``enabled``
     - Master switch. ``false`` (the default) emits nothing.
   * - ``backend``
     - Backend hint — ``generic`` for a plain OTLP endpoint, ``openobserve``
       for the Phase 2 add-on.
   * - ``endpoint``
     - OTLP base endpoint. Use the ``${VAR}`` form so the value comes from your
       ``.env`` rather than being committed to ``config.yml``.
   * - ``protocol``
     - OTLP transport. Defaults to ``http/protobuf``; set ``grpc`` if your
       collector prefers it.
   * - ``headers``
     - Extra OTLP headers (for example, routing or auth headers your backend
       requires).
   * - ``resource_attributes``
     - Attributes stamped onto every emitted record — useful for separating
       environments or agent instances in your backend.

Set the endpoint in your project ``.env`` and run the agent as usual:

.. code-block:: bash

   # .env
   OTEL_EXPORTER_OTLP_ENDPOINT=https://otel-collector.example.com

That is all Phase 1 requires — the agent begins emitting on its next run.

Phase 2 — The local OpenObserve add-on
======================================

If you do not already run an observability stack, OSPREY ships an opt-in
`OpenObserve <https://openobserve.ai/>`_ service: a single binary that ingests
OTLP directly and serves a browser UI, with no external dependencies. Everything
stays on the deploy host.

1. Enable the service
---------------------

The ``openobserve`` service is declared in ``config.yml`` but left out of
``deployed_services`` so it stays off until you opt in. Add it to the list:

.. code-block:: yaml

   services:
     openobserve:
       path: ./services/openobserve
       port: 5080          # host port for the UI + OTLP ingest

   deployed_services:
     - openobserve

2. Set the admin credentials
----------------------------

The OpenObserve root account doubles as the OTLP ingest credential, so set both
in your project ``.env``. The same two variables configure the container **and**
authenticate the agent's OTLP push — one source of truth:

.. code-block:: bash

   # .env
   ZO_ROOT_USER_EMAIL=you@example.com
   ZO_ROOT_USER_PASSWORD=choose-a-strong-password

3. Deploy it
------------

.. code-block:: bash

   osprey deploy up        # brings up openobserve alongside your other services

The UI is then available at ``http://localhost:5080`` (log in with the
credentials above). Verify the service is recognized with ``osprey health``.

4. Point the agent at it
------------------------

Wire the ``telemetry:`` block to ``backend: openobserve`` and supply the
OpenObserve auth in the ``openobserve:`` sub-block. **Do not set** ``endpoint``
for this backend — omit it and the agent derives the OTLP ingest URL
automatically per network context:

.. code-block:: yaml

   claude_code:
     telemetry:
       enabled: true
       backend: openobserve
       # No `endpoint:` here — for the openobserve backend it is derived
       # automatically (see the note below).
       protocol: http/protobuf
       openobserve:
         user: ${ZO_ROOT_USER_EMAIL}
         password: ${ZO_ROOT_USER_PASSWORD}
         org: default

.. important::

   For ``backend: openobserve`` the OTLP endpoint is **auto-derived** and you
   should not hardcode it. The derived form is ``http://<host>:5080/api/<org>``,
   where ``<host>`` is chosen for the running context:

   - **On the host** (``osprey web`` / ``osprey query`` on your machine) the host
     is ``localhost`` — the store's published port.
   - **Inside a bridge-networked container** (for example the containerized
     dispatch worker) the store is reachable only by its compose service DNS
     name, ``openobserve``. The framework's dispatch-worker service declares this
     by setting ``OSPREY_OTEL_OPENOBSERVE_HOST=openobserve`` in its compose
     environment. That explicit declaration — rather than sniffing the container
     runtime — is what makes emit work identically under Docker and Podman.

   If you run your own containerized emitter, set
   ``OSPREY_OTEL_OPENOBSERVE_HOST`` to whatever host reaches OpenObserve from
   inside that container: the compose service name on a bridge network, or
   ``localhost`` when the container uses host networking (``network_mode: host``),
   where the compose DNS name would not resolve. Leave it unset on a plain host
   run. Hardcoding ``endpoint:`` instead would point every context at the same
   host and silently drop records from the ones it doesn't fit.

On its next run the agent emits to OpenObserve, and its logs and metrics appear
in the UI.

Content capture
===============

By default the agent captures **full content** in its telemetry — operator
prompts, agent responses, tool calls, and raw provider bodies. Because the
Phase 2 store is local and air-gapped, this full-fidelity posture is the
default: nothing leaves the host, and complete transcripts make post-incident
review far more useful.

Four independent gates control it, all defaulting **on**. Set any to ``false``
to suppress that category from emitted telemetry:

.. code-block:: yaml

   claude_code:
     telemetry:
       enabled: true
       log_user_prompts: true          # operator chat prompts
       log_assistant_responses: true   # agent replies
       log_tool_details: true          # tool names + arguments
       log_raw_api_bodies: true        # raw provider request/response bodies

If you route telemetry to a shared or off-host backend (Phase 1), review these
gates and disable the categories you do not want to leave the machine.

Caveats
=======

- **Single admin.** The OpenObserve add-on provisions one root account from
  ``ZO_ROOT_USER_EMAIL`` / ``ZO_ROOT_USER_PASSWORD``. It is intended for a
  single operator or a small trusted team on the deploy host, not for
  multi-tenant access control.
- **Data volume growth is bounded by retention, not a size cap.** Ingested
  telemetry persists in a named container volume, which has no portable hard
  size limit — so the store is bounded by *age*: ``services.openobserve.retention_days``
  (default 14) sets ``ZO_COMPACT_DATA_RETENTION_DAYS`` in the container, dropping
  telemetry older than N days (OpenObserve's floor is 3 days). Raise it for
  longer history, but watch disk. Removing the volume still discards all history;
  back it up if you need retention across redeploys.
- **Health.** ``osprey health`` reports the store's ``/healthz`` readiness (a
  running container is not necessarily ready), the effective retention (warning
  if below OpenObserve's floor of 3 days), and the percentage-full of the disk
  the volume grows into.
- **Local by design.** The service binds to localhost by default. Do not expose
  port 5080 beyond the host without putting authentication and transport
  security in front of it.
