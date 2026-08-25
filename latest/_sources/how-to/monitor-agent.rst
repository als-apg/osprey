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
   - The store's two identities — the browser login and the telemetry ingest
     account — and what each one can read
   - Rotation, data-volume, and restart caveats

   **Prerequisites (Phase 2 only):** Docker (or Podman) installed locally.

Overview
========

The OSPREY agent can emit its operational telemetry — structured event logs and
runtime metrics — over the OpenTelemetry Protocol (OTLP). Projects built from
the bundled presets ship with telemetry **already enabled** and pointed at the
local OpenObserve store; the mechanism itself only stays off when a config has
no ``telemetry:`` block under ``claude_code:``. There are two ways to consume
it:

- **Phase 1 — any OTLP endpoint (backend-agnostic).** Point the agent at an
  OTLP collector or observability platform you already run. OSPREY only produces
  OTLP; it does not care what receives it.
- **Phase 2 — the local OpenObserve add-on (the scaffolded default).** A
  single-binary OpenObserve store deployed next to your project with
  ``osprey up``; all telemetry stays on the same host. This is the
  turn-key option when you have no existing observability stack.

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
     - OTLP transport. Defaults to ``http/protobuf``. ``grpc`` requires an
       explicit ``endpoint``: it is refused against the auto-derived
       ``openobserve`` endpoint, which is HTTP-only.
   * - ``headers``
     - Extra OTLP headers (for example, routing or auth headers your backend
       requires).
   * - ``resource_attributes``
     - Attributes stamped onto every emitted record — useful for separating
       environments or agent instances in your backend.

Set the endpoint in your profile's ``.env`` — the build derives the project's
from it — then run the agent as usual:

.. code-block:: bash

   # .env
   OTEL_EXPORTER_OTLP_ENDPOINT=https://otel-collector.example.com

That is all Phase 1 requires — the agent begins emitting on its next run.

Phase 2 — The local OpenObserve add-on
======================================

If you do not already run an observability stack, OSPREY ships an
`OpenObserve <https://openobserve.ai/>`_ service: a single binary that ingests
OTLP directly and serves a browser UI, with no external dependencies. Everything
stays on the deploy host.

1. Check the service is enabled
-------------------------------

Projects ship with the ``openobserve`` service already declared in
``config.yml`` **and** listed under ``deployed_services`` — it deploys by
default. If your project removed it, restore it:

.. code-block:: yaml

   services:
     openobserve:
       path: ./services/openobserve
       port: 5080          # host port for the UI + OTLP ingest

   deployed_services:
     - openobserve

2. Know the store's two identities
----------------------------------

The store carries two separate identities, and they do different jobs:

.. list-table::
   :header-rows: 1
   :widths: 16 40 44

   * - Identity
     - ``.env`` variables
     - What it is for
   * - **Root**
     - ``ZO_ROOT_USER_EMAIL``, ``ZO_ROOT_USER_PASSWORD``
     - Initializes the store on its first start, and logs *you* into the
       browser UI. It never leaves the deploy host: nothing on the wire
       authenticates as root.
   * - **Ingest**
     - ``ZO_INGEST_USER_EMAIL``, ``ZO_INGEST_SA_TOKEN``
     - The account the agent authenticates as when it ships telemetry. This is
       the credential that travels — into the rendered ``config.yml``, and into
       web-terminal environments.

You do not have to set either one. ``osprey up`` writes whatever is missing
into the project's ``.env``, so the answer to "what do I log in with" is always
that file:

.. code-block:: bash

   grep ZO_ ~/my-project/.env

**Root** is written before the store starts, because the store reads it to
initialize itself. The minted password is short on purpose: you read it off a
terminal and type it into a browser login form, so it is 12 characters drawn
from an alphabet with the easily-misread characters (``l I 1 O 0``) removed.
Every character still comes from a cryptographic random source, so it is short
to type without being easy to guess. Set the pair yourself, in the project's
``.env``, if you want specific values:

.. code-block:: bash

   # .env
   ZO_ROOT_USER_EMAIL=you@example.com
   ZO_ROOT_USER_PASSWORD=choose-a-strong-password

**Ingest** cannot work that way, and this is the asymmetry worth remembering:
its secret is issued by the *store*, not chosen by OSPREY, so it cannot exist
before the store is running. The deploy creates the account and reads its token
back — see step 3. Only the account *name* is written up front, and you may set
it to an address of your choosing; the token half is never yours to write.

.. warning::

   **The ingest identity can read everything the store holds.** It is a real
   OpenObserve service account and it is genuinely restricted — it cannot
   create users and it cannot delete them, so a leaked ingest token cannot mint
   itself a way back in or lock you out. But it is write-**plus**-read: it can
   search every log and metric already in the store, which means the agent's
   full conversation transcripts, and it can list the store's user roster.
   OpenObserve has no ingest-only role in any edition, so this is as narrow as
   the shipped identity gets. Guard the ingest token as closely as the
   telemetry itself. What it does buy you is that the root password stays on
   the deploy host and never reaches a config file, a container environment, or
   the network.

.. important::

   **Publishing this store beyond** ``localhost`` **exposes agent transcripts.**
   It holds full agent conversation transcripts, and both identities can read
   them. Reach it over an SSH tunnel, or put it behind TLS, rather than binding
   it to every interface. The minted password is a per-deploy random value, so
   it stays usable wherever the store is published — what exposure changes is
   who can reach the login, not how strong the password is.

3. Deploy it
------------

.. code-block:: bash

   osprey up        # brings up openobserve alongside your other services

``osprey up`` starts the store first, then provisions the ingest identity
against it and saves the token the store issues into the project's ``.env``.
There is nothing to set by hand. The run reports what it did under the
``openobserve`` group:

.. code-block:: text

   telemetry store started
   ingest identity verified        # the token already on file works
   ingest identity provisioned     # the deploy had to create, harvest, or reissue one

Only a run that *writes* says more, naming which of the three happened:

.. code-block:: text

   created the telemetry ingest account and saved its token → .env
   read the telemetry ingest token back from the store → .env
   the telemetry ingest token was refused, so a new one was issued → .env

A verified token is silent — no extra lines means nothing needed changing.

.. note::

   On a deployment that has never been started, ``ZO_INGEST_SA_TOKEN`` is
   simply absent from ``.env`` until this step runs. The deploy's preflight
   checks **defer** that one variable rather than refusing to start, because it
   is the one credential an operator cannot supply. For the same reason,
   ``osprey build`` against a never-started deployment warns that
   ``ZO_INGEST_SA_TOKEN`` is unresolved and leaves the telemetry auth header
   out; that warning is expected, and the value is resolved again when the
   agent starts.

If the store is unreachable, or the store refuses the root credential, the
deploy warns with a named remedy and carries on — telemetry is the only thing
affected, and every other service in the deployment comes up as usual. The
remedy is almost always to run ``osprey up`` again once the store is running.

The UI is then available at ``http://localhost:5080`` (log in with the **root**
credentials). Verify the service is recognized with ``osprey health``.

4. Point the agent at it
------------------------

Wire the ``telemetry:`` block to ``backend: openobserve`` and name the **ingest**
identity in the ``openobserve:`` sub-block — never the root account. **Do not
set** ``endpoint`` for this backend — omit it and the agent derives the OTLP
ingest URL automatically per network context:

.. code-block:: yaml

   claude_code:
     telemetry:
       enabled: true
       backend: openobserve
       # No `endpoint:` here — for the openobserve backend it is derived
       # automatically (see the note below).
       protocol: http/protobuf
       openobserve:
         user: ${ZO_INGEST_USER_EMAIL:-ingest@example.com}
         password: ${ZO_INGEST_SA_TOKEN}
         org: default

This is what the bundled presets already ship, so a project scaffolded from one
needs no edit here. The token reference deliberately carries **no** ``:-``
fallback: a default value would be a credential the store never issued, and
telemetry would fail with a plausible-looking secret on file instead of an
obviously missing one. The account name does carry a fallback, because it is a
name rather than a secret.

.. raw:: html
   :file: ../_diagrams/otel-endpoint-context.html

.. important::

   For ``backend: openobserve`` the OTLP endpoint is **auto-derived** and you
   should not hardcode it. The derived form is ``http://<host>:<port>/api/<org>``,
   with both halves chosen for the running context:

   - **On the host** (``osprey web`` / ``osprey query`` on your machine) the host
     is ``localhost`` and the port is the one the store **publishes** —
     ``services.openobserve.port``, so moving the store moves the exporter
     with it.
   - **Inside a host-networked container** (every web-terminal persona
     container, a dispatch worker on the host network) the container's
     loopback is the host's, so the address is the same published one. The
     framework's compose files declare the host explicitly
     (``OSPREY_OTEL_OPENOBSERVE_HOST=127.0.0.1`` or ``localhost``), because
     the emitter's own container detection would otherwise pick the compose
     DNS name, which resolves to nothing there.
   - **Inside a bridge-networked container** (the default containerized
     dispatch worker) the store is reachable only by its compose service DNS
     name, ``openobserve``, and on the port it **listens** on inside its own
     container (5080) rather than the published one. The dispatch-worker
     service declares both, ``OSPREY_OTEL_OPENOBSERVE_HOST=openobserve`` and
     ``OSPREY_OTEL_OPENOBSERVE_PORT=5080``. That explicit declaration — rather
     than sniffing the container runtime — is what makes emit work identically
     under Docker and Podman.

   If you run your own containerized emitter, set
   ``OSPREY_OTEL_OPENOBSERVE_HOST`` to whatever host reaches OpenObserve from
   inside that container — the compose service name on a bridge network
   (with ``OSPREY_OTEL_OPENOBSERVE_PORT`` naming the listen port), or
   ``localhost`` when the container uses host networking (``network_mode: host``),
   where the compose DNS name would not resolve and the published port
   applies. Leave both unset on a plain host run. Hardcoding ``endpoint:``
   instead would point every context at the same address and silently drop
   records from the ones it doesn't fit.

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

Upgrading an existing deployment
================================

Deployments made before the ingest account existed pointed their telemetry at
the root credentials. Rebuilding the project moves ``config.yml`` onto the
ingest account, and the next ``osprey up`` creates the account and saves its
token — both automatic.

One file is deliberately left alone: ``.env.users``, the environment your web
terminals run with. It is never regenerated once it exists, which is exactly
what keeps a file you have edited by hand intact. So if your deployment serves
web terminals that emit telemetry, the deploy prints an advisory naming
``ZO_INGEST_USER_EMAIL`` as missing from that file, and the fix is a write of
your own: **append** the variable.

.. code-block:: bash

   # use the same value the project's .env has
   grep '^ZO_INGEST_USER_EMAIL=' ~/my-project/.env >> ~/my-project/.env.users

A later assignment wins in an environment file, so appending disturbs nothing
already in it. ``osprey users env --output .env.users`` re-renders the file
instead — that **replaces it whole**, so reach for it only if nothing in the
file was added by hand. Do not delete the file to regenerate it; that discards
anything you added.

Until the variable is present, those terminals authenticate as whatever the
reference falls back to inside the container, and the store rejects their
records.

Caveats
=======

- **One administrator.** The OpenObserve add-on provisions a single root
  account from ``ZO_ROOT_USER_EMAIL`` / ``ZO_ROOT_USER_PASSWORD``. It is
  intended for a single operator or a small trusted team on the deploy host,
  not for multi-tenant access control.
- **Rotating the ingest token is delete-and-recreate.** OpenObserve cannot
  reissue a service account's token in place, so rotation means removing the
  account and creating it again. You only do the first half: delete the ingest
  account in the store's UI, and the next ``osprey up`` finds the token on file
  no longer works, issues a new one, and writes it back to ``.env``. A dead
  token in ``.env`` heals itself on the next start rather than needing repair.
  Every start checks the token it has before touching anything, so a working
  one is never rotated out from under something else that uses it.
- **Recreating the data volume destroys the ingest identity.** The account
  lives inside the ``openobserve_data`` volume; its token lives in ``.env``.
  Remove that volume and you are left with a credential on file for an account
  the store no longer has, and telemetry is refused. The next ``osprey up``
  detects exactly this and provisions a fresh identity, so no manual repair is
  needed. The same volume also holds the root password the store was first
  initialized with — a volume kept from an earlier deploy beside a freshly
  minted ``ZO_ROOT_USER_PASSWORD`` is reported with its own remedy.
- **``osprey restart`` provisions, but cannot deliver.** A restart re-runs the
  provisioning, so the store and ``.env`` agree afterwards, but a restarted
  container keeps the definition it already had — a token written during a
  restart reaches the containers at the next ``osprey up``.
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
