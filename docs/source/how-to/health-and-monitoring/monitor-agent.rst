.. _how-to-monitor-agent:

=========================
Monitor Your OSPREY Agent
=========================

How to emit the OSPREY agent's operational telemetry — logs, metrics and traces — over
OpenTelemetry (OTLP), and optionally keep it in a store deployed alongside your
project, together with the audit ledger.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What telemetry the agent emits and how it is transported (OTLP)
   - **Phase 1:** enabling emit against any OTLP-compatible backend
     (backend-agnostic)
   - **Phase 2:** the telemetry store and gateway the presets deploy by
     default, for an all-in-one, air-gapped record
   - How each sender logs in, and how the endpoint follows where it runs
   - The full-content-capture posture and how to suppress content categories

   **Prerequisites (Phase 2 only):** Docker (or Podman) installed locally.

Overview
========

The OSPREY agent can emit its operational telemetry — structured event logs,
runtime metrics and traces — over the OpenTelemetry Protocol (OTLP). Projects built from
the bundled presets ship with telemetry **already enabled** and pointed at the
local telemetry gateway; the mechanism itself only stays off when a config has
no ``telemetry:`` block under ``claude_code:``. There are two ways to consume
it:

- **Phase 1 — any OTLP endpoint (backend-agnostic).** Point the agent at an
  OTLP collector or observability platform you already run. OSPREY only produces
  OTLP; it does not care what receives it.
- **Phase 2 — the telemetry store and gateway (the scaffolded default).** Two
  containers deployed next to your project with ``osprey up``: a database that
  holds the record, and an OpenTelemetry Collector in front of it that admits
  each sender by its own login and also reads the audit ledger files. All
  telemetry stays on the same host.

.. note::

   Traces are exported to the same endpoint as logs and metrics; the exporter
   appends ``/v1/traces`` to it. The Phase 2 gateway takes all three. For a
   Phase 1 collector that takes no traces, leave them out with ``signals`` (see
   :ref:`monitor-agent-signals`).

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
       protocol: http/protobuf                    # default; or http/json, grpc
       auth:
         token_env: OTLP_COLLECTOR_TOKEN          # variable holding the bearer token
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
     - ``generic`` (the default when unset) for an endpoint you name, or
       ``gateway`` for the Phase 2 telemetry gateway.
   * - ``endpoint``
     - OTLP base endpoint. Use the ``${VAR}`` form so the value comes from your
       ``.env`` rather than being committed to ``config.yml``.
   * - ``protocol``
     - OTLP transport. Defaults to ``http/protobuf``. The gateway takes
       ``http/protobuf`` and ``http/json``; ``grpc`` is for ``generic`` only.
   * - ``signals``
     - Which of ``metrics``, ``logs`` and ``traces`` to export. Defaults to all
       three; see :ref:`monitor-agent-signals`.
   * - ``auth.token_env``
     - The environment variable holding the collector's bearer token; the agent
       sends ``Authorization: Bearer <value>``. Holds the variable's name, never
       the token. ``generic`` only: under ``gateway`` each sender logs in with
       its own login.
   * - ``resource_attributes``
     - Attributes stamped onto every emitted record — useful for separating
       environments or agent instances in your backend.

Set the endpoint in your profile's ``.env`` — the build derives the project's
from it — then run the agent as usual:

.. code-block:: text

   # .env
   OTEL_EXPORTER_OTLP_ENDPOINT=https://otel-collector.example.com
   OTLP_COLLECTOR_TOKEN=<the collector's token>

That is all Phase 1 requires — the agent begins emitting on its next run.

On a web-terminal deploy, every ``${VAR}`` in the ``telemetry`` block that has
no ``:-default`` must be set in the project's ``.env``; ``osprey up`` refuses
the deploy and names the key and the variable. None of these values is copied
into ``.env.users``, the one file every web terminal shares, so a value the
terminals need from this block does not reach them from there.

The web terminals receive the variable ``auth.token_env`` names from the
deploy ``.env`` through their own compose environment, never ``.env.users``,
and ``osprey up`` refuses the deploy when it is unset. Every other ``${VAR}``
the block references reaches the terminals the same way. A collector that needs
another header or another auth scheme sits behind an OpenTelemetry Collector
that adds it.

.. _monitor-agent-signals:

Export without traces
---------------------

A collector that takes no traces refuses every trace the agent sends. List
the signals it does take:

.. code-block:: yaml

   claude_code:
     telemetry:
       enabled: true
       endpoint: ${OTEL_EXPORTER_OTLP_ENDPOINT}
       signals: [metrics, logs]

Each signal left out is exported as ``none`` rather than left unset, so an
``OTEL_TRACES_EXPORTER`` in your shell or in the project's ``.env`` cannot
turn it back on. An empty list or any other name stops ``osprey build``; to
export nothing, set ``enabled: false``. Built-in tool output
(``log_tool_content``, under `Content capture`_) is recorded as trace span
events, so a list without ``traces`` leaves it out of the store. The osprey
tool-call record is written either way.


Phase 2 — The telemetry store and gateway
=========================================

If you do not already run an observability stack, OSPREY deploys one of its
own: two services that keep the agent's telemetry and the audit ledger on the
deploy host, queryable with SQL.

- ``telemetry_store`` — a ClickHouse database
  (``clickhouse/clickhouse-server``) holding the database ``telemetry``.
  ``osprey up`` creates its tables, its accounts and their limits. It publishes
  its HTTP port on the host's loopback only.
- ``telemetry_gateway`` — an OpenTelemetry Collector
  (``opentelemetry-collector-contrib``). It receives OTLP over HTTP from the
  agent, refuses any sender without a valid login, stamps every record with
  the sender's name, and writes to the store with an account that can only
  insert. It also reads every ledger file under ``var/audit/`` into the table
  ``audit_ledger``.

The tables are ``otel_logs``, ``otel_traces``, the five ``otel_metrics_*``
tables (the Collector's own table names and columns, so its upstream
documentation applies) and ``audit_ledger``. Every row carries the sender in
``User``, and the session columns ``SessionId``, ``TerminalSession`` and
``ToolUseId`` are the same on both sides, so the agent's own events and its
ledger records join on them. For the ledger table, see
:ref:`audit-trail-in-the-store`.

1. Check the services are deployed
----------------------------------

Projects ship with both services declared in ``config.yml`` **and** listed
under ``deployed_services``. If your project removed them, restore them:

.. code-block:: yaml

   services:
     telemetry_store:
       path: ./services/telemetry_store
       retention_days: 14       # the store drops rows older than this
       readers:
         - name: operator       # who may query it with SQL
     telemetry_gateway:
       path: ./services/telemetry_gateway

   deployed_services:
     - telemetry_store
     - telemetry_gateway

Both publish on fixed slots above ``deployment.port_base``: the store at + 804
(``services.telemetry_store.port``) on loopback, the gateway at + 51
(``services.telemetry_gateway.port``) on ``deployment.bind_address``. The store's
container memory is ``services.telemetry_store.memory`` (default ``2g``).

2. Build and deploy
-------------------

.. code-block:: bash

   osprey build
   osprey up

There is nothing to set by hand. Every account has its secret in a file of its
own at the repo root, mode 0600, outside the ``.env`` chain:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Whose
   * - ``.env.telemetry-admin``
     - The store's own administrator, ``osprey_admin``
   * - ``.env.telemetry-ingest``
     - The gateway's insert-only account
   * - ``.env.telemetry-sender.<identity>``
     - One sender's login: a roster user, a dispatch worker, or the host
       identity (``deployment.host_identity``) for runs on the host
   * - ``.env.telemetry-senders.htpasswd``
     - What the gateway checks the logins against
   * - ``.env.telemetry-archive``
     - The archive's read-only account
   * - ``.env.telemetry-history``, ``.env.telemetry-history-gate``
     - The history page, where it is deployed
   * - ``.env.telemetry-reader.<name>``
     - One reader's key (:doc:`read-the-record`)

``osprey build`` writes every file but the reader keys, which ``osprey up``
mints. Each file reaches exactly the container that acts as that account,
and no other: a terminal holds its own login and nothing else. The web
terminals, the dispatch workers and host runs each send under their own name.
To rotate a secret, delete its file and run ``osprey build`` and ``osprey up``;
the services that hold it are recreated. ``osprey reset`` keeps the files.

Verify both services with ``osprey health``: ``telemetry_store`` checks the
store answers, and ``telemetry_gateway`` checks the gateway's export queues and
refusal counters, and lists any audit directory it does not read yet.

3. Point the agent at it
------------------------

.. code-block:: yaml

   claude_code:
     telemetry:
       enabled: true
       backend: gateway
       protocol: http/protobuf

This is what the bundled presets already ship, so a project scaffolded from one
needs no edit here. The block names **no endpoint and no credential**: under
``backend: gateway`` the endpoint follows where the sender runs, and the login
comes from the sender's own file. Setting ``endpoint`` or ``auth.token_env``
under this backend stops the build.

.. raw:: html
   :file: ../../_diagrams/otel-endpoint-context.html

.. important::

   The gateway is reached under a different address depending on where the
   sender runs:

   - **On the host** (``osprey web`` / ``osprey chat`` on your machine), at the
     address the gateway publishes: ``deployment.bind_address`` (loopback for a
     wildcard bind) and ``services.telemetry_gateway.port``.
   - **Inside a host-networked container** (every web-terminal persona
     container, a dispatch worker on the host network), at that same published
     address. The container's compose file declares it as
     ``OSPREY_OTEL_GATEWAY_HOST`` and ``OSPREY_OTEL_GATEWAY_PORT``.
   - **Inside a bridge-networked container** (the default containerized
     dispatch worker), at the compose service name and the port the gateway
     listens on inside its own container: ``OSPREY_OTEL_GATEWAY_HOST=telemetry_gateway``,
     ``OSPREY_OTEL_GATEWAY_PORT=4318``.

   A process that carries neither variable is a host run. The pair is
   declared, never sniffed, so emit works the same under Docker and Podman. If
   you run your own containerized emitter, set both to whatever reaches the
   gateway from inside that container, and give it a sender login.

On its next run the agent's logs, metrics and traces land in the store. To
query them, see :doc:`read-the-record`.

4. Know what the gateway reads
------------------------------

The gateway reads one ledger directory per identity, ``var/audit/<identity>/``,
each with its own reader, so a broken file in one directory stalls only that
identity's feed. The set of directories is fixed when ``osprey build`` renders
the gateway: a directory that appears afterwards is **not fed until the next
build**, and ``osprey health`` lists it under ``telemetry_gateway`` until then.
A ``*.jsonl`` entry that is not a regular file — a FIFO, a symlink — is
reported by the same check, and ``osprey up`` renames it to
``<name>.refused-<time>`` (kept, out of the feed) before it starts the gateway;
it refuses to start when it cannot.

Each record is stamped from the login it arrived with (``User``); any
``osprey.*`` attribute a sender sets itself is dropped. A record dated before
the store's retention window or more than an hour ahead is dropped and counted.

Content capture
===============

By default the agent captures **full content** in its telemetry — operator
prompts, agent responses, tool calls, and raw provider bodies. Because the
Phase 2 store is local and air-gapped, this full-fidelity posture is the
default: nothing leaves the host, and complete transcripts make post-incident
review far more useful.

Five independent gates control it, all defaulting **on**. Set any to ``false``
to suppress that category from emitted telemetry:

.. code-block:: yaml

   claude_code:
     telemetry:
       enabled: true
       log_user_prompts: true          # operator chat prompts
       log_assistant_responses: true   # agent replies
       log_tool_details: true          # tool names + arguments
       log_tool_content: true          # built-in tool output (Read, Bash; Edit and Write with log_tool_details)
       log_raw_api_bodies: true        # raw provider request/response bodies

``log_tool_content`` is recorded on the traces signal: with ``traces`` left out
of ``signals`` it records nothing.

``content_max_length`` sets how many UTF-16 code units one content value may
carry before Claude Code truncates it; Claude Code's own limit, 61440, applies
when it is unset. The control-assistant preset sets 262144. MCP tools, the
osprey control-system tools among them, record no ``tool.output`` event, so
their results are not in this feed.
Their full arguments and results are in the osprey tool-call record
(``audit.tool_call.enabled``, on in the control-assistant preset; see
:doc:`/reference/contracts/audit-trail`).

If you route telemetry to a shared or off-host backend (Phase 1), review these
gates and disable the categories you do not want to leave the machine.


Caveats
=======

- **Local by design.** The store publishes on the host's loopback only. People
  query it through the readers' origin of the web tier, or over an SSH tunnel
  to the deploy host (:doc:`read-the-record`); its accounts, not the network,
  decide who reads what.
- **Data volume growth is bounded by retention, not a size cap.** The store
  keeps its tables in a named container volume, which has no portable size
  limit, so it is bounded by *age*: ``services.telemetry_store.retention_days``
  (default 14) drops rows older than that, including rows already stored when
  the value is lowered. Raise it for longer history, but watch disk. The daily
  archive keeps a copy beyond it (:doc:`agent-record`).
- **The store is down, the record is not.** The gateway queues on disk and
  retries while the store is unreachable; ``osprey health`` reports the store
  as degraded, not the deployment as broken. Fix the store before restarting
  the gateway when its export queue is over 80% full: a gateway stopped with a
  full ledger queue can lose the batch it was holding.
- **The dispatch dashboard links each run to the history page** where the page
  is deployed. A dispatched job is recorded under its worker's name,
  ``dispatch-worker-<i>``, so the link opens for people with ``history: all``
  and for readers (:ref:`multi-user-history-page`).
