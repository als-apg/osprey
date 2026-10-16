.. _how-to-read-the-record:

===============
Read the Record
===============

The telemetry store holds what the agent emitted and every line of the audit
ledger, as tables in the database ``telemetry``. People read it with SQL, under
an account of their own: a **reader**. This page sets readers up, says what each
one can and cannot see, and lists the limits every query runs under.

For what the store holds and how it is fed, see :doc:`monitor-agent`; for a
ready-made query that puts one session back together, see
:doc:`reconstruct-session`.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - How to name a reader, hand over its key, rotate it and withdraw it
   - Where a reader opens the store's query page, and how a script queries
   - The limits every reader query runs under, and the one shape they miss
   - What the record can and cannot vouch for

Name the readers
================

Readers are listed in the config. Each entry is one account:

.. code-block:: yaml

   services:
     telemetry_store:
       readers:
         - name: carol                 # a roster user: same name, same person
         - name: dana
           oidc_subject: 7f3c9e1a-...  # recognised by sign-on in the browser
           tables: [audit_ledger]      # default: otel_logs, otel_traces, audit_ledger
           days_back: 3                # default: retention_days

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Meaning
   * - ``name``
     - The account name. It may not be one of the store's own account names
       (``default``, ``osprey_admin``, ``ingest``, ``archive``, ``history``,
       and the role, profile and quota names) or start with ``view_``.
   * - ``oidc_subject``
     - The sign-on subject that is this reader in a browser, when the web
       terminals sign people on with ``oidc``. A reader with the same name as a
       roster user is that user already.
   * - ``tables``
     - The tables the reader may read, from ``otel_logs``, ``otel_traces``,
       ``audit_ledger`` and the five ``otel_metrics_*`` tables.
   * - ``days_back``
     - How far back the reader sees. Rows older than that are not visible to
       this account at all; it is a scope, not a limit.

The presets declare one reader: the deployment's admin roster user (``carol``)
in ``control-assistant``, and ``operator`` where a preset has no roster.

Hand over the key
-----------------

``osprey up`` mints each new reader's key into ``.env.telemetry-reader.<name>``
at the repo root, mode 0600, one line ``password=<key>``. The deploy prints the
file's path and the reader's query page, never the key. Read the file and give
the key to the person out of band:

.. code-block:: bash

   sed -n 's/^password=//p' .env.telemetry-reader.dana

- **Rotate** a key: delete the file and run ``osprey up``. The old key stops
  working and a new file is written.
- **Withdraw** a reader: delete the entry and run ``osprey up``. The account,
  its row policies and its browser view are dropped, and the key file is set
  aside as ``.env.telemetry-reader.<name>.withdrawn-<date>``, so re-adding the
  name later mints a fresh key.
- **Narrow** a reader: change ``tables`` or ``days_back`` and run
  ``osprey up``. A table taken away is refused on the next query.

``osprey up`` owns the accounts it made and nothing else: a store user of the
same name that ``osprey up`` did not make is refused by name, never taken over.

Open the query page
===================

The store's own SQL page, ``/play``, is where a person types a query.

- **With web terminals**, the page is on the **readers' origin**: a second web
  server of the web tier, on its own port (``modules.web_terminals.reader_port``,
  or the browser-facing origin ``modules.web_terminals.reader_origin`` names
  behind a load balancer). ``osprey up`` prints each reader's link,
  ``<readers' origin>/telemetry/play`` already pointed at that origin's query
  endpoint and naming the reader; the person types the key into the page. Pages on the terminals' origin, where agent-written HTML runs,
  cannot read from this origin.
- **Without web terminals**, the store answers on the deploy host's loopback
  only, at ``deployment.port_base`` + 804 (``services.telemetry_store.port``).
  ``osprey up`` prints the page as ``http://127.0.0.1:<port>/play?user=<name>``
  and, beside it, the SSH tunnel that brings it to another machine:

  .. code-block:: bash

     ssh -N -L 10804:127.0.0.1:10804 <deploy host>

The page runs each query and shows the result when it is complete: there is no
live progress bar, and the chart view is not served. **Download** saves the
result.

Query from a script
===================

Send the query as the request body, with the reader's name and key as HTTP
basic credentials. Through the readers' origin the query endpoint is exactly
``/telemetry/``:

.. code-block:: bash

   KEY="$(sed -n 's/^password=//p' .env.telemetry-reader.carol)"
   curl -sS --user "carol:$KEY" --data-binary @query.sql \
     "https://<readers' origin>/telemetry/"

On the deploy host, query the loopback port and add
``http_wait_end_of_query=1`` to the URL yourself:

.. code-block:: bash

   curl -sS --user "operator:$KEY" --data-binary @query.sql \
     "http://127.0.0.1:10804/?http_wait_end_of_query=1"

That setting makes the store hold the answer until the query has finished, so
a query refused for a limit answers with an error status and the limit's name
rather than a ``200`` carrying part of a result. The readers' origin adds it to
every request and cannot be told otherwise; on the loopback port nothing adds
it for you.

Every query a reader runs is in the store's ``system.query_log`` under the
reader's name. The archive copies that log every day.

What a reader sees
==================

A reader is a reader everywhere: the same ``tables`` and ``days_back`` bound it
in SQL and on the history page. A reader account is read-only. It cannot insert,
change a setting that loosens its limits, read another database, or reach
outside the store (``remote``, ``url``, ``file`` and the like are refused).
A query that reaches past the reader's scope — a join, a subquery, ``merge()``
or a date outside ``days_back`` — succeeds and returns no out-of-scope rows.

The history page in a browser is described in
:ref:`multi-user-history-page`.

Limits
======

Every reader query runs under these limits. Over one, the query is refused with
the limit named and an error status, never cut short and returned as if
complete. Each is a key under ``services.telemetry_store.limits``:

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Key
     - Default
     - What it bounds
   * - ``result_rows``
     - 1,000
     - Rows in one answer
   * - ``result_bytes``
     - 10,000,000
     - Bytes in one answer
   * - ``seconds``
     - 30
     - Run time of one query
   * - ``rows_read``
     - 1,000,000
     - Rows one query reads
   * - ``bytes_read``
     - 500,000,000
     - Bytes one query reads
   * - ``query_memory``
     - 200,000,000
     - Memory one query uses
   * - ``concurrent``
     - 2
     - Queries one reader runs at once
   * - ``per_minute``
     - 60
     - Queries one reader starts per minute
   * - ``reader_slots``
     - 2
     - Queries **all readers together** run at once

``reader_slots`` is a shared pool. When it is full, a further reader query waits
in a queue until a slot frees, rather than being refused. The archive and
``osprey up`` take no slot of the pool, so readers never hold them up.

The store's memory limit (``services.telemetry_store.memory``, default ``2g``)
must hold every query that can run at once: ``osprey build`` refuses a config
where ``reader_slots`` plus the history page's and the archive's queries, each
at ``query_memory``, do not fit, and names the keys to change.

.. important::

   **A query without** ``FROM`` **escapes two limits.** The store does not count
   such a query against ``per_minute``, and does not apply ``result_rows`` or
   ``result_bytes`` to it, so a scalar-subquery shape can return more than
   1,000 rows with a ``200``. It is still bound by ``rows_read``, ``bytes_read``,
   ``query_memory``, ``seconds``, ``concurrent`` and the reader's scope.

What the record can vouch for
=============================

The store is as trustworthy as the processes that write to it. Read it with
these in mind:

- **Containers are the boundary.** Each sender's container holds the login of the one
  identity it sends as and mounts only its own ledger directory. On a deployment
  whose agent runs on the host, the agent runs as the operator's own user, and
  the credential files and every identity's ledger directory are within its
  reach.
- **Within one container, any process can write that user's record.** The
  ledger directory and the login belong to the container, not to one process
  in it.
- **Prompts and events are reported by the sender.** The gateway stamps who
  sent a record from the login it arrived with; what the record says happened
  is the sender's account of it.
- **Agent-authored session plans run inside the queueserver container.** Its
  preflight rows are therefore trusted narrowly: on a person's own history page
  a preflight row appears only when it matches that person's own queue-submit
  call within 60 seconds. Readers and ``history: all`` viewers see every
  preflight row as stored.
- **The accounts are told apart by password**, not by the address they connect
  from.
- **The ledger feed is at least once**, and duplicates are dropped when reading
  (:ref:`the keys <agent-record-dedup>`). One path can lose a batch: a gateway
  stopped while a ledger batch waits on a full export queue. ``osprey health``
  degrades the gateway when that queue passes 80%, so fix the store before
  restarting the gateway.
- **The store keeps** ``retention_days`` **of history.** The daily archive keeps
  the rest (:doc:`agent-record`).
