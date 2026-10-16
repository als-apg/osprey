=====================
Reconstruct a Session
=====================

The telemetry store keeps two records of every agent session: the events the
agent harness emits (prompts, responses, API calls) in ``otel_logs``, and the
records its tools file in ``audit_ledger``. One query puts them back together
in the order they happened, and any configured reader can run it.

For what the store holds and how it is fed, see :doc:`monitor-agent`; for how
long it keeps it, see :doc:`agent-record`.

What you need
=============

* **A reader key.** Every entry under ``services.telemetry_store.readers`` gets
  a key that ``osprey up`` writes to ``.env.telemetry-reader.<name>`` at the
  repo root (mode 0600, one line ``password=<key>``). The account name is the
  reader's name.
* **The store's host port.** The store answers HTTP on loopback at
  ``deployment.port_base`` + 804 (``http://127.0.0.1:10804`` at the default port
  base), or at ``services.telemetry_store.port`` when that is set.
* **Whose session, and which.** The user name the session was recorded under
  and its session id. To list a user's recent sessions:

  .. code-block:: sql

     SELECT SessionId, min(Timestamp) AS started, count() AS events
     FROM telemetry.otel_logs
     WHERE User = 'alice' AND Timestamp > now() - INTERVAL 3 DAY
     GROUP BY SessionId
     ORDER BY started DESC

The query
=========

.. literalinclude:: ../../../../src/osprey/history/queries/reconstruct_session.sql
   :language: postgresql

Every value is a typed parameter, never text spliced into the SQL, so a user
name or session id with a quote in it is still one value.

Run it
======

Send the query by POST, with each parameter as a ``param_<name>`` field of the
URL. Run from the repo root, with the query above saved as ``reconstruct_session.sql``:

.. code-block:: bash

   KEY="$(sed -n 's/^password=//p' .env.telemetry-reader.operator)"
   curl -sS --user "operator:$KEY" --data-binary @reconstruct_session.sql \
     "http://127.0.0.1:10804/?http_wait_end_of_query=1&param_user=alice&param_session=SESSION_ID&param_after_at=2000-01-01%2000%3A00%3A00&param_after_source=&param_after_hash=0"

Replace ``alice`` and ``SESSION_ID`` with the user and session, URL-encoded.
The query travels in the request body and the parameters in the URL;
``http_wait_end_of_query=1`` makes a query over a limit answer with an error
rather than part of a result (:doc:`read-the-record`).

The answer is one row per event, tool call, approval or preflight check, as
tab-separated columns ``at``, ``source``, ``row_hash``, ``user``, ``kind``,
``tool_use_id``, ``terminal_session`` and ``detail``.

Read the next page
==================

A page holds at most 500 rows. When a page holds exactly 500, take the last
row's ``at``, ``source`` and ``row_hash`` as ``param_after_at``,
``param_after_source`` and ``param_after_hash`` and send the query again. A page
of fewer than 500 rows is the last one.
