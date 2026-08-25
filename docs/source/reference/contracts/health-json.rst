The ``osprey health --json`` Contract
=====================================

The exact shape of the document ``osprey health --json`` writes, the exit codes
that accompany it, and how to consume both from a CI job.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - The nine-key report envelope and what each field means
   - The per-check row shape, including which three keys are optional
   - The four check statuses and the five exit codes
   - The machine-clean stdout guarantee that makes piping to ``jq`` safe
   - How the agent's MCP health tools extend the same envelope

   **Prerequisites:** A deployment repository with a ``config.yml``.

Overview
--------

``osprey health --json`` runs the same suite as a plain ``osprey health`` and
writes the result as a single JSON document on stdout. The document is the
contract: the web dashboard and the agent's MCP health tools serve the same
shape, so anything you write against this page works against all three.

.. code-block:: bash

   osprey health --json

Everything the run produces for a human — the spinner, warnings, the failure
banner — goes to stderr. Stdout carries exactly one JSON document and a
trailing newline, so ``osprey health --json | jq ...`` never sees a stray line.

The report envelope
-------------------

The document always has these nine keys, in this order:

.. code-block:: json

   {
     "summary": "6/17 checks passed (2 warnings, 1 error, 8 skipped)",
     "ok": 6,
     "warnings": 2,
     "errors": 1,
     "skips": 8,
     "total": 17,
     "elapsed_ms": 194.8,
     "deadline_hit": false,
     "results": [
       {
         "name": "disk_space",
         "category": "file_system",
         "status": "ok",
         "message": "Disk 6% full (6986.2 GB free)"
       },
       {
         "name": "beam_current",
         "category": "beamline_services",
         "status": "ok",
         "message": "SR:CURRENT = 401.2 mA",
         "value": "401.2 mA",
         "latency_ms": 12.4
       },
       {
         "name": "config_file_exists",
         "category": "configuration",
         "status": "error",
         "message": "config.yml not found at /srv/osprey/my-agent/config.yml",
         "details": "Searched from: /srv/osprey/my-agent\nRun this from inside a deployment repository, or pass --project with the directory holding config.yml."
       }
     ]
   }

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Key
     - Type
     - Meaning
   * - ``summary``
     - string
     - The same one-line summary the human report prints, e.g.
       ``"10/15 checks passed (5 skipped)"``. Counts only; no status word.
   * - ``ok``
     - integer
     - Number of rows with status ``ok``.
   * - ``warnings``
     - integer
     - Number of rows with status ``warning``.
   * - ``errors``
     - integer
     - Number of rows with status ``error``.
   * - ``skips``
     - integer
     - Number of rows with status ``skip``.
   * - ``total``
     - integer
     - Number of rows in ``results`` — the sum of the four counts above.
   * - ``elapsed_ms``
     - number
     - Wall-clock time for the whole run in milliseconds, rounded to one
       decimal place.
   * - ``deadline_hit``
     - boolean
     - ``true`` when the run reached a suite-level deadline
       (``health.suite_timeout_s`` or ``health.on_demand_timeout_s``) before
       every check finished. Rows for checks that never ran are still present,
       so a ``true`` here is the signal that the counts describe a truncated
       run.
   * - ``results``
     - array
     - One object per check, in the order the suite produced them. See below.

Check rows
----------

Every object in ``results`` carries four keys unconditionally:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Key
     - Type
     - Meaning
   * - ``name``
     - string
     - Machine-readable check identifier, e.g. ``"disk_space"`` or
       ``"beam_current"``. Unique within a category.
   * - ``category``
     - string
     - The category the check belongs to, e.g. ``"file_system"``,
       ``"providers"``, or a facility-defined name from the ``health:`` block.
       This is the value ``--category NAME`` selects on.
   * - ``status``
     - string
     - One of ``ok``, ``warning``, ``error``, ``skip``. No other value is ever
       emitted.
   * - ``message``
     - string
     - Human-readable one-line result.

Three further keys are **optional** — they appear only when the check produced
them, and a consumer must treat a missing key as "not measured", not as an
error:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Key
     - Type
     - Present when
   * - ``value``
     - string
     - The check measured something worth reporting separately from the
       message, e.g. ``"401.2 mA"``. Omitted when empty. Most built-in checks
       do not set it; channel-read and probe checks do.
   * - ``latency_ms``
     - number
     - The check timed itself and the result was greater than zero, rounded to
       one decimal place. Omitted otherwise. Probe checks (HTTP, MCP, channel
       read, archiver freshness) set it; most local checks do not.
   * - ``details``
     - string
     - The check has extended diagnostic text — an exception string, a remedy
       hint, a searched-path list. Often multi-line. Omitted when empty.

.. note::

   Do not assume the three optional keys correlate with status. An ``ok`` row
   may carry ``value`` and ``latency_ms``; an ``error`` row may carry only
   ``message``. Guard every read of them, for example
   ``jq -r '.results[] | .latency_ms // "n/a"'``.

Check statuses
--------------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Status
     - Meaning
   * - ``ok``
     - The check ran and passed.
   * - ``warning``
     - The check ran and found something degraded but not broken — a missing
       optional file, a value outside its preferred range.
   * - ``error``
     - The check ran and failed, or could not run because of a fault it
       considers fatal (a failed ``config.yml`` load produces ``error`` rows in
       the ``configuration`` category).
   * - ``skip``
     - The check was not run. The usual reasons are an ``on_demand`` category
       without ``--full``, or a check whose configuration was unavailable. A
       skip does **not** fail the suite.

Exit codes
----------

The exit code is derived from the counts, so it and the document always agree.

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - Code
     - Meaning
     - When it occurs
   * - ``0``
     - Healthy
     - ``errors`` is ``0`` and ``warnings`` is ``0``. Skips do not count
       against this, so a default run with unexecuted ``on_demand`` categories
       still exits ``0``.
   * - ``1``
     - Degraded
     - ``warnings`` is greater than ``0`` and ``errors`` is ``0``.
   * - ``2``
     - Failing
     - ``errors`` is greater than ``0``. A ``config.yml`` that will not load
       lands here, with the reason in the ``configuration`` rows.
   * - ``3``
     - Command failure
     - The command itself raised before it could finish. **No JSON document is
       written** — the failure is reported on stderr instead. Add ``--verbose``
       for a traceback.
   * - ``130``
     - Interrupted
     - The run was interrupted (Ctrl-C). **No JSON document is written.**

.. important::

   "Healthy", "degraded", and "failing" are names for the exit codes ``0``,
   ``1``, and ``2`` in this table only. They are not values in the payload:
   the document carries counts and the four row statuses
   (``ok``/``warning``/``error``/``skip``), and there is no top-level status
   word anywhere in it. Derive the verdict from the exit code, or from
   ``.errors`` and ``.warnings``.

   Codes ``3`` and ``130`` are added by the command, not by the report. A
   consumer that reads stdout must therefore handle "no document at all", not
   just a document with errors in it.

Consuming it from CI
--------------------

Because ``osprey health`` exits non-zero on warnings, capture the exit code
before your shell's ``errexit`` sees it:

.. code-block:: bash

   set +e
   report=$(osprey health --json)
   code=$?
   set -e

   case "$code" in
     0) echo "healthy" ;;
     1) echo "degraded: $(echo "$report" | jq -r '.summary')" ;;
     2) echo "failing: $(echo "$report" | jq -r '.summary')"; exit 1 ;;
     *) echo "health check did not complete (exit $code)"; exit 1 ;;
   esac

Fail a pipeline on errors while tolerating warnings:

.. code-block:: bash

   osprey health --json | jq -e '.errors == 0' > /dev/null

List what actually broke, category-qualified:

.. code-block:: bash

   osprey health --json \
     | jq -r '.results[]
              | select(.status == "error" or .status == "warning")
              | "\(.status)\t\(.category)/\(.name)\t\(.message)"'

Pull the details block for one check:

.. code-block:: bash

   osprey health --json \
     | jq -r '.results[] | select(.name == "config_file_exists") | .details // "(none)"'

Watch for a truncated run — counts from a run that hit its deadline describe
fewer checks than the suite intended:

.. code-block:: bash

   osprey health --json | jq -e '.deadline_hit | not' > /dev/null \
     || echo "health suite hit its deadline; raise health.suite_timeout_s"

Scope a run to one category, which narrows both ``results`` and the counts:

.. code-block:: bash

   osprey health --json --category providers | jq -r '.summary'

The same envelope from the agent
--------------------------------

The agent reaches health through two MCP tools rather than the CLI, and they
serve the **same nine-key envelope** plus three keys of their own:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Type
     - Meaning
   * - ``cached``
     - boolean
     - The report came from the server's process cache rather than a fresh
       run.
   * - ``age_s``
     - number
     - Age of the served snapshot in seconds.
   * - ``refresh_suppressed``
     - boolean
     - A stale snapshot was served because a wedged worker thread blocked the
       refresh.

``health_check`` runs the poll tier and may serve a cached snapshot, so all
three keys vary. ``health_check_full`` runs the full suite fresh on every
call — the poll tier plus ``on_demand``; it is approval-gated — and therefore
always reports ``cached: false``, ``age_s: 0``, ``refresh_suppressed: false``.

.. note::

   The nine envelope keys and the check-row shape are the contract, shared by
   the CLI, the web dashboard, and both MCP tools. ``cached``, ``age_s``, and
   ``refresh_suppressed`` are specific to the MCP surface: ``osprey health
   --json`` never emits them. Write consumers against the nine; branch on the
   three only in code that talks to the MCP tools.

One behavioural difference is worth knowing when a category filter is in play.
Under ``health_check``, a filtered call recomputes the counts from the selected
rows but leaves ``elapsed_ms`` and ``deadline_hit`` describing the underlying
unfiltered suite run. ``health_check_full`` passes the filter into the run
itself and so behaves like the CLI, where ``--category`` scopes the run — all
nine keys describe the same scoped run.

.. seealso::

   :doc:`configure-health-checks`
       Add facility probe checks and plugins, and tune the suite's cost
       classes and timeouts.

   :doc:`cli-agent`
       The other machine-readable verb — full agent runs for CI.

   :doc:`/cli-reference/index`
       Full ``osprey health`` flag reference.
