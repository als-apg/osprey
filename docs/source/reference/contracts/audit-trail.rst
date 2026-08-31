.. _reference-audit-trail:

====================
Audit Trail Contract
====================

Every safety-relevant decision a deployment makes is one line under
``var/audit/`` in the deployment repository: one directory per identity, one
JSON-lines file per surface inside it. This page is the shape of that trail ---
which file a decision lands in, what each field holds, who can read the files,
and what the trail does not promise. For the policy that produces most of the
refusals see :ref:`the protected set <config-protected-set>`; for the
per-session, per-target posture that refuses control-system writes see
:ref:`web-terminal-session-posture`.

The zone is durable by construction: ``osprey build`` re-renders ``build/``
wholesale and never touches ``var/``, and ``osprey reset`` keeps ``var/audit``
unless you pass ``--purge-audit``.

.. _audit-trail-files:

Files, by surface
=================

``<identity>`` is whoever was acting, so on a multi-user deployment each
person's records stay in their own directory; ``<surface>`` is the layer that
decided, so a gallery refusal and a config refusal never share a file.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - File under ``var/audit/<identity>/``
     - What it records
   * - ``http_config.jsonl``
     - The settings drawer's Config tab --- ``PATCH`` and ``PUT`` on
       ``/api/config`` --- refusing protected keys in ``config.yml``, whether
       set, deleted, or reshaped by a whole-document replace
   * - ``setup_patch.jsonl``
     - The ``setup_patch`` MCP tool the agent calls directly, refusing
       protected keys in the file it was pointed at
   * - ``claude_setup.jsonl``
     - The Claude-setup file API (``/api/claude-setup``), which edits files
       under ``.claude/``, refusing reserved paths on both saving an existing
       file and creating a new one
   * - ``scaffold_gallery.jsonl``
     - The artifact galleries in the settings drawer (Behavior, Safety, Memory,
       Config), refusing reserved paths across all six of their write and
       delete operations
   * - ``scaffold_restore.jsonl``
     - The startup restore that puts durably-saved artifact bodies back,
       refusing store records that name a reserved path
   * - ``executor.jsonl``
     - Python runs a safety layer refused --- see
       :ref:`python-executor-protected-paths`
   * - ``<server>.jsonl``
     - Tool calls on an OSPREY MCP server, allowed and refused alike, in a file
       named for the server: ``osprey_workspace``, ``python``, ``bluesky``
   * - ``hook_writes_check.jsonl``, ``hook_approval.jsonl``,
       ``hook_limits.jsonl``, ``hook_memory_guard.jsonl``
     - What the safety hooks denied, and what they put in front of a person
   * - ``http_mutation.jsonl``, ``web_auth.jsonl``
     - Requests that changed something through a web API, and the 401s and 403s
       the login check itself refused
   * - ``auth_sidecar.jsonl`` (under ``var/audit/sidecar/``)
     - Logins and login refusals, where a deployment has a login wall

``decision`` reads ``allowed`` or ``refused`` on almost all of them, and ``ask``
in ``hook_approval.jsonl``, where the hook did neither: it put the call in front
of an operator. What the operator then said is visible in what follows --- an
approved call leaves its own record on the server that ran it, and a declined
one never reaches a server at all.

Some of that is chatter rather than safety: every request that changes state is
recorded, so moving a panel around the terminal leaves lines in
``http_mutation.jsonl`` next to the config edits. One asymmetry worth knowing
before you go looking: a *refused* WebSocket upgrade is recorded, an admitted
one is not --- a live session's activity is its tool records, not its
connection.

.. _audit-trail-record:

The record
==========

One record shape covers every file, so one reader handles a refused config key
and a refused control-system write alike. One JSON object per line:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - What it holds
   * - ``ts``
     - UTC timestamp, ``YYYY-MM-DDTHH:MM:SSZ``
   * - ``surface``
     - Which layer decided --- the file's own name
   * - ``actor``
     - Who was acting, from the terminal login where there is one
   * - ``posture``
     - The session's posture at the time, ``sandbox`` or ``writes``. The
       protected set is closed in both; this says what else was in force
   * - ``posture_source``
     - How that posture was established --- ``spawn`` (fixed when the session
       started), ``live`` (read from the session's setting at the time),
       ``app`` (a web request, which belongs to no session), ``process`` (no
       session posture at all, as in a CLI run). Never guessed from the
       posture value
   * - ``session``
     - The terminal session the posture belonged to, or ``null`` outside one
   * - ``subject``
     - What the decision was about: a dotted config key, a tool name, or the
       project-relative path when a whole file is the target
   * - ``decision``
     - ``allowed``, ``refused``, or ``ask``
   * - ``reason``
     - Short machine-readable reason --- ``protected_key``, ``reserved path``,
       ``reserved path in ownership store``; a control-system write the
       session posture refused reads ``posture`` on every surface (the hook,
       the MCP server and the Python executor all spell it the same way)
   * - ``detail``
     - Surface-specific context: for a protected-set refusal, the file the
       write was aimed at (``target=``) and the channel that owns it, named
       the same way the refusal message names it; on the web surfaces, the
       login the request came from --- see :ref:`audit-trail-identity-keys`

A ``PUT`` that would have changed many protected keys at once names the first
ten and counts the rest in the message, but **every changed key gets its own
line**. The cap trims the message, never the audit. A refused request leaves
exactly those lines and no others: the surface that decided files the record,
and the layers around it stand aside rather than filing the same refusal
again.

.. note::

   **Recording is best-effort; refusing is not.** An unwritable audit zone or an
   unreachable activity feed degrades the trail and never turns a refusal into
   a server error --- an error that reads like the gate malfunctioned is the one
   shape an operator could mistake for a gate that failed open.

.. _audit-trail-identity-keys:

The login behind the record
---------------------------

Where a deployment has a login wall, the two web surfaces ---
``http_mutation.jsonl`` and ``web_auth.jsonl`` --- record who the request came
from in ``detail``, as up to three keys. Which of them are present is itself
part of the message:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - What it holds
   * - ``account=``
     - The roster account the request is on --- the card named in the URL, and
       the name the login service checked the session against. Present on
       every record where the request carried a login at all
   * - ``expected_account=``
     - The account this container serves, written **only when the forwarded
       account is not it**. Its presence is the whole signal: one person's
       authorization arrived at another person's container. A request on the
       right card never writes the key. The same case logs a warning, and it
       is recorded rather than refused
   * - ``oidc_subject=``
     - Who proved the login, as the provider asserted it --- an opaque id or an
       email. Written only where that differs from the account, so a password
       deployment never sees this key, and a shared card records the person
       beside the card they opened

Two forwarded headers carry that from the login service through nginx:
``X-Osprey-Auth-Account`` names the card, ``X-Osprey-Auth-Subject`` names the
login that proved it. Under ``auth.method: password`` they hold the same value,
because the roster username *is* the proof; under ``oidc`` they part, and only
the account is a name a container can compare itself against.

A deployment that pins an older login-service image with ``auth.image`` gets no
account header. The container then falls back to comparing the subject, as it
did before this release. Where the mapped subject is not the roster name --- an
opaque ``sub`` or an email, the usual case --- it never matches, so
``expected_account=`` and a warning ride every audited request. Building against
this release's image is what clears it.

Who can read it
===============

In a multi-user deployment the directories are the isolation. Each user's
container is handed ``var/audit/<their name>/`` and nothing else under
``var/audit/``, so alice's terminal cannot read --- or rewrite --- bob's
records, even though both live in one project. The login service writes its
own ``var/audit/sidecar/``, which it owns as root and which no terminal mounts
at all; each dispatch worker and the Bluesky panel service write their own
subdirectory on the same terms.

**The deployment-wide view is the host's.** All of it is one tree on the deploy
host, so whoever has a shell there reads every subdirectory with ``grep``, and
that is the only place a question spanning several people can be answered.

Inside a container, the admin tier can read its own records without a shell:
``GET /api/audit/recent`` returns the newest records from that container's own
subdirectory, newest first, behind the same switch as the Config panel
(``web.config_panel.enabled``). It never reaches another user's.

What the trail is not
=====================

**It is append-only, not tamper-evident.** Lines are only ever appended, and
OSPREY never rewrites or prunes one. But there is no hash chain and no
signature: anyone who can write a subdirectory can edit or delete lines in it
--- the host's administrator included --- and nothing in a later read would
show that they had. It is an operational record of what the deployment
decided, which is what answers "what happened here". If you need something that
holds against someone with write access, ship the lines off the host as they
are written, to a collector the deployment cannot reach back into.

**Nothing rotates or expires it.** The files grow until you do something about
them. ``osprey reset --purge-audit`` empties the zone deliberately; rotation,
retention windows and forwarding are yours to arrange.

**One file carries a payload:** ``executor.jsonl``. Every other record holds
identifiers and config keys only --- a surface name, a username, a tool name, a
dotted key, a short reason --- never a config value, a prompt, or an agent
message. The Python executor is the deliberate exception: a refused run records
the code it refused, whole, in a ``source`` field (8000 characters, with
``source_truncated: true`` where a longer script was cut). A record of a refused
write that does not say what the write *was* is an alert, not an audit trail.
What that means for anyone reading or forwarding the trail: this one file
contains whatever the agent tried to run, including anything the conversation
put into the script. Give it the same care as the code itself, and expect it to
be the file that grows.
