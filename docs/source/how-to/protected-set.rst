.. _how-to-protected-set:

=================
The Protected Set
=================

The OSPREY agent authors a great deal inside a deployment: subagents, slash
commands, notes, analysis scripts, plots. What it may not do is rewrite the
framework that decides what it is allowed to do — the write gate, the approval
chain, the limits table, the safety hooks, and the config keys those read. That
closed list of files and keys is the **protected set**.

It is a policy about *writers*, not about a directory. One table answers the
question "may a running agent rewrite this", and every agent-side writer
consults it before it puts bytes on disk: the settings drawer's galleries, the
Claude-setup file API, the config panel, the ``setup_patch`` tool, and the
restore that runs when a session starts. A path refused in one is refused in
all of them.

.. note::

   Two neighbouring boundaries answer different questions. It is worth knowing
   which one you have hit, because the way past each is different:

   * :ref:`What executed code may not change <python-executor-protected-paths>`
     is about **zones**. A Python run may not write into the render zone
     (``build/``), the profile sources, or the audit ledger, wherever inside
     them the path lands.
   * :ref:`Paths the profile may not write <profile-reserved-paths>` is about
     **build channels**. It says which part of your profile is allowed to
     produce a given file, so two channels never both write one artifact.

   This page is the third question: may the running agent rewrite this file, or
   this config key, at all. The answer does not change with execution mode,
   session posture, or who is logged in.

Files, by name
==============

Eleven project paths are protected exactly. Each is owned by the channel that
*does* write it, and a refusal names that channel — the way in, if you have one:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Path
     - Written by
   * - ``config.yml``
     - the profile's ``config:`` block
   * - ``.claude/settings.json``
     - the profile's ``config:`` keys — ``claude_code.permissions``,
       ``claude_code.hooks``, ``artifacts.hooks``
   * - ``.claude/hooks/hook_config.json``
     - the framework render, from the enabled ``mcp_servers:`` and
       ``control_system.write_tools``. It is the runtime configuration of the
       write-safety layer: ``osprey_writes_check.py`` reads its ``write_tools``
   * - ``.mcp.json``
     - the profile's ``mcp_servers:`` block
   * - ``CLAUDE.md``
     - the profile's ``claude_md_template:`` key
   * - ``.env``, ``.env.example``
     - the profile's own ``.env`` file and ``env:`` keys
   * - ``.osprey-manifest.json``
     - the build itself — it stamps the manifest from the resolved profile
   * - ``data/simulation/channel_manifest.json``
     - the profile's ``data/`` directory
   * - ``data/simulation/channel_limits.json``
     - the profile's ``data/`` directory
   * - ``docker/web-terminal-context/base.md``
     - the profile's ``web-terminal-context/base.md`` slot

Whole classes of file
=====================

Some protected artifacts have no fixed name — you can add a rule or a skill, and
the new one has to be protected the moment it exists. Those are matched by
shape:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Pattern
     - Why it is closed
   * - ``.claude/hooks/osprey_*.py``
     - The ``osprey_`` hooks *are* the write-safety layer. An agent editing one
       would be disarming the check that guards its own writes.
   * - ``.claude/skills/**``
     - A skill is instruction text the agent would otherwise be rewriting for
       itself.
   * - ``.claude/rules/**``
     - A rule is instruction text the agent would otherwise be rewriting for
       itself.
   * - ``.claude/settings.local.json``
     - A local settings overlay silently widens the permissions the build
       rendered.
   * - ``data/channel_limits.json``
     - The limits table every setpoint is checked against before it reaches the
       control system.

Three things make the answer depend on which *file* is named rather than on how
the writer spelled it:

- **Paths are normalized first.** ``./x``, ``a//b`` and
  ``foo/../.claude/skills/x`` all ask the same question as their plain
  spellings, so nothing dodges the set by spelling a path the long way round.
- **Matching ignores case.** OSPREY runs on case-insensitive filesystems, where
  ``.CLAUDE/Skills/x`` opens exactly the file ``.claude/skills/x`` names. The
  answer is the same on every host.
- **A path that cannot be judged is refused, not allowed.** A path that is
  absolute, or that still climbs above the project root after normalization, is
  not project-relative; "not a path I can judge" must never read as "writable".

``.claude/agents/`` and ``.claude/commands/`` are deliberately **not** in the
set. Authoring a subagent or a slash command is the point of those directories,
and neither changes what the agent is permitted to do.

Config keys
===========

Two files carry keys an agent-side writer may not set.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - File
     - Protected key families
   * - ``config.yml``
     - ``control_system.*``, ``approval.*``, ``hooks.*``,
       ``claude_code.permissions.*``, ``claude_code.hooks.*``,
       ``claude_code.servers.*``, ``artifacts.hooks``, ``agent_data.*``,
       ``file_paths.*``, ``artifacts.*``, plus the runtime write paths below
   * - ``.mcp.json``
     - ``mcpServers.*.command``, ``mcpServers.*.args``, ``mcpServers.*.env.*``

How a pattern reads: a ``*`` segment stands for exactly one level, so
``mcpServers.*.command`` covers every server's command line. A *trailing* ``*``
also covers the family's own key, so replacing the whole ``control_system:``
block with a scalar is refused as well as editing a key inside it. And a key
that is a strict ancestor of a protected key is protected too, because writing
the ancestor rewrites what is underneath it — ``services`` is protected because
a runtime write path lives beneath it, while ``services.orbit.enabled`` is not.

The rule for what belongs in this table:

.. pull-quote::

   Any key that gates writes/approval/limits, anchors a filesystem path the
   safety layers derive zones from, or is rendered into
   ``.claude/settings.json``/``.mcp.json``.

Runtime write paths, derived rather than copied
-----------------------------------------------

Two ``config.yml`` keys name a path something writes to at run time:
``simulation.state_dir`` and
``services.channel_finder.pipelines.hierarchical.feedback.store_path``. They are
protected under the rule's second clause — repointing one moves the area a
safety layer treats as writable.

They are not typed out in the protected table. That list is *imported* from the
connector configuration, where the runtime write paths are already enumerated
once for their own reasons. Adding a key there protects it here with no second
edit, and no second copy can drift out of step with the first.

The one exemption: ``hooks.debug``
----------------------------------

``hooks.*`` is in the table for the hook *wiring*, and it over-matches one key
that does not meet the inclusion rule. ``hooks.debug`` toggles diagnostic
verbosity inside the hook scripts and does nothing else: it gates nothing, moves
no zone, and is rendered into no artifact. It is also a shipped operator control
— the Web Terminal's Hook Debug switch sets it. So it is exempted by name.

The exemption names exactly one key and has none of a pattern's reach.
``hooks`` itself stays protected: replacing the whole block still rewrites the
wiring.

Which surfaces ask
==================

Five writers consult the set. Each files its refusals under its own name in the
audit trail --- one file per surface --- so one query can be narrowed to a single
surface or span all of them:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Surface name
     - What it is
     - What it refuses
   * - ``http_config``
     - The settings drawer's Config tab — ``PATCH`` and ``PUT`` on
       ``/api/config``
     - Protected keys in ``config.yml``, whether set, deleted, or reshaped by a
       whole-document replace
   * - ``setup_patch``
     - The ``setup_patch`` MCP tool the agent calls directly
     - Protected keys in the file it was pointed at
   * - ``claude_setup``
     - The Claude-setup file API (``/api/claude-setup``), which edits files
       under ``.claude/``
     - Reserved paths, on both saving an existing file and creating a new one
   * - ``scaffold_gallery``
     - The artifact galleries in the settings drawer (Behavior, Safety, Memory,
       Config)
     - Reserved paths, across all six of its write and delete operations
   * - ``scaffold_restore``
     - The startup restore that puts durably-saved artifact bodies back
     - Store records naming a reserved path

The artifact galleries
----------------------

Every operation in the galleries that can put bytes on disk or take a file off
it is gated, and each refusal says what did not happen:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Operation
     - Refusal ends with
   * - Saving an artifact body
     - ``NOTHING WAS WRITTEN``
   * - Creating a new artifact
     - ``NOTHING WAS CREATED``
   * - Claiming a framework artifact
     - ``NOTHING WAS CLAIMED``
   * - Releasing a claim *and deleting the file*
     - ``NOTHING WAS DELETED``
   * - Deleting an untracked file
     - ``NOTHING WAS DELETED``
   * - Registering an untracked file as yours
     - ``NOTHING WAS REGISTERED``

Releasing a claim *without* deleting stays open: it changes no file, and the
protected set is about who writes the bytes.

**The judgement is on the resolved path.** A name that looks clean and
unprotected can still address a different file through a symlink — every writer
here follows the link, so a save opens the target and a delete removes what the
link points at. The gallery asks where the bytes actually land, which is what
holds the set against a link planted inside the render:
``.claude/agents/x.md`` pointing at ``../rules/safety.md`` is lexically an agent
and physically a rule.

**The read-only badge mirrors the gates exactly.** A card carrying the
``READ-ONLY`` badge is one the gates would refuse, because the badge is a
rename of the same answer rather than a second copy of the policy. A card
offering an edit that the save would then reject is worse than one that says
plainly it is not yours to edit.

Restoring artifacts at startup
------------------------------

Artifact bodies you save are kept in a durable store, so a container that is
recreated comes back with your copies rather than the ones the image shipped.
That restore runs in both deployment shapes — at Web Terminal startup on a bare
host, and in the container entrypoint before the server starts — and it is the
same code in both, so the answer cannot depend on how the deployment is run.

A store record naming a reserved path is skipped and audited, not obeyed. The
store sits on a volume the agent can write, so a poisoned record would otherwise
be a way to have the restore install agent-authored instruction text — and under
the container entrypoint, install it as root into a tree the agent cannot
otherwise reach. The restore skips that one record and carries on: one bad
record must not cost you every other body you saved, and a container that will
not start is a worse outcome than one that starts with a single file left as the
image shipped it.

The Claude-setup file API
-------------------------

``CLAUDE.md``, ``.mcp.json``, ``.claude/settings.json`` and the rules and skills
below ``.claude/`` are rendered by the build profile. Edit them in the profile
and rebuild the project; saves aimed at them here are refused, because a profile
that no longer describes the project it built is worse than an edit that did not
happen. Reserved entries are reported as read-only in the file listing, so the
surface never offers an edit it would then reject.

Where a refusal shows up
========================

Three things happen on every refusal, at whichever surface caught it.

**The message names the channel and says nothing happened.** A refusal an
operator has to interpret is one they will assume was partially carried out, so
the wording is explicit. An artifact refusal reads
``'<name>' belongs to <channel>. NOTHING WAS WRITTEN.``; a config refusal ends
``config.yml is unchanged — no field in this request was applied.`` Both point
at the profile and ``osprey build``, never at a retry.

**The agent-activity feed reports it.** Each refusal publishes one frame to the
Web Terminal's activity feed, so a blocked attempt is something you see rather
than something only the agent saw. Config refusals are phrased identically
whichever surface produced them — ``BLOCKED a protected config key`` — and carry
key names only, never values.

**The attempt is recorded in** ``var/audit/<identity>/<surface>.jsonl``. The
path is the whole filing system: ``<identity>`` is whoever was acting, so on a
multi-user deployment each person's refusals stay in their own directory, and
``<surface>`` is the writer that refused, so a gallery refusal and a config
refusal never share a file. The zone is durable by construction: ``osprey
build`` re-renders ``build/`` wholesale and never touches ``var/``, and
``osprey reset`` keeps ``var/audit`` unless you pass ``--purge-audit``.

The same record shape covers every safety decision OSPREY files, so one reader
handles a refused config key and a refused control-system write alike. One JSON
object per line:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - What it holds
   * - ``ts``
     - UTC timestamp, ``YYYY-MM-DDTHH:MM:SSZ``
   * - ``surface``
     - Which writer refused — one of the five surface names above
   * - ``actor``
     - Who was acting, from the terminal login where there is one
   * - ``posture``
     - The session's posture at the time, ``sandbox`` or ``writes``. The
       protected set is closed in both; this says what else was in force
   * - ``posture_source``
     - How that posture was established — ``spawn`` (fixed when the session
       started), ``live`` (read from the session's setting at the time),
       ``app`` (a web request, which belongs to no session), ``process`` (no
       session posture at all, as in a CLI run). Never guessed from the
       posture value
   * - ``session``
     - The terminal session the posture belonged to, or ``null`` outside one
   * - ``subject``
     - What was protected: a dotted config key, or the project-relative path
       when the whole file is the target
   * - ``decision``
     - ``refused`` on this page — the same field reads ``allowed`` on the
       records of calls that went through
   * - ``reason``
     - Short machine-readable reason — ``protected_key``, ``reserved path``,
       ``reserved path in ownership store``; a control-system write the
       session posture refused reads ``posture`` on every surface (the hook,
       the MCP server and the Python executor all spell it the same way)
   * - ``detail``
     - The file the write was aimed at (``target=``) and the channel that owns
       it, named the same way the refusal message names it

A ``PUT`` that would have changed many protected keys at once names the first
ten and counts the rest in the message, but **every changed key gets its own
line**. The cap trims the message, never the audit. A refused request leaves
exactly those lines and no others: the surface that decided files the record,
and the layers around it stand aside rather than filing the same refusal
again.

These five files are part of a larger trail — the same record shape covers
tool calls, hook decisions, logins and web mutations. What else is in it, who
can read it, and what it does not promise are :ref:`below <how-to-audit-trail>`.

.. note::

   **Recording is best-effort; refusing is not.** An unwritable audit zone or an
   unreachable feed degrades the trail and never turns a refusal into a server
   error — an error that reads like the gate malfunctioned is the one shape an
   operator could mistake for a gate that failed open.

.. _how-to-audit-trail:

The audit trail
===============

These refusals are one part of a single trail. Under ``var/audit/`` there is one
directory per identity, and one file per surface inside it. Every
safety-relevant decision the deployment makes is one line in one of them:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - File under ``var/audit/<identity>/``
     - What it records
   * - ``http_config.jsonl``, ``setup_patch.jsonl``, ``claude_setup.jsonl``,
       ``scaffold_gallery.jsonl``, ``scaffold_restore.jsonl``
     - The five protected-set surfaces above
   * - ``executor.jsonl``
     - Python runs a safety layer refused — see :ref:`python-executor-protected-paths`
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
of an operator. What the operator then said is visible in what follows — an
approved call leaves its own record on the server that ran it, and a declined
one never reaches a server at all.

Some of that is chatter rather than safety: every request that changes state is
recorded, so moving a panel around the terminal leaves lines in
``http_mutation.jsonl`` next to the config edits. One asymmetry worth knowing
before you go looking: a *refused* WebSocket upgrade is recorded, an admitted
one is not — a live session's activity is its tool records, not its connection.

Who can read it
---------------

In a multi-user deployment the directories are the isolation. Each user's
container is handed ``var/audit/<their name>/`` and nothing else under
``var/audit/``, so alice's terminal cannot read — or rewrite — bob's records,
even though both live in one project. The login service writes its own
``var/audit/sidecar/``, which it owns as root and which no terminal mounts at
all; each dispatch worker and the Bluesky panel service write their own
subdirectory on the same terms.

**The deployment-wide view is the host's.** All of it is one tree on the deploy
host, so whoever has a shell there reads every subdirectory with ``grep``, and
that is the only place a question spanning several people can be answered.

Inside a container, the admin tier can read its own records without a shell:
``GET /api/audit/recent`` returns the newest records from that container's own
subdirectory, newest first, behind the same switch as the Config panel
(``web.config_panel.enabled``). It never reaches another user's.

What the trail is not
---------------------

**It is append-only, not tamper-evident.** Lines are only ever appended, and
OSPREY never rewrites or prunes one. But there is no hash chain and no
signature: anyone who can write a subdirectory can edit or delete lines in it —
the host's administrator included — and nothing in a later read would show that
they had. It is an operational record of what the deployment decided, which is
what answers "what happened here". If you need something that holds against
someone with write access, ship the lines off the host as they are written, to a
collector the deployment cannot reach back into.

**Nothing rotates or expires it.** The files grow until you do something about
them. ``osprey reset --purge-audit`` empties the zone deliberately; rotation,
retention windows and forwarding are yours to arrange.

**One file carries a payload:** ``executor.jsonl``. Every other record holds
identifiers and config keys only — a surface name, a username, a tool name, a
dotted key, a short reason — never a config value, a prompt, or an agent
message. The Python executor is the deliberate exception: a refused run records
the code it refused, whole, in a ``source`` field (8000 characters, with
``source_truncated: true`` where a longer script was cut). A record of a refused
write that does not say what the write *was* is an alert, not an audit trail.
What that means for anyone reading or forwarding the trail: this one file
contains whatever the agent tried to run, including anything the conversation
put into the script. Give it the same care as the code itself, and expect it to
be the file that grows.

See also
========

- :ref:`What executed code may not change <python-executor-protected-paths>` —
  the zone boundary for Python runs, and the two layers that enforce it.
- :ref:`Paths the profile may not write <profile-reserved-paths>` — which build
  channel owns which artifact, checked at build time.
- :ref:`Sandbox one session <web-terminal-session-posture>` — the per-session
  posture that refuses control-system writes. It is a separate control: no
  posture opens the protected set, and no posture closes anything on this page
  further.
- :doc:`Privilege Tiers <multi-user/tiers>` — what each login may do, and
  the three layers that hold it. This page is the line every tier shares,
  admin included.
