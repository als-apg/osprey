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
   * - ``data/bluesky_devices.yml``
     - The device table that decides which channels a Bluesky plan may drive.

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
       ``file_paths.*``, ``artifacts.*``, ``services.*.devices_file``, plus the
       runtime write paths below
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

Five writers consult the set. Each records its own name in the audit ledger, so
one query can be narrowed to a single surface or span all of them:

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

**The attempt is recorded in** ``var/audit/protected-writes.jsonl``. It sits in
the state zone beside ``readonly-refusals.jsonl``, and it is durable by
construction: ``osprey build`` re-renders ``build/`` wholesale and never
touches ``var/``, and ``osprey reset`` keeps ``var/audit`` unless you pass
``--purge-audit``. The two ledgers are separate because they answer different
questions — "did the agent try to move the machine" versus "did the agent try to
rewrite the framework that constrains it".

One JSON object per line, six fields:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - What it holds
   * - ``ts``
     - UTC timestamp, ``YYYY-MM-DDTHH:MM:SSZ``
   * - ``surface``
     - Which writer refused — one of the five surface names above
   * - ``target_file``
     - The file the write was aimed at
   * - ``key_or_path``
     - What inside it was protected: a dotted config key, or the
       project-relative path when the whole file is the target
   * - ``channel``
     - The channel that owns the target, named the same way the refusal message
       names it
   * - ``reason``
     - Short machine-readable reason — ``protected_key``, ``reserved path``,
       ``reserved path in ownership store``

A ``PUT`` that would have changed many protected keys at once names the first
ten and counts the rest in the message, but every changed key reaches the
ledger. The cap trims the message, never the audit.

.. note::

   **Recording is best-effort; refusing is not.** An unwritable audit zone or an
   unreachable feed degrades the trail and never turns a refusal into a server
   error — an error that reads like the gate malfunctioned is the one shape an
   operator could mistake for a gate that failed open.

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
