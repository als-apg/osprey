.. _how-to-multi-user-tiers:

===============
Privilege Tiers
===============

Every card on the landing page opens a terminal with a fixed set of
capabilities — a **tier**. This page says what each tier may do, where a tier
comes from, and what stops a session from doing more than its tier allows. It
is written for the person deciding who gets which login; the mechanics behind
each boundary are one link away wherever they matter.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - The two questions that decide a tier
   - What the three tiers the ``control-assistant`` preset ships can and
     cannot do, login by login
   - That a tier is nothing more than a persona file — and what that means
     when you rename one or add one
   - Where the boundary is actually enforced, and the two places it is a
     matter of trust rather than enforcement

   **Prerequisites:** none. To watch the boundary act you need the stack
   running — see :doc:`index`.

Two questions decide a tier
===========================

A tier is the answer to two independent questions.

**May this session move hardware?** That is the reference monitor's master
write switch, ``control_system.writes_enabled``. Off, and every write surface
refuses — channel writes, read-write Python execution, plan arming, all of it,
from the single switch. On, and a write is *supervised*: it still passes the
writes-check hook, the per-channel limits, and a human approval prompt before
the connector executes it.

**May this session change the deployment it runs in?** That is a different
capability altogether — the ability to rewrite the ``config.yml`` every terminal
of this tier runs under, through the agent's ``setup_patch`` tool, the web
Config panel, or the scaffold gallery's editors. The preset's base profile
takes this away from every tier built on it, and hands it back to exactly one.

Put the two side by side and the tiers name themselves:

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 34 33 33

   * -
     - Cannot edit the deployment
     - Can edit the deployment
   * - **Control-system writes off**
     - **readonly**
     - —
   * - **Control-system writes on** (supervised)
     - **readwrite**
     - **admin**

The preset ships the three named cells. The empty one is not forbidden by
anything in OSPREY — a persona could be written for it — it simply has no use
the preset wanted to ship.

.. _multi-user-tiers:

The three tiers the preset ships
================================

Each tier is a self-contained OSPREY project with its **own** permissions,
because permissions are a property of a project's ``config.yml`` — the tiers
are genuinely different agents, not one agent with a UI toggle.

.. list-table::
   :header-rows: 1
   :widths: 16 26 26 32

   * - Tier (login)
     - Control-system writes
     - Editing the deployment
     - What the screen looks like
   * - **readonly** (bob)
     - Off. Every write surface refuses, from the single switch
     - No. The ``setup_patch`` tool is denied, the Config panel is off, and
       the scaffold gallery is readable but not writable
     - Chat-first ``simple`` layout, without the EVENTS and BLUESKY panels
   * - **readwrite** (alice)
     - On, and supervised: a channel write still passes the writes-check
       hook, the per-channel limits, and a human approval prompt before it
       executes
     - No — the same floor as readonly
     - Full ``expert`` workspace with the EVENTS and BLUESKY panels
   * - **admin** (carol)
     - On, on exactly the supervised terms above
     - Yes, and only here: the ``setup-mode`` skill, the ``setup_patch``
       tool, the web Config panel, and the gallery's edit, create and delete
       surfaces
     - Full ``expert`` workspace, with the Config panel and without the
       EVENTS and BLUESKY panels
   * - **ariel**
     - No control system behind it at all, so there is no write posture to
       compare
     - No — it inherits the same floor as the operator tiers
     - The standalone logbook terminal, opening on its ARIEL panel

Read the write column one row at a time rather than as a single statement
about the deployment. The posture is a property of the **session**, not of the
person holding it: which teammates get a write-capable login is your roster's
call, and the point is that the framework provisions genuinely different
postures out of one deployment.

That the admin login has no EVENTS or BLUESKY panels looks like an oversight
and is not one. Those two panels are declared only in the read-write delta
(``personas/readwrite.yml``), so they reach that tier and no other — the admin
delta never inherits them in the first place. It also suits what the admin
card is for: queueing plans and watching the event dispatcher is operator
work, done from an operator card.

**The default stays on the safe side.** ``default_persona`` is ``readonly``,
so a roster entry added in a hurry with no ``persona`` of its own gets the
read-only tier.

.. dropdown:: The exact tool names
   :icon: code

   The same table, as the rendered projects spell it. The single-user column
   is the one worth knowing: a plain ``osprey init --preset control-assistant``
   render is readwrite at the machine and floored at the deployment — no
   Config tab, no gallery writes. Admin in a single-user deployment means
   editing ``profile.yml`` and rebuilding.

   .. list-table::
      :header-rows: 1
      :widths: 28 16 18 18 20

      * - Capability
        - readonly
        - readwrite
        - admin
        - single-user base
      * - ``control_system.writes_enabled``
        - ``false``
        - ``true``
        - ``true``
        - ``true``
      * - ``mcp__controls__channel_write``
        - kill-switch deny
        - ask + approval hook
        - ask + approval hook
        - ask + approval hook
      * - Bluesky arming tools
        - kill-switch deny
        - ask
        - ask
        - ask
      * - ``mcp__python__execute``
        - read-only kernel only
        - both kernels
        - both kernels
        - both kernels
      * - ``mcp__osprey_workspace__setup_patch``
        - deny (floor)
        - deny (floor)
        - ask + approval hook
        - deny (floor)
      * - Config panel (``/api/config``)
        - 403
        - 403
        - enabled
        - 403
      * - Scaffold gallery edit / create / delete
        - 403 (read OK)
        - 403 (read OK)
        - enabled
        - 403 (read OK)
      * - Session posture toggle → *writes*
        - refused
        - confirm modal
        - confirm modal
        - confirm modal
      * - ``build/config.yml`` owner in the container
        - root
        - root
        - osprey
        - root
      * - :ref:`The protected set <config-protected-set>`
        - refused
        - refused
        - refused
        - refused

   The ``ariel`` persona is not in the table because it has no control system
   to gate: its deployment-editing floor is the same as the operator tiers'.

A tier is a persona file
========================

There is no ``tier:`` setting anywhere. A tier is what a **persona** turns out
to be once it is built.

.. raw:: html
   :file: ../../../_diagrams/tier-render.html

Each file under ``personas/`` holds only that persona's **differences** from
``profile.yml``. The base profile carries the floor — the three
deployment-editing surfaces switched off — and a persona sets what it needs on
top: ``readonly.yml`` turns control-system writes off, ``readwrite.yml`` leaves
them on, ``admin.yml`` leaves them on and lifts the floor. ``osprey build``
merges each delta over the base and renders it into its own project, and
``osprey up`` builds one container image from each. Every arrow in the drawing
is a build step, never a runtime lookup: the web server never asks *who is
this user and what tier are they?* Identity lives in the roster and the login
wall; capability lives in the render.

So the two words name two things. A **persona** is the mechanism: a named
configuration any number of roster entries can share, so ten operators who
need the same setup point at one file. A **tier** is what the preset's three
control personas *mean*. OSPREY itself enforces those two switches with
some care; the names ``readonly``, ``readwrite`` and ``admin``, and the fact
that there are three of them, belong to the preset. Rename a persona and
nothing in the framework notices; add a fourth and it inherits the floor and
the read-only default until its file says otherwise. :doc:`Build Profiles
<../../build-profiles>` covers what a persona file may contain and how the floor
is written so that a delta can lift it.

What makes it hold
==================

The persona file is the *input*. What a session actually meets when it tries
to do more than its tier allows is three layers, and the one that ultimately
holds is file ownership, not a permission list.

.. raw:: html
   :file: ../../../_diagrams/tier-layers.html

**Layer 1** is what makes the readonly tier certain. ``writes_enabled: false``
does not merely tell the agent not to write; it renders the control-system
write tools into a deny list that the agent runtime checks before any OSPREY
code runs, and no persona setting can subtract from that list.

**Layer 2** is the set of gates that give the agent a readable refusal —
``setup_patch`` denied by the floor, the Config panel and gallery refusing
with a 403, the Python executor refusing to write into the render zone in
either execution mode. Across all of them sits :ref:`the protected set
<config-protected-set>`: the files and config keys no agent-side writer may
rewrite on any tier.

**Layer 3** is what makes the readwrite-versus-admin line *true* rather than
asserted. In the container the rendered project is owned by root and the
agent runs as an unprivileged user, so a write that slipped every gate — or a
command the operator types at the terminal's own shell, which no permission
list sees — fails at the operating system. Only the admin image hands one
file, ``config.yml``, to the agent's user.

Two things are worth saying plainly, because the drawing would otherwise imply
more than is true:

.. note::

   **Readonly holds everywhere; readwrite-versus-admin holds in a container.**
   Run the same project on a bare host with ``osprey web`` and layer 3 is
   absent: the server runs as you, and the render is only as protected as the
   file ownership you give it. :ref:`The privilege split
   <containerize-privilege-split>` says what the container does and what a
   bare host would have to do instead.

.. note::

   **Admin is a trusted person, not a gated one.** Carol's ``config.yml`` is
   hers to write, so the protected set's refusal there is a gate, not a lock:
   her own shell can edit any key. The protected set defends against the
   *agent* rewriting its own safety layer; it does not defend against the
   deployment's administrator, and does not pretend to.

Watch it act
============

The boundary is enforced, not asserted — so you can watch it act. Open alice's
and bob's terminals and ask both agents to do the same two things:

**Read.** Ask either agent about a channel — a corrector setpoint, a BPM
reading. Both sessions answer identically: reads are ungated on both tiers.

**Write.** Ask each agent to change a setpoint. In alice's session the write
goes to a human approval prompt, then executes. In bob's session the same
request is **refused**: the write tool is denied in his project's rendered
permissions, and the refusal states plainly that writes are disabled in his
configuration.

Both agents carry the *same* tool surface — the readonly tier is not a
stripped-down agent that never heard of writing. It is the same agent whose
write path is switched off in its own project, which is exactly what you want
to demonstrate to a control room: the boundary holds at the enforcement layer,
not at the menu. (The readonly terminal's leaner look — no EVENTS/BLUESKY
tabs, chat-first layout — is presentation for the viewer tier, not the
boundary itself: the refusal above fires with or without it.)

What the admin tier really buys
===============================

Being able to edit the deployment is not the same as being able to change
anything, and it is worth being concrete about how much of an edit takes
effect while the stack is running.

**The protected set still refuses the admin login.** The keys that gate
writes, approval and limits — and the artifacts rendered from them — are
refused for every tier, this one included. Admin lifts a *tier floor*; it does
not open the safety layer. See :ref:`the protected set <config-protected-set>`
for what is protected and where a refusal is recorded.

**Most edits land on the next build.** The safety hook scripts run as a fresh
process per tool call, so each one reads ``config.yml`` as it stands at that
moment. Everything else — the terminal server, the tool servers, the agent's
rendered artifacts — reads its configuration once and caches it for the life
of the process, and nothing watches the file. So an edit outside those
hook-read keys takes effect on the next build and restart. The honest summary
is that the Config panel prepares a change; it is not a live control.

**A run-time edit is invisible to the drift check.** ``osprey up`` compares
``build/`` against ``profile.yml`` and the files that profile names — it never
fingerprints the rendered ``build/config.yml``. An edit the admin login makes
in the running deployment therefore leaves the drift check reading *in sync*,
because as far as it is concerned nothing about the profile changed. What does
record the edit is the copy every config write takes before it writes
anything, kept in the state zone at ``var/agent_data/config-backups/``: a copy
of the file as it stood immediately before the last write, one slot per file,
overwritten on each save — so it is a way back from the last change rather
than a history of them. Carry a change you want to keep back into
``profile.yml`` and rebuild, or the next build renders it away.

So the admin tier's real distinction is not a wider set of live knobs. It is
having the deployment-editing surfaces at all — ``setup-mode``, ``setup_patch``
and the Config panel — and being the login that drives the rebuild and restart
that make an edit real.

Checked, not merely conventional
================================

The one capability it would be worst to leave open — editing the deployment —
is guarded by the commands themselves, not only by the preset's choices.
``osprey profile validate`` and ``osprey build`` refuse a ``login: false``
entry that resolves to a persona holding either deployment-editing surface,
and refuse a privileged ``default_persona`` in a deployment that draws a tier
split at all; ``osprey up`` asks the same question of the render before it
starts anything. Each refusal names the user, or the persona, and the remedy.

.. dropdown:: Exactly what ``validate``, ``build`` and ``up`` refuse
   :icon: shield-check

   ``osprey profile validate`` and ``osprey build`` refuse any ``login: false``
   entry that resolves to a persona holding either deployment-editing surface
   — the agent's ``setup_patch`` tool or the web Config panel — naming the
   user. That refusal does not ask whether your profile drew a tier split: a
   profile that floors neither surface hands both of them to every persona it
   has, so an open terminal there is the most exposed version of this rather
   than an exempt one. What the split changes is the remedy the message can
   honestly offer. Where an unprivileged tier exists you are told to point the
   entry at it (or to give the entry a login); where none does, you are told
   to write the floor first — a ``claude_code.permissions.deny`` carrying
   ``mcp__osprey_workspace__setup_patch``, and
   ``web.config_panel.enabled: false``, in the profile's ``config:`` block —
   and to lift both only in the persona meant to hold them.

   A privileged ``default_persona`` is refused too, in a deployment that draws
   a privilege split at all — one whose profile floors those surfaces and lets
   a single persona lift them, which is what this preset ships. On a floorless
   profile that rule has no unprivileged tier to send the default to, so it
   stays quiet there and the entries actually exposed are the ones named. A
   persona named by either check that the command cannot read at all — a
   ``build_profile`` pointing outside ``personas/``, say — is refused rather
   than taken to hold nothing, naming the persona, the value it was given and
   the remedy — plus the path it tried, where the value resolved to one; where
   the unreadable persona is an open terminal's, that refusal too stands
   whatever the profile floors, so long as the deployment has a login wall for
   the entry to have opted out of.

   With ``auth.method: none`` there is no wall for an entry to be exempt from,
   so the exposure belongs to the deployment rather than to any one entry: it
   is reported as an advisory naming every privileged terminal instead of
   failing the build, and it is the one of these rules measured against the
   profile's own floor — a deployment that never drew a split would otherwise
   hear about every terminal it has, every time, for a posture it has always
   had. Both commands print advisories like this one with a ``⚠``, above their
   success line.

   ``osprey scaffold web-terminals lint`` asks the same questions of what was
   last *rendered*, and so does the render step itself. A ``login: false``
   exposure is an error there too; the message adds that this is what the last
   ``osprey build`` rendered, so a render made before the floor existed is
   refused until a rebuild puts the floor into it. A persona whose project has
   not been rendered yet is refused the same way where it is exposed, with
   ``osprey build`` as the remedy — or ``login: true`` for an entry that
   opted out of the wall — while behind a login it stays the plain warning it
   always was. A deployment that pulls its images from a registry has no
   persona render to read at all, so there only an open terminal is refused;
   its inherited default is judged where the deltas live, by ``osprey
   validate`` and ``osprey build``. A privileged ``default_persona`` in a
   render is advisory, because the entries it actually exposes are named by
   the rule that does block.

   ``osprey up`` asks the door question once more before it starts anything.
   A stack whose render serves an open privileged terminal, or one whose
   persona cannot be read there, does not come up — the refusal arrives with
   the other start-time problems, so one attempt reports them all — and the
   two advisories are printed beside it. Nothing else the lint finds stops a
   start: a duplicate port or a missing certificate belongs to the commands
   that author the render, and may already have been fixed by hand in it,
   while the open door is the one question those commands cannot answer for a
   tree that is about to be served.

Related pages
=============

- :ref:`The protected set <config-protected-set>` — the files and config keys
  no agent-side writer may rewrite, on any tier, and where a refusal is
  recorded.
- :ref:`The privilege split <containerize-privilege-split>` — what the
  container does with file ownership, and what a bare host does not.
- :ref:`What executed code may not change <python-executor-protected-paths>`
  — the Python executor's zone boundary, enforced in both execution modes.
- :ref:`Sandbox one session <web-terminal-session-posture>` — how an operator
  narrows a write-capable session for a while, never widens it.
- :doc:`../../build-profiles` — personas: what a delta file may contain, and how
  the floor is written so that one persona can lift it.
