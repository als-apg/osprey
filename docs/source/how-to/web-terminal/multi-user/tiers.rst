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
   - That a tier is nothing more than a small file under ``tiers/`` — and
     what that means when you rename one or add one
   - Where the boundary is actually enforced, and the two places it is a
     matter of trust rather than enforcement

   **Prerequisites:** none. To watch the boundary act you need the stack
   running — see :doc:`index`.

Two questions decide a tier
===========================

A tier is the answer to two independent questions.

**How far may this login write?** That is the tier key,
``control_system.writes: never | asked | allowed`` (:ref:`write-safety-config`),
the furthest position the login's switch may take. On ``never`` every write
surface refuses — channel writes, read-write Python execution, plan arming, all
of it. Otherwise the login may write to any machine the deployment configures,
but only where its own switch is armed: every machine starts at off, and the
person moves one from the chip. A write still passes the writes-check hook and
the per-channel limits, and at *asked* a human approval prompt.

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
   * - **Writes never**
     - **readonly**
     - —
   * - **Writes asked** (switch per machine, every write prompts)
     - **readwrite**
     - —
   * - **Writes allowed** (switch per machine, may run unprompted)
     - —
     - **admin**

The empty cells are not forbidden by anything in OSPREY — a tier file could be
written for any of them — the preset simply does not ship one.

.. _multi-user-tiers:

The three tiers the preset ships
================================

Each tier renders a self-contained OSPREY project with its **own**
permissions, because permissions are a property of a project's ``config.yml``
— one agent, built once per access level, not one terminal with a UI toggle.

.. list-table::
   :header-rows: 1
   :widths: 16 26 26 32

   * - Tier (login)
     - Control-system writes
     - Editing the deployment
     - What the screen looks like
   * - **readonly** (bob)
     - ``never``. Every write surface refuses
     - No. The ``setup_patch`` tool is denied, the Config panel is off, and
       the scaffold gallery is readable but not writable
     - Operator chat, without the EVENTS and BLUESKY panels
   * - **readwrite** (alice)
     - ``asked``. Alice writes to a machine once she moves its switch to
       *asked* from the chip, and nowhere else. A channel write still passes the
       writes-check hook, the per-channel limits, and a human approval prompt
       before it executes
     - No — the same floor as readonly
     - Operator chat, with the EVENTS and BLUESKY panels
   * - **admin** (carol)
     - ``allowed``: as above, and the switch may also go to *allowed*
     - Yes, and only here: the ``setup-mode`` skill, the ``setup_patch``
       tool, the web Config panel, and the gallery's edit, create and delete
       surfaces
     - Operator chat, with the EVENTS and BLUESKY panels, plus
       the Config panel

.. _multi-user-tiers-posture:

Read the write column one row at a time rather than as a single statement
about the deployment. The posture is a property of the **session**, not of the
person holding it: which teammates get a write-capable login is your roster's
call, and the point is that the framework provisions genuinely different
postures out of one deployment.

Only the read-only login is built without the EVENTS and BLUESKY panels. The
base profile declares neither, and the two write-capable deltas,
``tiers/readwrite.yml`` and ``tiers/admin.yml``, each add both:
queueing a plan and watching its results belong on a card that can arm a
write. The admin card is the read-write operator desk plus deployment editing.

**The default stays on the safe side.** ``default_tier`` is ``readonly``,
so a roster entry added in a hurry with no ``tier`` of its own gets the
read-only tier.

.. dropdown:: The exact tool names
   :icon: code

   The same table, as the rendered projects spell it. The single-user column
   is the one worth knowing: a plain ``osprey init --preset control-assistant``
   render is readwrite at the machine and floored at the deployment — no
   Config tab, no gallery writes. It opens on the simulator with every switch
   off. Admin in a single-user deployment means editing ``profile.yml`` and
   rebuilding.

   .. list-table::
      :header-rows: 1
      :widths: 22 20 20 20 18

      * - Capability
        - readonly
        - readwrite
        - admin
        - single-user base
      * - ``control_system.writes``
        - ``never``
        - ``asked``
        - ``allowed``
        - ``asked``
      * - ``mcp__controls__channel_write``
        - kill-switch deny
        - approval hook prompts at asked; refused ``switch_off`` at off
        - as readwrite; runs unprompted at allowed
        - as readwrite
      * - Bluesky arming tools
        - kill-switch deny
        - ask; refused ``switch_off`` where the lane's machine is off
        - as readwrite
        - as readwrite
      * - ``mcp__python__execute``
        - read-only kernel only
        - both kernels; a read-write run is refused where the switch is off
        - as readwrite
        - as readwrite
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
      * - Control-target chip: move a machine to *asked*
        - locked (*kept read-only by the deployment*)
        - confirm modal
        - as readwrite
        - confirm modal
      * - Control-target chip: move a machine to *allowed*
        - locked
        - disabled (*your tier caps writes at asked*)
        - confirm modal, every time
        - disabled
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

   The shared logins — ``logbook``, ``knowledge`` and ``channel-finder`` — are
   not in the table: every login ``shared:`` names renders
   ``control_system.writes: never`` and closed to deployment editing, whatever
   its file or the host sets.

A tier is a file under ``tiers/``
=================================

A tier is an access level of the deployment's own agent, written as a small
file.

.. raw:: html
   :file: ../../../_diagrams/tier-render.html

Each file under ``tiers/`` holds only that tier's **differences** from
``profile.yml``. The base profile carries the floor — the three
deployment-editing surfaces switched off — and a tier adds what it needs on
top: ``readonly.yml`` sets ``control_system.writes: never``, ``readwrite.yml``
sets it ``asked``, ``admin.yml`` sets it ``allowed`` and lifts the floor.
A tier adds lists by name and its config keys win; it never subtracts and never
carries a deployment fact such as services, data or ports. ``osprey build``
lays each tier over the base and renders it into its own project, and
``osprey up`` builds one container image from each. Every arrow in the drawing
is a build step, never a runtime lookup: the web server never asks *who is
this user and what tier are they?* Identity lives in the roster and the login
wall; capability lives in the render.

Any number of roster entries can share a tier, so ten operators who need the
same setup point at one file. OSPREY enforces the two switches with some care;
the names ``readonly``, ``readwrite`` and ``admin``, and the fact that there
are three of them, belong to the preset. Rename a tier and nothing in the
framework notices; add a fourth and it inherits
the floor until its file says otherwise. :doc:`Build Profiles
<../../build-profiles>` covers what a tier file may contain and how the floor
is written so that a tier can lift it.

A **shared login** is the same kind of file, named by
``modules.web_terminals.shared`` and opened from a card everyone on the roster
can use. A shared card never writes: the build seals its render
``control_system.writes: never`` whatever the file says. The bundled
``logbook``, ``knowledge`` and ``channel-finder`` logins say ``lists: own``, so
their lists are the whole selection rather than additions to the
deployment's. A login's ``apps:`` block picks which of the deployment's
applications its card shows; a shown application writes only where the
login's tier is ``asked`` or ``allowed`` and the operator's switch is on for
that machine. :doc:`Build Profiles <../../build-profiles>` lists every key a
login file may carry, and ``osprey init --revalidate`` moves a login an older
repository keeps under ``apps/`` into ``tiers/``.

A roster entry can also name a **role** instead of a tier, and let the role
carry the mapping — ``operator: {tier: readwrite}`` written once, rather than
a tier pinned on every entry. A role resolves to a tier before anything is
built, so which tier a card runs and which container it opens are the same
either way. What the role adds is that
it survives the login: it travels with the session, reaches the terminal as
identity headers, is shown beside the user's name in the terminal's session
menu — together with where it came from, the roster entry or the identity
provider's claim — and is named on the login record in the audit trail. A
session that signed in before its role was set goes on showing the role on its
own, without its origin, until the person logs in again. Where a facility runs
single sign-on, the provider's own groups can be what decides it — see
:ref:`Let single sign-on pick the tier <multi-user-role-from-sso>`.

What makes it hold
==================

The tier file is the *input*. What a session actually meets when it tries
to do more than its tier allows is three layers, and the one that ultimately
holds is file ownership, not a permission list.

.. raw:: html
   :file: ../../../_diagrams/tier-layers.html

**Layer 1** is what makes the readonly tier certain. A ``never`` tier does not
merely tell the agent not to write; it renders the framework servers'
write-gated tools into a deny list that the agent runtime checks before any
OSPREY code runs, and no tier setting can subtract from that list. Tools a
project adds under ``control_system.write_tools`` are not in that list in any
render — they are refused by the writes-check hook, which is Layer 2.

An armed tier cannot get that static deny: the switch changes at run time,
so the boundary is carried per call instead — the safety hook asks the verdict,
and the connector asks it again behind it. Layer 1 is the stronger guarantee,
which is why the read-only tier, and not a write-capable one, is the one to hand
out when the requirement is "this login can never move anything". A shell is
denied on an armed tier, and lifting it fails the build.

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

The boundary is enforced, not asserted — so you can watch it act. Both sessions
open on the simulator, which is the target that decides the outcome below. Open
alice's and bob's terminals and ask both agents to do the same two things:

**Read.** Ask either agent about a channel — a corrector setpoint, a BPM
reading. Both sessions answer identically: reads are ungated on both tiers.

**Write.** Ask each agent to change a setpoint on the simulator. In alice's
session the write is refused: her switch is off, and the refusal says to turn
it on from the chip. She moves the simulator to *asked*, asks again, and the
write goes to a human approval prompt, then executes. In bob's session the
same request is **refused** whatever he does: the write tool is denied in his
project's rendered permissions, and his chip has no switch to turn on.

**Switch.** In alice's terminal, use the control-target chip to switch to the
Live machine or the Rehearsal, and ask for the same write. It is refused —
by the safety hook and the connector, not by a rendered deny — because her
switch for that machine is still off.

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
than a history of them. The *history* is in :ref:`the audit trail
<reference-audit-trail>`: every request that changes something through a web API
leaves a line naming the route and who was acting, and a refused protected key
leaves one of its own. Neither carries the values that were written — that is
what the backup copy is for. Between them: the trail says an edit happened and
who made it, the backup says what the file looked like just before, and the
drift check says nothing at all. Carry a change you want to keep back into
``profile.yml`` and rebuild, or the next build renders it away.

**It is also the tier that can read the audit trail from inside a container.**
``GET /api/audit/recent`` sits behind the same switch as the Config panel, and
returns the newest safety records from *that container's own* subdirectory —
never another user's, which no container can reach. A view across the whole
deployment is the deploy host's shell, not a tier.

So the admin tier's real distinction is not a wider set of live knobs. It is
having the deployment-editing surfaces at all — ``setup-mode``, ``setup_patch``
and the Config panel — and being the login that drives the rebuild and restart
that make an edit real.

Checked, not merely conventional
================================

The one capability it would be worst to leave open — editing the deployment —
is guarded by the commands themselves, not only by the preset's choices.
``osprey profile validate`` and ``osprey build`` refuse a shared card — one
whose ``access:`` admits anybody but the card's own login — that resolves to
a tier holding any deployment-editing
surface, and refuse a privileged ``default_tier`` in a deployment that
draws a tier split at all; ``osprey up`` asks the same question of the render
before it starts anything. Each refusal names the user, or the tier, and
the remedy.

.. dropdown:: Exactly what ``validate``, ``build`` and ``up`` refuse
   :icon: shield-check

   ``osprey profile validate`` and ``osprey build`` refuse any shared card
   that resolves to a tier holding any of the three deployment-editing
   surfaces — the agent's ``setup_patch`` tool, the web Config panel or the
   scaffold gallery's editors — naming the user.
   Shared means the card's ``access:`` admits anybody but its own login,
   whether that is ``any``, one ``domain:`` or a single ``user:``; the
   question the rule asks is how far the card reaches, not which words say
   so. That refusal does not ask whether your
   profile drew a tier split: a profile that floors none of them hands every
   one to every tier it has, so a shared card there is the most exposed
   version of this rather than an exempt one. The remedy is the card's own
   key — ``access: own``, or dropping the key — or pointing the entry at a
   tier that holds none of them. A ``shared:`` card never reaches this rule:
   the build closes all three surfaces on every shared login, whatever the
   host sets.

   A privileged ``default_tier`` is refused too, in a deployment that draws
   a privilege split at all — one whose profile floors those surfaces and lets
   a single tier lift them, which is what this preset ships. On a floorless
   profile that rule has no unprivileged tier to send the default to, so it
   stays quiet there and the entries actually exposed are the ones named. A
   tier named by either check that the command cannot read at all — a name
   with no file under ``tiers/``, say, or a render whose ``project_path``
   holds no ``config.yml`` — is refused rather
   than taken to hold nothing, naming the tier, the value it was given and
   the remedy — plus the path it tried, where the value resolved to one; where
   the unreadable tier is a shared card's, that refusal too stands whatever
   the profile floors, so long as the deployment has a login wall for the card
   to be shared behind.

   Where the deployment has no login page at all — ``auth.method`` of
   ``token`` (the default) or ``none`` — sharing changes nothing, so the
   exposure belongs to the deployment rather than to any one entry: it is
   reported as an advisory naming every privileged terminal
   instead of failing the build, and it is the one of these rules measured against the
   profile's own floor — a deployment that never drew a split would otherwise
   hear about every terminal it has, every time, for a posture it has always
   had. Both commands print advisories like this one with a ``⚠``, above their
   success line.

   ``osprey scaffold web-terminals lint`` asks the same questions of what was
   last *rendered*, and so does the render step itself. A shared privileged
   card is an error there too. A tier whose project has not been rendered
   yet is refused the same way where a shared card resolves to it, with
   ``osprey build`` as the remedy — or ``access: own`` for the card — while
   on an own card it stays the plain warning it always was. A deployment that
   pulls its images from a registry has no tier render to read at all, so
   there only a shared card is refused;
   its inherited default is judged where the deltas live, by ``osprey
   validate`` and ``osprey build``. A privileged ``default_tier`` in a
   render is advisory, because the entries it actually exposes are named by
   the rule that does block.

   ``osprey up`` asks the door question once more before it starts anything.
   A stack whose render shares a privileged card, or one whose tier cannot
   be read there, does not come up — the refusal arrives with
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
- :ref:`The audit trail <reference-audit-trail>` — what every tier's sessions
  leave behind, who can read it, and what it does not promise.
- :ref:`The privilege split <containerize-privilege-split>` — what the
  container does with file ownership, and what a bare host does not.
- :ref:`What executed code may not change <python-executor-protected-paths>`
  — the Python executor's zone boundary, enforced in both execution modes.
- :ref:`The control-target chip <web-terminal-session-posture>` — how an
  operator takes writes away from one control target for their own session,
  and can never gain more than the tier permits.
- :doc:`../../build-profiles` — logins: what a login file may contain,
  and how the floor is written so that one tier can lift it.
