Send and Retrieve Feedback
==========================

Anyone using the Web Terminal can report a problem, or ask for something, from
the terminal itself — no bug tracker account, no separate form. The **Feedback**
control sits in the utility cluster at the far end of the panel rail (bottom in
the default left-hand rail, right-hand end when the rail runs along the top),
next to the **Documentation** link. Clicking it opens a dialog with a text box,
a channel picker, and two attachment checkboxes.

Every submission is recorded on the deployment it was sent from, whichever
channel the sender picks. Facility staff read those records back with
``osprey feedback list`` and ``osprey feedback export`` — see
:ref:`feedback-retrieval` below.

Choose a channel
----------------

The dialog offers three ways out. The choice is the sender's, and switching
between them keeps the typed text and both checkboxes as they were.

.. list-table::
   :header-rows: 1
   :widths: 14 40 46

   * - Channel
     - What it does
     - Use it when
   * - **Local**
     - Writes the report to this deployment and nothing else. **Send** is the
       only button.
     - The report is for the people who run this deployment — the usual case in
       a control room, and the only one that works with outbound access
       blocked.
   * - **GitHub**
     - Also opens a prefilled new-issue tab on the configured repository, and
       puts the whole report — the text plus the session context — on the
       clipboard to paste in.
     - The report belongs upstream, and the sender has a GitHub account.
   * - **Email**
     - Also opens a prefilled mail draft to the configured address, again with
       the whole report on the clipboard.
     - There is a maintainer address to write to and no issue tracker in the
       loop.

GitHub and Email never send anything by themselves. OSPREY prefills a tab or a
draft in the sender's own browser and mail client; the sender still reviews it
and presses send. The server posts nothing off the host, on any channel.

Both outbound channels write the same record on the deployment as a Local send
does, so the facility keeps its own copy of every report even when the
conversation continues in a public issue.

If a channel is not configured for the deployment, its radio button still
appears but the action is refused with a short explanation rather than aiming
the report somewhere nobody reads. See :ref:`feedback-configuration`.

What gets attached
------------------

The dialog's ``(?)`` control opens a plain-language disclosure of exactly what
leaves the page. It says the same thing this section does, and it is the
authority the sender sees at the moment of deciding.

**Always sent**, with no checkbox to turn it off:

- the feedback text that was typed,
- a timestamp,
- the sender's username, when the deployment knows it (in a single-user or
  development setup this falls back to the account the server runs under).

**Deployment metadata** — the checkbox is **on** by default. It adds the OSPREY
version and the application name, and that is all it governs: turning it off
does not remove the text, the timestamp, or the username, which travel either
way.

**Session context** — the checkbox is **off** by default, and is greyed out with
"no session yet" when there is no live session to attach. It adds:

- the session's tool-call and agent event log,
- its chat history,
- the terminal scrollback,
- an artifact inventory listing **titles and ids only** — artifacts tagged to
  this session, plus artifacts created on this deployment during the same time
  window, which may include work from other terminal tabs or the chat panel.

That last clause is worth reading twice, and the dialog states it in those
words. OSPREY does not tag every artifact with the session that produced it, so
the inventory falls back to a time window: an artifact somebody else made in a
different tab during the same minutes can be listed. Only its title and id
appear — never its contents — but a sender attaching context to a **public**
GitHub issue should know that the list is not strictly their own work.

Session context is triaged newest-first and truncated to fit a size budget, so a
long session is attached in part rather than in full. The copy destined for a
GitHub issue or a mail draft is trimmed harder than the copy kept on the
deployment: the clipboard payload has to stay pasteable, while the local record
holds a much larger version.

Where the session had no readable transcript, the dialog says so and the
**outbound** copy — the clipboard payload behind a GitHub issue or a mail draft
— carries the text, the metadata and a one-line note in place of the four
context tiers, never a silently empty context. The deployment's own record is
not so trimmed: the terminal scrollback the browser uploaded is stored with it
either way. On-disk and outbound are two different things, and a sender deciding
what to attach should read the checkbox as governing both.

Copy session context
--------------------

On the GitHub and Email channels a second button, **Copy session context**,
puts the same trimmed context on the clipboard without sending anything and
without writing a record. It is enabled only when the session-context checkbox
is ticked and a session exists; otherwise it stays visible but greyed, so it
never looks like a control that disappeared.

Use it when the prefilled issue or draft is too small to carry the context
inline: the prefilled body already contains a ``paste the copied session context
here`` placeholder to drop it into.

.. _feedback-retrieval:

Read the feedback a deployment collected
----------------------------------------

Feedback is stored where the sender's own data lives, not in a central inbox.
Two verbs read it back, both run from the deployment repository (or with
``--repo DIRECTORY`` naming another one):

.. code-block:: bash

   osprey feedback list                        # one line per submission, newest first
   osprey feedback export --output feedback.json
   osprey feedback export > feedback.json      # same document, redirected

``list`` prints one row per submission: the record id, who sent it, when (UTC),
which channel it went out on, and the start of the message. It reads submission
headers only, so it stays quick on a large store. A deployment nobody has
written to yet prints one sentence rather than an empty table.

``export`` writes the whole record — who sent it, when, from which session and
which version of the application, how it went out, where it was truncated — and
the captured session alongside it, as one JSON array with one object per
submission. Records are written as they are read, so a large store never has to
fit in memory. With ``--output`` the file holds the export and everything else
stays on the terminal; without it, stdout carries the JSON document and nothing
else, with errors and the closing summary on standard error.

Where the records live
~~~~~~~~~~~~~~~~~~~~~~

The two deployment shapes are read in two different ways, and the command picks
the right one for you. What decides is whether the deployment has a web-terminal
roster — a non-empty ``modules.web_terminals.users`` list:

**Multi-user deployment.** Each person on the roster has their own web terminal
and their own workspace volume, and their feedback stays in it. Reading the
deployment's feedback starts a short-lived, read-only container on each person's
workspace in turn and reads the store through it. **The deployment's container
runtime has to be running**; with it stopped the command fails loudly instead of
reporting a deployment with no feedback in it. A workspace that cannot be read
is named on screen and the rest of the deployment is still listed or exported,
so one broken volume never hides everybody else's reports. The export's closing
summary distinguishes a workspace that contributed nothing from one that was
only partly readable. If *nothing at all* could be read, the command stops
rather than printing "no feedback".

A roster user's workspace volume is created when their web terminal container
first starts, so a repository that has been built but never brought up with
``osprey up`` has no volumes to read. Every user then reads as unreadable, and
the command stops with that rather than reporting an empty deployment.

**Single-user deployment.** There is one store, on this machine, under the
deployment's own agent-data directory, and it is read straight off disk — no
container, nothing to start first. This is also the shape a development setup
has.

.. warning::

   ``agent_data.base_dir`` is honoured differently by the two branches, and only
   one of them can warn you.

   The **single-user** reader resolves the store through that key, so relocating
   it relocates the store and the reader together — and a ``~``-relative value,
   which puts the data where the reader cannot follow, is refused loudly with
   that explanation.

   The **multi-user** reader never consults the key at all: it mounts each
   person's workspace volume and reads a fixed path inside it. A rostered
   deployment whose ``agent_data.base_dir`` is ``~``-relative writes its
   feedback into a different volume — one these commands do not mount — and
   ``osprey feedback list`` then exits ``0`` reporting that no feedback has been
   sent. Nothing detects that case, so on a multi-user deployment give
   ``agent_data.base_dir`` a path relative to the deployment repo, or an
   absolute one — anything but ``~``-relative — and do not read an empty listing
   as proof that nobody has written in.

Exit status is ``0`` when the read succeeded — including a deployment that has
collected no feedback — and non-zero when the repository has no build, when
every workspace failed to read, and when an export stopped part-way. A part-way
export still parses: it holds what was read, and the summary says how much that
was. On a single-user deployment a ``~``-relative ``agent_data.base_dir`` is
also non-zero; on a multi-user one it is not detected at all, per the warning
above.

Keeping the store bounded
~~~~~~~~~~~~~~~~~~~~~~~~~

Session contexts are much larger than the reports they belong to, so the store
has a ceiling (``web.feedback.max_store_bytes``, 256 MB by default). When it is
exceeded, the oldest saved **contexts** are dropped; their submissions are
never dropped. An export marks a submission whose context was dropped this way
with ``context_pruned``, and one missing its context for any other reason with
``context_missing`` — the submission itself is exported either way, so the
history of what people reported stays complete.

.. _feedback-configuration:

Point the channels at your facility
-----------------------------------

Three string keys under ``web`` aim the two rail controls, and a fourth bounds
the store they fill:

.. code-block:: yaml

   web:
     docs_url: https://als-apg.github.io/osprey   # the Documentation control
     feedback:
       github_repo: als-apg/osprey                # owner/repo for the GitHub channel
       email: thellert@lbl.gov                    # recipient of the mail draft
       max_store_bytes: 268435456                 # 256 MB store ceiling

Leaving a key out gives you the shipped default above. Setting one of the three
**string** keys to an **explicitly blank** value (``docs_url: ""``) means "this
deployment has no such target" and removes the affordance: an air-gapped control
room blanks ``web.docs_url`` and gets no documentation link rather than one that
opens a dead tab, and blanking ``web.feedback.github_repo`` retires the GitHub
channel instead of aiming reports at the upstream maintainers' tracker.
``max_store_bytes`` has no such posture — it wants a positive byte count, and
anything else (blank included) is logged and replaced by the default. The Local
channel is always available and needs no configuration.

:doc:`web-terminal/operate` carries the full table, including what a key written
with no value at all does.

.. note::

   The feedback dialog is behind the terminal's own gate: only a browser
   holding a session can open it, the same as for any other terminal action. It
   draws no line *within* a session, though — anyone sitting at a signed-in
   terminal can send a report — so read a stored report as coming from that
   terminal, not from a named person.

Named future work
-----------------

Two deliberate gaps, both recorded so nobody has to rediscover them:

- **Downloadable ``.txt`` context attachments.** The session context travels as
  a trimmed clipboard payload today. If teams report that the trimmed bundle is
  not enough for a maintainer to act on, a downloadable full attachment is the
  planned next step.
- **Locally hosted documentation.** ``web.docs_url`` points at whatever site you
  give it, so a facility can already host its own copy — but OSPREY does not yet
  *build and serve* a local documentation bundle for you. That key is the
  plug-in point when it does.

.. seealso::

   :doc:`web-terminal/operate`
      Running the terminal the feedback dialog lives in, and the full ``web``
      settings table.

   :doc:`multi-user`
      Per-user web terminals and the roster that decides how feedback is stored
      and read.

   :doc:`/cli-reference/index`
      ``osprey feedback list`` and ``osprey feedback export`` with every option.
