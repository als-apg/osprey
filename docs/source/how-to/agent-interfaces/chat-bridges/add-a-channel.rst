.. _how-to-add-a-channel:

=========================
Add Your Own Chat Service
=========================

Nextcloud Talk and Google Chat ship with Osprey. If your team lives somewhere
else — Slack, Mattermost, Zulip, or plain email — this page explains what
connecting it involves.

What You Get, and What You Write
================================

Almost everything a bridge does is already written and shared — deduplication,
conversation history, retries, honest give-up, and clean recovery after a crash
all live in the engine and are not rewritten per service.

What you write is the part that is genuinely different from one chat service to
the next, and it comes to two things:

**An arrival loop** — how new messages reach you. This is the one piece that
does not generalise, which is why the engine has no opinion about it. The two
shipped bridges show the two common shapes: Nextcloud Talk runs one long-poll
thread per room, and Google Chat holds a single streaming pull on a message
queue.

**A set of small functions** — how to read a message, post a reply, send a file,
and acknowledge that a question arrived. These are gathered in one place, a class
called ``ChannelOps``. Your service's version of it is the only file that knows
anything about your chat service's API.

Four Decisions
==============

Before writing code, work out the answers to these. They shape everything else.

**How do messages arrive?** A queue you pull from, a connection you hold open, a
loop that asks for new messages, a mailbox you poll. If your service can only
*push* to a public web address, note that both shipped bridges deliberately avoid
that, and say so — it costs you the no-open-port property.

**How do you address a reply?** The engine never learns what a "destination" is.
Nextcloud posts to a room token, Google Chat to a space and thread, email replies
to a sender with the right headers. Your adapter records whatever it needs when
the question arrives and gets it handed back when it is time to answer.

**What does "on it" look like?** The agent may take a while, so the bridge
signals that it heard you before the long wait begins. Google Chat and Talk post
a short message; Slack could add an emoji reaction. Email has no natural way to
do this, and that is a valid answer — the step is allowed to do nothing.

**How do files come back?** This is a safety decision, not a technical detail.
Nextcloud Talk shares a plot with the room and nobody else. Google Chat has to
publish it to a public web address, because Google itself must be able to fetch
the image to display it. Find out which of those your service forces on you
before you build anything, and write it down where deployers will see it.

The Functions You Implement
===========================

``ChannelOps`` is a list of about a dozen small functions: parse an incoming
message, resolve a quoted message, post the acknowledgement, download any
attached files, post the answer, deliver the output files, post the "parked,
will retry" notice, post the "gave up" notice, and a couple of helpers for
grouping repeated questions.

The important thing about them is not what they do but **how they are allowed to
fail**, and it is not the same for all of them. Getting this wrong is the main
way a new bridge ends up looking healthy while quietly losing answers.

.. dropdown:: How each function must behave when it fails
   :icon: alert

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Function
        - On failure it must…
      * - ``parse_event``
        - Return "not for us" rather than raise. Bad input is normal.
      * - ``resolve_reply_context``
        - Return nothing. Quoted context is a bonus, never essential.
      * - ``post_ack``
        - Swallow the error. A missing acknowledgement is cosmetic.
      * - ``download_inputs``
        - Record the file as skipped. Never raise.
      * - ``post_answer``
        - **Raise.** The engine turns the raise into "keep it queued and try
          again". Swallowing it silently throws the answer away.
      * - ``deliver_files``
        - **Never raise.** The answer text has already been posted; a failed
          upload must not undo it. Degrade to text only.
      * - ``post_queued``
        - May raise. The question is parked before the notice is posted, so a
          lost notice never loses the question.
      * - ``post_giveup``
        - **Raise.** The question is only marked finished once the notice lands.
          A lost give-up notice is silence, which is worse than a rare duplicate.
      * - ``post_superseded``
        - Swallow the error. The decision it announces is already committed.

   The pattern behind these: anything the user must see is required to raise so
   the engine can retry it, and anything cosmetic is required not to raise so it
   cannot undo work that already succeeded.

Wiring It Into a Build
======================

A new bridge is not a drop-in folder. Alongside the adapter package itself, it
has to be taught to the build system so that a profile can switch it on and
``osprey up`` can start it.

.. dropdown:: The files a new channel touches
   :icon: file-directory

   .. list-table::
      :header-rows: 1
      :widths: 45 55

      * - Location
        - What goes there
      * - ``src/osprey/bridges/<channel>/``
        - The adapter: its config, its API client, its ``ChannelOps``, its
          arrival loop, and a ``__main__.py`` that reads the environment,
          refuses to start on an incomplete one, and hands the arrival loop to
          the shared ``run_forever``.
      * - ``src/osprey/templates/services/<channel>_bridge/``
        - ``Dockerfile`` and ``docker-compose.yml.j2`` for the service, copied
          into a project at build time.
      * - ``cli/build_profile_model.py``
        - The profile block's shape, plus its validation. The existing
          ``_validate_chat_bridge`` helper already covers the two checks every
          bridge needs — that a ``dispatch:`` block exists, and that the named
          trigger is really declared.
      * - ``cli/build_profile_load.py``
        - Reading that block out of a profile file.
      * - ``cli/build_profile_emit.py``
        - The commented example block written into generated profiles.
      * - ``cli/build_injectors.py``
        - The injector: copy the service template into the project, register the
          service in ``config.yml``, and add it to ``deployed_services``.

   The two shipped injectors are deliberate mirrors of each other, so the second
   one is largely the first with the names changed.

Two Sketches
============

**Slack** is close to the shipped bridges. Socket Mode gives you a connection
your app opens outward, so you keep the no-open-port property. Replies go into a
thread, emoji reactions make a natural acknowledgement, and uploaded files are
shared with the channel rather than published — so on the question that matters
most, Slack behaves like Nextcloud Talk, not like Google Chat.

**Email** is the useful stress test, because it breaks assumptions chat makes.
Messages arrive by polling a mailbox. There is no acknowledgement idiom worth
having, so that step does nothing. A reply is addressed by headers rather than by
a room. And answer text and attachments go out together in one message, instead
of as a reply followed by separate file uploads — which is exactly why the
engine treats posting the answer and delivering the files as two steps it is
allowed to fill differently.

.. seealso::

   :doc:`nextcloud-talk` and :doc:`google-chat`
       The two shipped bridges, and the two arrival shapes worth copying.

   :doc:`../add-mcp-server`
       The other way to extend what the agent can reach — new tools rather than
       a new place to ask from.
