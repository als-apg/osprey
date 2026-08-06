====================
Deploy a Chat Bridge
====================

How to let your team ask the OSPREY agent questions from a Nextcloud Talk chat
room and get answers — including plots and files — back in the same room.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What the Nextcloud Talk bridge does, and what it deliberately does not do
   - How to create the bot account the bridge speaks as
   - How to enable the bridge in a build profile and bring it up
   - Which questions the bridge answers, and who is allowed to ask

   **Prerequisites:** A Nextcloud instance with the Talk app, permission to
   create a user account on it, and a project whose profile has a ``dispatch:``
   block (see :doc:`event-dispatch`). Docker or Podman for the container path.

Overview
========

The bridge turns a chat room into a way of talking to the agent. Someone
mentions the bot in a Talk room, the bridge hands that question to the event
dispatch pipeline, and the answer is posted back as a reply in the room.

It is a **poller, not a server**: it asks Nextcloud for new messages and waits
for them, so it opens no network port of its own and nothing has to be able to
reach it. It only makes outbound calls — to Nextcloud and to the dispatcher.

.. mermaid::

   flowchart LR
       T[Nextcloud Talk room] -->|message mentioning the bot| B[nextcloud-bridge]
       B -->|POST /webhook/trigger| D[Event dispatcher]
       D --> W[Dispatch worker]
       W -->|answer + any files| B
       B -->|reply in the room| T

The bridge is the piece that remembers things. Each question is recorded before
it is dispatched, so a restart in the middle of one does not answer it twice or
drop it; the recent exchanges in a conversation travel with each new question, so
"now plot that over 24 hours" makes sense; and each room's reading position is
saved, so messages posted while the bridge was down are picked up rather than
missed.

Enable It in a Profile
======================

Add a ``nextcloud_bridge:`` block to your build profile. The only setting is
which dispatcher trigger the bridge fires — that trigger decides what the agent
is allowed to do with a chat question:

.. code-block:: yaml

   nextcloud_bridge:
     trigger: nextcloud-question    # default; must exist in your triggers file

   env:
     required:
       - NEXTCLOUD_BASE_URL
       - NEXTCLOUD_BOT_ACCOUNT
       - NEXTCLOUD_APP_PASSWORD
       - NEXTCLOUD_ROOMS

Rooms and credentials are **not** profile settings. They are runtime values you
supply, because they differ per deployment and the password must never be baked
into a build. Listing them under ``env.required`` makes ``osprey build`` write
them into the project's ``.env`` (created mode ``0600``, readable only by you)
for you to fill in.

Two mistakes are caught at **build** time rather than at runtime: declaring the
bridge without a ``dispatch:`` block, and naming a trigger that your triggers
file does not declare. Both fail the build with a message naming the problem,
instead of producing a project that deploys and then fails on every message.

Runtime settings
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Meaning
   * - ``NEXTCLOUD_BASE_URL``
     - Your Nextcloud instance, with no trailing slash, e.g.
       ``https://cloud.example.org``. HTTPS is the production expectation; a
       plain ``http://`` URL still starts but logs a warning at startup, because
       the bot's password and every message cross the network unencrypted.
   * - ``NEXTCLOUD_BOT_ACCOUNT``
     - The Nextcloud user id the bridge signs in as — the account people mention
       to ask a question.
   * - ``NEXTCLOUD_APP_PASSWORD``
     - An app password for that account (not the account's login password).
   * - ``NEXTCLOUD_ROOMS``
     - Comma-separated Talk room tokens to watch. A room's token is the last
       part of its URL: in ``…/call/a1b2c3d4`` the token is ``a1b2c3d4``.
   * - ``DISPATCH_TRIGGER``
     - The trigger to fire. Filled in for you from the profile block above.
   * - ``EVENT_DISPATCHER_TOKEN``
     - Shared secret for talking to the dispatcher. ``osprey deploy up``
       generates it when unset.
   * - ``DISPATCH_WORKER_TOKEN``
     - Shared secret for talking to the worker. Also auto-generated.

Bring It Up
===========

**1. Create the bot account.** In Nextcloud, add a regular user for the agent to
speak as — the display name is what your team sees replying, so make it obvious
(for example ``OSPREY agent``). Sign in as that user once, then create an app
password under *Settings → Security*. An app password can be revoked on its own
without disturbing the account, which is why the bridge uses one.

**2. Invite it to the rooms it should serve.** The bot only sees rooms it is a
member of. Add it to each room you want served, and collect those rooms' tokens.

**3. Fill in ``.env``** with the four values from the table above:

.. code-block:: bash

   NEXTCLOUD_BASE_URL=https://cloud.example.org
   NEXTCLOUD_BOT_ACCOUNT=osprey-agent
   NEXTCLOUD_APP_PASSWORD=xxxxx-xxxxx-xxxxx-xxxxx-xxxxx
   NEXTCLOUD_ROOMS=a1b2c3d4,e5f6g7h8

**4. Bring the stack up.** The bridge is registered in ``deployed_services``, so
it starts with everything else:

.. code-block:: bash

   osprey deploy up        # add --dev to bake in a local osprey checkout

Then mention the bot in one of the listed rooms and ask it something. If nothing
happens, check the service's logs first: a missing credential stops the bridge at
startup with the missing variable named, rather than letting it run in a broken
state.

.. important::

   The bridge keeps what it remembers — which questions it has answered, recent
   conversation, and each room's reading position — in a named volume mounted at
   ``/data``. Do not remove that volume. Without it, a restart forgets everything
   and the room's history is either replayed from the beginning or skipped past.

Who Can Ask, and What Is Shared
===============================

**Room membership is the access gate.** The bridge answers questions from any
room listed in ``NEXTCLOUD_ROOMS``, which means anyone who can post in one of
those rooms can reach the agent. Choose those rooms as deliberately as you would
choose who gets an account: adding a room grants its members access, and
Nextcloud — not OSPREY — decides who is in it.

**In a group room, only messages that mention the bot are answered.** Everything
else is ignored, so the bridge can sit in a busy room without reacting to
ordinary conversation. The check is made against the mention Talk itself records,
not against the message text, so writing the bot's name in passing does not
trigger it, and neither does ``@all``. In a one-to-one conversation there is
nobody else to address, so every message counts as a question. The bridge also
ignores its own messages, which is what stops it answering itself in a loop.

**What the agent may do with a chat question is set by the trigger, not by the
question.** The trigger's tool allowlist lives in your triggers file, which is
mounted **read-only** into the dispatcher — so a request arriving through chat
cannot widen what the agent is permitted to do, and the worker's own denylist
applies on top of it (see :doc:`event-dispatch`).

**Files are shared into the room, never published.** When an answer includes a
plot or a file, the bridge uploads it to the bot's own storage and shares it with
the room, so only that room's members can open it. **No world-readable link is
ever created** — there is no public URL to leak, forward, or index.

.. seealso::

   :doc:`event-dispatch`
       The dispatcher and worker the bridge hands questions to, and how to
       author the trigger it fires.

   :doc:`deploy-project`
       Container deployment mechanics for all OSPREY services, including image
       overrides.

   :doc:`../cli-reference/index`
       Full ``osprey build`` and ``osprey deploy`` reference.
