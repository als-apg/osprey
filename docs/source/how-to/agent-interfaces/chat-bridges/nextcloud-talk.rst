.. _how-to-nextcloud-talk:

==============
Nextcloud Talk
==============

How to let your team ask the Osprey agent questions from a Nextcloud Talk room,
and get answers, plots, and files back in the same room.

.. dropdown:: Before you start
   :color: info
   :icon: checklist

   - A project whose build profile has a ``dispatch:`` block — see
     :doc:`../event-dispatch`.
   - Docker or Podman, for the container path.
   - A Nextcloud instance with the Talk app, and permission to create a user
     account on it.

Overview
========

The bridge turns a Talk room into a way of talking to the agent. Someone
mentions the bot in the room, the bridge hands that question to the
:doc:`event dispatch pipeline <../event-dispatch>`, and the answer is posted
back as a reply in the room.

It is a **poller, not a server**: it asks Nextcloud for new messages and waits
for them, making only outbound calls.

Like every bridge, it :ref:`remembers each question and conversation
<bridge-memory>`. What is particular to Talk is that each room's reading position
is saved too, so messages posted while the bridge was down are picked up rather
than missed.

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
into a build. Listing them under ``env.required`` documents them in
``.env.example``; fill the values into the **profile's** ``.env``, which is
where a secret survives a rebuild. The build derives the project's ``.env``
from it (created mode ``0600``, readable only by you).

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
     - Shared secret for talking to the dispatcher. ``osprey up``
       generates it when unset.
   * - ``DISPATCH_WORKER_TOKEN``
     - Shared secret for talking to the worker. Also auto-generated.

.. dropdown:: Optional settings you can usually leave alone
   :icon: gear

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Variable
        - Meaning
      * - ``DISPATCHER_URL``, ``WORKER_URL``
        - Where the dispatcher and worker are. Filled in for you when they run
          in the same stack; when they run elsewhere you must set them, and the
          bridge refuses to start without them.
      * - ``DISPATCH_TIMEOUT_SEC``
        - How long the worker may spend on one run. Comes from your project
          configuration, so raising it raises it for both halves at once.
      * - ``POLL_BUDGET``
        - How long the bridge waits for an answer before giving up on it.
          Defaults to 30 seconds more than the worker's own limit, and may never
          be less than that limit — the bridge refuses to start if it is.
      * - ``POLL_INTERVAL``
        - Seconds between checks on an answer in progress (default 2).
      * - ``DRAIN_INTERVAL``
        - Seconds between sweeps of the queue of questions that could not be
          handed off yet (default 60).
      * - ``RETRY_MIN_AGE``
        - How long a failed hand-off is held before it is retried, so a brief
          outage has time to clear (default 20 minutes).
      * - ``RETRY_GIVE_UP``
        - Age at which a question that still cannot be handed off is abandoned
          (default 48 hours).
      * - ``RETRY_LIFETIME_CAP``
        - Hard ceiling on how long anything may sit in that queue, whatever its
          state (default 7 days).
      * - ``BRIDGE_TRUST_ENV``
        - Set to ``1`` only if this host's outbound calls must go through your
          site's web proxy. Off by default, so a proxy inherited from a shell or
          a CI runner cannot quietly place itself in front of Nextcloud.
      * - ``DEDUP_PATH``, ``HISTORY_PATH``
        - Where the bridge keeps what it remembers. Both default to files under
          ``/data``, its own volume; change them only if you deliberately
          relocate that state.
      * - ``TZ``
        - Timezone, taken from your project configuration so timestamps match
          the rest of the stack.

Bring It Up
===========

**1. Create the bot account.** In Nextcloud, add a regular user for the agent to
speak as — the display name is what your team sees replying, so make it obvious
(for example ``OSPREY agent``). Sign in as that user once, then create an app
password under *Settings → Security*. An app password can be revoked on its own
without disturbing the account, which is why the bridge uses one.

**2. Invite it to the rooms it should serve.** The bot only sees rooms it is a
member of. Add it to each room you want served, and collect those rooms' tokens.

**3. Fill in the environment file.** Set the four values from the table above in
the **profile's** ``.env`` (the build derives the project's ``.env`` from it):

.. code-block:: bash

   NEXTCLOUD_BASE_URL=https://cloud.example.org
   NEXTCLOUD_BOT_ACCOUNT=osprey-agent
   NEXTCLOUD_APP_PASSWORD=xxxxx-xxxxx-xxxxx-xxxxx-xxxxx
   NEXTCLOUD_ROOMS=a1b2c3d4,e5f6g7h8

**4. Bring the stack up.** The bridge is registered in ``deployed_services``, so
it starts with everything else:

.. code-block:: bash

   osprey up        # add --dev to bake in a local osprey checkout

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

Who may reach the agent, and what the trigger lets it do, is the same for every
bridge — see :ref:`bridge-access`. What is particular to Nextcloud Talk:

**The rooms are exactly the ones you list.** Adding a room token to
``NEXTCLOUD_ROOMS`` grants access to everyone Nextcloud says is in that room, so
choose rooms as deliberately as you would choose who gets an account.

**Mentions are matched against the mention Talk itself records**, not against the
message text, so writing the bot's name in passing does not trigger it, and
neither does ``@all``. The bridge also ignores its own messages, which is what
stops it answering itself in a loop.

**Files are shared into the room, never published.** When an answer includes a
plot or a file, the bridge uploads it to the bot's own storage and shares it with
the room, so only that room's members can open it. **No world-readable link is
ever created** — there is no public URL to leak, forward, or index.
