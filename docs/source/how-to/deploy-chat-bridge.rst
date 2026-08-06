====================
Deploy a Chat Bridge
====================

How to let your team ask the OSPREY agent questions from a chat room — in
Nextcloud Talk or in Google Chat — and get answers, including plots and files,
back in the same room.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What a chat bridge does, and what it deliberately does not do
   - How to give the agent an identity to speak as in your chat system
   - How to enable a bridge in a build profile and bring it up
   - Which questions a bridge answers, who may ask, and what gets shared

   **Prerequisites:** A project whose profile has a ``dispatch:`` block (see
   :doc:`event-dispatch`), and Docker or Podman for the container path. For
   Nextcloud Talk you also need an instance with the Talk app and permission to
   create a user account on it. For Google Chat you need a Google Cloud project
   in which you can create a chat app, a service account, and a message queue.

Two chat systems are supported, each with its own section below. They work the
same way and are configured the same way; what differs is the identity the agent
speaks as, the credentials you supply, and — this one matters — how files come
back. In Nextcloud Talk a file is shared with the room and nobody else. In
Google Chat it is published as a public link. Read the last part of whichever
section applies to you before you deploy.

Nextcloud Talk
==============

Overview
--------

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
----------------------

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
~~~~~~~~~~~~~~~~

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
-----------

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
-------------------------------

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

Google Chat
===========

Overview
--------

The bridge turns a Google Chat conversation into a way of talking to the agent.
Someone mentions the chat app in a space, Google puts that message on a queue,
the bridge picks it up and hands the question to the event dispatch pipeline,
and the answer is posted back in the same conversation thread.

It is a **reader of a queue, not a server**. Google never calls it; it asks
Google for waiting messages. So it opens no network port of its own, needs no
public address, and nothing has to be able to reach it. It only makes outbound
calls — to Google and to the dispatcher.

.. mermaid::

   flowchart LR
       C[Google Chat space] -->|message mentioning the app| Q[Message queue]
       Q -->|picked up| B[gchat-bridge]
       B -->|question| D[Event dispatcher]
       D --> W[Dispatch worker]
       W -->|answer + any files| B
       B -->|reply in the thread| C

The bridge is the piece that remembers things. Each question is recorded before
it is dispatched, so a restart in the middle of one does not answer it twice or
drop it, and the recent exchanges in a conversation travel with each new
question, so "now plot that over 24 hours" makes sense. There is no reading
position to keep here: a message the bridge has not finished with stays on the
queue, so anything sent while the bridge was down is waiting for it when it
comes back.

Enable It in a Profile
----------------------

Add a ``gchat_bridge:`` block to your build profile. As with Talk, the only
setting is which dispatcher trigger the bridge fires — that trigger decides what
the agent is allowed to do with a chat question — and the block is only
meaningful next to a ``dispatch:`` block:

.. code-block:: yaml

   gchat_bridge:
     trigger: gchat-question        # default; must exist in your triggers file

   dispatch:
     triggers: my_triggers.yml      # the file that trigger must be declared in
     worker_count: 1

   env:
     required:
       - GCHAT_SA_KEY
       - GCHAT_SUBSCRIPTION
       - GCHAT_APP_ID

The queue, the service-account key and the bucket for files are **not** profile
settings. Which queue this deployment reads, which key file it signs in with,
and which storage bucket files are published through all differ per deployment —
and the key is a secret that must never be baked into a build. Listing them
under ``env.required`` makes ``osprey build`` write them into the project's
``.env`` (created mode ``0600``, readable only by you) for you to fill in.

The same two mistakes are caught at **build** time rather than at runtime:
declaring the bridge without a ``dispatch:`` block, and naming a trigger your
triggers file does not declare. Both fail the build with a message naming the
problem.

Runtime settings
~~~~~~~~~~~~~~~~

Everything the bridge reads at runtime is listed below. The first three are
yours to create in Google Cloud; most of the rest are filled in for you or can
be left alone.

.. warning::

   **One bridge per queue.** Deploy exactly one bridge against a given
   subscription. Google hands each message to only *one* of a subscription's
   readers, so a second deployment pointed at the same subscription — a staging
   stack, a container someone forgot to remove, another facility reusing the
   name — does not get its own copy of every message. It **silently splits**
   them: each bridge answers only the messages it happened to receive, the other
   questions look to your team like they were ignored, and nothing anywhere logs
   an error. Give every deployment its own subscription.

.. list-table::
   :header-rows: 1
   :widths: 26 18 56

   * - Variable
     - Required?
     - Meaning
   * - ``GCHAT_SA_KEY``
     - Yes
     - Path to the service-account key file the bridge signs in with. The file
       must exist at that path on the deployment host; it is mounted read-only
       into the container at the same path, so one value names it on both sides.
   * - ``GCHAT_SUBSCRIPTION``
     - Yes
     - The full name of the queue subscription to read, in the form
       ``projects/your-project/subscriptions/your-subscription``. A short name
       on its own does not work; the bridge warns at startup if what you set
       does not look like the full form.
   * - ``GCHAT_APP_ID``
     - Yes
     - The chat app's own user id, e.g. ``users/1234567890``. This is also what
       an @mention is matched against, so a wrong value makes the bridge ignore
       every message instead of failing loudly — worth double-checking.
   * - ``DISPATCH_TRIGGER``
     - Filled in
     - The trigger to fire. Comes from the profile block above.
   * - ``EVENT_DISPATCHER_TOKEN``
     - Yes
     - Shared secret for talking to the dispatcher. ``osprey deploy up``
       generates it when unset.
   * - ``DISPATCH_WORKER_TOKEN``
     - Yes
     - Shared secret for talking to the worker. Also auto-generated.
   * - ``DISPATCHER_URL``
     - Only if separate
     - Where the dispatcher is. Filled in for you when it runs in the same
       stack; when it runs elsewhere you must set it, and the bridge refuses to
       start without it.
   * - ``WORKER_URL``
     - Only if separate
     - Where the worker is — the bridge collects answers and files from it
       directly. Same rule as above.
   * - ``DISPATCH_TIMEOUT_SEC``
     - Filled in
     - How long the worker may spend on one run. Comes from your project
       configuration, so raising it raises it for both halves at once.
   * - ``POLL_BUDGET``
     - Optional
     - How long the bridge waits for an answer before giving up on it. Defaults
       to 30 seconds more than the worker's own limit, and may never be less
       than that limit — the bridge refuses to start if it is.
   * - ``POLL_INTERVAL``
     - Optional
     - Seconds between checks on an answer in progress (default 2).
   * - ``DRAIN_INTERVAL``
     - Optional
     - Seconds between sweeps of the queue of questions that could not be handed
       off yet (default 60).
   * - ``RETRY_MIN_AGE``
     - Optional
     - How long a failed hand-off is held before it is retried, so a brief
       outage has time to clear (default 20 minutes).
   * - ``RETRY_GIVE_UP``
     - Optional
     - Age at which a question that still cannot be handed off is abandoned
       (default 48 hours).
   * - ``RETRY_LIFETIME_CAP``
     - Optional
     - Hard ceiling on how long anything may sit in that queue, whatever its
       state (default 7 days).
   * - ``BRIDGE_TRUST_ENV``
     - Optional
     - Set to ``1`` only if this host's outbound calls must go through your
       site's web proxy. Off by default, so a proxy inherited from a shell or a
       CI runner cannot quietly place itself in front of Google.
   * - ``GITLAB_URL``, ``GITLAB_PROJECT``, ``GITLAB_ISSUES_TOKEN``
     - Optional
     - Where to file an issue when a question is finally given up on. Leave
       unset if you have no such host: nothing is filed and nothing is checked.
   * - ``GCS_BUCKET``
     - Optional
     - The Cloud Storage bucket that plots and files are published to. Without
       it the agent still answers, text only. **Read the sharing note at the end
       of this section before setting one.**
   * - ``GCS_PROJECT``
     - Optional
     - The Google Cloud project that owns that bucket.
   * - ``APP_VERSION_DISPLAY``
     - Optional
     - A release label shown with each acknowledgement, so a conversation shows
       which version answered. Omitted when unset.
   * - ``DEDUP_PATH``, ``HISTORY_PATH``
     - Optional
     - Where the bridge keeps what it remembers. Both default to files under
       ``/data``, its own volume; change them only if you deliberately relocate
       that state.
   * - ``TZ``
     - Filled in
     - Timezone, taken from your project configuration so timestamps match the
       rest of the stack.

Bring It Up
-----------

**1. Create the chat app and the identity it speaks as.** In your Google Cloud
project, create a Chat app for the agent — its display name is what your team
sees replying, so make it obvious (for example ``OSPREY agent``). Create a
service account for it, download a key file for that account, and note the app's
own user id (the ``users/…`` value).

**2. Give this deployment its own queue.** Configure the chat app to publish its
events to a topic, and create one subscription on that topic for *this*
deployment — not one shared with any other. Allow the service account to read
that subscription. See the warning above for what a shared subscription does.

**3. Add the app to the spaces it should serve.** It only sees conversations it
has been added to. Add it to each space you want served; people can also message
it directly without any setup.

**4. Optional: create a bucket for plots and files.** Allow the service account
to write to it. Anything the agent publishes there has to be readable without
signing in — see the sharing note below before you decide to do this at all.
Skip this step and the agent answers text only.

**5. Fill in ``.env``** with the values from the table above:

.. code-block:: bash

   GCHAT_SA_KEY=/etc/osprey/gchat-service-account.json
   GCHAT_SUBSCRIPTION=projects/my-gcp-project/subscriptions/osprey-chat-events
   GCHAT_APP_ID=users/1234567890
   GCS_BUCKET=my-osprey-chat-artifacts   # optional; enables plots and files
   GCS_PROJECT=my-gcp-project            # optional; the bucket's project

**6. Bring the stack up.** The bridge is registered in ``deployed_services``, so
it starts with everything else:

.. code-block:: bash

   osprey deploy up        # add --dev to bake in a local osprey checkout

Then mention the app in one of its spaces and ask it something. If nothing
happens, check the service's logs first: a missing credential stops the bridge at
startup with the missing variable named. A wrong app id is the quieter failure —
the bridge runs happily and ignores every message, because nothing it sees looks
like a mention of itself.

.. important::

   The bridge keeps what it remembers — which questions it has answered and the
   recent conversation — in a named volume mounted at ``/data``. Do not remove
   that volume. Without it, a restart in the middle of a question can answer it
   twice or drop it, and conversations lose their thread of context.

Who Can Ask, and What Is Shared
-------------------------------

**Space membership is the access gate.** Anyone who can post in a space the app
belongs to can reach the agent, and so can anyone who can message it directly.
Google — not OSPREY — decides who those people are. Add the app to spaces as
deliberately as you would choose who gets an account.

**In a space, only messages that @mention the app are answered.** Everything
else is ignored, so the app can sit in a busy space without reacting to ordinary
conversation. The check is made against the mention Google itself records, not
against the message text, so writing the app's name in passing does not trigger
it. In a direct message there is nobody else to address, so every message counts
as a question.

**What the agent may do with a chat question is set by the trigger, not by the
question.** The trigger's tool allowlist lives in your triggers file, which is
mounted **read-only** into the dispatcher — so a request arriving through chat
cannot widen what the agent is permitted to do, and the worker's own denylist
applies on top of it (see :doc:`event-dispatch`).

.. warning::

   **Plots and files come back as public links.** This is where a Google Chat
   deployment differs sharply from Nextcloud Talk. Chat can only show an image
   in a message if Google itself can fetch it, so every plot or file the agent
   produces is uploaded to the bucket you configured and posted as an ordinary
   web address that **anyone who has the link can open** — no sign-in, no
   membership check, no expiry. Forwarding the message forwards that access with
   it, and a link that leaks stays usable until you delete the object.

   Treat that bucket as published material. Give it a bucket of its own, holding
   nothing else, in a project with nothing sensitive in it, and consider a rule
   that deletes objects after a while so you are not accumulating a permanent
   public archive of your plots. If publishing this way is not acceptable at
   your facility, leave ``GCS_BUCKET`` unset: the agent then answers text only
   and publishes nothing.

.. seealso::

   :doc:`event-dispatch`
       The dispatcher and worker the bridge hands questions to, and how to
       author the trigger it fires.

   :doc:`deploy-project`
       Container deployment mechanics for all OSPREY services, including image
       overrides.

   :doc:`../cli-reference/index`
       Full ``osprey build`` and ``osprey deploy`` reference.
