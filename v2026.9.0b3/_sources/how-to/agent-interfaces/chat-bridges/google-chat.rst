.. _how-to-google-chat:

===========
Google Chat
===========

How to let your team ask the Osprey agent questions from a Google Chat space,
and get answers, plots, and files back in the same conversation.

.. dropdown:: Before you start
   :color: info
   :icon: checklist

   - A project whose build profile has a ``dispatch:`` block — see
     :doc:`../event-dispatch`.
   - Docker or Podman, for the container path.
   - A Google Cloud project in which you can create a chat app, a service
     account, and a message queue.

Overview
========

The bridge turns a Google Chat conversation into a way of talking to the agent.
Someone mentions the chat app in a space, Google puts that message on a queue,
the bridge picks it up and hands the question to the :doc:`event dispatch
pipeline <../event-dispatch>`, and the answer is posted back in the same
conversation thread.

It is a **reader of a queue, not a server**. Google never calls it; it asks
Google for waiting messages, making only outbound calls.

Like every bridge, it :ref:`remembers each question and conversation
<bridge-memory>`. What is particular to Google Chat is that there is no reading
position to keep: a message the bridge has not finished with stays on the queue,
so anything sent while the bridge was down is waiting when it comes back.

Enable It in a Profile
======================

Add a ``gchat_bridge:`` block to your build profile. The only setting is which
dispatcher trigger the bridge fires — that trigger decides what the agent is
allowed to do with a chat question — and the block is only meaningful next to a
``dispatch:`` block:

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
under ``env.required`` documents them in ``.env.example``; fill the values into
the **profile's** ``.env``, which is where a secret survives a rebuild. The
build derives the project's ``.env`` from it (created mode ``0600``, readable
only by you).

Two mistakes are caught at **build** time rather than at runtime: declaring the
bridge without a ``dispatch:`` block, and naming a trigger your triggers file
does not declare. Both fail the build with a message naming the problem.

Runtime settings
----------------

These are the ones you create and set yourself.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Meaning
   * - ``GCHAT_SA_KEY``
     - Path to the service-account key file the bridge signs in with. The file
       must exist at that path on the deployment host; it is mounted read-only
       into the container at the same path, so one value names it on both sides.
   * - ``GCHAT_SUBSCRIPTION``
     - The full name of the queue subscription to read, in the form
       ``projects/your-project/subscriptions/your-subscription``. A short name
       on its own does not work; the bridge warns at startup if what you set
       does not look like the full form.
   * - ``GCHAT_APP_ID``
     - The chat app's own user id, e.g. ``users/1234567890``. This is also what
       an @mention is matched against, so a wrong value makes the bridge ignore
       every message instead of failing loudly — worth double-checking.
   * - ``GCS_BUCKET``, ``GCS_PROJECT``
     - Optional. The Cloud Storage bucket that plots and files are published to,
       and the project that owns it. Without a bucket the agent still answers,
       text only. **Read** :ref:`what-is-shared-gchat` **before setting one.**

.. warning::

   **One bridge per queue.** Deploy exactly one bridge against a given
   subscription. Google hands each message to only *one* of a subscription's
   readers, so a second deployment pointed at the same subscription — a staging
   stack, a container someone forgot to remove, another facility reusing the
   name — does not get its own copy of every message. It **silently splits**
   them: each bridge answers only the messages it happened to receive, the other
   questions look to your team like they were ignored, and nothing anywhere logs
   an error. Give every deployment its own subscription.

.. dropdown:: Settings that are filled in for you, or safe to leave alone
   :icon: gear

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Variable
        - Meaning
      * - ``DISPATCH_TRIGGER``
        - The trigger to fire. Comes from the profile block.
      * - ``EVENT_DISPATCHER_TOKEN``, ``DISPATCH_WORKER_TOKEN``
        - The two shared secrets the bridge needs to reach the dispatcher and
          the worker, generated for you when unset — see
          :ref:`Authentication <event-dispatch-auth>`.
      * - ``DISPATCHER_URL``, ``WORKER_URL``
        - Where the dispatcher and worker are — the bridge collects answers and
          files from the worker directly. Filled in for you when they run in the
          same stack; when they run elsewhere you must set them, and the bridge
          refuses to start without them.
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
          a CI runner cannot quietly place itself in front of Google.
      * - ``GITLAB_URL``, ``GITLAB_PROJECT``, ``GITLAB_ISSUES_TOKEN``
        - Where to file an issue when a question is finally given up on. Leave
          unset if you have no such host: nothing is filed and nothing is
          checked.
      * - ``APP_VERSION_DISPLAY``
        - A release label shown with each acknowledgement, so a conversation
          shows which version answered. Omitted when unset.
      * - ``DEDUP_PATH``, ``HISTORY_PATH``
        - Where the bridge keeps what it remembers. Both default to files under
          ``/data``, its own volume; change them only if you deliberately
          relocate that state.
      * - ``TZ``
        - Timezone, taken from your project configuration so timestamps match
          the rest of the stack.

Bring It Up
===========

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
signing in — see :ref:`what-is-shared-gchat` before you decide to do this at all.
Skip this step and the agent answers text only.

**5. Fill in the environment file.** Set the values from the table above in the
**profile's** ``.env`` (the build derives the project's ``.env`` from it):

.. code-block:: bash

   GCHAT_SA_KEY=/etc/osprey/gchat-service-account.json
   GCHAT_SUBSCRIPTION=projects/my-gcp-project/subscriptions/osprey-chat-events
   GCHAT_APP_ID=users/1234567890
   GCS_BUCKET=my-osprey-chat-artifacts   # optional; enables plots and files
   GCS_PROJECT=my-gcp-project            # optional; the bucket's project

**6. Bring the stack up.** The bridge is registered in ``deployed_services``, so
it starts with everything else:

.. code-block:: bash

   osprey up        # add --dev to bake in a local osprey checkout

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

.. _what-is-shared-gchat:

Who Can Ask, and What Is Shared
===============================

Who may reach the agent, and what the trigger lets it do, is the same for every
bridge — see :ref:`bridge-access`. What is particular to Google Chat:

**The spaces are the ones you added the app to**, plus anyone who messages it
directly. Google — not Osprey — decides who is in those spaces, so add the app
as deliberately as you would choose who gets an account.

**Mentions are matched against the mention Google itself records**, not against
the message text, so writing the app's name in passing does not trigger it.

.. warning::

   **Plots and files come back as public links.** This is where a Google Chat
   deployment differs sharply from :doc:`Nextcloud Talk <nextcloud-talk>`. Chat
   can only show an image in a message if Google itself can fetch it, so every
   plot or file the agent produces is uploaded to the bucket you configured and
   posted as an ordinary web address that **anyone who has the link can open** —
   no sign-in, no membership check, no expiry. Forwarding the message forwards
   that access with it, and a link that leaks stays usable until you delete the
   object.

   Treat that bucket as published material. Give it a bucket of its own, holding
   nothing else, in a project with nothing sensitive in it, and consider a rule
   that deletes objects after a while so you are not accumulating a permanent
   public archive of your plots. If publishing this way is not acceptable at
   your facility, leave ``GCS_BUCKET`` unset: the agent then answers text only
   and publishes nothing.
