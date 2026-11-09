.. _how-to-slack:

=====
Slack
=====

How to let your team ask the OSPREY agent questions from Slack, and get
answers, plots, and files back in the same thread.

.. dropdown:: Before you start
   :color: info
   :icon: checklist

   - A project whose build profile has a ``dispatch:`` block — see
     :doc:`../event-dispatch`.
   - Docker or Podman, for the container path.
   - A Slack workspace in which you may create and install an app (some
     workspaces need an administrator to approve it).

Overview
========

Someone mentions the app in a channel, or messages it directly; the bridge
hands the question to the :doc:`event dispatch pipeline <../event-dispatch>`
and posts the answer back in the thread.

The bridge uses Slack's **Socket Mode**: it opens a connection *to* Slack and
Slack pushes events down it. Every connection is outbound, so the bridge opens
no port and needs no public address.

Like every bridge, it :ref:`remembers each question and conversation
<bridge-memory>`. It also writes each event to disk before telling Slack it has
it, so a question already received survives a crash or restart.

Create the Slack App
====================

**1. Create the app from a manifest.** At `api.slack.com/apps
<https://api.slack.com/apps>`_, choose *Create New App* → *From a manifest*,
pick your workspace, and paste (newer Slack UIs call the option *Paste a
manifest*):

.. code-block:: yaml

   display_information:
     name: OSPREY agent
   features:
     app_home:
       home_tab_enabled: false
       messages_tab_enabled: true
       messages_tab_read_only_enabled: false   # lets people message the app
     bot_user:
       display_name: osprey
       always_online: true
   oauth_config:
     scopes:
       bot:
         - app_mentions:read
         - chat:write
         - channels:history
         - groups:history
         - im:history
         - mpim:history
         - channels:read
         - groups:read
         - im:read
         - mpim:read
         - users:read
         - files:read
         - files:write
   settings:
     event_subscriptions:
       bot_events:
         - app_mention
         - message.im
         - message.mpim
     interactivity:
       is_enabled: false
     org_deploy_enabled: false
     socket_mode_enabled: true
     token_rotation_enabled: false

The display name is what your team sees replying, so make it obvious. Leave
token rotation off: with it on, the tokens expire every 12 hours.

What each scope is for — an administrator approving the app will ask:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Scope
     - Why the bridge needs it
   * - ``app_mentions:read``
     - Receive messages that mention the app in a channel.
   * - ``chat:write``
     - Post the acknowledgement, the answer and status notices.
   * - ``channels:history``, ``groups:history``
     - Read the message a thread started from, so a reply in a thread arrives
       with its context (public and private channels).
   * - ``im:history``, ``mpim:history``
     - Receive direct and group direct messages, and read their threads.
   * - ``channels:read``, ``groups:read``, ``im:read``, ``mpim:read``
     - List who is in a conversation, so the agent knows who it is talking to.
   * - ``users:read``
     - Look up those people's names.
   * - ``files:read``
     - Download files someone attaches to a question.
   * - ``files:write``
     - Upload the plots and files a run produces into the thread.

**2. Create the app-level token.** Under *Basic Information* → *App-Level
Tokens*, generate a token with the scope ``connections:write``. It starts with
``xapp-`` and is what the bridge opens its Socket Mode connection with. This
token cannot be declared in the manifest.

**3. Install the app.** Under *Install App*, install it to the workspace (or
request approval). Copy the *Bot User OAuth Token*, which starts with
``xoxb-``.

**4. Invite the app to channels.** It only sees channels it is a member of:
``/invite @OSPREY agent`` in each one. Direct messages need no setup.

.. warning::

   **One bridge per app, one app per deployment.** Slack spreads an app's
   events across *every* open Socket Mode connection. A second bridge running
   with the same app token — a staging stack, a forgotten container, another
   facility reusing the app — does not get its own copy of each message. It
   **silently splits** them: each bridge answers only what it received, and
   nothing logs an error. Create a separate app for every deployment.

Enable It in a Profile
======================

Add a ``slack_bridge:`` block next to your ``dispatch:`` block. The trigger
decides what the agent may do with a Slack question; ``mentions`` decides
whether it may @mention people:

.. code-block:: yaml

   slack_bridge:
     trigger: slack-question        # default; must exist in your triggers file
     mentions: true                 # default; false posts @mentions as plain text

   dispatch:
     triggers: my_triggers.yml      # the file that trigger must be declared in
     worker_count: 1

   env:
     required:
       - SLACK_BOT_TOKEN
       - SLACK_APP_TOKEN

.. note::

   The bridge and the dispatch pair must sit on the same network. When the
   workers go on the host's network (:ref:`deployment-network-attachment`), add
   ``services.slack_bridge.network: host`` to your profile's ``config:`` block.
   ``osprey build`` refuses a split pair and names the key to change.

The two tokens are secrets, not profile settings. Listing them under
``env.required`` documents them in ``.env.example``; put the values in the
**profile's** ``.env``, from which the build derives the project's ``.env``
(mode ``0600``).

Declaring the bridge without a ``dispatch:`` block, or naming a trigger your
triggers file does not declare, fails the **build** with a message naming the
problem. A minimal trigger; list the tools the agent may use, or it can only
answer from what it knows:

.. code-block:: yaml

   triggers:
     - name: slack-question
       source: webhook
       action:
         prompt: >-
           A colleague asked you this question in Slack. Answer it in the
           thread; plots and files you save are posted there too.
         allowed_tools:
           - mcp__python__execute
           - mcp__osprey_workspace__create_static_plot

Runtime settings
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Meaning
   * - ``SLACK_BOT_TOKEN``
     - The bot token (``xoxb-…``) from *Install App*. Used to read and post.
   * - ``SLACK_APP_TOKEN``
     - The app-level token (``xapp-…``) with ``connections:write``. Used only
       to open the Socket Mode connection.

Neither is generated for you. A missing token stops the bridge at startup with
the variable named; a refused one stops it with Slack's reason.

.. dropdown:: Settings that are filled in for you, or safe to leave alone
   :icon: gear

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Variable
        - Meaning
      * - ``DISPATCH_TRIGGER``
        - The trigger to fire. Comes from the profile block.
      * - ``SLACK_MENTIONS``
        - Whether @mentions are on. Comes from the profile block.
      * - ``EVENT_DISPATCHER_TOKEN``, ``DISPATCH_WORKER_TOKEN``
        - The shared secrets for the dispatcher and the worker, generated for
          you when unset — see :ref:`Authentication <event-dispatch-auth>`.
      * - ``DISPATCHER_URL``, ``WORKER_URL``
        - Where the dispatcher and worker are. Filled in when they run in the
          same stack; required when they run elsewhere.
      * - ``DISPATCH_TIMEOUT_SEC``
        - How long the worker may spend on one run, from your project
          configuration.
      * - ``POLL_BUDGET``
        - How long the bridge waits for an answer. Defaults to 30 seconds more
          than the worker's limit, and may never be less.
      * - ``POLL_INTERVAL``, ``DRAIN_INTERVAL``
        - Seconds between checks on an answer (default 2) and between retries of
          questions not yet handed off (default 60).
      * - ``RETRY_MIN_AGE``, ``RETRY_GIVE_UP``, ``RETRY_LIFETIME_CAP``
        - When a failed hand-off is retried (20 minutes), abandoned (48 hours),
          and dropped whatever its state (7 days).
      * - ``BRIDGE_TRUST_ENV``
        - Set to ``1`` only if outbound calls must go through your site's web
          proxy — see :ref:`slack-proxy`.
      * - ``HISTORY_ANSWER_LIMIT``
        - Longest earlier answer, in characters, sent back in full with a
          follow-up (default 3000; ``0`` sends every answer in full).
      * - ``GITLAB_URL``, ``GITLAB_PROJECT``, ``GITLAB_ISSUES_TOKEN``
        - Where to file an issue when a question is given up on. Leave unset to
          file nothing.
      * - ``APP_VERSION_DISPLAY``
        - Release label shown with each acknowledgement. Defaults to the
          installed OSPREY version.
      * - ``DEDUP_PATH``
        - Where the bridge keeps the questions it has answered, under ``/data``
          by default.
      * - ``TZ``
        - Timezone, from your project configuration.

.. _slack-proxy:

Behind a Web Proxy
------------------

If the host reaches the internet only through a proxy, set ``BRIDGE_TRUST_ENV=1``
and the usual ``HTTPS_PROXY`` variables. The proxy must allow ``CONNECT`` to:

- ``slack.com`` — the Web API;
- ``wss.slack.com`` (and its ``wss-*.slack.com`` variants) — the Socket Mode
  connection;
- ``files.slack.com`` and ``files-edge.slack.com`` — file downloads and
  uploads.

Calls to the dispatcher and worker never use the proxy.

Bring It Up
===========

**1. Fill in the environment file.** In the **profile's** ``.env``:

.. code-block:: bash

   SLACK_BOT_TOKEN=xoxb-...
   SLACK_APP_TOKEN=xapp-...

**2. Build and start.** The bridge is registered in ``deployed_services``, so
it starts with everything else:

.. code-block:: bash

   osprey up        # add --dev to bake in a local osprey checkout

Then mention the app in a channel it was invited to, or message it directly.
It acknowledges with "On it — working on this now." and replies in the thread.
If nothing happens, check the service's logs.

.. important::

   The bridge keeps what it remembers — which questions it has answered, the
   events it has received but not yet handled, and the recent conversation — in
   a named volume mounted at ``/data``. Do not remove that volume.

Who Is in the Channel, and @Mentions
====================================

Each question reaches the agent with who asked it and who is in the
conversation: the people Slack lists as members (up to 200), named by their
display name, else their real name. Bots and deactivated accounts are left out.
A one-to-one direct message has no member list.

The agent may @mention a member of the same conversation only when someone asks
it to pass something on or notify someone. Set ``mentions: false`` in the
profile block to post every mention as plain text.

.. _what-is-shared-slack:

Who Can Ask, and What Is Shared
===============================

Who may reach the agent, and what the trigger lets it do, is the same for every
bridge — see :ref:`bridge-access`. What is particular to Slack:

**Direct messages** from anyone in the workspace are answered without a
mention, at the top level unless asked in a thread.

**In channels and group direct messages, every message must mention the app** —
including replies in a thread the app is already answering. A reply without
the mention is not seen. Answers go in the thread of the question.

**Shared channels are ignored.** In a Slack Connect channel shared with another
organisation, the bridge answers nobody — not even people from your own
workspace. Messages from bots, edits and join notices are ignored everywhere.

**Plots and files stay inside Slack.** They are uploaded into the thread and
visible to that conversation's members; nothing is published as a public link.
Files someone attaches are passed to the agent up to 10 MiB each; anything
larger, or not hosted by Slack, is skipped with a note naming it.

When the Bridge Is Down
=======================

Slack redelivers an event the bridge did not acknowledge only for a few
minutes. A question posted while the bridge is down for longer than that is
**never answered**, and nothing in Slack shows it was missed. Restart or
upgrade the bridge at quiet times, and ask people to repeat a question that got
no acknowledgement.
