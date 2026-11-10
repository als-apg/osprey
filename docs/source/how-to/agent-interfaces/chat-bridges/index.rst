.. _how-to-chat-bridges:

============
Chat Bridges
============

A chat bridge lets your team ask the OSPREY agent questions from a chat room
they already sit in. Someone mentions the agent in the room, and the answer
comes back in the same conversation — plots and other files included. A bridge
is a good way to give a whole team read access to the machine, and a poor way
to hand out control of it.

Four chat systems are supported today: **Nextcloud Talk**, **Google Chat**,
**Microsoft Teams** and **Slack**. They answer the same way and are switched on the same way in
a profile. You can add others.

What a Bridge Is Not
====================

A bridge is not a second :doc:`Web Terminal <../../web-terminal/index>`. There is
no session to keep open, and nobody is asked to approve anything while an answer
is being worked out.

A chat question is one headless run of the agent. What the agent may do during
that run is decided in advance by the dispatcher trigger the bridge fires, not
by the question and not by who asked it.

How It Works
============

A bridge sits between a chat room and the :doc:`event dispatch pipeline
<../event-dispatch>`. It never runs the agent itself. It hands the question to
the dispatcher, waits, and posts back what comes out.

.. raw:: html
   :file: ../../../_diagrams/chat-bridge-pipeline.html

Only the top box changes between chat systems. Each system has its own small
**adapter**, which knows how messages arrive there and how to post a reply
there. Everything below it — remembering questions, keeping the conversation,
handling failures — is shared code that every bridge uses unchanged.

Notice that every arrow leaving a bridge points outward. A bridge is never
called: it fetches its messages from the chat system, from a queue the chat
system feeds, or down a connection it opened itself. So it opens no network
port, needs no public address, and nothing has to be able to reach it. For Teams the one public piece is the
relay, and that lives in Azure, not in your stack.

.. _bridge-memory:

What the Bridge Remembers
-------------------------

The bridge is the only part of the system that sees a whole conversation, so it
is the part that remembers. It keeps three things on its own disk volume: which
questions it has already answered, the recent exchanges in each conversation,
and (for Nextcloud Talk) how far it has read in each room.

That is what makes follow-up questions work — "now plot that over 24 hours"
arrives at the agent with the previous exchange attached. It is also what makes
a restart safe: a question interrupted halfway through is picked up again rather
than answered twice or lost.

In a shared room it also remembers who asked each question. Every question
reaches the agent with its asker's name, and every remembered exchange keeps
the name of the person who asked it, so a conversation among several people
reads as one.

A long earlier answer — a big table, say — goes back to the agent with a
follow-up as its opening and a note. The agent reads the rest only when the
follow-up needs it, and the bridge's own copy keeps every answer in full.

.. _bridge-access:

Who Can Ask
===========

**Membership in the room is the access gate.** Anyone who can post in a room the
agent has been added to can ask it questions. Your chat system, not Osprey,
decides who those people are — so add the agent to rooms as deliberately as you
would decide who gets an account.

In a group room, only messages that mention the agent are answered; everything
else is ignored, so it can sit in a busy room quietly. In a one-to-one
conversation there is nobody else to address, so every message counts.

**What the agent may do is set by the trigger.** The trigger's list of permitted
tools is mounted read-only, so a question arriving through chat cannot widen what
the agent is allowed to do, however it is phrased.

Choosing Your Platform
======================

All four bridges answer questions equally well. The differences that matter when
you pick one:

.. list-table::
   :header-rows: 1
   :widths: 12 22 22 22 22

   * -
     - Nextcloud Talk
     - Google Chat
     - Microsoft Teams
     - Slack
   * - The agent speaks as
     - A normal Nextcloud user account you create
     - A Chat app backed by a service account
     - A single-tenant Azure Bot registration in your own tenant
     - A Slack app installed in your workspace
   * - Messages arrive by
     - The bridge asking Nextcloud for them
     - A Google Cloud message queue
     - An Azure Function relay that checks each message came from Microsoft and
       puts it on a Service Bus queue the bridge reads
     - A Socket Mode connection the bridge opens to Slack
   * - Plots and files
     - Shared with the room, visible to its members only
     - **Published as a public link** anyone can open
     - Plots inside the conversation; other files in one SharePoint library,
       shared with the conversation's members
     - Uploaded into the thread, visible to its members only
   * - You need
     - A Nextcloud instance with the Talk app
     - A Google Cloud project
     - An Azure subscription, and permission to install a Teams app
     - Permission to install an app in the workspace

The plots-and-files row is the one to read twice:

- **Nextcloud Talk** keeps plots and files in the room.
- **Google Chat** publishes them at a public link, because Google must fetch an
  image to display it. The Google Chat page shows how to turn files off and
  keep text answers.
- **Microsoft Teams** returns plots inside the conversation and keeps other
  files in your own tenant, readable by the conversation's members. It needs a
  small relay in Azure, because a Teams bot is reached only over the public
  internet. Nothing in your stack is exposed. The Microsoft Teams page walks
  through it.
- **Slack** uploads plots and files into the thread, visible to its members
  only.

Learn More
==========

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Nextcloud Talk
      :link: nextcloud-talk
      :link-type: doc
      :class-header: bg-info text-white
      :shadow: md

      Deploy a bridge into Nextcloud Talk rooms, where files stay private to the
      room.

   .. grid-item-card:: Google Chat
      :link: google-chat
      :link-type: doc
      :class-header: bg-primary text-white
      :shadow: md

      Deploy a bridge into Google Chat spaces, and decide how plots and files
      are shared.

   .. grid-item-card:: Microsoft Teams
      :link: microsoft-teams
      :link-type: doc
      :class-header: bg-secondary text-white
      :shadow: md

      Publish the relay into Azure, then deploy a bridge into Teams channels and
      chats, where plots stay in the conversation.

   .. grid-item-card:: Slack
      :link: slack
      :link-type: doc
      :class-header: bg-warning text-white
      :shadow: md

      Create a Slack app and deploy a bridge into channels and direct messages,
      where files stay in the thread.

   .. grid-item-card:: Add Your Own
      :link: /contributing/extending-osprey
      :link-type: doc
      :class-header: bg-success text-white
      :shadow: md

      What it takes to connect Mattermost, email, or any other service the agent
      should answer from -- the extension point for developers, in the
      Contributing guide.

.. seealso::

   :doc:`../event-dispatch`
       The dispatcher and worker every bridge hands its questions to, and how to
       write the trigger it fires.

.. toctree::
   :maxdepth: 2
   :hidden:

   nextcloud-talk
   google-chat
   microsoft-teams
   slack
