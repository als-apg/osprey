.. _how-to-read-the-record:

===============
Read the Record
===============

Use this page to let a person read the agent record beyond their own sessions.
You name them as a reader, hand over their key, and later rotate or withdraw
it. The page also says what a reader can and cannot see, and what the record
can vouch for.

The **agent record** holds what the agent was asked, what it answered and what
it did. The ``agent-record`` service keeps it as entries of named kinds. A
**reader** is a name in the config with a key of its own.

For how long the record keeps entries and the daily archive that keeps the
rest, see :doc:`agent-record`.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - How to name a reader, hand over its key, rotate it and withdraw it
   - What a reader sees, and where it reads
   - What the record can and cannot vouch for

Name the readers
================

List readers in the profile's top-level ``agent_record:`` block. Each entry is
one reader:

.. code-block:: yaml

   agent_record:
     readers:
       - name: carol                 # a roster user: same name, same person
       - name: dana
         oidc_subject: 7f3c9e1a-...  # recognised by sign-on in the browser
         kinds: [audit, run]         # default: every kind
         days_back: 3                # default: retention_days

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Meaning
   * - ``name``
     - The reader's name: lowercase letters, digits, ``_`` and ``-``, starting
       with a letter or digit. Each reader is named once.
   * - ``oidc_subject``
     - The sign-on subject that is this reader in a browser, when the web
       terminals sign people on with ``oidc``. A reader with the same name as a
       roster user is that user already, and may not carry a subject.
   * - ``kinds``
     - The kinds of entry the reader may read, from the record's kinds
       (``prompt``, ``answer``, ``tool_call``, ``audit``, ``run`` and the rest).
   * - ``days_back``
     - How far back the reader sees, at most ``agent_record.retention_days``.
       Entries older than that are not visible to this reader at all; it is a
       scope, not a limit.

The control-assistant preset declares one reader, the deployment's admin
roster user (``carol``).

Hand over the key
-----------------

To give a new reader their key:

#. Run ``osprey up``. It mints each new reader's key into
   ``.env.agent-record.reader.<name>`` at the repo root, with mode 0600. The
   deploy prints the file's path, never the key.
#. Read the file. It holds one line,
   ``OSPREY_AGENT_RECORD_LOGIN=reader.<name>:<key>``.
#. Give the key to the person out of band.

To change a reader later:

- **Rotate** a key: delete the file and run ``osprey up``. The old key stops
  working and ``osprey up`` writes a new file.
- **Withdraw** a reader: delete the entry and run ``osprey up``. The key stops
  working. The key file is set aside as
  ``.env.agent-record.reader.<name>.withdrawn-<date>``, so re-adding the name
  later mints a fresh key.
- **Narrow** a reader: change ``kinds`` or ``days_back`` and run ``osprey up``.

What a reader sees
==================

A reader is a reader everywhere: the same ``kinds`` and ``days_back`` bound it
on the history page and with its key. Within them a reader sees every
identity's sessions. A reader writes nothing: its key reads, and no entry
is ever filed under it.

In a browser, a reader reads on the history page, served at ``/history/`` on
the **history page's origin**. That origin is the web tier's second server on
its own port (``modules.web_terminals.history_port``). Behind a load balancer,
it is the browser-facing origin ``modules.web_terminals.history_origin`` names.
Pages on the terminals' origin, where agent-written HTML runs, cannot read from
it.

A roster user named as a reader signs in as themselves. A reader off the roster
is recognised by its ``oidc_subject``. For the page, and who else sees what on
it, see :ref:`multi-user-history-page`.

What the record can vouch for
=============================

The record is as trustworthy as the processes that write to it. Read it with
these in mind:

- **Containers are the boundary.** Each sender's container holds the login of
  the one identity it records as, and only its own spool and transcripts. On a
  deployment whose agent runs on the host, the agent runs as the operator's
  own user, and the credential files are within its reach.
- **Within one container, any process can write that user's record.** The
  login belongs to the container, not to one process in it.
- **Prompts and events are reported by the sender.** The record files an
  entry under the identity it arrived from: its spool directory, its login or
  its transcript root. It never uses a name the sender wrote. What the entry
  says happened is the sender's account of it.
- **Agent-authored session plans run inside the queueserver container.** So
  the record trusts its preflight entries narrowly. On a person's own history
  page, a preflight entry appears only when it matches that person's own
  queue-submit call within 60 seconds. Readers and ``history: all`` viewers see
  every preflight entry as stored.
- **The logins are told apart by their tokens**, not by the address they
  connect from.
- **Each fact is stored once.** An entry's source fixes its key, and the
  record never stores a key twice. So a restart, a resumed session or a
  rewritten transcript adds no second copy. While the record's store is
  unreachable, the senders keep their spools, and ``osprey health`` shows each
  source's backlog.
- **The record keeps** ``retention_days`` **of history.** The daily archive
  keeps the rest (:doc:`agent-record`).
