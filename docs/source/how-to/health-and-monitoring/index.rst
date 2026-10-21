=====================
Health and Monitoring
=====================

Two questions, two tools. *Is it up?* is answered by the health suite: a run of
diagnostics over configuration, the Python environment, container
infrastructure, the framework services you enabled, and any probes a facility
declares, read in the browser's SYSTEM panel or parsed from ``osprey health
--json`` by a CI job. *What is it doing?* is answered by telemetry: the logs
and metrics the agent emits over OpenTelemetry as it works, sent to a backend
you already run or to a local store the presets deploy alongside the project.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Configure Health Checks
      :link: configure-health-checks
      :link-type: doc
      :shadow: md

      What the built-in diagnostics cover, and how a ``health:`` block in
      ``config.yml`` -- set through the profile's ``config:`` block -- adds a
      facility's own probes and tunes the suite's timing.

   .. grid-item-card:: Monitor Your OSPREY Agent
      :link: monitor-agent
      :link-type: doc
      :shadow: md

      Emitting the agent's telemetry over OTLP to any compatible backend, and
      the local telemetry store and gateway the presets deploy by default for
      storing and querying it.

   .. grid-item-card:: Keep the Agent Record
      :link: agent-record
      :link-type: doc
      :shadow: md

      Every store of what the agent was asked and did, where it lives, what
      deletes it, and the daily append-only archive that keeps a copy.

   .. grid-item-card:: Read the Record
      :link: read-the-record
      :link-type: doc
      :shadow: md

      Name the people who may query the telemetry store with SQL, hand over
      their keys, and know the limits every query runs under.

   .. grid-item-card:: Reconstruct a Session
      :link: reconstruct-session
      :link-type: doc
      :shadow: md

      One SQL query, run as a reader, that puts a session's events, tool
      calls, approvals and preflight checks back in the order they happened.

.. toctree::
   :hidden:

   configure-health-checks
   monitor-agent
   agent-record
   read-the-record
   reconstruct-session
