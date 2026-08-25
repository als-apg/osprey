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
      the local OpenObserve store the presets deploy by default for storing
      and viewing it.

.. toctree::
   :hidden:

   configure-health-checks
   monitor-agent
