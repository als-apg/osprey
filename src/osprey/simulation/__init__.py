"""Scenario application, scenario plots and the simulation engine plug-ins.

:mod:`osprey.simulation.apply` composes a set of the render's scenarios and
makes their telemetry and logbook live, :mod:`osprey.simulation.plots` draws a
scenario plot spec, :mod:`osprey.simulation.sessions` parses and places a
scenario's past agent sessions, :mod:`osprey.simulation.status` reads every served model's
status on one control target, and ``osprey.simulation.engines`` holds the
engine plug-ins a served model is built from. The package root exports one
name, :data:`DEFAULT_SCENARIOS_KEY`: the config key's one declaration, which the
runtime and the build both import without loading the runtime's data stack.
"""

__all__ = ["DEFAULT_SCENARIOS_KEY"]

#: The config key naming the scenarios a deployment that never chose a set starts in.
DEFAULT_SCENARIOS_KEY = "simulation.default_scenarios"
