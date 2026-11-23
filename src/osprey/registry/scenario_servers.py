"""The check of a scenario's sessions against the framework's MCP server names.

A scenario's ``sessions`` block replays agent turns whose tool steps name the
MCP server that answered them. The facility description checks the rest of
the block (:func:`osprey.facility.scenarios.check_scenario_sessions`) and knows
no server names; the framework holds them in
:data:`osprey.registry.mcp.FRAMEWORK_SERVERS` and supplies this check to the
build through its ``extra_checks`` keyword, so its stops join the scenario
errors at the point the build runs them.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey.facility.errors import FacilityBuildError
    from osprey.facility.validate import Validated

__all__ = ["check_scenario_servers", "scenario_server_errors"]


def check_scenario_servers(validated: Validated) -> list[FacilityBuildError]:
    """Check every tool step of every scenario session names an OSPREY server.

    Args:
        validated: What the build's stages produced; its document is set.

    Returns:
        One ``reference-missing`` per unknown server, in scenario, session,
        part, turn and step order.
    """
    from osprey.facility.validate import need
    from osprey.registry.mcp import FRAMEWORK_SERVERS

    return scenario_server_errors(need(validated.document), FRAMEWORK_SERVERS)


def scenario_server_errors(
    document: Mapping[str, Any], servers: Set[str] | Mapping[str, Any]
) -> list[FacilityBuildError]:
    """The tool steps of a document's scenario sessions naming no server in *servers*.

    A ``sessions`` block that does not parse is left to the description's own
    session check, which stops on it; this check stays silent on it.

    Args:
        document: The combined document.
        servers: The known server names.

    Returns:
        One ``reference-missing`` per unknown server.
    """
    from osprey.facility.errors import FacilityBuildError
    from osprey.simulation.sessions import ToolStep, parse_sessions

    known = ", ".join(sorted(servers))
    errors: list[FacilityBuildError] = []
    for scenario in document.get("scenarios") or []:
        if not isinstance(scenario, dict) or scenario.get("sessions") is None:
            continue
        name = str(scenario.get("name"))
        files = [f"scenarios/{name}.yaml"]
        try:
            sessions = parse_sessions(scenario["sessions"], prefix=f"{files[0]} `sessions`")
        except ValueError:
            continue
        for s, session in enumerate(sessions):
            for p, part in enumerate(session.parts):
                for t, turn in enumerate(part.turns):
                    slot = f"sessions[{s}].parts[{p}].turns[{t}]"
                    for index, step in enumerate(turn.steps):
                        if not isinstance(step, ToolStep) or step.server in servers:
                            continue
                        errors.append(
                            FacilityBuildError(
                                "reference-missing",
                                name,
                                files,
                                f"name one of {known} in `{slot}.steps[{index}].tool.server`",
                                record_kind="scenario",
                                detail=f"{files[0]} `{slot}.steps[{index}].tool.server` names "
                                f"server {step.server}, not an OSPREY server",
                            )
                        )
    return errors
