"""The closing summary card for the OSPREY lifecycle verbs.

``init``, ``build``, ``up -d``, ``restart -d`` and ``down`` all end the same
way: one short card saying which deployment this was, what state it is in now,
where it answers, where the command output went, and which command comes next.
One renderer for all five, so the five cannot describe the same deployment in
five shapes.

It prints through the installed phase reporter, after the last phase has
closed, so it shares that module's plain-line path -- color only on a terminal,
and nothing at all under the global ``--verbose``, where the real output was
streamed instead of spooled. Attached (non-``-d``) starts print no card at all:
``compose up`` replaces this process and the terminal belongs to the log stream.
"""

from __future__ import annotations

from pathlib import Path

from osprey.deployment.subprocess_capture import SPOOL_DIR
from osprey.utils.logger import get_logger

from .phase_reporter import NullReporter, current_reporter
from .styles import Styles

logger = get_logger("cli.summary_card")

#: What to do next, per state. Only ``running`` is reachable-anywhere, so it is
#: the only state whose card carries endpoints.
_NEXT_STEPS = {
    "created": "osprey build · osprey up -d",
    "built": "osprey up -d · osprey status",
    "running": "osprey status · osprey logs · osprey down",
    "stopped": "osprey up -d · osprey status",
}


def owns_summary_card() -> bool:
    """True when the calling verb is the outermost one, so the card is its own.

    Asked BEFORE :func:`~osprey.cli.main.lifecycle_reporter`, and by the same
    rule that decides who owns the reporter: the verb that still finds the quiet
    default installed is the one about to install a reporter. A chained verb
    (``init --up`` invokes ``build`` and ``up``) finds the outer verb's reporter
    and leaves the card to it, so a chain prints one card at its end rather than
    one per verb.
    """
    active = current_reporter()
    return isinstance(active, NullReporter) and not active.verbose


def format_summary_card(repo_root: Path | str, state: str) -> list[str]:
    """Render the card for ``repo_root`` in ``state`` as plain lines.

    :param repo_root: The deployment repo -- its directory name IS the
        deployment's name, the same one the compose project carries
    :param state: One of ``created``, ``built``, ``running``, ``stopped``
    """
    from osprey.deployment.deploy_summary import as_built_endpoint_entries

    root = Path(repo_root)
    rows = []
    if state == "running":
        # URLs only: a card is the "what now" surface, and the addresses someone
        # can act on are the ones they can open. The full list, bare addresses
        # and the not-configured facts included, is `osprey status`.
        rows += [(s, a) for s, a in as_built_endpoint_entries(root) if a.startswith("http://")]
    rows.append(("command output", str(root / SPOOL_DIR)))
    rows.append(("next", _NEXT_STEPS.get(state, "osprey status")))

    width = max(len(label) for label, _ in rows)
    return [f"{root.name} — {state}"] + [
        f"  {label.ljust(width)}   {value}" for label, value in rows
    ]


def print_summary_card(repo_root: Path | str, state: str) -> None:
    """Print the card through the installed reporter; advisory, never raises.

    A verb that has done its work has succeeded, and a card that cannot be
    rendered -- an unreadable render, a compose file that moved -- must not turn
    that into a failure.
    """
    try:
        lines = format_summary_card(repo_root, state)
    except Exception as exc:
        logger.debug("Summary card skipped: %s", exc)
        return
    reporter = current_reporter()
    reporter.emit("")
    reporter.emit(lines[0], style=Styles.BOLD)
    for line in lines[1:]:
        reporter.emit(line)
