"""The closing summary card for the OSPREY lifecycle verbs.

``init``, ``build``, ``up -d``, ``restart -d`` and ``down`` all end the same
way: one short card saying which deployment this was, what state it is in now,
where it answers, where the command output went, and which command comes next.
One renderer for all five, so the five cannot describe the same deployment in
five shapes.

It prints after the last phase has closed, through the CLI's renderer
(:mod:`osprey.cli.output`) rather than through the phase reporter. The card is
the verb's own output, not decoration on its progress: it says what the run
left behind, so it is echo-class and prints under every reporter -- including
the one the global ``--verbose`` installs, where the phase record itself is
swallowed. Attached (non-``-d``) starts print no card at all: ``compose up``
replaces this process and the terminal belongs to the log stream.

The rows are a key/value block, which is what :func:`osprey.cli.output.section`
renders, so the card is laid out by the same code as every other block of facts
the CLI prints. Only the wording stays here: which states exist and what to run
next are this module's business, not the renderer's.
"""

from __future__ import annotations

from pathlib import Path

from osprey.deployment.subprocess_capture import SPOOL_DIR
from osprey.utils.logger import get_logger
from osprey.utils.workspace import BUILD_DIR_NAME

from . import output
from .phase_reporter import NullReporter, current_reporter

logger = get_logger("cli.summary_card")

#: What to do next, per state. Only ``running`` is reachable-anywhere, so it is
#: the only state whose card carries endpoints.
_NEXT_STEPS = {
    "created": "osprey build · osprey up -d",
    "built": "osprey up -d · osprey status",
    "running": "osprey status · osprey logs · osprey down",
    "stopped": "osprey up -d · osprey status",
}

#: Where a rendered project keeps the demo scenarios ``osprey sim apply`` reads.
#: Relative to the build directory, which is the tree the build wrote, not the
#: repo the operator is standing in.
_SCENARIOS_DIR = Path("data") / "simulation" / "scenarios"

#: The command that seeds the demo logbook from those scenarios. Appended to the
#: ``built`` card's next step, so the one next-step surface carries it instead of
#: a line of its own.
_SEED_DEMO_LOGBOOK = "osprey sim apply nominal"


def _next_step(root: Path, state: str) -> str:
    """What to run next for a deployment in ``state``, rooted at ``root``.

    Fixed per state, with one exception the map cannot hold: a build that
    rendered demo scenarios can seed its logbook from them, and one that did not
    must not be told to run a command with nothing to apply. So the answer is
    read off the rendered tree rather than declared.

    :param root: The deployment repo. The build it rendered is one directory in.
    :param state: One of ``created``, ``built``, ``running``, ``stopped``
    """
    step = _NEXT_STEPS.get(state, "osprey status")
    if state == "built" and (root / BUILD_DIR_NAME / _SCENARIOS_DIR).is_dir():
        return f"{step} · {_SEED_DEMO_LOGBOOK}"
    return step


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


def _card_parts(repo_root: Path | str, state: str) -> tuple[str, list[tuple[str, str]]]:
    """The card's heading and its rows, before either is laid out.

    The one derivation of what the card says. Both the printer and the
    plain-text renderer start here, so there is no way for the card an operator
    reads to differ from the card a test reads.

    :param repo_root: The deployment repo -- its directory name IS the
        deployment's name, the same one the compose project carries
    :param state: One of ``created``, ``built``, ``running``, ``stopped``
    """
    from osprey.deployment.deploy_summary import as_built_endpoint_entries

    root = Path(repo_root)
    rows: list[tuple[str, str]] = []
    if state == "running":
        # URLs only: a card is the "what now" surface, and the addresses someone
        # can act on are the ones they can open. The full list, bare addresses
        # and the not-configured facts included, is `osprey status`.
        rows += [(s, a) for s, a in as_built_endpoint_entries(root) if a.startswith("http://")]
    rows.append(("command output", str(root / SPOOL_DIR)))
    rows.append(("next", _next_step(root, state)))
    return f"{root.name} — {state}", rows


def format_summary_card(repo_root: Path | str, state: str) -> list[str]:
    """Render the card for ``repo_root`` in ``state`` as plain lines.

    The same lines :func:`print_summary_card` puts on screen, with no styling
    on them -- for a caller that wants the card as text rather than as output.
    The row shape is :func:`osprey.cli.output.section`'s, which is what makes
    the two agree; ``test_summary_card.py`` pins them against each other so
    they cannot drift apart.

    :param repo_root: The deployment repo -- its directory name IS the
        deployment's name, the same one the compose project carries
    :param state: One of ``created``, ``built``, ``running``, ``stopped``
    """
    title, rows = _card_parts(repo_root, state)
    width = max(len(label) for label, _ in rows)
    return [title] + [f"  {label.ljust(width)}   {value}" for label, value in rows]


def print_summary_card(repo_root: Path | str, state: str) -> None:
    """Print the card through the CLI renderer; advisory, never raises.

    A verb that has done its work has succeeded, and a card that cannot be
    rendered -- an unreadable render, a compose file that moved -- must not turn
    that into a failure.
    """
    try:
        title, rows = _card_parts(repo_root, state)
    except Exception as exc:
        logger.debug("Summary card skipped: %s", exc)
        return
    output.report("")
    output.section(title, rows)
