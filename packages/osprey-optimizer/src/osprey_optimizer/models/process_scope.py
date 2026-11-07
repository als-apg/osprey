"""Whether this process may talk to more than one control-system target.

A process serves one target by default: an environment plugin binds the process
to the first target it connects to and refuses a different one, so the operating
system, not in-process bookkeeping, keeps targets apart. A process that has to
serve several (the API) says so with ``allow_multiple_targets()``, and the
plugin then hands each target's work to a child process of its own. That is an
in-process call on purpose: an environment variable would be inherited by every
job worker the process spawns.
"""

_multiple_targets_allowed = False


def allow_multiple_targets() -> None:
    """Let this process talk to more than one control-system target."""
    global _multiple_targets_allowed
    _multiple_targets_allowed = True


def multiple_targets_allowed() -> bool:
    """Whether ``allow_multiple_targets()`` has been called in this process."""
    return _multiple_targets_allowed
