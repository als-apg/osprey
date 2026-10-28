"""Identity and header helpers shared by the Bluesky write-tool suites."""

from osprey.utils.owner_header import OWNER_HEADER
from osprey_connectors.identity import TERMINAL_USER_ENV
from osprey_connectors.posture_store import CONTROL_CONTEXT_TREE_ENV_VAR, CONTROL_OWNER_ENV_VAR


def headers_without_owner(m) -> dict[str, str]:
    """The headers the mocked request ``m`` carried, minus the owner stamp.

    Every write to the bridge carries ``X-Osprey-Owner`` whenever the identity
    ladder can name an owner, and under test the process account always can.
    Rows about gating or the launch token compare the rest of the header dict
    exactly, so a stray credential or third header still fails them, and leave
    the owner half to the rows that pin it.
    """
    headers = m.call_args.kwargs["headers"] or {}
    return {name: value for name, value in headers.items() if name != OWNER_HEADER}


def as_terminal_user(monkeypatch, user: str) -> None:
    """Put the process in a terminal container: the roster account, no stamp over it."""
    monkeypatch.delenv(CONTROL_OWNER_ENV_VAR, raising=False)
    monkeypatch.delenv(CONTROL_CONTEXT_TREE_ENV_VAR, raising=False)
    monkeypatch.setenv(TERMINAL_USER_ENV, user)


def as_owner_less_tree_container(monkeypatch, tmp_path) -> None:
    """Put the process in a container holding the state tree and owning nothing.

    The account such a container runs as names nobody whose chip anyone set, so
    the ladder answers NO_OWNER rather than falling through to that account.
    """
    monkeypatch.delenv(CONTROL_OWNER_ENV_VAR, raising=False)
    monkeypatch.setenv(CONTROL_CONTEXT_TREE_ENV_VAR, str(tmp_path / "control-context"))
