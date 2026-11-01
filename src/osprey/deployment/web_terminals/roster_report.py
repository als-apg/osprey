"""Who can open what: the roster, the shared cards and the access between them.

The answer is spread over a deployment's files and one rule that lives only in
code: the roster names people and their renders, ``shared`` names the apps, each
render states its own write posture, and every app is admitted to the whole
roster by :func:`~osprey.deployment.web_terminals.roster.resolve_cards`. This
module joins them into one document, for ``osprey users list``.

It decides nothing of its own. Cards come from :func:`resolve_cards`, principals
from :func:`resolve_access_principals`, ports from :func:`allocate_ports`, and
identities are compared by the login service's own comparators, so the report
cannot drift from what the render and the sign-in path enforce.

**The matrix only means something behind a login wall.** ``access`` is enforced
by the auth sidecar, which runs under ``password`` and ``oidc``. Under ``none``
nginx vouches for every terminal, and under ``token`` a card opens to whoever
holds its URL, so the document says so rather than presenting a grid nothing
enforces.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from osprey.agent_record import credentials as creds
from osprey.agent_record import identities as ids
from osprey.agent_record.client import RecordClient
from osprey.agent_record.config import parse_agent_record_block
from osprey.audit.envelope import DECISION_ALLOWED
from osprey.deployment.runtime_helper import get_runtime_command
from osprey.deployment.web_terminals.naming import web_container_name
from osprey.deployment.web_terminals.ports import allocate_ports, base_ports_from_config
from osprey.deployment.web_terminals.roster import (
    entry_is_shared,
    is_app_entry,
    resolve_access_principals,
    resolve_cards,
)
from osprey.port_layout import resolve_port_base
from osprey.services.auth_sidecar.identity_headers import same_domain, same_identity

#: Auth methods whose login service enforces ``access``.
WALLED_METHODS = frozenset({"password", "oidc"})

#: The claim identities compare under when the deployment configures none.
_DEFAULT_CLAIM = "email"

#: How long one container inspect may take before its state reads as unknown.
_LIVE_TIMEOUT_S = 5

#: The audit surface and detail form the login service files under.
AUTH_SIDECAR_SURFACE = "auth_sidecar"
_OPENER = re.compile(r"opener=(\S+)")

#: How many of the newest login records one report reads.
_SIGN_IN_ROWS = 1000

_USER_PREFIX = "user:"
_DOMAIN_PREFIX = "domain:"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _render_writes(repo_root: Path, render: Mapping[str, Any]) -> str | None:
    """A render's ``control_system.writes``, or ``None`` when it cannot be read."""
    project_path = render.get("project_path")
    if not isinstance(project_path, str):
        return None
    try:
        document = yaml.safe_load((repo_root / project_path / "config.yml").read_text())
    except (OSError, yaml.YAMLError):
        return None
    writes = _as_dict(_as_dict(document).get("control_system")).get("writes")
    return writes if isinstance(writes, str) else None


def _principals(entry: Mapping[str, Any]) -> list[str]:
    """The entry's principals, sorted; owner-only when they cannot be read.

    A read-only report degrades here rather than refusing: the build refuses a
    malformed ``access``, and owner-only is the reading that claims the least.
    """
    try:
        return sorted(resolve_access_principals(entry))
    except ValueError:
        return ["self"]


def _opened_by(principals: list[str]) -> str:
    """The principals in words, for the table's last column."""
    words = {"roster": "any roster login", "self": "self"}
    return ", ".join(words.get(member, member) for member in principals)


def _may_open(person: Mapping[str, Any], card: Mapping[str, Any], *, claim: str) -> bool:
    """Whether a person on the roster may open a card, under the sidecar's rules."""
    identity = person.get("identity") or person["name"]
    for member in card["access"]:
        if member == "roster":
            return True
        if member == "self" and person["name"] == card["name"]:
            return True
        if member.startswith(_USER_PREFIX) and same_identity(
            identity, member[len(_USER_PREFIX) :], claim=claim
        ):
            return True
        if member.startswith(_DOMAIN_PREFIX) and same_domain(
            identity, member[len(_DOMAIN_PREFIX) :]
        ):
            return True
    return False


def build_roster_report(
    config: Mapping[str, Any],
    repo_root: Path,
    *,
    project_name: str,
    app_titles: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The whole answer for one deployment, as a JSON-ready document.

    Args:
        config: The rendered ``build/config.yml``.
        repo_root: The deployment repo root, which render paths are relative to.
        project_name: The deployment's project name.
        app_titles: Each app's title, as :func:`app_titles_in` reads them.

    Returns:
        ``deployment``, ``auth``, ``default_tier``, ``cards`` (every card, with
        ``kind`` ``user`` or ``app``), ``people`` (each named login with the
        cards it may open), ``off_roster`` (cards a ``user:`` or ``domain:``
        principal opens beyond the roster) and ``unused_tiers``.
    """
    modules = _as_dict(config.get("modules"))
    web_terminals = _as_dict(modules.get("web_terminals"))
    auth = _as_dict(web_terminals.get("auth"))
    method = auth.get("method") or "token"
    claim = _as_dict(auth.get("oidc")).get("claim") or _DEFAULT_CLAIM
    catalog = _as_dict(web_terminals.get("renders"))
    registry = _as_dict(web_terminals.get("registry"))
    base_ports = base_ports_from_config(web_terminals, base=resolve_port_base(config))

    cards: list[dict[str, Any]] = []
    for entry in resolve_cards(
        web_terminals,
        registry,
        strict=False,
        project_name=project_name,
        app_titles=app_titles,
    ):
        render = entry.get("render")
        try:
            port = allocate_ports(base_ports, entry["index"])["web"]
        except ValueError:
            port = None
        principals = _principals(entry)
        is_app = is_app_entry(entry)
        cards.append(
            {
                "name": entry["name"],
                "kind": "app" if is_app else "user",
                "display_name": entry.get("display_name"),
                "identity": entry.get("oidc_subject"),
                "control_identity": entry.get("control_identity"),
                "render": render,
                "writes": _render_writes(repo_root, _as_dict(catalog.get(render))),
                "port": port,
                "access": principals,
                "opened_by": _opened_by(principals),
                "shared": is_app or entry_is_shared(dict(entry)),
            }
        )

    people = []
    for card in cards:
        if card["kind"] != "user" or card["shared"]:
            continue
        opens = [other["name"] for other in cards if _may_open(card, other, claim=claim)]
        people.append(
            {
                "name": card["name"],
                "identity": card["identity"],
                "render": card["render"],
                "writes": card["writes"],
                "opens": opens,
            }
        )

    used = {card["render"] for card in cards}
    used.add(_as_dict(_as_dict(config.get("services")).get("dispatch_worker")).get("tier"))
    unused_tiers = sorted(
        name
        for name, render in catalog.items()
        if _as_dict(render).get("kind") == "tier" and name not in used
    )
    off_roster = {
        card["name"]: [
            member for member in card["access"] if member.startswith((_USER_PREFIX, _DOMAIN_PREFIX))
        ]
        for card in cards
        if any(member.startswith((_USER_PREFIX, _DOMAIN_PREFIX)) for member in card["access"])
    }
    return {
        "deployment": project_name,
        "auth": {
            "method": method,
            "claim": claim if method == "oidc" else None,
            "enforced": method in WALLED_METHODS,
        },
        "default_tier": web_terminals.get("default_tier"),
        "cards": cards,
        "people": people,
        "off_roster": off_roster,
        "unused_tiers": unused_tiers,
    }


def _container_state(runtime: str, container: str) -> str | None:
    """The container's state, or ``None`` when the runtime cannot say."""
    try:
        result = subprocess.run(
            [runtime, "inspect", "--format", "{{.State.Status}}", container],
            capture_output=True,
            text=True,
            check=False,
            timeout=_LIVE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (result.stdout or "").strip().lower() or None if result.returncode == 0 else None


def last_sign_ins(entries: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Each name's newest admitted sign-in, from the login service's audit entries.

    A login files under the card it unlocked. A shared card also names who opened
    it in ``detail``, so that opener counts as signing in too. *entries* may come
    in any order; the newest timestamp wins.
    """
    newest: dict[str, str] = {}

    def note(name: str | None, when: str) -> None:
        if name and when > newest.get(name, ""):
            newest[name] = when

    for entry in entries:
        body = _as_dict(entry.get("body"))
        when = entry.get("ts")
        if body.get("decision") != DECISION_ALLOWED or not isinstance(when, str):
            continue
        note(body.get("subject"), when)
        opener = _OPENER.search(str(body.get("detail") or ""))
        if opener:
            note(opener.group(1), when)
    return newest


def _read_sign_ins(config: Mapping[str, Any], repo_root: Path) -> dict[str, str] | None:
    """The record's sign-ins; ``None`` when no running record answers.

    Reads with the operator login the build minted and never starts a record: a
    report must not bring up a service to fill in one column.
    """
    login = ids.login_for(ids.OPERATOR_IDENTITY)
    token = creds.read_login(repo_root / creds.login_file(login), login)
    if token is None:
        return None
    try:
        settings = parse_agent_record_block(config, base=resolve_port_base(config))
    except ValueError:
        return None
    client = RecordClient(f"http://127.0.0.1:{settings.port}", login, token)
    answer = client.recent_audit(surface=AUTH_SIDECAR_SURFACE, limit=_SIGN_IN_ROWS)
    if not answer.available or not isinstance(answer.data, list):
        return None
    return last_sign_ins(entry for entry in answer.data if isinstance(entry, Mapping))


def attach_live(report: dict[str, Any], config: Mapping[str, Any], repo_root: Path) -> None:
    """Add each card's container state and last sign-in, in place.

    Both degrade to ``None`` off the host or with the stack down: the static
    report stays true either way, so a missing runtime is not a failure.
    """
    try:
        # The compose command's first word is the runtime; inspect is not a
        # compose verb.
        runtime = get_runtime_command(config)[0]
    except Exception:  # no runtime on this machine: the state column stays empty.
        runtime = None
    sign_ins = _read_sign_ins(config, repo_root)
    for card in report["cards"]:
        container = web_container_name(report["deployment"], card["name"])
        card["state"] = _container_state(runtime, container) if runtime else None
        card["last_sign_in"] = (sign_ins or {}).get(card["name"])
    for person in report["people"]:
        person["last_sign_in"] = (sign_ins or {}).get(person["name"])
    report["live"] = {"sign_ins_available": sign_ins is not None}
