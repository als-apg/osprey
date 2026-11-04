"""``osprey write-authorization``: approve, list and revoke a program's write authorization.

A program that writes through the connector with no person at a prompt — an
optimizer run — asks for a bounded authorization by writing a request
(:func:`osprey_connectors.write_authorization.write_request`). The web
terminal's card approves it; this group is the same approval without a
terminal: on the deployment host, against the deployment's control-context
tree, or on a laptop with no deployment at all, against this account's own
state directory.

* ``approve <request>`` — a request file, or a request id looked up in the
  holders' request directory. The request is checked as the card checks it
  (a configured control target, the owner's ceiling, the limits database,
  overlap with the owner's grants and the programs' self-issued ones), shown,
  confirmed, and the grant issued under the
  owner's record lock with ``approver.identity`` this process's acting identity
  and ``surface: "cli"``.
* ``list`` — the pending requests and the authorizations, each marked in force
  or not; grants that ended more than an hour ago are pruned on the way. The
  programs' self-issued grants (written by a program at ``allowed`` into the
  holders' write ledger) are listed too, marked self-issued.
* ``revoke <id>`` — mark one authorization revoked, a self-issued one in the
  write ledger included; the holder's next write is refused. Turning the
  machine's switch off revokes every one on it.

Where things are: inside a deployment repo (or with ``--repo``), the owner's
directory is ``control_target/<owner>/`` under the deployment's agent-data
root, and the requests are in ``var/write_ledger/requests``; outside one, the
owner is this account and everything is in its own state directory.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from osprey.cli.output import fail, machine_mode, note, report, table, warn
from osprey.cli.repo_resolver import RepoNotFoundError, find_repo_root, repo_option
from osprey.cli.styles import Styles, data_table

__all__ = ["write_authorization"]


@dataclass(frozen=True)
class _Scope:
    """Where this run reads and writes: a deployment's tree, or this host alone.

    Attributes:
        repo_root: The deployment repo, or ``None`` on a host with none.
        config: The rendered config (the deployment's, or the host's), or ``None``.
        config_path: Its file, the anchor a relative limits path resolves on.
        tree: The deployment's control-context tree, or ``None``.
        requests: The holders' request directory, or ``None``.
        ledger: The holders' ledger directory, or ``None``.
    """

    repo_root: Path | None
    config: dict[str, Any] | None
    config_path: Path | None
    tree: Path | None
    requests: Path | None
    ledger: Path | None

    def record_file(self, owner: str) -> Path | None:
        """*owner*'s control-context record: in the tree, else ``None`` (this host's own)."""
        if self.tree is None or self.repo_root is None:
            return None
        from osprey.deployment.compose_generator import control_target_identity_dir
        from osprey_connectors.control_context import RECORD_FILENAME

        directory: Path = control_target_identity_dir(self.config or {}, self.repo_root, owner)
        return directory / RECORD_FILENAME

    def grants_dir(self, owner: str) -> Path | None:
        """Where *owner*'s grants are, as this run reaches them."""
        from osprey_connectors import write_authorization as wa

        record = self.record_file(owner)
        if record is not None:
            return record.parent / wa.AUTHORIZATIONS_DIR_NAME
        return wa.authorizations_dir(owner)

    def request_dirs(self, logins: list[str]) -> list[Path]:
        """Every place a request naming one of *logins* is filed, each once.

        The holders' request directory (a program in a container, its ledger
        bound), then each login's own grants directory: a program run on the
        host with no ledger bound files there, and keeps its ledger there.
        """
        from osprey_connectors import write_authorization as wa

        places = [] if self.requests is None else [self.requests]
        for login in logins:
            directory = self.grants_dir(login)
            if directory is not None:
                places.append(directory / wa.REQUESTS_DIR_NAME)
        return _unique(places)

    def ledger_dirs(self, owner: str) -> list[Path]:
        """Where the ledger of a grant *owner* approved can be: the holders', then the owner's own."""
        directory = self.grants_dir(owner)
        places = [] if self.ledger is None else [self.ledger]
        return _unique([*places, *([] if directory is None else [directory])])

    def holder_in_container(self, request_file: Path) -> bool:
        """Whether *request_file* was filed by a program whose ledger the deployment binds.

        Such a program runs in a shared container, under the control-context
        tree, and its writes are judged by the login map the build staged; a
        request anywhere else comes from a program on this host.
        """
        if self.repo_root is None or self.requests is None:
            return False
        try:
            return request_file.resolve().parent == self.requests.resolve()
        except OSError:
            return False


def _unique(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    kept = []
    for path in paths:
        key = os.path.normpath(path)
        if key not in seen:
            seen.add(key)
            kept.append(path)
    return kept


def _load(path: Path) -> dict[str, Any] | None:
    from osprey_connectors import yaml_loader

    try:
        document = yaml_loader.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    except Exception:  # a config that cannot be parsed is no config
        return None
    return document if isinstance(document, dict) else None


def _scope(repo: Path | None, config_option: Path | None) -> _Scope:
    """The deployment the operator stands in, or this host when there is none."""
    from osprey_connectors import write_authorization as wa

    try:
        root: Path | None = find_repo_root(repo)
    except RepoNotFoundError:
        if repo is not None:
            raise
        root = None
    if root is not None:
        from osprey.deployment.compose_generator import (
            control_target_tree_dir,
            write_ledger_relpath,
            write_requests_relpath,
        )
        from osprey_connectors.workspace import rendered_config_path

        config_path = config_option or rendered_config_path(root)
        config = _load(config_path) if config_path.is_file() else None
        return _Scope(
            repo_root=root,
            config=config,
            config_path=config_path,
            tree=control_target_tree_dir(config or {}, root),
            requests=root / write_requests_relpath(),
            ledger=root / write_ledger_relpath(),
        )
    from osprey_connectors.workspace import resolve_config_path

    config_path = config_option or resolve_config_path()
    return _Scope(
        repo_root=None,
        config=_load(config_path) if config_path.is_file() else None,
        config_path=config_path,
        tree=None,
        requests=wa.requests_dir(),
        ledger=wa.ledger_dir(),
    )


def _ceiling(scope: _Scope, owner: str, *, in_container: bool) -> str:
    """*owner*'s write ceiling for the program asking: ``never``, ``asked``, ``allowed`` or ``unknown``.

    Judged as that program's own writes will be: the deployment's tier, then,
    for a program in a shared container (*in_container*), the owner's rung in
    the login map those containers enforce (staged by the build); for one on
    this host, the rendered config, as the web terminal's card reads it.
    """
    from osprey_connectors import posture_store
    from osprey_connectors.types import tier_writes

    section = (scope.config or {}).get("control_system")
    try:
        if tier_writes(section) == "never":
            return "never"
        if in_container and scope.repo_root is not None:
            from osprey.deployment.status_display import _mapped_tier, _staged_login_tiers

            staged = _staged_login_tiers(scope.repo_root)
            if staged is not None:
                return str(_mapped_tier(staged, owner))
        rung = posture_store.owner_tier_term(section, owner)
        if rung in ("never", "unknown"):
            return rung
        return posture_store.write_ceiling(section, owner)
    except Exception:  # an unreadable tier is not an allowed one
        return "unknown"


def _configured_targets(scope: _Scope) -> tuple[str, ...]:
    """The control targets the config describes, as the chip's approval checks them; ``()`` for none."""
    from osprey_connectors.types import configured_targets

    try:
        return tuple(configured_targets((scope.config or {}).get("control_system")))
    except Exception:  # an unreadable config configures no targets
        return ()


def _find_request(scope: _Scope, request: str, logins: list[str]) -> Path:
    """The request file *request* names: a path, or an id in a place a request is filed."""
    candidate = Path(request)
    if candidate.suffix == ".json" or candidate.exists():
        return candidate
    places = scope.request_dirs(logins)
    if not places:
        raise click.ClickException(
            f"No request directory is reachable to look {request!r} up in; name the file."
        )
    for directory in places:
        if (directory / f"{request}.json").exists():
            return directory / f"{request}.json"
    return places[0] / f"{request}.json"


def _no_request(scope: _Scope, request: str, path: Path, logins: list[str]) -> str:
    """Why *request* names no pending request, as the operator can act on it.

    A request named by its file is reported by its file. One named by id is
    looked up among the owners' authorizations: an approved request's grant
    carries its id, so a grant there means it was approved already.
    """
    if Path(request).suffix == ".json" or os.sep in request:
        return f"No request at {path}."
    for login in logins:
        directory = scope.grants_dir(login)
        if directory is not None and (directory / f"{request}.json").exists():
            return f"Request {request} was already approved."
    return (
        f"No pending request {request}: it was already approved, withdrawn when its run "
        "ended, or never filed. `osprey write-authorization list` shows what is pending."
    )


def _show_card(card: dict[str, Any]) -> None:
    from osprey.write_authorization import LIMITS_COST_SENTENCE

    report(
        f"{card['holder']} asks to write on {card['target']} for {card['owner']}"
        + (f" — {card['label']}" if card["label"] else "")
    )
    grid = data_table()
    grid.add_column("Channel", style=Styles.ACCENT, no_wrap=True)
    grid.add_column("Range", style=Styles.PRIMARY)
    grid.add_column("Limits", style=Styles.INFO)
    grid.add_column("Problem", style=Styles.WARNING)
    for channel in card["channels"]:
        low, high = channel["limit_min"], channel["limit_max"]
        limits = "none" if low is None and high is None else f"[{low}, {high}]"
        grid.add_row(
            channel["address"],
            f"[{channel['min']}, {channel['max']}]",
            limits,
            channel["problem"] or "",
        )
    table(grid)
    report(f"  Steps: {card['steps']} write calls; at the end the run does: {card['completion']}")
    report(f"  Expires: {card['expires_at']}; lease {card['lease_s']:g} s; holder {card['holder']}")
    note(LIMITS_COST_SENTENCE)


@click.group("write-authorization")
def write_authorization() -> None:
    """Approve, list and revoke programs' bounded write authorizations.

    A program (an optimizer run) asks to write a few channels, inside ranges,
    a number of times, until an expiry; approving issues that and nothing more.
    Inside a deployment repo this acts on the deployment's control-context
    tree; outside one, on this account's own.

    \b
    Examples:
      osprey write-authorization list
      osprey write-authorization approve 0123…cdef --owner alice
      osprey write-authorization revoke 0123…cdef
    """


@write_authorization.command("approve")
@click.argument("request")
@click.option("--owner", help="The login approving. Default: the request's owner.")
@click.option(
    "--config",
    "config_option",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Config whose limits database bounds the ranges. Default: the deployment's render.",
)
@click.option("--yes", is_flag=True, help="Approve without asking to confirm.")
@repo_option
def approve(
    request: str, owner: str | None, config_option: Path | None, yes: bool, repo: Path | None
) -> None:
    """Approve REQUEST (a request file, or its id) as its owner."""
    from osprey import write_authorization as approval
    from osprey_connectors import write_authorization as wa
    from osprey_connectors.identity import acting_identity

    scope = _scope(repo, config_option)
    logins = [owner] if owner else _owners(scope, None)
    path = _find_request(scope, request, logins)
    try:
        wanted = wa.read_request(path)
    except FileNotFoundError:
        raise click.ClickException(_no_request(scope, request, path, logins)) from None
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"The request at {path} cannot be read: {exc}") from None
    login = owner or wanted.owner
    if login != wanted.owner:
        raise click.ClickException(f"The request is {wanted.owner}'s to approve, not {login}'s.")
    if scope.tree is None and login != acting_identity():
        raise click.ClickException(
            f"Outside a deployment only this account ({acting_identity()}) can approve; "
            f"the request names {login}. Run this in the deployment repo, or pass --repo."
        )
    ceiling = _ceiling(scope, login, in_container=scope.holder_in_container(path))
    if ceiling in ("never", "unknown"):
        raise click.ClickException(
            f"{login}'s write tier is {ceiling}: no write authorization can be approved for it."
        )
    if scope.config is None:
        raise click.ClickException(
            f"No config to read the limits database from at {scope.config_path}; pass --config."
        )
    configured = _configured_targets(scope)
    if wanted.target not in configured:
        raise click.ClickException(
            f"The request names {wanted.target!r}, which is not a control target of this "
            f"deployment ({', '.join(configured) or 'none configured'}); nothing was approved."
        )
    try:
        limits = approval.load_limits(scope.config, scope.config_path)
    except approval.ApprovalRefused as refused:
        raise click.ClickException(refused.detail) from None
    checks = approval.check_channels(wanted, limits)
    _show_card(approval.card(wanted, checks))
    problem = approval.refusal(checks)
    if problem is not None:
        raise click.ClickException(f"Not approved ({problem.reason}): {problem.detail}")
    if not yes and not click.confirm(f"Approve {wanted.id} for {wanted.holder}?", default=False):
        warn("Not approved.")
        raise click.Abort()
    peers = None if scope.ledger is None else (scope.ledger,)
    try:
        grant = wa.approve_request(
            wanted, surface="cli", record_file=scope.record_file(login), peers=peers
        )
    except wa.AuthorizationRefused as exc:
        raise click.ClickException(f"Not approved ({exc.reason}): {exc}") from None
    except FileExistsError:
        raise click.ClickException(f"{wanted.id} is approved already.") from None
    except (OSError, RuntimeError) as exc:
        fail("Not approved: nothing was written", str(exc))
        raise click.Abort() from None
    report(
        f"Approved {grant.id}: {grant.holder} may write {grant.steps} times on {grant.target} "
        f"(approver {grant.approver.identity}, cli)."
    )


def _owners(scope: _Scope, owner: str | None) -> list[str]:
    from osprey_connectors.identity import acting_identity

    if owner is not None:
        return [owner]
    if scope.tree is None:
        return [acting_identity()]
    try:
        return sorted(
            child.name
            for child in scope.tree.iterdir()
            if child.is_dir() and not child.is_symlink() and not child.name.startswith(".")
        )
    except OSError:
        return []


@write_authorization.command("list")
@click.option("--owner", help="Only this login's. Default: every login in the deployment.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of tables.")
@repo_option
def list_(owner: str | None, as_json: bool, repo: Path | None) -> None:
    """List the pending requests and the authorizations, pruning what ended long ago."""
    from osprey.write_authorization import grant_summary
    from osprey_connectors import control_context
    from osprey_connectors import write_authorization as wa

    scope = _scope(repo, None)
    now = time.time_ns()
    grants: list[dict[str, Any]] = []
    records: dict[str, Any] = {}

    def record_of(login: str) -> Any:
        if login not in records:
            record_file = scope.record_file(login)
            records[login] = (
                control_context.read_record_status(path=record_file)
                if record_file is not None
                else control_context.read_record_status()
            )[0]
        return records[login]

    owner_dirs: list[Path] = []
    for login in _owners(scope, owner):
        directory = scope.grants_dir(login)
        if directory is None:
            continue
        try:
            wa.prune_grants(directory, now_ns=now)
        except (OSError, RuntimeError) as exc:
            warn(f"{login}'s authorizations cannot be pruned: {exc}")
        owner_dirs.append(directory)
    # The programs' self-issued grants are in the holders' write ledger, beside
    # the ledgers that count every grant's steps.
    ledgers = [] if scope.ledger is None else [scope.ledger]
    found, unlisted = wa.listed_grants(owner_dirs, ledgers)
    for path in unlisted:
        warn(f"Cannot be read: {path}")
    for grant, _directory in found:
        if owner is not None and grant.owner != owner:
            continue
        standing = wa.grant_standing(
            grant, record_of(grant.owner), scope.ledger_dirs(grant.owner), now_ns=now
        )
        row = grant_summary(grant, used=standing.used, now_ns=now)
        row["in_force"] = standing.in_force
        row["self_issued"] = grant.approver.surface == wa.SELF_ISSUE_SURFACE
        grants.append(row)
    pending: list[dict[str, Any]] = []
    issued = {row["id"] for row in grants}
    for directory in scope.request_dirs(_owners(scope, owner)):
        try:
            requests, _unreadable = wa.list_requests(directory)
        except OSError:
            requests, _unreadable = (), ()
        for request in requests:
            if request.id in issued or now >= request.expires_at_ns:
                continue
            if owner is not None and request.owner != owner:
                continue
            issued.add(request.id)
            pending.append(
                {
                    "id": request.id,
                    "owner": request.owner,
                    "holder": request.holder,
                    "target": request.target,
                    "steps": request.steps,
                    "label": request.label,
                }
            )
    if as_json:
        with machine_mode():
            # ALLOWLISTED raw click.echo, not a renderer primitive: a --json
            # verb's stdout is one document.
            click.echo(json.dumps({"requests": pending, "authorizations": grants}, indent=2))
        return
    if pending:
        grid = data_table()
        for column in ("Request", "Owner", "Holder", "Target", "Steps", "Label"):
            grid.add_column(column)
        for row in pending:
            grid.add_row(
                row["id"],
                row["owner"],
                row["holder"],
                row["target"],
                str(row["steps"]),
                row["label"],
            )
        report("Requests waiting for approval:")
        table(grid)
    if grants:
        grid = data_table()
        for column in ("Authorization", "Owner", "Holder", "Target", "Steps", "State"):
            grid.add_column(column)
        for row in grants:
            spent = "?" if row["used"] is None else str(row["used"])
            state = "in force" if row["in_force"] else ("revoked" if row["revoked"] else "ended")
            grid.add_row(
                row["id"],
                row["owner"],
                row["holder"] + (" (self-issued)" if row["self_issued"] else ""),
                row["target"],
                f"{spent}/{row['steps']}",
                state,
            )
        report("Authorizations:")
        table(grid)
    if not pending and not grants:
        note("No request is waiting and no authorization is recorded.")


@write_authorization.command("revoke")
@click.argument("authorization")
@click.option("--owner", help="The login whose authorization it is. Default: found by id.")
@repo_option
def revoke(authorization: str, owner: str | None, repo: Path | None) -> None:
    """Revoke AUTHORIZATION: the holder's next write is refused."""
    from osprey_connectors import write_authorization as wa
    from osprey_connectors.identity import acting_identity

    scope = _scope(repo, None)
    by = acting_identity()
    places: list[tuple[str | None, Path | None]] = [
        (login, scope.grants_dir(login)) for login in _owners(scope, owner)
    ]
    # A program's self-issued grant is in the holders' write ledger, which this
    # host writes; a web terminal only reads it, and ends one through the switch.
    places.append((None, scope.ledger))
    for login, directory in places:
        if directory is None or not (directory / f"{authorization}.json").exists():
            continue
        try:
            marked = wa.revoke_grants(by=by, directory=directory, ids=[authorization])
        except (OSError, RuntimeError) as exc:
            fail("The revocation did not take effect", str(exc))
            raise click.Abort() from None
        whose = "self-issued" if login is None else f"{login}'s"
        if marked:
            report(f"Revoked {authorization} ({whose}), by {by}: its next write is refused.")
        else:
            note(f"{authorization} was already revoked.")
        return
    raise click.ClickException(f"No authorization {authorization} is recorded here.")
