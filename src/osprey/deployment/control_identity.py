"""Per-card control identity: the name uid 1000 resolves to inside a card.

A walled web-terminal card may carry a ``control_identity`` roster key. At boot
the entrypoint's root phase makes uid 1000 resolve to that name, so the EPICS
Channel Access user (and every other ``getpwuid`` consumer) reports the person
behind the card instead of the shared ``osprey`` account.

This module is a stdlib-only leaf: it is copied byte-for-byte into the build
directory and run inside containers by path, where the ``osprey`` package may
not be importable. It must never import anything from ``osprey.*``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile

__all__ = [
    "BASE_IMAGE_ACCOUNTS",
    "CANONICAL_NAME",
    "CONTROL_IDENTITY_CONTAINER_PATH",
    "IDENTITY_RE",
    "RESERVED_RE",
    "SKIPPED_ENV_VAR",
    "append_group_members",
    "apply",
    "health_fields",
    "main",
    "rewrite",
    "validate_identity",
]

CONTROL_IDENTITY_CONTAINER_PATH = "/opt/osprey/control_identity.py"
"""Where the staged copy of this module is mounted inside a container."""

IDENTITY_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}\Z")
"""A portable POSIX login name: lowercase, at most 32 characters."""

RESERVED_RE = re.compile(r"^(root|osprey(-[a-z0-9_-]*)?)\Z")
"""Names OSPREY owns: ``root``, ``osprey``, and the generated ``osprey-*`` service names."""

BASE_IMAGE_ACCOUNTS: frozenset[str] = frozenset(
    {
        # Debian base-passwd accounts shared by python:3.12-slim (project image,
        # plus its apt layer) and python:3.11-slim (bluesky image).
        "daemon",
        "bin",
        "sys",
        "sync",
        "games",
        "man",
        "lp",
        "mail",
        "news",
        "uucp",
        "proxy",
        "www-data",
        "backup",
        "list",
        "irc",
        "_apt",
        "nobody",
        # Present in bookworm-based slim images; kept so an image pinned to an
        # older base still refuses it.
        "gnats",
    }
)
"""Accounts already in either image's ``/etc/passwd``, minus ``root``/``osprey``.

A roster name that collides with one of these must be refused when the roster
is linted, not discovered as a crash-looping card at boot. The set is the
union over both images, so an account present in only one of them is still
refused everywhere.
"""

CANONICAL_NAME: dict[int, str] = {1000: "osprey", 0: "root"}
"""The image account each rewritable uid is canonically named."""

SKIPPED_ENV_VAR = "OSPREY_CONTROL_IDENTITY_SKIPPED"
"""Set by the entrypoint, to its reason, when it could not apply the intended identity."""


def health_fields() -> dict[str, str | None]:
    """The control-identity fields a service's ``/health`` reports.

    ``ca_user`` is the name this process's uid resolves to — the name the
    control system sees its writes arrive under — read in-process so it
    reflects what the entrypoint actually applied. ``control_identity_skipped``
    is the entrypoint's reason when it could not apply the intended identity,
    ``None`` otherwise. ``osprey health`` compares both with the rendered
    config.
    """
    import pwd  # POSIX-only; imported where used so the module loads anywhere.

    return {
        "ca_user": pwd.getpwuid(os.getuid()).pw_name,
        "control_identity_skipped": os.environ.get(SKIPPED_ENV_VAR),
    }


def validate_identity(name: object, *, allow_service: bool = False) -> None:
    """Raise ``ValueError`` unless ``name`` may be written into ``/etc/passwd``.

    Refused: a non-string, a name outside :data:`IDENTITY_RE` (which also
    excludes ``:`` and newlines, the passwd field and record separators), a
    reserved name (:data:`RESERVED_RE`), and a name in
    :data:`BASE_IMAGE_ACCOUNTS`.

    ``allow_service=True`` admits the generated ``osprey-*`` service names; it
    never admits ``root`` or ``osprey`` themselves.
    """
    if not isinstance(name, str):
        raise ValueError(f"control identity must be a string, got {type(name).__name__}")
    if ":" in name or "\n" in name or "\r" in name:
        raise ValueError(f"control identity {name!r} contains a passwd separator")
    if IDENTITY_RE.fullmatch(name) is None:
        raise ValueError(
            f"control identity {name!r} must match {IDENTITY_RE.pattern} "
            "(lowercase letter or underscore first, then up to 31 of a-z 0-9 _ -)"
        )
    if RESERVED_RE.fullmatch(name) is not None:
        service = name not in CANONICAL_NAME.values()
        if not (allow_service and service):
            raise ValueError(f"control identity {name!r} is reserved for OSPREY")
    if name in BASE_IMAGE_ACCOUNTS:
        raise ValueError(f"control identity {name!r} is already an account in the base image")


def rewrite(text: str, uid: int, name: str) -> str:
    """Return ``/etc/passwd`` text in which ``uid`` resolves to ``name``.

    The identity line is the canonical line for ``uid`` (``osprey`` for 1000,
    ``root`` for 0) with only its name field replaced, placed directly above
    the canonical line so ``getpwuid`` finds it first while the canonical
    account still resolves by name. Every other non-canonical line carrying
    ``uid`` is dropped, so the result is idempotent and a changed identity
    replaces the previous one. Lines that are not passwd records are kept.

    Raises ``ValueError`` for a name :func:`validate_identity` refuses (service
    names allowed), a ``uid`` with no canonical account, a missing canonical
    line, or a name already held by a different uid.
    """
    validate_identity(name, allow_service=True)
    canonical = CANONICAL_NAME.get(uid)
    if canonical is None:
        raise ValueError(f"uid {uid!r} has no canonical account to rewrite")
    uid_field = str(uid)

    out: list[str] = []
    found = False
    for line in text.splitlines():
        fields = line.split(":")
        if len(fields) != 7:
            out.append(line)
            continue
        if fields[0] == canonical and fields[2] == uid_field and not found:
            found = True
            out.append(":".join([name, *fields[1:]]))
            out.append(line)
        elif fields[2] == uid_field and fields[0] != canonical:
            continue
        elif fields[0] == name:
            raise ValueError(
                f"control identity {name!r} is already an account with uid {fields[2]}"
            )
        else:
            out.append(line)
    if not found:
        raise ValueError(f"/etc/passwd has no {canonical!r} line with uid {uid}")
    result = "\n".join(out)
    if text.endswith("\n"):
        result += "\n"
    return result


def append_group_members(text: str, canonical: str, name: str) -> str:
    """Return ``/etc/group`` text in which ``name`` shares ``canonical``'s memberships.

    For every group record whose member list contains ``canonical`` as a whole
    member, ``name`` is appended unless it is already a member. Members are
    never removed or reordered: the identity is fixed when the container is
    created and a recreate restores ``/etc/group`` from the image, so nothing
    legitimately calls for a strip, and one could delete a real member. The
    result is idempotent. Primary groups need no entry, because the passwd
    gid field already carries the canonical gid. Lines that are not group
    records are kept.

    Raises ``ValueError`` for a name :func:`validate_identity` refuses (service
    names allowed) or a ``canonical`` that is not a canonical account.
    """
    validate_identity(name, allow_service=True)
    if canonical not in CANONICAL_NAME.values():
        raise ValueError(f"{canonical!r} is not a canonical account")

    out: list[str] = []
    for line in text.splitlines():
        fields = line.split(":")
        if len(fields) != 4:
            out.append(line)
            continue
        members = fields[3].split(",") if fields[3] else []
        if canonical in members and name not in members:
            fields[3] = ",".join([*members, name])
            out.append(":".join(fields))
        else:
            out.append(line)
    result = "\n".join(out)
    if text.endswith("\n"):
        result += "\n"
    return result


SYSTEM_PASSWD = "/etc/passwd"
"""The passwd file NSS reads; only a write to it is checked through ``getpwuid``."""

SYSTEM_GROUP = "/etc/group"
"""The group file :func:`apply` rewrites by default."""

_FILE_MODE = 0o644


def _atomic_write(path: str, text: str) -> None:
    """Replace ``path`` with ``text`` via a same-directory temp file (mode 0644)."""
    directory, base = os.path.split(os.path.abspath(path))
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{base}.", suffix=".tmp")
    try:
        os.fchmod(fd, _FILE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _nss_name(uid: int) -> str:
    """Return the name NSS resolves ``uid`` to."""
    import pwd  # POSIX-only; imported where used so the module loads anywhere.

    return pwd.getpwuid(uid).pw_name


def _post_check(text: str, uid: int, name: str, canonical: str) -> None:
    """Raise ``RuntimeError`` unless ``name`` is the first passwd line for ``uid``."""
    uid_field = str(uid)
    names = [
        fields[0]
        for fields in (line.split(":") for line in text.splitlines())
        if len(fields) == 7 and fields[2] == uid_field
    ]
    if not names or names[0] != name:
        found = names[0] if names else None
        raise RuntimeError(f"post-check: uid {uid} resolves to {found!r}, not {name!r}")
    if canonical not in names:
        raise RuntimeError(f"post-check: canonical {canonical!r} line for uid {uid} is gone")


def apply(
    uid: int,
    name: str,
    *,
    passwd: str | os.PathLike[str] = SYSTEM_PASSWD,
    group: str | os.PathLike[str] = SYSTEM_GROUP,
) -> None:
    """Make ``uid`` resolve to ``name`` in ``passwd`` and give it ``group`` memberships.

    Both files are read and transformed (:func:`rewrite`,
    :func:`append_group_members`) before either is written, so a refused name
    or a malformed file changes nothing. Each file is then replaced atomically
    with mode 0644. The written passwd is re-parsed to confirm ``name`` is the
    first line for ``uid`` and the canonical line survives; when ``passwd`` is
    :data:`SYSTEM_PASSWD`, ``getpwuid`` must also report ``name``.

    Raises ``ValueError`` for a refused name or uid, ``OSError`` for an I/O
    failure, and ``RuntimeError`` when a post-check fails.
    """
    passwd_path = os.fspath(passwd)
    group_path = os.fspath(group)
    new_passwd = rewrite(_read(passwd_path), uid, name)
    canonical = CANONICAL_NAME[uid]
    new_group = append_group_members(_read(group_path), canonical, name)

    _atomic_write(passwd_path, new_passwd)
    _atomic_write(group_path, new_group)

    _post_check(_read(passwd_path), uid, name, canonical)
    if os.path.realpath(passwd_path) == os.path.realpath(SYSTEM_PASSWD):
        resolved = _nss_name(uid)
        if resolved != name:
            raise RuntimeError(f"post-check: getpwuid({uid}) is {resolved!r}, not {name!r}")


class _UsageError(Exception):
    """An argument error, reported as one stderr line instead of argparse's usage block."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise _UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="control_identity.py", add_help=False)
    commands = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)
    apply_cmd = commands.add_parser("apply", add_help=False)
    apply_cmd.add_argument("--uid", type=int, required=True)
    apply_cmd.add_argument("--name", required=True)
    apply_cmd.add_argument("--passwd", default=SYSTEM_PASSWD)
    apply_cmd.add_argument("--group", default=SYSTEM_GROUP)
    return parser


def main(argv: list[str]) -> int:
    """Run ``apply --uid <n> --name <id> [--passwd PATH] [--group PATH]``.

    Success prints nothing and returns 0. Any failure, including a bad or
    missing argument, prints exactly one line to stderr and returns non-zero,
    so a caller that chains ``apply ... || exit 1`` fails closed. There is no
    ``--help``: an exit 0 that applied nothing would read as success.
    """
    try:
        args = _parser().parse_args(argv)
        apply(args.uid, args.name, passwd=args.passwd, group=args.group)
    except _UsageError as exc:
        _fail(
            f"usage: control_identity.py apply --uid <n> --name <id> "
            f"[--passwd PATH] [--group PATH]: {exc}"
        )
        return 2
    except Exception as exc:  # every failure is one stderr line
        _fail(f"{type(exc).__name__}: {exc}")
        return 1
    return 0


def _fail(message: str) -> None:
    sys.stderr.write("control_identity: " + " ".join(message.split()) + "\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
