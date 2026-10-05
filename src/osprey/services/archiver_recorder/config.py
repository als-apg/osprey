"""What the recorder reads out of a deployment's rendered ``config.yml``.

Two blocks, both written by the profile's ``va_archiver:`` block (see
``osprey/cli/build_profile_archiver.py``) and neither duplicated here:

* ``archiver.mongodb_archiver.*`` — where the store is and how to authenticate,
  the same keys ``MongoDBArchiverConnector`` reads, so the writer and the reader
  cannot end up pointed at different collections.
* ``va_archiver.*`` — the cadences and retention spans that decide the shape of
  what gets written.

Nothing in this package carries a default for either. A missing key is an
error naming the key, not a silent fallback: a recorder that invented its own
cadence would write an archive whose density disagreed with the seeded half of
the very same collection, and nothing downstream would notice.

One value here does not come from the file: the store's address. The rendered
connection block is written for the HOST side — the agent runs there and reaches
the store on loopback at its published port — which inside the compose network
names this container's own loopback. The compose template hands the container
the store's network alias and container port instead, and those win.

That override is NOT defined here. It is read through
``MongoDBArchiverConnector``'s :func:`~osprey.connectors.archiver.mongodb_archiver_connector.address_overrides`,
where the contract is stated, so this service and the agent's connector cannot
drift apart on the variable names, on what an empty value means, or on how the
port is typed. Two readers of one convention have to agree on its edges.

Settings are resolved once. The one thing re-read on every poll is
:class:`RecordingFacts` — who is on the other end of the Channel Access
connection — and it is read out of the same file through the same resolvers the
roster and the target switch use, for the same reason: a guard that works out
privately what the reader it guards resolves can disagree with it, and the
disagreement is the bypass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from osprey.connectors.archiver.mongodb_archiver_connector import address_overrides

# The recorded endpoint is derived, never re-read: `derive_endpoints` is the same
# resolver the roster's label and the target switch use, so this service and the
# surfaces an operator reads cannot come to different answers about where the
# `standin` target lands. Imported at module scope because it costs nothing this
# service does not already pay — it pulls in `osprey_connectors` and the standard
# library and nothing else, no Channel Access stack, no PVA, no Mongo driver.
from osprey.mcp_server.control_system.target_eligibility import (
    derive_endpoints,
    endpoint_is_live_standin,
)
from osprey_connectors.connection import read_connection_settings

# Whose past the store holds, decided in one place for the recorder's compose
# entry, this enablement gate and the deploy-time archive seed alike. A guard
# that worked it out privately could disagree with the seed it shares a
# collection with.
from osprey_connectors.standin import archive_belongs_to_standin
from osprey_connectors.types import TARGET_STANDIN

from .field_names import _unstorable_field_name

#: Where the compose template mounts the render's simulator view,
#: ``build/data/simulator``: the same directory the virtual accelerator serves
#: from, so the recorder and the IOC read one ``addresses.json``.
DEFAULT_DATA_DIR = "/data/simulator"

#: The view's document listing the addresses the simulator serves.
ADDRESSES_FILE = "addresses.json"

#: Rendered-config subtree holding the connection keys (mirrors
#: ``build_profile_archiver.CONNECTION_CONFIG_PREFIX``).
_CONNECTION_PREFIX = ("archiver", "mongodb_archiver")

#: The login under the connection block.
_AUTH_PREFIX = (*_CONNECTION_PREFIX, "auth")

#: Rendered-config subtree holding the archive-shape knobs (mirrors
#: ``build_profile_archiver.KNOBS_CONFIG_PREFIX``).
_KNOBS_PREFIX = ("va_archiver",)


class RecorderConfigError(RuntimeError):
    """The deployment's config cannot back a recorder.

    Raised only from startup paths. Once the recorder is running, a config that
    goes unreadable is a torn read to be waited out, not a reason to stop
    recording — see :func:`read_recording_facts`.
    """


@dataclass(frozen=True)
class RecorderSettings:
    """Everything the recorder needs, resolved once at startup.

    Deliberately not re-read on the poll interval: a cadence or retention
    change alters the shape of the archive, so it goes through the seeder's
    fingerprint check and a redeploy, which restarts this service anyway. What
    IS re-read is :class:`RecordingFacts`, which changes only *who* is being
    recorded, not what the record looks like.
    """

    host: str
    port: int
    database: str
    collection: str
    auth_source: str
    username: str
    password_env: str
    timeout_s: int
    cadence_sec: int
    tail_cadence_sec: int
    poll_sec: int
    hot_span_hours: int
    retention_days: int
    #: A CA file to trust for the store (``tls.ca_bundle``); ``None`` keeps TLS off.
    ca_bundle: str | None = None


def load_settings(config_path: Path) -> RecorderSettings:
    """Resolve :class:`RecorderSettings` from a rendered ``config.yml``.

    Raises:
        RecorderConfigError: if the file is unreadable, is not a mapping, or is
            missing a key the recorder has no honest default for. The message
            names the key and the block it belongs to, because the fix is
            always a profile edit and a rebuild. Also raised when the block
            names its store by ``url``: that store is one this deployment reads,
            never one it writes.
    """
    config = _load_mapping(config_path)
    connection = _subtree(config, _CONNECTION_PREFIX, config_path)
    knobs = _subtree(config, _KNOBS_PREFIX, config_path)
    if connection.get("url"):
        raise RecorderConfigError(
            f"{config_path}: `{'.'.join((*_CONNECTION_PREFIX, 'url'))}` is set. The recorder "
            "writes only to the store this deployment runs, addressed by host and port."
        )
    auth, ca_bundle = _login_block(connection, config_path)

    # The in-network address override, read through the connector's own helper
    # so this service and the agent's connector cannot come to differ on what an
    # empty value means or on how the port is typed — one convention, one reader.
    override_host, override_port = address_overrides()

    host = override_host or _require(connection, "host", _CONNECTION_PREFIX, config_path)
    port = (
        override_port
        if override_port is not None
        else _as_int(
            _require(connection, "port", _CONNECTION_PREFIX, config_path),
            "port",
            _CONNECTION_PREFIX,
            config_path,
        )
    )

    return RecorderSettings(
        host=str(host),
        port=port,
        database=str(_require(connection, "name", _CONNECTION_PREFIX, config_path)),
        collection=str(_require(connection, "collection", _CONNECTION_PREFIX, config_path)),
        auth_source=str(_require(auth, "source", _AUTH_PREFIX, config_path)),
        username=str(_require(auth, "username", _AUTH_PREFIX, config_path)),
        password_env=str(_require(auth, "password_env", _AUTH_PREFIX, config_path)),
        timeout_s=_int_key(connection, "timeout_s", _CONNECTION_PREFIX, config_path),
        ca_bundle=ca_bundle,
        cadence_sec=_int_key(knobs, "recorder_cadence_sec", _KNOBS_PREFIX, config_path),
        tail_cadence_sec=_int_key(knobs, "recorder_tail_cadence_sec", _KNOBS_PREFIX, config_path),
        poll_sec=_int_key(knobs, "recorder_poll_sec", _KNOBS_PREFIX, config_path),
        hot_span_hours=_int_key(knobs, "hot_span_hours", _KNOBS_PREFIX, config_path),
        retention_days=_int_key(knobs, "retention_days", _KNOBS_PREFIX, config_path),
    )


@dataclass(frozen=True)
class RecordingFacts:
    """What the mounted config says about the machine on the other end.

    Two facts rather than one, and only one of them moves when an operator runs
    ``osprey set connector=epics``: the rendered ``control_system.type`` is
    rewritten, while a deployment that records its own stand-in goes on
    recording the same machine it always did. See
    :data:`~osprey.services.archiver_recorder.recorder.RECORDING_CONTROL_SYSTEM`
    for what the recorder does with the pair.

    Both are derived from a single parse of a single file. Two reads could
    straddle a config write and answer from two different configs, and the
    machine described by neither of them would be the one being recorded.
    """

    #: ``control_system.type`` exactly as written, stripped, ``''`` when unset.
    control_system_type: str
    #: Whether the machine this deployment records is its own stand-in — the
    #: deployment stood one up, it runs this recorder, and the ``standin``
    #: target's gateways still select that stand-in. See
    #: :func:`_recorded_target_is_standin`.
    live_standin: bool


def read_recording_facts(config_path: Path) -> RecordingFacts:
    """Read the enablement facts out of the mounted config, right now.

    This is the enablement question, asked again on every poll rather than
    answered once at startup, because both flipping ``control_system.type`` and
    repointing the ``standin`` target's gateways are documented post-build steps
    and neither must need a redeploy to take effect.

    Raises:
        RecorderConfigError: if the file cannot be read or parsed *at this
            moment*, or has no ``control_system:`` block. Callers keep their
            last known answer instead of acting on it: config writes are
            truncate-in-place, not atomic, so a poll that lands mid-write sees a
            torn file — and treating that as "the machine changed" would stop
            and restart recording on a file write that changed nothing.
    """
    config = _load_mapping(config_path)
    return RecordingFacts(
        control_system_type=_control_system_type(config, config_path),
        live_standin=_recorded_target_is_standin(config),
    )


def read_control_system_type(config_path: Path) -> str:
    """Read ``control_system.type`` from the mounted config, right now.

    The narrow question, for a caller that wants only the type. Enablement is
    decided from :func:`read_recording_facts`, which answers this one alongside
    the recorded machine's identity out of the same parse; asking here and there
    would be two reads of a file that is rewritten under both of them.

    Raises:
        RecorderConfigError: on the same terms as :func:`read_recording_facts`.
    """
    return _control_system_type(_load_mapping(config_path), config_path)


def resolve_channel_addresses(data_dir: Path | None = None) -> list[str]:
    """The addresses to record, in the order the simulator view lists them.

    The source is the ``channels`` list of the build's simulator view,
    ``addresses.json``, the file the virtual accelerator serves its namespace
    from, so the recorder covers precisely the facility's channels. The view's
    ``status`` list names the simulator's own model status channels and is not
    recorded. There is no fallback: a view that is missing or unreadable is
    refused, because the only channel set this service could pick on its own is
    one that is not this facility's, and an archive filled with those addresses
    under this facility's name is indistinguishable, later, from a real record
    of the facility.

    An address the archive cannot hold as a field name is refused here on the
    same terms (see :func:`~osprey.services.archiver_recorder.field_names._unstorable_field_name`),
    rather than started and discovered a tick at a time. ``osprey build`` stops
    on such an address whenever the recorder is configured, so a built
    deployment never reaches this refusal.

    Args:
        data_dir: The simulator view's directory; :data:`DEFAULT_DATA_DIR`
            when ``None``.

    Raises:
        RecorderConfigError: if ``addresses.json`` cannot be read, holds no
            ``channels`` list of strings, or lists an address the archive
            cannot store as a field name.
    """
    root = Path(data_dir) if data_dir is not None else Path(DEFAULT_DATA_DIR)
    path = root / ADDRESSES_FILE
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RecorderConfigError(
            f"cannot record: the simulator view's {ADDRESSES_FILE} ({path}) could not be "
            f"loaded: {exc}. `osprey build` writes it under build/data/simulator."
        ) from exc
    channels = document.get("channels") if isinstance(document, dict) else None
    if not isinstance(channels, list) or not all(isinstance(c, str) for c in channels):
        raise RecorderConfigError(
            f"cannot record: the simulator view's {ADDRESSES_FILE} ({path}) holds no "
            f"`channels` list of addresses. Rebuild the project with `osprey build`."
        )

    for address in channels:
        refusal = _unstorable_field_name(address)
        if refusal is not None:
            raise RecorderConfigError(
                f"cannot record: the simulator view's {ADDRESSES_FILE} ({path}) lists "
                f"{address!r}, which cannot be stored: it {refusal}. Every tick is one "
                f"document with a field per address, so this channel would be dropped "
                f"from the archive -- or take the whole write with it -- rather than "
                f"recorded. Rename the channel in the facility description and rebuild."
            )
    return list(channels)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _control_system_type(config: dict[str, Any], config_path: Path) -> str:
    """``control_system.type`` as written, or an error naming what is missing."""
    control_system = config.get("control_system")
    if not isinstance(control_system, dict):
        raise RecorderConfigError(
            f"{config_path} has no `control_system:` block; cannot tell what is being recorded"
        )
    return str(control_system.get("type", "")).strip()


def _recorded_target_is_standin(config: dict[str, Any]) -> bool:
    """Whether the machine this deployment records is its own stand-in.

    **The archive belongs to the machine it records, and a model has no past.**
    Nothing here asks whether ``live`` is secretly something else — the stand-in
    is its own ``standin`` target, and an operator on it is told ``standin``.
    The question is whose history the store this recorder writes into already
    holds, and that is
    :func:`~osprey_connectors.standin.archive_belongs_to_standin`: a deployment
    that stood a stand-in up *and* runs this recorder records the stand-in,
    because the machine whose present is sampled and the machine whose past was
    seeded are then the same one. Neither conjunct is restated here — the
    recorder's compose entry and the deploy-time seed bind to that same
    predicate, and three readings of one question are three chances to disagree
    about a single collection.

    The predicate decides *whether*; the derivation decides *where*, and it
    reads the ``standin`` target — its own ``control_system.connector.live_standin``
    block — never the facility's authored ``epics`` block, which always means
    the real machine. The endpoint is derived by
    :func:`~osprey.mcp_server.control_system.target_eligibility.derive_endpoints`
    and judged by
    :func:`~osprey.mcp_server.control_system.target_eligibility.endpoint_is_live_standin`,
    the same step the roster's label is minted through, so a recorder cannot
    believe it is sampling a stand-in while an operator is told otherwise.

    A ``standin`` block whose gateways have been moved off this host is a real
    machine, whatever the deployment once stood up, and recording it into a
    synthesized past is the one thing an archive must never hold — so that
    answers ``False``, the direction every honesty predicate in this stack
    fails: an endpoint is a real machine until the config proves otherwise.
    """
    if not archive_belongs_to_standin(config):
        return False
    try:
        derivation = derive_endpoints(config, TARGET_STANDIN)
    except ValueError:
        return False
    return endpoint_is_live_standin(config, derivation.selected_endpoint())


def _load_mapping(config_path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise RecorderConfigError(f"cannot read {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise RecorderConfigError(f"{config_path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise RecorderConfigError(f"{config_path} does not contain a YAML mapping")
    return raw


def _subtree(config: dict[str, Any], prefix: tuple[str, ...], config_path: Path) -> dict[str, Any]:
    node: Any = config
    for key in prefix:
        node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            raise RecorderConfigError(
                f"{config_path} has no `{'.'.join(prefix)}:` block. The recorder reads its "
                f"store and its cadences from the profile's `va_archiver:` block; a project "
                f"without one has no archive to record into."
            )
    if not isinstance(node, dict):
        raise RecorderConfigError(f"{config_path}: `{'.'.join(prefix)}` is not a mapping")
    return node


def _login_block(
    connection: dict[str, Any], config_path: Path
) -> tuple[dict[str, Any], str | None]:
    """The block's ``auth:`` mapping and ``tls.ca_bundle``, checked by the shared reader.

    Each of ``source``, ``username`` and ``password_env`` stays required and is
    checked by the caller, so a missing one is named before the reader's own
    rules apply.
    """
    raw = connection.get("auth")
    auth = raw if isinstance(raw, dict) else {}
    for leaf in ("source", "username", "password_env"):
        _require(auth, leaf, _AUTH_PREFIX, config_path)
    try:
        settings = read_connection_settings(
            connection,
            where=".".join(_CONNECTION_PREFIX),
            logins=frozenset({"password"}),
            unsupported_because="the archive store takes a username and auth.password_env",
            extra_auth_keys=frozenset({"source"}),
        )
    except ValueError as exc:
        raise RecorderConfigError(f"{config_path}: {exc}") from exc
    ca_bundle = str(settings.ca_bundle) if settings.ca_bundle is not None else None
    return auth, ca_bundle


def _require(block: dict[str, Any], key: str, prefix: tuple[str, ...], config_path: Path) -> Any:
    value = block.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise RecorderConfigError(
            f"{config_path}: `{'.'.join((*prefix, key))}` is missing or empty"
        )
    return value


def _int_key(block: dict[str, Any], key: str, prefix: tuple[str, ...], config_path: Path) -> int:
    return _as_int(_require(block, key, prefix, config_path), key, prefix, config_path)


def _as_int(value: Any, key: str, prefix: tuple[str, ...], config_path: Path) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RecorderConfigError(
            f"{config_path}: `{'.'.join((*prefix, key))}` must be an integer, got {value!r}"
        ) from exc
    if parsed <= 0:
        raise RecorderConfigError(
            f"{config_path}: `{'.'.join((*prefix, key))}` must be positive, got {parsed}"
        )
    return parsed
