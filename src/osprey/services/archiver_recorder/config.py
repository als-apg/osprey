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
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from osprey.connectors.archiver.mongodb_archiver_connector import address_overrides

#: Where the compose template mounts the project's ``data/simulation`` tree,
#: matching the virtual accelerator's own mount so a relative
#: ``VA_CHANNELS_FILE`` resolves to the same file on both sides.
DEFAULT_DATA_DIR = "/data/simulation"

#: Rendered-config subtree holding the connection keys (mirrors
#: ``build_profile_archiver.CONNECTION_CONFIG_PREFIX``).
_CONNECTION_PREFIX = ("archiver", "mongodb_archiver")

#: Rendered-config subtree holding the archive-shape knobs (mirrors
#: ``build_profile_archiver.KNOBS_CONFIG_PREFIX``).
_KNOBS_PREFIX = ("va_archiver",)


class RecorderConfigError(RuntimeError):
    """The deployment's config cannot back a recorder.

    Raised only from startup paths. Once the recorder is running, a config that
    goes unreadable is a torn read to be waited out, not a reason to stop
    recording — see :func:`read_control_system_type`.
    """


@dataclass(frozen=True)
class RecorderSettings:
    """Everything the recorder needs, resolved once at startup.

    Deliberately not re-read on the poll interval: a cadence or retention
    change alters the shape of the archive, so it goes through the seeder's
    fingerprint check and a redeploy, which restarts this service anyway. The
    one setting that IS re-read is ``control_system.type``, which changes only
    who is being recorded, not what the record looks like.
    """

    host: str
    port: int
    database: str
    collection: str
    auth_database: str
    username: str
    password_env: str
    timeout_sec: int
    cadence_sec: int
    tail_cadence_sec: int
    poll_sec: int
    hot_span_hours: int
    retention_days: int


def load_settings(config_path: Path) -> RecorderSettings:
    """Resolve :class:`RecorderSettings` from a rendered ``config.yml``.

    Raises:
        RecorderConfigError: if the file is unreadable, is not a mapping, or is
            missing a key the recorder has no honest default for. The message
            names the key and the block it belongs to, because the fix is
            always a profile edit and a rebuild.
    """
    config = _load_mapping(config_path)
    connection = _subtree(config, _CONNECTION_PREFIX, config_path)
    knobs = _subtree(config, _KNOBS_PREFIX, config_path)

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
        auth_database=str(_require(connection, "auth", _CONNECTION_PREFIX, config_path)),
        username=str(_require(connection, "username", _CONNECTION_PREFIX, config_path)),
        password_env=str(_require(connection, "password_env", _CONNECTION_PREFIX, config_path)),
        timeout_sec=_int_key(connection, "timeout", _CONNECTION_PREFIX, config_path),
        cadence_sec=_int_key(knobs, "recorder_cadence_sec", _KNOBS_PREFIX, config_path),
        tail_cadence_sec=_int_key(knobs, "recorder_tail_cadence_sec", _KNOBS_PREFIX, config_path),
        poll_sec=_int_key(knobs, "recorder_poll_sec", _KNOBS_PREFIX, config_path),
        hot_span_hours=_int_key(knobs, "hot_span_hours", _KNOBS_PREFIX, config_path),
        retention_days=_int_key(knobs, "retention_days", _KNOBS_PREFIX, config_path),
    )


def read_control_system_type(config_path: Path) -> str:
    """Read ``control_system.type`` from the mounted config, right now.

    This is the enablement question, asked again on every poll rather than
    answered once at startup, because flipping ``control_system.type`` is a
    documented post-build step and it must not need a redeploy to take effect.

    Raises:
        RecorderConfigError: if the file cannot be read or parsed *at this
            moment*. Callers keep their last known answer instead of acting on
            it: config writes are truncate-in-place, not atomic, so a poll that
            lands mid-write sees a torn file — and treating that as "the
            control system changed" would stop and restart recording on a file
            write that changed nothing.
    """
    config = _load_mapping(config_path)
    control_system = config.get("control_system")
    if not isinstance(control_system, dict):
        raise RecorderConfigError(
            f"{config_path} has no `control_system:` block; cannot tell what is being recorded"
        )
    return str(control_system.get("type", "")).strip()


def resolve_channel_addresses(data_dir: Path | None = None) -> list[str]:
    """The addresses to record, in the order the channel source lists them.

    The source is the build-generated channel MANIFEST, named by
    ``VA_CHANNELS_FILE`` exactly as the virtual accelerator reads it, so the
    recorder covers precisely the channels the IOC serves. Unset or empty means
    the image's built-in manifest, again matching the IOC.

    It must never be ``channel_limits.json``. That file is a *write-safety
    projection* of the same manifest rather than a second copy of it: it holds
    one entry per address, read-only ones included, alongside top-level metadata
    keys (``_comment``, ``_version``, ``_description``, ``defaults``). Its key
    set is therefore not a channel list, and recording from it would ask the
    IOC for names that are not channels at all. The manifest is the one source
    the IOC and this service share; reading anything else makes a second source
    that is free to drift from what is actually being served.

    Raises:
        RecorderConfigError: if the named manifest cannot be loaded. A recorder
            that fell back to the built-in channel set here would record
            another facility's namespace into this facility's archive.
    """
    root = Path(data_dir) if data_dir is not None else Path(DEFAULT_DATA_DIR)
    raw = os.environ.get("VA_CHANNELS_FILE", "").strip()

    # Imported here rather than at module scope: the manifest package pulls in
    # the channel-database parsers, which a config-only caller has no use for.
    from osprey.services.virtual_accelerator.manifest import build_manifest
    from osprey.services.virtual_accelerator.manifest.loaders import (
        ManifestFileError,
        load_manifest_file,
    )

    if not raw:
        channels = build_manifest()["channels"]
    else:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        try:
            channels = load_manifest_file(path)
        except (ManifestFileError, json.JSONDecodeError, OSError) as exc:
            raise RecorderConfigError(
                f"cannot record: the channel manifest named by VA_CHANNELS_FILE ({path}) "
                f"could not be loaded: {exc}"
            ) from exc

    return [str(channel["address"]) for channel in channels]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


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
