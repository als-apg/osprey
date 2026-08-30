"""Runtime channel limits validation engine - simplified single-layer design."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey_connectors.logger import get_logger
from osprey_connectors.types import (
    LimitsPosture,
    most_restrictive_limits_posture,
    target_limits_posture,
    type_limits_posture,
)

logger = get_logger("limits_validator")


# Reserved metadata fields (underscore prefix)
# These are for documentation only and don't affect validation
METADATA_FIELDS = {"_comment", "_version", "_last_updated", "_description"}

# Special functional field (not metadata)
DEFAULTS_FIELD = "defaults"


@dataclass
class ChannelLimitsConfig:
    """Configuration for a single channel's limits."""

    channel_address: str
    min_value: float | None = None
    max_value: float | None = None
    max_step: float | None = None  # Optional: requires channel read (I/O overhead)
    writable: bool = True


class LimitsValidator:
    """Core limits validation engine.

    Performs synchronous validation - no async I/O to keep it simple.
    Raises exceptions directly instead of returning result objects.
    """

    def __init__(
        self,
        limits_database: dict[str, ChannelLimitsConfig],
        policy_config: dict,
        raw_db: dict | None = None,
        failsafe_reason: str | None = None,
    ):
        self.limits = limits_database
        self.policy = policy_config
        self._raw_db = raw_db  # Keep raw database for confirm-policy access
        # Set only by the empty-DB failsafe paths in from_config: the database
        # could not be loaded, so every write is blocked. validate() uses it to
        # refuse with a load-failure message instead of the unlisted-channel
        # one — the two conditions need different operator responses.
        self.failsafe_reason = failsafe_reason

    @staticmethod
    def resolve_database_path(
        db_path: str, project_root: str | None, config_path: str | None = None
    ) -> str:
        """Resolve a relative ``database_path`` the same way for every caller.

        A relative ``database_path`` is written alongside the config it appears
        in, so the anchor is the directory of the config that is actually in
        use: *config_path*, when the caller can name the file it loaded (see
        :func:`osprey_connectors.config.default_config_path`). This is what
        makes the resolution independent of how the process was launched — a
        Claude Code hook, an MCP server, and a host verb anchored via
        ``config_anchored_at`` all load the render's config and find the
        ``data/`` tree the build copied next to it.

        Fallbacks, for callers that cannot name the loaded config:
        ``Path(CONFIG_FILE).parent`` when that variable is set (a container
        deploy names the config mounted inside the container while
        ``project_root`` may record a host path the container does not have),
        else the config's own ``project_root`` key. Note that under the
        four-zone repo layout the render lives at ``<repo>/build/config.yml``
        while ``project_root`` names ``<repo>`` — the project_root branch is a
        last resort, not an equivalent spelling.

        Returns ``db_path`` unchanged if it is already absolute, or if no base
        is available.
        """
        db_path_obj = Path(db_path)
        if db_path_obj.is_absolute():
            return db_path
        if config_path:
            return str(Path(config_path).parent / db_path)
        config_file = os.environ.get("CONFIG_FILE")
        if config_file:
            return str(Path(config_file).parent / db_path)
        if project_root:
            return str(Path(project_root) / db_path)
        return db_path

    @classmethod
    def _load_configured_database(
        cls,
    ) -> tuple[tuple[dict[str, ChannelLimitsConfig], dict] | None, str | None]:
        """Resolve and load the deployment-wide limits database.

        The database path is deployment-wide even where the posture is not: a
        deployment mounts one limits file (``compose_generator.py`` binds a
        single path), so a per-type block changes policy and never the data.
        This is the half of validator construction that is the same for every
        posture, kept in one place so that the deployment-wide and per-type
        entry points cannot drift in how they resolve a relative path or in
        which load failures they treat as fatal.

        Failure is reported rather than raised because both entry points answer
        it the same way — with a fail-safe validator that blocks every write.
        Returning ``None`` instead would leave writes unchecked, and raising
        would crash a connector on a deployment that never writes at all.

        Returns:
            A ``(loaded, failsafe_reason)`` pair with exactly one half set:
            either the ``(limits_database, raw_database)`` tuple, or a reason
            string naming what could not be read, ready to hand to
            ``failsafe_reason``.
        """
        from osprey_connectors.config import default_config_path, get_config_value

        db_path = get_config_value("control_system.limits_checking.database_path", None)
        # Validate db_path is actually a string path (not None, False, or other types)
        if not db_path or not isinstance(db_path, str):
            logger.warning(
                "Limits checking enabled but no database path configured - blocking all writes"
            )
            return None, (
                "limits checking is enabled but "
                "control_system.limits_checking.database_path is not configured"
            )

        project_root = get_config_value("project_root", None)
        resolved_path = cls.resolve_database_path(
            db_path, project_root, config_path=default_config_path()
        )
        if resolved_path != db_path:
            logger.debug(f"Resolved limits database path: {resolved_path}")
        db_path = resolved_path

        try:
            limits_db, raw_db = cls._load_limits_database(db_path)
        except ValueError as e:
            # Same failsafe as the missing-database-path case above: a
            # missing/unparseable database must never crash the connector
            # (a read-only deployment needs no limits) nor silently disable
            # checking (returning None would leave writes unchecked) — an
            # empty DB blocks every write with a clear refusal instead.
            logger.warning(
                f"Limits checking enabled but the database at {db_path} could not "
                f"be read or parsed - blocking all writes. {e}"
            )
            return None, (f"the limits database at {db_path} could not be read or parsed: {e}")

        logger.debug(f"Loaded limits database with {len(limits_db)} channels")
        return (limits_db, raw_db), None

    @classmethod
    def from_config(
        cls,
        *,
        connector_type: str | None = None,
        target: str | None = None,
    ) -> "LimitsValidator | None":
        """Load a validator for the limits posture the caller acts under.

        Three call shapes, one per identity a caller can hold:

        - **Neither argument** asks the deployment-wide question,
          ``control_system.limits_checking``, and never resolves a target. It is
          what a caller holding no machine of its own asks — the connector
          registry's log line, the executor's config helper — and it is the
          shape every caller had before per-type blocks existed, so a deployment
          that wrote none behaves as it did — except that a deployment-wide
          leaf written as something other than a literal boolean now blocks
          every write instead of being read as truthy.
        - **connector_type** is what a connector, factory or IPC child asks:
          they know which connector type they are and never which target
          selected it.
        - **target** is what a tool, hook, roster or the executor asks: they
          follow the session's control target, which is resolved to its
          connector type here so both shapes read one posture.

        A present ``control_system.connector.<type>.limits_checking`` block
        overrides the deployment-wide pair whole; absent, the deployment-wide
        block answers. The key that answered travels into ``policy`` so a later
        refusal names the config line an operator can edit.

        Args:
            connector_type: The connector type to resolve the posture for, as
                the config's ``control_system.connector`` table keys it.
            target: The session control target to resolve the posture for, one
                of :data:`osprey_connectors.types.CONTROL_TARGETS`. A target
                that does not resolve to a type gets the deployment-wide block,
                as the write posture does.

        Returns:
            An enforcing validator; a fail-safe validator that blocks every
            write when the resolved block is incomplete or the limits database
            is unavailable; or ``None`` when limits checking is off for this
            posture or the config could not be read at all.

        Raises:
            TypeError: If both *connector_type* and *target* are given. A target
                already names a type, so stating both states the posture twice
                and leaves it undefined which one the caller meant. Raised
                before any config is read, so it surfaces as the caller bug it
                is rather than as a missing-config ``None``.
        """
        if connector_type is not None and target is not None:
            raise TypeError(
                "LimitsValidator.from_config() takes connector_type or target, not both: "
                "a target resolves to a connector type, so passing both states the "
                "posture twice"
            )
        try:
            from osprey_connectors.config import get_config_value

            section = get_config_value("control_system", {})
        except (FileNotFoundError, KeyError, RuntimeError) as e:
            # Config not available (test environment, etc.) - limits checking disabled
            logger.debug(f"Limits validator not initialized (config unavailable): {e}")
            return None

        if target is not None:
            posture = target_limits_posture(section, target)
        else:
            posture = type_limits_posture(section, connector_type)
        return cls._from_posture(posture)

    @classmethod
    def from_config_most_restrictive(cls) -> "LimitsValidator | None":
        """Load a validator for the posture that holds across every reachable target.

        What a caller with no target of its own has to assume. The stdlib limits
        hook is the one that needs it: when the session's control target cannot
        be read — none written yet, an unreadable state directory, a target that
        resolves to nothing — it still has to decide about a write, and the
        machine it is deciding about is any of the ones a session here can
        select.

        The fold is :func:`osprey_connectors.types.most_restrictive_limits_posture`'s:
        limits checking is on when any reachable target has it on, and unlisted
        channels are allowed only where every reachable target allows them. The
        result names the deployment-wide keys, because no per-type line decides
        a union and naming one would send an operator to edit a single machine
        rather than the answer.

        Returns:
            An enforcing validator, a fail-safe one, or ``None`` — the same
            envelope :meth:`from_config` returns, for the same reasons.
        """
        try:
            from osprey_connectors.config import get_config_value

            section = get_config_value("control_system", {})
        except (FileNotFoundError, KeyError, RuntimeError) as e:
            # Config not available (test environment, etc.) - limits checking disabled
            logger.debug(f"Limits validator not initialized (config unavailable): {e}")
            return None

        return cls._from_posture(most_restrictive_limits_posture(section))

    @classmethod
    def _from_posture(cls, posture: LimitsPosture) -> "LimitsValidator | None":
        """Build a validator for an already-resolved limits posture.

        The caller resolves the posture for the thing it is acting on — a
        connector for its own type, a tool or hook for the session's target —
        and this turns that answer into an enforcing validator. The posture's
        answering key travels into ``policy`` so that a later refusal names the
        config line an operator can edit: on a deployment that relaxed unlisted
        channels for its simulator alone, quoting the deployment-wide key would
        send them to flip a line the per-type block overrides.

        An incomplete block is checked before ``enabled``, and on purpose. Such
        a block answered neither leaf, so ``enabled`` is ``None`` and the
        disabled branch would wave every write through on exactly the config
        nobody has finished writing. Blocking instead is the same choice the
        missing-database-path branch makes.

        A block is incomplete two ways, and the second is why this branch comes
        first. A per-type block may omit a leaf. Either block may write one as
        something no reader can turn into a boolean — a quoted ``'true'``, a
        ``1``, an unexpanded ``'${LIMITS_ON}'``, which is the shape environment
        expansion leaves behind when nothing set the variable. That second one
        is a deployment trying to switch limits checking *on*; reading it as an
        unset ``enabled`` would take the disabled branch and check nothing.

        Args:
            posture: The resolved posture, from the ``types`` resolver family.

        Returns:
            An enforcing validator; a fail-safe validator that blocks every
            write when the block is incomplete or the database is unavailable;
            or ``None`` when limits checking is off for this posture or the
            config could not be read at all.
        """
        try:
            if posture.incomplete:
                reason = (
                    f"{posture.block_key} does not state "
                    f"{', '.join(posture.incomplete)} as true/false"
                )
                logger.warning(f"Incomplete limits block - blocking all writes: {reason}")
                return cls({}, {}, {}, failsafe_reason=reason)

            if posture.enabled is not True:
                return None

            loaded, failsafe_reason = cls._load_configured_database()
            if loaded is None:
                # Empty DB = blocks all (failsafe)
                return cls({}, {}, {}, failsafe_reason=failsafe_reason)
            limits_db, raw_db = loaded

            # Both entries are JSON-serialisable: the python executor embeds
            # this dict verbatim into the sandbox (wrapper.py), so the
            # tri-state rides as null rather than as anything richer.
            policy = {
                "allow_unlisted_channels": posture.allow_unlisted,
                "allow_unlisted_key": posture.key("allow_unlisted_channels"),
            }

            return cls(limits_db, policy, raw_db)
        except (FileNotFoundError, KeyError, RuntimeError) as e:
            # Config not available (test environment, etc.) - limits checking disabled
            logger.debug(f"Limits validator not initialized (config unavailable): {e}")
            return None

    def get_limits_config(self, channel_address: str) -> dict | None:
        """Get raw limits configuration for a channel (with defaults merged).

        Returns the channel's configuration dictionary with defaults applied,
        or None if channel is not in database and unlisted channels are allowed.

        Args:
            channel_address: Channel address to look up

        Returns:
            Configuration dictionary with defaults merged, or None if not found
        """
        channel_config = self.limits.get(channel_address)
        if channel_config is None:
            return None

        # Convert ChannelLimitsConfig dataclass to dict for compatibility
        return {
            "channel_address": channel_config.channel_address,
            "min_value": channel_config.min_value,
            "max_value": channel_config.max_value,
            "max_step": channel_config.max_step,
            "writable": channel_config.writable,
        }

    def resolve_confirm(self, channel_address: str) -> bool:
        """Whether a write to this channel must be confirmed by re-reading it.

        Resolution: the channel's own ``confirm`` → the ``defaults`` block's
        ``confirm`` → ``True``. Read off the raw database, which is where
        ``confirm`` lives: it is write policy, not a limit, so it never enters
        :class:`ChannelLimitsConfig`.

        Args:
            channel_address: Channel address being written

        Returns:
            True when the write must be confirmed (the fleet default).
        """
        raw_db = getattr(self, "_raw_db", None)
        if not isinstance(raw_db, dict):
            return True

        channel_config = raw_db.get(channel_address)
        if isinstance(channel_config, dict) and "confirm" in channel_config:
            return bool(channel_config["confirm"])

        defaults_config = raw_db.get(DEFAULTS_FIELD)
        if isinstance(defaults_config, dict) and "confirm" in defaults_config:
            return bool(defaults_config["confirm"])

        return True

    @staticmethod
    def _validate_channel_config(channel_name: str, config_dict: dict) -> None:
        """Validate a single channel configuration structure.

        Args:
            channel_name: Channel address being validated
            config_dict: Configuration dictionary for the channel

        Raises:
            ValueError: If configuration is invalid with descriptive error message
        """
        valid_fields = {"min_value", "max_value", "max_step", "writable", "confirm"}
        # Any '_'-prefixed key is documentation metadata and stays legal, whatever
        # it is called. Every other unrecognised key fails the load: a limits file
        # is a safety artifact, and a key the loader does not understand is a key
        # whose intent was never applied.
        unknown_fields = {k for k in config_dict if not k.startswith("_")} - valid_fields

        if unknown_fields:
            detail = (
                f"Channel '{channel_name}' has unknown fields: {sorted(unknown_fields)}. "
                f"Valid fields are: {sorted(valid_fields)}"
            )
            if "verification" in unknown_fields:
                raise ValueError(
                    f"Channel '{channel_name}': 'verification' was replaced by "
                    f"'confirm: true|false'. {detail}"
                )
            raise ValueError(detail)

        # Validate numeric fields
        for field in ["min_value", "max_value", "max_step"]:
            if field in config_dict:
                value = config_dict[field]
                if value is not None and not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Field '{field}' must be numeric, got {type(value).__name__} = {value}"
                    )

        # Validate boolean fields
        for field in ["writable", "confirm"]:
            if field in config_dict:
                value = config_dict[field]
                if not isinstance(value, bool):
                    raise ValueError(
                        f"Field '{field}' must be boolean, got {type(value).__name__} = {value}"
                    )

    @staticmethod
    def _load_limits_database(db_path: str) -> tuple[dict[str, ChannelLimitsConfig], dict]:
        """Load and validate limits database from JSON file.

        The database supports:
        - Channel-specific configurations
        - 'defaults' field for common settings (functional, not metadata)
        - Metadata fields with underscore prefix (_comment, _version, etc.)
        - Per-channel confirm policy (stored in raw DB)

        Args:
            db_path: Path to JSON database file

        Returns:
            Tuple of (parsed_limits_db, raw_db):
            - parsed_limits_db: Dict mapping channel addresses to ChannelLimitsConfig
            - raw_db: Raw dict from JSON file (for confirm-policy access)

        Raises:
            ValueError: If database structure is invalid
        """
        try:
            path = Path(db_path).expanduser()
            if not path.exists():
                logger.error(f"Limits database not found: {db_path}")
                raise ValueError(f"Channel limits database not found: {db_path}")

            with open(path) as f:
                raw_db = json.load(f)

            if not isinstance(raw_db, dict):
                raise ValueError(
                    f"Limits database must be a JSON object/dict, got {type(raw_db).__name__}"
                )

            # Validate 'defaults' field if present
            defaults_config: dict = {}
            if DEFAULTS_FIELD in raw_db:
                defaults_config = raw_db[DEFAULTS_FIELD]
                if not isinstance(defaults_config, dict):
                    raise ValueError(
                        f"'{DEFAULTS_FIELD}' field must be a dictionary, "
                        f"got {type(defaults_config).__name__}"
                    )
                try:
                    LimitsValidator._validate_channel_config(DEFAULTS_FIELD, defaults_config)
                    logger.debug(f"Loaded defaults configuration: {list(defaults_config.keys())}")
                except ValueError as e:
                    raise ValueError(f"Invalid '{DEFAULTS_FIELD}' configuration: {e}") from e

            # Load channel configurations
            limits_db = {}
            for channel_name, config_dict in raw_db.items():
                # Skip metadata fields (underscore prefix)
                if channel_name in METADATA_FIELDS or channel_name.startswith("_"):
                    logger.debug(f"Skipping metadata field: {channel_name}")
                    continue

                # Skip the defaults field (handled separately, not a channel)
                if channel_name == DEFAULTS_FIELD:
                    continue

                # Validate it's a dict
                if not isinstance(config_dict, dict):
                    raise ValueError(
                        f"Invalid config for channel '{channel_name}': "
                        f"must be a dictionary, got {type(config_dict).__name__}"
                    )

                try:
                    # Validate configuration structure
                    LimitsValidator._validate_channel_config(channel_name, config_dict)

                    # Merge the 'defaults' block under the channel's own config so
                    # the channel inherits any default field it does not override.
                    # Shallow merge: the channel's own keys take precedence, and a
                    # channel that declares 'confirm' overrides the default.
                    merged = {**defaults_config, **config_dict}

                    # Create validated config object
                    config = ChannelLimitsConfig(
                        channel_address=channel_name,
                        min_value=merged.get("min_value"),
                        max_value=merged.get("max_value"),
                        max_step=merged.get("max_step"),
                        writable=merged.get("writable", True),
                    )

                    # Log performance warning for max_step
                    if config.max_step is not None:
                        logger.debug(
                            f"Channel '{channel_name}' has max_step={config.max_step} configured "
                            f"(will require channel read, adds ~50-100ms latency)"
                        )

                    limits_db[channel_name] = config

                except (TypeError, ValueError, KeyError) as e:
                    # One malformed entry fails the whole load. Skipping it used
                    # to drop the channel from the database, which - with
                    # allow_unlisted_channels - silently removed its limits.
                    raise ValueError(f"Invalid config for channel '{channel_name}': {e}") from e

            logger.info(
                f"Successfully loaded {len(limits_db)} channel configurations from {db_path}"
            )
            return limits_db, raw_db

        except json.JSONDecodeError as e:
            logger.error("=" * 80)
            logger.error("CRITICAL: CHANNEL LIMITS DATABASE HAS INVALID JSON")
            logger.error(f"   File: {db_path}")
            logger.error(f"   Error: {e}")
            logger.error("   Impact: ALL channel writes will be BLOCKED (fail-safe mode)")
            logger.error("   Fix: Correct the JSON syntax in your limits database file")
            logger.error("=" * 80)
            raise ValueError(f"Invalid JSON in channel limits database: {e}") from e
        except Exception as e:
            logger.error(f"Failed to load limits database: {e}")
            raise ValueError(f"Failed to load channel limits database: {e}") from e

    def validate(self, channel_address: str, value: Any) -> None:
        """Validate a channel write operation (synchronous, optional I/O for max_step).

        Raises ChannelLimitsViolationError if validation fails.
        Returns None if validation passes.

        Note: If max_step is configured for the channel, this performs one synchronous
        read to get the current value. This adds ~50-100ms latency but
        provides important step-size safety checking.

        Args:
            channel_address: Channel address to validate
            value: Value to write

        Raises:
            ChannelLimitsViolationError: If any validation check fails
        """
        from osprey_connectors.errors import ChannelLimitsViolationError

        # Check 1: Channel exists in database?
        channel_config = self.limits.get(channel_address)

        if channel_config is None:
            # A database that failed to load blocks everything — say so, rather
            # than refusing with the unlisted-channel message: "not in limits
            # database" sends the operator chasing a data problem when the
            # actual failure is that no database was loaded at all.
            if self.failsafe_reason:
                logger.warning(
                    f"Blocked write (limits database unavailable): {channel_address}={value}"
                )
                raise ChannelLimitsViolationError(
                    channel_address=channel_address,
                    value=value,
                    violation_type="LIMITS_DATABASE_UNAVAILABLE",
                    violation_reason=(
                        f"Limits database unavailable — all writes are blocked as a "
                        f"failsafe ({self.failsafe_reason}). This is not a statement "
                        f"about channel '{channel_address}'."
                    ),
                )
            # Unlisted channel - check policy. Only an explicit `True` is
            # permission: the policy carries the posture's tri-state verbatim
            # so that `channel_limits` can report an unstated answer as `null`,
            # and unstated is nobody's permission to write an unlisted channel.
            if self.policy.get("allow_unlisted_channels") is True:
                return  # Allow unlisted channel
            else:
                # FAILSAFE: Block unlisted channels. Name the key that actually
                # answered — a deployment may set this per connector type, and
                # quoting the deployment-wide key there would send an operator
                # to flip a line the per-type block overrides. A validator built
                # from a bare policy dict carries no key; the deployment-wide
                # one is the honest answer for it.
                answering_key = self.policy.get(
                    "allow_unlisted_key", "control_system.limits_checking.allow_unlisted_channels"
                )
                logger.warning(f"Blocked write to unlisted channel: {channel_address}={value}")
                raise ChannelLimitsViolationError(
                    channel_address=channel_address,
                    value=value,
                    violation_type="UNLISTED_CHANNEL",
                    violation_reason=(
                        f"Channel '{channel_address}' not in limits database "
                        f"('{answering_key}' does not allow unlisted channels)"
                    ),
                )

        # Check 2: Channel is writable?
        if not channel_config.writable:
            logger.warning(f"Blocked write to read-only channel: {channel_address}={value}")
            raise ChannelLimitsViolationError(
                channel_address=channel_address,
                value=value,
                violation_type="READ_ONLY_CHANNEL",
                violation_reason="Channel is marked as read-only",
            )

        # Check 3: Min/Max bounds (numeric values only)
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            # Non-numeric value - skip numeric checks
            return

        if channel_config.min_value is not None and numeric_value < channel_config.min_value:
            logger.warning(
                f"Blocked write below minimum: {channel_address}={numeric_value} "
                f"(min={channel_config.min_value})"
            )
            raise ChannelLimitsViolationError(
                channel_address=channel_address,
                value=value,
                violation_type="MIN_EXCEEDED",
                violation_reason=f"Value {numeric_value} below minimum {channel_config.min_value}",
                min_value=channel_config.min_value,
                max_value=channel_config.max_value,
            )

        if channel_config.max_value is not None and numeric_value > channel_config.max_value:
            logger.warning(
                f"Blocked write above maximum: {channel_address}={numeric_value} "
                f"(max={channel_config.max_value})"
            )
            raise ChannelLimitsViolationError(
                channel_address=channel_address,
                value=value,
                violation_type="MAX_EXCEEDED",
                violation_reason=f"Value {numeric_value} above maximum {channel_config.max_value}",
                min_value=channel_config.min_value,
                max_value=channel_config.max_value,
            )

        # Check 4: Step size limit (OPTIONAL - only if configured, requires I/O)
        if channel_config.max_step is not None:
            try:
                import epics

                # Read current value (I/O operation)
                logger.debug(f"Reading current value for step check: {channel_address}")
                current_value = epics.caget(channel_address, timeout=2.0)

                if current_value is None:
                    # FAILSAFE: Can't read current value → block write
                    logger.warning(
                        f"Cannot verify step size for {channel_address} - "
                        f"channel read returned None"
                    )
                    raise ChannelLimitsViolationError(
                        channel_address=channel_address,
                        value=value,
                        violation_type="STEP_CHECK_FAILED",
                        violation_reason="Cannot read current channel value to verify step size",
                    )

                # Check step size (numeric values only)
                try:
                    numeric_current = float(current_value)
                    step_size = abs(numeric_value - numeric_current)

                    if step_size > channel_config.max_step:
                        logger.warning(
                            f"Blocked write exceeding max step: {channel_address} "
                            f"step={step_size:.3f} > max={channel_config.max_step}"
                        )
                        raise ChannelLimitsViolationError(
                            channel_address=channel_address,
                            value=value,
                            violation_type="MAX_STEP_EXCEEDED",
                            violation_reason=(
                                f"Step size {step_size:.3f} exceeds maximum "
                                f"{channel_config.max_step} (current={numeric_current}, "
                                f"requested={numeric_value})"
                            ),
                            current_value=current_value,
                            max_step=channel_config.max_step,
                            min_value=channel_config.min_value,
                            max_value=channel_config.max_value,
                        )

                except (ValueError, TypeError):
                    # Non-numeric current value - skip step check
                    logger.debug(
                        f"Skipping step check for non-numeric values: "
                        f"{channel_address} current={current_value}, new={value}"
                    )

            except ChannelLimitsViolationError:
                # Re-raise limits violations (from max_step check)
                raise
            except ImportError:
                # FAILSAFE: Can't import epics → block write if step checking required
                logger.error(
                    f"Cannot verify step size for {channel_address} - pyepics not available"
                )
                raise ChannelLimitsViolationError(
                    channel_address=channel_address,
                    value=value,
                    violation_type="STEP_CHECK_FAILED",
                    violation_reason="pyepics not available for step size verification",
                ) from None
            except Exception as e:
                # FAILSAFE: Any error during read → block write
                logger.error(
                    f"Failed to read current value for step check: {channel_address} - {e}"
                )
                raise ChannelLimitsViolationError(
                    channel_address=channel_address,
                    value=value,
                    violation_type="STEP_CHECK_FAILED",
                    violation_reason=f"Channel read failed: {str(e)}",
                ) from e

        # All checks passed!
        logger.debug(f"Validated write: {channel_address}={value}")
