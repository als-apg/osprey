"""Runtime channel limits validation engine - simplified single-layer design."""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Reserved metadata fields (underscore prefix)
# These are for documentation only and don't affect validation
METADATA_FIELDS = {'_comment', '_version', '_last_updated', '_description'}

# Special functional field (not metadata)
DEFAULTS_FIELD = 'defaults'


@dataclass
class ChannelLimitsConfig:
    """Configuration for a single channel's limits."""
    channel_address: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    max_step: Optional[float] = None  # Optional: requires channel read (I/O overhead)
    writable: bool = True


class LimitsValidator:
    """Core limits validation engine.

    Performs synchronous validation - no async I/O to keep it simple.
    Raises exceptions directly instead of returning result objects.
    """

    def __init__(self, limits_database: dict[str, ChannelLimitsConfig], policy_config: dict):
        self.limits = limits_database
        self.policy = policy_config
        # Validation behavior: "error" (raise exception) or "skip" (return False, log warning)
        self.on_violation = policy_config.get('on_violation', 'error')

    @classmethod
    def from_config(cls):
        """Load validator from Osprey configuration."""
        from osprey.utils.config import get_config_value

        enabled = get_config_value('control_system.limits_checking.enabled', False)
        if not enabled:
            return None

        db_path = get_config_value('control_system.limits_checking.database_path', None)
        if not db_path:
            logger.warning("Limits checking enabled but no database path configured - blocking all writes")
            return cls({}, {})  # Empty DB = blocks all (failsafe)

        limits_db = cls._load_limits_database(db_path)
        logger.debug(f"Loaded limits database with {len(limits_db)} channels")

        policy = {
            'allow_unlisted_channels': get_config_value(
                'control_system.limits_checking.allow_unlisted_channels', False
            ),
            'on_violation': get_config_value(
                'control_system.limits_checking.on_violation', 'skip'  # Default to skip for resilience
            )
        }

        return cls(limits_db, policy)

    @staticmethod
    def _validate_channel_config(channel_name: str, config_dict: dict) -> None:
        """Validate a single channel configuration structure.

        Args:
            channel_name: Channel address being validated
            config_dict: Configuration dictionary for the channel

        Raises:
            ValueError: If configuration is invalid with descriptive error message
        """
        valid_fields = {'min_value', 'max_value', 'max_step', 'writable', 'verification'}
        unknown_fields = set(config_dict.keys()) - valid_fields

        if unknown_fields:
            logger.warning(
                f"Channel '{channel_name}' has unknown fields: {unknown_fields}. "
                f"Valid fields are: {valid_fields}"
            )

        # Validate numeric fields
        for field in ['min_value', 'max_value', 'max_step']:
            if field in config_dict:
                value = config_dict[field]
                if value is not None and not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Field '{field}' must be numeric, got {type(value).__name__} = {value}"
                    )

        # Validate boolean fields
        if 'writable' in config_dict:
            value = config_dict['writable']
            if not isinstance(value, bool):
                raise ValueError(
                    f"Field 'writable' must be boolean, got {type(value).__name__} = {value}"
                )

        # Validate nested verification config (if present)
        if 'verification' in config_dict:
            verification = config_dict['verification']
            if not isinstance(verification, dict):
                raise ValueError(
                    f"Field 'verification' must be a dictionary, got {type(verification).__name__}"
                )

    @staticmethod
    def _load_limits_database(db_path: str) -> dict[str, ChannelLimitsConfig]:
        """Load and validate limits database from JSON file.

        The database supports:
        - Channel-specific configurations
        - 'defaults' field for common settings (functional, not metadata)
        - Metadata fields with underscore prefix (_comment, _version, etc.)

        Args:
            db_path: Path to JSON database file

        Returns:
            Dictionary mapping channel addresses to validated configurations

        Raises:
            ValueError: If database structure is invalid
        """
        try:
            path = Path(db_path).expanduser()
            if not path.exists():
                logger.error(f"Limits database not found: {db_path}")
                raise ValueError(f"Channel limits database not found: {db_path}")

            with open(path, 'r') as f:
                raw_db = json.load(f)

            if not isinstance(raw_db, dict):
                raise ValueError(
                    f"Limits database must be a JSON object/dict, got {type(raw_db).__name__}"
                )

            # Validate 'defaults' field if present
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
                if channel_name in METADATA_FIELDS or channel_name.startswith('_'):
                    logger.debug(f"Skipping metadata field: {channel_name}")
                    continue

                # Skip the defaults field (handled separately, not a channel)
                if channel_name == DEFAULTS_FIELD:
                    continue

                # Validate it's a dict
                if not isinstance(config_dict, dict):
                    logger.error(
                        f"Invalid config for channel '{channel_name}': "
                        f"must be a dictionary, got {type(config_dict).__name__} - SKIPPING"
                    )
                    continue

                try:
                    # Validate configuration structure
                    LimitsValidator._validate_channel_config(channel_name, config_dict)

                    # Create validated config object
                    config = ChannelLimitsConfig(
                        channel_address=channel_name,
                        min_value=config_dict.get('min_value'),
                        max_value=config_dict.get('max_value'),
                        max_step=config_dict.get('max_step'),
                        writable=config_dict.get('writable', True)
                    )

                    # Log performance warning for max_step
                    if config.max_step is not None:
                        logger.debug(
                            f"Channel '{channel_name}' has max_step={config.max_step} configured "
                            f"(will require channel read, adds ~50-100ms latency)"
                        )

                    limits_db[channel_name] = config

                except (TypeError, ValueError, KeyError) as e:
                    logger.error(f"Invalid config for channel '{channel_name}': {e} - SKIPPING")

            logger.info(f"Successfully loaded {len(limits_db)} channel configurations from {db_path}")
            return limits_db

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
        # Import here to avoid circular dependency
        from osprey.services.python_executor.exceptions import ChannelLimitsViolationError

        # Check 1: Channel exists in database?
        channel_config = self.limits.get(channel_address)

        if channel_config is None:
            # Unlisted channel - check policy
            if self.policy.get('allow_unlisted_channels', False):
                return  # Allow unlisted channel
            else:
                # FAILSAFE: Block unlisted channels
                logger.warning(f"Blocked write to unlisted channel: {channel_address}={value}")
                raise ChannelLimitsViolationError(
                    channel_address=channel_address,
                    value=value,
                    violation_type="UNLISTED_CHANNEL",
                    violation_reason=f"Channel '{channel_address}' not in limits database"
                )

        # Check 2: Channel is writable?
        if not channel_config.writable:
            logger.warning(f"Blocked write to read-only channel: {channel_address}={value}")
            raise ChannelLimitsViolationError(
                channel_address=channel_address,
                value=value,
                violation_type="READ_ONLY_CHANNEL",
                violation_reason="Channel is marked as read-only"
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
                max_value=channel_config.max_value
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
                max_value=channel_config.max_value
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
                        violation_reason="Cannot read current channel value to verify step size"
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
                            max_value=channel_config.max_value
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
                    violation_reason="pyepics not available for step size verification"
                )
            except Exception as e:
                # FAILSAFE: Any error during read → block write
                logger.error(
                    f"Failed to read current value for step check: {channel_address} - {e}"
                )
                raise ChannelLimitsViolationError(
                    channel_address=channel_address,
                    value=value,
                    violation_type="STEP_CHECK_FAILED",
                    violation_reason=f"Channel read failed: {str(e)}"
                )

        # All checks passed!
        logger.debug(f"Validated write: {channel_address}={value}")

