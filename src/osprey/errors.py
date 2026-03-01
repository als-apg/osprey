"""Error classification and handling for the Osprey Framework."""

from typing import Any


class ChannelLimitsViolationError(Exception):
    """Raised when a channel write violates configured safety limits.

    Covers min/max range violations, read-only channel writes,
    excessive step sizes, and writes to unlisted channels.
    """

    def __init__(
        self,
        channel_address: str,
        value: Any,
        violation_type: str,
        violation_reason: str,
        min_value: float | None = None,
        max_value: float | None = None,
        max_step: float | None = None,
        current_value: Any | None = None,
    ):
        self.channel_address = channel_address
        self.attempted_value = value
        self.violation_type = violation_type
        self.violation_reason = violation_reason
        self.min_value = min_value
        self.max_value = max_value
        self.max_step = max_step
        self.current_value = current_value

        message = self._format_violation_message()

        super().__init__(message)

    def _format_violation_message(self) -> str:
        """Format a user-friendly violation message with all relevant details."""
        msg = [
            "\n" + "=" * 70,
            "CHANNEL LIMITS VIOLATION DETECTED",
            "=" * 70,
            f"Channel Address: {self.channel_address}",
            f"Attempted Value: {self.attempted_value}",
        ]

        if self.current_value is not None:
            msg.append(f"Current Value: {self.current_value}")

        msg.append(f"Violation: {self.violation_reason}")

        if self.min_value is not None or self.max_value is not None:
            msg.append(f"Allowed Range: [{self.min_value}, {self.max_value}]")

        if self.max_step is not None:
            msg.append(f"Maximum Step Size: {self.max_step}")

        msg.extend(
            [
                "=" * 70,
                "⚠️  Write operation BLOCKED for safety",
                "=" * 70,
            ]
        )

        return "\n".join(msg)


class RegistryError(Exception):
    """Exception for registry-related errors.

    Raised when issues occur with component registration, lookup, or
    management within the framework's registry system.
    """

    pass


class ConfigurationError(Exception):
    """Exception for configuration-related errors.

    Raised when configuration files are invalid, missing required settings,
    or contain incompatible values that prevent proper system operation.
    """

    pass
