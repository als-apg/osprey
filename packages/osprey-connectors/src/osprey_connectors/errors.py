"""Error classification and handling for the Osprey Framework.

Stability: the exception class names and ``reason`` codes defined here are
public API. Removing or renaming any of them is a major version bump; adding
new ones is a minor version bump.
"""

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


class ChannelWriteBlockedError(Exception):
    """Raised when a channel write was refused and no value was written.

    A refusal always means the channel was left alone, but not always that the
    control system went unasked. Most refusals come from the reference monitor
    before anything is sent — writes are disabled, a limits check failed, or
    validation raised. "CONTROL_SYSTEM_REFUSED" is the exception: the control
    system itself was asked and denied the write (EPICS access security, for
    instance). Distinct from ChannelWriteFailedError, which means the write was
    attempted and its outcome is a failure rather than a denial.

    reason is one of: "WRITES_DISABLED", "LIMITS", "VALIDATION_ERROR",
    "CONTROL_SYSTEM_REFUSED".
    """

    _VALID_REASONS = (
        "WRITES_DISABLED",
        "LIMITS",
        "VALIDATION_ERROR",
        "CONTROL_SYSTEM_REFUSED",
    )

    def __init__(self, channel_address: str, reason: str, message: str | None = None):
        self.channel_address = channel_address
        self.reason = reason
        # The fallback names the refuser the reason implies. Every production
        # path passes an explicit message, but a bare construction must not be
        # able to reintroduce the misattribution this class exists to avoid.
        refuser = (
            "the control system" if reason == "CONTROL_SYSTEM_REFUSED" else "reference monitor"
        )
        text = message or f"Write to '{channel_address}' refused by {refuser} ({reason})"
        super().__init__(text)


class ChannelWriteFailedError(Exception):
    """Raised when a channel write was attempted but did not verifiably succeed.

    The control system was asked to write but the write failed or its readback
    did not verify. Distinct from ChannelWriteBlockedError (a refusal — no value
    was written). A scan consumer must abort on this.

    reason is one of: "WRITE_FAILED", "READBACK_UNVERIFIED". Both are
    protocol-neutral on purpose — the same codes describe an EPICS, DOOCS, or
    simulated write.
    """

    _VALID_REASONS = ("WRITE_FAILED", "READBACK_UNVERIFIED")

    def __init__(self, channel_address: str, reason: str, message: str | None = None):
        self.channel_address = channel_address
        self.reason = reason
        text = message or f"Write to '{channel_address}' failed ({reason})"
        super().__init__(text)


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


class BuildProfileError(ConfigurationError):
    """Raised when a build profile is invalid or missing required files."""

    pass
