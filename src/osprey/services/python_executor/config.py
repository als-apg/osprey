"""
Python Executor Configuration Module

This module contains configuration classes for the Python executor service.
Separated from service.py to avoid circular import issues.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PythonExecutorConfig:
    """Configuration for Python Executor Service.

    Manages essential configuration settings for the Python executor service,
    including retry limits and execution timeouts. Values can be overridden
    via framework configuration.
    """

    def __init__(self, configurable: dict[str, Any] = None):
        config = configurable or {}
        executor_config = config.get("python_executor", {})

        # Retry configuration - how many times to retry failed operations
        self.max_generation_retries = executor_config.get("max_generation_retries", 3)
        self.max_execution_retries = executor_config.get("max_execution_retries", 3)

        # Timeout configuration - how long to wait for operations
        self.execution_timeout_seconds = executor_config.get("execution_timeout_seconds", 600)  # 10 minutes

        # Limits validator - lazy-loaded from config
        self._limits_validator = None

    @property
    def limits_validator(self):
        """Get limits validator (lazy-loaded from config).

        Returns the LimitsValidator instance if runtime channel limits checking
        is enabled in the configuration, or None if disabled. The validator is
        loaded only once and cached for subsequent accesses.

        :return: LimitsValidator instance or None if disabled
        :rtype: LimitsValidator | None
        """
        if self._limits_validator is None:
            from osprey.services.python_executor.execution.limits_validator import (
                LimitsValidator
            )
            self._limits_validator = LimitsValidator.from_config()

            if self._limits_validator:
                logger.info("Runtime channel limits checking ENABLED")
            else:
                logger.debug("Runtime channel limits checking DISABLED")

        return self._limits_validator
