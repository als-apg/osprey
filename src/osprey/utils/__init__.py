"""Configuration Management Package.

This package provides configuration management capabilities
for the Osprey Framework, supporting both LangGraph contexts
and standalone execution.

Modules:
    config: Main configuration builder and access functions
    logger: Logging configuration utilities
    streaming: Streaming configuration utilities
    log_filter: Flexible logging filter utilities
    epics_utils: EPICS utilities for PV operations and caching
    epics_gateway_config: EPICS gateway configuration for different facilities
"""

# Make the main modules available at package level
from . import config, log_filter, logger, streaming

# EPICS utilities are optional - only import if needed
try:
    from . import epics_utils, epics_gateway_config
    __all__ = ['config', 'logger', 'streaming', 'log_filter', 'epics_utils', 'epics_gateway_config']
except ImportError:
    # EPICS utilities not available (missing dependencies)
    __all__ = ['config', 'logger', 'streaming', 'log_filter']
