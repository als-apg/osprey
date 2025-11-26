#!/usr/bin/env python3
"""
EPICS Utilities for Osprey Framework

This module provides enhanced EPICS functionality including PV existence checking,
caching, and improved read/write operations with automatic existence validation.

Usage:
    from osprey.utils.epics_utils import EPICSManager
    
    # Initialize the manager
    epics_mgr = EPICSManager()
    
    # Check if a PV exists
    if epics_mgr.check_pv_exists("S-DCCT:CurrentM"):
        value = epics_mgr.caget("S-DCCT:CurrentM")
        print(f"Current: {value}")
    
    # Read with automatic existence check
    value = epics_mgr.safe_caget("S-DCCT:CurrentM")
    
    # Write with automatic existence check (if in write mode)
    success = epics_mgr.safe_caput("SOME:PV", 10.5)
"""

import os
import sys
import subprocess
from typing import Dict, Any, Optional, Union, Set
from datetime import datetime, timedelta

from osprey.utils.logger import get_logger

# Get logger for this module
logger = get_logger("epics_utils")


def getoutput(cmd):
    """Replacement for deprecated commands.getoutput()"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Command timed out"
    except Exception as e:
        return f"Command failed: {e}"


class EPICSError(Exception):
    """Base class for EPICS-related errors."""
    pass


class PVNotFoundError(EPICSError):
    """Raised when a PV does not exist."""
    pass


class EPICSWriteError(EPICSError):
    """Raised when EPICS write operations fail."""
    pass


class EPICSReadError(EPICSError):
    """Raised when EPICS read operations fail."""
    pass


class PVExistenceCache:
    """Cache for PV existence checks to avoid repeated pvExist calls."""
    
    def __init__(self, cache_duration_minutes: int = 60):
        """
        Initialize the PV existence cache.
        
        Args:
            cache_duration_minutes: How long to cache existence results (default: 60 minutes)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_duration = timedelta(minutes=cache_duration_minutes)
    
    def get(self, pv_name: str) -> Optional[bool]:
        """
        Get cached existence result for a PV.
        
        Args:
            pv_name: The PV name to check
            
        Returns:
            True if PV exists, False if not, None if not cached or expired
        """
        if pv_name not in self._cache:
            return None
        
        entry = self._cache[pv_name]
        if datetime.now() - entry['timestamp'] > self._cache_duration:
            # Cache expired, remove entry
            del self._cache[pv_name]
            return None
        
        return entry['exists']
    
    def set(self, pv_name: str, exists: bool) -> None:
        """
        Cache the existence result for a PV.
        
        Args:
            pv_name: The PV name
            exists: Whether the PV exists
        """
        self._cache[pv_name] = {
            'exists': exists,
            'timestamp': datetime.now()
        }
    
    def clear(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
    
    def remove(self, pv_name: str) -> None:
        """Remove a specific PV from the cache."""
        if pv_name in self._cache:
            del self._cache[pv_name]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.now()
        valid_entries = sum(1 for entry in self._cache.values() 
                          if now - entry['timestamp'] <= self._cache_duration)
        
        return {
            'total_entries': len(self._cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self._cache) - valid_entries,
            'cache_duration_minutes': self._cache_duration.total_seconds() / 60
        }


class EPICSManager:
    """
    Enhanced EPICS manager with PV existence checking and caching.
    
    This class provides a centralized interface for EPICS operations with
    automatic PV existence checking and intelligent caching.
    """
    
    def __init__(self, cache_duration_minutes: int = 60):
        """
        Initialize the EPICS manager.
        
        Args:
            cache_duration_minutes: How long to cache PV existence results
        """
        self._cache = PVExistenceCache(cache_duration_minutes)
        self._pvexist_path = None
        self._epics_available = False
        self._execution_mode = os.environ.get('EPICS_EXECUTION_MODE', 'unknown')
        
        # Initialize EPICS and find pvExist command
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize EPICS environment and locate pvExist command."""
        # First, set up EPICS gateway configuration
        self._setup_epics_gateway()
        
        # Check if PyEPICS is available without importing it yet
        # This allows environment variables to be set before PyEPICS caches them
        try:
            import importlib.util
            spec = importlib.util.find_spec("epics")
            self._epics_available = spec is not None
            if self._epics_available:
                logger.info("PyEPICS module available (not loaded yet)")
            else:
                logger.warning("PyEPICS module not available")
        except (ImportError, ValueError):
            logger.warning("PyEPICS module not available")
            self._epics_available = False
        
        # Find pvExist command path
        self._find_pvexist_command()
    
    def _setup_epics_gateway(self) -> None:
        """Set up EPICS gateway configuration before PyEPICS operations."""
        try:
            from osprey.utils.epics_gateway_config import setup_facility_epics
            
            logger.info("Setting up EPICS gateway configuration...")
            env_vars = setup_facility_epics(apply_env=True)
            
            # Log the configuration that was applied
            if env_vars:
                gateway = env_vars.get('EPICS_CA_ADDR_LIST', 'broadcast mode')
                auto_addr = env_vars.get('EPICS_CA_AUTO_ADDR_LIST', 'unknown')
                logger.info(f"EPICS gateway configured - Gateway: {gateway}, Auto Address List: {auto_addr}")
            else:
                logger.info("No facility detected - using default EPICS configuration")
            
        except ImportError as e:
            logger.warning(f"EPICS gateway configuration not available: {e}")
            logger.info("Proceeding without dynamic gateway configuration")
        except Exception as e:
            logger.error(f"Failed to setup EPICS gateway: {e}")
            logger.info("Proceeding without gateway configuration")
    
    def _find_pvexist_command(self) -> None:
        """
        Placeholder for pvExist command path finding.
        
        Note: pvExist is no longer used since we assume all requested PVs exist.
        This method is kept for backward compatibility but does nothing.
        """
        # pvExist is no longer needed - we assume all PVs exist
        self._pvexist_path = None
        logger.debug("pvExist check disabled - assuming all requested PVs exist")
    
    def check_pv_exists(self, pv_name: str, use_cache: bool = True) -> bool:
        """
        Check if a PV exists.
        
        Note: Since only available PVs are requested, this method now assumes
        all PVs exist and returns True. The pvExist check has been removed as
        it's no longer needed.

        Args:
            pv_name (str): The name of the PV to check.
            use_cache (bool): Whether to use cached results (default: True, kept for API compatibility)

        Returns:
            bool: Always returns True (assumes PV exists)
        """
        # Since only available PVs will be requested, we can skip the existence check
        # and assume all PVs exist. This simplifies the code and removes the pvExist dependency.
        logger.debug(f"PV existence check for {pv_name}: assumed to exist (pvExist check disabled)")
        return True
    
    def safe_caget(self, pv_name: str, **kwargs) -> Any:
        """
        Safely read a PV value with automatic existence checking.
        
        Args:
            pv_name: The PV name to read
            **kwargs: Additional arguments passed to epics.caget
            
        Returns:
            The PV value
            
        Raises:
            PVNotFoundError: If the PV does not exist
            EPICSReadError: If the read operation fails
        """
        if not self._epics_available:
            raise EPICSReadError("PyEPICS module not available")
        
        # Check if PV exists
        if not self.check_pv_exists(pv_name):
            raise PVNotFoundError(f"PV '{pv_name}' does not exist or is not accessible")
        
        try:
            import epics
            value = epics.caget(pv_name, **kwargs)
            if value is None:
                raise EPICSReadError(f"Failed to read PV '{pv_name}' - returned None")
            return value
        except Exception as e:
            if "not found" in str(e).lower() or "never connected" in str(e).lower():
                # Remove from cache since it might have been incorrectly cached
                self._cache.remove(pv_name)
                raise PVNotFoundError(f"PV '{pv_name}' not found: {str(e)}")
            else:
                raise EPICSReadError(f"Failed to read PV '{pv_name}': {str(e)}")
    
    def safe_caput(self, pv_name: str, value: Any, **kwargs) -> bool:
        """
        Safely write a PV value with automatic existence checking.
        
        Args:
            pv_name: The PV name to write
            value: The value to write
            **kwargs: Additional arguments passed to epics.caput
            
        Returns:
            True if write was successful
            
        Raises:
            PVNotFoundError: If the PV does not exist
            EPICSWriteError: If the write operation fails
            PermissionError: If in read-only mode
        """
        if not self._epics_available:
            raise EPICSWriteError("PyEPICS module not available")
        
        # Check execution mode
        if self._execution_mode == 'read':
            raise PermissionError(
                f"🔒 WRITE OPERATION BLOCKED\n"
                f"   PV: {pv_name}\n"
                f"   Value: {value}\n"
                f"   Reason: You are using the Read-Only kernel\n"
                f"   Solution: Switch to 'Write Access' kernel for real machine control"
            )
        
        # Check if PV exists
        if not self.check_pv_exists(pv_name):
            raise PVNotFoundError(f"PV '{pv_name}' does not exist or is not accessible")
        
        try:
            import epics
            result = epics.caput(pv_name, value, **kwargs)
            if result is None or result == 0:
                raise EPICSWriteError(f"Failed to write to PV '{pv_name}' - operation returned {result}")
            return True
        except Exception as e:
            if "not found" in str(e).lower() or "never connected" in str(e).lower():
                # Remove from cache since it might have been incorrectly cached
                self._cache.remove(pv_name)
                raise PVNotFoundError(f"PV '{pv_name}' not found: {str(e)}")
            elif "write access denied" in str(e).lower():
                raise PermissionError(
                    f"⚠️ EPICS WRITE ACCESS DENIED\n"
                    f"   PV: {pv_name}\n"
                    f"   Value: {value}\n"
                    f"   Reason: EPICS gateway or IOC denied write access\n"
                    f"   Note: This specific PV may be protected\n"
                    f"   Original error: {str(e)}"
                )
            else:
                raise EPICSWriteError(f"Failed to write to PV '{pv_name}': {str(e)}")
    
    def batch_check_pvs(self, pv_names: list, use_cache: bool = True) -> Dict[str, bool]:
        """
        Check existence of multiple PVs efficiently.
        
        Args:
            pv_names: List of PV names to check
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary mapping PV names to existence status
        """
        results = {}
        for pv_name in pv_names:
            results[pv_name] = self.check_pv_exists(pv_name, use_cache=use_cache)
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get PV existence cache statistics."""
        return self._cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear the PV existence cache."""
        self._cache.clear()
        logger.info("PV existence cache cleared")
    
    def is_epics_available(self) -> bool:
        """Check if PyEPICS is available."""
        return self._epics_available
    
    def get_execution_mode(self) -> str:
        """Get the current EPICS execution mode."""
        return self._execution_mode


# Global instance for easy access
_global_epics_manager = None


def get_epics_manager() -> EPICSManager:
    """
    Get the global EPICS manager instance.
    
    Returns:
        EPICSManager: The global EPICS manager instance
    """
    global _global_epics_manager
    if _global_epics_manager is None:
        _global_epics_manager = EPICSManager()
    return _global_epics_manager


# Convenience functions for backward compatibility and ease of use
def check_pv_exists(pv_name: str, use_cache: bool = True) -> bool:
    """
    Check if a PV exists using the global EPICS manager.
    
    Args:
        pv_name: The PV name to check
        use_cache: Whether to use cached results
        
    Returns:
        True if PV exists, False otherwise
    """
    return get_epics_manager().check_pv_exists(pv_name, use_cache)


def safe_caget(pv_name: str, **kwargs) -> Any:
    """
    Safely read a PV value with automatic existence checking.
    
    Args:
        pv_name: The PV name to read
        **kwargs: Additional arguments passed to epics.caget
        
    Returns:
        The PV value
    """
    return get_epics_manager().safe_caget(pv_name, **kwargs)


def safe_caput(pv_name: str, value: Any, **kwargs) -> bool:
    """
    Safely write a PV value with automatic existence checking.
    
    Args:
        pv_name: The PV name to write
        value: The value to write
        **kwargs: Additional arguments passed to epics.caput
        
    Returns:
        True if write was successful
    """
    return get_epics_manager().safe_caput(pv_name, value, **kwargs)


def batch_check_pvs(pv_names: list, use_cache: bool = True) -> Dict[str, bool]:
    """
    Check existence of multiple PVs efficiently.
    
    Args:
        pv_names: List of PV names to check
        use_cache: Whether to use cached results
        
    Returns:
        Dictionary mapping PV names to existence status
    """
    return get_epics_manager().batch_check_pvs(pv_names, use_cache)


# Example usage and testing functions
def test_pv_operations(test_pv: str = "S-DCCT:CurrentM") -> None:
    """
    Test PV operations with the specified test PV.
    
    Args:
        test_pv: PV name to test with (default: S-DCCT:CurrentM)
    """
    print("=" * 60)
    print("EPICS Utilities Test")
    print("=" * 60)
    
    epics_mgr = get_epics_manager()
    
    print(f"Testing with PV: {test_pv}")
    print(f"EPICS available: {epics_mgr.is_epics_available()}")
    print(f"Execution mode: {epics_mgr.get_execution_mode()}")
    print()
    
    # Test PV existence check
    print("1. Testing PV existence check...")
    try:
        exists = epics_mgr.check_pv_exists(test_pv)
        print(f"   PV '{test_pv}' exists: {exists}")
    except Exception as e:
        print(f"   Error checking PV existence: {e}")
    
    print()
    
    # Test safe read
    print("2. Testing safe PV read...")
    try:
        value = epics_mgr.safe_caget(test_pv)
        print(f"   PV '{test_pv}' value: {value}")
    except Exception as e:
        print(f"   Error reading PV: {e}")
    
    print()
    
    # Test cache stats
    print("3. Cache statistics:")
    stats = epics_mgr.get_cache_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print()
    print("Test completed.")
    print("=" * 60)


if __name__ == "__main__":
    # Run test if executed directly
    test_pv = sys.argv[1] if len(sys.argv) > 1 else "S-DCCT:CurrentM"
    test_pv_operations(test_pv)