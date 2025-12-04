"""
Jupyter Startup Script for ALS Agent Environment

This script runs on Jupyter kernel startup to configure the environment
for safe EPICS operations based on the execution mode.
"""

import logging
import os
import sys
import traceback  # Ensure traceback is imported for the final except block
from pathlib import Path

# Try to configure logging, but don't fail if we can't write the log file
log_file_path = '/home/jovyan/work/startup_script.log'
try:
    # Try to set up file logging
    logging.basicConfig(filename=log_file_path,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
except Exception as e:
    # If file logging fails, fall back to console logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                        stream=sys.stdout)
    print(f"Warning: Could not create log file at {log_file_path}: {e}")
    print("Logging to console instead.")

logger = logging.getLogger(__name__)

logger.info("--- Custom Jupyter kernel startup script (via IPython profile) BEGIN ---")
logger.debug(f"Python executable: {sys.executable}")
logger.debug(f"Initial sys.path: {sys.path}")
logger.debug(f"Current working directory: {os.getcwd()}")
logger.debug(f"__file__ (path to this script when executed by kernel): {__file__}")

def print_environment_info():
    """Print environment information for debugging."""
    print("=" * 70)
    print("ALS Agent Jupyter Environment")
    print("=" * 70)

    execution_mode = os.environ.get('EPICS_EXECUTION_MODE', 'unknown')
    epics_addr = os.environ.get('EPICS_CA_ADDR_LIST', 'Not configured')
    epics_port = os.environ.get('EPICS_CA_SERVER_PORT', 'Not configured')

    print(f"📍 Current execution mode: {execution_mode}")
    print(f"🌐 EPICS Gateway: {epics_addr}:{epics_port}")

    # Container-specific information
    if execution_mode == "read":
        print("🔒 READ CONTAINER: Read-only operations only")
        print("")
        print("   📋 Purpose:")
        print("   - Safe data analysis and visualization")
        print("   - Available kernels: Read-Only")
        print("   - EPICS write operations are BLOCKED for safety")
        print("")

    elif execution_mode == "write_access":
        print("⚠️  WRITE CONTAINER: DEV ONLY - Can modify live systems!")
        print("")
        print("   📋 Available kernels:")
        print("   - Available kernels: Read-Only, Write Access")
        print("   - 💡 Start with Read-Only kernel for safety")
        print("")

    print("📚 Execution modes:")
    print("   🔒 Read-Only: Safe for data analysis (caget commands)")
    print("   ⚠️  Write Access: Can modify control systems (caput commands) - USE WITH EXTREME CAUTION")
    print("")
    print("   Recommended workflow: Start with Read-Only → Use Write Access carefully when needed")

    print("\n" + "="*80)

# Make common imports available
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    print("✓ Standard scientific libraries loaded")
    import epics
    print("✓ EPICS library loaded and ready")
except ImportError as e:
    print(f"⚠️  Some standard libraries not available: {e}")

# Initialize NLTK packages to avoid download messages in notebooks
# TODO: cleanly separate from osprey startup!
try:
    from src.applications.als_assistant.services.pv_finder.util import initialize_nltk_resources
    initialize_nltk_resources()
    print("✓ NLTK packages initialized")
except ImportError as e:
    print(f"⚠️  Could not import NLTK initialization function: {e}")
except Exception as e:
    print(f"⚠️  NLTK initialization failed: {e}")
    # Continue anyway - not critical for core functionality

# Setup EPICS
def setup_epics():
    """Setup EPICS with improved error messages based on kernel mode."""
    try:
        import epics
        execution_mode = os.environ.get('EPICS_EXECUTION_MODE', 'unknown')

        # Store original caput function
        _original_caput = epics.caput

        def enhanced_caput(pvname, value, wait=False, timeout=30, **kwargs):
            """Enhanced caput with user-friendly error messages."""

            # Check if this is the read-only kernel (no simulation mode)
            if execution_mode == 'read':
                # This is the read-only kernel
                raise PermissionError(
                    f"🔒 WRITE OPERATION BLOCKED\n"
                    f"   PV: {pvname}\n"
                    f"   Value: {value}\n"
                    f"   Reason: You are using the Read-Only kernel\n"
                    f"   Solution: Switch to '🧪 EPICS Simulation' kernel to test writes safely\n"
                    f"            or '⚠️ Write Access' kernel for real machine control"
                ) from None  # Suppress original traceback

            else:
                # This is write-access kernel - use original function
                try:
                    return _original_caput(pvname, value, wait=wait, timeout=timeout, **kwargs)
                except Exception as e:
                    if "Write access denied" in str(e):
                        # Even in write kernel, if EPICS denies the write, provide helpful info
                        raise PermissionError(
                            f"⚠️ EPICS WRITE ACCESS DENIED\n"
                            f"   PV: {pvname}\n"
                            f"   Value: {value}\n"
                            f"   Reason: EPICS gateway or IOC denied write access\n"
                            f"   Note: You are in Write Access kernel but this specific PV may be protected\n"
                            f"   Original error: {str(e)}"
                        ) from None  # Suppress original traceback
                    else:
                        # Re-raise other exceptions unchanged
                        raise

        # Replace epics.caput with our enhanced version
        epics.caput = enhanced_caput

        # Make it available globally
        globals()['caput'] = enhanced_caput

        print(f"✓ EPICS configured with enhanced error handling (mode: {execution_mode})")
        if execution_mode == 'read':
            print("  🔒 Read-only mode - writes will be blocked with helpful messages")
        elif execution_mode == 'write_access':
            print("  ⚠️ Write access mode - REAL WRITES ENABLED (use with caution)")

    except ImportError:
        print("⚠️ PyEPICS not available - skipping EPICS error handling setup")
    except Exception as e:
        print(f"⚠️ Failed to setup EPICS error handling: {e}")

setup_epics()

# Add utility functions for users
def kernel_info():
    """Display current kernel mode and capabilities."""
    execution_mode = os.environ.get('EPICS_EXECUTION_MODE', 'unknown')
    epics_addr = os.environ.get('EPICS_CA_ADDR_LIST', 'Not configured')
    epics_port = os.environ.get('EPICS_CA_SERVER_PORT', 'Not configured')

    print("=" * 50)
    print("🔍 CURRENT KERNEL STATUS")
    print("=" * 50)

    if execution_mode == 'read':
        print("🔒 KERNEL: Read-Only")
        print("📊 EPICS Reads: ✅ Real data from storage ring")
        print("✏️  EPICS Writes: ❌ Blocked (will show helpful error)")
        print("🎯 Use Case: Safe data analysis and monitoring")
    elif execution_mode == 'write_access':
        print("⚠️  KERNEL: Write Access")
        print("📊 EPICS Reads: ✅ Real data from storage ring")
        print("✏️  EPICS Writes: ⚠️  REAL WRITES TO HARDWARE")
        print("🎯 Use Case: Actual machine control (DANGEROUS)")
    else:
        print(f"❓ KERNEL: Unknown mode ({execution_mode})")

    print(f"🌐 EPICS Gateway: {epics_addr}:{epics_port}")
    print("=" * 50)
    print("💡 Call kernel_info() anytime to see this information")

# Make utility functions globally available
globals()['kernel_info'] = kernel_info

print_environment_info()
print("Environment setup complete. Ready for agent-generated code execution.")
print("")
print("🛠️  AVAILABLE HELPER FUNCTIONS:")
print("   📋 kernel_info() - Check current kernel mode and capabilities")
print("   📊 get_archiver_data(pv_list, start_date, end_date) - Retrieve archived data")
print("   ✏️  epics.caput() or caput() - Enhanced with user-friendly error messages")

try:
    logger.info("Attempting to add project root to sys.path and import 'get_archiver_data'.")


    # Print local directory structure
    logger.info("Local directory structure:")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Contents of current directory: {os.listdir(os.getcwd())}")

    # Check if /jupyter/repo_src exists (copied source during build)
    if Path("/jupyter/repo_src").exists():
        logger.info(f"Contents of /jupyter/repo_src: {os.listdir('/jupyter/repo_src')}")

    # The project source is copied to /jupyter/repo_src via the build system
    repo_src = Path("/jupyter/repo_src").resolve()
    logger.debug(f"Target repo_src: {repo_src}")

    if not repo_src.exists():
        logger.error(f"Project root {repo_src} does not exist. Cannot import 'get_archiver_data'.")
        logger.error("Looking for alternative paths...")

        # Log what's available in /jupyter
        if Path("/jupyter").exists():
            logger.info(f"Contents of /jupyter: {os.listdir('/jupyter')}")
    else:
        logger.info(f"Project root exists. Contents: {os.listdir(repo_src)}")

        if str(repo_src) not in sys.path:
            sys.path.insert(0, str(repo_src))
            logger.info(f"Project root '{repo_src}' ADDED to sys.path.")
        else:
            logger.info(f"Project root '{repo_src}' was ALREADY in sys.path.")
        logger.debug(f"Updated sys.path: {sys.path}")

        # Test import of nbformat directly first
        try:
            import nbformat
            logger.info(f"Successfully imported 'nbformat'. Version: {getattr(nbformat, '__version__', 'unknown')}")
        except ImportError as nie:
            logger.error(f"Failed to import 'nbformat': {nie}")
            logger.error(f"Detailed traceback for nbformat import error: {traceback.format_exc()}")
            logger.error("Please ensure 'nbformat' is listed in requirements_jupiter.txt and installed.")

except ImportError as e:
    logger.error(f"ImportError during startup: {e}")
    logger.error(f"Detailed traceback: {traceback.format_exc()}")
    logger.error("Check sys.path and ensure the module 'services.ALS_assistant.agent_tools' and its dependencies (like nbformat) are accessible and installed.")
    logger.debug(f"sys.path at time of error: {sys.path}")
    if 'repo_src' in locals():
        logger.debug(f"Project_root during error: {repo_src}")

except Exception as e:
    logger.error(f"An unexpected error occurred during startup script execution: {e}")
    logger.error(f"Detailed traceback: {traceback.format_exc()}")

finally:
    logger.info("--- Custom Jupyter kernel startup script (via IPython profile) END ---")
