"""Version information utilities for Osprey Framework.

This module provides utilities for retrieving version information
for the Osprey framework and its key dependencies.
"""

import importlib.metadata as metadata
import sys
from typing import Any, Dict


def get_osprey_version() -> str:
    """Get the Osprey framework version.
    
    Returns:
        Version string from osprey.__version__ or 'Unknown'
    """
    try:
        import osprey
        return osprey.__version__
    except (ImportError, AttributeError):
        return "Unknown"


def get_package_version(package_name: str) -> str:
    """Get version of an installed package.
    
    Args:
        package_name: Name of the package (e.g., 'langgraph', 'langchain-core')
        
    Returns:
        Version string or 'Not installed' if package is not found
    """
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "Not installed"


def get_core_dependencies() -> Dict[str, str]:
    """Get versions of core framework dependencies.
    
    Returns:
        Dictionary mapping package names to version strings
    """
    core_packages = [
        'langgraph',
        'langchain-core',
        'langgraph-sdk',
        'pydantic-ai',
        'rich',
        'click',
        'PyYAML',
    ]
    
    return {pkg: get_package_version(pkg) for pkg in core_packages}


def get_optional_dependencies() -> Dict[str, str]:
    """Get versions of optional framework dependencies.
    
    Returns:
        Dictionary mapping package names to version strings
    """
    optional_packages = [
        'openai',
        'anthropic',
        'google-generativeai',
        'ollama',
        'pandas',
        'numpy',
        'matplotlib',
        'mem0ai',
        'pymongo',
        'neo4j',
    ]
    
    return {pkg: get_package_version(pkg) for pkg in optional_packages}


def get_python_version() -> str:
    """Get Python version string.
    
    Returns:
        Python version (e.g., '3.11.5')
    """
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_all_versions() -> Dict[str, Any]:
    """Get comprehensive version information.
    
    Returns:
        Dictionary with version information for:
        - osprey: Osprey framework version
        - python: Python version
        - core: Core dependencies
        - optional: Optional dependencies
    """
    return {
        'osprey': get_osprey_version(),
        'python': get_python_version(),
        'core': get_core_dependencies(),
        'optional': get_optional_dependencies(),
    }


def format_version_info(include_optional: bool = False) -> str:
    """Format version information as a readable string.
    
    Args:
        include_optional: Whether to include optional dependencies
        
    Returns:
        Formatted version information string
    """
    versions = get_all_versions()
    
    lines = [
        f"Osprey Framework: {versions['osprey']}",
        f"Python: {versions['python']}",
        "",
        "Core Dependencies:",
    ]
    
    for pkg, ver in versions['core'].items():
        lines.append(f"  {pkg}: {ver}")
    
    if include_optional:
        lines.extend([
            "",
            "Optional Dependencies:",
        ])
        for pkg, ver in versions['optional'].items():
            if ver != "Not installed":
                lines.append(f"  {pkg}: {ver}")
    
    return "\n".join(lines)


def print_version_info(include_optional: bool = False):
    """Print version information to console.
    
    Args:
        include_optional: Whether to include optional dependencies
    """
    print(format_version_info(include_optional=include_optional))