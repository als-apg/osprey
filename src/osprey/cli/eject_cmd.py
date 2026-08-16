"""Eject command for copying native framework components to local projects.

This command allows users to copy framework-native services to their local
project for customization. This is useful when the default framework behavior
needs to be modified beyond what prompt customization allows.

Usage:
    osprey eject list                          # List ejectable components
    osprey eject service channel_finder        # Copy service to local project
"""

import importlib
import shutil
from pathlib import Path

import click

from osprey.utils.logger import get_logger

from .output import fail, note, report, section, warn

logger = get_logger("osprey.cli")


def _get_project_package_dir() -> Path | None:
    """Find the user's project package directory (src/<package_name>/).

    Returns:
        Path to the package directory, or None if not found.
    """
    src_dir = Path("src")
    if not src_dir.exists():
        return None

    # Find the first package directory under src/
    for item in src_dir.iterdir():
        if item.is_dir() and (item / "__init__.py").exists():
            return item

    return None


def _get_module_source_path(module_path: str) -> Path | None:
    """Get the filesystem path for a Python module.

    Args:
        module_path: Dotted module path (e.g., "osprey.capabilities.channel_finding")

    Returns:
        Path to the module's source file or directory, or None if not found.
    """
    try:
        spec = importlib.util.find_spec(module_path)
        if spec and spec.origin:
            return Path(spec.origin)
        elif spec and spec.submodule_search_locations:
            # Package - return directory
            locations = list(spec.submodule_search_locations)
            if locations:
                return Path(locations[0])
    except (ModuleNotFoundError, ValueError):
        logger.debug("Could not resolve module path: %s", module_path)
    return None


@click.group()
def eject():
    """Copy framework components out for customization.

    This command copies native framework services to your local project
    directory so you can modify them.

    Examples:

    \b
      osprey eject list                          List available components
      osprey eject service channel_finder        Copy channel finder service
    """
    pass


@eject.command("list")
def eject_list():
    """List all ejectable framework capabilities and services."""
    from osprey.registry.builtins import FrameworkRegistryProvider

    provider = FrameworkRegistryProvider()
    config = provider.get_registry_config()

    report("")
    section("Ejectable services", {svc.name: svc.description for svc in config.services})
    report("")
    report("Usage:")
    note("osprey eject service <name>   Copy a service into your project")


@eject.command("service")
@click.argument("name")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output directory path (default: ./src/<package>/services/<name>/)",
)
@click.option("--include-tests", is_flag=True, default=False, help="Also copy related test files")
def eject_service(name: str, output: str | None, include_tests: bool):
    """Copy a framework service to local project for customization.

    This copies the entire service directory including all sub-modules.
    """
    from osprey.registry.builtins import FrameworkRegistryProvider

    # Find the service in the framework registry
    provider = FrameworkRegistryProvider()
    config = provider.get_registry_config()

    svc_reg = None
    for svc in config.services:
        if svc.name == name:
            svc_reg = svc
            break

    if not svc_reg:
        available = [svc.name for svc in config.services]
        fail(f"Unknown service '{name}'", f"Available: {', '.join(available)}")
        raise SystemExit(1)

    # Find source directory (service module path points to the service.py file)
    # We want the parent directory (the service package)
    source_path = _get_module_source_path(svc_reg.module_path)
    if not source_path or not source_path.exists():
        fail(f"Could not find the source for '{svc_reg.module_path}'")
        raise SystemExit(1)

    # Service source is the parent directory of the module file
    source_dir = source_path.parent
    if not source_dir.is_dir():
        fail(f"Expected a directory at {source_dir}")
        raise SystemExit(1)

    # Determine output path
    if output:
        output_dir = Path(output)
    else:
        pkg_dir = _get_project_package_dir()
        if not pkg_dir:
            fail(
                "Could not find the project package directory",
                None,
                "Run this from the project root, or name a target with --output.",
            )
            raise SystemExit(1)
        output_dir = pkg_dir / "services" / name

    # Copy the entire service directory
    if output_dir.exists():
        warn(f"{output_dir} already exists", "Its contents were replaced.")
        shutil.rmtree(output_dir)

    shutil.copytree(source_dir, output_dir)

    # Count files copied
    file_count = sum(1 for _ in output_dir.rglob("*.py"))
    report(f"Ejected service '{name}' to {output_dir} ({file_count} Python files)")

    # Copy tests if requested
    if include_tests:
        test_dir = Path("tests/services") / name
        test_dir.mkdir(parents=True, exist_ok=True)
        # source_dir: src/osprey/services/name/ -> 4 parents to project root
        project_root = source_dir.parent.parent.parent.parent
        tests_root = project_root / "tests" / "services" / name
        if tests_root.exists():
            shutil.copytree(tests_root, test_dir, dirs_exist_ok=True)
            test_count = sum(1 for _ in test_dir.rglob("test_*.py"))
            note(f"Copied {test_count} test files to {test_dir}")

    report("")
    report("Next steps:")
    note(f"1. Modify files in {output_dir} for your needs")
    note("2. Update imports in your capabilities to use the local service")
    note("3. Run 'osprey health' to check it")
