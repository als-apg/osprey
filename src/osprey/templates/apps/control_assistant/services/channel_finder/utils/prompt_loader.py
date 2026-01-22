"""Dynamic prompt loader for facility-specific or generic prompts."""

import logging
import sys
from pathlib import Path
from typing import Any

# Use Osprey's config system for path resolution
from osprey.utils.config import get_config_builder

logger = logging.getLogger(__name__)


def load_prompts(config: dict, require_query_splitter: bool = True) -> Any:
    """Load prompts based on configuration.

    Priority order:
    1. Pipeline-specific prompts (channel_finder.pipelines.<mode>.prompts.path)
    2. Facility-specific prompts (facility.path)
    3. Generic prompts (fallback)

    Args:
        config: Configuration dictionary
        require_query_splitter: Whether query_splitter is required (False if query splitting is disabled)

    Returns:
        Prompts module with query_splitter, channel_matcher, correction
    """
    # Get pipeline mode to determine required prompt files
    pipeline_mode = config.get("channel_finder", {}).get("pipeline_mode", "in_context")

    # 1. Try pipeline-specific prompts first (highest priority)
    pipeline_config = config.get("channel_finder", {}).get("pipelines", {}).get(pipeline_mode, {})
    pipeline_prompts_config = pipeline_config.get("prompts", {})
    pipeline_prompts_path = pipeline_prompts_config.get("path", "")

    if pipeline_prompts_path:
        prompts = _try_load_prompts_directly(
            pipeline_prompts_path, pipeline_mode, require_query_splitter
        )
        if prompts:
            logger.info(
                f"[dim]✓ Loaded pipeline-specific prompts from {pipeline_prompts_path}[/dim]"
            )
            return prompts
        else:
            logger.warning(f"⚠ Pipeline prompts configured but not found: {pipeline_prompts_path}")

    # 2. Try facility-specific prompts if requested
    facility_config = config.get("facility", {})
    prompt_source = facility_config.get("prompts", "generic")
    facility_path = facility_config.get("path", "")

    if prompt_source == "facility" and facility_path:
        prompts = _try_load_facility_prompts(facility_path, pipeline_mode, require_query_splitter)
        if prompts:
            logger.info(
                f"[dim]✓ Loaded facility-specific prompts from {facility_path}/prompts[/dim]"
            )
            return prompts
        else:
            logger.warning("⚠ Facility prompts not found or incomplete, falling back to generic")
    elif prompt_source == "facility":
        logger.warning("⚠ Prompts: facility requested but no facility.path configured")

    # 3. No prompts found - this should not happen in normal operation
    # Both pipelines should have prompts configured in channel_finder.pipelines.<mode>.prompts.path
    logger.error("❌ No prompts could be loaded! Check configuration.")
    logger.error("   Expected: channel_finder.pipelines.<mode>.prompts.path in config.yml")
    raise RuntimeError(
        f"No prompts found for {pipeline_mode} pipeline. "
        "Please configure prompts.path in channel_finder.pipelines section of config.yml"
    )


def _try_load_prompts_directly(
    prompts_path: str, pipeline_mode: str = "in_context", require_query_splitter: bool = True
) -> Any | None:
    """Try to load prompts module directly from specified path.

    Args:
        prompts_path: Direct path to prompts directory (not facility root)
        pipeline_mode: Pipeline mode ('in_context' or 'hierarchical')
        require_query_splitter: Whether query_splitter is required

    Returns:
        Prompts module if found, None otherwise
    """
    try:
        # Build path to prompts using Osprey config
        config_builder = get_config_builder()
        project_root = Path(config_builder.get("project_root"))
        prompts_path_obj = Path(prompts_path)

        if prompts_path_obj.is_absolute():
            prompts_dir = prompts_path_obj
        else:
            prompts_dir = project_root / prompts_path

        if not prompts_dir.exists():
            logger.debug(f"Prompts directory does not exist: {prompts_dir}")
            return None

        # Determine required files based on pipeline mode and query_splitting setting
        base_files = ["__init__.py", "system.py"]
        if require_query_splitter:
            base_files.append("query_splitter.py")

        if pipeline_mode == "hierarchical":
            required_files = base_files
        elif pipeline_mode == "middle_layer":
            # Middle layer uses React agent, only needs query_splitter (if enabled)
            required_files = base_files
        else:
            # in_context and other modes need full prompt set
            required_files = base_files + ["channel_matcher.py", "correction.py"]

        missing_files = [f for f in required_files if not (prompts_dir / f).exists()]
        if missing_files:
            logger.warning(f"⚠ Prompts incomplete. Missing: {missing_files}")
            logger.debug(f"  Required for {pipeline_mode} mode: {required_files}")
            return None

        # Add to sys.path temporarily and import
        prompts_dir_str = str(prompts_dir.parent)
        sys.path.insert(0, prompts_dir_str)

        try:
            # Import the prompts module
            import importlib

            module_name = prompts_dir.name
            prompts_module = importlib.import_module(module_name)

            # Verify it has the required attributes based on pipeline mode
            if pipeline_mode in ["hierarchical", "middle_layer"]:
                required_attrs = ["query_splitter"] if require_query_splitter else []
            else:
                required_attrs = ["channel_matcher", "correction"]
                if require_query_splitter:
                    required_attrs.insert(0, "query_splitter")

            missing_attrs = [attr for attr in required_attrs if not hasattr(prompts_module, attr)]
            if missing_attrs:
                logger.warning(f"⚠ Prompts missing required submodules: {missing_attrs}")
                logger.debug(f"  Required for {pipeline_mode} mode: {required_attrs}")
                return None

            return prompts_module

        finally:
            # Clean up sys.path
            if prompts_dir_str in sys.path:
                sys.path.remove(prompts_dir_str)

    except Exception as e:
        logger.debug(f"Failed to load prompts directly: {e}")
        return None


def _try_load_facility_prompts(facility_path: str, pipeline_mode: str = "in_context") -> Any | None:
    """Try to load facility-specific prompts module.

    Args:
        facility_path: Path to facility directory
        pipeline_mode: Pipeline mode ('in_context' or 'hierarchical')

    Returns:
        Prompts module if found, None otherwise
    """
    try:
        # Build path to facility prompts using Osprey config
        config_builder = get_config_builder()
        project_root = Path(config_builder.get("project_root"))
        facility_path_obj = Path(facility_path)

        if facility_path_obj.is_absolute():
            facility_root = facility_path_obj
        else:
            facility_root = project_root / facility_path

        prompts_dir = facility_root / "prompts"

        if not prompts_dir.exists():
            return None

        # Determine required files based on pipeline mode
        # All pipelines need: __init__.py, system.py (facility description), query_splitter.py
        base_files = ["__init__.py", "system.py", "query_splitter.py"]

        if pipeline_mode == "hierarchical":
            # Hierarchical pipeline uses base files + hierarchical_context.py (loaded separately by pipeline)
            required_files = base_files
        elif pipeline_mode == "middle_layer":
            # Middle layer uses React agent, only needs query_splitter (no channel matcher/correction)
            required_files = base_files
        else:
            # In-context and other pipelines need base files + channel matching and correction
            required_files = base_files + ["channel_matcher.py", "correction.py"]

        missing_files = [f for f in required_files if not (prompts_dir / f).exists()]
        if missing_files:
            logger.warning(f"⚠ Facility prompts incomplete. Missing: {missing_files}")
            logger.debug(f"  Required for {pipeline_mode} mode: {required_files}")
            return None

        # Add to sys.path temporarily and import
        prompts_dir_str = str(prompts_dir.parent)
        sys.path.insert(0, prompts_dir_str)

        try:
            # Import the prompts module
            import importlib

            prompts_module = importlib.import_module("prompts")

            # Verify it has the required attributes based on pipeline mode
            if pipeline_mode in ["hierarchical", "middle_layer"]:
                required_attrs = ["query_splitter"]
            else:
                required_attrs = ["query_splitter", "channel_matcher", "correction"]

            missing_attrs = [attr for attr in required_attrs if not hasattr(prompts_module, attr)]
            if missing_attrs:
                logger.warning(f"⚠ Facility prompts missing required submodules: {missing_attrs}")
                logger.debug(f"  Required for {pipeline_mode} mode: {required_attrs}")
                return None

            return prompts_module

        finally:
            # Clean up sys.path
            if prompts_dir_str in sys.path:
                sys.path.remove(prompts_dir_str)

    except Exception as e:
        logger.debug(f"Failed to load facility prompts: {e}")
        return None
