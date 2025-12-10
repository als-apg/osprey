#!/usr/bin/env python3
"""
Project Discovery and Unified Configuration Generation for Osprey Framework GUI

This module provides functionality to discover osprey projects in subdirectories
and generate unified configuration and registry files for multi-project setups.
"""

import sys
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

from osprey.utils.logger import get_logger
from osprey.interfaces.pyqt.gui_utils import load_config_safe

logger = get_logger("project_discovery")


def discover_projects(base_dir: Path, max_dirs: int = 50) -> List[Dict[str, Any]]:
    """Discover osprey projects in immediate subdirectories.
    
    This performs a SHALLOW, non-recursive search (1 level deep only) for
    config.yml files in subdirectories, similar to the CLI's discover_nearby_projects().
    
    Args:
        base_dir: Base directory to search in
        max_dirs: Maximum number of subdirectories to check
        
    Returns:
        List of project info dictionaries with keys:
        - name: Project directory name
        - path: Full path to project directory
        - config_path: Path to config.yml
        - registry_path: Path to registry file (if found in config)
        - capabilities: List of capability names (extracted from registry)
        - models: Dict of model configurations (from config.yml)
    """
    projects = []
    
    # Directories to ignore
    ignore_dirs = {
        'node_modules', 'venv', '.venv', 'env', '.env',
        '__pycache__', '.git', '.svn', '.hg',
        'build', 'dist', '.egg-info', 'site-packages',
        '.pytest_cache', '.mypy_cache', '.tox',
        'docs', '_agent_data', '.cache', 'temp_configs'
    }
    
    try:
        checked_count = 0
        subdirs = []
        
        # Get all immediate subdirectories
        for item in base_dir.iterdir():
            if not item.is_dir():
                continue
            if item.name.startswith('.'):
                continue
            if item.name in ignore_dirs:
                continue
            subdirs.append(item)
        
        # Sort for consistent ordering
        subdirs.sort(key=lambda p: p.name.lower())
        
        # Check each subdirectory for config.yml
        for subdir in subdirs:
            if checked_count >= max_dirs:
                logger.debug(f"Stopped after checking {max_dirs} directories")
                break
            
            try:
                config_file = subdir / 'config.yml'
                
                if config_file.exists() and config_file.is_file():
                    # Double-check: skip if this is a docs directory (even if it has config.yml)
                    if subdir.name == 'docs' or subdir.name.startswith('doc'):
                        logger.debug(f"Skipping docs directory: {subdir.name}")
                        continue
                    
                    # Found a project!
                    project_info = {
                        'name': subdir.name,
                        'path': str(subdir),
                        'config_path': str(config_file)
                    }
                    
                    # Try to extract registry_path and other info from config
                    config = load_config_safe(str(config_file))
                    if config and isinstance(config, dict):
                        registry_path = config.get('registry_path')
                        if registry_path:
                            # Resolve relative to project directory
                            if not Path(registry_path).is_absolute():
                                registry_path = str(subdir / registry_path)
                            project_info['registry_path'] = registry_path
                        
                        # Extract capabilities from registry file
                        project_info['capabilities'] = _extract_capabilities(subdir, registry_path if registry_path else None)
                        
                        # Extract model configurations
                        project_info['models'] = _extract_models(config)
                    
                    projects.append(project_info)
            
            except (PermissionError, OSError):
                pass
            
            checked_count += 1
    
    except Exception as e:
        logger.warning(f"Error during project discovery: {e}")
    
    logger.info(f"Discovered {len(projects)} projects: {[p['name'] for p in projects]}")
    return projects


def _extract_capabilities(project_dir: Path, registry_path: str | None) -> List[str]:
    """Extract capability names from a project's registry file.
    
    Args:
        project_dir: Path to project directory
        registry_path: Path to registry file (if available)
    
    Returns:
        List of capability names
    """
    capabilities = []
    
    if not registry_path:
        return capabilities
    
    try:
        registry_file = Path(registry_path)
        if not registry_file.exists():
            return capabilities
        
        # Read the registry file and extract capability names
        with open(registry_file, 'r') as f:
            content = f.read()
        
        # Look for CapabilityRegistration patterns
        import re
        # Match patterns like: CapabilityRegistration(name="capability_name"
        pattern = r'CapabilityRegistration\s*\(\s*name\s*=\s*["\']([^"\']+)["\']'
        matches = re.findall(pattern, content)
        capabilities.extend(matches)
        
        logger.debug(f"Extracted {len(capabilities)} capabilities from {registry_file.name}")
        
    except Exception as e:
        logger.debug(f"Could not extract capabilities: {e}")
    
    return capabilities


def _extract_models(config: dict) -> Dict[str, str]:
    """Extract model configurations from project config.
    
    Args:
        config: Project configuration dictionary
    
    Returns:
        Dictionary mapping model step names to model IDs
    """
    models = {}
    
    try:
        # Extract from models section
        models_config = config.get('models', {})
        if isinstance(models_config, dict):
            for step_name, step_config in models_config.items():
                if isinstance(step_config, dict):
                    model_id = step_config.get('model_id')
                    provider = step_config.get('provider')
                    if model_id:
                        # Format as "provider/model_id" for clarity
                        if provider:
                            models[step_name] = f"{provider}/{model_id}"
                        else:
                            models[step_name] = model_id
        
        logger.debug(f"Extracted {len(models)} model configurations")
        
    except Exception as e:
        logger.debug(f"Could not extract models: {e}")
    
    return models


def _adjust_channel_finder_paths(channel_finder_config: dict, project_dir: Path, unified_config_dir: Path) -> dict:
    """Adjust paths in channel_finder configuration to be relative to unified config location.
    
    Args:
        channel_finder_config: The channel_finder section from project config
        project_dir: Path to the project directory
        unified_config_dir: Path to directory where unified config will be created
    
    Returns:
        Adjusted channel_finder configuration
    """
    import copy
    config = copy.deepcopy(channel_finder_config)
    
    # Adjust paths in pipelines section
    if 'pipelines' in config:
        for pipeline_name, pipeline_config in config['pipelines'].items():
            if isinstance(pipeline_config, dict):
                # Adjust database path
                if 'database' in pipeline_config and 'path' in pipeline_config['database']:
                    old_path = pipeline_config['database']['path']
                    if not Path(old_path).is_absolute():
                        # Convert to absolute, then back to relative from unified config dir
                        abs_path = (project_dir / old_path).resolve()
                        try:
                            new_path = abs_path.relative_to(unified_config_dir)
                            pipeline_config['database']['path'] = str(new_path)
                        except ValueError:
                            # Can't make relative, use absolute
                            pipeline_config['database']['path'] = str(abs_path)
                
                # Adjust prompts path
                if 'prompts' in pipeline_config and 'path' in pipeline_config['prompts']:
                    old_path = pipeline_config['prompts']['path']
                    if not Path(old_path).is_absolute():
                        abs_path = (project_dir / old_path).resolve()
                        try:
                            new_path = abs_path.relative_to(unified_config_dir)
                            pipeline_config['prompts']['path'] = str(new_path)
                        except ValueError:
                            pipeline_config['prompts']['path'] = str(abs_path)
                
                # Adjust benchmark dataset path
                if 'benchmark' in pipeline_config and 'dataset_path' in pipeline_config['benchmark']:
                    old_path = pipeline_config['benchmark']['dataset_path']
                    if not Path(old_path).is_absolute():
                        abs_path = (project_dir / old_path).resolve()
                        try:
                            new_path = abs_path.relative_to(unified_config_dir)
                            pipeline_config['benchmark']['dataset_path'] = str(new_path)
                        except ValueError:
                            pipeline_config['benchmark']['dataset_path'] = str(abs_path)
    
    # Adjust benchmark output results_dir
    if 'benchmark' in config and 'output' in config['benchmark'] and 'results_dir' in config['benchmark']['output']:
        old_path = config['benchmark']['output']['results_dir']
        if not Path(old_path).is_absolute():
            abs_path = (project_dir / old_path).resolve()
            try:
                new_path = abs_path.relative_to(unified_config_dir)
                config['benchmark']['output']['results_dir'] = str(new_path)
            except ValueError:
                config['benchmark']['output']['results_dir'] = str(abs_path)
    
    return config


def _deep_merge_dict(target: dict, source: dict) -> None:
    """Deep merge source dictionary into target dictionary.
    
    This recursively merges nested dictionaries, allowing proper combination
    of complex configuration structures like channel_finder.pipelines.in_context.
    
    Args:
        target: Dictionary to merge into (modified in place)
        source: Dictionary to merge from
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            _deep_merge_dict(target[key], value)
        else:
            # Overwrite or add new key
            target[key] = value


def create_unified_config(projects: List[Dict[str, Any]], output_path: Optional[Path] = None) -> str:
    """Create a unified configuration file from multiple projects.
    
    Args:
        projects: List of project info dictionaries from discover_projects()
        output_path: Optional path where unified_config.yml should be created.
                    If None, creates in the current working directory (project root)
        
    Returns:
        Path to the created unified config file
    """
    if not projects:
        raise ValueError("No projects provided for unified config generation")
    
    # Default to current working directory (project root) if no output path specified
    if output_path is None:
        output_path = Path.cwd() / "unified_config.yml"
    
    # Use first project's config as base
    base_config_path = Path(projects[0]['config_path'])
    unified_config = load_config_safe(str(base_config_path)) or {}
    
    # Adjust paths in the base project's channel_finder section
    if 'channel_finder' in unified_config:
        unified_config['channel_finder'] = _adjust_channel_finder_paths(
            unified_config['channel_finder'],
            Path(projects[0]['path']),
            output_path.parent
        )
    
    # Set registry_path to unified registry (in same directory as config)
    unified_registry_path = output_path.parent / "unified_registry.py"
    unified_config['registry_path'] = str(unified_registry_path)
    
    # Merge configurations from other projects
    for project in projects[1:]:
        project_config = load_config_safe(project['config_path'])
        if not project_config:
            continue

        # Adjust paths in channel_finder section to be relative to unified config location
        if 'channel_finder' in project_config:
            project_config['channel_finder'] = _adjust_channel_finder_paths(
                project_config['channel_finder'],
                Path(project['path']),
                output_path.parent
            )

        # Merge specific sections (models, api, services, etc.)
        # Include all common configuration sections that capabilities might need
        for section in ['models', 'api', 'execution', 'file_paths', 'services', 'channel_finder', 'approval', 'python_executor', 'logging']:
            if section in project_config:
                if section not in unified_config:
                    unified_config[section] = {}
                if isinstance(project_config[section], dict):
                    # Deep merge for nested configurations
                    _deep_merge_dict(unified_config[section], project_config[section])
    
    # Write unified config
    header = f"""# ============================================================
# Unified Multi-Project Osprey Configuration
# ============================================================
# Automatically generated from {len(projects)} project(s)
# Projects: {', '.join(p['name'] for p in projects)}
# ============================================================

"""
    
    try:
        with open(output_path, 'w') as f:
            f.write(header)
            yaml.dump(unified_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Created unified config at: {output_path}")
        return str(output_path)
    
    except Exception as e:
        logger.error(f"Failed to write unified config: {e}")
        raise


def create_unified_registry(projects: List[Dict[str, Any]], output_path: Optional[Path] = None) -> str:
    """Create a unified registry file that combines all project registries.
    
    Args:
        projects: List of project info dictionaries from discover_projects()
        output_path: Optional path where unified_registry.py should be created.
                    If None, creates in the current working directory (project root)
        
    Returns:
        Path to the created unified registry file
    """
    if not projects:
        raise ValueError("No projects provided for unified registry generation")
    
    # Default to current working directory (project root) if no output path specified
    if output_path is None:
        output_path = Path.cwd() / "unified_registry.py"
    
    # Filter projects that have registry_path
    projects_with_registry = [p for p in projects if 'registry_path' in p]
    
    if not projects_with_registry:
        raise ValueError("No projects have registry_path defined in their config")
    
    # Generate the unified registry code
    registry_code = _generate_unified_registry_code(projects_with_registry, output_path.parent)
    
    # Write the file
    try:
        with open(output_path, 'w') as f:
            f.write(registry_code)
        
        logger.info(f"Created unified registry at: {output_path}")
        return str(output_path)
    
    except Exception as e:
        logger.error(f"Failed to write unified registry: {e}")
        raise


def _generate_unified_registry_code(projects: List[Dict[str, Any]], base_dir: Path) -> str:
    """Generate Python code for the unified registry."""
    lines = [
        '"""',
        'Unified Multi-Project Registry',
        'Automatically generated to combine all discovered project registries.',
        '"""',
        '',
        'import sys',
        'from pathlib import Path',
        '',
        'from osprey.registry import (',
        '    extend_framework_registry,',
        '    CapabilityRegistration,',
        '    ContextClassRegistration,',
        '    RegistryConfig,',
        '    RegistryConfigProvider',
        ')',
        '',
        '',
        'class UnifiedMultiProjectRegistryProvider(RegistryConfigProvider):',
        '    """Unified registry combining all discovered projects."""',
        '    ',
        '    def get_registry_config(self) -> RegistryConfig:',
        '        """Combine all project registries into one."""',
        '        # Add project src directories to sys.path for imports',
        '        _base_dir = Path(__file__).parent',
        '        _project_src_dirs = [',
    ]
    
    # Add each project's src directory
    for project in projects:
        project_path = Path(project['path'])
        rel_path = project_path.relative_to(base_dir)
        
        # Check if project has src/ directory
        src_dir = project_path / 'src'
        if src_dir.exists():
            lines.append(f"            _base_dir / '{rel_path}' / 'src',")
        else:
            # Use project directory itself
            lines.append(f"            _base_dir / '{rel_path}',")
    
    lines.extend([
        '        ]',
        '',
        '        for src_dir in _project_src_dirs:',
        '            src_dir_str = str(src_dir.resolve())',
        '            if src_dir.exists() and src_dir_str not in sys.path:',
        '                sys.path.insert(0, src_dir_str)',
        '        ',
        '        # Import project registry providers dynamically',
    ])
    
    # Generate imports for each project
    provider_classes = []
    for i, project in enumerate(projects):
        registry_path = Path(project['registry_path'])
        
        # Extract module name from registry path
        # e.g., /path/to/project/src/my_project/registry.py -> my_project
        module_name = registry_path.parent.name
        
        # Generate provider class name
        provider_class = f"{module_name.replace('_', ' ').replace('-', ' ').title().replace(' ', '')}RegistryProvider"
        provider_classes.append((module_name, provider_class, i))
        
        lines.append(f"        from {module_name}.registry import {provider_class}")
    
    lines.extend([
        '        ',
        '        # Collect all components from projects',
        '        all_capabilities = []',
        '        all_context_classes = []',
        '        all_data_sources = []',
        '        all_services = []',
        '        all_prompt_providers = []',
        '        ',
    ])
    
    # Add code to load each project's registry
    for module_name, provider_class, i in provider_classes:
        project_name = projects[i]['name']
        lines.extend([
            f"        # Load {project_name} registry",
            f"        project{i}_config = {provider_class}().get_registry_config()",
            f"        all_capabilities.extend(project{i}_config.capabilities)",
            f"        all_context_classes.extend(project{i}_config.context_classes)",
            f"        if project{i}_config.data_sources:",
            f"            all_data_sources.extend(project{i}_config.data_sources)",
            f"        if project{i}_config.services:",
            f"            all_services.extend(project{i}_config.services)",
            f"        if project{i}_config.framework_prompt_providers:",
            f"            all_prompt_providers.extend(project{i}_config.framework_prompt_providers)",
            '        ',
        ])
    
    lines.extend([
        '        # Return extended registry with all projects',
        '        return extend_framework_registry(',
        '            capabilities=all_capabilities,',
        '            context_classes=all_context_classes,',
        '            data_sources=all_data_sources,',
        '            services=all_services,',
        '            framework_prompt_providers=all_prompt_providers',
        '        )',
        ''
    ])
    
    return '\n'.join(lines)