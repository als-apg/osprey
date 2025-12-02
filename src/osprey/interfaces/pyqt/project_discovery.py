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
                    # Found a project!
                    project_info = {
                        'name': subdir.name,
                        'path': str(subdir),
                        'config_path': str(config_file)
                    }
                    
                    # Try to extract registry_path from config
                    try:
                        with open(config_file, 'r') as f:
                            config = yaml.safe_load(f)
                            if config and isinstance(config, dict):
                                registry_path = config.get('registry_path')
                                if registry_path:
                                    # Resolve relative to project directory
                                    if not Path(registry_path).is_absolute():
                                        registry_path = str(subdir / registry_path)
                                    project_info['registry_path'] = registry_path
                    except Exception as e:
                        logger.warning(f"Could not read config from {config_file}: {e}")
                    
                    projects.append(project_info)
            
            except (PermissionError, OSError):
                pass
            
            checked_count += 1
    
    except Exception as e:
        logger.warning(f"Error during project discovery: {e}")
    
    logger.info(f"Discovered {len(projects)} projects: {[p['name'] for p in projects]}")
    return projects


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
    
    try:
        with open(base_config_path, 'r') as f:
            unified_config = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Could not load base config from {base_config_path}: {e}")
        unified_config = {}
    
    # Set registry_path to unified registry (in same directory as config)
    unified_registry_path = output_path.parent / "unified_registry.py"
    unified_config['registry_path'] = str(unified_registry_path)
    
    # Merge configurations from other projects
    for project in projects[1:]:
        try:
            with open(project['config_path'], 'r') as f:
                project_config = yaml.safe_load(f)
                if not project_config:
                    continue
                
                # Merge specific sections (models, api, etc.)
                for section in ['models', 'api', 'execution', 'file_paths']:
                    if section in project_config:
                        if section not in unified_config:
                            unified_config[section] = {}
                        if isinstance(project_config[section], dict):
                            unified_config[section].update(project_config[section])
        
        except Exception as e:
            logger.warning(f"Could not merge config from {project['name']}: {e}")
    
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