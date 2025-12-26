"""
Runtime Override Manager

Manages runtime overrides for execution behavior that should be applied
at runtime but NOT written to project configuration files.

This allows the GUI to modify execution behavior without polluting
version control or affecting other users of the same project.
"""

from typing import Dict, Any
from copy import deepcopy
from osprey.utils.logger import get_logger

logger = get_logger("runtime_overrides")


class RuntimeOverrideManager:
    """
    Manages runtime overrides for execution behavior.
    
    These overrides are applied to the configuration at runtime but are
    NEVER written to disk. This prevents:
    - Version control pollution
    - Multi-project conflicts
    - Unintended configuration changes
    
    The overrides are stored in memory and applied when creating the
    base_config for agent execution.
    """
    
    def __init__(self):
        """Initialize runtime override manager."""
        self.overrides = {}
        logger.info("Initialized RuntimeOverrideManager")
    
    def set_override(self, key: str, value: Any) -> None:
        """
        Set a runtime override.
        
        Args:
            key: Override key (can be dot-separated path like 'execution_control.limits.max_retries')
            value: Override value
        """
        self.overrides[key] = value
        logger.debug(f"Set runtime override: {key} = {value}")
    
    def set_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Set multiple runtime overrides at once.
        
        Args:
            overrides: Dictionary of overrides
        """
        self.overrides.update(overrides)
        logger.debug(f"Set {len(overrides)} runtime overrides")
    
    def get_override(self, key: str, default: Any = None) -> Any:
        """
        Get a runtime override value.
        
        Args:
            key: Override key
            default: Default value if not found
            
        Returns:
            Override value or default
        """
        return self.overrides.get(key, default)
    
    def remove_override(self, key: str) -> None:
        """
        Remove a runtime override.
        
        Args:
            key: Override key to remove
        """
        if key in self.overrides:
            del self.overrides[key]
            logger.debug(f"Removed runtime override: {key}")
    
    def clear_overrides(self) -> None:
        """Clear all runtime overrides."""
        self.overrides.clear()
        logger.info("Cleared all runtime overrides")
    
    def get_all_overrides(self) -> Dict[str, Any]:
        """
        Get all runtime overrides.
        
        Returns:
            Dictionary of all overrides
        """
        return self.overrides.copy()
    
    def apply_to_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply runtime overrides to a configuration dictionary.
        
        This creates a NEW dictionary with overrides applied, leaving the
        original configuration unchanged.
        
        Args:
            config_dict: Original configuration dictionary
            
        Returns:
            New configuration dictionary with overrides applied
        """
        # Deep copy to avoid modifying original
        result = deepcopy(config_dict)
        
        # Apply each override
        for key, value in self.overrides.items():
            self._set_nested_value(result, key, value)
        
        return result
    
    def _set_nested_value(self, d: Dict[str, Any], path: str, value: Any) -> None:
        """
        Set a value in a nested dictionary using dot-separated path.
        
        Args:
            d: Dictionary to modify
            path: Dot-separated path (e.g., 'execution_control.limits.max_retries')
            value: Value to set
        """
        keys = path.split('.')
        current = d
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def create_agent_control_defaults(self) -> Dict[str, Any]:
        """
        Create agent_control_defaults dictionary from overrides.
        
        This is used to populate the 'agent_control_defaults' section
        of the base_config that gets passed to the agent.
        
        Returns:
            Dictionary of agent control settings
        """
        return {
            # Agent Control
            'planning_mode_enabled': self.get_override('planning_mode_enabled', False),
            'epics_writes_enabled': self.get_override('epics_writes_enabled', False),
            'task_extraction_bypass_enabled': self.get_override('task_extraction_bypass_enabled', False),
            'capability_selection_bypass_enabled': self.get_override('capability_selection_bypass_enabled', False),
            'parallel_execution_enabled': self.get_override('parallel_execution_enabled', False),
            
            # Approval
            'approval_global_mode': self.get_override('approval_global_mode', 'selective'),
            'python_execution_approval_enabled': self.get_override('python_execution_approval_enabled', True),
            'python_execution_approval_mode': self.get_override('python_execution_approval_mode', 'all_code'),
            'memory_approval_enabled': self.get_override('memory_approval_enabled', True),
            
            # Execution Limits
            'max_reclassifications': self.get_override('max_reclassifications', 1),
            'max_planning_attempts': self.get_override('max_planning_attempts', 2),
            'max_step_retries': self.get_override('max_step_retries', 0),
            'max_execution_time_seconds': self.get_override('max_execution_time_seconds', 300),
            'max_concurrent_classifications': self.get_override('max_concurrent_classifications', 5),
            
            # Development
            'debug_mode': self.get_override('debug_mode', False),
            'verbose_logging': self.get_override('verbose_logging', False),
            'raise_raw_errors': self.get_override('raise_raw_errors', False),
            'print_prompts': self.get_override('print_prompts', False),
            'show_prompts': self.get_override('show_prompts', False),
            'prompts_latest_only': self.get_override('prompts_latest_only', True),
            
            # Routing (for backward compatibility)
            'enable_routing_cache': self.get_override('enable_routing_cache', True),
            'cache_max_size': self.get_override('cache_max_size', 100),
            'cache_ttl_seconds': self.get_override('cache_ttl_seconds', 3600.0),
            'cache_similarity_threshold': self.get_override('cache_similarity_threshold', 0.85),
            'enable_advanced_invalidation': self.get_override('enable_advanced_invalidation', True),
            'enable_adaptive_ttl': self.get_override('enable_adaptive_ttl', True),
            'enable_probabilistic_expiration': self.get_override('enable_probabilistic_expiration', True),
            'enable_event_driven_invalidation': self.get_override('enable_event_driven_invalidation', True),
            'enable_semantic_analysis': self.get_override('enable_semantic_analysis', True),
            'semantic_similarity_threshold': self.get_override('semantic_similarity_threshold', 0.5),
            'topic_similarity_threshold': self.get_override('topic_similarity_threshold', 0.6),
            'max_context_history': self.get_override('max_context_history', 20),
            'orchestration_max_parallel': self.get_override('orchestration_max_parallel', 3),
            'analytics_max_history': self.get_override('analytics_max_history', 1000),
        }