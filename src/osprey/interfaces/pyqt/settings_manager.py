"""
Settings Manager for Osprey GUI

Handles all GUI settings including:
- Agent control settings
- Approval settings
- Execution limits
- GUI preferences
- Development/debug settings
- Routing settings
- Loading from and saving to config files
"""

from dataclasses import dataclass
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from osprey.utils.logger import get_logger

logger = get_logger("settings_manager")


@dataclass
class AgentControlSettings:
    """Agent control settings."""
    planning_mode_enabled: bool = False
    epics_writes_enabled: bool = False
    task_extraction_bypass_enabled: bool = False
    capability_selection_bypass_enabled: bool = False


@dataclass
class ApprovalSettings:
    """Approval settings."""
    approval_global_mode: str = 'selective'
    python_execution_approval_enabled: bool = True
    python_execution_approval_mode: str = 'all_code'
    memory_approval_enabled: bool = True


@dataclass
class ExecutionLimits:
    """Execution limit settings."""
    max_reclassifications: int = 1
    max_planning_attempts: int = 2
    max_step_retries: int = 0
    max_execution_time_seconds: int = 300
    max_concurrent_classifications: int = 5


@dataclass
class GUISettings:
    """GUI-specific settings."""
    use_persistent_conversations: bool = True
    conversation_storage_mode: str = 'json'
    redirect_output_to_gui: bool = True
    suppress_terminal_output: bool = False
    group_system_messages: bool = True
    enable_routing_feedback: bool = True


@dataclass
class DevelopmentSettings:
    """Development and debug settings."""
    debug_mode: bool = False
    verbose_logging: bool = False
    raise_raw_errors: bool = False
    print_prompts: bool = False
    show_prompts: bool = False
    prompts_latest_only: bool = True


@dataclass
class RoutingSettings:
    """Advanced routing settings."""
    # Cache settings
    enable_routing_cache: bool = True
    cache_max_size: int = 100
    cache_ttl_seconds: float = 3600.0
    cache_similarity_threshold: float = 0.85

    # Advanced invalidation
    enable_advanced_invalidation: bool = True
    enable_adaptive_ttl: bool = True
    enable_probabilistic_expiration: bool = True
    enable_event_driven_invalidation: bool = True

    # Semantic analysis
    enable_semantic_analysis: bool = True
    semantic_similarity_threshold: float = 0.5
    topic_similarity_threshold: float = 0.6
    max_context_history: int = 20

    # Orchestration
    orchestration_max_parallel: int = 3

    # Analytics
    analytics_max_history: int = 1000


class SettingsManager:
    """
    Manages all GUI settings.

    Handles:
    - Settings initialization with defaults
    - Loading settings from config files
    - Saving settings to config files
    - Providing settings as a dictionary for backward compatibility
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize settings manager.

        Args:
            config_path: Optional path to config file to load settings from
        """
        self.config_path = config_path

        # Initialize all settings with defaults
        self.agent_control = AgentControlSettings()
        self.approval = ApprovalSettings()
        self.execution_limits = ExecutionLimits()
        self.gui = GUISettings()
        self.development = DevelopmentSettings()
        self.routing = RoutingSettings()

        # Memory monitoring settings (not in a dataclass, stored directly)
        self.memory_monitor_enabled = True
        self.memory_warning_threshold_mb = 500
        self.memory_critical_threshold_mb = 1000
        self.memory_check_interval_seconds = 5

        # Load from config if provided
        if config_path:
            self.load_from_config(config_path)

    def get_all_settings(self) -> Dict[str, Any]:
        """
        Get all settings as a flat dictionary for backward compatibility.

        Returns:
            Dictionary with all settings
        """
        settings = {}

        # Agent control
        settings.update({
            'planning_mode_enabled': self.agent_control.planning_mode_enabled,
            'epics_writes_enabled': self.agent_control.epics_writes_enabled,
            'task_extraction_bypass_enabled': self.agent_control.task_extraction_bypass_enabled,
            'capability_selection_bypass_enabled': self.agent_control.capability_selection_bypass_enabled,
        })

        # Approval
        settings.update({
            'approval_global_mode': self.approval.approval_global_mode,
            'python_execution_approval_enabled': self.approval.python_execution_approval_enabled,
            'python_execution_approval_mode': self.approval.python_execution_approval_mode,
            'memory_approval_enabled': self.approval.memory_approval_enabled,
        })

        # Execution limits
        settings.update({
            'max_reclassifications': self.execution_limits.max_reclassifications,
            'max_planning_attempts': self.execution_limits.max_planning_attempts,
            'max_step_retries': self.execution_limits.max_step_retries,
            'max_execution_time_seconds': self.execution_limits.max_execution_time_seconds,
            'max_concurrent_classifications': self.execution_limits.max_concurrent_classifications,
        })

        # GUI settings
        settings.update({
            'use_persistent_conversations': self.gui.use_persistent_conversations,
            'conversation_storage_mode': self.gui.conversation_storage_mode,
            'redirect_output_to_gui': self.gui.redirect_output_to_gui,
            'suppress_terminal_output': self.gui.suppress_terminal_output,
            'group_system_messages': self.gui.group_system_messages,
            'enable_routing_feedback': self.gui.enable_routing_feedback,
        })

        # Development
        settings.update({
            'debug_mode': self.development.debug_mode,
            'verbose_logging': self.development.verbose_logging,
            'raise_raw_errors': self.development.raise_raw_errors,
            'print_prompts': self.development.print_prompts,
            'show_prompts': self.development.show_prompts,
            'prompts_latest_only': self.development.prompts_latest_only,
        })

        # Routing
        settings.update({
            'enable_routing_cache': self.routing.enable_routing_cache,
            'cache_max_size': self.routing.cache_max_size,
            'cache_ttl_seconds': self.routing.cache_ttl_seconds,
            'cache_similarity_threshold': self.routing.cache_similarity_threshold,
            'enable_advanced_invalidation': self.routing.enable_advanced_invalidation,
            'enable_adaptive_ttl': self.routing.enable_adaptive_ttl,
            'enable_probabilistic_expiration': self.routing.enable_probabilistic_expiration,
            'enable_event_driven_invalidation': self.routing.enable_event_driven_invalidation,
            'enable_semantic_analysis': self.routing.enable_semantic_analysis,
            'semantic_similarity_threshold': self.routing.semantic_similarity_threshold,
            'topic_similarity_threshold': self.routing.topic_similarity_threshold,
            'max_context_history': self.routing.max_context_history,
            'orchestration_max_parallel': self.routing.orchestration_max_parallel,
            'analytics_max_history': self.routing.analytics_max_history,
        })

        # Memory monitoring
        settings.update({
            'memory_monitor_enabled': self.memory_monitor_enabled,
            'memory_warning_threshold_mb': self.memory_warning_threshold_mb,
            'memory_critical_threshold_mb': self.memory_critical_threshold_mb,
            'memory_check_interval_seconds': self.memory_check_interval_seconds,
        })

        return settings

    def update_from_dict(self, settings_dict: Dict[str, Any]):
        """
        Update settings from a dictionary.

        Args:
            settings_dict: Dictionary with settings to update
        """
        # Agent control
        if 'planning_mode_enabled' in settings_dict:
            self.agent_control.planning_mode_enabled = settings_dict['planning_mode_enabled']
        if 'epics_writes_enabled' in settings_dict:
            self.agent_control.epics_writes_enabled = settings_dict['epics_writes_enabled']
        if 'task_extraction_bypass_enabled' in settings_dict:
            self.agent_control.task_extraction_bypass_enabled = settings_dict['task_extraction_bypass_enabled']
        if 'capability_selection_bypass_enabled' in settings_dict:
            self.agent_control.capability_selection_bypass_enabled = settings_dict['capability_selection_bypass_enabled']

        # Approval
        if 'approval_global_mode' in settings_dict:
            self.approval.approval_global_mode = settings_dict['approval_global_mode']
        if 'python_execution_approval_enabled' in settings_dict:
            self.approval.python_execution_approval_enabled = settings_dict['python_execution_approval_enabled']
        if 'python_execution_approval_mode' in settings_dict:
            self.approval.python_execution_approval_mode = settings_dict['python_execution_approval_mode']
        if 'memory_approval_enabled' in settings_dict:
            self.approval.memory_approval_enabled = settings_dict['memory_approval_enabled']

        # Execution limits
        if 'max_reclassifications' in settings_dict:
            self.execution_limits.max_reclassifications = settings_dict['max_reclassifications']
        if 'max_planning_attempts' in settings_dict:
            self.execution_limits.max_planning_attempts = settings_dict['max_planning_attempts']
        if 'max_step_retries' in settings_dict:
            self.execution_limits.max_step_retries = settings_dict['max_step_retries']
        if 'max_execution_time_seconds' in settings_dict:
            self.execution_limits.max_execution_time_seconds = settings_dict['max_execution_time_seconds']
        if 'max_concurrent_classifications' in settings_dict:
            self.execution_limits.max_concurrent_classifications = settings_dict['max_concurrent_classifications']

        # GUI settings
        if 'use_persistent_conversations' in settings_dict:
            self.gui.use_persistent_conversations = settings_dict['use_persistent_conversations']
        if 'conversation_storage_mode' in settings_dict:
            self.gui.conversation_storage_mode = settings_dict['conversation_storage_mode']
        if 'redirect_output_to_gui' in settings_dict:
            self.gui.redirect_output_to_gui = settings_dict['redirect_output_to_gui']
        if 'suppress_terminal_output' in settings_dict:
            self.gui.suppress_terminal_output = settings_dict['suppress_terminal_output']
        if 'group_system_messages' in settings_dict:
            self.gui.group_system_messages = settings_dict['group_system_messages']
        if 'enable_routing_feedback' in settings_dict:
            self.gui.enable_routing_feedback = settings_dict['enable_routing_feedback']

        # Development
        if 'debug_mode' in settings_dict:
            self.development.debug_mode = settings_dict['debug_mode']
        if 'verbose_logging' in settings_dict:
            self.development.verbose_logging = settings_dict['verbose_logging']
        if 'raise_raw_errors' in settings_dict:
            self.development.raise_raw_errors = settings_dict['raise_raw_errors']
        if 'print_prompts' in settings_dict:
            self.development.print_prompts = settings_dict['print_prompts']
        if 'show_prompts' in settings_dict:
            self.development.show_prompts = settings_dict['show_prompts']
        if 'prompts_latest_only' in settings_dict:
            self.development.prompts_latest_only = settings_dict['prompts_latest_only']

        # Routing
        if 'enable_routing_cache' in settings_dict:
            self.routing.enable_routing_cache = settings_dict['enable_routing_cache']
        if 'cache_max_size' in settings_dict:
            self.routing.cache_max_size = settings_dict['cache_max_size']
        if 'cache_ttl_seconds' in settings_dict:
            self.routing.cache_ttl_seconds = settings_dict['cache_ttl_seconds']
        if 'cache_similarity_threshold' in settings_dict:
            self.routing.cache_similarity_threshold = settings_dict['cache_similarity_threshold']
        if 'enable_advanced_invalidation' in settings_dict:
            self.routing.enable_advanced_invalidation = settings_dict['enable_advanced_invalidation']
        if 'enable_adaptive_ttl' in settings_dict:
            self.routing.enable_adaptive_ttl = settings_dict['enable_adaptive_ttl']
        if 'enable_probabilistic_expiration' in settings_dict:
            self.routing.enable_probabilistic_expiration = settings_dict['enable_probabilistic_expiration']
        if 'enable_event_driven_invalidation' in settings_dict:
            self.routing.enable_event_driven_invalidation = settings_dict['enable_event_driven_invalidation']
        if 'enable_semantic_analysis' in settings_dict:
            self.routing.enable_semantic_analysis = settings_dict['enable_semantic_analysis']
        if 'semantic_similarity_threshold' in settings_dict:
            self.routing.semantic_similarity_threshold = settings_dict['semantic_similarity_threshold']
        if 'topic_similarity_threshold' in settings_dict:
            self.routing.topic_similarity_threshold = settings_dict['topic_similarity_threshold']
        if 'max_context_history' in settings_dict:
            self.routing.max_context_history = settings_dict['max_context_history']
        if 'orchestration_max_parallel' in settings_dict:
            self.routing.orchestration_max_parallel = settings_dict['orchestration_max_parallel']
        if 'analytics_max_history' in settings_dict:
            self.routing.analytics_max_history = settings_dict['analytics_max_history']

        # Memory monitoring settings (missing from original implementation)
        if 'memory_monitor_enabled' in settings_dict:
            self.memory_monitor_enabled = settings_dict['memory_monitor_enabled']
        if 'memory_warning_threshold_mb' in settings_dict:
            self.memory_warning_threshold_mb = settings_dict['memory_warning_threshold_mb']
        if 'memory_critical_threshold_mb' in settings_dict:
            self.memory_critical_threshold_mb = settings_dict['memory_critical_threshold_mb']
        if 'memory_check_interval_seconds' in settings_dict:
            self.memory_check_interval_seconds = settings_dict['memory_check_interval_seconds']

    def load_from_config(self, config_path: str) -> bool:
        """
        Load settings from a YAML config file.

        Args:
            config_path: Path to the config file

        Returns:
            True if successful, False otherwise
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_file}")
                return False

            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f) or {}

            # Load agent control settings
            agent_control = config_data.get('execution_control', {}).get('agent_control', {})
            self.agent_control.task_extraction_bypass_enabled = agent_control.get('task_extraction_bypass_enabled', False)
            self.agent_control.capability_selection_bypass_enabled = agent_control.get('capability_selection_bypass_enabled', False)

            epics = config_data.get('execution_control', {}).get('epics', {})
            self.agent_control.epics_writes_enabled = epics.get('writes_enabled', False)

            # Load approval settings
            approval = config_data.get('approval', {})
            self.approval.approval_global_mode = approval.get('global_mode', 'selective')

            python_exec = approval.get('capabilities', {}).get('python_execution', {})
            self.approval.python_execution_approval_enabled = python_exec.get('enabled', True)
            self.approval.python_execution_approval_mode = python_exec.get('mode', 'all_code')

            memory = approval.get('capabilities', {}).get('memory', {})
            self.approval.memory_approval_enabled = memory.get('enabled', True)

            # Load execution limits
            limits = config_data.get('execution_control', {}).get('limits', {})
            self.execution_limits.max_reclassifications = limits.get('max_reclassifications', 1)
            self.execution_limits.max_planning_attempts = limits.get('max_planning_attempts', 2)
            self.execution_limits.max_step_retries = limits.get('max_step_retries', 0)
            self.execution_limits.max_execution_time_seconds = limits.get('max_execution_time_seconds', 300)
            self.execution_limits.max_concurrent_classifications = limits.get('max_concurrent_classifications', 5)

            # Load GUI settings
            gui = config_data.get('gui', {})
            self.gui.use_persistent_conversations = gui.get('use_persistent_conversations', True)
            self.gui.conversation_storage_mode = gui.get('conversation_storage_mode', 'json')
            self.gui.redirect_output_to_gui = gui.get('redirect_output_to_gui', True)
            self.gui.group_system_messages = gui.get('group_system_messages', True)
            self.gui.suppress_terminal_output = gui.get('suppress_terminal_output', False)

            # Load development settings
            dev = config_data.get('development', {})
            self.development.debug_mode = dev.get('debug', False)
            self.development.raise_raw_errors = dev.get('raise_raw_errors', False)

            prompts = dev.get('prompts', {})
            self.development.print_prompts = prompts.get('print_all', False)
            self.development.show_prompts = prompts.get('show_all', False)
            self.development.prompts_latest_only = prompts.get('latest_only', True)

            # Load routing settings
            routing = config_data.get('routing', {})

            cache = routing.get('cache', {})
            self.routing.enable_routing_cache = cache.get('enabled', True)
            self.routing.cache_max_size = cache.get('max_size', 100)
            self.routing.cache_ttl_seconds = cache.get('ttl_seconds', 3600.0)
            self.routing.cache_similarity_threshold = cache.get('similarity_threshold', 0.85)

            invalidation = routing.get('advanced_invalidation', {})
            self.routing.enable_advanced_invalidation = invalidation.get('enabled', True)
            self.routing.enable_adaptive_ttl = invalidation.get('adaptive_ttl', True)
            self.routing.enable_probabilistic_expiration = invalidation.get('probabilistic_expiration', True)
            self.routing.enable_event_driven_invalidation = invalidation.get('event_driven', True)

            semantic = routing.get('semantic_analysis', {})
            self.routing.enable_semantic_analysis = semantic.get('enabled', True)
            self.routing.semantic_similarity_threshold = semantic.get('similarity_threshold', 0.5)
            self.routing.topic_similarity_threshold = semantic.get('topic_similarity_threshold', 0.6)
            self.routing.max_context_history = semantic.get('max_context_history', 20)

            orchestration = routing.get('orchestration', {})
            self.routing.orchestration_max_parallel = orchestration.get('max_parallel', 3)

            analytics = routing.get('analytics', {})
            self.routing.analytics_max_history = analytics.get('max_history', 1000)

            feedback = routing.get('feedback', {})
            self.gui.enable_routing_feedback = feedback.get('enabled', True)

            # Load memory monitoring settings
            memory_monitoring = config_data.get('memory_monitoring', {})
            self.memory_monitor_enabled = memory_monitoring.get('enabled', True)
            self.memory_warning_threshold_mb = memory_monitoring.get('warning_threshold_mb', 500)
            self.memory_critical_threshold_mb = memory_monitoring.get('critical_threshold_mb', 1000)
            self.memory_check_interval_seconds = memory_monitoring.get('check_interval_seconds', 5)

            logger.info(f"Loaded settings from {config_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to load settings from config: {e}")
            return False

    def save_to_config(self, config_path: str) -> bool:
        """
        Save settings to a YAML config file.

        Args:
            config_path: Path to the config file

        Returns:
            True if successful, False otherwise
        """
        try:
            config_file = Path(config_path)

            # Read existing config
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                config_data = {}

            # Update execution_control section
            if 'execution_control' not in config_data:
                config_data['execution_control'] = {}

            if 'agent_control' not in config_data['execution_control']:
                config_data['execution_control']['agent_control'] = {}

            config_data['execution_control']['agent_control']['task_extraction_bypass_enabled'] = self.agent_control.task_extraction_bypass_enabled
            config_data['execution_control']['agent_control']['capability_selection_bypass_enabled'] = self.agent_control.capability_selection_bypass_enabled

            if 'epics' not in config_data['execution_control']:
                config_data['execution_control']['epics'] = {}

            config_data['execution_control']['epics']['writes_enabled'] = self.agent_control.epics_writes_enabled

            # Update approval section
            if 'approval' not in config_data:
                config_data['approval'] = {}

            config_data['approval']['global_mode'] = self.approval.approval_global_mode

            if 'capabilities' not in config_data['approval']:
                config_data['approval']['capabilities'] = {}

            if 'python_execution' not in config_data['approval']['capabilities']:
                config_data['approval']['capabilities']['python_execution'] = {}

            config_data['approval']['capabilities']['python_execution']['enabled'] = self.approval.python_execution_approval_enabled
            config_data['approval']['capabilities']['python_execution']['mode'] = self.approval.python_execution_approval_mode

            if 'memory' not in config_data['approval']['capabilities']:
                config_data['approval']['capabilities']['memory'] = {}

            config_data['approval']['capabilities']['memory']['enabled'] = self.approval.memory_approval_enabled

            # Update execution limits
            if 'limits' not in config_data['execution_control']:
                config_data['execution_control']['limits'] = {}

            config_data['execution_control']['limits']['max_reclassifications'] = self.execution_limits.max_reclassifications
            config_data['execution_control']['limits']['max_planning_attempts'] = self.execution_limits.max_planning_attempts
            config_data['execution_control']['limits']['max_step_retries'] = self.execution_limits.max_step_retries
            config_data['execution_control']['limits']['max_execution_time_seconds'] = self.execution_limits.max_execution_time_seconds
            config_data['execution_control']['limits']['max_concurrent_classifications'] = self.execution_limits.max_concurrent_classifications

            # Update GUI section
            if 'gui' not in config_data:
                config_data['gui'] = {}

            config_data['gui']['use_persistent_conversations'] = self.gui.use_persistent_conversations
            config_data['gui']['conversation_storage_mode'] = self.gui.conversation_storage_mode
            config_data['gui']['redirect_output_to_gui'] = self.gui.redirect_output_to_gui
            config_data['gui']['group_system_messages'] = self.gui.group_system_messages
            config_data['gui']['suppress_terminal_output'] = self.gui.suppress_terminal_output

            # Update development section
            if 'development' not in config_data:
                config_data['development'] = {}

            config_data['development']['debug'] = self.development.debug_mode
            config_data['development']['raise_raw_errors'] = self.development.raise_raw_errors

            if 'prompts' not in config_data['development']:
                config_data['development']['prompts'] = {}

            config_data['development']['prompts']['print_all'] = self.development.print_prompts
            config_data['development']['prompts']['show_all'] = self.development.show_prompts
            config_data['development']['prompts']['latest_only'] = self.development.prompts_latest_only

            # Update routing section
            if 'routing' not in config_data:
                config_data['routing'] = {}

            if 'cache' not in config_data['routing']:
                config_data['routing']['cache'] = {}

            config_data['routing']['cache']['enabled'] = self.routing.enable_routing_cache
            config_data['routing']['cache']['max_size'] = self.routing.cache_max_size
            config_data['routing']['cache']['ttl_seconds'] = self.routing.cache_ttl_seconds
            config_data['routing']['cache']['similarity_threshold'] = self.routing.cache_similarity_threshold

            if 'advanced_invalidation' not in config_data['routing']:
                config_data['routing']['advanced_invalidation'] = {}

            config_data['routing']['advanced_invalidation']['enabled'] = self.routing.enable_advanced_invalidation
            config_data['routing']['advanced_invalidation']['adaptive_ttl'] = self.routing.enable_adaptive_ttl
            config_data['routing']['advanced_invalidation']['probabilistic_expiration'] = self.routing.enable_probabilistic_expiration
            config_data['routing']['advanced_invalidation']['event_driven'] = self.routing.enable_event_driven_invalidation

            if 'semantic_analysis' not in config_data['routing']:
                config_data['routing']['semantic_analysis'] = {}

            config_data['routing']['semantic_analysis']['enabled'] = self.routing.enable_semantic_analysis
            config_data['routing']['semantic_analysis']['similarity_threshold'] = self.routing.semantic_similarity_threshold
            config_data['routing']['semantic_analysis']['topic_similarity_threshold'] = self.routing.topic_similarity_threshold
            config_data['routing']['semantic_analysis']['max_context_history'] = self.routing.max_context_history

            if 'orchestration' not in config_data['routing']:
                config_data['routing']['orchestration'] = {}

            config_data['routing']['orchestration']['max_parallel'] = self.routing.orchestration_max_parallel

            if 'analytics' not in config_data['routing']:
                config_data['routing']['analytics'] = {}

            config_data['routing']['analytics']['max_history'] = self.routing.analytics_max_history

            if 'feedback' not in config_data['routing']:
                config_data['routing']['feedback'] = {}

            config_data['routing']['feedback']['enabled'] = self.gui.enable_routing_feedback

            # Update memory monitoring section
            if 'memory_monitoring' not in config_data:
                config_data['memory_monitoring'] = {}

            config_data['memory_monitoring']['enabled'] = self.memory_monitor_enabled
            config_data['memory_monitoring']['warning_threshold_mb'] = self.memory_warning_threshold_mb
            config_data['memory_monitoring']['critical_threshold_mb'] = self.memory_critical_threshold_mb
            config_data['memory_monitoring']['check_interval_seconds'] = self.memory_check_interval_seconds

            # Write back to file
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, indent=2)

            logger.info(f"Saved settings to {config_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save settings to config: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value by key (for backward compatibility).

        Args:
            key: Setting key
            default: Default value if key not found

        Returns:
            Setting value or default
        """
        settings = self.get_all_settings()
        return settings.get(key, default)
