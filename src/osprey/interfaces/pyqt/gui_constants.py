"""
GUI Constants for Osprey Framework

Centralized constants for GUI dimensions, colors, defaults, and magic numbers.
This eliminates hardcoded values scattered throughout the codebase.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowDimensions:
    """Main window dimensions and positioning."""
    
    # Main window
    MAIN_X: int = 100
    MAIN_Y: int = 100
    MAIN_WIDTH: int = 1200
    MAIN_HEIGHT: int = 800
    
    # Settings dialog
    SETTINGS_WIDTH: int = 750
    SETTINGS_HEIGHT: int = 750  # Updated to match actual usage
    SETTINGS_MIN_WIDTH: int = 600
    SETTINGS_MIN_HEIGHT: int = 650  # Updated to match actual usage


@dataclass(frozen=True)
class UILayout:
    """UI layout constants."""
    
    # Input field
    INPUT_FIELD_LINES: int = 4
    INPUT_FIELD_PADDING: int = 10
    INPUT_FIELD_MAX_WIDTH: int = 800
    
    # Splitter sizes (History | Conversation | Status)
    SPLITTER_HISTORY_WIDTH: int = 250
    SPLITTER_CONVERSATION_WIDTH: int = 650
    SPLITTER_STATUS_WIDTH: int = 400
    
    # Button sizes
    SMALL_BUTTON_WIDTH: int = 50
    CACHE_BUTTON_HEIGHT: int = 25
    
    # Table row height
    DEFAULT_ROW_HEIGHT: int = 80


@dataclass(frozen=True)
class ColorPalette:
    """Centralized color definitions for the GUI."""
    
    # Component colors
    BASE: str = '#FFFFFF'
    CONTEXT: str = '#AFD7FF'
    ROUTER: str = '#FF00FF'
    ORCHESTRATOR: str = '#00FFFF'
    MONITOR: str = '#CD8500'
    CLASSIFIER: str = '#FFA07A'
    TASK_EXTRACTION: str = '#D8BFD8'
    ERROR: str = '#FF0000'
    GATEWAY: str = '#FFA07A'
    APPROVAL: str = '#FFA07A'
    TIME_RANGE_PARSING: str = '#1E90FF'
    MEMORY: str = '#FFA07A'
    PYTHON: str = '#FFA07A'
    RESPOND: str = '#D8BFD8'
    CLARIFY: str = '#D8BFD8'
    
    # Message colors
    USER_MESSAGE: str = '#D8BFD8'
    AGENT_MESSAGE: str = '#FFFFFF'
    SYSTEM_MESSAGE: str = '#808080'
    SUCCESS_MESSAGE: str = '#00FF00'
    WARNING_MESSAGE: str = '#FFA500'
    ERROR_MESSAGE: str = '#FF0000'
    INFO_MESSAGE: str = '#00FFFF'
    QUEUED_MESSAGE: str = '#FFD700'
    PROCESSING_MESSAGE: str = '#808080'
    
    # UI element colors
    TIMESTAMP: str = '#808080'
    SEPARATOR: str = '#3F3F46'
    BORDER: str = '#3F3F46'
    BACKGROUND_DARK: str = '#1E1E1E'
    BACKGROUND_MEDIUM: str = '#2D2D30'
    HIGHLIGHT: str = '#0078D4'
    
    # Status colors
    ENABLED: str = '#00FF00'
    DISABLED: str = '#808080'
    
    # Tool usage colors
    TOOL_LABEL: str = '#FFD700'
    TOOL_CAPABILITY: str = '#00FFFF'
    TOOL_SUCCESS: str = '#00FF00'
    TOOL_FAILURE: str = '#FF0000'
    TOOL_TIMING: str = '#FFA500'
    
    @classmethod
    def get_component_color(cls, component: str) -> str:
        """
        Get color for a component type.
        
        Args:
            component: Component name (case-insensitive)
            
        Returns:
            Color hex string, defaults to BASE if not found
        """
        component_upper = component.upper().replace(' ', '_')
        return getattr(cls, component_upper, cls.BASE)


@dataclass(frozen=True)
class CacheDefaults:
    """Default values for caching configuration."""
    
    # Routing cache
    CACHE_MAX_SIZE: int = 100
    CACHE_TTL_SECONDS: float = 3600.0
    CACHE_SIMILARITY_THRESHOLD: float = 0.85
    
    # Image cache
    IMAGE_CACHE_MAX_SIZE_MB: int = 100
    IMAGE_CACHE_RETENTION_HOURS: int = 24
    
    # Analytics
    ANALYTICS_MAX_HISTORY: int = 1000
    ANALYTICS_MAX_ENTRIES: int = 1000
    ANALYTICS_RETENTION_DAYS: int = 7


@dataclass(frozen=True)
class ConversationDefaults:
    """Default values for conversation management."""
    
    MAX_MESSAGES_PER_CONVERSATION: int = 50
    MAX_LLM_DETAILS_ENTRIES: int = 100
    MAX_TOOL_USAGE_ENTRIES: int = 50
    MAX_CONTEXT_HISTORY: int = 20


@dataclass(frozen=True)
class ExecutionDefaults:
    """Default values for execution limits."""
    
    MAX_RECLASSIFICATIONS: int = 1
    MAX_PLANNING_ATTEMPTS: int = 2
    MAX_STEP_RETRIES: int = 0
    MAX_EXECUTION_TIME_SECONDS: int = 300
    MAX_CONCURRENT_CLASSIFICATIONS: int = 5
    ORCHESTRATION_MAX_PARALLEL: int = 3


@dataclass(frozen=True)
class TimerDelays:
    """Timer delay constants in milliseconds."""
    
    FRAMEWORK_INIT_DELAY_MS: int = 100
    PROJECTS_REFRESH_DELAY_MS: int = 200
    PROJECT_SELECTOR_UPDATE_DELAY_MS: int = 300
    CONVERSATION_DISPLAY_LOAD_DELAY_MS: int = 100


@dataclass(frozen=True)
class MemoryDefaults:
    """Default values for memory monitoring."""
    
    WARNING_THRESHOLD_MB: int = 500
    CRITICAL_THRESHOLD_MB: int = 1000
    CHECK_INTERVAL_SECONDS: int = 5


@dataclass(frozen=True)
class SettingsKeys:
    """Centralized settings key constants to avoid magic strings."""
    
    # Agent Control
    PLANNING_MODE_ENABLED: str = 'planning_mode_enabled'
    EPICS_WRITES_ENABLED: str = 'epics_writes_enabled'
    TASK_EXTRACTION_BYPASS_ENABLED: str = 'task_extraction_bypass_enabled'
    CAPABILITY_SELECTION_BYPASS_ENABLED: str = 'capability_selection_bypass_enabled'
    
    # Approval
    APPROVAL_GLOBAL_MODE: str = 'approval_global_mode'
    PYTHON_EXECUTION_APPROVAL_ENABLED: str = 'python_execution_approval_enabled'
    PYTHON_EXECUTION_APPROVAL_MODE: str = 'python_execution_approval_mode'
    MEMORY_APPROVAL_ENABLED: str = 'memory_approval_enabled'
    
    # Execution Limits
    MAX_RECLASSIFICATIONS: str = 'max_reclassifications'
    MAX_PLANNING_ATTEMPTS: str = 'max_planning_attempts'
    MAX_STEP_RETRIES: str = 'max_step_retries'
    MAX_EXECUTION_TIME_SECONDS: str = 'max_execution_time_seconds'
    MAX_CONCURRENT_CLASSIFICATIONS: str = 'max_concurrent_classifications'
    
    # GUI Settings
    USE_PERSISTENT_CONVERSATIONS: str = 'use_persistent_conversations'
    CONVERSATION_STORAGE_MODE: str = 'conversation_storage_mode'
    REDIRECT_OUTPUT_TO_GUI: str = 'redirect_output_to_gui'
    SUPPRESS_TERMINAL_OUTPUT: str = 'suppress_terminal_output'
    GROUP_SYSTEM_MESSAGES: str = 'group_system_messages'
    ENABLE_ROUTING_FEEDBACK: str = 'enable_routing_feedback'
    MAX_MESSAGES_PER_CONVERSATION: str = 'max_messages_per_conversation'
    MAX_LLM_DETAILS_ENTRIES: str = 'max_llm_details_entries'
    MAX_TOOL_USAGE_ENTRIES: str = 'max_tool_usage_entries'
    LLM_DETAILS_AUTO_CLEAR: str = 'llm_details_auto_clear'
    IMAGE_CACHE_MAX_SIZE_MB: str = 'image_cache_max_size_mb'
    IMAGE_CACHE_RETENTION_HOURS: str = 'image_cache_retention_hours'
    IMAGE_CACHE_AUTO_CLEANUP: str = 'image_cache_auto_cleanup'
    
    # Development
    DEBUG_MODE: str = 'debug_mode'
    VERBOSE_LOGGING: str = 'verbose_logging'
    RAISE_RAW_ERRORS: str = 'raise_raw_errors'
    PRINT_PROMPTS: str = 'print_prompts'
    SHOW_PROMPTS: str = 'show_prompts'
    PROMPTS_LATEST_ONLY: str = 'prompts_latest_only'
    
    # Routing
    ENABLE_ROUTING_CACHE: str = 'enable_routing_cache'
    CACHE_MAX_SIZE: str = 'cache_max_size'
    CACHE_TTL_SECONDS: str = 'cache_ttl_seconds'
    CACHE_SIMILARITY_THRESHOLD: str = 'cache_similarity_threshold'
    ENABLE_ADVANCED_INVALIDATION: str = 'enable_advanced_invalidation'
    ENABLE_ADAPTIVE_TTL: str = 'enable_adaptive_ttl'
    ENABLE_PROBABILISTIC_EXPIRATION: str = 'enable_probabilistic_expiration'
    ENABLE_EVENT_DRIVEN_INVALIDATION: str = 'enable_event_driven_invalidation'
    ENABLE_SEMANTIC_ANALYSIS: str = 'enable_semantic_analysis'
    SEMANTIC_SIMILARITY_THRESHOLD: str = 'semantic_similarity_threshold'
    TOPIC_SIMILARITY_THRESHOLD: str = 'topic_similarity_threshold'
    MAX_CONTEXT_HISTORY: str = 'max_context_history'
    ORCHESTRATION_MAX_PARALLEL: str = 'orchestration_max_parallel'
    ANALYTICS_MAX_HISTORY: str = 'analytics_max_history'
    ANALYTICS_MAX_ENTRIES: str = 'analytics_max_entries'
    ANALYTICS_RETENTION_DAYS: str = 'analytics_retention_days'
    ANALYTICS_AUTO_CLEANUP: str = 'analytics_auto_cleanup'
    MEMORY_MONITOR_ENABLED: str = 'memory_monitor_enabled'
    MEMORY_WARNING_THRESHOLD_MB: str = 'memory_warning_threshold_mb'
    MEMORY_CRITICAL_THRESHOLD_MB: str = 'memory_critical_threshold_mb'
    MEMORY_CHECK_INTERVAL_SECONDS: str = 'memory_check_interval_seconds'


# Convenience instances for easy access
WINDOW = WindowDimensions()
LAYOUT = UILayout()
COLORS = ColorPalette()
CACHE = CacheDefaults()
CONVERSATION = ConversationDefaults()
EXECUTION = ExecutionDefaults()
MEMORY = MemoryDefaults()
TIMERS = TimerDelays()
KEYS = SettingsKeys()


# Backward compatibility: Export individual constants
__all__ = [
    'WindowDimensions',
    'UILayout',
    'ColorPalette',
    'CacheDefaults',
    'ConversationDefaults',
    'ExecutionDefaults',
    'MemoryDefaults',
    'TimerDelays',
    'SettingsKeys',
    'WINDOW',
    'LAYOUT',
    'COLORS',
    'CACHE',
    'CONVERSATION',
    'EXECUTION',
    'MEMORY',
    'TIMERS',
    'KEYS',
]