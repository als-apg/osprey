"""
Enums and Constants for Osprey PyQt GUI

This module centralizes all magic strings, colors, and constants used throughout
the GUI to improve type safety and maintainability.
"""

from enum import Enum


class StorageMode(Enum):
    """Conversation storage modes."""
    JSON = 'json'
    POSTGRESQL = 'postgresql'


class MessageType(Enum):
    """Message types with associated display properties."""
    USER = ('user', '👤 You: ', '#D8BFD8')
    AGENT = ('agent', '🤖 ', '#FFFFFF')
    STATUS = ('status', 'ℹ️ ', '#00FFFF')
    ERROR = ('error', '❌ ', '#FF0000')
    SUCCESS = ('success', '✅ ', '#00FF00')
    WARNING = ('warning', '⚠️ ', '#FFA500')
    PROCESSING = ('processing', '▶️ ', '#00FFFF')
    
    def __init__(self, type_id: str, prefix: str, color: str):
        self.type_id = type_id
        self.prefix = prefix
        self.color = color


class ComponentType(Enum):
    """Component types for status updates and color coding."""
    BASE = ('base', '#FFFFFF')
    ORCHESTRATOR = ('orchestrator', '#FFD700')
    ROUTER = ('router', '#00FFFF')
    CLASSIFIER = ('classifier', '#FF69B4')
    TASK_EXTRACTOR = ('task_extractor', '#98FB98')
    CLARIFIER = ('clarifier', '#FFA500')
    RESPONDER = ('responder', '#87CEEB')
    ERROR = ('error', '#FF0000')
    
    def __init__(self, component_id: str, color: str):
        self.component_id = component_id
        self.color = color


class LLMEventType(Enum):
    """LLM event types for detailed logging."""
    LLM_START = ('llm_start', '#00FFFF')
    LLM_END = ('llm_end', '#00FF00')
    LLM_STREAM = ('llm_stream', '#FFFF00')
    CLASSIFICATION = ('classification', '#FFD700')
    STATUS = ('status', '#FFFFFF')
    BASE = ('base', '#FFFFFF')
    
    def __init__(self, event_id: str, color: str):
        self.event_id = event_id
        self.color = color


class Colors:
    """Color constants for GUI elements."""
    # Message colors
    USER_MESSAGE = '#D8BFD8'
    AGENT_MESSAGE = '#FFFFFF'
    STATUS_MESSAGE = '#00FFFF'
    ERROR_MESSAGE = '#FF0000'
    SUCCESS_MESSAGE = '#00FF00'
    WARNING_MESSAGE = '#FFA500'
    
    # Component colors
    ORCHESTRATOR = '#FFD700'
    ROUTER = '#00FFFF'
    CLASSIFIER = '#FF69B4'
    TASK_EXTRACTOR = '#98FB98'
    CLARIFIER = '#FFA500'
    RESPONDER = '#87CEEB'
    
    # LLM detail colors
    LLM_START = '#00FFFF'
    LLM_END = '#00FF00'
    LLM_STREAM = '#FFFF00'
    CLASSIFICATION = '#FFD700'
    
    # Tool usage colors
    TOOL_SUCCESS = '#00FF00'
    TOOL_FAILURE = '#FF6B6B'
    TOOL_TIMING = '#FFD700'
    TOOL_CAPABILITY = '#00FFFF'
    TOOL_LABEL = '#FFA500'
    
    # UI element colors
    TIMESTAMP = '#808080'
    SEPARATOR = '#404040'
    BACKGROUND = '#1E1E1E'
    FOREGROUND = '#FFFFFF'


class UIConstants:
    """UI layout and sizing constants."""
    # Panel widths
    HISTORY_PANEL_WIDTH = 250
    CONVERSATION_PANEL_WIDTH = 650
    STATUS_PANEL_WIDTH = 400
    
    # Input field
    INPUT_FIELD_LINES = 4
    INPUT_FIELD_PADDING = 10
    
    # Timeouts
    STATUS_MESSAGE_TIMEOUT = 5000  # milliseconds
    QUEUED_MESSAGE_DELAY = 100  # milliseconds
    
    # Limits
    MAX_CONVERSATION_HISTORY = 1000
    MAX_STATUS_ENTRIES = 500


class EventTypes:
    """Event types for the event bus."""
    # Message events
    MESSAGE_RECEIVED = 'message_received'
    MESSAGE_SENT = 'message_sent'
    
    # Status events
    STATUS_UPDATE = 'status_update'
    ERROR_OCCURRED = 'error_occurred'
    PROCESSING_COMPLETE = 'processing_complete'
    
    # LLM events
    LLM_DETAIL = 'llm_detail'
    TOOL_USAGE = 'tool_usage'
    
    # Conversation events
    CONVERSATION_CREATED = 'conversation_created'
    CONVERSATION_LOADED = 'conversation_loaded'
    CONVERSATION_DELETED = 'conversation_deleted'
    CONVERSATION_UPDATED = 'conversation_updated'
    
    # Project events
    PROJECT_LOADED = 'project_loaded'
    PROJECT_SWITCHED = 'project_switched'
    
    # Settings events
    SETTINGS_CHANGED = 'settings_changed'
    MODEL_PREFERENCES_CHANGED = 'model_preferences_changed'


class InfrastructureSteps:
    """Infrastructure step names for model preferences."""
    ORCHESTRATOR = 'orchestrator'
    ROUTER = 'router'
    CLASSIFIER = 'classifier'
    TASK_EXTRACTOR = 'task_extractor'
    CLARIFIER = 'clarifier'
    RESPONDER = 'responder'
    
    @classmethod
    def all_steps(cls):
        """Get list of all infrastructure steps."""
        return [
            cls.ORCHESTRATOR,
            cls.ROUTER,
            cls.CLASSIFIER,
            cls.TASK_EXTRACTOR,
            cls.CLARIFIER,
            cls.RESPONDER
        ]