"""
Event Bus Pattern for Osprey PyQt GUI

This module provides a centralized event bus for decoupling GUI components.
Instead of components holding direct references to each other, they communicate
through events, reducing circular dependencies and improving testability.

Usage:
    # Subscribe to events
    event_bus.subscribe('message_received', handler.on_message)
    
    # Publish events
    event_bus.publish('message_received', {'content': 'Hello'})
    
    # Unsubscribe
    event_bus.unsubscribe('message_received', handler.on_message)
"""

from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict
from osprey.utils.logger import get_logger

logger = get_logger("event_bus")


class EventBus:
    """Centralized event bus for decoupled component communication."""
    
    def __init__(self):
        """Initialize the event bus."""
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_log: List[Dict[str, Any]] = []
        self._logging_enabled = False
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """
        Subscribe a handler to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Callable to invoke when event is published
        """
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)
            logger.debug(f"Subscribed {handler.__name__} to '{event_type}'")
    
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """
        Unsubscribe a handler from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler to remove
        """
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            logger.debug(f"Unsubscribed {handler.__name__} from '{event_type}'")
    
    def publish(self, event_type: str, data: Any = None) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event_type: Type of event to publish
            data: Optional data to pass to handlers
        """
        if self._logging_enabled:
            self._event_log.append({
                'type': event_type,
                'data': data
            })
        
        handlers = self._subscribers.get(event_type, [])
        logger.debug(f"Publishing '{event_type}' to {len(handlers)} handler(s)")
        
        for handler in handlers:
            try:
                # Call handler with data if provided
                if data is not None:
                    handler(data)
                else:
                    handler()
            except Exception as e:
                logger.exception(f"Error in event handler {handler.__name__} for '{event_type}': {e}")
    
    def clear_subscribers(self, event_type: Optional[str] = None) -> None:
        """
        Clear all subscribers for an event type, or all subscribers if no type specified.
        
        Args:
            event_type: Optional event type to clear. If None, clears all.
        """
        if event_type:
            self._subscribers[event_type].clear()
            logger.debug(f"Cleared all subscribers for '{event_type}'")
        else:
            self._subscribers.clear()
            logger.debug("Cleared all event subscribers")
    
    def enable_logging(self, enabled: bool = True) -> None:
        """
        Enable or disable event logging for debugging.
        
        Args:
            enabled: Whether to enable event logging
        """
        self._logging_enabled = enabled
        if not enabled:
            self._event_log.clear()
    
    def get_event_log(self) -> List[Dict[str, Any]]:
        """
        Get the event log (only if logging is enabled).
        
        Returns:
            List of logged events
        """
        return self._event_log.copy()
    
    def get_subscriber_count(self, event_type: str) -> int:
        """
        Get the number of subscribers for an event type.
        
        Args:
            event_type: Event type to check
            
        Returns:
            Number of subscribers
        """
        return len(self._subscribers.get(event_type, []))