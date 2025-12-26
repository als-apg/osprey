"""
Conversation History Management for Osprey PyQt GUI

This module handles conversation history persistence:
- Saving conversation history to JSON
- Loading conversation history from JSON
- Loading conversation list from checkpointer
- Managing conversation storage modes
"""

from datetime import datetime
from osprey.utils.logger import get_logger
from osprey.interfaces.pyqt.gui_utils import get_gui_data_dir

logger = get_logger("conversation_history")


class ConversationHistory:
    """Manages conversation history persistence for the GUI."""

    def __init__(self, gui):
        """
        Initialize the conversation history manager.

        Args:
            gui: Reference to the main OspreyGUI instance
        """
        self.gui = gui

    def save_conversation_history(self):
        """Save conversation metadata and optionally messages to persistent storage."""
        storage_mode = self.gui.settings_manager.get('conversation_storage_mode', 'json')

        # Only save to JSON file if using JSON storage mode
        if storage_mode != 'json':
            logger.debug(f"Skipping JSON save - using {storage_mode} storage mode (messages stored in database)")
            return

        try:
            # Save to GUI-specific conversations file (not project-specific)
            # Use framework-relative directory to store GUI conversations
            gui_conversations_dir = get_gui_data_dir() / 'conversations'
            gui_conversations_dir.mkdir(parents=True, exist_ok=True)
            conversations_file = gui_conversations_dir / 'conversations.json'

            # Use ConversationManager's save method
            if self.gui.conversation_manager.save_to_json(conversations_file):
                logger.debug(f"Saved conversation data with messages to {conversations_file}")
            else:
                logger.warning("Failed to save conversation data")
        except Exception as e:
            logger.warning(f"Failed to save conversation data: {e}")

    def load_conversation_history(self):
        """Load conversation metadata and optionally messages from persistent storage."""
        storage_mode = self.gui.settings_manager.get('conversation_storage_mode', 'json')

        # Only load from JSON file if using JSON storage mode
        if storage_mode != 'json':
            logger.debug(f"Skipping JSON load - using {storage_mode} storage mode (messages stored in database)")
            return

        try:
            # Load from GUI-specific conversations file (not project-specific)
            gui_conversations_dir = get_gui_data_dir() / 'conversations'
            conversations_file = gui_conversations_dir / 'conversations.json'

            logger.info(f"Loading conversations from: {conversations_file}")

            # Use ConversationManager's load method
            if self.gui.conversation_manager.load_from_json(conversations_file):
                logger.info(f"✅ Loaded {len(self.gui.conversation_manager.conversations)} conversation(s) with messages from JSON file")
                # Debug: Log message counts for each conversation
                for thread_id, conv in self.gui.conversation_manager.conversations.items():
                    logger.info(f"  📂 Conversation '{conv.name}' ({thread_id}): {len(conv.messages)} messages")
                    for i, msg in enumerate(conv.messages):
                        logger.debug(f"    Message {i+1}: type={msg.type}, content_length={len(msg.content)}")
            else:
                logger.info(f"No conversation history file found at {conversations_file}")

        except Exception as e:
            logger.warning(f"Failed to load conversation history: {e}")

    def load_conversation_list(self):
        """Load list of conversations from checkpointer."""
        try:
            if not self.gui.graph or not hasattr(self.gui.graph, 'checkpointer'):
                logger.debug("No checkpointer available for loading conversations")
                return

            checkpointer = self.gui.graph.checkpointer

            # Try to get all thread IDs from the checkpointer
            # Different checkpointer types have different methods
            thread_ids = set()

            # For MemorySaver checkpointer
            if hasattr(checkpointer, 'storage') and isinstance(checkpointer.storage, dict):
                # MemorySaver stores data as {(thread_id, checkpoint_ns): checkpoint}
                for key in checkpointer.storage.keys():
                    if isinstance(key, tuple) and len(key) >= 1:
                        thread_ids.add(key[0])

            # For PostgreSQL checkpointer (AsyncPostgresSaver)
            elif hasattr(checkpointer, 'conn'):
                # PostgreSQL checkpointer - we'd need to query the database
                # This is more complex and would require async operations
                logger.info("PostgreSQL checkpointer detected - loading conversations from database")
                # For now, we'll skip this and rely on on-demand loading
                return

            # Load conversations for each thread ID found
            loaded_count = 0
            for thread_id in thread_ids:
                try:
                    # Create config for this thread
                    config = {
                        "configurable": {
                            **self.gui.base_config["configurable"],
                            "thread_id": thread_id,
                            "session_id": thread_id
                        },
                        "recursion_limit": self.gui.base_config.get("recursion_limit", 100)
                    }

                    # Get state from checkpointer
                    state = self.gui.graph.get_state(config=config)

                    if state and state.values:
                        messages = state.values.get('messages', [])

                        if messages and len(messages) > 0:
                            # Create conversation entry
                            # Try to extract a meaningful name from the first user message
                            first_user_msg = None
                            for msg in messages:
                                if hasattr(msg, 'type') and msg.type == 'human':
                                    first_user_msg = msg.content[:50] if hasattr(msg, 'content') else None
                                    break

                            conv_name = first_user_msg if first_user_msg else f"Conversation {len(self.gui.conversations) + 1}"

                            # Get timestamp from state metadata if available
                            timestamp = datetime.now()
                            if hasattr(state, 'created_at') and state.created_at:
                                try:
                                    timestamp = datetime.fromisoformat(state.created_at)
                                except:
                                    pass

                            # Add to conversations dict
                            self.gui.conversations[thread_id] = {
                                'name': conv_name,
                                'messages': [],  # We'll load these on-demand
                                'timestamp': timestamp,
                                'thread_id': thread_id
                            }
                            loaded_count += 1

                except Exception as e:
                    logger.warning(f"Failed to load conversation {thread_id}: {e}")
                    continue

            if loaded_count > 0:
                logger.info(f"Loaded {loaded_count} conversation(s) from checkpointer")
                self.gui.conversation_display_mgr.update_conversation_list()
            else:
                logger.info("No existing conversations found in checkpointer")

        except Exception as e:
            logger.error(f"Failed to load conversation list: {e}")
