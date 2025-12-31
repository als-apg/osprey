"""
Conversation Display Manager for Osprey PyQt GUI

This module handles all conversation display operations including:
- Loading and displaying conversation messages
- Switching between conversations
- Managing conversation UI updates
- Handling different storage modes (JSON/PostgreSQL)
"""

from PyQt5.QtGui import QColor

from osprey.utils.logger import get_logger

logger = get_logger("conversation_display")


class ConversationDisplayManager:
    """Manages conversation display operations for the GUI."""

    def __init__(self, gui):
        """
        Initialize the conversation display manager.

        Args:
            gui: Reference to the main OspreyGUI instance
        """
        self.gui = gui

    def load_and_display_conversation(
        self, thread_id: str, show_header: bool = True, clear_display: bool = False
    ):
        """
        Unified method to load and display conversation messages.

        Args:
            thread_id: Thread ID of the conversation to load
            show_header: Whether to show conversation header
            clear_display: Whether to clear display before loading
        """
        try:
            if clear_display:
                self.gui.conversation_display.clear()

            conv = self.gui.conversation_manager.get_conversation(thread_id)
            if not conv:
                logger.warning(f"Conversation {thread_id} not found")
                return

            # Show header if requested
            if show_header:
                self.gui._append_colored_message(
                    "=" * 80 + "\n" + f"📂 {conv.name}\n" + "=" * 80 + "\n", "#00FFFF"
                )

            # Load messages based on storage mode
            storage_mode = self.gui.settings_manager.get("conversation_storage_mode", "json")

            if storage_mode == "json" and conv.messages:
                # Load from JSON storage (in-memory)
                logger.info(f"Loading {len(conv.messages)} messages from JSON storage")
                for msg in conv.messages:
                    if msg.type == "user":
                        self.gui._append_colored_message(
                            f"👤 You: {msg.content}", "#D8BFD8", auto_open_plots=False
                        )
                    else:
                        # Check if message has special formatting
                        if msg.formatting == "orchestrated":
                            # Apply orchestrated formatting
                            self.gui._append_colored_message(
                                "🤖 Combined Answer:", "#00FF00", auto_open_plots=False
                            )
                            self.gui._append_colored_message(
                                "", "#FFFFFF", auto_open_plots=False
                            )  # Empty line
                            self.gui.orchestration_ui._display_formatted_result(msg.content)
                        else:
                            # Regular message - DON'T auto-open plots for historical messages
                            self.gui._append_colored_message(
                                f"🤖 {msg.content}", "#FFFFFF", auto_open_plots=False
                            )
            elif (
                storage_mode == "postgresql"
                and self.gui.settings_manager.get("use_persistent_conversations", True)
                and self.gui.graph
            ):
                # Load from PostgreSQL checkpointer
                logger.info("Loading messages from PostgreSQL checkpointer")
                self.load_from_checkpointer(thread_id)
            else:
                self.gui._append_colored_message(
                    "Welcome! Start a conversation by typing a message below.", "#00FFFF"
                )

        except Exception as e:
            logger.error(f"Failed to load conversation display: {e}")
            self.gui._append_colored_message(f"⚠️ Failed to load conversation: {e}", "#FFA500")

    def load_from_checkpointer(self, thread_id: str):
        """
        Load messages from checkpointer and display them.

        Args:
            thread_id: Thread ID of the conversation to load
        """
        try:
            # Create a config with the specific thread_id
            config = {
                "configurable": {
                    **self.gui.base_config["configurable"],
                    "thread_id": thread_id,
                    "session_id": thread_id,
                },
                "recursion_limit": self.gui.base_config.get("recursion_limit", 100),
            }

            # Get state from checkpointer
            state = self.gui.graph.get_state(config=config)

            if state and state.values:
                messages = state.values.get("messages", [])

                if messages:
                    message_count = 0
                    for msg in messages:
                        if hasattr(msg, "content") and msg.content:
                            if hasattr(msg, "type") and msg.type == "human":
                                self.gui._append_colored_message(
                                    f"👤 You: {msg.content}", "#D8BFD8", auto_open_plots=False
                                )
                                message_count += 1
                            else:
                                self.gui._append_colored_message(
                                    f"🤖 {msg.content}", "#FFFFFF", auto_open_plots=False
                                )
                                message_count += 1

                    logger.info(f"Loaded {message_count} messages from checkpointer")
                    self.gui.add_status(f"✅ Loaded {message_count} messages from database", "base")
                else:
                    self.gui._append_colored_message(
                        "No messages in this conversation yet.", "#808080"
                    )
            else:
                self.gui._append_colored_message("No messages in this conversation yet.", "#808080")

        except Exception as e:
            logger.error(f"Failed to load from checkpointer: {e}")
            self.gui._append_colored_message(
                f"⚠️ Could not load conversation history: {e}", "#FFA500"
            )
            self.gui.add_status(f"❌ Failed to load from database: {e}", "error")

    def load_from_memory(self, messages: list):
        """
        Load messages from in-memory storage and display them.

        Args:
            messages: List of message dictionaries with 'type' and 'content' keys
        """
        try:
            for msg in messages:
                if msg["type"] == "user":
                    self.gui._append_colored_message(
                        f"👤 You: {msg['content']}", "#D8BFD8", auto_open_plots=False
                    )
                else:
                    self.gui._append_colored_message(
                        msg["content"], "#FFFFFF", auto_open_plots=False
                    )

            logger.debug(f"Loaded {len(messages)} messages from memory")
        except Exception as e:
            logger.error(f"Failed to load from memory: {e}")
            self.gui._append_colored_message(f"⚠️ Could not load messages: {e}", "#FFA500")

    def load_current_conversation_display(self):
        """Load the current conversation messages into the display."""
        if not self.gui.current_conversation_id:
            return

        conv = self.gui.conversation_manager.get_conversation(self.gui.current_conversation_id)
        if not conv:
            return

        # Use unified loading method
        self.load_and_display_conversation(
            self.gui.current_conversation_id, show_header=True, clear_display=True
        )

    def update_conversation_list(self):
        """Update the conversation history list."""
        self.gui.conversation_list.clear()

        # Use ConversationManager to get sorted conversations
        sorted_convs = self.gui.conversation_manager.list_conversations(sort_by_timestamp=True)

        for conv in sorted_convs:
            thread_id = conv.thread_id
            name = conv.name
            timestamp = conv.timestamp.strftime("%Y-%m-%d %H:%M")
            msg_count = len(conv.messages)

            is_current = thread_id == self.gui.current_conversation_id

            prefix = "▶ " if is_current else "  "
            item_text = f"{prefix}{name}\n   {timestamp} • {msg_count} messages"

            from PyQt5.QtCore import Qt
            from PyQt5.QtWidgets import QListWidgetItem

            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, thread_id)

            if is_current:
                item.setForeground(QColor("#00FF00"))
            else:
                item.setForeground(QColor("#FFD700"))

            self.gui.conversation_list.addItem(item)

    def switch_conversation(self, item):
        """
        Switch to a different conversation and reload all messages.

        Args:
            item: QListWidgetItem containing the conversation thread_id
        """
        from PyQt5.QtCore import Qt

        thread_id = item.data(Qt.UserRole)

        # Use ConversationManager to switch conversation
        if not self.gui.conversation_manager.switch_conversation(thread_id):
            return

        try:
            # Update thread ID and config FIRST before loading messages
            self.gui.current_conversation_id = thread_id
            self.gui.thread_id = thread_id

            if self.gui.base_config:
                self.gui.base_config["configurable"]["thread_id"] = self.gui.thread_id
                self.gui.base_config["configurable"]["session_id"] = self.gui.thread_id

            # Clear display
            self.gui.conversation_display.clear()

            # Get conversation from manager
            conv = self.gui.conversation_manager.get_conversation(thread_id)
            if not conv:
                return

            self.gui._append_colored_message(
                "=" * 80
                + "\n"
                + f"📂 LOADED CONVERSATION: {conv.name}\n"
                + f"   Created: {conv.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                + "=" * 80
                + "\n",
                "#00FFFF",
            )

            # Load and display all messages from this conversation
            storage_mode = self.gui.settings_manager.get("conversation_storage_mode", "json")

            if storage_mode == "json" and conv.messages:
                # Load from JSON storage (in-memory)
                message_count = len(conv.messages)
                self.gui.add_status(
                    f"Loading {message_count} messages from JSON storage...", "base"
                )

                for msg in conv.messages:
                    if msg.type == "user":
                        self.gui._append_colored_message(
                            f"👤 You: {msg.content}", "#D8BFD8", auto_open_plots=False
                        )
                    else:
                        # Check if message has special formatting
                        if msg.formatting == "orchestrated":
                            # Apply orchestrated formatting
                            self.gui._append_colored_message(
                                "🤖 Combined Answer:", "#00FF00", auto_open_plots=False
                            )
                            self.gui._append_colored_message(
                                "", "#FFFFFF", auto_open_plots=False
                            )  # Empty line
                            self.gui.orchestration_ui._display_formatted_result(msg.content)
                        else:
                            # Regular message - DON'T auto-open plots for historical messages
                            self.gui._append_colored_message(
                                f"🤖 {msg.content}", "#FFFFFF", auto_open_plots=False
                            )

                self.gui.add_status(f"✅ Loaded {message_count} messages", "base")

            elif (
                storage_mode == "postgresql"
                and self.gui.settings_manager.get("use_persistent_conversations", True)
                and self.gui.graph
            ):
                # Load from PostgreSQL checkpointer
                self.gui.add_status("Loading messages from PostgreSQL...", "base")
                self.load_from_checkpointer(thread_id)

            else:
                # No messages to load
                self.gui._append_colored_message(
                    "No messages in this conversation yet. Start chatting below!", "#808080"
                )

            self.update_conversation_list()
            self.gui.update_session_info()

            self.gui.add_status(f"Switched to conversation: {conv.name}", "base")

        except Exception as e:
            logger.exception(f"Error switching conversation: {e}")

    def display_parallel_execution_start(self, step_count: int, capabilities: list):
        """
        Display visual indicator when parallel execution starts.

        Args:
            step_count: Number of steps executing in parallel
            capabilities: List of capability names being executed
        """
        from PyQt5.QtWidgets import QApplication

        self.gui._append_colored_message(
            f"\n⚡ Parallel Execution: {step_count} steps running simultaneously", "#00FFFF"
        )
        for i, cap in enumerate(capabilities, 1):
            self.gui._append_colored_message(f"   {i}. {cap} ⏳", "#FFD700")
        QApplication.processEvents()

    def display_parallel_execution_complete(self, step_count: int, duration: float):
        """
        Display completion indicator for parallel execution.

        Args:
            step_count: Number of steps that completed
            duration: Time taken in seconds
        """
        from PyQt5.QtWidgets import QApplication

        speedup = step_count / max(duration, 0.1) if duration > 0 else 1
        self.gui._append_colored_message(
            f"✅ Parallel execution complete: {step_count} steps in {duration:.2f}s "
            f"(~{speedup:.1f}× speedup)",
            "#00FF00",
        )
        QApplication.processEvents()

    def display_execution_mode(self, is_parallel: bool):
        """
        Display current execution mode indicator.

        Args:
            is_parallel: Whether parallel execution is enabled
        """
        if is_parallel:
            self.gui._append_colored_message("⚡ Parallel Execution Mode: ON", "#00FFFF")
        else:
            self.gui._append_colored_message("🔄 Sequential Execution Mode", "#808080")
