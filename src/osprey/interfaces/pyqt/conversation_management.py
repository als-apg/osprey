"""
Conversation Management for Osprey PyQt GUI

This module handles conversation lifecycle operations:
- Creating new conversations
- Deleting conversations
- Renaming conversations
- Clearing conversation history
- Managing conversation storage
"""

import asyncio
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QInputDialog, QMessageBox

from osprey.utils.logger import get_logger

logger = get_logger("conversation_management")


class ConversationManagement:
    """Manages conversation lifecycle operations for the GUI."""

    def __init__(self, gui):
        """
        Initialize the conversation management.

        Args:
            gui: Reference to the main OspreyGUI instance
        """
        self.gui = gui

    def clear_conversation(self):
        """Clear the conversation display and history."""
        reply = QMessageBox.question(
            self.gui,
            "Clear Conversation",
            "Are you sure you want to clear the conversation history?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.gui.conversation_display.clear()

            if self.gui.current_conversation_id:
                # Clear messages using ConversationManager
                conv = self.gui.conversation_manager.get_conversation(
                    self.gui.current_conversation_id
                )
                if conv:
                    conv.messages = []
                    conv.timestamp = datetime.now()
                    self.gui.save_conversation_history()
                    self.gui.conversation_display_mgr.update_conversation_list()

            self.gui.add_status("Conversation history cleared", "base")

    def start_new_conversation(self):
        """Start a new conversation."""
        self.create_new_conversation()

    def create_new_conversation(self):
        """Create a new conversation."""
        try:
            # Use ConversationManager to create new conversation
            self.gui.thread_id = self.gui.conversation_manager.create_conversation()
            self.gui.current_conversation_id = self.gui.thread_id

            self.gui.save_conversation_history()

            if self.gui.base_config:
                self.gui.base_config["configurable"]["thread_id"] = self.gui.thread_id
                self.gui.base_config["configurable"]["session_id"] = self.gui.thread_id

            self.gui.conversation_display.clear()

            self.gui._append_colored_message(
                "=" * 80 + "\n" + "🔄 NEW CONVERSATION STARTED\n" + "=" * 80 + "\n", "#00FFFF"
            )

            self.gui.add_status(f"New conversation started (Thread: {self.gui.thread_id})", "base")
            self.gui.conversation_display_mgr.update_conversation_list()
            self.gui.update_session_info()
            self.gui.conversation_display_mgr.load_current_conversation_display()
            self.gui.input_field.setFocus()

        except Exception as e:
            logger.exception(f"Error starting new conversation: {e}")
            self.gui.add_status(f"❌ Failed to start new conversation: {e}", "error")
            QMessageBox.warning(self.gui, "Error", f"Failed to start new conversation:\n{e}")

    def delete_selected_conversation(self):
        """Delete the currently selected conversation(s)."""
        selected_items = self.gui.conversation_list.selectedItems()
        if not selected_items:
            QMessageBox.information(
                self.gui, "No Selection", "Please select one or more conversations to delete."
            )
            return

        # Get thread IDs and names of selected conversations
        selected_convs = []
        for item in selected_items:
            thread_id = item.data(Qt.UserRole)
            conv = self.gui.conversation_manager.get_conversation(thread_id)
            if conv:
                selected_convs.append({"thread_id": thread_id, "name": conv.name})

        if not selected_convs:
            return

        # Check if trying to delete all conversations
        if len(selected_convs) == len(self.gui.conversation_manager.conversations):
            QMessageBox.warning(
                self.gui,
                "Cannot Delete",
                "Cannot delete all conversations. At least one must remain.",
            )
            return

        # Build confirmation message
        if len(selected_convs) == 1:
            message = f"Are you sure you want to delete '{selected_convs[0]['name']}'?"
        else:
            conv_names = "\n  • ".join([conv["name"] for conv in selected_convs])
            message = f"Are you sure you want to delete {len(selected_convs)} conversations?\n\n  • {conv_names}"

        reply = QMessageBox.question(
            self.gui,
            "Delete Conversation(s)",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Check if current conversation is being deleted
            current_being_deleted = self.gui.current_conversation_id in [
                conv["thread_id"] for conv in selected_convs
            ]

            if current_being_deleted:
                # Switch to a conversation that's not being deleted
                for other_id in self.gui.conversation_manager.conversations:
                    if other_id not in [conv["thread_id"] for conv in selected_convs]:
                        for i in range(self.gui.conversation_list.count()):
                            item = self.gui.conversation_list.item(i)
                            if item.data(Qt.UserRole) == other_id:
                                self.gui.switch_conversation(item)
                                break
                        break

            # Delete all selected conversations using ConversationManager
            deleted_names = []
            for conv in selected_convs:
                thread_id = conv["thread_id"]

                # Delete using ConversationManager
                if self.gui.conversation_manager.delete_conversation(thread_id):
                    deleted_names.append(conv["name"])

                # Delete from persistent storage (database or JSON)
                self._delete_conversation_from_storage(thread_id)

            self.gui.conversation_display_mgr.update_conversation_list()

            # Log deletion
            if len(deleted_names) == 1:
                self.gui.add_status(f"Deleted conversation: {deleted_names[0]}", "base")
            else:
                self.gui.add_status(f"Deleted {len(deleted_names)} conversations", "base")

    def _delete_conversation_from_storage(self, thread_id: str):
        """
        Delete a conversation from persistent storage.

        Args:
            thread_id: Thread ID of the conversation to delete
        """
        storage_mode = self.gui.settings_manager.get("conversation_storage_mode", "json")

        try:
            if storage_mode == "json":
                # For JSON storage, just save the updated conversations dict
                # (the conversation was already removed from self.conversations)
                self.gui.save_conversation_history()
                logger.debug(f"Deleted conversation {thread_id} from JSON storage")
            elif storage_mode == "postgresql":
                # For PostgreSQL storage, delete from the database checkpointer
                if self.gui.graph and hasattr(self.gui.graph, "checkpointer"):
                    checkpointer = self.gui.graph.checkpointer

                    # Check if checkpointer has a delete method
                    if hasattr(checkpointer, "delete"):
                        # Use the checkpointer's delete method if available
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            # Create config for this thread
                            config = {
                                "configurable": {"thread_id": thread_id, "session_id": thread_id}
                            }
                            # Delete the checkpoint
                            loop.run_until_complete(checkpointer.delete(config))
                            logger.info(
                                f"Deleted conversation {thread_id} from PostgreSQL database"
                            )
                        finally:
                            loop.close()
                    else:
                        # Checkpointer doesn't have delete method - manual cleanup needed
                        logger.warning(
                            f"PostgreSQL checkpointer doesn't support deletion. "
                            f"Conversation {thread_id} removed from GUI but may remain in database. "
                            f"Manual cleanup may be required."
                        )
                else:
                    logger.warning(
                        f"No checkpointer available for PostgreSQL deletion of {thread_id}"
                    )
        except Exception as e:
            logger.error(f"Failed to delete conversation from storage: {e}")
            # Don't raise - the conversation is already removed from memory
            # Show warning to user
            QMessageBox.warning(
                self.gui,
                "Deletion Warning",
                f"Conversation removed from GUI but may not be fully deleted from database:\n{e}\n\n"
                f"The conversation will not appear in the GUI, but database cleanup may be needed.",
            )

    def rename_selected_conversation(self):
        """Rename the currently selected conversation."""
        current_item = self.gui.conversation_list.currentItem()
        if not current_item:
            QMessageBox.information(
                self.gui, "No Selection", "Please select a conversation to rename."
            )
            return

        thread_id = current_item.data(Qt.UserRole)
        conv = self.gui.conversation_manager.get_conversation(thread_id)
        if not conv:
            return

        old_name = conv.name
        new_name, ok = QInputDialog.getText(
            self.gui, "Rename Conversation", "Enter new name:", text=old_name
        )

        if ok and new_name.strip():
            # Use ConversationManager to rename
            if self.gui.conversation_manager.rename_conversation(thread_id, new_name):
                self.gui.conversation_display_mgr.update_conversation_list()
                self.gui.save_conversation_history()
                self.gui.add_status(f"Renamed conversation: '{old_name}' → '{new_name}'", "base")
