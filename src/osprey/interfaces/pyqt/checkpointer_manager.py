"""PostgreSQL Checkpointer Manager for Osprey Framework GUI.

This module handles checkpointer creation and PostgreSQL setup for conversation storage.
"""

import os
import socket

from langgraph.checkpoint.memory import MemorySaver
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox

from osprey.graph import create_async_postgres_checkpointer
from osprey.utils.logger import get_logger

logger = get_logger("checkpointer_manager")


class CheckpointerManager:
    """Manages checkpointer creation and PostgreSQL setup for conversation storage."""

    @staticmethod
    def create_checkpointer(settings_manager, parent_widget=None):
        """
        Create checkpointer based on settings.

        Args:
            settings_manager: SettingsManager instance with conversation storage settings
            parent_widget: Parent QWidget for showing dialogs (optional)

        Returns:
            Checkpointer instance (MemorySaver or PostgreSQL checkpointer)
        """
        storage_mode = settings_manager.get("conversation_storage_mode", "json")

        # If using JSON storage mode, use in-memory checkpointer (messages saved to JSON)
        if storage_mode == "json":
            logger.info("📝 Using JSON file storage for conversations (in-memory checkpointer)")
            logger.info("💡 Conversation messages will be saved to conversations.json")
            return MemorySaver()

        # If using PostgreSQL storage mode
        if storage_mode == "postgresql" and settings_manager.get(
            "use_persistent_conversations", True
        ):
            # Check if PostgreSQL URI is configured
            postgres_uri = os.getenv("POSTGRESQL_URI")

            if postgres_uri:
                try:
                    # Use PostgreSQL checkpointer if URI is configured
                    checkpointer = create_async_postgres_checkpointer(postgres_uri)
                    logger.info("✅ Using PostgreSQL checkpointer for persistent conversations")
                    return checkpointer
                except Exception as e:
                    logger.warning(f"⚠️  Failed to create PostgreSQL checkpointer: {e}")
                    CheckpointerManager.show_postgresql_setup_guidance(parent_widget)
                    logger.info("📝 Falling back to JSON storage mode")
                    settings_manager.update_from_dict({"conversation_storage_mode": "json"})
                    return MemorySaver()
            else:
                # Check if local PostgreSQL is running before attempting connection
                if CheckpointerManager.is_postgres_running():
                    try:
                        # Attempt to connect to local PostgreSQL
                        local_uri = "postgresql://postgres:postgres@localhost:5432/osprey"
                        checkpointer = create_async_postgres_checkpointer(local_uri)
                        logger.info(
                            "✅ Using local PostgreSQL checkpointer for persistent conversations"
                        )
                        logger.info(f"💡 Database: {local_uri}")
                        return checkpointer
                    except Exception as e:
                        # Fall back to JSON storage if connection fails
                        logger.warning(f"⚠️  PostgreSQL connection failed: {e}")
                        CheckpointerManager.show_postgresql_setup_guidance(parent_widget)
                        logger.info("📝 Falling back to JSON storage mode")
                        settings_manager.update_from_dict({"conversation_storage_mode": "json"})
                        return MemorySaver()
                else:
                    # PostgreSQL not running - show guidance and fall back to JSON
                    logger.warning("⚠️  PostgreSQL is not running")
                    CheckpointerManager.show_postgresql_setup_guidance(parent_widget)
                    logger.info("📝 Falling back to JSON storage mode")
                    settings_manager.update_from_dict({"conversation_storage_mode": "json"})
                    return MemorySaver()
        else:
            logger.info("📝 Using in-memory checkpointer (persistence disabled in settings)")
            return MemorySaver()

    @staticmethod
    def show_postgresql_setup_guidance(parent_widget=None):
        """
        Show guidance for setting up PostgreSQL for conversation storage.

        Args:
            parent_widget: Parent QWidget for showing dialog (optional)
        """
        logger.info("=" * 80)
        logger.info("PostgreSQL Setup Required")
        logger.info("=" * 80)
        logger.info("To use PostgreSQL for conversation storage, you need to:")
        logger.info("")
        logger.info("1. Install PostgreSQL:")
        logger.info("   • Ubuntu/Debian: sudo apt-get install postgresql")
        logger.info("   • macOS: brew install postgresql")
        logger.info("   • Windows: Download from https://www.postgresql.org/download/")
        logger.info("")
        logger.info("2. Start PostgreSQL service:")
        logger.info("   • Ubuntu/Debian: sudo systemctl start postgresql")
        logger.info("   • macOS: brew services start postgresql")
        logger.info("   • Windows: Start via Services or pg_ctl")
        logger.info("")
        logger.info("3. Create the 'osprey' database:")
        logger.info("   createdb osprey")
        logger.info("")
        logger.info("4. (Optional) Set custom connection via environment variable:")
        logger.info("   export POSTGRESQL_URI='postgresql://user:pass@host:port/dbname'")
        logger.info("")
        logger.info("5. Restart the GUI to use PostgreSQL storage")
        logger.info("=" * 80)

        # Also show a GUI dialog if parent widget is provided
        if parent_widget:
            QTimer.singleShot(
                1000, lambda: CheckpointerManager._show_postgresql_setup_dialog(parent_widget)
            )

    @staticmethod
    def _show_postgresql_setup_dialog(parent_widget):
        """
        Show a GUI dialog with PostgreSQL setup instructions.

        Args:
            parent_widget: Parent QWidget for the dialog
        """
        msg = QMessageBox(parent_widget)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("PostgreSQL Setup Required")
        msg.setText("PostgreSQL is not available for conversation storage.")
        msg.setInformativeText(
            "To use PostgreSQL for storing conversation messages:\n\n"
            "1. Install PostgreSQL on your system\n"
            "2. Start the PostgreSQL service\n"
            "3. Create the 'osprey' database: createdb osprey\n"
            "4. Restart the GUI\n\n"
            "For now, conversations will be saved to JSON files.\n"
            "See the System Information tab for detailed instructions."
        )
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    @staticmethod
    def is_postgres_running(host="localhost", port=5432, timeout=1):
        """
        Check if PostgreSQL is running by attempting a socket connection.

        Args:
            host: PostgreSQL host (default: localhost)
            port: PostgreSQL port (default: 5432)
            timeout: Connection timeout in seconds (default: 1)

        Returns:
            bool: True if PostgreSQL is running, False otherwise
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    @staticmethod
    def acquire_conversation_lock(db_path):
        """
        Acquire a lock file to prevent conflicts with other GUI instances.

        Args:
            db_path: Path to the database file

        Returns:
            File handle for the lock file, or None if locking failed
        """
        try:
            import fcntl
        except ImportError:
            # Windows doesn't have fcntl, skip locking
            logger.debug("File locking not available on this platform")
            return None

        lock_file_path = db_path.parent / f".{db_path.name}.lock"
        try:
            lock_file = open(lock_file_path, "w")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_file.write(f"{os.getpid()}\n")
            lock_file.flush()
            logger.debug(f"Acquired conversation lock: {lock_file_path}")
            return lock_file
        except OSError as e:
            logger.warning(
                f"Could not acquire exclusive lock (another GUI instance may be running): {e}"
            )
            # Continue anyway - PostgreSQL handles concurrent access
            if lock_file:
                lock_file.close()
            return None

    @staticmethod
    def release_conversation_lock(lock_file):
        """
        Release the conversation lock file.

        Args:
            lock_file: File handle for the lock file
        """
        if lock_file:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                logger.debug("Released conversation lock")
            except Exception as e:
                logger.warning(f"Error releasing conversation lock: {e}")
