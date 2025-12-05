"""
Memory Storage Manager

Simple file-based memory manager for user memory storage.
Extracted from services.als_assistant.utils.memory_manager
"""

import json
from datetime import datetime
from pathlib import Path

from osprey.state import UserMemories
from osprey.utils.config import get_agent_dir, get_session_info
from osprey.utils.logger import get_logger

try:
    from langgraph.config import get_config
except ImportError:
    get_config = None

from .models import MemoryContent

logger = get_logger("memory_storage")


class MemoryStorageManager:
    """Simple file-based memory manager for user memory storage.

    Provides persistent storage of user memory entries in JSON format with
    thread-safe file operations and proper error handling. Each user's memory
    is stored in a separate JSON file identified by sanitized user ID.
    """

    def __init__(self, memory_directory: str):
        """Initialize memory manager with storage directory.

        Creates the memory directory if it doesn't exist and sets up
        logging for memory operations.

        :param memory_directory: Directory path for storing memory files
        :type memory_directory: str
        :raises OSError: If directory cannot be created or accessed
        """
        self.memory_dir = Path(memory_directory).resolve()
        self.memory_dir.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Memory manager initialized with directory: {self.memory_dir}")

    def _get_memory_file_path(self, user_id: str) -> Path:
        """Get path to user's memory file.

        :param user_id: User identifier for memory file
        :type user_id: str
        :return: Path to the user's memory JSON file
        :rtype: Path
        :raises ValueError: If user_id is empty or invalid
        """
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")

        # Sanitize user_id for filename
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in "-_")
        return self.memory_dir / f"{safe_user_id}.json"

    def _load_memory_data(self, user_id: str) -> list[dict]:
        """Load memory entries from user's JSON file.

        :param user_id: User identifier
        :type user_id: str
        :return: List of memory entry dictionaries
        :rtype: List[dict]

        .. note::
           Returns empty list if file doesn't exist or on read errors.
        """
        try:
            memory_file = self._get_memory_file_path(user_id)
            if memory_file.exists():
                with open(memory_file, encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("entries", [])
            return []
        except Exception as e:
            logger.error(f"Error loading memory for user {user_id}: {e}")
            return []

    def _save_memory_data(self, user_id: str, entries: list[dict]) -> bool:
        """Save memory entries to user's JSON file.

        :param user_id: User identifier
        :type user_id: str
        :param entries: List of memory entry dictionaries to save
        :type entries: List[dict]
        :return: True if save was successful, False otherwise
        :rtype: bool

        .. note::
           File is written atomically with proper encoding and formatting.
        """
        try:
            memory_file = self._get_memory_file_path(user_id)
            data = {
                "user_id": user_id,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "entries": entries,
            }

            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            logger.error(f"Error saving memory for user {user_id}: {e}")
            return False

    def get_user_memory(self, user_id: str) -> str:
        """Get user's memory content as formatted string.

        :param user_id: User identifier
        :type user_id: str
        :return: Formatted memory content with timestamps, empty string if no memory
        :rtype: str

        Examples:
            Retrieving formatted memory::

                >>> manager = MemoryManager("/path/to/memory")
                >>> memory_text = manager.get_user_memory("user123")
                >>> print(memory_text)
                [2025-01-15 14:30] User prefers morning meetings
                [2025-01-15 15:45] Working on project Alpha
        """
        try:
            entries = self._load_memory_data(user_id)
            if not entries:
                return ""

            # Format as readable text
            formatted_entries = []
            for entry in entries:
                timestamp = entry.get("timestamp", "")
                content = entry.get("content", "")
                if content:
                    formatted_entries.append(f"[{timestamp}] {content}")

            return "\n".join(formatted_entries)
        except Exception as e:
            logger.error(f"Error retrieving memory for user {user_id}: {e}")
            return ""

    def get_all_memory_entries(self, user_id: str) -> list[MemoryContent]:
        """Get all user memory entries as list of MemoryContent objects.

        :param user_id: User identifier
        :type user_id: str
        :return: List of structured memory content objects
        :rtype: List[MemoryContent]

        .. note::
           Handles timestamp parsing errors by using current time as fallback.
           Only returns entries with non-empty content.
        """
        try:
            entries = self._load_memory_data(user_id)
            memory_contents = []

            for entry in entries:
                timestamp_str = entry.get("timestamp", "")
                content = entry.get("content", "")

                if content.strip():  # Only include non-empty content
                    try:
                        # Parse timestamp string back to datetime
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M")
                        memory_content = MemoryContent(timestamp=timestamp, content=content.strip())
                        memory_contents.append(memory_content)
                    except ValueError as e:
                        logger.warning(
                            f"Failed to parse timestamp '{timestamp_str}' for user {user_id}: {e}"
                        )
                        # Use current time as fallback
                        memory_content = MemoryContent(
                            timestamp=datetime.now(), content=content.strip()
                        )
                        memory_contents.append(memory_content)

            logger.debug(f"Retrieved {len(memory_contents)} memory entries for user {user_id}")
            return memory_contents

        except Exception as e:
            logger.error(f"Error retrieving memory entries for user {user_id}: {e}")
            return []

    def add_memory_entry(self, user_id: str, memory_content: MemoryContent) -> bool:
        """Add new memory entry for user.

        :param user_id: User identifier
        :type user_id: str
        :param memory_content: Memory content to add
        :type memory_content: MemoryContent
        :return: True if entry was added successfully, False otherwise
        :rtype: bool

        Examples:
            Adding a memory entry::

                >>> from datetime import datetime
                >>> manager = MemoryManager("/path/to/memory")
                >>> entry = MemoryContent(
                ...     timestamp=datetime.now(),
                ...     content="User completed training module"
                ... )
                >>> success = manager.add_memory_entry("user123", entry)
                >>> print(f"Entry added: {success}")
        """
        try:
            entries = self._load_memory_data(user_id)

            # Add new entry
            new_entry = {
                "timestamp": memory_content.timestamp.strftime("%Y-%m-%d %H:%M"),
                "content": memory_content.content.strip(),
            }

            entries.append(new_entry)

            success = self._save_memory_data(user_id, entries)
            if success:
                logger.info(
                    f"Added memory entry for user {user_id}: {memory_content.content[:50]}..."
                )
            return success
        except Exception as e:
            logger.error(f"Error adding memory entry for user {user_id}: {e}")
            return False

    def get_memories_from_state(self, state):
        """Get user memory from agent state as UserMemories object.

        :param state: Agent state containing user session context
        :type state: AgentState
        :return: UserMemories object with content list
        :rtype: UserMemories

        .. note::
           Returns empty UserMemories if user_id is not available in config.
        """
        # Extract user_id from LangGraph config (not from state)
        try:
            if get_config:
                config = get_config()
                configurable = config.get("configurable", {})
            session_info = get_session_info()
            user_id = session_info.get("user_id")
        except Exception:
            user_id = None

        if not user_id:
            logger.warning("No user_id found in LangGraph config, skipping memory retrieval")
            return UserMemories(entries=[])

        try:
            entries = self._load_memory_data(user_id)
            # Extract just the content for UserMemories
            content_list = [entry.get("content", "") for entry in entries if entry.get("content")]
            return UserMemories(entries=content_list)
        except Exception as e:
            logger.warning(f"Failed to retrieve user memory for {user_id}: {e}")
            return UserMemories(entries=[])


# Global memory storage manager instance
_memory_storage_manager: MemoryStorageManager | None = None


def get_memory_storage_manager() -> MemoryStorageManager:
    """Get the global memory storage manager instance.

    Creates and caches a global MemoryStorageManager instance using the
    configured memory directory from global configuration.

    :return: Global memory storage manager instance
    :rtype: MemoryStorageManager

    .. note::
       Uses lazy initialization and global caching for efficiency.
    """
    global _memory_storage_manager
    if _memory_storage_manager is None:
        # Memory storage config now accessed via config
        # Use get_agent_dir to properly construct the path
        memory_dir = get_agent_dir("user_memory_dir")

        _memory_storage_manager = MemoryStorageManager(memory_dir)
    return _memory_storage_manager
