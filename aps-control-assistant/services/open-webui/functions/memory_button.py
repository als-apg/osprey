"""
title: Memory Manager Action
author: ALS Assistant Team
version: 1.0.0
required_open_webui_version: 0.5.1
icon_url: data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDNjNS41IDAgMTAgMS41IDEwIDMuNVYxOWMwIDItNC41IDMuNS0xMCAzLjVTMiAyMSAyIDE5VjYuNUMyIDQuNSA2LjUgMyAxMiAzem0wIDJjLTQuNDEgMC04IDEuMTItOCAycy0zLjU5IDIgOCAyIDgtMS4xMiA4LTItMy41OS0yLTgtMnptOCA5YzAgLjg4LTMuNTkgMi04IDJzLTgtMS4xMi04LTJWMTBjMCAuODggMy41OSAyIDggMnM4LTEuMTIgOC0yek0yMCA4djJjMCAuODgtMy41OSAyLTggMnMtOC0xLjEyLTgtMlY4YzAgLjg4IDMuNTkgMiA4IDJzOC0xLjEyIDgtMnoiIGZpbGw9ImN1cnJlbnRDb2xvciIvPgo8L3N2Zz4K
Description: View and edit your personal memory entries. Display, add, modify, or delete memories stored in your personal memory file.
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

# Set up logger for this function module
logger = logging.getLogger(__name__)


class Action:
    class Valves(BaseModel):
        memory_base_path: str = Field(
            default=os.getenv("USER_MEMORY_DIR", "/app/user_memory"),
            description="Base directory path for user memory files (mounted from host user_memory directory)"
        )

    class UserValves(BaseModel):
        show_timestamps: bool = Field(
            default=True, description="Show timestamps for memory entries"
        )
        max_display_entries: int = Field(
            default=50, description="Maximum number of entries to display at once"
        )
        enable_editing: bool = Field(
            default=True, description="Allow editing and modifying memories"
        )
        auto_backup: bool = Field(
            default=True, description="Create automatic backups before editing"
        )

    def __init__(self):
        self.valves = self.Valves()

    def _get_memory_file_path(self, user_id: str) -> Path:
        """Get path to user's memory file with safe filename generation."""
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")

        # Sanitize user_id for filename (same logic as existing memory manager)
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in "-_")
        return Path(self.valves.memory_base_path) / f"{safe_user_id}.json"

    def _load_memory_data(self, user_id: str) -> dict:
        """Load memory data from user's JSON file."""
        try:
            # DEBUG: Print extensive debug information
            import os
            logger.info(f"DEBUG: Current working directory: {os.getcwd()}")
            logger.info(f"DEBUG: Memory base path setting: {self.valves.memory_base_path}")
            logger.info(f"DEBUG: User ID received: '{user_id}'")

            memory_file = self._get_memory_file_path(user_id)
            logger.info(f"DEBUG: Full memory file path: {memory_file}")
            logger.info(f"DEBUG: Memory file absolute path: {memory_file.absolute()}")
            logger.info(f"DEBUG: Memory file exists: {memory_file.exists()}")

            # Check if directory exists
            memory_dir = memory_file.parent
            logger.info(f"DEBUG: Memory directory: {memory_dir}")
            logger.info(f"DEBUG: Memory directory exists: {memory_dir.exists()}")

            # List contents of memory directory if it exists
            if memory_dir.exists():
                try:
                    dir_contents = list(memory_dir.iterdir())
                    logger.info(f"DEBUG: Memory directory contents: {[str(f) for f in dir_contents]}")
                except Exception as dir_e:
                    logger.error(f"DEBUG: Error listing directory contents: {dir_e}")

            if memory_file.exists():
                logger.info("DEBUG: Memory file found, attempting to read...")
                with open(memory_file, encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"DEBUG: Successfully loaded memory data with {len(data.get('entries', []))} entries")
                return data
            else:
                logger.info("DEBUG: Memory file does not exist, returning empty structure")
                # Return empty structure if file doesn't exist
                return {
                    "user_id": user_id,
                    "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "entries": []
                }
        except Exception as e:
            logger.error(f"DEBUG: Exception in _load_memory_data: {type(e).__name__}: {e}")
            logger.error(f"Error loading memory for user {user_id}: {e}")
            raise

    def _validate_memory_data(self, data: dict) -> bool:
        """Validate memory data structure before saving."""
        try:
            # Check required fields
            if not isinstance(data, dict):
                return False

            required_fields = ["user_id", "entries"]
            for field in required_fields:
                if field not in data:
                    return False

            # Validate entries structure
            entries = data.get("entries", [])
            if not isinstance(entries, list):
                return False

            for entry in entries:
                if not isinstance(entry, dict):
                    return False
                if "timestamp" not in entry or "content" not in entry:
                    return False
                if not isinstance(entry["timestamp"], str) or not isinstance(entry["content"], str):
                    return False

            return True
        except Exception as e:
            logger.error(f"Error validating memory data: {e}")
            return False

    def _save_memory_data(self, user_id: str, data: dict) -> bool:
        """Save memory data to user's JSON file with validation."""
        try:
            # Validate data structure
            if not self._validate_memory_data(data):
                logger.error(f"Invalid memory data structure for user {user_id}")
                return False

            memory_file = self._get_memory_file_path(user_id)
            memory_file.parent.mkdir(exist_ok=True, parents=True)

            # Update last_updated timestamp
            data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")

            # Add created timestamp if missing
            if "created" not in data:
                data["created"] = data["last_updated"]

            # Atomic write: write to temp file first, then rename
            temp_file = memory_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Rename temp file to actual file (atomic operation on most filesystems)
            temp_file.rename(memory_file)

            logger.info(f"Successfully saved memory for user {user_id} with {len(data.get('entries', []))} entries")
            return True
        except Exception as e:
            logger.error(f"Error saving memory for user {user_id}: {e}")
            # Clean up temp file if it exists
            temp_file = self._get_memory_file_path(user_id).with_suffix('.tmp')
            if temp_file.exists():
                temp_file.unlink()
            return False

    def _create_backup(self, user_id: str) -> bool:
        """Create a backup of the user's memory file in the .backups folder."""
        try:
            memory_file = self._get_memory_file_path(user_id)
            if memory_file.exists():
                # Create .backups directory if it doesn't exist
                backup_dir = memory_file.parent / ".backups"
                backup_dir.mkdir(exist_ok=True)

                # Single backup file per user (overwrites previous backup)
                backup_file = backup_dir / f"{memory_file.stem}.json"
                shutil.copy2(memory_file, backup_file)
                logger.info(f"Created/updated backup: {backup_file}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error creating backup for user {user_id}: {e}")
            return False

    def _restore_from_backup(self, user_id: str) -> bool:
        """Restore user's memory from backup file."""
        try:
            memory_file = self._get_memory_file_path(user_id)
            backup_dir = memory_file.parent / ".backups"
            backup_file = backup_dir / f"{memory_file.stem}.json"

            if backup_file.exists():
                shutil.copy2(backup_file, memory_file)
                logger.info(f"Restored memory from backup: {backup_file}")
                return True
            else:
                logger.warning(f"No backup file found for user {user_id}")
                return False
        except Exception as e:
            logger.error(f"Error restoring backup for user {user_id}: {e}")
            return False

    def _format_memory_display(self, data: dict, user_valves) -> str:
        """Format memory data as readable markdown."""
        entries = data.get("entries", [])
        user_id = data.get("user_id", "Unknown")
        created = data.get("created", "Unknown")
        last_updated = data.get("last_updated", "Unknown")

        markdown = "# 🧠 Personal Memory Manager\n\n"
        markdown += f"**User ID:** `{user_id}`\n"
        markdown += f"**Created:** {created}\n"
        markdown += f"**Last Updated:** {last_updated}\n"
        markdown += f"**Total Memories:** {len(entries)}\n\n"

        if not entries:
            markdown += "## 📝 No memories found\n\n"
            markdown += "Your memory file is empty. Memories you save will appear here.\n\n"
            if user_valves.enable_editing:
                markdown += "💡 *Tip: Use the memory button to add your first memory!*\n"
            return markdown

        markdown += "---\n\n"

        # Display entries (limit based on user preference)
        display_count = min(len(entries), user_valves.max_display_entries)
        if display_count < len(entries):
            markdown += f"## 📋 Recent {display_count} of {len(entries)} Memories\n\n"
        else:
            markdown += f"## 📋 All {len(entries)} Memories\n\n"

        # Sort entries by timestamp (newest first)
        sorted_entries = sorted(entries, key=lambda x: x.get("timestamp", ""), reverse=True)

        for i, entry in enumerate(sorted_entries[:display_count], 1):
            timestamp = entry.get("timestamp", "Unknown time")
            content = entry.get("content", "").strip()

            if user_valves.show_timestamps:
                markdown += f"### 🕒 {timestamp}\n"
            else:
                markdown += f"### 📌 Memory #{i}\n"

            markdown += f"{content}\n\n"

            if i < display_count:
                markdown += "---\n\n"

        if display_count < len(entries):
            markdown += f"\n💡 *Showing {display_count} of {len(entries)} total memories. Adjust max_display_entries to see more.*\n"

        # Add edit hint
        if user_valves.enable_editing:
            markdown += "\n\n🔧 **Want to edit your memories?** Click the memory button again to open the interactive editor.\n"

        return markdown

    async def create_memory_editor_interface(
        self,
        data: dict,
        user_id: str,
        __event_emitter__=None,
        __event_call__=None,
    ) -> dict | None:
        """Create an interactive memory editing interface using JavaScript."""

        entries = data.get("entries", [])
        entries_json = json.dumps(entries).replace('"', '\\"').replace('\n', '\\n')

        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "Opening interactive memory editor...", "done": False},
            }
        )

        editor_js = f"""
        try {{
            // Professional color palette - customize these variables
            const colors = {{
                primary: '#669bbc',      // Indigo - main action buttons
                primaryHover: '#003049', // Darker indigo for hover
                secondary: '#b1a7a6',    // Gray - secondary actions
                secondaryHover: '#161a1d', // Darker gray for hover
                success: '#669bbc',      // Green - save/confirm actions
                successHover: '#003049', // Darker green for hover
                danger: '#C1121F',       // Red - delete/destructive actions
                dangerHover: '#780000',  // Darker red for hover
                background: '#ffffff',   // White backgrounds
                border: '#e5e7eb',       // Light gray borders
                text: '#374151',         // Dark gray text
                textLight: '#6b7280'     // Light gray text
            }};

            // Remove any existing memory editor
            const existingEditor = document.getElementById('memory-editor-overlay');
            if (existingEditor) {{
                existingEditor.remove();
            }}

            // Create overlay
            const overlay = document.createElement('div');
            overlay.id = 'memory-editor-overlay';
            overlay.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.7);
                z-index: 10000;
                display: flex;
                justify-content: center;
                align-items: center;
            `;

            // Create editor container
            const editor = document.createElement('div');
            editor.style.cssText = `
                background: white;
                border-radius: 12px;
                padding: 24px;
                width: 90%;
                max-width: 800px;
                max-height: 80vh;
                overflow-y: auto;
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                color: #333;
            `;

            // Parse the memories data
            let memories = JSON.parse("{entries_json}");

            // Create editor HTML
            editor.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 2px solid ${{colors.border}};">
                    <h2 style="margin: 0; color: ${{colors.text}}; font-size: 24px; font-weight: bold;">Edit Memories for {user_id}</h2>
                    <button id="memory-close-btn" style="background: ${{colors.danger}}; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 500; transition: background-color 0.2s;">✕ Close</button>
                </div>

                <div style="margin-bottom: 20px;">
                    <button id="memory-add-btn" style="background: ${{colors.success}}; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; margin-right: 10px; font-weight: 500; font-size: 14px; transition: background-color 0.2s;">+ Add Memory</button>
                    <button id="memory-save-btn" style="background: ${{colors.primary}}; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; margin-right: 10px; font-weight: 500; font-size: 14px; transition: background-color 0.2s;">Save Changes</button>
                    <button id="memory-cancel-btn" style="background: ${{colors.secondary}}; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 500; font-size: 14px; transition: background-color 0.2s;">Cancel</button>
                </div>

                <div id="memories-container"></div>
            `;

            // Function to render memories
            function renderMemories() {{
                const container = document.getElementById('memories-container');
                container.innerHTML = '';

                if (memories.length === 0) {{
                    container.innerHTML = '<p style="text-align: center; color: ' + colors.textLight + '; font-style: italic; padding: 40px;">No memories yet. Click "Add Memory" to create your first one!</p>';
                    return;
                }}

                // Sort memories by timestamp (newest first) - same as main display
                const sortedMemories = [...memories].sort((a, b) => {{
                    return new Date(b.timestamp) - new Date(a.timestamp);
                }});

                sortedMemories.forEach((memory, displayIndex) => {{
                    // Find the actual index in the original array for operations
                    const actualIndex = memories.findIndex(m => m === memory);
                    const memoryDiv = document.createElement('div');
                    memoryDiv.style.cssText = `
                        border: 2px solid ${{colors.border}};
                        padding: 20px;
                        margin-bottom: 16px;
                        border-radius: 8px;
                        background-color: #f9fafb;
                        position: relative;
                    `;

                    memoryDiv.innerHTML = `
                        <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
                            <button onclick="deleteMemory(${{actualIndex}})" style="background: ${{colors.danger}}; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; transition: background-color 0.2s;" onmouseover="this.style.backgroundColor='${{colors.dangerHover}}'" onmouseout="this.style.backgroundColor='${{colors.danger}}'">Delete</button>
                        </div>
                        <div style="margin-bottom: 12px;">
                            <label style="display: block; font-weight: 600; margin-bottom: 6px; color: ${{colors.text}};">Timestamp:</label>
                            <input type="text" value="${{memory.timestamp}}" onchange="updateTimestamp(${{actualIndex}}, this.value)" style="width: 100%; padding: 8px 12px; border: 1px solid ${{colors.border}}; border-radius: 6px; font-size: 14px;">
                        </div>
                                                 <div>
                             <label style="display: block; font-weight: 600; margin-bottom: 6px; color: ${{colors.text}};">Content:</label>
                             <textarea onchange="updateContent(${{actualIndex}}, this.value)" oninput="autoResizeTextarea(this)" style="width: 100%; min-height: 40px; padding: 12px; border: 1px solid ${{colors.border}}; border-radius: 6px; resize: vertical; font-family: inherit; font-size: 14px; line-height: 1.5; overflow: hidden;">${{memory.content}}</textarea>
                         </div>
                    `;

                    container.appendChild(memoryDiv);
                }});

                // Auto-resize all textareas after rendering
                container.querySelectorAll('textarea').forEach(textarea => {{
                    autoResizeTextarea(textarea);
                }});
            }}

            // Global functions for memory operations
            window.updateTimestamp = function(index, value) {{
                memories[index].timestamp = value;
            }};

            window.updateContent = function(index, value) {{
                memories[index].content = value;
            }};

            window.deleteMemory = function(index) {{
                if (confirm('Are you sure you want to delete this memory?')) {{
                    memories.splice(index, 1);
                    renderMemories();
                }}
            }};

            // Auto-resize textarea function
            window.autoResizeTextarea = function(textarea) {{
                // Reset height to auto to get the correct scrollHeight
                textarea.style.height = 'auto';
                // Set height to scrollHeight (content height) with minimum of 40px
                textarea.style.height = Math.max(40, textarea.scrollHeight) + 'px';
            }};



            // Add new memory function with popup dialog
            function addNewMemory() {{
                // Add visual feedback to button
                const addBtn = document.getElementById('memory-add-btn');
                const originalText = addBtn.innerHTML;
                addBtn.innerHTML = '⏳ Creating...';
                addBtn.disabled = true;

                // Create new memory popup
                createNewMemoryPopup().then((newMemory) => {{
                    // Reset button
                    addBtn.innerHTML = originalText;
                    addBtn.disabled = false;

                    if (newMemory) {{
                        // Add new memory to the beginning of the list (newest first)
                        memories.unshift(newMemory);
                        renderMemories();
                    }}
                }});
            }}

            // Create new memory popup dialog
            function createNewMemoryPopup() {{
                return new Promise((resolve) => {{
                    // Create overlay for new memory dialog
                    const newMemoryOverlay = document.createElement('div');
                    newMemoryOverlay.style.cssText = `
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background: rgba(0, 0, 0, 0.8);
                        z-index: 20000;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    `;

                    // Create new memory dialog
                    const dialog = document.createElement('div');
                    dialog.style.cssText = `
                        background: white;
                        border-radius: 12px;
                        padding: 32px;
                        width: 90%;
                        max-width: 600px;
                        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
                        color: #333;
                        animation: slideIn 0.3s ease-out;
                    `;

                    // Add keyframe animation
                    if (!document.getElementById('memory-dialog-styles')) {{
                        const style = document.createElement('style');
                        style.id = 'memory-dialog-styles';
                        style.textContent = `
                            @keyframes slideIn {{
                                from {{ transform: translateY(-50px); opacity: 0; }}
                                to {{ transform: translateY(0); opacity: 1; }}
                            }}
                        `;
                        document.head.appendChild(style);
                    }}

                    const now = new Date();
                    const currentTimestamp = now.getFullYear() + '-' +
                        String(now.getMonth() + 1).padStart(2, '0') + '-' +
                        String(now.getDate()).padStart(2, '0') + ' ' +
                        String(now.getHours()).padStart(2, '0') + ':' +
                        String(now.getMinutes()).padStart(2, '0');

                    dialog.innerHTML = `
                        <div style="text-align: center; margin-bottom: 24px;">
                            <h3 style="margin: 0; color: ${{colors.success}}; font-size: 20px; font-weight: bold;">✨ Create New Memory</h3>
                            <p style="margin: 8px 0 0 0; color: ${{colors.textLight}}; font-size: 14px;">Add a new memory to your personal collection</p>
                        </div>

                        <div style="margin-bottom: 20px;">
                            <label style="display: block; font-weight: 600; margin-bottom: 8px; color: ${{colors.text}};">Timestamp:</label>
                            <input type="text" id="new-memory-timestamp" value="${{currentTimestamp}}" style="width: 100%; padding: 12px; border: 2px solid ${{colors.border}}; border-radius: 8px; font-size: 14px; transition: border-color 0.2s;">
                        </div>

                        <div style="margin-bottom: 32px;">
                            <label style="display: block; font-weight: 600; margin-bottom: 8px; color: ${{colors.text}};">Memory Content:</label>
                            <textarea id="new-memory-content" placeholder="What would you like me to remember?" oninput="autoResizeTextarea(this)" style="width: 100%; min-height: 50px; padding: 12px; border: 2px solid ${{colors.border}}; border-radius: 8px; resize: vertical; font-family: inherit; font-size: 14px; line-height: 1.5; transition: border-color 0.2s; overflow: hidden;"></textarea>
                        </div>

                        <div style="display: flex; gap: 12px; justify-content: flex-end;">
                            <button id="cancel-new-memory" style="background: ${{colors.secondary}}; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 500; font-size: 14px; transition: background-color 0.2s;">Cancel</button>
                            <button id="save-new-memory" style="background: ${{colors.success}}; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 500; font-size: 14px; transition: background-color 0.2s;">Add Memory</button>
                        </div>
                    `;

                    // Add focus and hover effects
                    const timestampInput = dialog.querySelector('#new-memory-timestamp');
                    const contentTextarea = dialog.querySelector('#new-memory-content');
                    const saveBtn = dialog.querySelector('#save-new-memory');
                    const cancelBtn = dialog.querySelector('#cancel-new-memory');

                                         // Focus effects using color variables
                     [timestampInput, contentTextarea].forEach(input => {{
                         input.addEventListener('focus', () => input.style.borderColor = colors.success);
                         input.addEventListener('blur', () => input.style.borderColor = colors.border);
                     }});

                                         // Hover effects using color variables
                     saveBtn.addEventListener('mouseenter', () => saveBtn.style.backgroundColor = colors.successHover);
                     saveBtn.addEventListener('mouseleave', () => saveBtn.style.backgroundColor = colors.success);
                     cancelBtn.addEventListener('mouseenter', () => cancelBtn.style.backgroundColor = colors.secondaryHover);
                     cancelBtn.addEventListener('mouseleave', () => cancelBtn.style.backgroundColor = colors.secondary);

                    // Event handlers
                    function closeDialog(result) {{
                        newMemoryOverlay.remove();
                        resolve(result);
                    }}

                    cancelBtn.onclick = () => closeDialog(null);

                    saveBtn.onclick = () => {{
                        const timestamp = timestampInput.value.trim();
                        const content = contentTextarea.value.trim();

                                                 if (!content) {{
                             contentTextarea.style.borderColor = colors.danger;
                             contentTextarea.focus();
                             return;
                         }}

                        closeDialog({{
                            timestamp: timestamp || currentTimestamp,
                            content: content
                        }});
                    }};

                    // Close on overlay click
                    newMemoryOverlay.onclick = (e) => {{
                        if (e.target === newMemoryOverlay) {{
                            closeDialog(null);
                        }}
                    }};

                    // Close on Escape key
                    document.addEventListener('keydown', function escapeHandler(e) {{
                        if (e.key === 'Escape') {{
                            document.removeEventListener('keydown', escapeHandler);
                            closeDialog(null);
                        }}
                    }});

                                         // Append to document and focus
                     newMemoryOverlay.appendChild(dialog);
                     document.body.appendChild(newMemoryOverlay);

                     // Auto-resize the textarea and focus after a brief delay
                     setTimeout(() => {{
                         autoResizeTextarea(contentTextarea);
                         contentTextarea.focus();
                     }}, 100);
                }});
            }}

            // Set up event listeners and return a promise
            function setupEventListeners() {{
                return new Promise((resolve) => {{
                    // Get button elements
                    const addBtn = document.getElementById('memory-add-btn');
                    const saveBtn = document.getElementById('memory-save-btn');
                    const cancelBtn = document.getElementById('memory-cancel-btn');
                    const closeBtn = document.getElementById('memory-close-btn');

                    // Set up click handlers
                    addBtn.onclick = addNewMemory;

                    closeBtn.onclick = function() {{
                        overlay.remove();
                        resolve({{ action: 'cancel' }});
                    }};

                    cancelBtn.onclick = function() {{
                        overlay.remove();
                        resolve({{ action: 'cancel' }});
                    }};

                    saveBtn.onclick = function() {{
                        overlay.remove();
                        resolve({{ action: 'save', memories: memories }});
                    }};

                    // Add hover effects to main buttons
                    addBtn.addEventListener('mouseover', () => addBtn.style.backgroundColor = colors.successHover);
                    addBtn.addEventListener('mouseout', () => addBtn.style.backgroundColor = colors.success);

                    saveBtn.addEventListener('mouseover', () => saveBtn.style.backgroundColor = colors.primaryHover);
                    saveBtn.addEventListener('mouseout', () => saveBtn.style.backgroundColor = colors.primary);

                    cancelBtn.addEventListener('mouseover', () => cancelBtn.style.backgroundColor = colors.secondaryHover);
                    cancelBtn.addEventListener('mouseout', () => cancelBtn.style.backgroundColor = colors.secondary);

                    closeBtn.addEventListener('mouseover', () => closeBtn.style.backgroundColor = colors.dangerHover);
                    closeBtn.addEventListener('mouseout', () => closeBtn.style.backgroundColor = colors.danger);

                    // Close on overlay click
                    overlay.onclick = function(e) {{
                        if (e.target === overlay) {{
                            overlay.remove();
                            resolve({{ action: 'cancel' }});
                        }}
                    }};
                }});
            }}

            // Append to document and render
            overlay.appendChild(editor);
            document.body.appendChild(overlay);
            renderMemories();

            // Return the promise that resolves when user interacts
            return await setupEventListeners();

        }} catch (error) {{
            return {{ action: 'error', message: 'Error creating editor: ' + error.message }};
        }}
        """

        try:
            result = await __event_call__(
                {
                    "type": "execute",
                    "data": {"code": editor_js},
                }
            )
            return result
        except Exception as e:
            logger.error(f"Error creating memory editor interface: {e}")
            return {"action": "error", "message": str(e)}

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> dict | None:
        """Main action handler for memory manager."""
        logger.info(f"Memory Manager Action - User: {__user__.get('name', 'Unknown')}, ID: {__user__.get('id', 'Unknown')}")

        user_valves = __user__.get("valves")
        if not user_valves:
            user_valves = self.UserValves()

        # Get user ID - use email prefix if available, otherwise use ID
        user_id = __user__.get("id")

        if __user__.get("email"):
            user_email = __user__.get("email")
            if "@" in user_email:
                user_id = user_email.split("@")[0]
                logger.info(f"Using email prefix as user ID: '{user_id}'")
            else:
                logger.warning(f"Email found but no @ symbol: {user_email}")
        else:
            logger.info(f"No email found, using user ID: {user_id}")

        logger.info(f"Memory Manager - User: '{user_id}'")

        if not user_id:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Error: User ID not available", "done": True},
                }
            )
            return

        try:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Loading memory data...", "done": False},
                }
            )

            data = self._load_memory_data(user_id)

            # Check if this is a second click (editing mode)
            # For now, we'll assume if they have memories and editing is enabled, they want to edit
            if data.get("entries") and user_valves.enable_editing:
                # Create backup if enabled
                if user_valves.auto_backup:
                    backup_created = self._create_backup(user_id)
                    if backup_created:
                        await __event_emitter__(
                            {
                                "type": "notification",
                                "data": {"type": "info", "content": "Backup created before editing"},
                            }
                        )

                # Open editor interface
                editor_result = await self.create_memory_editor_interface(
                    data, user_id, __event_emitter__, __event_call__
                )

                if editor_result and editor_result.get("action") == "save":
                    # Save the updated memories
                    data["entries"] = editor_result.get("memories", [])
                    success = self._save_memory_data(user_id, data)

                    if success:
                        await __event_emitter__(
                            {
                                "type": "notification",
                                "data": {"type": "success", "content": "Memories saved successfully!"},
                            }
                        )
                    else:
                        await __event_emitter__(
                            {
                                "type": "notification",
                                "data": {"type": "error", "content": "Failed to save memories"},
                            }
                        )
                elif editor_result and editor_result.get("action") == "error":
                    await __event_emitter__(
                        {
                            "type": "notification",
                            "data": {"type": "error", "content": f"Editor error: {editor_result.get('message', 'Unknown error')}"},
                        }
                    )
                else:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "Edit cancelled", "done": True},
                        }
                    )
            else:
                # No memories or editing disabled - just open editor anyway
                if user_valves.enable_editing:
                    # Open editor interface even for empty memories
                    editor_result = await self.create_memory_editor_interface(
                        data, user_id, __event_emitter__, __event_call__
                    )

                    if editor_result and editor_result.get("action") == "save":
                        # Save the updated memories
                        data["entries"] = editor_result.get("memories", [])
                        success = self._save_memory_data(user_id, data)

                        if success:
                            await __event_emitter__(
                                {
                                    "type": "notification",
                                    "data": {"type": "success", "content": "Memories saved successfully!"},
                                }
                            )
                        else:
                            await __event_emitter__(
                                {
                                    "type": "notification",
                                    "data": {"type": "error", "content": "Failed to save memories"},
                                }
                            )
                    elif editor_result and editor_result.get("action") == "error":
                        await __event_emitter__(
                            {
                                "type": "notification",
                                "data": {"type": "error", "content": f"Editor error: {editor_result.get('message', 'Unknown error')}"},
                            }
                        )
                else:
                    await __event_emitter__(
                        {
                            "type": "notification",
                            "data": {"type": "info", "content": "Memory editing is disabled"},
                        }
                    )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Memory operation completed", "done": True},
                }
            )

        except Exception as e:
            logger.error(f"Memory Manager error for user {user_id}: {e}")
            error_message = f"# ❌ Error Loading Memory\n\nFailed to load memory data: {str(e)}"
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": error_message},
                }
            )
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Memory Manager failed", "done": True},
                }
            )
