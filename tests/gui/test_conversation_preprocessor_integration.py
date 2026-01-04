"""
Integration test for ConversationPreprocessor with GUI

Tests that the GUI actually calls the preprocessor when sending messages.
"""

import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Set Qt to use offscreen platform for headless testing
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Skip if PyQt5 is not available
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication


def test_gui_calls_preprocessor_on_send_message():
    """Test that GUI.send_message() actually calls the preprocessor."""
    
    # Create QApplication if it doesn't exist
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Import GUI after QApplication is created
    from osprey.interfaces.pyqt.gui import OspreyGUI
    from osprey.interfaces.pyqt.conversation_preprocessor import ConversationPreprocessor
    
    # Mock the config file check to avoid FileNotFoundError
    with patch('osprey.interfaces.pyqt.gui.Path') as mock_path:
        mock_path.return_value.parent = MagicMock()
        mock_path.return_value.exists.return_value = True
        
        # Create GUI instance with mocked config
        with patch('osprey.interfaces.pyqt.gui.OspreyGUI.initialize_framework'):
            with patch('osprey.interfaces.pyqt.gui.OspreyGUI.setup_ui'):
                gui = OspreyGUI.__new__(OspreyGUI)
                
                # Set up minimal required attributes
                gui.input_field = MagicMock()
                gui.input_field.toPlainText.return_value = "What was the weather when I asked earlier?"
                
                gui.conversation_manager = MagicMock()
                gui.current_conversation_id = "test_conv"
                gui._agent_processing = False
                gui._queued_message = None
                
                # Mock the conversation manager to return a conversation with history
                mock_conversation = MagicMock()
                timestamp = datetime(2026, 1, 4, 10, 30, 0)
                mock_conversation.messages = [
                    {"role": "user", "content": "What's the weather?", "timestamp": timestamp}
                ]
                gui.conversation_manager.get_conversation.return_value = mock_conversation
                gui.conversation_manager.add_message = MagicMock()
                
                # Mock other required methods
                gui._append_colored_message = MagicMock()
                gui.conversation_display_mgr = MagicMock()
                gui.save_conversation_history = MagicMock()
                gui.project_manager = MagicMock()
                gui.project_manager.get_enabled_projects.return_value = []
                gui.status_bar = MagicMock()
                
                # Patch the preprocessor to track if it was called
                with patch.object(ConversationPreprocessor, 'preprocess_message') as mock_preprocess:
                    mock_preprocess.return_value = (
                        "What was the weather at 2026-01-04 10:30:00 when I asked?",
                        "📝 Resolved conversation reference"
                    )
                    
                    # Call send_message
                    try:
                        gui.send_message()
                    except Exception:
                        # We expect this to fail due to missing project setup
                        # We just want to verify the preprocessor was called
                        pass
                    
                    # Verify preprocessor was called
                    assert mock_preprocess.called, "ConversationPreprocessor.preprocess_message should be called"
                    
                    # Verify it was called with correct arguments
                    call_args = mock_preprocess.call_args
                    assert call_args[0][0] == "What was the weather when I asked earlier?"
                    assert call_args[0][1] == gui.conversation_manager
                    assert call_args[0][2] == "test_conv"
                    
                    # Verify explanation was shown to user
                    assert gui._append_colored_message.called
                    explanation_call = [
                        call for call in gui._append_colored_message.call_args_list
                        if "Resolved conversation reference" in str(call)
                    ]
                    assert len(explanation_call) > 0, "Explanation should be shown to user"
                    
                    print("✅ Integration test passed!")
                    print("   - GUI.send_message() calls ConversationPreprocessor.preprocess_message()")
                    print("   - Preprocessor is called with correct arguments")
                    print("   - Explanation is shown to user")


if __name__ == "__main__":
    test_gui_calls_preprocessor_on_send_message()