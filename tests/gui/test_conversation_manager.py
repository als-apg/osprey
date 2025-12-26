"""
Unit tests for Conversation Manager

Tests the conversation management functionality including creating,
switching, deleting conversations and managing messages.
"""

import pytest
from pathlib import Path
from datetime import datetime
import tempfile
import json

from osprey.interfaces.pyqt.conversation_manager import (
    ConversationManager,
    Conversation,
    ConversationMessage
)


class TestConversationMessage:
    """Test suite for ConversationMessage dataclass."""
    
    def test_create_message(self):
        """Test creating a conversation message."""
        msg = ConversationMessage(type='user', content='Hello')
        assert msg.type == 'user'
        assert msg.content == 'Hello'
        assert isinstance(msg.timestamp, datetime)
    
    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = ConversationMessage(type='agent', content='Hi there')
        data = msg.to_dict()
        
        assert data['type'] == 'agent'
        assert data['content'] == 'Hi there'
        assert 'timestamp' in data
        assert isinstance(data['timestamp'], str)
    
    def test_message_from_dict(self):
        """Test creating message from dictionary."""
        data = {
            'type': 'user',
            'content': 'Test message',
            'timestamp': '2024-01-01T12:00:00'
        }
        msg = ConversationMessage.from_dict(data)
        
        assert msg.type == 'user'
        assert msg.content == 'Test message'
        assert isinstance(msg.timestamp, datetime)


class TestConversation:
    """Test suite for Conversation dataclass."""
    
    def test_create_conversation(self):
        """Test creating a conversation."""
        conv = Conversation(thread_id='test_123', name='Test Conv')
        assert conv.thread_id == 'test_123'
        assert conv.name == 'Test Conv'
        assert conv.messages == []
        assert isinstance(conv.timestamp, datetime)
    
    def test_conversation_to_dict(self):
        """Test converting conversation to dictionary."""
        msg = ConversationMessage(type='user', content='Hello')
        conv = Conversation(
            thread_id='test_123',
            name='Test',
            messages=[msg]
        )
        data = conv.to_dict()
        
        assert data['thread_id'] == 'test_123'
        assert data['name'] == 'Test'
        assert len(data['messages']) == 1
        assert data['messages'][0]['content'] == 'Hello'
    
    def test_conversation_from_dict(self):
        """Test creating conversation from dictionary."""
        data = {
            'thread_id': 'test_123',
            'name': 'Test',
            'messages': [
                {'type': 'user', 'content': 'Hi', 'timestamp': '2024-01-01T12:00:00'}
            ],
            'timestamp': '2024-01-01T12:00:00'
        }
        conv = Conversation.from_dict(data)
        
        assert conv.thread_id == 'test_123'
        assert conv.name == 'Test'
        assert len(conv.messages) == 1
        assert conv.messages[0].content == 'Hi'


class TestConversationManager:
    """Test suite for ConversationManager class."""
    
    def test_initialization(self):
        """Test manager initializes correctly."""
        manager = ConversationManager()
        assert manager.storage_mode == 'json'
        assert manager.conversations == {}
        assert manager.current_conversation_id is None
    
    def test_create_conversation(self):
        """Test creating a new conversation."""
        manager = ConversationManager()
        thread_id = manager.create_conversation('Test Conv')
        
        assert thread_id in manager.conversations
        assert manager.conversations[thread_id].name == 'Test Conv'
        assert manager.current_conversation_id == thread_id
    
    def test_create_conversation_auto_name(self):
        """Test creating conversation with auto-generated name."""
        manager = ConversationManager()
        thread_id = manager.create_conversation()
        
        assert thread_id in manager.conversations
        assert manager.conversations[thread_id].name == 'Conversation 1'
    
    def test_create_multiple_conversations(self):
        """Test creating multiple conversations."""
        manager = ConversationManager()
        id1 = manager.create_conversation('Conv 1')
        id2 = manager.create_conversation('Conv 2')
        id3 = manager.create_conversation('Conv 3')
        
        assert len(manager.conversations) == 3
        assert manager.current_conversation_id == id3
    
    def test_get_conversation(self):
        """Test getting a conversation by ID."""
        manager = ConversationManager()
        thread_id = manager.create_conversation('Test')
        
        conv = manager.get_conversation(thread_id)
        assert conv is not None
        assert conv.name == 'Test'
    
    def test_get_nonexistent_conversation(self):
        """Test getting conversation that doesn't exist."""
        manager = ConversationManager()
        conv = manager.get_conversation('nonexistent')
        assert conv is None
    
    def test_get_current_conversation(self):
        """Test getting current active conversation."""
        manager = ConversationManager()
        thread_id = manager.create_conversation('Current')
        
        current = manager.get_current_conversation()
        assert current is not None
        assert current.thread_id == thread_id
    
    def test_switch_conversation(self):
        """Test switching between conversations."""
        manager = ConversationManager()
        id1 = manager.create_conversation('Conv 1')
        id2 = manager.create_conversation('Conv 2')
        
        # Switch back to first conversation
        result = manager.switch_conversation(id1)
        assert result is True
        assert manager.current_conversation_id == id1
    
    def test_switch_to_nonexistent_conversation(self):
        """Test switching to conversation that doesn't exist."""
        manager = ConversationManager()
        result = manager.switch_conversation('nonexistent')
        assert result is False
    
    def test_delete_conversation(self):
        """Test deleting a conversation."""
        manager = ConversationManager()
        id1 = manager.create_conversation('Conv 1')
        id2 = manager.create_conversation('Conv 2')
        
        result = manager.delete_conversation(id1)
        assert result is True
        assert id1 not in manager.conversations
        assert len(manager.conversations) == 1
    
    def test_delete_current_conversation_switches_to_another(self):
        """Test that deleting current conversation switches to another."""
        manager = ConversationManager()
        id1 = manager.create_conversation('Conv 1')
        id2 = manager.create_conversation('Conv 2')
        
        # Delete current conversation (id2)
        result = manager.delete_conversation(id2)
        assert result is True
        assert manager.current_conversation_id == id1
    
    def test_cannot_delete_last_conversation(self):
        """Test that last conversation cannot be deleted."""
        manager = ConversationManager()
        thread_id = manager.create_conversation('Only Conv')
        
        result = manager.delete_conversation(thread_id)
        assert result is False
        assert thread_id in manager.conversations
    
    def test_delete_nonexistent_conversation(self):
        """Test deleting conversation that doesn't exist."""
        manager = ConversationManager()
        result = manager.delete_conversation('nonexistent')
        assert result is False
    
    def test_rename_conversation(self):
        """Test renaming a conversation."""
        manager = ConversationManager()
        thread_id = manager.create_conversation('Old Name')
        
        result = manager.rename_conversation(thread_id, 'New Name')
        assert result is True
        assert manager.conversations[thread_id].name == 'New Name'
    
    def test_rename_nonexistent_conversation(self):
        """Test renaming conversation that doesn't exist."""
        manager = ConversationManager()
        result = manager.rename_conversation('nonexistent', 'New Name')
        assert result is False
    
    def test_add_message(self):
        """Test adding a message to conversation."""
        manager = ConversationManager()
        thread_id = manager.create_conversation('Test')
        
        result = manager.add_message(thread_id, 'user', 'Hello')
        assert result is True
        
        messages = manager.get_messages(thread_id)
        assert len(messages) == 1
        assert messages[0].type == 'user'
        assert messages[0].content == 'Hello'
    
    def test_add_multiple_messages(self):
        """Test adding multiple messages."""
        manager = ConversationManager()
        thread_id = manager.create_conversation('Test')
        
        manager.add_message(thread_id, 'user', 'Hello')
        manager.add_message(thread_id, 'agent', 'Hi there')
        manager.add_message(thread_id, 'user', 'How are you?')
        
        messages = manager.get_messages(thread_id)
        assert len(messages) == 3
        assert messages[0].content == 'Hello'
        assert messages[1].content == 'Hi there'
        assert messages[2].content == 'How are you?'
    
    def test_add_message_to_nonexistent_conversation(self):
        """Test adding message to conversation that doesn't exist."""
        manager = ConversationManager()
        result = manager.add_message('nonexistent', 'user', 'Hello')
        assert result is False
    
    def test_get_messages_empty_conversation(self):
        """Test getting messages from empty conversation."""
        manager = ConversationManager()
        thread_id = manager.create_conversation('Empty')
        
        messages = manager.get_messages(thread_id)
        assert messages == []
    
    def test_get_messages_nonexistent_conversation(self):
        """Test getting messages from nonexistent conversation."""
        manager = ConversationManager()
        messages = manager.get_messages('nonexistent')
        assert messages == []
    
    def test_list_conversations(self):
        """Test listing all conversations."""
        manager = ConversationManager()
        manager.create_conversation('Conv 1')
        manager.create_conversation('Conv 2')
        manager.create_conversation('Conv 3')
        
        conversations = manager.list_conversations()
        assert len(conversations) == 3
    
    def test_list_conversations_sorted_by_timestamp(self):
        """Test that conversations are sorted by timestamp (newest first)."""
        manager = ConversationManager()
        id1 = manager.create_conversation('First')
        id2 = manager.create_conversation('Second')
        id3 = manager.create_conversation('Third')
        
        conversations = manager.list_conversations(sort_by_timestamp=True)
        # Most recent should be first
        assert conversations[0].thread_id == id3
        assert conversations[1].thread_id == id2
        assert conversations[2].thread_id == id1
    
    def test_save_to_json(self):
        """Test saving conversations to JSON file."""
        manager = ConversationManager()
        thread_id = manager.create_conversation('Test Conv')
        manager.add_message(thread_id, 'user', 'Hello')
        manager.add_message(thread_id, 'agent', 'Hi')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / 'conversations.json'
            result = manager.save_to_json(file_path)
            
            assert result is True
            assert file_path.exists()
            
            # Verify JSON content
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            assert thread_id in data
            assert data[thread_id]['name'] == 'Test Conv'
            assert len(data[thread_id]['messages']) == 2
    
    def test_load_from_json(self):
        """Test loading conversations from JSON file."""
        # First save some conversations
        manager1 = ConversationManager()
        id1 = manager1.create_conversation('Conv 1')
        manager1.add_message(id1, 'user', 'Message 1')
        id2 = manager1.create_conversation('Conv 2')
        manager1.add_message(id2, 'user', 'Message 2')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / 'conversations.json'
            manager1.save_to_json(file_path)
            
            # Load into new manager
            manager2 = ConversationManager()
            result = manager2.load_from_json(file_path)
            
            assert result is True
            assert len(manager2.conversations) == 2
            assert id1 in manager2.conversations
            assert id2 in manager2.conversations
            assert len(manager2.get_messages(id1)) == 1
            assert len(manager2.get_messages(id2)) == 1
    
    def test_load_from_nonexistent_file(self):
        """Test loading from file that doesn't exist."""
        manager = ConversationManager()
        result = manager.load_from_json(Path('/nonexistent/path.json'))
        assert result is False
    
    def test_clear_all(self):
        """Test clearing all conversations."""
        manager = ConversationManager()
        manager.create_conversation('Conv 1')
        manager.create_conversation('Conv 2')
        manager.create_conversation('Conv 3')
        
        manager.clear_all()
        
        # Should have one new default conversation
        assert len(manager.conversations) == 1
        conversations = manager.list_conversations()
        assert conversations[0].name == 'Initial Conversation'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])