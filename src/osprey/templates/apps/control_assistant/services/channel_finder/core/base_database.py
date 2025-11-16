"""
Abstract base class for all database implementations.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseDatabase(ABC):
    """Abstract base class for all channel databases."""
    
    def __init__(self, db_path: str):
        """
        Initialize database.
        
        Args:
            db_path: Path to database file
        """
        self.db_path = db_path
        self.load_database()
    
    @abstractmethod
    def load_database(self):
        """Load database from file."""
        pass
    
    @abstractmethod
    def get_channel(self, channel_name: str) -> Optional[Dict]:
        """
        Get channel information by name.
        
        Args:
            channel_name: Channel name to lookup
            
        Returns:
            Channel dict or None if not found
        """
        pass
    
    @abstractmethod
    def get_all_channels(self) -> List[Dict]:
        """
        Get all channels in the database.
        
        Returns:
            List of channel dictionaries
        """
        pass
    
    @abstractmethod
    def validate_channel(self, channel_name: str) -> bool:
        """
        Check if a channel exists.
        
        Args:
            channel_name: Channel name to validate
            
        Returns:
            True if channel exists, False otherwise
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dict with statistics (total_channels, etc.)
        """
        pass

