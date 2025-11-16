"""
Abstract base class for all channel finder pipelines.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from .models import ChannelFinderResult


class BasePipeline(ABC):
    """Abstract base class for all channel finder pipelines."""
    
    def __init__(self, database, model_config: dict, **kwargs):
        """
        Initialize pipeline.
        
        Args:
            database: Database instance (specific to pipeline type)
            model_config: LLM model configuration
            **kwargs: Pipeline-specific configuration
        """
        self.database = database
        self.model_config = model_config
    
    @abstractmethod
    async def process_query(self, query: str) -> ChannelFinderResult:
        """
        Process a natural language query and return matching channels.
        
        Args:
            query: Natural language query string
            
        Returns:
            ChannelFinderResult with found channels and metadata
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Return pipeline statistics.
        
        Returns:
            Dict with pipeline-specific statistics
        """
        pass
    
    @property
    @abstractmethod
    def pipeline_name(self) -> str:
        """Return the pipeline name."""
        pass

