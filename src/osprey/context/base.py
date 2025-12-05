"""
Pydantic-based Context Base Classes

Clean, production-ready context system using Pydantic for automatic serialization,
validation, and type safety. Eliminates complex custom serialization logic.
"""

import logging
from abc import abstractmethod
from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CapabilityContext(BaseModel):
    """
    Base class for all capability context objects. Uses Pydantic for automatic
    serialization/deserialization and type validation.

    This class provides:
    - Automatic JSON serialization via .model_dump()
    - Automatic deserialization via .model_validate()
    - Type validation on field assignment
    - Consistent interface for all context types
    """

    model_config = {
        # Enforce JSON-compatible types only (no complex Python objects)
        "arbitrary_types_allowed": False,
        # Allow field names for compatibility
        "populate_by_name": True,
        # Use enum values for serialization
        "use_enum_values": True,
        # JSON encoders for specific types (Pydantic v2 syntax)
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        },
    }

    # Class constants - using ClassVar to exclude from model fields
    CONTEXT_TYPE: ClassVar[str] = ""
    CONTEXT_CATEGORY: ClassVar[str] = ""

    @property
    def context_type(self) -> str:
        """Return the context type identifier"""
        return self.CONTEXT_TYPE

    @abstractmethod
    def get_access_details(self, key: str) -> dict:
        """
        Get detailed access information for this context data.

        Args:
            key: The context key this data is stored under

        Returns:
            Dictionary with access details including summary, capabilities, etc.
        """
        pass

    def get_summary(self) -> dict:
        """
        Get a summary of this context data for human display/LLM consumption.

        Returns:
            Dictionary with summary information about the context. The content
            and structure are completely up to the context class implementation.

        Note:
            This method is used by get_summaries() to create human-readable displays
            and LLM prompts. Include whatever information is most useful for your
            specific context type.
        """
        # Check if subclass overrides get_human_summary (legacy support)
        # We check __dict__ to see if the method is actually implemented in this class
        if "get_human_summary" in self.__class__.__dict__:
            import warnings

            warnings.warn(
                f"get_human_summary() is deprecated in {self.__class__.__name__}. "
                f"Please rename the method to get_summary(). "
                f"This backwards compatibility will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Call the legacy method directly - it may have key parameter
            legacy_method = self.__class__.__dict__["get_human_summary"]
            # Try calling with no args first (new style), fall back to with key
            try:
                return legacy_method(self)
            except TypeError:
                # Legacy method expects key parameter
                return legacy_method(self, None)
        else:
            # This should be implemented by subclasses
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement get_summary() method. "
                f"If you have a get_human_summary() method, please rename it to get_summary()."
            )
