"""
ContextManager - Simplified LangGraph-Native Context Management

Ultra-simplified context manager using Pydantic for automatic serialization.
This eliminates 90% of the complexity from the previous implementation while
maintaining full LangGraph compatibility for checkpointing and serialization.

Key simplifications:
- Uses Pydantic's .model_dump() and .model_validate() for serialization
- No custom reflection-based serialization logic
- No complex type conversion or property detection
- No DotDict utilities needed
- Direct registry lookup without extensive validation
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.context.base import CapabilityContext
    from osprey.state.state import AgentState

logger = get_logger("osprey")


# ===================================================================
# ==================== SHARED UTILITY FUNCTIONS ==================
# ===================================================================


class DictNamespace:
    """A namespace that supports both dot access and subscript access.

    Used as a fallback when context classes can't be loaded from the registry
    (e.g., in subprocess execution environments). Provides dict-like access
    patterns while also supporting attribute access.

    Example:
        >>> ns = DictNamespace({"foo": {"bar": 1}})
        >>> ns.foo.bar  # Dot access
        1
        >>> ns["foo"]["bar"]  # Subscript access
        1
        >>> "foo" in ns  # Containment check
        True
    """

    def __init__(self, data: dict):
        self._data = data
        for k, v in data.items():
            setattr(self, k, self._convert(v))

    def _convert(self, v):
        if isinstance(v, dict):
            return DictNamespace(v)
        elif isinstance(v, list):
            return [self._convert(item) for item in v]
        return v

    def __getitem__(self, key):
        """Support subscript access: obj['key']"""
        return getattr(self, key)

    def __contains__(self, key):
        """Support 'in' operator: 'key' in obj"""
        return key in self._data

    def __iter__(self):
        """Support iteration over keys."""
        return iter(self._data.keys())

    def keys(self):
        """Support .keys() method."""
        return self._data.keys()

    def values(self):
        """Support .values() method."""
        return [getattr(self, k) for k in self._data.keys()]

    def items(self):
        """Support .items() method."""
        return [(k, getattr(self, k)) for k in self._data.keys()]

    def get(self, key, default=None):
        """Support .get() method."""
        return getattr(self, key, default)

    def get_summary(self) -> dict:
        """Return a summary dict for compatibility with CapabilityContext."""
        return {"type": "raw_data", "data": self._data}

    def __repr__(self):
        return f"DictNamespace({self._data!r})"


def recursively_summarize_data(data, max_depth: int = 3, current_depth: int = 0):
    """
    Recursively summarize data structures to prevent massive context overflow.

    This utility function is shared across all context classes to ensure consistent
    behavior when creating summaries of large nested data structures.

    Args:
        data: The data structure to summarize
        max_depth: Maximum recursion depth to prevent infinite loops
        current_depth: Current recursion depth

    Returns:
        Summarized version of the data structure
    """
    # Configuration constants - clearly visible "knobs" for tuning behavior
    LARGE_LIST_THRESHOLD = 10  # Lists larger than this will be truncated
    LARGE_DICT_THRESHOLD = 10  # Dicts larger than this will be truncated
    LONG_STRING_THRESHOLD = 200  # Strings longer than this will be truncated
    LIST_SAMPLE_SIZE = 3  # Number of items to show from large lists
    DICT_SAMPLE_SIZE = 3  # Number of keys to show from large dicts
    STRING_PREVIEW_SIZE = 100  # Number of characters to show from long strings

    # Prevent infinite recursion
    if current_depth >= max_depth:
        return f"<Max depth {max_depth} reached: {type(data).__name__}>"

    # Handle lists
    if isinstance(data, list):
        if len(data) > LARGE_LIST_THRESHOLD:
            # For large lists, show count and first few items
            sample_items = [
                recursively_summarize_data(item, max_depth, current_depth + 1)
                for item in data[:LIST_SAMPLE_SIZE]
            ]
            return f"List with {len(data):,} items: {sample_items}... (truncated)"
        else:
            # For small lists, recursively summarize each item
            return [recursively_summarize_data(item, max_depth, current_depth + 1) for item in data]

    # Handle dictionaries
    elif isinstance(data, dict):
        if len(data) > LARGE_DICT_THRESHOLD:
            # For large dicts, show count and first few keys
            keys = list(data.keys())[:DICT_SAMPLE_SIZE]
            sample_data = {
                k: recursively_summarize_data(data[k], max_depth, current_depth + 1) for k in keys
            }
            return f"Dict with {len(data)} keys: {sample_data}... (showing first {DICT_SAMPLE_SIZE} keys only)"
        else:
            # For small dicts, recursively summarize each value
            return {
                k: recursively_summarize_data(v, max_depth, current_depth + 1)
                for k, v in data.items()
            }

    # Handle strings
    elif isinstance(data, str):
        if len(data) > LONG_STRING_THRESHOLD:
            return f"{data[:STRING_PREVIEW_SIZE]}... (truncated from {len(data)} chars)"
        else:
            return data

    # Handle other types (int, float, bool, None, etc.)
    else:
        return data


class ContextManager:
    """Simplified LangGraph-native context manager using Pydantic serialization.

    This class provides sophisticated functionality over dictionary data while storing
    everything in LangGraph-compatible dictionary format. It uses Pydantic's built-in
    serialization capabilities to eliminate complex custom logic.

    The data is stored as: {context_type: {context_key: {field: value}}}
    """

    def __init__(self, state: "AgentState"):
        """Initialize ContextManager with agent state.

        Args:
            state: Full AgentState containing capability_context_data

        Raises:
            TypeError: If state is not an AgentState dictionary
            ValueError: If state doesn't contain capability_context_data key
        """
        if not isinstance(state, dict):
            raise TypeError(f"ContextManager expects AgentState dictionary, got {type(state)}")

        if "capability_context_data" not in state:
            raise ValueError("AgentState must contain 'capability_context_data' key")

        self._data = state["capability_context_data"]
        self._object_cache: dict[str, dict[str, CapabilityContext]] = {}

    def __getattr__(self, context_type: str):
        """Enable dot notation access to context data with lazy namespace creation."""
        if context_type.startswith("_"):
            # For private attributes, use normal attribute access
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{context_type}'"
            )

        if context_type in self._data:
            # Create a namespace for this context type with lazy object reconstruction
            namespace = ContextNamespace(self, context_type)
            return namespace

        # If not found in _data, raise AttributeError to maintain normal Python behavior
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{context_type}'"
        )

    def set_context(
        self,
        context_type: str,
        key: str,
        value: "CapabilityContext",
        skip_validation: bool = False,
        task_objective: str | None = None,
    ) -> None:
        """Store context using Pydantic's built-in serialization.

        Args:
            context_type: Type of context (e.g., "PV_ADDRESSES")
            key: Unique key for this context instance
            value: CapabilityContext object to store
            skip_validation: Skip registry validation (useful for testing)
            task_objective: Optional task description from execution plan step.
                           Stored as metadata to help orchestrator understand
                           what this context was created for (context reuse optimization).
        """
        # Validate using registry (unless skipped for testing)
        if not skip_validation:
            try:
                # Import registry here to avoid circular imports
                from osprey.registry import get_registry

                registry = get_registry()
                # Check if registry is initialized before validation
                if hasattr(registry, "_registries") and registry._registries:
                    # Validate context type is recognized
                    if not registry.is_valid_context_type(context_type):
                        raise ValueError(
                            f"Unknown context type: {context_type}. Valid types: {registry.get_all_context_types()}"
                        )

                    # Validate value is correct type for context type
                    expected_type = registry.get_context_class(context_type)
                    if expected_type is not None and not isinstance(value, expected_type):
                        raise ValueError(
                            f"Context type {context_type} expects {expected_type}, got {type(value)}"
                        )
                else:
                    # Registry not initialized - just log a warning and continue
                    logger.warning(
                        f"Registry not initialized, skipping validation for {context_type}"
                    )

            except ImportError:
                # If registry is not available yet, skip validation
                logger.debug(f"Registry not available, skipping validation for {context_type}")

        # Use Pydantic's built-in .model_dump() method for serialization
        if context_type not in self._data:
            self._data[context_type] = {}

        context_dict = value.model_dump()

        # Store task_objective as metadata for orchestrator context reuse optimization
        # This helps the orchestrator understand what each context was created for,
        # enabling intelligent reuse decisions without exposing raw data
        if task_objective:
            context_dict["_meta"] = {"task_objective": task_objective}

        self._data[context_type][key] = context_dict

        # Update cache
        if context_type not in self._object_cache:
            self._object_cache[context_type] = {}
        self._object_cache[context_type][key] = value

        logger.debug(f"Stored context: {context_type}.{key} = {type(value).__name__}")

    def get_context(self, context_type: str, key: str) -> Optional["CapabilityContext"]:
        """Retrieve using Pydantic's .model_validate() for reconstruction.

        Args:
            context_type: Type of context to retrieve
            key: Key of the context instance

        Returns:
            Reconstructed CapabilityContext object or None if not found
        """
        # Check cache first
        if context_type in self._object_cache and key in self._object_cache[context_type]:
            cached_obj = self._object_cache[context_type][key]
            logger.debug(
                f"Retrieved cached context: {context_type}.{key} = {type(cached_obj).__name__}"
            )
            return cached_obj

        # Get raw dictionary data
        raw_data = self._data.get(context_type, {}).get(key)
        if raw_data is None:
            return None

        # Get context class from registry
        context_class = self._get_context_class(context_type)
        if context_class is None:
            # Fallback: return raw dict data wrapped in a DictNamespace for both
            # dot-access (obj.key) AND subscript access (obj["key"])
            # This allows code to work without registry (e.g., subprocess execution)
            logger.info(f"Using raw dict data for {context_type}.{key} (registry not available)")
            return DictNamespace(raw_data)

        # Use Pydantic's model_validate for reconstruction
        try:
            # Strip _meta from raw_data before Pydantic validation
            # _meta contains framework metadata (task_objective) that isn't part of the context schema
            data_for_validation = {k: v for k, v in raw_data.items() if k != "_meta"}
            reconstructed_obj = context_class.model_validate(data_for_validation)

            # Cache the reconstructed object
            if context_type not in self._object_cache:
                self._object_cache[context_type] = {}
            self._object_cache[context_type][key] = reconstructed_obj

            logger.debug(
                f"Retrieved and cached context: {context_type}.{key} = {type(reconstructed_obj).__name__}"
            )
            return reconstructed_obj
        except Exception as e:
            logger.error(f"Failed to reconstruct {context_type}: {e}")
            return None

    def get_all_of_type(self, context_type: str) -> dict[str, "CapabilityContext"]:
        """Get all contexts of a specific type as reconstructed objects.

        Args:
            context_type: Type of context to retrieve

        Returns:
            Dictionary of key -> CapabilityContext objects
        """
        result = {}
        context_keys = self._data.get(context_type, {}).keys()

        for key in context_keys:
            context_obj = self.get_context(context_type, key)
            if context_obj:
                result[key] = context_obj

        return result

    def get_context_metadata(self, context_type: str, key: str) -> dict[str, Any] | None:
        """Get metadata for a specific context (e.g., task_objective).

        Args:
            context_type: Type of context
            key: Key of the context instance

        Returns:
            Metadata dictionary containing task_objective, or None if not found
        """
        raw_data = self._data.get(context_type, {}).get(key)
        if raw_data is None:
            return None
        return raw_data.get("_meta")

    def get_all_context_metadata(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Get metadata for all contexts, organized by type and key.

        Returns:
            Dictionary: {context_type: {key: {task_objective: "..."}}}
            Only includes contexts that have metadata.
        """
        result: dict[str, dict[str, dict[str, Any]]] = {}
        for context_type, contexts in self._data.items():
            if context_type.startswith("_"):
                continue  # Skip internal keys like _execution_config
            for key, context_data in contexts.items():
                meta = context_data.get("_meta")
                if meta:
                    if context_type not in result:
                        result[context_type] = {}
                    result[context_type][key] = meta
        return result

    def get_all(self) -> dict[str, Any]:
        """Get all context data in flattened format for reporting/summary purposes.

        Returns:
            Dictionary with flattened keys in format "context_type.key" -> context object
        """
        flattened = {}
        for context_type in self._data.keys():
            contexts_dict = self.get_all_of_type(context_type)
            for key, context in contexts_dict.items():
                flattened_key = f"{context_type}.{key}"
                flattened[flattened_key] = context
        return flattened

    def get_context_access_description(
        self, context_filter: list[dict[str, str]] | None = None
    ) -> str:
        """Create detailed description of available context data for use in prompts.

        Args:
            context_filter: Optional list of context filter dictionaries

        Returns:
            Formatted string description of available context data
        """
        if not self._data:
            return "No context data available."

        description_parts = []
        description_parts.append(
            "The agent context is available via the 'context' object with dot notation access:"
        )
        description_parts.append("")

        # Determine which contexts to show based on context_filter
        contexts_to_show = {}

        if context_filter and isinstance(context_filter, list) and context_filter:
            # Filter to only show contexts referenced in context_filter
            for filter_dict in context_filter:
                for context_type, context_key in filter_dict.items():
                    if context_type in self._data and context_key in self._data[context_type]:
                        if context_type not in contexts_to_show:
                            contexts_to_show[context_type] = {}
                        # Reconstruct the object for access details
                        context_obj = self.get_context(context_type, context_key)
                        if context_obj:
                            contexts_to_show[context_type][context_key] = context_obj
        else:
            # Show all contexts (reconstruct all objects)
            for context_type in self._data.keys():
                contexts_to_show[context_type] = self.get_all_of_type(context_type)

        if not contexts_to_show:
            return "No relevant context data available for the specified context filter."

        for context_type, contexts_dict in contexts_to_show.items():
            if isinstance(contexts_dict, dict):
                description_parts.append(f"• context.{context_type}:")

                for key, context_obj in contexts_dict.items():
                    # Use the get_access_details method with the actual key name
                    if hasattr(context_obj, "get_access_details"):
                        try:
                            details = context_obj.get_access_details(key)
                            if isinstance(details, dict):
                                description_parts.append(f"  └── {key}")

                                details_str = json.dumps(details, indent=6, default=str)
                                description_parts.append(f"      └── Details: {details_str}")
                            else:
                                description_parts.append(f"  └── {key}: {str(details)}")
                        except Exception as e:
                            description_parts.append(
                                f"  └── {key}: {type(context_obj).__name__} object (get_access_details error: {e})"
                            )
                    else:
                        description_parts.append(
                            f"  └── {key}: {type(context_obj).__name__} object (no get_access_details method available)"
                        )

                description_parts.append("")

        return "\n".join(description_parts)

    def get_summaries(self, step: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Get summaries of contexts for human display/LLM consumption.

        Args:
            step: Optional step dict. If provided, extract contexts from step.inputs.
                  If None, get summaries for all available contexts.

        Returns:
            List of context summary dicts. Each context class defines its own
            summary structure via the get_summary() method.

        Example:
            [
                {"type": "PV Addresses", "total_pvs": 5, "pv_list": [...]},
                {"type": "Current Weather", "location": "San Francisco", "temp": 15.0, "conditions": "Sunny"},
                {"type": "Current Weather", "location": "New York", "temp": 20.0, "conditions": "Cloudy"}
            ]
        """
        # Get contexts (filtered by step or all)
        if step is not None:
            try:
                step_contexts = self.extract_from_step(step, {})
            except Exception as e:
                logger.error(f"Error extracting step contexts: {e}")
                step_contexts = self.get_all()
        else:
            step_contexts = self.get_all()

        # Flatten any lists and collect summaries
        summaries = []
        for context_or_list in step_contexts.values():
            # Handle both single contexts and lists
            contexts_list = (
                context_or_list if isinstance(context_or_list, list) else [context_or_list]
            )
            for context in contexts_list:
                summaries.append(context.get_summary())

        return summaries

    def add_execution_config(self, config: dict) -> None:
        """Add execution configuration to context data.

        This config is saved alongside capability contexts and can be
        accessed by runtime utilities for reproducible execution.

        Args:
            config: Execution configuration dictionary
        """
        # Store in special _execution_config key
        if "_execution_config" not in self._data:
            self._data["_execution_config"] = {}

        self._data["_execution_config"] = config

    def get_raw_data(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Get the raw dictionary data for state updates.

        Returns:
            The raw dictionary data for LangGraph state updates
        """
        return self._data

    def save_context_to_file(self, folder_path: Path, filename: str = "context.json") -> Path:
        """Save capability context data to a JSON file in the specified folder.

        This method always saves the current context data to ensure it reflects
        the latest state. It uses the same serialization format as the state system.

        Args:
            folder_path: Path to the folder where the context file should be saved
            filename: Name of the context file (default: "context.json")

        Returns:
            Path to the saved context file

        Raises:
            OSError: If file cannot be written
            TypeError: If context data cannot be serialized
            ValueError: If filename is empty or contains invalid characters
        """
        if not isinstance(folder_path, Path):
            folder_path = Path(folder_path)

        if not filename or not filename.strip():
            raise ValueError("Filename cannot be empty")

        # Ensure filename has .json extension if not provided
        if not filename.endswith(".json"):
            filename = f"{filename}.json"

        # Ensure folder exists
        folder_path.mkdir(parents=True, exist_ok=True)

        context_file = folder_path / filename

        try:
            # Save using standard JSON (data is already JSON-serializable via Pydantic)
            with open(context_file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Saved context data to: {context_file}")
            return context_file

        except Exception as e:
            logger.error(f"Failed to save context to {context_file}: {e}")
            raise

    def _get_context_class(self, context_type: str) -> type | None:
        """Get context class from registry or direct mapping.

        Args:
            context_type: The context type string

        Returns:
            Context class or None if not found (graceful fallback)
        """
        try:
            # Import registry here to avoid circular imports
            from osprey.registry import get_registry

            registry = get_registry()
            return registry.get_context_class(context_type)
        except Exception as e:
            # Graceful fallback: return None instead of raising
            # This allows get_context() to return raw dict data when registry isn't available
            # (e.g., in subprocess execution environments)
            logger.warning(
                f"Registry not available for {context_type}, will use raw dict data: {e}"
            )
            return None

    def extract_from_step(
        self,
        step: dict[str, Any],
        state: dict[str, Any],
        constraints: list[str | tuple[str, str]] | None = None,
        constraint_mode: str = "hard",
    ) -> dict[str, Union["CapabilityContext", list["CapabilityContext"]]]:
        """Extract all contexts specified in step.inputs with optional type and cardinality constraints.

        This method consolidates the common pattern of extracting context data from
        step inputs and validating against expected types and instance counts. It replaces
        repetitive validation logic across capabilities.

        Args:
            step: Execution step with inputs list like ``[{"PV_ADDRESSES": "key1"}, {"TIME_RANGE": "key2"}]``
            state: Current agent state (for error checking)
            constraints: Optional list of required context types. Each item can be:
                - String: context type with no cardinality restriction (e.g., ``"PV_ADDRESSES"``)
                - Tuple: ``(context_type, cardinality)`` where cardinality is:
                    - ``"single"``: Must be exactly one instance (not a list)
                    - ``"multiple"``: Must be multiple instances (must be a list)
                Example: ``["PV_ADDRESSES", ("TIME_RANGE", "single")]``
            constraint_mode: ``"hard"`` (all constraints required) or ``"soft"`` (at least one constraint required)

        Returns:
            Dict mapping context_type to:
            - Single context object if only one of that type
            - List of context objects if multiple of that type

        Raises:
            ValueError: If contexts not found, constraints not met, or cardinality violated
                       (capability converts to specific error)

        Example:

        .. code-block:: python

            # Simple extraction without constraints
            contexts = context_manager.extract_from_step(step, state)
            pv_context = contexts["PV_ADDRESSES"]

            # With cardinality constraints (eliminates isinstance checks)
            try:
                contexts = context_manager.extract_from_step(
                    step, state,
                    constraints=[
                        ("PV_ADDRESSES", "single"),  # Must be exactly one
                        ("TIME_RANGE", "single")     # Must be exactly one
                    ],
                    constraint_mode="hard"
                )
                pv_context = contexts["PV_ADDRESSES"]  # Guaranteed to be single object
                time_context = contexts["TIME_RANGE"]  # Guaranteed to be single object
            except ValueError as e:
                raise ArchiverDependencyError(str(e))

            # Mixed constraints (some with cardinality, some without)
            try:
                contexts = context_manager.extract_from_step(
                    step, state,
                    constraints=[
                        ("CHANNEL_ADDRESSES", "single"),  # Must be single
                        "ARCHIVER_DATA"                   # Any cardinality ok
                    ]
                )
            except ValueError as e:
                raise DataValidationError(str(e))

            # Soft constraints (at least one required, no cardinality restriction)
            try:
                contexts = context_manager.extract_from_step(
                    step, state,
                    constraints=["PV_VALUES", "ARCHIVER_DATA"],
                    constraint_mode="soft"
                )
            except ValueError as e:
                raise DataValidationError(str(e))
        """
        # Extract all contexts from step.inputs
        step_inputs = step.get("inputs", [])
        if not step_inputs:
            # Check if we need any data at all
            if constraints:
                capability_context_data = state.get("capability_context_data", {})
                if not capability_context_data:
                    raise ValueError("No context data available and no step inputs specified")
            return {}

        # First, build nested structure to detect duplicates
        nested_results = {}  # {context_type: {key: obj}}

        for input_dict in step_inputs:
            for context_type, context_key in input_dict.items():
                context_obj = self.get_context(context_type, context_key)
                if context_obj:
                    if context_type not in nested_results:
                        nested_results[context_type] = {}
                    nested_results[context_type][context_key] = context_obj
                else:
                    raise ValueError(f"Context {context_type}.{context_key} not found")

        # Then flatten: single object or list (preserves insertion order in Python 3.7+)
        results = {}
        for context_type, contexts_dict in nested_results.items():
            values = list(contexts_dict.values())
            if len(values) > 1:
                logger.debug(
                    f"Multiple contexts of type {context_type} detected: {len(values)} instances"
                )
            results[context_type] = values[0] if len(values) == 1 else values

        # Apply constraints if specified
        if constraints:
            # Parse constraints to separate type names from cardinality requirements
            required_types = set()
            cardinality_constraints = {}  # {context_type: cardinality}

            for constraint in constraints:
                if isinstance(constraint, tuple):
                    context_type, cardinality = constraint
                    if cardinality not in ("single", "multiple"):
                        raise ValueError(
                            f"Invalid cardinality '{cardinality}' for {context_type}. "
                            f"Must be 'single' or 'multiple'"
                        )
                    required_types.add(context_type)
                    cardinality_constraints[context_type] = cardinality
                else:
                    # String constraint - no cardinality restriction
                    required_types.add(constraint)

            found_types = set(results.keys())

            # Validate type presence
            if constraint_mode == "hard":
                # All constraints must be satisfied
                missing_types = required_types - found_types
                if missing_types:
                    raise ValueError(f"Missing required context types: {list(missing_types)}")
            elif constraint_mode == "soft":
                # At least one constraint must be satisfied
                if not (required_types & found_types):
                    raise ValueError(
                        f"None of the required context types found: {list(required_types)}"
                    )
            else:
                raise ValueError(
                    f"Invalid constraint_mode: {constraint_mode}. Use 'hard' or 'soft'"
                )

            # Validate cardinality constraints
            for context_type, required_cardinality in cardinality_constraints.items():
                if context_type in results:
                    context_value = results[context_type]
                    is_list = isinstance(context_value, list)

                    if required_cardinality == "single" and is_list:
                        raise ValueError(
                            f"Context type '{context_type}' expected single instance "
                            f"but got {len(context_value)} instances"
                        )
                    elif required_cardinality == "multiple" and not is_list:
                        raise ValueError(
                            f"Context type '{context_type}' expected multiple instances "
                            f"but got single instance"
                        )

                    logger.debug(
                        f"Cardinality validation passed for {context_type}: "
                        f"required={required_cardinality}, actual={'list' if is_list else 'single'}"
                    )

        return results


class ContextNamespace:
    """Namespace object that provides dot notation access to context objects."""

    def __init__(self, context_manager: ContextManager, context_type: str):
        self._context_manager = context_manager
        self._context_type = context_type

    def __getattr__(self, key: str):
        """Get context object by key with lazy reconstruction."""
        context_obj = self._context_manager.get_context(self._context_type, key)
        if context_obj is not None:
            return context_obj
        raise AttributeError(f"Context '{self._context_type}' has no key '{key}'")

    def __setattr__(self, key: str, value):
        """Set context object by key."""
        if key.startswith("_"):
            # Set private attributes normally
            super().__setattr__(key, value)
        else:
            # This would require the value to be a CapabilityContext object
            # For now, raise an error as direct assignment should go through set_context
            raise AttributeError(
                f"Cannot directly assign to context key '{key}'. Use context_manager.set_context() instead."
            )
