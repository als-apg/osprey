"""Serialization utilities for the Python executor.

Robust JSON serialization for execution results: scientific types (numpy,
pandas, matplotlib figures) are converted to JSON-safe representations, with
a fallback save path when full serialization fails.
"""

import json
from typing import Any

from osprey.utils.logger import get_logger

logger = get_logger("osprey")


def make_json_serializable(obj: Any) -> Any:
    """Convert complex objects to JSON-serializable format using modern Python patterns.

    This is a standalone function that can be imported and used by execution wrappers
    and other components that need robust JSON serialization.

    Args:
        obj: Any Python object to make JSON-serializable

    Returns:
        JSON-serializable representation of the object

    Examples:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Handle numpy arrays
        >>> arr = np.array([1, 2, 3])
        >>> serializable = make_json_serializable(arr)
        >>> print(serializable)  # [1, 2, 3]
        >>>
        >>> # Handle matplotlib figures
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> serializable = make_json_serializable(fig)
        >>> print(serializable['_type'])  # 'matplotlib_figure'
    """
    from datetime import datetime
    from pathlib import Path

    def enhanced_json_serializer(obj):
        """Enhanced serializer supporting scientific computing objects."""
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()

        # Handle numpy arrays first (they also have 'item' method)
        elif hasattr(obj, "tolist"):
            try:
                return obj.tolist()
            except (ValueError, AttributeError):
                # Fallback for objects that have tolist but it fails
                pass

        # Handle numpy scalars (single-element arrays)
        elif hasattr(obj, "item"):
            try:
                return obj.item()
            except ValueError:
                # Multi-element arrays will fail here, should have been caught above
                pass

        # Handle matplotlib figures
        elif _is_matplotlib_figure(obj):
            return _serialize_matplotlib_figure(obj)

        # Handle pandas DataFrames and Series
        elif hasattr(obj, "to_dict") and hasattr(obj, "index"):
            return obj.to_dict()

        # Handle pathlib objects
        elif isinstance(obj, Path):
            return str(obj)

        # Handle sets and frozensets
        elif isinstance(obj, (set, frozenset)):
            return {"items": list(obj), "_type": type(obj).__name__}

        # Handle complex numbers
        elif isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag, "_type": "complex"}

        # Handle custom objects with to_dict method
        elif hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()

        # Default to string representation with type info
        else:
            return {
                "value": str(obj),
                "type": type(obj).__name__,
                "module": getattr(type(obj), "__module__", "unknown"),
                "_serialization_note": "converted_to_string",
            }

    # Use json.loads(json.dumps()) to leverage Python's built-in serialization
    # This is more efficient and handles nested structures automatically
    try:
        return json.loads(json.dumps(obj, default=enhanced_json_serializer, ensure_ascii=False))
    except (TypeError, ValueError) as e:
        # If JSON serialization completely fails, return a structured fallback
        return {
            "_serialization_failed": True,
            "_original_type": type(obj).__name__,
            "_error": str(e),
            "_string_representation": str(obj),
        }


def _is_matplotlib_figure(obj) -> bool:
    """Check if object is a matplotlib figure."""
    return hasattr(obj, "savefig") and hasattr(obj, "get_axes") and type(obj).__name__ == "Figure"


def _serialize_matplotlib_figure(fig) -> dict:
    """Serialize matplotlib figure to JSON-compatible dict."""
    try:
        # Extract basic figure information
        axes_info = []
        for ax in fig.get_axes():
            ax_info = {
                "title": ax.get_title(),
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "xlim": ax.get_xlim(),
                "ylim": ax.get_ylim(),
            }

            # Extract line data if present
            lines_data = []
            for line in ax.get_lines():
                line_data = {
                    "xdata": (
                        line.get_xdata().tolist()
                        if hasattr(line.get_xdata(), "tolist")
                        else list(line.get_xdata())
                    ),
                    "ydata": (
                        line.get_ydata().tolist()
                        if hasattr(line.get_ydata(), "tolist")
                        else list(line.get_ydata())
                    ),
                    "label": line.get_label(),
                }
                lines_data.append(line_data)
            ax_info["lines"] = lines_data
            axes_info.append(ax_info)

        return {
            "_type": "matplotlib_figure",
            "figure_size": fig.get_size_inches().tolist(),
            "dpi": fig.get_dpi(),
            "axes": axes_info,
            "_note": "Matplotlib figure serialized with basic plot data",
        }
    except Exception as e:
        return {
            "_type": "matplotlib_figure",
            "_error": f"Failed to serialize figure: {str(e)}",
            "_fallback": str(fig),
        }


def serialize_results_to_file(results: Any, file_path: str) -> dict:
    """Serialize results and save to JSON file with comprehensive error handling.

    This function is designed to be called from execution wrappers and provides
    robust serialization with detailed error reporting.

    Args:
        results: The results object to serialize
        file_path: Path where to save the JSON file

    Returns:
        dict: Metadata about the serialization operation

    Examples:
        >>> # Called from execution wrapper
        >>> metadata = serialize_results_to_file(results, 'results.json')
        >>> if metadata['success']:
        >>>     print(f"Results saved successfully to {metadata['file_path']}")
        >>> else:
        >>>     print(f"Serialization failed: {metadata['error']}")
    """

    metadata = {
        "success": False,
        "file_path": file_path,
        "error": None,
        "serialization_warnings": [],
    }

    try:
        # Convert to JSON-serializable format
        serializable_results = make_json_serializable(results)

        # Save to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        metadata["success"] = True
        return metadata

    except Exception as e:
        metadata["error"] = str(e)
        metadata["error_type"] = type(e).__name__

        # Try to save a minimal fallback
        try:
            fallback_data = {
                "_serialization_failed": True,
                "_original_error": str(e),
                "_fallback_representation": str(results),
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(fallback_data, f, indent=2, ensure_ascii=False)
            metadata["fallback_saved"] = True
        except Exception as fallback_error:
            metadata["fallback_error"] = str(fallback_error)

        return metadata


async def serialize_results_to_file_async(results: Any, file_path: str) -> dict:
    """Async version of serialize_results_to_file for non-blocking JSON writing.

    This function is designed to be called from async execution contexts to avoid
    blocking the event loop during file I/O operations.

    Args:
        results: The results object to serialize
        file_path: Path where to save the JSON file

    Returns:
        dict: Metadata about the serialization operation

    Examples:
        >>> # Called from async execution wrapper
        >>> metadata = await serialize_results_to_file_async(results, 'results.json')
        >>> if metadata['success']:
        >>>     print(f"Results saved successfully to {metadata['file_path']}")
        >>> else:
        >>>     print(f"Serialization failed: {metadata['error']}")
    """
    import asyncio

    import aiofiles

    metadata = {
        "success": False,
        "file_path": file_path,
        "error": None,
        "serialization_warnings": [],
    }

    try:
        # Convert to JSON-serializable format (CPU-bound, use to_thread)
        serializable_results = await asyncio.to_thread(make_json_serializable, results)

        # Save to file asynchronously
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            json_str = await asyncio.to_thread(
                json.dumps, serializable_results, indent=2, ensure_ascii=False
            )
            await f.write(json_str)

        metadata["success"] = True
        return metadata

    except Exception as e:
        metadata["error"] = str(e)
        metadata["error_type"] = type(e).__name__

        # Try to save a minimal fallback
        try:
            fallback_data = {
                "_serialization_failed": True,
                "_original_error": str(e),
                "_fallback_representation": str(results),
            }
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                json_str = await asyncio.to_thread(
                    json.dumps, fallback_data, indent=2, ensure_ascii=False
                )
                await f.write(json_str)
            metadata["fallback_saved"] = True
        except Exception as fallback_error:
            metadata["fallback_error"] = str(fallback_error)

        return metadata
