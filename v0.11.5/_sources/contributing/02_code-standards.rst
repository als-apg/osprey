Code Standards
==============

Python style guidelines, testing requirements, and documentation standards.

Python Style
------------

We follow PEP 8 with these configurations:

- **Line length**: 100 characters (configured in Ruff)
- **Type hints**: Encouraged but not required
- **Classes**: PascalCase (``CapabilityManager``)
- **Functions**: snake_case (``get_data``)
- **Constants**: UPPER_SNAKE_CASE (``MAX_RETRIES``)
- **Private members**: Leading underscore (``_internal_method``)

**Import organization:**

.. code-block:: python

   # 1. Standard library
   import os
   from typing import Optional

   # 2. Third-party
   import numpy as np
   from langgraph.graph import StateGraph

   # 3. Local
   from osprey.base import Capability

Docstrings
----------

All public functions, classes, and methods need docstrings in Google style:

.. code-block:: python

   def capability_function(param1: str, param2: int) -> bool:
       """Short description of function.

       Longer description providing context about usage.

       Args:
           param1: Description of first parameter
           param2: Description of second parameter

       Returns:
           Description of return value

       Raises:
           ValueError: When parameter is invalid

       Example:
           >>> result = capability_function("test", 42)
           >>> print(result)
           True
       """
       pass

Testing
-------

All new functionality must include tests.

**Test Types:**

- **Unit Tests** - Fast, no external dependencies, mock APIs
- **Integration Tests** - Test component interactions
- **E2E Tests** - Full workflows, real services (costly: $0.10-$0.25 per run)

**When to use:**

- Unit tests: Default for most code
- Integration tests: Component interactions
- E2E tests: Critical workflows only, sparingly

**Test structure:**

.. code-block:: python

   import pytest
   from osprey.module import YourClass

   class TestYourClass:
       """Test suite for YourClass."""

       def test_basic_functionality(self):
           """Test basic functionality works as expected."""
           obj = YourClass()
           result = obj.method()
           assert result == expected_value

       def test_error_handling(self):
           """Test error handling for invalid input."""
           obj = YourClass()
           with pytest.raises(ValueError):
               obj.method(invalid_input)

**Run tests:**

.. code-block:: bash

   # Unit tests
   pytest tests/ --ignore=tests/e2e -v

   # With coverage
   pytest tests/ --ignore=tests/e2e --cov=src/osprey

   # E2E tests (requires API keys)
   pytest tests/e2e/ -v

Linting and Formatting
-----------------------

We use Ruff for linting and formatting:

.. code-block:: bash

   # Check
   ruff check src/ tests/
   ruff format --check src/ tests/

   # Auto-fix
   ruff check --fix src/ tests/
   ruff format src/ tests/

   # Type checking (optional)
   mypy src/ --no-error-summary

Pre-Commit Checks
-----------------

Run before every commit:

.. code-block:: bash

   ./scripts/premerge_check.sh

This checks for:

- Debug code (print, pdb, breakpoint)
- Commented-out code
- Hardcoded secrets
- CHANGELOG.md was modified
- TODOs without issue links
- Tests passing
- Code formatting

Next Steps
----------

- :doc:`01_git-and-github` - Git and GitHub workflow
- :doc:`03_ai-assisted-development` - AI workflows
